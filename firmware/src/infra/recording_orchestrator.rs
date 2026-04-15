use embassy_sync::{blocking_mutex::raw::NoopRawMutex, signal::Signal, watch};
use embassy_time::{Duration, Instant, Timer};
use log::info;

use crate::app::config::NetOutChannel;
use crate::domain::{CtrlMsg, NetOut};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum OrchestratorState {
    Idle,
    OnSegment {
        started: Instant,
    },
    OffSegment {
        cycle_id: u32,
        planned_ms: u32,
        started: Instant,
    },
}

#[embassy_executor::task]
pub async fn recording_orchestrator(
    net_out: &'static NetOutChannel,
    recording_state_tx: watch::DynSender<'static, bool>,
    mut net_ready_rx: watch::DynReceiver<'static, bool>,
    mode: wire_protocol::RecordingModeKind,
    mut pump_state_rx: Option<watch::DynReceiver<'static, bool>>,
    flush_req: watch::DynSender<'static, u32>,
    flush_ack: &'static Signal<NoopRawMutex, ()>,
) {
    match mode {
        wire_protocol::RecordingModeKind::Continuous => {
            // Gate recording strictly on network readiness and emit a single initial Segment
            let mut net_ready = net_ready_rx.get_and(|b| *b).await;
            let mut sent_initial = false;
            if net_ready && !sent_initial {
                let _ = net_out
                    .send(NetOut::Ctrl(CtrlMsg::Segment {
                        cycle_id: 0,
                        label: wire_protocol::SegmentLabel::Undefined,
                        prev_reason: wire_protocol::StopReason::Normal,
                        plan_ms: 0,
                    }))
                    .await;
                sent_initial = true;
            }
            recording_state_tx.send(net_ready);
            loop {
                if let Some(n) = net_ready_rx.try_changed() {
                    net_ready = n;
                    if net_ready && !sent_initial {
                        let _ = net_out
                            .send(NetOut::Ctrl(CtrlMsg::Segment {
                                cycle_id: 0,
                                label: wire_protocol::SegmentLabel::Undefined,
                                prev_reason: wire_protocol::StopReason::Normal,
                                plan_ms: 0,
                            }))
                            .await;
                        sent_initial = true;
                    }
                    recording_state_tx.send(net_ready);
                }
                Timer::after(Duration::from_millis(50)).await;
            }
        }
        wire_protocol::RecordingModeKind::PumpGated => {
            info!("Recording orchestrator (pump-gated) started");
            let mut state = OrchestratorState::Idle;
            let mut cycle_id: u32 = 0;
            let mut flush_seq: u32 = 0;
            let mut current_pump_on = if let Some(rx) = pump_state_rx.as_mut() {
                rx.get_and(|v| *v).await
            } else {
                false
            };

            // Initialize network readiness and recording state
            let mut net_ready = net_ready_rx.get_and(|b| *b).await;
            recording_state_tx.send(false);
            let mut sent_initial = false;
            if net_ready && !sent_initial {
                // Emit initial Segment representing current state; do not start recording unless ON
                let label = if current_pump_on {
                    wire_protocol::SegmentLabel::On
                } else {
                    wire_protocol::SegmentLabel::Off
                };
                let _ = net_out
                    .send(NetOut::Ctrl(CtrlMsg::Segment {
                        cycle_id,
                        label,
                        prev_reason: wire_protocol::StopReason::Normal,
                        plan_ms: 0,
                    }))
                    .await;
                if current_pump_on {
                    state = OrchestratorState::OnSegment {
                        started: Instant::now(),
                    };
                    recording_state_tx.send(true);
                } else {
                    state = OrchestratorState::Idle;
                    recording_state_tx.send(false);
                }
                sent_initial = true;
            }

            let mut orch_tick: u32 = 0;
            const ORCH_DIAG_EVERY: u32 = 150; // ~3s at 20ms tick

            loop {
                // React to network readiness changes
                if let Some(n) = net_ready_rx.try_changed() {
                    net_ready = n;
                    if !net_ready {
                        recording_state_tx.send(false);
                    } else {
                        if !sent_initial {
                            let label = if current_pump_on {
                                wire_protocol::SegmentLabel::On
                            } else {
                                wire_protocol::SegmentLabel::Off
                            };
                            let _ = net_out
                                .send(NetOut::Ctrl(CtrlMsg::Segment {
                                    cycle_id,
                                    label,
                                    prev_reason: wire_protocol::StopReason::Normal,
                                    plan_ms: 0,
                                }))
                                .await;
                            if current_pump_on {
                                state = OrchestratorState::OnSegment {
                                    started: Instant::now(),
                                };
                                recording_state_tx.send(true);
                            } else {
                                state = OrchestratorState::Idle;
                                recording_state_tx.send(false);
                            }
                            sent_initial = true;
                        } else {
                            match state {
                                OrchestratorState::OnSegment { .. }
                                | OrchestratorState::OffSegment { .. } => {
                                    recording_state_tx.send(true);
                                }
                                _ => {}
                            }
                        }
                    }
                }

                // Poll pump state changes
                if let Some(rx) = pump_state_rx.as_mut() {
                    if let Some(next) = rx.try_changed() {
                        if next != current_pump_on {
                            current_pump_on = next;
                            if current_pump_on {
                                // OFF -> ON (do not pause recording here)
                                if net_ready {
                                    let prev_reason = if let OrchestratorState::OffSegment {
                                        planned_ms,
                                        started,
                                        ..
                                    } = state
                                    {
                                        let elapsed = (Instant::now() - started).as_millis() as u32;
                                        if elapsed < planned_ms {
                                            wire_protocol::StopReason::Canceled
                                        } else {
                                            wire_protocol::StopReason::Normal
                                        }
                                    } else {
                                        wire_protocol::StopReason::Normal
                                    };
                                    // Start new ON cycle (increment cycle id always on OFF->ON)
                                    cycle_id = cycle_id.wrapping_add(1);
                                    // Flush current partial audio to ensure boundary ordering
                                    flush_seq = flush_seq.wrapping_add(1);
                                    flush_req.send(flush_seq);
                                    flush_ack.wait().await;
                                    flush_ack.reset();
                                    log::info!(
                                        "[{:08} ms] ORCH: boundary ON start prev={:?} backlog={}",
                                        crate::infra::instrumentation::now_ms(),
                                        prev_reason,
                                        crate::infra::instrumentation::backlog()
                                    );
                                    let _ = net_out
                                        .send(NetOut::Ctrl(CtrlMsg::Segment {
                                            cycle_id,
                                            label: wire_protocol::SegmentLabel::On,
                                            prev_reason,
                                            plan_ms: 0,
                                        }))
                                        .await;
                                    crate::infra::instrumentation::inc_enq(1);
                                    recording_state_tx.send(true);
                                    state = OrchestratorState::OnSegment {
                                        started: Instant::now(),
                                    };
                                } else {
                                    recording_state_tx.send(false);
                                    state = OrchestratorState::Idle;
                                }
                            } else {
                                // ON -> OFF (keep recording; only pause when OFF follow-up completes)
                                if net_ready {
                                    if let OrchestratorState::OnSegment { started } = state {
                                        let on_ms = (Instant::now() - started).as_millis() as u32;
                                        // Flush current partial audio to ensure boundary ordering
                                        flush_seq = flush_seq.wrapping_add(1);
                                        flush_req.send(flush_seq);
                                        flush_ack.wait().await;
                                        flush_ack.reset();
                                        log::info!(
                                            "[{:08} ms] ORCH: boundary OFF start on_ms={} backlog={}",
                                            crate::infra::instrumentation::now_ms(),
                                            on_ms,
                                            crate::infra::instrumentation::backlog()
                                        );
                                        let _ = net_out
                                            .send(NetOut::Ctrl(CtrlMsg::Segment {
                                                cycle_id,
                                                label: wire_protocol::SegmentLabel::Off,
                                                prev_reason: wire_protocol::StopReason::Normal,
                                                plan_ms: on_ms,
                                            }))
                                            .await;
                                        crate::infra::instrumentation::inc_enq(1);
                                        if on_ms > 0 {
                                            state = OrchestratorState::OffSegment {
                                                cycle_id,
                                                planned_ms: on_ms,
                                                started: Instant::now(),
                                            };
                                        } else {
                                            state = OrchestratorState::Idle;
                                        }
                                    } else {
                                        // Not in ON; remain idle
                                        state = OrchestratorState::Idle;
                                    }
                                } else {
                                    recording_state_tx.send(false);
                                    state = OrchestratorState::Idle;
                                }
                            }
                        }
                    }
                }

                // If OFF segment is active, check timer for planned end (only if net is ready)
                if net_ready {
                    if let OrchestratorState::OffSegment {
                        cycle_id,
                        planned_ms,
                        started,
                    } = state
                    {
                        if (Instant::now() - started).as_millis() as u32 >= planned_ms {
                            // Planned OFF finished; pause recording and emit boundary to Off (no new plan)
                            recording_state_tx.send(false);
                            // Wait for recorder to flush and acknowledge pause
                            flush_ack.wait().await;
                            flush_ack.reset();
                            log::info!(
                                "[{:08} ms] ORCH: boundary OFF finish backlog={}",
                                crate::infra::instrumentation::now_ms(),
                                crate::infra::instrumentation::backlog()
                            );
                            let _ = net_out
                                .send(NetOut::Ctrl(CtrlMsg::Segment {
                                    cycle_id,
                                    label: wire_protocol::SegmentLabel::Off,
                                    prev_reason: wire_protocol::StopReason::Normal,
                                    plan_ms: 0,
                                }))
                                .await;
                            crate::infra::instrumentation::inc_enq(1);
                            state = OrchestratorState::Idle;
                        }
                    }
                } else {
                    // Ensure paused when not ready
                    recording_state_tx.send(false);
                }

                orch_tick = orch_tick.wrapping_add(1);
                if orch_tick % ORCH_DIAG_EVERY == 0 {
                    log::debug!(
                        "ORCH: state={:?} pump={} net_ready={}",
                        state, current_pump_on, net_ready
                    );
                }

                Timer::after(Duration::from_millis(20)).await;
            }
        }
    }
}
