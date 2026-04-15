//! I2S audio capture and chunking for streaming.
//!
//! Format notes:
//! - The I2S peripheral is configured as Philips standard with `Data32Channel32`.
//! - Each frame contains two 32-bit channel slots (left/right). Many MEMS mics mirror mono data
//!   across both or place the valid 24-bit sample in the MSBs.
//! - This task supports both 16-bit and 24-bit precision based on compile-time configuration.
//! - 16-bit mode extracts the high 16 bits of each 32-bit word.
//! - 24-bit mode extracts the full 24-bit precision from the microphone data.
//!
//! Timing/metrics constants are grouped at the top for easier tuning.

mod drain;
mod metrics;
mod processors;

use alloc::vec::Vec;

use embassy_sync::{blocking_mutex::raw::NoopRawMutex, signal::Signal, watch};
use embassy_time::{Duration, Instant};
use log::{debug, error, info};
use metrics::{STREAM_LATE_INTERVAL_MS, StreamLateLog, maybe_emit_metrics};
use processors::{
    MonoBeamformerProcessor,
    Processor,
    StereoHigh16Processor,
    StereoHigh24Processor,
};

use crate::domain::{AudioData, NetOut};

// Feature-gated config imports
#[cfg(feature = "streaming")]
use crate::app::config::{
    BEAMFORM_DELAY_SAMPLES,
    BEAMFORM_GAIN,
    BEAMFORM_HPF_FC_HZ,
    BEAMFORM_LPF_ENABLED,
    BEAMFORM_LPF_FC_HZ,
    BEAMFORM_RING_SAMPLES,
    NetOutChannel,
    is_16bit_mode,
    is_24bit_mode,
};
#[cfg(feature = "inference")]
use crate::pump_app::config::{
    BEAMFORM_DELAY_SAMPLES,
    BEAMFORM_GAIN,
    BEAMFORM_HPF_FC_HZ,
    BEAMFORM_LPF_ENABLED,
    BEAMFORM_LPF_FC_HZ,
    BEAMFORM_RING_SAMPLES,
    NetOutChannel,
    is_16bit_mode,
    is_24bit_mode,
};

use crate::recover_late_pop;

const LATE_DRAIN_ATTEMPTS: usize = 10;
const LATE_RECOVERY_ATTEMPTS: usize = 20;

#[derive(Copy, Clone)]
struct RecorderTuning {
    late_drain_attempts: usize,
    late_recovery_attempts: usize,
}

impl Default for RecorderTuning {
    fn default() -> Self {
        Self {
            late_drain_attempts: LATE_DRAIN_ATTEMPTS,
            late_recovery_attempts: LATE_RECOVERY_ATTEMPTS,
        }
    }
}

/// Streaming configuration for readability when passing multiple params around.
struct RecorderCfg {
    sample_rate: u32,
    channels: u8,
    chunk_samples: usize,
}

struct RecorderSession {
    start: Instant,
    last_metrics: Instant,
    total_samples: usize,
    late: StreamLateLog,
}

impl RecorderSession {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_metrics: now,
            total_samples: 0,
            late: StreamLateLog::new(Duration::from_millis(STREAM_LATE_INTERVAL_MS)),
        }
    }

    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis()
    }

    fn record_late(&mut self, drained: usize, suffix: &str) {
        self.late.inc();
        self.late.add_drained(drained);
        self.late.maybe_log(self.total_samples, self.start, suffix);
    }
}

fn begin_streaming<S>(cfg: &RecorderCfg, audio_chunk: &mut Vec<S>) -> RecorderSession {
    info!("Starting continuous audio streaming");
    debug!(
        "Streaming parameters: rate={}Hz channels={} chunk_samples={}",
        cfg.sample_rate, cfg.channels, cfg.chunk_samples
    );
    audio_chunk.clear();
    RecorderSession::new()
}

fn pause_streaming<S>(session: RecorderSession, audio_chunk: &mut Vec<S>) {
    info!(
        "Audio streaming paused: {} total samples in {}ms",
        session.total_samples,
        session.elapsed_ms()
    );
    audio_chunk.clear();
}

fn update_recording_state<S, W: FnOnce(Vec<S>) -> crate::domain::AudioData + Copy>(
    recording_state: &mut watch::DynReceiver<'static, bool>,
    is_recording: &mut bool,
    session: &mut Option<RecorderSession>,
    audio_chunk: &mut Vec<S>,
    cfg: &RecorderCfg,
    net_out_channel: &'static NetOutChannel,
    wrap: W,
    flush_ack: Option<&'static Signal<NoopRawMutex, ()>>,
) {
    if let Some(next_state) = recording_state.try_changed() {
        if next_state == *is_recording {
            return;
        }

        if next_state {
            *session = Some(begin_streaming(cfg, audio_chunk));
        } else if let Some(active) = session.take() {
            // Flush final partial chunk, if present, before pausing to minimize tail loss
            if !audio_chunk.is_empty() {
                info!(
                    "Sending final partial chunk on pause: {} samples",
                    audio_chunk.len()
                );
                let audio_data = wrap(core::mem::take(audio_chunk));
                if let Err(_full) = net_out_channel.try_send(NetOut::Audio(audio_data)) {
                    log::info!("Dropping final partial chunk due to full channel");
                }
            }
            pause_streaming(active, audio_chunk);
            if let Some(sig) = flush_ack {
                sig.signal(());
            }
        } else {
            audio_chunk.clear();
        }

        *is_recording = next_state;
    }
}

fn handle_flush_request<S>(
    flush_rx: &mut watch::DynReceiver<'static, u32>,
    audio_chunk: &mut Vec<S>,
    net_out_channel: &'static NetOutChannel,
    wrap: impl FnOnce(Vec<S>) -> AudioData + Copy,
    flush_ack: &Signal<NoopRawMutex, ()>,
) {
    if let Some(seq) = flush_rx.try_changed() {
        log::info!(
            "[{:08} ms] Recorder: flush_req={} received",
            crate::infra::instrumentation::now_ms(),
            seq
        );
        if !audio_chunk.is_empty() {
            let audio_data = wrap(core::mem::take(audio_chunk));
            if net_out_channel.try_send(NetOut::Audio(audio_data)).is_ok() {
                crate::infra::instrumentation::inc_enq(1);
            }
        }
        flush_ack.signal(());
    }
}

#[inline]
async fn flush_full_chunks<S>(
    audio_chunk: &mut Vec<S>,
    cfg: &RecorderCfg,
    net_out_channel: &'static NetOutChannel,
    total_samples_streamed: usize,
    streaming_start: Instant,
    last_metrics: &mut Instant,
    wrap: impl FnOnce(Vec<S>) -> AudioData + Copy,
) {
    while audio_chunk.len() >= cfg.chunk_samples {
        let chunks_sent = total_samples_streamed / cfg.chunk_samples;
        maybe_emit_metrics(
            last_metrics,
            total_samples_streamed,
            streaming_start,
            cfg.sample_rate,
            cfg.channels,
            chunks_sent,
        );

        let head = crate::util::collect_to_external_vec(audio_chunk.drain(..cfg.chunk_samples));
        let audio_data = wrap(head);
        // Prefer dropping over blocking to avoid I2S DMA late errors.
        match net_out_channel.try_send(NetOut::Audio(audio_data)) {
            Ok(()) => {
                crate::infra::instrumentation::inc_enq(1);
            }
            Err(_full) => {
                // Channel is full: drop this chunk to keep up with real-time capture.
                // Use debug-level to avoid log floods; late metrics will still reflect pressure.
                log::info!("Dropping audio chunk due to full channel (backpressure)");
            }
        }
        audio_chunk.reserve(cfg.chunk_samples);
    }
}

#[inline]
async fn send_final_partial_chunk<S>(
    audio_chunk: Vec<S>,
    net_out_channel: &'static NetOutChannel,
    wrap: impl FnOnce(Vec<S>) -> AudioData,
) {
    if !audio_chunk.is_empty() {
        debug!("Sending final partial chunk: {} samples", audio_chunk.len());
        let audio_data = wrap(audio_chunk);
        if let Err(_full) = net_out_channel.try_send(NetOut::Audio(audio_data)) {
            log::info!("Dropping final partial chunk due to full channel");
        } else {
            crate::infra::instrumentation::inc_enq(1);
        }
    }
}

async fn run_recorder<S, P: Processor<S>>(
    i2s_rx: esp_hal::i2s::master::I2sRx<'static, esp_hal::Async>,
    buffer: &'static mut [u8],
    net_out_channel: &'static NetOutChannel,
    mut recording_state: watch::DynReceiver<'static, bool>,
    mut flush_rx: watch::DynReceiver<'static, u32>,
    sample_rate: u32,
    channels: u8,
    streaming_chunk_samples: usize,
    mut processor: P,
    wrap: impl Fn(Vec<S>) -> AudioData + Copy,
    flush_ack: &'static Signal<NoopRawMutex, ()>,
) {
    let tuning = RecorderTuning::default();
    let buffer_len = buffer.len();
    // Scratch buffer used for DMA pop() copies. In esp-hal, `pop()` requires the caller buffer to
    // be >= the currently available bytes, so allocate at least the full DMA ring length.
    let i2s_data = crate::util::alloc_external_slice_zeroed_aligned(buffer_len, 8);
    let cfg = RecorderCfg {
        sample_rate,
        channels,
        chunk_samples: streaming_chunk_samples,
    };

    let mut audio_chunk = Vec::<S>::with_capacity(cfg.chunk_samples);
    let mut discard_count: u32 = 0;
    const DISCARD_LOG_EVERY: u32 = 500;

    info!("Audio recorder task ready - waiting for recording control signal...");
    let mut is_recording = recording_state.get_and(|state| *state).await;
    let mut session = if is_recording {
        Some(begin_streaming(&cfg, &mut audio_chunk))
    } else {
        None
    };

    info!(
        "Creating I2S DMA circular transaction with buffer size: {} bytes",
        buffer_len
    );
    let mut transaction = match i2s_rx.read_dma_circular_async(buffer) {
        Ok(tx) => {
            info!("I2S DMA transaction created successfully");
            tx
        }
        Err(e) => {
            error!("Failed to start I2S DMA: {:?}", e);
            return;
        }
    };

    loop {
        update_recording_state(
            &mut recording_state,
            &mut is_recording,
            &mut session,
            &mut audio_chunk,
            &cfg,
            net_out_channel,
            wrap,
            Some(flush_ack),
        );
        handle_flush_request(
            &mut flush_rx,
            &mut audio_chunk,
            net_out_channel,
            wrap,
            flush_ack,
        );

        match transaction.available().await {
            Ok(_) => {}
            Err(esp_hal::i2s::master::Error::DmaError(esp_hal::dma::DmaError::Late)) => {
                let (recovery_drained, _recovered_bytes) =
                    recover_late_pop!(transaction, i2s_data, tuning.late_recovery_attempts);
                if let Some(active_session) = session.as_mut() {
                    active_session.record_late(recovery_drained, "");
                }
                continue;
            }
            Err(e) => {
                let streamed = session
                    .as_ref()
                    .map(|s| s.total_samples)
                    .unwrap_or_default();
                let elapsed_ms = session.as_ref().map(|s| s.elapsed_ms()).unwrap_or(0);
                error!(
                    "I2S available() failed: {:?}; streamed={} elapsed_ms={}",
                    e, streamed, elapsed_ms
                );
                if let Some(active_session) = session.take() {
                    info!(
                        "Audio streaming stopped: {} total samples in {}ms",
                        active_session.total_samples,
                        active_session.elapsed_ms()
                    );
                    if !audio_chunk.is_empty() {
                        send_final_partial_chunk(
                            core::mem::take(&mut audio_chunk),
                            net_out_channel,
                            wrap,
                        )
                        .await;
                    }
                }
                return;
            }
        }

        update_recording_state(
            &mut recording_state,
            &mut is_recording,
            &mut session,
            &mut audio_chunk,
            &cfg,
            net_out_channel,
            wrap,
            Some(flush_ack),
        );
        handle_flush_request(
            &mut flush_rx,
            &mut audio_chunk,
            net_out_channel,
            wrap,
            flush_ack,
        );

        let bytes_read = match transaction.pop(i2s_data).await {
            Ok(bytes) => bytes,
            Err(esp_hal::i2s::master::Error::DmaError(esp_hal::dma::DmaError::Late)) => {
                let (recovery_drained, recovered_bytes) =
                    recover_late_pop!(transaction, i2s_data, tuning.late_drain_attempts);
                if let Some(active_session) = session.as_mut() {
                    active_session.record_late(recovery_drained, "(pop)");
                }
                if recovered_bytes > 0 {
                    recovered_bytes
                } else {
                    continue;
                }
            }
            Err(e) => {
                let streamed = session
                    .as_ref()
                    .map(|s| s.total_samples)
                    .unwrap_or_default();
                let elapsed_ms = session.as_ref().map(|s| s.elapsed_ms()).unwrap_or(0);
                error!(
                    "I2S pop() failed: {:?}; streamed={} elapsed_ms={}",
                    e, streamed, elapsed_ms
                );
                if let Some(active_session) = session.take() {
                    info!(
                        "Audio streaming stopped: {} total samples in {}ms",
                        active_session.total_samples,
                        active_session.elapsed_ms()
                    );
                    if !audio_chunk.is_empty() {
                        send_final_partial_chunk(
                            core::mem::take(&mut audio_chunk),
                            net_out_channel,
                            wrap,
                        )
                        .await;
                    }
                }
                return;
            }
        };

        update_recording_state(
            &mut recording_state,
            &mut is_recording,
            &mut session,
            &mut audio_chunk,
            &cfg,
            net_out_channel,
            wrap,
            Some(flush_ack),
        );
        handle_flush_request(
            &mut flush_rx,
            &mut audio_chunk,
            net_out_channel,
            wrap,
            flush_ack,
        );

        if !is_recording {
            audio_chunk.clear();
            discard_count = discard_count.wrapping_add(1);
            if discard_count % DISCARD_LOG_EVERY == 0 {
                log::debug!("Recorder: discarded {} DMA pops (is_recording=false)", discard_count);
            }
            continue;
        }
        discard_count = 0;

        let session = session
            .as_mut()
            .expect("Recorder session must exist when recording is active");
        let added = processor.process_block(&i2s_data[..bytes_read], &mut audio_chunk);
        session.total_samples += added;

        flush_full_chunks(
            &mut audio_chunk,
            &cfg,
            net_out_channel,
            session.total_samples,
            session.start,
            &mut session.last_metrics,
            wrap,
        )
        .await;
    }
}

async fn audio_recorder_16(
    i2s_rx: esp_hal::i2s::master::I2sRx<'static, esp_hal::Async>,
    buffer: &'static mut [u8],
    net_out_channel: &'static NetOutChannel,
    recording_state: watch::DynReceiver<'static, bool>,
    flush_rx: watch::DynReceiver<'static, u32>,
    sample_rate: u32,
    channels: u8,
    streaming_chunk_samples: usize,
    flush_ack: &'static Signal<NoopRawMutex, ()>,
) {
    run_recorder::<i16, StereoHigh16Processor>(
        i2s_rx,
        buffer,
        net_out_channel,
        recording_state,
        flush_rx,
        sample_rate,
        channels,
        streaming_chunk_samples,
        StereoHigh16Processor,
        AudioData::new_16bit,
        flush_ack,
    )
    .await;
}

async fn audio_recorder_24(
    i2s_rx: esp_hal::i2s::master::I2sRx<'static, esp_hal::Async>,
    buffer: &'static mut [u8],
    net_out_channel: &'static NetOutChannel,
    recording_state: watch::DynReceiver<'static, bool>,
    flush_rx: watch::DynReceiver<'static, u32>,
    sample_rate: u32,
    channels: u8,
    streaming_chunk_samples: usize,
    flush_ack: &'static Signal<NoopRawMutex, ()>,
) {
    run_recorder::<i32, StereoHigh24Processor>(
        i2s_rx,
        buffer,
        net_out_channel,
        recording_state,
        flush_rx,
        sample_rate,
        channels,
        streaming_chunk_samples,
        StereoHigh24Processor,
        AudioData::new_24bit,
        flush_ack,
    )
    .await;
}

async fn audio_recorder_mono_24(
    i2s_rx: esp_hal::i2s::master::I2sRx<'static, esp_hal::Async>,
    buffer: &'static mut [u8],
    net_out_channel: &'static NetOutChannel,
    recording_state: watch::DynReceiver<'static, bool>,
    flush_rx: watch::DynReceiver<'static, u32>,
    sample_rate: u32,
    channels: u8,
    streaming_chunk_samples: usize,
    flush_ack: &'static Signal<NoopRawMutex, ()>,
) {
    let bf = MonoBeamformerProcessor::new(
        BEAMFORM_RING_SAMPLES,
        sample_rate as f32,
        BEAMFORM_HPF_FC_HZ,
        BEAMFORM_LPF_ENABLED,
        BEAMFORM_LPF_FC_HZ,
        BEAMFORM_DELAY_SAMPLES,
        BEAMFORM_GAIN,
    );
    run_recorder::<i32, MonoBeamformerProcessor>(
        i2s_rx,
        buffer,
        net_out_channel,
        recording_state,
        flush_rx,
        sample_rate,
        channels,
        streaming_chunk_samples,
        bf,
        AudioData::new_24bit,
        flush_ack,
    )
    .await;
}

/// I2S audio recorder task migrated from example.
#[embassy_executor::task]
pub async fn audio_recorder(
    i2s_rx: esp_hal::i2s::master::I2sRx<'static, esp_hal::Async>,
    buffer: &'static mut [u8],
    net_out_channel: &'static NetOutChannel,
    recording_state: watch::DynReceiver<'static, bool>,
    flush_rx: watch::DynReceiver<'static, u32>,
    sample_rate: u32,
    channels: u8,
    streaming_chunk_samples: usize,
    flush_ack: &'static Signal<NoopRawMutex, ()>,
) {
    if is_16bit_mode() {
        audio_recorder_16(
            i2s_rx,
            buffer,
            net_out_channel,
            recording_state,
            flush_rx,
            sample_rate,
            channels,
            streaming_chunk_samples,
            flush_ack,
        )
        .await;
    } else if is_24bit_mode() {
        if channels == 1 {
            audio_recorder_mono_24(
                i2s_rx,
                buffer,
                net_out_channel,
                recording_state,
                flush_rx,
                sample_rate,
                channels,
                streaming_chunk_samples,
                flush_ack,
            )
            .await;
        } else {
            audio_recorder_24(
                i2s_rx,
                buffer,
                net_out_channel,
                recording_state,
                flush_rx,
                sample_rate,
                channels,
                streaming_chunk_samples,
                flush_ack,
            )
            .await;
        }
    }
}
