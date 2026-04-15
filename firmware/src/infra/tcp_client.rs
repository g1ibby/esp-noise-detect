use core::net::Ipv4Addr;

use embassy_net::tcp::{Error as TcpError, TcpSocket};
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, signal::Signal, watch};
use embassy_time::{Duration, Instant, Timer, with_timeout};
use embedded_io_async::Write;
use wire_protocol::{
    HEADER_LEN, Header as ProtoHeader, MessageType, RecordingModeKind, SegmentMsg,
};

use crate::{
    app::config::NetOutChannel,
    domain::{CtrlMsg, NetOut},
    infra::instrumentation,
};

const TCP_TX_BUF_SIZE: usize = 8192;
const TCP_RX_BUF_SIZE: usize = 1024;
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(5);
const RECONNECT_DELAY: Duration = Duration::from_secs(2);
const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);
const SOCKET_KEEPALIVE: Duration = Duration::from_secs(30);
const STALL_BACKOFF_THRESHOLD_MS: u32 = 3000; // consider reconnect if one write blocks >3s

pub async fn send_message(
    socket: &mut TcpSocket<'_>,
    msg_type: MessageType,
    payload: &[u8],
    sequence: u32,
    flush_now: bool,
) -> Result<u32, TcpError> {
    let header = ProtoHeader {
        msg_type,
        length: payload.len() as u32,
        sequence,
    };
    let mut header_buf = [0u8; HEADER_LEN];
    header.encode_into(&mut header_buf);
    let start = Instant::now();
    // Block until the full header and payload are written. This applies backpressure
    // through the TCP/IP stack instead of short manual retries that can hammer WiFi.
    socket.write_all(&header_buf).await?;
    if !payload.is_empty() {
        socket.write_all(payload).await?;
    }
    if flush_now {
        // Flush can exert pressure on the lower layers; avoid doing it per-packet.
        // We only flush on keepalives and periodically during audio streaming.
        let _ = socket.flush().await;
    }
    Ok(start.elapsed().as_millis() as u32)
}

/// Send audio data directly to the socket without a heap-allocated intermediate
/// buffer. Samples are serialized through a small stack buffer to avoid competing
/// with WiFi for internal SRAM via the global allocator.
pub async fn send_audio(
    socket: &mut TcpSocket<'_>,
    audio: &crate::domain::AudioData,
    sequence: u32,
) -> Result<(u32, usize), TcpError> {
    let payload_len = audio.len() * audio.bytes_per_sample();
    let header = ProtoHeader {
        msg_type: MessageType::Audio,
        length: payload_len as u32,
        sequence,
    };
    let mut header_buf = [0u8; HEADER_LEN];
    header.encode_into(&mut header_buf);
    let start = Instant::now();
    socket.write_all(&header_buf).await?;

    // Write samples through a stack buffer (no heap allocation).
    const BUF_LEN: usize = 512;
    let mut tmp = [0u8; BUF_LEN];
    let mut pos = 0usize;

    match audio {
        crate::domain::AudioData::Sample16(data) => {
            for &s in data.iter() {
                tmp[pos..pos + 2].copy_from_slice(&s.to_le_bytes());
                pos += 2;
                if pos + 2 > BUF_LEN {
                    socket.write_all(&tmp[..pos]).await?;
                    pos = 0;
                }
            }
        }
        crate::domain::AudioData::Sample24(data) => {
            for &s in data.iter() {
                tmp[pos..pos + 4].copy_from_slice(&s.to_le_bytes());
                pos += 4;
                if pos + 4 > BUF_LEN {
                    socket.write_all(&tmp[..pos]).await?;
                    pos = 0;
                }
            }
        }
    }
    if pos > 0 {
        socket.write_all(&tmp[..pos]).await?;
    }

    Ok((start.elapsed().as_millis() as u32, payload_len))
}

/// Network transmitter task - sends audio data over WiFi/TCP using wire-protocol
#[embassy_executor::task]
pub async fn network_transmitter(
    stack: &'static embassy_net::Stack<'static>,
    net_out_channel: &'static NetOutChannel,
    net_ready: watch::DynSender<'static, bool>,
    recording_state: watch::DynSender<'static, bool>,
    connected_event: &'static Signal<NoopRawMutex, ()>,
    server_ip: [u8; 4],
    server_port: u16,
    device_id: &'static str,
    protocol_version: &'static str,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    _chunk_samples: usize,
    recording_mode: RecordingModeKind,
) {
    let mut rx_buffer = [0; TCP_RX_BUF_SIZE];
    let mut tx_buffer = [0; TCP_TX_BUF_SIZE];

    // Wait for initial WiFi connection
    connected_event.wait().await;
    connected_event.reset();

    // Wait for link up
    loop {
        if stack.is_link_up() {
            break;
        }
        Timer::after(Duration::from_millis(500)).await;
    }

    // Small delay to ensure network is fully ready
    Timer::after(Duration::from_secs(2)).await;

    // Reconnection loop
    loop {
        if !stack.is_link_up() {
            connected_event.wait().await;
            connected_event.reset();
            continue;
        }

        let mut socket = TcpSocket::new(*stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(SOCKET_TIMEOUT));
        socket.set_keep_alive(Some(SOCKET_KEEPALIVE));

        let ip = Ipv4Addr::from(server_ip);
        if let Err(_e) = socket.connect((ip, server_port)).await {
            // Backoff and retry
            Timer::after(RECONNECT_DELAY).await;
            continue;
        }

        // Send HELLO and METADATA
        let hello = wire_protocol::Hello {
            device_id,
            version: protocol_version,
        };
        let mut hbuf = [0u8; 64];
        if let Ok(n) = hello.encode_into(&mut hbuf) {
            let _ = send_message(&mut socket, MessageType::Hello, &hbuf[..n], 0, true).await;
        }

        let meta = wire_protocol::Metadata {
            sample_rate,
            channels: channels as u16,
            bits_per_sample: bits_per_sample as u16,
            recording_mode,
        };
        let mut mbuf = [0u8; 32];
        if let Ok(n) = meta.encode_into(&mut mbuf) {
            let _ = send_message(&mut socket, MessageType::Metadata, &mbuf[..n], 0, true).await;
        }

        // Mark network ready for orchestrator (after HELLO/METADATA)
        net_ready.send(true);

        // Reset reconnect delay on success
        let mut sequence_counter = 0u32;
        let mut last_keepalive = embassy_time::Instant::now();
        let mut last_health = embassy_time::Instant::now();
        let mut reconnects: u32 = 0;
        let mut stalls: u32 = 0;
        let mut bytes_sent: u64 = 0;
        let mut packets_sent: u64 = 0;
        let mut total_backoff_ms: u64 = 0;
        // Instrumentation: backlog statistics between health logs
        let mut backlog_max: usize = 0;
        let mut backlog_sum: u64 = 0;
        let mut backlog_samples: u64 = 0;

        loop {
            // Keepalive
            if last_keepalive.elapsed() >= KEEPALIVE_INTERVAL {
                if let Ok(backoff) = send_message(
                    &mut socket,
                    MessageType::Keepalive,
                    &[],
                    sequence_counter,
                    true,
                )
                .await
                {
                    total_backoff_ms = total_backoff_ms.saturating_add(backoff as u64);
                } else {
                    break;
                }
                sequence_counter = sequence_counter.wrapping_add(1);
                last_keepalive = embassy_time::Instant::now();
            }

            // Try to receive a NetOut; time out to keep keepalives flowing when idle
            // Sample current backlog
            let bl = crate::infra::instrumentation::backlog();
            if bl > backlog_max {
                backlog_max = bl;
            }
            backlog_sum = backlog_sum.saturating_add(bl as u64);
            backlog_samples = backlog_samples.saturating_add(1);

            let item =
                match with_timeout(Duration::from_millis(250), net_out_channel.receive()).await {
                    Ok(i) => i,
                    Err(_) => {
                        continue;
                    }
                };
            // Dequeue observed
            instrumentation::inc_deq(1);

            match item {
                NetOut::Ctrl(ctrl) => {
                    let mut buf = [0u8; 64];
                    let (msg_type, n) = match ctrl {
                        CtrlMsg::Segment {
                            cycle_id,
                            label,
                            prev_reason,
                            plan_ms,
                        } => {
                            let m = SegmentMsg {
                                cycle_id,
                                label,
                                prev_reason,
                                plan_ms,
                            };
                            let n = m.encode_into(&mut buf).unwrap_or(0);
                            (MessageType::Segment, n)
                        }
                    };
                    match send_message(&mut socket, msg_type, &buf[..n], sequence_counter, true)
                        .await
                    {
                        Ok(backoff) => {
                            packets_sent = packets_sent.saturating_add(1);
                            total_backoff_ms = total_backoff_ms.saturating_add(backoff as u64);
                            sequence_counter = sequence_counter.wrapping_add(1);
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
                NetOut::Audio(audio) => {
                    match send_audio(&mut socket, &audio, sequence_counter).await {
                        Ok((backoff, payload_len)) => {
                            packets_sent = packets_sent.saturating_add(1);
                            bytes_sent = bytes_sent.saturating_add(payload_len as u64);
                            total_backoff_ms = total_backoff_ms.saturating_add(backoff as u64);
                            if backoff > STALL_BACKOFF_THRESHOLD_MS {
                                break;
                            }
                            sequence_counter = sequence_counter.wrapping_add(1);
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
            }

            // Periodic health log
            if last_health.elapsed() >= Duration::from_secs(30) {
                let backlog_avg = if backlog_samples > 0 {
                    backlog_sum / backlog_samples
                } else {
                    0
                };
                log::info!(
                    "Net: pkts={} bytes={}KB backoff_ms={} stalls={} reconnects={} drops={} bl_max={} bl_avg={}",
                    packets_sent,
                    bytes_sent / 1024,
                    total_backoff_ms,
                    stalls,
                    reconnects,
                    0u64,
                    backlog_max,
                    backlog_avg
                );
                last_health = embassy_time::Instant::now();
                // reset rolling counters (keep totals if desired)
                packets_sent = 0;
                bytes_sent = 0;
                total_backoff_ms = 0;
                stalls = 0;
                backlog_max = 0;
                backlog_sum = 0;
                backlog_samples = 0;
            }

            // Loop continues while socket healthy; any send error breaks out
        }
        // On disconnect/error, pause recording and loop back to reconnect
        recording_state.send(false);
        net_ready.send(false);
        reconnects = reconnects.saturating_add(1);
        let _ = reconnects; // keep compiler happy about the updated value
        // Try to flush any remaining coalesced data (best-effort)
        // No access to socket here; coalesce will be dropped.
        Timer::after(RECONNECT_DELAY).await;
    }
}
