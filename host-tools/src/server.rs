use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use tokio::{
    io::AsyncReadExt,
    net::{TcpListener, TcpStream},
    sync::Mutex,
    time::timeout,
};
use tracing::{debug, error, info, warn};
use wire_protocol::{
    Header, Hello, MessageType, Metadata, RecordingModeKind, SegmentMsg, StopReason,
};

use crate::{
    session::{AudioSession, RECORDING_DURATION_SECS},
    storage::{Sink, WavSink},
    util::{format_bytes, format_duration},
};

pub const DEFAULT_PORT: u16 = 3000;
pub const MAX_MESSAGE_SIZE: usize = 8192;
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

type ServerState = Arc<Mutex<HashMap<SocketAddr, AudioSession>>>;
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[derive(Parser, Debug, Clone)]
#[command(name = "wifi_audio_server")]
#[command(about = "TCP server for receiving WiFi audio streams from ESP32-S3")]
pub struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    pub port: u16,

    /// Output directory for WAV files
    #[arg(short, long, default_value = "./recordings")]
    pub output_dir: PathBuf,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

pub async fn run(args: Args) -> Result<()> {
    info!("═══════════════════════════════════════════════════════");
    info!("   WiFi Audio Server for ESP32-S3 (5-min recordings)");
    info!("     Press Ctrl+C to gracefully stop and save files");
    info!("═══════════════════════════════════════════════════════");

    if !args.output_dir.exists() {
        std::fs::create_dir_all(&args.output_dir).context("Failed to create output directory")?;
    }
    info!("🗂️  Output directory: {}", args.output_dir.display());

    let state: ServerState = Arc::new(Mutex::new(HashMap::new()));
    let addr = format!("[::]:{}", args.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    info!("🚀 Server listening on: {}", addr);
    info!("💡 Connect your ESP32-S3 to this server");
    info!(
        "🎙️  Recording {} minute segments (or until Ctrl+C)",
        RECORDING_DURATION_SECS / 60
    );

    // Graceful shutdown
    let state_clone = Arc::clone(&state);
    let output_dir_for_shutdown = args.output_dir.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("\n\n💫 Graceful shutdown requested (Ctrl+C detected)");
        info!("💾 Saving partial recordings...");
        SHUTDOWN.store(true, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_secs(2)).await;
        let mut sessions = state_clone.lock().await;
        let session_count = sessions.len();
        if session_count > 0 {
            info!(
                "💾 Processing {} active recording session(s)...",
                session_count
            );
            let mut sink = WavSink::new(output_dir_for_shutdown);
            for (addr, mut session) in sessions.drain() {
                if session.has_audio_samples() {
                    match session.save_current_segment(
                        &mut sink,
                        session.current_label,
                        session.current_cycle,
                        Some(StopReason::Timeout),
                    ) {
                        Ok(Some(_)) => {
                            info!("✅ [{}] Saved partial recording on shutdown", addr)
                        }
                        Ok(None) => {}
                        Err(e) => {
                            error!("❌ [{}] Failed to save shutdown recording: {}", addr, e)
                        }
                    }
                }
            }
        }
        info!("✅ Shutdown complete. All recordings saved.");
        std::process::exit(0);
    });

    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept().await {
            Ok((stream, addr)) => {
                let state_clone = Arc::clone(&state);
                let output_dir = args.output_dir.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, addr, state_clone, output_dir).await {
                        warn!("⏰ [{}] Connection error: {}", addr, e);
                    }
                    info!("🔌 [{}] Connection closed", addr);
                });
            }
            Err(e) => error!("❌ Failed to accept connection: {}", e),
        }
    }
    Ok(())
}

async fn handle_client(
    mut stream: TcpStream,
    addr: SocketAddr,
    state: ServerState,
    output_dir: PathBuf,
) -> Result<()> {
    info!("🔌 New connection from: {}", addr);

    let session = AudioSession::new(format!("esp32_{}", addr.port()));

    {
        let mut sessions = state.lock().await;
        sessions.insert(addr, session.clone());
    }

    let mut sink = WavSink::new(output_dir.clone());
    loop {
        let mut header_buf = [0u8; 8];
        match timeout(READ_TIMEOUT, stream.read_exact(&mut header_buf)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                finalize_on_disconnect(addr, &state, &mut sink).await;
                return Err(e.into());
            }
            Err(_) => {
                finalize_on_disconnect(addr, &state, &mut sink).await;
                return Err(anyhow!("Read timeout"));
            }
        }

        let header = Header::parse(&header_buf)?;
        if header.msg_type != MessageType::Audio {
            debug!(
                "📨 [{}] Received {:?} message (seq: {}, len: {})",
                addr, header.msg_type, header.sequence, header.length
            );
        }

        // Enforce maximum allowed payload size
        if header.length as usize > MAX_MESSAGE_SIZE {
            warn!(
                "🚫 [{}] Message too large: {} bytes (max: {})",
                addr, header.length, MAX_MESSAGE_SIZE
            );
            finalize_on_disconnect(addr, &state, &mut sink).await;
            return Err(anyhow!(
                "message too large: {} > {}",
                header.length,
                MAX_MESSAGE_SIZE
            ));
        }

        // Read payload
        let payload = if header.length > 0 {
            let mut payload_buf = vec![0u8; header.length as usize];
            match timeout(READ_TIMEOUT, stream.read_exact(&mut payload_buf)).await {
                Ok(Ok(_)) => payload_buf,
                Ok(Err(e)) => {
                    finalize_on_disconnect(addr, &state, &mut sink).await;
                    return Err(e.into());
                }
                Err(_) => {
                    finalize_on_disconnect(addr, &state, &mut sink).await;
                    return Err(anyhow!("Payload read timeout"));
                }
            }
        } else {
            Vec::new()
        };

        // Update session based on message
        let mut sessions = state.lock().await;
        let entry = sessions.get_mut(&addr).expect("session exists");

        match header.msg_type {
            MessageType::Hello => {
                let hello = Hello::parse(&payload)?;
                info!(
                    "👋 [{}] Device: {} (version: {})",
                    addr, hello.device_id, hello.version
                );
                entry.device_id = hello.device_id.to_string();
            }
            MessageType::Metadata => {
                let meta = Metadata::parse(&payload)?;
                info!(
                    "📝 [{}] Audio format: {}Hz, {} channels, {} bits | mode={:?}",
                    addr,
                    meta.sample_rate,
                    meta.channels,
                    meta.bits_per_sample,
                    meta.recording_mode
                );
                entry.metadata = Some(meta);
                entry.recording_mode = Some(meta.recording_mode);
            }
            MessageType::Segment => {
                let seg = SegmentMsg::parse(&payload)?;
                info!(
                    "🔁 [{}] Segment boundary -> cycle={} label={:?} prev_reason={:?} plan_ms={}",
                    addr, seg.cycle_id, seg.label, seg.prev_reason, seg.plan_ms
                );
                let prev_label = entry.current_label;
                let prev_cycle = entry.current_cycle;
                // Close current chunk to avoid mixing across labels
                if entry.has_audio_samples() {
                    if let Err(e) = entry.save_current_segment(
                        &mut sink,
                        prev_label,
                        prev_cycle,
                        Some(seg.prev_reason),
                    ) {
                        warn!("⚠️ [{}] Failed to roll chunk on boundary: {}", addr, e);
                    } else {
                        // Ensure next save cannot overwrite the previous file even if more samples arrive
                        entry.chunk_number = entry.chunk_number.saturating_add(1);
                    }
                }
                // Reset chunk numbering only when the segment truly changes (label or cycle)
                if prev_label != Some(seg.label) || prev_cycle != Some(seg.cycle_id) {
                    entry.chunk_number = 0;
                }
                entry.current_label = Some(seg.label);
                entry.current_cycle = Some(seg.cycle_id);
                entry.current_segment_start = std::time::SystemTime::now();
            }
            MessageType::Audio => {
                // Ensure we have metadata so we know how to parse payload
                let bits_per_sample = match &entry.metadata {
                    Some(m) => m.bits_per_sample,
                    None => {
                        finalize_on_disconnect(addr, &state, &mut sink).await;
                        return Err(anyhow!(
                            "audio received before metadata (len={})",
                            payload.len()
                        ));
                    }
                };
                let bytes_per_sample = match bits_per_sample {
                    16 => 2usize,
                    24 => 4usize, // transported as 32-bit little-endian
                    other => {
                        finalize_on_disconnect(addr, &state, &mut sink).await;
                        return Err(anyhow!("unsupported bits_per_sample: {}", other));
                    }
                };
                if payload.len() % bytes_per_sample != 0 {
                    finalize_on_disconnect(addr, &state, &mut sink).await;
                    return Err(anyhow!(
                        "audio payload not divisible by {} bytes: {}",
                        bytes_per_sample,
                        payload.len()
                    ));
                }
                entry.add_audio_data(&payload)?;
                if entry.should_auto_save() {
                    // In pump_gated mode, treat auto-save as SafetyCutoff; continuous remains None
                    let reason = match entry.recording_mode {
                        Some(RecordingModeKind::PumpGated) => Some(StopReason::SafetyCutoff),
                        _ => None,
                    };
                    if let Err(e) = entry.save_current_segment(
                        &mut sink,
                        entry.current_label,
                        entry.current_cycle,
                        reason,
                    ) {
                        warn!("⚠️ [{}] Failed to auto-save segment: {}", addr, e);
                    }
                }
            }
            MessageType::End => {
                info!("🏁 [{}] End marker received (saving current segment)", addr);
                if entry.should_auto_save() || entry.has_audio_samples() {
                    if let Err(e) = entry.save_current_segment(
                        &mut sink,
                        entry.current_label,
                        entry.current_cycle,
                        None,
                    ) {
                        warn!("⚠️ [{}] Failed to save end segment: {}", addr, e);
                    }
                }
            }
            MessageType::Keepalive => {
                entry.last_keepalive = std::time::SystemTime::now();
                if header.sequence % 100 == 0 {
                    debug!(
                        "💓 [{}] Keepalive received (seq: {})",
                        addr, header.sequence
                    );
                }
            }
            MessageType::PumpStatus => {
                // The main server ignores pump status messages; they are for the monitor.
                debug!("💦 [{}] Pump status received (ignoring)", addr);
            }
        }
        drop(sessions);
    }
}

async fn finalize_on_disconnect(addr: SocketAddr, state: &ServerState, sink: &mut impl Sink) {
    let mut sessions = state.lock().await;
    if let Some(mut session) = sessions.remove(&addr) {
        if session.has_audio_samples() {
            match session.save_current_segment(
                sink,
                session.current_label,
                session.current_cycle,
                Some(StopReason::Timeout),
            ) {
                Ok(Some(_)) => info!("✅ [{}] Saved partial recording on disconnect", addr),
                Ok(None) => {}
                Err(e) => error!(
                    "❌ [{}] Failed to save recording on disconnect: {}",
                    addr, e
                ),
            }
        }
        info!("✅ [{}] Streaming session complete:", addr);
        // Report total audio duration if metadata is available
        if let Some(meta) = &session.metadata {
            let total_samples = session.total_samples_saved + session.sample_count();
            let total_secs =
                total_samples as f32 / (meta.sample_rate as f32 * meta.channels as f32);
            info!("   • Total audio: {}", format_duration(total_secs));
        } else {
            let total_duration = session.session_start.elapsed().unwrap_or_default();
            info!(
                "   • Total time: {}",
                format_duration(total_duration.as_secs_f32())
            );
        }
        info!("   • Files saved: {}", session.total_files_saved);
        info!("   • Packets received: {}", session.packets_received);
        info!(
            "   • Data processed: {}",
            format_bytes(session.bytes_received)
        );
    }
}
