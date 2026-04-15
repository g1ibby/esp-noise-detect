use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::net::SocketAddr;
use tokio::{
    io::AsyncReadExt,
    net::{TcpListener, TcpStream},
    time::timeout,
};
use tracing::{debug, error, info, warn};
use wire_protocol::{Header, Hello, MessageType, PumpState, PumpStatusMsg};

pub const DEFAULT_PORT: u16 = 3000;
pub const MAX_MESSAGE_SIZE: usize = 1024;
pub const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Parser, Debug, Clone)]
#[command(name = "pump_monitor")]
#[command(about = "TCP server for receiving Pump Status from ESP32-S3")]
pub struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    pub port: u16,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with default "info" level if RUST_LOG is not set
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let args = Args::parse();

    info!("═══════════════════════════════════════════════════════");
    info!("   Pump Monitor Server for ESP32-S3");
    info!("     Press Ctrl+C to stop");
    info!("═══════════════════════════════════════════════════════");

    let addr = format!("[::]:{}", args.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    info!("🚀 Server listening on: {}", addr);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, addr).await {
                        warn!("⏰ [{}] Connection error: {}", addr, e);
                    }
                    info!("🔌 [{}] Connection closed", addr);
                });
            }
            Err(e) => error!("❌ Failed to accept connection: {}", e),
        }
    }
}

async fn handle_client(mut stream: TcpStream, addr: SocketAddr) -> Result<()> {
    info!("🔌 New connection from: {}", addr);

    loop {
        let mut header_buf = [0u8; 8];
        match timeout(READ_TIMEOUT, stream.read_exact(&mut header_buf)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err(anyhow!("Read timeout")),
        }

        let header = Header::parse(&header_buf)?;

        // Enforce maximum allowed payload size
        if header.length as usize > MAX_MESSAGE_SIZE {
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
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => return Err(anyhow!("Payload read timeout")),
            }
        } else {
            Vec::new()
        };

        match header.msg_type {
            MessageType::Hello => {
                let hello = Hello::parse(&payload)?;
                info!(
                    "👋 [{}] Device: {} (version: {})",
                    addr, hello.device_id, hello.version
                );
            }
            MessageType::PumpStatus => {
                let status_msg = PumpStatusMsg::parse(&payload)?;
                let status_str = match status_msg.status {
                    PumpState::On => "ON",
                    PumpState::Off => "OFF",
                };
                info!("💦 [{}] Pump Status Event: {}", addr, status_str);
            }
            MessageType::Keepalive => {
                if header.sequence % 10 == 0 {
                    debug!("💓 [{}] Keepalive", addr);
                }
            }
            _ => {
                debug!(
                    "📨 [{}] Received other message type: {:?}",
                    addr, header.msg_type
                );
            }
        }
    }
}
