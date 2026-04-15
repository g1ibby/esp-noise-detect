use anyhow::Result;
use clap::Parser;
use host_tools::{run, Args};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing subscriber with env override and --verbose fallback
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(if args.verbose { "debug" } else { "info" }))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_level(true)
        .compact()
        .init();

    run(args).await
}
