//! `nn-rs eval` — evaluate a TinyConv checkpoint.
//!
//! Loads a checkpoint saved by the training loop, walks the requested
//! split, computes per-file and (optionally) per-window binary metrics,
//! and writes `metrics.json` / `calibration.json` / `threshold.json`
//! next to the checkpoint.
//!
//! Typical invocation:
//!
//! ```sh
//! # Apple Silicon (Metal):
//! cargo run --release -p nn-rs --no-default-features \
//!     --features "std,metal" --bin eval -- \
//!     --config nn-rs/configs/robust_session.yaml \
//!     --checkpoint runs/robust_session/checkpoints/best \
//!     --manifest path/to/manifest.jsonl \
//!     --split val \
//!     [--threshold 0.5] [--no-calibrate] \
//!     [--window-metrics] [--audit] \
//!     [--preds-csv preds.csv] [--window-preds-csv preds_win.csv]
//!
//! # Linux / NVIDIA (native CUDA):
//! cargo run --release -p nn-rs --no-default-features \
//!     --features "std,cuda" --bin eval -- \
//!     --config nn-rs/configs/robust_session.yaml \
//!     --checkpoint runs/robust_session/checkpoints/best
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cubecl::Runtime;

use nn_rs::data::Split;
use nn_rs::eval::{AggregateMode, EvalOptions, Evaluator};
use nn_rs::train::TrainAppConfig;
use nn_rs::train::runner::{SelectedDevice, SelectedRuntime};

fn usage() -> &'static str {
    "usage: eval --config <yaml> --checkpoint <path> \
     [--manifest <path>] [--split val|test] [--aggregate mean|max] \
     [--threshold <float>] [--no-calibrate] \
     [--window-metrics] [--audit] [--audit-top-n N] \
     [--metrics-json <path>] [--calibration-json <path>] \
     [--preds-csv <path>] [--window-preds-csv <path>] \
     [--batch-size N] [--num-workers N]"
}

#[derive(Default)]
struct Cli {
    config: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    manifest: Option<String>,
    split: Option<String>,
    aggregate: Option<String>,
    threshold: Option<f32>,
    no_calibrate: bool,
    window_metrics: bool,
    audit: bool,
    audit_top_n: usize,
    metrics_json: Option<PathBuf>,
    calibration_json: Option<PathBuf>,
    preds_csv: Option<PathBuf>,
    window_preds_csv: Option<PathBuf>,
    batch_size: Option<usize>,
    num_workers: Option<usize>,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli {
        audit_top_n: 10,
        ..Default::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => cli.config = Some(PathBuf::from(need(&mut args, "--config")?)),
            "--checkpoint" => {
                cli.checkpoint = Some(PathBuf::from(need(&mut args, "--checkpoint")?))
            }
            "--manifest" => cli.manifest = Some(need(&mut args, "--manifest")?),
            "--split" => cli.split = Some(need(&mut args, "--split")?),
            "--aggregate" => cli.aggregate = Some(need(&mut args, "--aggregate")?),
            "--threshold" => {
                cli.threshold = Some(
                    need(&mut args, "--threshold")?
                        .parse()
                        .map_err(|e| format!("--threshold: {e}"))?,
                )
            }
            "--calibrate" => {} // default; kept for symmetry with --no-calibrate
            "--no-calibrate" => cli.no_calibrate = true,
            "--window-metrics" => cli.window_metrics = true,
            "--audit" => cli.audit = true,
            "--audit-top-n" => {
                cli.audit_top_n = need(&mut args, "--audit-top-n")?
                    .parse()
                    .map_err(|e| format!("--audit-top-n: {e}"))?
            }
            "--metrics-json" => {
                cli.metrics_json = Some(PathBuf::from(need(&mut args, "--metrics-json")?))
            }
            "--calibration-json" => {
                cli.calibration_json =
                    Some(PathBuf::from(need(&mut args, "--calibration-json")?))
            }
            "--preds-csv" => {
                cli.preds_csv = Some(PathBuf::from(need(&mut args, "--preds-csv")?))
            }
            "--window-preds-csv" => {
                cli.window_preds_csv =
                    Some(PathBuf::from(need(&mut args, "--window-preds-csv")?))
            }
            "--batch-size" => {
                cli.batch_size = Some(
                    need(&mut args, "--batch-size")?
                        .parse()
                        .map_err(|e| format!("--batch-size: {e}"))?,
                )
            }
            "--num-workers" => {
                cli.num_workers = Some(
                    need(&mut args, "--num-workers")?
                        .parse()
                        .map_err(|e| format!("--num-workers: {e}"))?,
                )
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {a}\n{}", usage())),
        }
    }
    if cli.config.is_none() {
        return Err(format!("missing --config\n{}", usage()));
    }
    if cli.checkpoint.is_none() {
        return Err(format!("missing --checkpoint\n{}", usage()));
    }
    Ok(cli)
}

fn need<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{name} takes an argument"))
}

fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let config_path = cli.config.clone().expect("checked above");
    let mut cfg = match TrainAppConfig::from_yaml_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load {}: {e}", config_path.display());
            return ExitCode::from(2);
        }
    };
    if let Some(m) = &cli.manifest {
        cfg.dataset.manifest_path = Some(m.clone());
    }
    if let Some(b) = cli.batch_size {
        cfg.dm.batch_size = b;
    }
    if let Some(n) = cli.num_workers {
        cfg.dm.num_workers = Some(n);
    }
    resolve_relative_manifest(&mut cfg, &config_path);

    let split = match cli.split.as_deref().unwrap_or("val") {
        "val" => Split::Val,
        "test" => Split::Test,
        "train" => Split::Train,
        other => {
            eprintln!("unknown --split {other} (expected val|test|train)");
            return ExitCode::from(2);
        }
    };
    let aggregate = match cli.aggregate.as_deref().unwrap_or("mean") {
        s => match AggregateMode::parse(s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        },
    };

    let checkpoint = cli.checkpoint.clone().expect("checked above");
    let ckpt_dir = checkpoint
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let metrics_json = cli
        .metrics_json
        .clone()
        .unwrap_or_else(|| ckpt_dir.join("metrics.json"));
    let calibration_json = cli
        .calibration_json
        .clone()
        .unwrap_or_else(|| ckpt_dir.join("calibration.json"));

    let opts = EvalOptions {
        checkpoint,
        split,
        aggregate,
        threshold: cli.threshold,
        calibrate: !cli.no_calibrate,
        window_metrics: cli.window_metrics,
        audit: cli.audit,
        audit_top_n: cli.audit_top_n,
        metrics_json,
        calibration_json,
        preds_csv: cli.preds_csv,
        window_preds_csv: cli.window_preds_csv,
    };

    let device = SelectedDevice::default();
    let client = <SelectedRuntime as Runtime>::client(&device);
    let evaluator: Evaluator<SelectedRuntime, f32, i32, u8> =
        Evaluator::new(cfg, client, device);

    match evaluator.evaluate(&opts) {
        Ok(out) => {
            println!(
                "\nevaluation complete — files={} windows={} macro_f1={:.4} acc={:.4} threshold={:.3}",
                out.num_files,
                out.num_windows,
                out.file.macro_f1,
                out.file.accuracy,
                out.threshold,
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("evaluation failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn resolve_relative_manifest(cfg: &mut TrainAppConfig, config_path: &Path) {
    let Some(m) = cfg.dataset.manifest_path.clone() else {
        return;
    };
    let p = PathBuf::from(&m);
    if p.is_absolute() {
        return;
    }
    if let Some(parent) = config_path.parent() {
        cfg.dataset.manifest_path = Some(parent.join(&p).to_string_lossy().to_string());
    }
}
