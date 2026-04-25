//! `nn-rs train` — CLI for the TinyConv training loop.
//!
//! ```sh
//! # Apple Silicon (Metal):
//! cargo run --release -p nn-rs --no-default-features \
//!     --features "std,metal" --bin train -- \
//!     --config nn-rs/configs/robust_session.yaml \
//!     [--manifest /path/to/manifest.jsonl] \
//!     [--epochs 30] \
//!     [--batch-size 128] \
//!     [--artifact-dir runs/robust_session]
//!
//! # Linux / NVIDIA (native CUDA):
//! cargo run --release -p nn-rs --no-default-features \
//!     --features "std,cuda" --bin train -- \
//!     --config nn-rs/configs/robust_session.yaml
//! ```
//!
//! Flags override fields in the loaded YAML; unknown flags error out
//! with a usage message. See `nn-rs/src/train/config.rs` for the config
//! schema.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cubecl::Runtime;

use nn_rs::train::runner::{SelectedAutodiff, SelectedDevice, SelectedRuntime};
use nn_rs::train::{TrainAppConfig, Trainer};

fn usage() -> &'static str {
    "usage: train --config <yaml> [--manifest <path>] [--epochs N] \
     [--batch-size N] [--num-workers N] [--artifact-dir <path>] [--lr F] [--seed N] \
     [--profile-stages]"
}

fn parse_args() -> Result<Overrides, String> {
    let mut args = std::env::args().skip(1);
    let mut ov = Overrides::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => ov.config = Some(PathBuf::from(require(&mut args, "--config")?)),
            "--manifest" => ov.manifest = Some(require(&mut args, "--manifest")?),
            "--epochs" => ov.epochs = Some(parse_num(&mut args, "--epochs")?),
            "--batch-size" => ov.batch_size = Some(parse_num(&mut args, "--batch-size")?),
            "--num-workers" => ov.num_workers = Some(parse_num(&mut args, "--num-workers")?),
            "--artifact-dir" => ov.artifact_dir = Some(require(&mut args, "--artifact-dir")?),
            "--lr" => ov.lr = Some(parse_float(&mut args, "--lr")?),
            "--seed" => ov.seed = Some(parse_num(&mut args, "--seed")? as u64),
            "--profile-stages" => ov.profile_stages = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {a}\n{}", usage())),
        }
    }
    if ov.config.is_none() {
        return Err(format!("missing --config\n{}", usage()));
    }
    Ok(ov)
}

#[derive(Default)]
struct Overrides {
    config: Option<PathBuf>,
    manifest: Option<String>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    num_workers: Option<usize>,
    artifact_dir: Option<String>,
    lr: Option<f64>,
    seed: Option<u64>,
    profile_stages: bool,
}

fn require<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{name} takes an argument"))
}

fn parse_num<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<usize, String> {
    require(args, name)?
        .parse::<usize>()
        .map_err(|e| format!("{name}: {e}"))
}

fn parse_float<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<f64, String> {
    require(args, name)?
        .parse::<f64>()
        .map_err(|e| format!("{name}: {e}"))
}

fn apply_overrides(cfg: &mut TrainAppConfig, ov: &Overrides) {
    if let Some(m) = &ov.manifest {
        cfg.dataset.manifest_path = Some(m.clone());
    }
    if let Some(e) = ov.epochs {
        cfg.trainer.max_epochs = e;
    }
    if let Some(b) = ov.batch_size {
        cfg.dm.batch_size = b;
    }
    if let Some(n) = ov.num_workers {
        cfg.dm.num_workers = Some(n);
    }
    if let Some(a) = &ov.artifact_dir {
        cfg.trainer.artifact_dir = a.clone();
    }
    if let Some(lr) = ov.lr {
        cfg.optim.lr = lr;
    }
    if let Some(s) = ov.seed {
        cfg.trainer.seed = s;
    }
}

fn main() -> ExitCode {
    let ov = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let config_path = ov.config.clone().expect("checked above");
    let mut cfg = match TrainAppConfig::from_yaml_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load {}: {e}", config_path.display());
            return ExitCode::from(2);
        }
    };
    apply_overrides(&mut cfg, &ov);
    resolve_relative_manifest(&mut cfg, &config_path);

    let device = SelectedDevice::default();
    let client = <SelectedRuntime as Runtime>::client(&device);
    let trainer: Trainer<SelectedAutodiff, SelectedRuntime, f32, i32, u8> = Trainer::new(
        cfg,
        client,
        device.clone(),
        device,
    )
    .with_profile_stages(ov.profile_stages);

    match trainer.fit() {
        Ok(out) => {
            println!(
                "\ntraining complete — best epoch {} val_f1_macro={:.4} val_loss={:.4} val_acc={:.4}\n\
                 best checkpoint: {}",
                out.best_epoch,
                out.best_val_macro_f1,
                out.best_val_loss,
                out.best_val_acc,
                out.best_checkpoint.display(),
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("training failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// If the YAML's `dataset.manifest_path` is relative, resolve it against
/// the config file's parent directory (keeps configs portable).
fn resolve_relative_manifest(cfg: &mut TrainAppConfig, config_path: &Path) {
    let Some(m) = cfg.dataset.manifest_path.clone() else {
        return;
    };
    let p = PathBuf::from(&m);
    if p.is_absolute() {
        return;
    }
    if let Some(parent) = config_path.parent() {
        cfg.dataset.manifest_path = Some(
            parent
                .join(&p)
                .to_string_lossy()
                .to_string(),
        );
    }
}
