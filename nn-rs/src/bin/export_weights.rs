//! `nn-rs export_weights` — dump a trained TinyConv checkpoint to a
//! PyTorch-compatible `.safetensors` file.
//!
//! The Python ONNX bridge (`python/burn_to_onnx.py`) loads the
//! safetensors blob into a reconstructed PyTorch `TinyConv` and runs
//! `torch.onnx.export` + `onnxslim`.
//!
//! Key remapping — Burn stores parameters as
//! `blocks.{N}.{conv_down|bn_down|conv_refine|bn_refine}.…`, but
//! PyTorch's `TinyConv.features` uses sequential indices:
//! `features.{2N}.0.weight` (conv_down), `features.{2N}.1.weight`
//! (bn_down gamma), etc. Rename rules are generated at runtime from
//! the config's `channels` length.
//!
//! `BurnToPyTorchAdapter` handles within-tensor adaptations:
//! * Linear weight transpose `[d_in, d_out]` → `[d_out, d_in]`.
//! * BatchNorm `gamma` → `weight`, `beta` → `bias`.
//!
//! Typical invocation:
//!
//! ```sh
//! # Apple Silicon (Metal):
//! cargo run --release -p nn-rs --no-default-features \
//!     --features "std,metal" --bin export_weights -- \
//!     --config nn-rs/configs/robust_session.yaml \
//!     --checkpoint runs/robust_session/checkpoints/best \
//!     --out runs/robust_session/export/tinyconv.safetensors
//!
//! # Linux / NVIDIA (native CUDA):
//! cargo run --release -p nn-rs --no-default-features \
//!     --features "std,cuda" --bin export_weights -- \
//!     --config nn-rs/configs/robust_session.yaml \
//!     --checkpoint runs/robust_session/checkpoints/best \
//!     --out runs/robust_session/export/tinyconv.safetensors
//! ```
//!
//! The `--checkpoint` path is the stem passed to Burn's
//! `CompactRecorder` (no `.mpk` extension — Burn appends it). We accept
//! either form and strip the extension ourselves.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::module::Module;
use burn::prelude::Tensor;
use burn::record::{CompactRecorder, Recorder};
use burn_store::{BurnToPyTorchAdapter, KeyRemapper, ModuleSnapshot, SafetensorsStore};
use cubecl::Runtime;

use nn_rs::model::{TinyConv, TinyConvConfig};
use nn_rs::train::TrainAppConfig;
use nn_rs::train::runner::{SelectedCube, SelectedDevice, SelectedRuntime};

fn usage() -> &'static str {
    "usage: export_weights --config <yaml> --checkpoint <path> --out <safetensors> \
     [--verify-logits <json-path>]"
}

struct Cli {
    config: PathBuf,
    checkpoint: PathBuf,
    out: PathBuf,
    /// Optional JSON dump of a forward-pass on a canary input — used by
    /// `burn_to_onnx.py --compare-logits` to verify the exported
    /// weights reproduce the model's output after a safetensors
    /// roundtrip.
    verify_logits: Option<PathBuf>,
}

fn parse_args() -> Result<Cli, String> {
    let mut config: Option<PathBuf> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut verify_logits: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = Some(PathBuf::from(need(&mut args, "--config")?)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(need(&mut args, "--checkpoint")?)),
            "--out" => out = Some(PathBuf::from(need(&mut args, "--out")?)),
            "--verify-logits" => {
                verify_logits = Some(PathBuf::from(need(&mut args, "--verify-logits")?))
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {a}\n{}", usage())),
        }
    }
    let config = config.ok_or_else(|| format!("missing --config\n{}", usage()))?;
    let checkpoint = checkpoint.ok_or_else(|| format!("missing --checkpoint\n{}", usage()))?;
    let out = out.ok_or_else(|| format!("missing --out\n{}", usage()))?;
    Ok(Cli { config, checkpoint, out, verify_logits })
}

fn need<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{name} takes an argument"))
}

/// `CompactRecorder::load` expects a stem (it appends `.mpk`). If the
/// user passed the full file, strip the extension.
fn strip_mpk(path: &Path) -> PathBuf {
    if path.extension().map(|s| s == "mpk").unwrap_or(false) {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

/// Build the remap rules that turn Burn's `blocks.N.{...}.X` paths
/// into `features.{2N|2N+1}.{0|1}.X`. One pattern per (block, role)
/// pair.
fn build_remapper(n_blocks: usize) -> KeyRemapper {
    let mut remap = KeyRemapper::new();
    for n in 0..n_blocks {
        let down_idx = 2 * n;
        let refine_idx = 2 * n + 1;
        remap = remap
            .add_pattern(
                format!(r"^blocks\.{n}\.conv_down\."),
                format!("features.{down_idx}.0."),
            )
            .expect("static regex should compile")
            .add_pattern(
                format!(r"^blocks\.{n}\.bn_down\."),
                format!("features.{down_idx}.1."),
            )
            .expect("static regex should compile")
            .add_pattern(
                format!(r"^blocks\.{n}\.conv_refine\."),
                format!("features.{refine_idx}.0."),
            )
            .expect("static regex should compile")
            .add_pattern(
                format!(r"^blocks\.{n}\.bn_refine\."),
                format!("features.{refine_idx}.1."),
            )
            .expect("static regex should compile");
    }
    remap
}

fn run(cli: Cli) -> std::io::Result<()> {
    let cfg = TrainAppConfig::from_yaml_file(&cli.config).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("failed to load {}: {e}", cli.config.display()),
        )
    })?;

    let device = SelectedDevice::default();
    let _client = <SelectedRuntime as Runtime>::client(&device);
    let model_cfg = TinyConvConfig {
        channels: cfg.model.channels.clone(),
        dropout: cfg.model.dropout,
        n_classes: cfg.dataset.class_names.len(),
    };
    let ckpt_stem = strip_mpk(&cli.checkpoint);
    let record = CompactRecorder::new()
        .load::<<TinyConv<SelectedCube> as Module<SelectedCube>>::Record>(
            ckpt_stem.clone(),
            &device,
        )
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("load checkpoint {}: {e}", ckpt_stem.display()),
            )
        })?;
    let model: TinyConv<SelectedCube> = model_cfg.init::<SelectedCube>(&device).load_record(record);

    if let Some(parent) = cli.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let remap = build_remapper(cfg.model.channels.len());
    let mut store = SafetensorsStore::from_file(cli.out.clone())
        .with_to_adapter(BurnToPyTorchAdapter)
        .remap(remap)
        .overwrite(true)
        .metadata("producer", "nn-rs export_weights")
        .metadata("model", "TinyConv")
        .metadata("channels", format!("{:?}", cfg.model.channels))
        .metadata("n_classes", cfg.dataset.class_names.len().to_string());

    model.save_into(&mut store).map_err(|e| {
        std::io::Error::other(format!("save safetensors {}: {e}", cli.out.display()))
    })?;

    println!("Wrote PyTorch-format weights to {}", cli.out.display());

    if let Some(path) = &cli.verify_logits {
        // Canary forward: zero-input `(1, 1, n_mels, n_frames)`. A zero
        // input still exercises every layer (BatchNorm adds the bias
        // and subtracts running_mean/σ, the Linear head adds its
        // bias), so any weight-mismatch between the Burn record and
        // the safetensors roundtrip surfaces here.
        let n_mels = cfg.features.n_mels;
        let n_frames = mel_frame_count(&cfg);
        let input: Tensor<SelectedCube, 4> =
            Tensor::<SelectedCube, 4>::zeros([1, 1, n_mels, n_frames], &device);
        let logits = model.forward(input);
        let dims = logits.dims();
        let logits_vec: Vec<f32> = logits
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("into_data should yield f32");
        write_logits_json(path, &logits_vec, &dims)?;
        println!(
            "Wrote canary logits ({}x{}) to {}",
            dims[0], dims[1], path.display(),
        );
    }

    println!(
        "Next step: python/burn_to_onnx.py --weights {} --config {} --out-dir <dir>",
        cli.out.display(),
        cli.config.display(),
    );
    Ok(())
}

/// Number of mel frames emitted by `MelExtractor` for a 1-window input
/// at `sample_rate` with `center=True` reflect padding.
fn mel_frame_count(cfg: &TrainAppConfig) -> usize {
    let window_samples =
        (cfg.dataset.window_s * cfg.dataset.sample_rate as f32).round() as usize;
    let hop = (cfg.features.hop_length_ms / 1000.0 * cfg.dataset.sample_rate as f32).round()
        as usize;
    let padded = if cfg.features.center {
        window_samples + cfg.features.n_fft
    } else {
        window_samples
    };
    // floor((padded - n_fft) / hop) + 1
    (padded.saturating_sub(cfg.features.n_fft) / hop) + 1
}

fn write_logits_json(path: &Path, logits: &[f32], dims: &[usize; 2]) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let values = logits
        .iter()
        .map(|v| format!("{v:.8e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let body = format!(
        "{{\n  \"shape\": [{}, {}],\n  \"input\": \"zeros\",\n  \"logits\": [{}]\n}}\n",
        dims[0], dims[1], values,
    );
    std::fs::write(path, body)
}

fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("export failed: {e}");
            ExitCode::FAILURE
        }
    }
}
