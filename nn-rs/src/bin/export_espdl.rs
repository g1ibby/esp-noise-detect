//! `nn-rs export_espdl` - native Burn checkpoint to ESP-DL exporter.
//!
//! This replaces the legacy `Burn -> safetensors -> PyTorch -> ONNX ->
//! esp-ppq Docker -> .espdl` path with one Rust process:
//!
//! 1. load the production `TinyConv` Burn checkpoint,
//! 2. lower it to `burn-espdl-export`'s model-agnostic IR,
//! 3. collect calibration mel windows from the manifest or read `.npy`
//!    calibration tensors from disk,
//! 4. hand the graph + windows to `burn-espdl-export`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn_espdl_export::EspdlExporter;
use cubecl::Runtime;
use nn_rs::data::{AudioBatcher, DatasetConfig, Split, WindowedAudioDataset, load_manifest};
use nn_rs::espdl::tinyconv_to_burn_graph;
use nn_rs::espdl_calib::collect_calibration_windows_from_npy;
use nn_rs::mel::{MelConfig, MelExtractor};
use nn_rs::model::{TinyConv, TinyConvConfig};
use nn_rs::train::TrainAppConfig;
use nn_rs::train::runner::{SelectedCube, SelectedDevice, SelectedRuntime};

const DEFAULT_CONFIG: &str = "nn-rs/configs/robust_session.yaml";

fn usage() -> &'static str {
    "usage: export_espdl --checkpoint <path.mpk|stem> --manifest <manifest.jsonl> \
     [--config nn-rs/configs/robust_session.yaml] [--out <model.espdl> | --out-dir <dir>] \
     [--target esp32s3] [--num-bits 8] [--calib-split train|val] [--calib-windows N] \
     [--calib-dir <dir-with-npy>]\n\
     If --out/--out-dir is omitted, output defaults to a sibling export/model.espdl \
     for checkpoints under a checkpoints/ directory."
}

#[derive(Debug)]
struct Cli {
    config: PathBuf,
    checkpoint: PathBuf,
    manifest: PathBuf,
    out: PathBuf,
    target: String,
    num_bits: u8,
    calib_split: Split,
    calib_windows: usize,
    calib_dir: Option<PathBuf>,
}

fn parse_args() -> Result<Cli, String> {
    let mut config = PathBuf::from(DEFAULT_CONFIG);
    let mut checkpoint = None;
    let mut manifest = None;
    let mut out = None;
    let mut out_dir = None;
    let mut target = "esp32s3".to_string();
    let mut num_bits = 8_u8;
    let mut calib_split = Split::Train;
    let mut calib_windows = 512_usize;
    let mut calib_dir = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = PathBuf::from(need(&mut args, "--config")?),
            "--checkpoint" => checkpoint = Some(PathBuf::from(need(&mut args, "--checkpoint")?)),
            "--manifest" => manifest = Some(PathBuf::from(need(&mut args, "--manifest")?)),
            "--out" => out = Some(PathBuf::from(need(&mut args, "--out")?)),
            "--out-dir" => out_dir = Some(PathBuf::from(need(&mut args, "--out-dir")?)),
            "--target" => target = need(&mut args, "--target")?,
            "--num-bits" => {
                num_bits = need(&mut args, "--num-bits")?
                    .parse()
                    .map_err(|_| "--num-bits must be 8".to_string())?
            }
            "--calib-split" => calib_split = parse_split(&need(&mut args, "--calib-split")?)?,
            "--calib-windows" => {
                calib_windows = need(&mut args, "--calib-windows")?
                    .parse()
                    .map_err(|_| "--calib-windows must be an integer".to_string())?
            }
            "--calib-dir" => calib_dir = Some(PathBuf::from(need(&mut args, "--calib-dir")?)),
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {a}\n{}", usage())),
        }
    }

    if target != "esp32s3" {
        return Err(format!(
            "unsupported --target {target:?}; native exporter currently supports esp32s3"
        ));
    }
    if num_bits != 8 {
        return Err(
            "--num-bits must be 8; INT16 export is not exposed until parity-verified".to_string(),
        );
    }
    if calib_windows == 0 {
        return Err("--calib-windows must be greater than zero".to_string());
    }
    if out.is_some() && out_dir.is_some() {
        return Err("use only one of --out or --out-dir".to_string());
    }
    let checkpoint = checkpoint.ok_or_else(|| format!("missing --checkpoint\n{}", usage()))?;
    let out = match (out, out_dir) {
        (Some(path), None) => path,
        (None, Some(dir)) => dir.join("model.espdl"),
        (None, None) => default_out_for_checkpoint(&checkpoint),
        (Some(_), Some(_)) => unreachable!("validated above"),
    };

    Ok(Cli {
        config,
        checkpoint,
        manifest: manifest.unwrap_or_default(),
        out,
        target,
        num_bits,
        calib_split,
        calib_windows,
        calib_dir,
    })
}

fn need<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} takes an argument"))
}

fn parse_split(s: &str) -> Result<Split, String> {
    match s {
        "train" => Ok(Split::Train),
        "val" | "valid" | "validation" => Ok(Split::Val),
        other => Err(format!(
            "unsupported --calib-split {other:?}; use train or val"
        )),
    }
}

fn strip_mpk(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|s| s == "mpk") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn default_out_for_checkpoint(checkpoint: &Path) -> PathBuf {
    let parent = checkpoint.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(parent) = parent
        && parent.file_name().is_some_and(|name| name == "checkpoints")
        && let Some(run_dir) = parent.parent()
    {
        return run_dir.join("export").join("model.espdl");
    }
    parent
        .map(|p| p.join("model.espdl"))
        .unwrap_or_else(|| PathBuf::from("model.espdl"))
}

fn run(cli: Cli) -> std::io::Result<()> {
    let mut cfg = TrainAppConfig::from_yaml_file(&cli.config).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("failed to load {}: {e}", cli.config.display()),
        )
    })?;
    if !cli.manifest.as_os_str().is_empty() {
        cfg.dataset.manifest_path = Some(cli.manifest.display().to_string());
    }
    let manifest = cfg.resolved_manifest().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "missing --manifest and config has no dataset.manifest_path",
        )
    })?;

    let device = SelectedDevice::default();
    let client = <SelectedRuntime as Runtime>::client(&device);
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

    let mel_cfg = to_mel_config(&cfg.features);
    let window_samples = (cfg.dataset.window_s * cfg.dataset.sample_rate as f32).round() as usize;
    let mel = MelExtractor::<SelectedRuntime>::new(
        client,
        device.clone(),
        mel_cfg,
        cfg.dataset.sample_rate,
    );
    let n_mels = cfg.features.n_mels;
    let n_frames = mel.num_frames(window_samples);
    let input_shape = [1, 1, n_mels, n_frames];

    println!("==[1/3] Load Burn checkpoint =========================================");
    println!("  config     : {}", cli.config.display());
    println!("  checkpoint : {}", cli.checkpoint.display());
    println!("  manifest   : {}", manifest.display());
    println!("  target     : {} int{}", cli.target, cli.num_bits);
    println!("  model      : TinyConv channels={:?}", cfg.model.channels);
    println!("  input      : {:?}", input_shape);

    let graph = tinyconv_to_burn_graph(&model, input_shape);

    println!("==[2/3] Collect calibration windows ==================================");
    let windows = if let Some(calib_dir) = &cli.calib_dir {
        collect_calibration_windows_from_npy(calib_dir, n_mels, n_frames, cli.calib_windows)?
    } else {
        collect_calibration_windows(&cfg, &mel, &device, cli.calib_split, cli.calib_windows)?
    };
    println!(
        "  calibration: {} window(s) from {}",
        windows.len(),
        cli.calib_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| format!("{:?} split", cli.calib_split))
    );

    println!("==[3/3] Calibrate and write ESP-DL ===================================");
    let artifacts = EspdlExporter::esp32s3_int8()
        .export_graph::<SelectedCube>(&graph, &windows, &device)
        .map_err(|e| std::io::Error::other(format!("export failed: {e}")))?;
    artifacts.write_to_model_path(&cli.out)?;
    let json_path = cli.out.with_extension("json");
    let info_path = cli.out.with_extension("info");

    println!();
    println!("========================================================================");
    println!("Native Burn -> ESP-DL export complete");
    println!("========================================================================");
    println!("  ESP-DL model : {}", cli.out.display());
    println!("  Quant config : {}", json_path.display());
    println!("  Graph info   : {}", info_path.display());
    println!("  Python/ONNX/Docker were not invoked.");
    println!("========================================================================");

    Ok(())
}

fn collect_calibration_windows(
    cfg: &TrainAppConfig,
    mel: &MelExtractor<SelectedRuntime>,
    device: &<SelectedCube as burn::tensor::backend::Backend>::Device,
    split: Split,
    limit: usize,
) -> std::io::Result<Vec<Vec<f32>>> {
    let manifest_path = cfg
        .resolved_manifest()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing manifest"))?;
    let items = load_manifest(&manifest_path)?;
    let dataset = WindowedAudioDataset::new(items, to_dataset_config(cfg), split)
        .map_err(|e| std::io::Error::other(format!("build calibration dataset: {e}")))?;
    if dataset.len() == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("manifest has no windows for {:?} split", split),
        ));
    }

    let batch_size = cfg.dm.batch_size.max(1).min(limit);
    let batcher = AudioBatcher::<SelectedCube>::new();
    let mut windows = Vec::with_capacity(limit.min(dataset.len()));
    let mut idx = 0_usize;
    while idx < dataset.len() && windows.len() < limit {
        let mut items = Vec::with_capacity(batch_size);
        while idx < dataset.len() && items.len() < batch_size && windows.len() + items.len() < limit
        {
            let item = dataset.get(idx).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to decode calibration window {idx}"),
                )
            })?;
            items.push(item);
            idx += 1;
        }
        if items.is_empty() {
            break;
        }
        let batch = batcher.batch(items, device);
        let tensor = mel.forward::<f32, i32, u8>(batch.waveforms);
        let dims = tensor.dims();
        let per_window = dims[1] * dims[2] * dims[3];
        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("mel tensor should be readable as f32");
        for chunk in values.chunks(per_window) {
            windows.push(chunk.to_vec());
        }
    }
    Ok(windows)
}

fn to_mel_config(cfg: &nn_rs::train::config::MelCfg) -> MelConfig {
    MelConfig {
        n_mels: cfg.n_mels,
        fmin: cfg.fmin,
        fmax: cfg.fmax,
        n_fft: cfg.n_fft,
        hop_length_ms: cfg.hop_length_ms,
        log: cfg.log,
        eps: cfg.eps,
        normalize: cfg.normalize,
        center: cfg.center,
    }
}

fn to_dataset_config(cfg: &TrainAppConfig) -> DatasetConfig {
    DatasetConfig {
        sample_rate: cfg.dataset.sample_rate,
        window_s: cfg.dataset.window_s,
        hop_s: cfg.dataset.hop_s,
        class_names: cfg.dataset.class_names.clone(),
        manifest_path: cfg.dataset.manifest_path.as_ref().map(PathBuf::from),
    }
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
