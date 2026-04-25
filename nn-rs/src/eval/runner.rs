//! End-to-end evaluation pipeline.
//!
//! Runs on the `CubeBackend` (no autodiff — eval doesn't need
//! gradients), reusing `MelExtractor<R>` directly. The CLI takes a
//! single composed YAML plus a handful of explicit flags.
//!
//! The evaluator loads a checkpoint saved by the training loop
//! (`CompactRecorder`-based), walks the requested split one batch at
//! a time, pulls `softmax(logits)[:, 1]` back to the host, and
//! accumulates per-file and (optionally) per-window predictions.
//! Aggregation to file-level is mean (default) or max.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use cubecl::client::ComputeClient;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::data::{
    AudioBatch, AudioBatcher, DatasetConfig, Split, WindowedAudioDataset, WindowedAudioItem,
    load_manifest,
};
use crate::mel::{MelConfig, MelExtractor};
use crate::model::{TinyConv, TinyConvConfig};
use crate::train::config::{MelCfg, ModelCfg, TrainAppConfig};

use super::audit::print_audit;
use super::metrics::{BinaryMetrics, binary_metrics, calibrate_threshold};

/// Window-probability aggregation mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AggregateMode {
    Mean,
    Max,
}

impl AggregateMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            _ => Err(format!("unknown aggregate mode: {s} (expected `mean` or `max`)")),
        }
    }

    fn apply(&self, probs: &[f32]) -> f32 {
        if probs.is_empty() {
            return 0.0;
        }
        match self {
            Self::Mean => probs.iter().sum::<f32>() / probs.len() as f32,
            Self::Max => probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

/// Everything the CLI passes to [`Evaluator::evaluate`].
///
/// Metrics land in `metrics.json` under the artifact dir, calibration
/// in `calibration.json`, and a copy goes next to the checkpoint as
/// `threshold.json` so it survives a checkpoint move.
#[derive(Clone, Debug)]
pub struct EvalOptions {
    pub checkpoint: PathBuf,
    pub split: Split,
    pub aggregate: AggregateMode,
    /// `Some(t)` uses that threshold as-is; `None` either calibrates
    /// on the split (when `calibrate=true`) or falls back to the
    /// checkpoint-adjacent `threshold.json` then 0.5.
    pub threshold: Option<f32>,
    pub calibrate: bool,
    pub window_metrics: bool,
    pub audit: bool,
    pub audit_top_n: usize,
    pub metrics_json: PathBuf,
    pub calibration_json: PathBuf,
    pub preds_csv: Option<PathBuf>,
    pub window_preds_csv: Option<PathBuf>,
}

/// Subset of the metrics bundle we surface back to the CLI / tests.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalOutcome {
    pub file: BinaryMetrics,
    pub window: Option<BinaryMetrics>,
    pub threshold: f32,
    pub num_files: usize,
    pub num_windows: usize,
    pub metrics_json_path: PathBuf,
}

/// Inner-backend evaluator. Holds the compute client, device, and mel
/// extractor; [`evaluate`] is the end-to-end entry point.
pub struct Evaluator<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    pub cfg: TrainAppConfig,
    pub client: ComputeClient<R>,
    pub device: <CubeBackend<R, F, I, BT> as burn::tensor::backend::Backend>::Device,
    pub mel: MelExtractor<R>,
}

type Backend<R, F, I, BT> = CubeBackend<R, F, I, BT>;

impl<R, F, I, BT> Evaluator<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    pub fn new(
        cfg: TrainAppConfig,
        client: ComputeClient<R>,
        device: <Backend<R, F, I, BT> as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        let mel_cfg = to_mel_config(&cfg.features);
        let mel = MelExtractor::<R>::new(
            client.clone(),
            device.clone(),
            mel_cfg,
            cfg.dataset.sample_rate,
        );
        Self {
            cfg,
            client,
            device,
            mel,
        }
    }

    /// Load the manifest, run the forward pass over `split`, compute
    /// metrics, calibrate / resolve the threshold, and write the
    /// `metrics.json` / `calibration.json` / CSV artifacts.
    pub fn evaluate(&self, opts: &EvalOptions) -> std::io::Result<EvalOutcome> {
        let manifest_path = self
            .cfg
            .resolved_manifest()
            .expect("dataset.manifest_path is required (set via YAML or --manifest)");
        let items = load_manifest(&manifest_path)?;

        let ds_cfg = to_dataset_config(&self.cfg);
        let dataset = WindowedAudioDataset::new(items, ds_cfg, opts.split).map_err(io_err)?;

        let batch_size = self.cfg.dm.batch_size;
        let num_workers = self.cfg.dm.resolved_num_workers().max(1);
        let loader: Arc<dyn DataLoader<Backend<R, F, I, BT>, AudioBatch<Backend<R, F, I, BT>>>> =
            build_loader::<Backend<R, F, I, BT>>(dataset, batch_size, num_workers);

        let model = self.load_checkpoint(&opts.checkpoint)?;

        // Window-level storage. Always populated; file-level aggregation
        // pulls from it. Our production split is a few thousand windows
        // so keeping them all in memory is cheap.
        let mut probs_by_file: HashMap<PathBuf, Vec<f32>> = HashMap::new();
        let mut label_by_file: HashMap<PathBuf, u8> = HashMap::new();
        let mut window_probs: Vec<f32> = Vec::new();
        let mut window_labels: Vec<u8> = Vec::new();
        let mut window_files: Vec<PathBuf> = Vec::new();
        let mut window_starts: Vec<f32> = Vec::new();
        let mut window_ends: Vec<f32> = Vec::new();

        for batch in loader.iter() {
            let n = batch.files.len();
            let logits = model.forward(self.mel.forward(batch.waveforms));
            // softmax over dim=1 (n_classes); take column 1 (pump_on).
            let probs_tensor = softmax(logits, 1).slice([0..n, 1..2]).reshape([n]);
            let probs_vec: Vec<f32> = probs_tensor
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let labels_vec: Vec<i64> = batch
                .labels
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .unwrap();

            for i in 0..n {
                let f = batch.files[i].clone();
                let p = probs_vec[i];
                let y = labels_vec[i] as u8;
                probs_by_file.entry(f.clone()).or_default().push(p);
                label_by_file.insert(f.clone(), y);
                window_probs.push(p);
                window_labels.push(y);
                window_files.push(f);
                window_starts.push(batch.starts[i]);
                window_ends.push(batch.ends[i]);
            }
        }

        // Deterministic order for CSV / JSON consumers.
        let mut files_sorted: Vec<PathBuf> = probs_by_file.keys().cloned().collect();
        files_sorted.sort();
        let y_prob: Vec<f32> = files_sorted
            .iter()
            .map(|f| opts.aggregate.apply(&probs_by_file[f]))
            .collect();
        let y_true: Vec<u8> = files_sorted.iter().map(|f| label_by_file[f]).collect();

        let ckpt_dir = opts.checkpoint.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let threshold = resolve_threshold(
            opts,
            &y_true,
            &y_prob,
            &ckpt_dir,
            files_sorted.len(),
        )?;

        let metrics_file = binary_metrics(&y_true, &y_prob, threshold);
        let metrics_window = if opts.window_metrics && !window_probs.is_empty() {
            Some(binary_metrics(&window_labels, &window_probs, threshold))
        } else {
            None
        };

        // Write metrics.json
        if let Some(parent) = opts.metrics_json.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let payload = match &metrics_window {
            Some(m) => json!({ "file": metrics_file, "window": m }),
            None => json!({ "file": metrics_file }),
        };
        std::fs::write(
            &opts.metrics_json,
            serde_json::to_string_pretty(&payload).map_err(io_err)?,
        )?;

        // Optional predictions CSVs.
        if let Some(path) = &opts.preds_csv {
            write_file_preds_csv(path, &files_sorted, &y_prob, &y_true, threshold)?;
        }
        if let Some(path) = &opts.window_preds_csv {
            write_window_preds_csv(
                path,
                &window_files,
                &window_starts,
                &window_ends,
                &window_probs,
                &window_labels,
                threshold,
            )?;
        }

        // Report.
        println!(
            "File metrics: {}",
            serde_json::to_string_pretty(&metrics_file).map_err(io_err)?
        );
        if let Some(m) = &metrics_window {
            println!(
                "Window metrics: {}",
                serde_json::to_string_pretty(m).map_err(io_err)?
            );
        }
        println!("Wrote metrics to {}", opts.metrics_json.display());

        if opts.audit {
            let file_refs: Vec<&Path> = window_files.iter().map(PathBuf::as_path).collect();
            print_audit(
                &file_refs,
                &window_probs,
                &window_labels,
                threshold,
                opts.audit_top_n,
            );
        }

        // Final summary block.
        println!();
        println!("{}", "=".repeat(72));
        println!("Evaluation complete (split={:?})", opts.split);
        println!("{}", "=".repeat(72));
        println!("  Checkpoint       : {}", opts.checkpoint.display());
        println!("  Metrics JSON     : {}", opts.metrics_json.display());
        let thr_path = ckpt_dir.join("threshold.json");
        if thr_path.exists() {
            println!("  Threshold JSON   : {}", thr_path.display());
        }
        let cal_path = ckpt_dir.join("calibration.json");
        if cal_path.exists() {
            println!("  Calibration JSON : {}", cal_path.display());
        }
        if let Some(p) = &opts.preds_csv {
            println!("  File preds CSV   : {}", p.display());
        }
        if let Some(p) = &opts.window_preds_csv {
            println!("  Window preds CSV : {}", p.display());
        }
        println!("{}", "=".repeat(72));

        Ok(EvalOutcome {
            file: metrics_file,
            window: metrics_window,
            threshold,
            num_files: files_sorted.len(),
            num_windows: window_probs.len(),
            metrics_json_path: opts.metrics_json.clone(),
        })
    }

    fn load_checkpoint(&self, path: &Path) -> std::io::Result<TinyConv<Backend<R, F, I, BT>>> {
        let recorder = CompactRecorder::new();
        let record = recorder
            .load(path.to_path_buf(), &self.device)
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("load checkpoint {}: {e}", path.display()),
                )
            })?;
        let model_cfg = to_tinyconv_config(&self.cfg.model);
        Ok(model_cfg
            .init::<Backend<R, F, I, BT>>(&self.device)
            .load_record(record))
    }
}

/// Resolve the threshold to use:
///
/// 1. `opts.threshold` if explicitly set.
/// 2. If `calibrate=true` — run the coarse-to-fine macro-F1 search and
///    persist `calibration.json` / `threshold.json`.
/// 3. Otherwise, try to load `threshold.json` next to the checkpoint.
/// 4. Fallback: `0.5`.
fn resolve_threshold(
    opts: &EvalOptions,
    y_true: &[u8],
    y_prob: &[f32],
    ckpt_dir: &Path,
    num_files: usize,
) -> std::io::Result<f32> {
    if let Some(t) = opts.threshold {
        return Ok(t);
    }
    if opts.calibrate {
        let best_t = calibrate_threshold(y_true, y_prob);
        let calibration = json!({
            "best_threshold": best_t,
            "split": format!("{:?}", opts.split).to_lowercase(),
            "aggregate": match opts.aggregate {
                AggregateMode::Mean => "mean",
                AggregateMode::Max => "max",
            },
            "num_files": num_files,
        });
        if let Some(parent) = opts.calibration_json.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(
            &opts.calibration_json,
            serde_json::to_string_pretty(&calibration).map_err(io_err)?,
        )?;
        println!(
            "Calibrated best_threshold={best_t:.3} -> wrote {}",
            opts.calibration_json.display()
        );
        // Persist next to the checkpoint too, so infer / export can pick it up.
        std::fs::create_dir_all(ckpt_dir)?;
        let thr_path = ckpt_dir.join("threshold.json");
        std::fs::write(
            &thr_path,
            serde_json::to_string_pretty(&calibration).map_err(io_err)?,
        )?;
        println!("Wrote calibrated threshold next to checkpoint: {}", thr_path.display());
        return Ok(best_t);
    }
    let thr_path = ckpt_dir.join("threshold.json");
    if thr_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&thr_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(t) = v.get("best_threshold").and_then(|x| x.as_f64()) {
                    return Ok(t as f32);
                }
                if let Some(t) = v.get("threshold").and_then(|x| x.as_f64()) {
                    return Ok(t as f32);
                }
            }
        }
    }
    Ok(0.5)
}

fn write_file_preds_csv(
    path: &Path,
    files: &[PathBuf],
    probs: &[f32],
    labels: &[u8],
    threshold: f32,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut body = String::from("file,prob_on,label,pred\n");
    for i in 0..files.len() {
        let pred = if probs[i] >= threshold { 1 } else { 0 };
        body.push_str(&format!(
            "{},{},{},{}\n",
            csv_escape(&files[i].to_string_lossy()),
            probs[i],
            labels[i],
            pred,
        ));
    }
    std::fs::write(path, body)?;
    println!("Wrote predictions CSV to {}", path.display());
    Ok(())
}

fn write_window_preds_csv(
    path: &Path,
    files: &[PathBuf],
    starts: &[f32],
    ends: &[f32],
    probs: &[f32],
    labels: &[u8],
    threshold: f32,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut body = String::from("file,start_s,end_s,prob_on,label,pred\n");
    for i in 0..files.len() {
        let pred = if probs[i] >= threshold { 1 } else { 0 };
        body.push_str(&format!(
            "{},{},{},{},{},{}\n",
            csv_escape(&files[i].to_string_lossy()),
            starts[i],
            ends[i],
            probs[i],
            labels[i],
            pred,
        ));
    }
    std::fs::write(path, body)?;
    println!("Wrote window predictions CSV to {}", path.display());
    Ok(())
}

/// Minimal RFC 4180 escape for the file column. The other columns are
/// numeric / single-digit and never need quoting.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

fn build_loader<B: burn::tensor::backend::Backend>(
    dataset: WindowedAudioDataset,
    batch_size: usize,
    num_workers: usize,
) -> Arc<dyn DataLoader<B, AudioBatch<B>>> {
    let batcher = AudioBatcher::<B>::new();
    DataLoaderBuilder::<B, WindowedAudioItem, AudioBatch<B>>::new(batcher)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(dataset)
}

fn to_mel_config(cfg: &MelCfg) -> MelConfig {
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

fn to_tinyconv_config(cfg: &ModelCfg) -> TinyConvConfig {
    TinyConvConfig {
        channels: cfg.channels.clone(),
        dropout: cfg.dropout,
        n_classes: 2,
    }
}

fn io_err<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_mean_and_max() {
        assert_eq!(AggregateMode::Mean.apply(&[0.2, 0.4, 0.9]), (0.2 + 0.4 + 0.9) / 3.0);
        assert_eq!(AggregateMode::Max.apply(&[0.2, 0.4, 0.9]), 0.9);
        assert_eq!(AggregateMode::Mean.apply(&[]), 0.0);
    }

    #[test]
    fn aggregate_mode_parse() {
        assert_eq!(AggregateMode::parse("mean").unwrap(), AggregateMode::Mean);
        assert_eq!(AggregateMode::parse("max").unwrap(), AggregateMode::Max);
        assert!(AggregateMode::parse("median").is_err());
    }
}
