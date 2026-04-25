//! End-to-end eval smoke test.
//!
//! Trains a micro run (2 epochs, 4–6 files per split, batch size 8)
//! so we have a real `CompactRecorder` checkpoint to evaluate against,
//! then runs [`Evaluator::evaluate`] on the same manifest and asserts:
//!
//! * `metrics.json` is produced with the expected schema.
//! * `calibration.json` is written when `calibrate=true`.
//! * `threshold.json` lands next to the checkpoint.
//! * Aggregation covers at least one file (no empty split).
//!
//! Skipped when the dataset manifest isn't available (CI without the
//! mounted dataset). Set `DATASET_MANIFEST=...` to run.

use std::path::{Path, PathBuf};

use burn_cubecl::CubeBackend;
use cubecl::{Runtime, TestRuntime};

use nn_rs::data::{Split, load_manifest};
use nn_rs::eval::{AggregateMode, EvalOptions, Evaluator};
use nn_rs::train::{TrainAppConfig, Trainer};

// Backend-agnostic aliases: TestRuntime is whichever runtime was selected
// at `cargo test` time via --features test-{metal,vulkan,cuda,cpu}.
type TestCube = CubeBackend<TestRuntime, f32, i32, u8>;
type TestAutodiff = burn::backend::Autodiff<TestCube>;

fn dataset_manifest_path() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("DATASET_MANIFEST") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn tempdir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "nn-rs-eval-smoke-{}-{n}",
        std::process::id(),
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Write a sliced manifest with at most `max_per_split` rows per split.
fn write_sliced_manifest(src: &Path, dst_dir: &Path, max_per_split: usize) -> PathBuf {
    let all = load_manifest(src).expect("manifest load");
    let mut seen_train = 0usize;
    let mut seen_val = 0usize;
    let mut seen_test = 0usize;
    let sliced: Vec<_> = all
        .into_iter()
        .filter(|it| match it.split.unwrap_or(Split::Train) {
            Split::Train => {
                seen_train += 1;
                seen_train <= max_per_split
            }
            Split::Val => {
                seen_val += 1;
                seen_val <= max_per_split
            }
            Split::Test => {
                seen_test += 1;
                seen_test <= max_per_split
            }
        })
        .collect();
    let manifest_path = dst_dir.join("manifest.jsonl");
    let mut out = String::new();
    for it in sliced {
        let split = match it.split {
            Some(Split::Train) => "train",
            Some(Split::Val) => "val",
            Some(Split::Test) => "test",
            None => "train",
        };
        out.push_str(&format!(
            "{{\"audio_path\": \"{}\", \"label\": \"{}\", \"split\": \"{}\"}}\n",
            it.audio_path.display(),
            it.label,
            split,
        ));
    }
    std::fs::write(&manifest_path, out).unwrap();
    manifest_path
}

fn base_config(manifest: &Path, artifact_dir: &Path, max_epochs: usize) -> TrainAppConfig {
    let mut cfg = TrainAppConfig::default();
    cfg.dataset.manifest_path = Some(manifest.to_string_lossy().into_owned());
    cfg.dm.batch_size = 8;
    cfg.dm.num_workers = Some(0);
    cfg.trainer.max_epochs = max_epochs;
    cfg.trainer.early_stopping_patience = max_epochs + 1;
    cfg.trainer.artifact_dir = artifact_dir.to_string_lossy().into_owned();
    cfg.optim.lr = 1e-3;
    cfg.augment.train.enabled = false;
    cfg
}

#[test]
fn evaluate_produces_metrics_and_calibration_json() {
    let Some(source_manifest) = dataset_manifest_path() else {
        eprintln!(
            "[skip] DATASET_MANIFEST not set — skipping eval smoke test"
        );
        return;
    };

    let tmp = tempdir();
    let manifest = write_sliced_manifest(&source_manifest, &tmp, /* max_per_split */ 4);
    let artifact_dir = tmp.join("run");
    let cfg = base_config(&manifest, &artifact_dir, /* max_epochs */ 2);

    // --- train to produce a checkpoint --------------------------------
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let trainer: Trainer<TestAutodiff, TestRuntime, f32, i32, u8> =
        Trainer::new(cfg.clone(), client.clone(), device.clone(), device.clone());
    let outcome = trainer.fit().expect("2-epoch training");
    eprintln!(
        "[eval-smoke] trained — best_epoch={} val_f1={:.4} ckpt={}",
        outcome.best_epoch,
        outcome.best_val_macro_f1,
        outcome.best_checkpoint.display()
    );
    assert!(outcome.best_checkpoint.parent().unwrap().exists());

    // --- eval against the saved checkpoint ----------------------------
    let evaluator: Evaluator<TestRuntime, f32, i32, u8> =
        Evaluator::new(cfg, client, device);
    let ckpt_dir = outcome.best_checkpoint.parent().unwrap().to_path_buf();
    let metrics_json = ckpt_dir.join("metrics.json");
    let calibration_json = ckpt_dir.join("calibration.json");
    let preds_csv = ckpt_dir.join("preds.csv");
    let window_preds_csv = ckpt_dir.join("preds_window.csv");

    let opts = EvalOptions {
        checkpoint: outcome.best_checkpoint.clone(),
        split: Split::Val,
        aggregate: AggregateMode::Mean,
        threshold: None,
        calibrate: true,
        window_metrics: true,
        audit: true,
        audit_top_n: 5,
        metrics_json: metrics_json.clone(),
        calibration_json: calibration_json.clone(),
        preds_csv: Some(preds_csv.clone()),
        window_preds_csv: Some(window_preds_csv.clone()),
    };

    let outcome = evaluator.evaluate(&opts).expect("evaluate");
    eprintln!(
        "[eval-smoke] files={} windows={} macro_f1={:.4} acc={:.4} thr={:.3}",
        outcome.num_files,
        outcome.num_windows,
        outcome.file.macro_f1,
        outcome.file.accuracy,
        outcome.threshold,
    );

    // --- artifact checks ----------------------------------------------
    assert!(metrics_json.exists(), "metrics.json not written");
    assert!(calibration_json.exists(), "calibration.json not written");
    assert!(
        ckpt_dir.join("threshold.json").exists(),
        "threshold.json not written next to checkpoint"
    );
    assert!(preds_csv.exists(), "file preds CSV missing");
    assert!(window_preds_csv.exists(), "window preds CSV missing");

    // --- schema check --------------------------------------------------
    let text = std::fs::read_to_string(&metrics_json).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let file = v.get("file").expect("file metrics block");
    for k in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_pos",
        "f1_neg",
        "auroc",
        "auprc",
        "tp",
        "tn",
        "fp",
        "fn",
        "threshold",
    ] {
        assert!(file.get(k).is_some(), "metrics.json missing key `{k}`");
    }
    let window = v.get("window").expect("window metrics block (window_metrics=true)");
    assert!(window.get("macro_f1").is_some());

    // At least one file should have been evaluated (val slice is 4 rows).
    assert!(outcome.num_files >= 1, "no files aggregated");
    assert!(outcome.num_windows >= 1, "no windows evaluated");
    assert!(
        outcome.threshold.is_finite() && (0.0..=1.0).contains(&outcome.threshold),
        "calibrated threshold out of range: {}",
        outcome.threshold
    );
}
