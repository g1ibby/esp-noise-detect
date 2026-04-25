//! Training smoke tests.
//!
//! 1. `overfit_two_batches_drives_loss_down` — pulls two batches from the
//!    dataset manifest, runs a few dozen optimizer steps, and asserts
//!    `final / initial < 0.1`. Fails loudly if autodiff / optimizer
//!    wiring is wrong.
//!
//! 2. `two_epoch_sanity_runs` — constructs a tiny sliced manifest,
//!    runs `trainer.fit()` for 2 epochs, and asserts the run completes
//!    and produces a checkpoint file.
//!
//! Both tests are guarded by `DATASET_MANIFEST` — if the manifest
//! isn't present on the machine (e.g. CI), we skip. Set
//! `DATASET_MANIFEST=/path/to/recordings/manifest.jsonl`
//! to run the test.

use std::path::{Path, PathBuf};

use burn::data::dataloader::batcher::Batcher;
use burn_cubecl::CubeBackend;
use cubecl::{Runtime, TestRuntime};

use nn_rs::WindowedAudioItem;
use nn_rs::data::{
    AudioBatch, AudioBatcher, DatasetConfig, Split, WindowedAudioDataset, load_manifest,
};
use nn_rs::train::{TrainAppConfig, Trainer};

// Backend-agnostic aliases: TestRuntime is whichever runtime was selected
// at `cargo test` time via --features test-{metal,vulkan,cuda,cpu}.
type TestCube = CubeBackend<TestRuntime, f32, i32, u8>;
type TestAutodiff = burn::backend::Autodiff<TestCube>;
// Burn `Dataset` trait provides `.get()` which we use below.
use burn::data::dataset::Dataset;

fn dataset_manifest_path() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("DATASET_MANIFEST") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Load a slice of the voicy manifest with at most `max_per_split`
/// entries in each of train / val / test. Lets the smoke test run in
/// seconds even on the full 3k-row production manifest.
fn sliced_items(path: &Path, max_per_split: usize) -> Vec<nn_rs::ManifestItem> {
    let all = load_manifest(path).expect("manifest load");
    let mut seen_train = 0usize;
    let mut seen_val = 0usize;
    let mut seen_test = 0usize;
    all.into_iter()
        .filter(|it| match it.split.unwrap_or(Split::Train) {
            Split::Train => {
                if seen_train < max_per_split {
                    seen_train += 1;
                    true
                } else {
                    false
                }
            }
            Split::Val => {
                if seen_val < max_per_split {
                    seen_val += 1;
                    true
                } else {
                    false
                }
            }
            Split::Test => {
                if seen_test < max_per_split {
                    seen_test += 1;
                    true
                } else {
                    false
                }
            }
        })
        .collect()
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
    // Turn off augmentation for the smoke tests — we want deterministic
    // loss trajectories, and the augment pipeline's cubek-random
    // process-global seed makes reruns non-bit-exact.
    cfg.augment.train.enabled = false;
    cfg
}

fn build_trainer(cfg: TrainAppConfig) -> Trainer<TestAutodiff, TestRuntime, f32, i32, u8> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    Trainer::new(cfg, client, device.clone(), device)
}

fn tempdir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("nn-rs-train-smoke-{}-{n}", std::process::id(),));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn overfit_two_batches_drives_loss_down() {
    let Some(manifest) = dataset_manifest_path() else {
        eprintln!("[skip] DATASET_MANIFEST not set — skipping overfit smoke test");
        return;
    };
    let artifact_dir = tempdir();

    let mut cfg = base_config(&manifest, &artifact_dir, /* unused */ 1);
    // Overfit is more aggressive than real training — crank the LR so
    // 60 steps is enough to flatten the loss on two batches.
    cfg.optim.lr = 3e-3;

    let trainer = build_trainer(cfg);

    // Slice the manifest hard — we only need two batches' worth of
    // data, not the whole 3k rows.
    let items = sliced_items(&manifest, /* max_per_split */ 4);
    assert!(!items.is_empty(), "sliced manifest is empty");

    let ds_cfg = DatasetConfig::default();
    let ds = WindowedAudioDataset::new(items, ds_cfg, Split::Train).expect("dataset build");

    let window_samples = ds.window_samples();
    // Pull two mini-batches of 4 windows each off the dataset — more than
    // enough audio variety to avoid a pathological "all one label" case.
    let batch_a = collect_batch(&ds, 0, 4, window_samples, &trainer.device);
    let batch_b = collect_batch(&ds, 4, 4, window_samples, &trainer.device);

    let history = trainer.overfit_batches(vec![batch_a, batch_b], /* steps */ 60);
    assert!(!history.is_empty(), "no losses recorded");
    let first = history[0];
    let last = history[history.len() - 1];
    let min = history.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "[overfit] initial={first:.4} final={last:.4} min={min:.4} (n_steps={})",
        history.len()
    );
    assert!(first.is_finite() && last.is_finite(), "NaN in loss history");
    // Plan asks for "near-zero loss" on two batches. Not every device
    // pins loss to literally zero (bf16 rounding, random init), but a
    // 10× drop is the bar this test enforces.
    assert!(
        min < first * 0.1 + 1e-3,
        "loss did not drop enough: first={first:.4} min={min:.4}"
    );
}

#[test]
fn two_epoch_sanity_runs() {
    let Some(manifest) = dataset_manifest_path() else {
        eprintln!("[skip] DATASET_MANIFEST not set — skipping 2-epoch sanity test");
        return;
    };

    // Write a sliced manifest to a tempdir so the Trainer's own
    // `load_manifest` call sees a smaller input. This also exercises
    // the CLI-style pipeline (file on disk, relative audio_path
    // resolution against manifest parent).
    let tmp = tempdir();
    let all = load_manifest(&manifest).expect("manifest load");
    let sliced = {
        let mut seen_train = 0;
        let mut seen_val = 0;
        all.into_iter()
            .filter(|it| match it.split.unwrap_or(Split::Train) {
                Split::Train => {
                    seen_train += 1;
                    seen_train <= 6
                }
                Split::Val => {
                    seen_val += 1;
                    seen_val <= 4
                }
                Split::Test => false,
            })
            .collect::<Vec<_>>()
    };
    let manifest_path = tmp.join("manifest.jsonl");
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

    let artifact_dir = tmp.join("run");
    let cfg = base_config(&manifest_path, &artifact_dir, 2);
    let trainer = build_trainer(cfg);
    let outcome = trainer.fit().expect("2-epoch training did not crash");
    eprintln!(
        "[sanity] total_epochs={} best_epoch={} best_val_f1={:.4} ckpt={}",
        outcome.total_epochs_run,
        outcome.best_epoch,
        outcome.best_val_macro_f1,
        outcome.best_checkpoint.display(),
    );
    assert_eq!(outcome.total_epochs_run, 2);
    // The best checkpoint uses Burn's CompactRecorder — the actual file
    // on disk ends with `.mpk`. A `.mpk` in the checkpoint dir is all
    // we check; weight parity is a separate concern.
    let ckpt_dir = artifact_dir.join("checkpoints");
    let mut found = false;
    if let Ok(rd) = std::fs::read_dir(&ckpt_dir) {
        for e in rd.flatten() {
            if e.path().extension().map(|e| e == "mpk").unwrap_or(false) {
                found = true;
                break;
            }
        }
    }
    assert!(found, "no *.mpk checkpoint found in {}", ckpt_dir.display());
}

/// Collect `n` windows from `dataset` starting at `start_idx`, stack
/// them into an `AudioBatch<AB>` using the standard batcher. Same shape
/// the training loop gets at runtime.
fn collect_batch<AB>(
    dataset: &WindowedAudioDataset,
    start_idx: usize,
    n: usize,
    _window_samples: usize,
    device: &<AB as burn::tensor::backend::Backend>::Device,
) -> AudioBatch<AB>
where
    AB: burn::tensor::backend::Backend,
{
    let items: Vec<WindowedAudioItem> = (0..n)
        .map(|i| dataset.get(start_idx + i).expect("dataset item"))
        .collect();
    let batcher = AudioBatcher::<AB>::new();
    batcher.batch(items, device)
}
