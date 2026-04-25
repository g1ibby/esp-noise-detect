//! Integration test for the data pipeline.
//!
//! Writes synthetic 24-bit mono WAV files to a tempdir, builds a
//! manifest pointing at them, and exercises the full
//! manifest → dataset → batcher chain: indexing, label decoding,
//! and window count for known file lengths.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;

use nn_rs::data::{
    AudioBatch, AudioBatcher, DatasetConfig, ManifestItem, Split,
    WindowedAudioDataset, WindowedAudioItem, load_manifest,
};

mod common;
use common::{Backend, device};

/// Write a 24-bit mono WAV at `sample_rate` whose samples are a
/// linear ramp from `start_val` to `start_val + num_samples` in
/// 24-bit signed integer space. Returns the full path. The ramp lets
/// us verify we decoded the correct byte range per window (samples
/// round-trip exactly at ≤24-bit scale).
fn write_ramp_wav(
    dir: &Path,
    name: &str,
    sample_rate: u32,
    num_samples: usize,
    start_val: i32,
) -> PathBuf {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let path = dir.join(name);
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for i in 0..num_samples {
        writer.write_sample(start_val + i as i32).unwrap();
    }
    writer.finalize().unwrap();
    path
}

fn write_manifest(dir: &Path, name: &str, lines: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let mut f = File::create(&path).unwrap();
    for l in lines {
        writeln!(f, "{l}").unwrap();
    }
    path
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("nn-rs-data-pipeline-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Default config used across tests: 1 s window / 0.5 s hop @ 32 kHz,
/// exactly matching `configs/experiment/robust_session.yaml`.
fn cfg() -> DatasetConfig {
    DatasetConfig::default()
}

#[test]
fn manifest_and_dataset_index_one_aligned_file() {
    let tmp = TempDir::new();
    // 2 seconds of audio at 32 kHz = 64000 samples, window=32000, hop=16000.
    // Expected windows: 1 + (64000-32000)/16000 = 1 + 2 = 3.
    write_ramp_wav(tmp.path(), "f.wav", 32_000, 64_000, 0);
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[r#"{"audio_path": "f.wav", "label": "pump_off", "split": "train"}"#],
    );

    let items = load_manifest(&manifest).unwrap();
    assert_eq!(items.len(), 1);
    let ds = WindowedAudioDataset::new(items, cfg(), Split::Train).unwrap();
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.window_samples(), 32_000);
    assert_eq!(ds.hop_samples(), 16_000);
}

#[test]
fn dataset_pads_tail_window() {
    let tmp = TempDir::new();
    // 40000 samples = 1.25 s. window=32000, hop=16000.
    // 1 + (40000-32000)/16000 = 1 (no integer division), remainder 8000 > 0,
    // so we get a second padded window: total 2.
    write_ramp_wav(tmp.path(), "f.wav", 32_000, 40_000, 0);
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[r#"{"audio_path": "f.wav", "label": "pump_on"}"#],
    );

    let items = load_manifest(&manifest).unwrap();
    let ds = WindowedAudioDataset::new(items, cfg(), Split::Train).unwrap();
    assert_eq!(ds.len(), 2);

    let first: WindowedAudioItem = ds.get(0).unwrap();
    let second: WindowedAudioItem = ds.get(1).unwrap();
    assert_eq!(first.waveform.len(), 32_000);
    assert_eq!(second.waveform.len(), 32_000);

    // Second window starts at 16000, file ends at 40000, so only
    // samples [16000, 40000) are non-zero and the tail [24000, 32000)
    // is zero-padded.
    let non_zero_count = second.waveform.iter().filter(|v| **v != 0.0).count();
    assert_eq!(non_zero_count, 24_000);
    for v in &second.waveform[24_000..] {
        assert_eq!(*v, 0.0);
    }
    // end_s caps at file length (40000/32000 = 1.25s), not the window
    // end (48000/32000 = 1.5s).
    assert!((second.end_s - 1.25).abs() < 1e-6);
    assert!((first.start_s - 0.0).abs() < 1e-6);
}

#[test]
fn sub_window_file_produces_one_padded_window() {
    let tmp = TempDir::new();
    // 10000 samples = 0.3125 s < window_samples (32000). Still get
    // exactly one padded window, matching Python behaviour.
    write_ramp_wav(tmp.path(), "short.wav", 32_000, 10_000, 0);
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[r#"{"audio_path": "short.wav", "label": "pump_off"}"#],
    );

    let ds =
        WindowedAudioDataset::new(load_manifest(&manifest).unwrap(), cfg(), Split::Train).unwrap();
    assert_eq!(ds.len(), 1);
    let item = ds.get(0).unwrap();
    assert_eq!(item.waveform.len(), 32_000);
    // Samples [10000, 32000) are zero-padded.
    for v in &item.waveform[10_000..] {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn split_filter_selects_only_matching_rows() {
    let tmp = TempDir::new();
    write_ramp_wav(tmp.path(), "t.wav", 32_000, 32_000, 0);
    write_ramp_wav(tmp.path(), "v.wav", 32_000, 32_000, 0);
    write_ramp_wav(tmp.path(), "nosplit.wav", 32_000, 32_000, 0);

    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[
            r#"{"audio_path": "t.wav", "label": "pump_off", "split": "train"}"#,
            r#"{"audio_path": "v.wav", "label": "pump_on", "split": "val"}"#,
            // No split → default to train, matching Python's
            // `(item.split or "train")` shorthand.
            r#"{"audio_path": "nosplit.wav", "label": "pump_on"}"#,
        ],
    );
    let items = load_manifest(&manifest).unwrap();

    let train_ds =
        WindowedAudioDataset::new(items.clone(), cfg(), Split::Train).unwrap();
    let val_ds = WindowedAudioDataset::new(items.clone(), cfg(), Split::Val).unwrap();
    let test_ds = WindowedAudioDataset::new(items, cfg(), Split::Test).unwrap();

    assert_eq!(train_ds.len(), 2); // t.wav + nosplit.wav
    assert_eq!(val_ds.len(), 1); // v.wav
    assert_eq!(test_ds.len(), 0);
}

#[test]
fn labels_decode_per_class_names_order() {
    let tmp = TempDir::new();
    write_ramp_wav(tmp.path(), "off.wav", 32_000, 32_000, 0);
    write_ramp_wav(tmp.path(), "on.wav", 32_000, 32_000, 0);

    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[
            r#"{"audio_path": "off.wav", "label": "pump_off"}"#,
            r#"{"audio_path": "on.wav", "label": "pump_on"}"#,
        ],
    );
    let ds = WindowedAudioDataset::new(
        load_manifest(&manifest).unwrap(),
        cfg(),
        Split::Train,
    )
    .unwrap();

    // class_names default order is ["pump_off", "pump_on"], so
    // off → 0, on → 1.
    let items: Vec<_> = (0..ds.len()).map(|i| ds.get(i).unwrap()).collect();
    let by_name: std::collections::HashMap<_, _> = items
        .iter()
        .map(|it| (it.file.file_name().unwrap().to_owned(), it.label))
        .collect();
    assert_eq!(by_name[std::ffi::OsStr::new("off.wav")], 0);
    assert_eq!(by_name[std::ffi::OsStr::new("on.wav")], 1);
}

#[test]
fn unknown_label_rejected_at_construction() {
    let tmp = TempDir::new();
    write_ramp_wav(tmp.path(), "x.wav", 32_000, 32_000, 0);
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[r#"{"audio_path": "x.wav", "label": "pump_wat"}"#],
    );
    let items = load_manifest(&manifest).unwrap();
    match WindowedAudioDataset::new(items, cfg(), Split::Train) {
        Ok(_) => panic!("expected UnknownLabel error"),
        Err(e) => assert!(
            format!("{e}").contains("pump_wat"),
            "unexpected error: {e}"
        ),
    }
}

#[test]
fn mismatched_sample_rate_rejected_lazily() {
    let tmp = TempDir::new();
    // Write a 16 kHz file with a 32 kHz target config.
    write_ramp_wav(tmp.path(), "lowsr.wav", 16_000, 16_000, 0);
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[r#"{"audio_path": "lowsr.wav", "label": "pump_off"}"#],
    );
    let ds = WindowedAudioDataset::new(
        load_manifest(&manifest).unwrap(),
        cfg(),
        Split::Train,
    )
    .unwrap();
    assert_eq!(ds.len(), 1); // lazy-failure fallback
    assert!(
        ds.get(0).is_none(),
        "expected None from get() on SR-mismatched file",
    );
}

#[test]
fn batcher_stacks_windows_and_labels() {
    let tmp = TempDir::new();
    write_ramp_wav(tmp.path(), "a.wav", 32_000, 32_000, 0);
    write_ramp_wav(tmp.path(), "b.wav", 32_000, 32_000, 1000);
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[
            r#"{"audio_path": "a.wav", "label": "pump_off"}"#,
            r#"{"audio_path": "b.wav", "label": "pump_on"}"#,
        ],
    );
    let ds = WindowedAudioDataset::new(
        load_manifest(&manifest).unwrap(),
        cfg(),
        Split::Train,
    )
    .unwrap();

    let items: Vec<WindowedAudioItem> = (0..ds.len()).map(|i| ds.get(i).unwrap()).collect();
    assert_eq!(items.len(), 2);

    let batcher = AudioBatcher::<Backend>::new();
    let batch: AudioBatch<Backend> = batcher.batch(items, &device());

    assert_eq!(batch.waveforms.dims(), [2, 32_000]);
    assert_eq!(batch.labels.dims(), [2]);
    assert_eq!(batch.files.len(), 2);
    assert_eq!(batch.starts, vec![0.0, 0.0]);
    assert_eq!(batch.ends, vec![1.0, 1.0]);

    // Verify labels survived the upload intact. The test backend uses
    // i32 as its IntElem (see tests/common/mod.rs).
    let labels: Vec<i32> = batch
        .labels
        .to_data()
        .to_vec::<i32>()
        .expect("label tensor should decode to i32");
    // Order is manifest order, Python-parity: a→pump_off→0, b→pump_on→1.
    assert_eq!(labels, vec![0, 1]);

    // And the ramp values: first sample of a.wav is 0, first sample of b.wav
    // is 1000 (in 24-bit integer space, divided by 2^23).
    let flat: Vec<f32> = batch
        .waveforms
        .to_data()
        .to_vec::<f32>()
        .expect("waveform tensor should decode to f32");
    let scale = 1.0_f32 / (1u64 << 23) as f32;
    assert!((flat[0] - 0.0).abs() < 1e-6);
    assert!((flat[32_000] - 1000.0 * scale).abs() < 1e-6);
}

#[test]
fn production_shaped_manifest_round_trips() {
    // Fixture mirroring the production manifest shape (mixed splits,
    // multi-second files, the `pump_off` / `pump_on` labels). Checks
    // that a realistic multi-row manifest produces the expected total
    // index size across splits and that a first-window decode lands
    // at the right shape.
    let tmp = TempDir::new();
    // Five files, mixed splits / labels / durations.
    write_ramp_wav(tmp.path(), "a.wav", 32_000, 2 * 32_000, 0); // 2 s → 3 windows
    write_ramp_wav(tmp.path(), "b.wav", 32_000, 3 * 32_000, 100); // 3 s → 5 windows
    write_ramp_wav(tmp.path(), "c.wav", 32_000, 32_000, 200); // 1 s → 1 window
    write_ramp_wav(tmp.path(), "d.wav", 32_000, 40_000, 300); // 1.25 s → 2 windows (padded)
    write_ramp_wav(tmp.path(), "e.wav", 32_000, 32_000, 400); // 1 s → 1 window
    let manifest = write_manifest(
        tmp.path(),
        "m.jsonl",
        &[
            r#"{"audio_path": "a.wav", "label": "pump_off", "split": "train"}"#,
            r#"{"audio_path": "b.wav", "label": "pump_on", "split": "train"}"#,
            r#"{"audio_path": "c.wav", "label": "pump_off", "split": "val"}"#,
            r#"{"audio_path": "d.wav", "label": "pump_on", "split": "val"}"#,
            r#"{"audio_path": "e.wav", "label": "pump_on", "split": "test"}"#,
        ],
    );

    let items = load_manifest(&manifest).unwrap();
    assert_eq!(items.len(), 5);

    let train = WindowedAudioDataset::new(items.clone(), cfg(), Split::Train).unwrap();
    let val = WindowedAudioDataset::new(items.clone(), cfg(), Split::Val).unwrap();
    let test = WindowedAudioDataset::new(items, cfg(), Split::Test).unwrap();

    assert_eq!(train.len(), 3 + 5); // a + b
    assert_eq!(val.len(), 1 + 2); // c + d
    assert_eq!(test.len(), 1); // e

    let first = train.get(0).unwrap();
    assert_eq!(first.waveform.len(), 32_000);
    // And unused to silence the warning; keep ManifestItem import live.
    let _: &ManifestItem = &ManifestItem {
        audio_path: first.file.clone(),
        label: "pump_off".into(),
        split: Some(Split::Train),
        start_s: None,
        end_s: None,
    };
}
