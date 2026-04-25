//! JSONL manifest loader.
//!
//! Required row fields: `audio_path` (string) and `label` (string).
//! Optional: `split` (`"train" | "val" | "test"`), `start_s` / `end_s`
//! (floats).
//!
//! Parsing is lenient — rows without the required fields, rows with
//! malformed JSON, and rows with an unknown `split` value are skipped
//! or coerced to `None`. Rows with a `label` outside the caller's
//! known set survive at this layer; `WindowedAudioDataset::new` is
//! where unknown labels are rejected.
//!
//! Relative `audio_path` values are resolved against the manifest
//! file's parent directory.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Train / val / test split tag. Anything outside this set is stored
/// as `None`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "train" => Some(Self::Train),
            "val" => Some(Self::Val),
            "test" => Some(Self::Test),
            _ => None,
        }
    }
}

/// Parsed manifest row. `audio_path` is always absolute after
/// [`load_manifest`]; relative paths in the JSONL are resolved against
/// the manifest's parent directory.
#[derive(Clone, Debug)]
pub struct ManifestItem {
    pub audio_path: PathBuf,
    pub label: String,
    pub split: Option<Split>,
    pub start_s: Option<f32>,
    pub end_s: Option<f32>,
}

#[derive(Deserialize)]
struct RawRow {
    audio_path: Option<String>,
    label: Option<String>,
    split: Option<String>,
    start_s: Option<f32>,
    end_s: Option<f32>,
}

/// Load and validate `manifest.jsonl`.
///
/// One pass over the file, skipping malformed lines and rows that
/// don't carry the required `audio_path` / `label` fields.
pub fn load_manifest(path: &Path) -> io::Result<Vec<ManifestItem>> {
    let path = path.to_path_buf();
    let base_dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let file = File::open(&path)?;
    let reader = BufReader::new(file);

    let mut items = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: RawRow = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue, // malformed JSON → skip.
        };
        let Some(audio_path) = row.audio_path else {
            continue;
        };
        let Some(label) = row.label else {
            continue;
        };
        let split = row.split.as_deref().and_then(Split::from_str);
        let audio_path = resolve_audio_path(&audio_path, &base_dir);
        items.push(ManifestItem {
            audio_path,
            label,
            split,
            start_s: row.start_s,
            end_s: row.end_s,
        });
    }
    Ok(items)
}

fn resolve_audio_path(audio_path: &str, base_dir: &Path) -> PathBuf {
    let p = PathBuf::from(audio_path);
    if p.is_absolute() {
        p
    } else {
        base_dir.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_manifest(dir: &Path, contents: &str) -> PathBuf {
        let p = dir.join("manifest.jsonl");
        let mut f = File::create(&p).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_valid_rows_and_skips_invalid() {
        let tmp = tempdir();
        let contents = r#"{"audio_path": "a.wav", "label": "pump_off", "split": "train"}
{"audio_path": "b.wav", "label": "pump_on"}
{"label": "pump_on"}
not-even-json
{"audio_path": "c.wav", "label": "pump_on", "split": "val", "start_s": 0.0, "end_s": 1.0}
"#;
        let path = write_manifest(tmp.path(), contents);
        let items = load_manifest(&path).unwrap();
        assert_eq!(items.len(), 3);

        assert_eq!(items[0].audio_path, tmp.path().join("a.wav"));
        assert_eq!(items[0].label, "pump_off");
        assert_eq!(items[0].split, Some(Split::Train));

        assert_eq!(items[1].label, "pump_on");
        assert_eq!(items[1].split, None);

        assert_eq!(items[2].split, Some(Split::Val));
        assert_eq!(items[2].start_s, Some(0.0));
        assert_eq!(items[2].end_s, Some(1.0));
    }

    #[test]
    fn unknown_split_becomes_none() {
        let tmp = tempdir();
        let path = write_manifest(
            tmp.path(),
            r#"{"audio_path": "a.wav", "label": "pump_off", "split": "holdout"}
"#,
        );
        let items = load_manifest(&path).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].split, None);
    }

    #[test]
    fn absolute_audio_path_preserved() {
        let tmp = tempdir();
        let path = write_manifest(
            tmp.path(),
            r#"{"audio_path": "/tmp/abs.wav", "label": "pump_on"}
"#,
        );
        let items = load_manifest(&path).unwrap();
        assert_eq!(items[0].audio_path, PathBuf::from("/tmp/abs.wav"));
    }

    // Minimal in-repo tempdir — avoid pulling in a dep for a single test
    // helper. Drop cleans up on panic just like `tempfile::tempdir`.
    fn tempdir() -> TempDir {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "nn-rs-manifest-test-{}-{n}",
            std::process::id(),
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
