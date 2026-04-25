//! YAML-backed training config.
//!
//! The pipeline consumes one pre-composed YAML file. Every field has a
//! sane default so a minimal YAML (`dataset: {manifest_path: ...}`)
//! is sufficient to run. Unknown fields are ignored via
//! `#[serde(default)]`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Top-level config — one file per experiment.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainAppConfig {
    pub dataset: DatasetCfg,
    pub features: MelCfg,
    pub model: ModelCfg,
    pub augment: AugmentCfg,
    pub dm: DataModuleCfg,
    pub optim: OptimCfg,
    pub sched: SchedCfg,
    pub trainer: TrainerCfg,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetCfg {
    pub sample_rate: u32,
    pub window_s: f32,
    pub hop_s: f32,
    pub class_names: Vec<String>,
    pub manifest_path: Option<String>,
}

impl Default for DatasetCfg {
    fn default() -> Self {
        Self {
            sample_rate: 32_000,
            window_s: 1.0,
            hop_s: 0.5,
            class_names: vec!["pump_off".into(), "pump_on".into()],
            manifest_path: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MelCfg {
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: Option<f32>,
    pub n_fft: usize,
    pub hop_length_ms: f32,
    pub log: bool,
    pub eps: f32,
    pub normalize: bool,
    pub center: bool,
}

impl Default for MelCfg {
    fn default() -> Self {
        Self {
            n_mels: 64,
            fmin: 50.0,
            fmax: None,
            n_fft: 1024,
            hop_length_ms: 10.0,
            log: true,
            eps: 1e-10,
            normalize: true,
            center: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelCfg {
    pub channels: Vec<usize>,
    pub dropout: f64,
}

impl Default for ModelCfg {
    fn default() -> Self {
        Self {
            channels: vec![16, 32, 64],
            dropout: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AugmentCfg {
    pub train: AugmentPipelineCfg,
    pub val: AugmentPipelineCfg,
    pub test: AugmentPipelineCfg,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AugmentPipelineCfg {
    pub enabled: bool,
    pub shuffle: bool,
    pub same_on_batch: bool,
    pub transforms: Vec<AugmentTransformCfg>,
}

impl Default for AugmentPipelineCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            shuffle: true,
            same_on_batch: false,
            transforms: Vec::new(),
        }
    }
}

/// A single transform entry. `params` carries scalar / list-of-scalar
/// values; the augment builder (`crate::train::augment`) applies
/// transform-specific parsing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AugmentTransformCfg {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default = "default_prob")]
    pub probability: f64,
    #[serde(default)]
    pub params: BTreeMap<String, serde_yaml::Value>,
}

fn default_prob() -> f64 {
    0.5
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DataModuleCfg {
    pub batch_size: usize,
    /// `None` (YAML `null` or field omitted) → auto-pick based on
    /// physical CPU count. `Some(n)` respects the exact value — use
    /// `Some(0)` if you want Burn's single-threaded dataloader.
    pub num_workers: Option<usize>,
    pub pin_memory: bool,
    pub use_cache: bool,
    pub cache_dir: Option<String>,
}

impl Default for DataModuleCfg {
    fn default() -> Self {
        Self {
            batch_size: 64,
            num_workers: None,
            pin_memory: true,
            use_cache: false,
            cache_dir: None,
        }
    }
}

impl DataModuleCfg {
    /// Resolve `num_workers` into a concrete value, auto-picking when
    /// the field is `None`.
    ///
    /// Heuristic:
    /// * On Apple Silicon, use the performance-core count reported by
    ///   `sysctl hw.perflevel0.physicalcpu`. Decode work landing on
    ///   E-cores runs ~3× slower than on P-cores, so oversubscribing
    ///   past the P-core count just forces the scheduler onto E-cores
    ///   and hurts throughput. An M4 (4P + 6E) lands on 4.
    /// * Otherwise, fall back to `physical_cores - 1` (reserve one
    ///   for the main thread / GPU driver).
    /// * Clamp to `[2, 4]`. Past 4 workers the dataloader contends on
    ///   the dataset `Arc` and the decode-cache `Mutex` with no
    ///   throughput gain — our M4 bench showed 8 auto-picked workers
    ///   running ~2% slower than 4. Machines with slow storage where
    ///   decode is the real bottleneck should set `num_workers: N`
    ///   explicitly.
    pub fn resolved_num_workers(&self) -> usize {
        if let Some(n) = self.num_workers {
            return n;
        }
        let base = performance_core_count()
            .unwrap_or_else(|| num_physical_cpus().saturating_sub(1));
        base.clamp(2, 4)
    }
}

fn num_physical_cpus() -> usize {
    // sysinfo exposes physical-core count on macOS/Linux/Windows.
    // Fall back to logical cores via `std::thread::available_parallelism`
    // if the probe fails (unlikely in practice but cheap to guard).
    sysinfo::System::physical_core_count()
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(|n| n.get())
        })
        .unwrap_or(4)
}

/// Apple-Silicon performance-core count via `sysctl
/// hw.perflevel0.physicalcpu`. Returns `None` on non-macOS hosts and
/// on Intel Macs, where the `perflevel` namespace is absent.
fn performance_core_count() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        std::str::from_utf8(&out.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimCfg {
    pub name: String,
    pub lr: f64,
    pub weight_decay: f32,
    pub momentum: f32,
}

impl Default for OptimCfg {
    fn default() -> Self {
        Self {
            name: "adamw".into(),
            lr: 1e-3,
            weight_decay: 1e-2,
            momentum: 0.9,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedCfg {
    pub name: String,
    pub t_max: usize,
    pub min_lr: f64,
}

impl Default for SchedCfg {
    fn default() -> Self {
        Self {
            name: "none".into(),
            t_max: 100,
            min_lr: 1e-5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainerCfg {
    pub max_epochs: usize,
    pub gradient_clip_val: f32,
    pub seed: u64,
    pub early_stopping_patience: usize,
    pub artifact_dir: String,
}

impl Default for TrainerCfg {
    fn default() -> Self {
        Self {
            max_epochs: 20,
            gradient_clip_val: 1.0,
            seed: 42,
            early_stopping_patience: 5,
            artifact_dir: "runs/nn-rs".into(),
        }
    }
}

impl TrainAppConfig {
    /// Load a config from a YAML file on disk.
    pub fn from_yaml_file(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = serde_yaml::from_str(&text).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("yaml parse: {e}"))
        })?;
        Ok(cfg)
    }

    pub fn resolved_manifest(&self) -> Option<PathBuf> {
        self.dataset.manifest_path.as_ref().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let cfg = TrainAppConfig::default();
        assert_eq!(cfg.dataset.sample_rate, 32_000);
        assert_eq!(cfg.features.n_mels, 64);
        assert_eq!(cfg.features.n_fft, 1024);
        assert_eq!(cfg.model.channels, vec![16, 32, 64]);
        assert_eq!(cfg.optim.name, "adamw");
        assert_eq!(cfg.sched.name, "none");
        assert_eq!(cfg.trainer.max_epochs, 20);
    }

    #[test]
    fn parses_robust_session_overrides() {
        let yaml = r#"
dataset:
  window_s: 1.0
  hop_s: 0.5
dm:
  batch_size: 128
  use_cache: false
trainer:
  max_epochs: 30
optim:
  lr: 5.0e-4
  weight_decay: 1.0e-2
sched:
  name: cosine
  t_max: 30
  min_lr: 1.0e-5
augment:
  train:
    enabled: true
    shuffle: true
    transforms:
      - type: gain
        probability: 0.8
        params:
          min_gain_in_db: -12.0
          max_gain_in_db: 12.0
      - type: polarity_inversion
        probability: 0.5
"#;
        let cfg: TrainAppConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.dm.batch_size, 128);
        assert_eq!(cfg.trainer.max_epochs, 30);
        assert_eq!(cfg.sched.name, "cosine");
        assert_eq!(cfg.augment.train.transforms.len(), 2);
        assert_eq!(cfg.augment.train.transforms[0].ty, "gain");
        assert!(!cfg.dm.use_cache);
    }
}
