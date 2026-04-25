//! Training loop.
//!
//! * [`config`] — YAML-backed config tree.
//! * [`augment`] — maps `AugmentPipelineCfg` to a
//!   `burn-audiomentations::Compose` of concrete transforms.
//! * [`metrics`] — host-side macro-F1 / accuracy / mean-loss tally for
//!   per-epoch reporting and early stopping.
//! * [`runner`] — the [`Trainer`](runner::Trainer): data loaders,
//!   autodiff wiring, optimizer, cosine LR, checkpointing.

pub mod augment;
pub mod config;
pub mod metrics;
pub mod runner;

pub use config::TrainAppConfig;
pub use metrics::BinaryClassificationStats;
pub use runner::{StageTimings, TrainOutcome, Trainer};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
))]
pub use runner::{
    AutodiffBackendConcrete, Backend, SelectedAutodiff, SelectedCube, SelectedDevice,
    SelectedRuntime, TrainerConcrete, WgpuAutodiff, WgpuCube,
};
