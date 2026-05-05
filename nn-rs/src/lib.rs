//! `nn-rs` — pump-noise classifier pipeline in Burn.
//!
//! Project-local crate. Everything specific to the pump-noise task lives
//! here: the mel front-end, the TinyConv model, the training / eval CLIs
//! and the weight-export bridge. Reusable building blocks live in the
//! `cubek-*` and `burn-audiomentations` crates under `crates/`.
//!
//! Modules: [`mel`], [`data`], [`model`], [`train`], [`eval`]. The weight
//! export bridge ships as a binary under `src/bin/export_weights.rs`; the
//! native ESP-DL exporter ships as `src/bin/export_espdl.rs`.
//!
//! `model` is the only module a downstream crate can pull in without
//! selecting a GPU backend feature — the rest depend on cubecl runtimes
//! (`mel`, `data`, `train`) or on `train::config` (`eval`). The
//! `burn-espdl-export` crate consumes `model` through the
//! `default-features = false` workspace dep to read TinyConv parameters
//! on the host without dragging cuda/wgpu into its dependency tree.

pub mod model;

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub mod data;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub mod espdl;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub mod espdl_calib;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub mod eval;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub mod mel;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub mod train;

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub use data::{
    AudioBatch, AudioBatcher, DatasetConfig, ManifestItem, Split, WindowedAudioDataset,
    WindowedAudioItem, load_manifest, seconds_to_samples,
};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub use eval::{AggregateMode, BinaryMetrics, EvalOptions, EvalOutcome, Evaluator};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub use mel::{MelConfig, MelExtractor};
pub use model::{TinyConv, TinyConvBlock, TinyConvConfig};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
))]
pub use train::{TrainAppConfig, TrainOutcome, Trainer};
