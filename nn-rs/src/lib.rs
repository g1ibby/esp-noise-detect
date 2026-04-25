//! `nn-rs` — pump-noise classifier pipeline in Burn.
//!
//! Project-local crate. Everything specific to the pump-noise task lives
//! here: the mel front-end, the TinyConv model, the training / eval CLIs
//! and the weight-export bridge. Reusable building blocks live in the
//! `cubek-*` and `burn-audiomentations` crates under `crates/`.
//!
//! Modules: [`mel`], [`data`], [`model`], [`train`], [`eval`]. The weight
//! export bridge ships as a binary under `src/bin/export_weights.rs`.

pub mod data;
pub mod eval;
pub mod mel;
pub mod model;
pub mod train;

pub use data::{
    AudioBatch, AudioBatcher, DatasetConfig, ManifestItem, Split, WindowedAudioDataset,
    WindowedAudioItem, load_manifest, seconds_to_samples,
};
pub use eval::{AggregateMode, BinaryMetrics, EvalOptions, EvalOutcome, Evaluator};
pub use mel::{MelConfig, MelExtractor};
pub use model::{TinyConv, TinyConvBlock, TinyConvConfig};
pub use train::{TrainAppConfig, TrainOutcome, Trainer};
