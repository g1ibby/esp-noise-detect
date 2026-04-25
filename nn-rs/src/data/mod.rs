//! Data pipeline: manifest loading, windowed dataset, and batching.
//!
//! * [`config::DatasetConfig`] — sample rate, window/hop sizes, class names.
//! * [`manifest::load_manifest`] — reads `manifest.jsonl` leniently.
//! * [`dataset::WindowedAudioDataset`] — per-window indexer: WAV decode
//!   via `hound`, mono mix-down, strict source-SR check, zero-padded
//!   tail windowing. Implements the Burn `Dataset` trait.
//! * [`batcher::AudioBatcher`] — stacks windows into `(batch, time)` and
//!   labels into `(batch,)`. Implements the Burn `Batcher` trait.

pub mod batcher;
pub mod config;
pub mod dataset;
pub mod manifest;

pub use batcher::{AudioBatch, AudioBatcher};
pub use config::{DatasetConfig, seconds_to_samples};
pub use dataset::{WindowedAudioDataset, WindowedAudioItem};
pub use manifest::{ManifestItem, Split, load_manifest};
