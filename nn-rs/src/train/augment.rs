//! Build a `burn-audiomentations::Compose` from an [`AugmentPipelineCfg`].
//!
//! Maps `{type, probability, params}` entries in the YAML to concrete
//! `Box<dyn Transform<R>>` objects. Unknown / unsupported transform types
//! are skipped with a warning to stderr, keeping partial configs
//! functional.
//!
//! Supported transforms:
//!
//! | YAML `type`             | Rust transform         |
//! |-------------------------|------------------------|
//! | `gain`                  | `Gain`                 |
//! | `polarity_inversion`    | `PolarityInversion`    |
//! | `add_colored_noise`     | `AddColoredNoise`      |
//! | `highpass`              | `HighPassFilter`       |
//! | `lowpass`               | `LowPassFilter`        |
//! | `pitch_shift`           | `PitchShift`           |
//! | `time_masking`          | `TimeMasking`          |
//!
//! Unsupported types (`add_background_noise`, `shift`, etc.) are dropped.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::ir::StorageType;
use cubecl::prelude::CubePrimitive;

use burn_audiomentations::{
    AddColoredNoise, Compose, Gain, HighPassFilter, LowPassFilter, PitchShift, PolarityInversion,
    TimeMasking, Transform,
};

use super::config::{AugmentPipelineCfg, AugmentTransformCfg};

/// Default number of quantization buckets for mel-spaced filter banks.
const DEFAULT_FILTER_BUCKETS: u32 = 32;

/// Build a `Compose<R>` from a pipeline config. Returns `None` when the
/// pipeline is disabled or has no transforms — saves the training loop a
/// per-batch call into a no-op apply.
pub fn build_pipeline<R: Runtime>(
    client: &ComputeClient<R>,
    sample_rate: u32,
    pipeline: &AugmentPipelineCfg,
) -> Option<Compose<R>> {
    if !pipeline.enabled || pipeline.transforms.is_empty() {
        return None;
    }
    let dtype = f32::as_type_native_unchecked().storage_type();
    let mut transforms: Vec<Box<dyn Transform<R>>> = Vec::new();
    for t in &pipeline.transforms {
        if let Some(built) = build_transform::<R>(client, sample_rate, t, dtype) {
            transforms.push(built);
        } else {
            eprintln!(
                "[augment] skipping unsupported transform type `{}` — \
                 implement it in burn-audiomentations if you need it",
                t.ty
            );
        }
    }
    if transforms.is_empty() {
        return None;
    }
    Some(Compose::new(transforms).with_shuffle(pipeline.shuffle))
}

fn build_transform<R: Runtime>(
    client: &ComputeClient<R>,
    sample_rate: u32,
    cfg: &AugmentTransformCfg,
    dtype: StorageType,
) -> Option<Box<dyn Transform<R>>> {
    let p = cfg.probability;
    match cfg.ty.as_str() {
        "gain" => {
            let min_db = get_f32(cfg, "min_gain_in_db", -12.0);
            let max_db = get_f32(cfg, "max_gain_in_db", 12.0);
            Some(Box::new(Gain::new(min_db, max_db, p)))
        }
        "polarity_inversion" => Some(Box::new(PolarityInversion::new(p))),
        "add_colored_noise" => {
            let min_snr = get_f32(cfg, "min_snr_in_db", 3.0);
            let max_snr = get_f32(cfg, "max_snr_in_db", 30.0);
            let min_f = get_f32(cfg, "min_f_decay", -2.0);
            let max_f = get_f32(cfg, "max_f_decay", 2.0);
            Some(Box::new(AddColoredNoise::new(
                min_snr,
                max_snr,
                min_f,
                max_f,
                sample_rate,
                p,
            )))
        }
        "highpass" => {
            let min_hz = get_f32(cfg, "min_cutoff_freq", 40.0);
            let max_hz = get_f32(cfg, "max_cutoff_freq", 200.0);
            Some(Box::new(HighPassFilter::new(
                client.clone(),
                min_hz,
                max_hz,
                sample_rate,
                p,
                DEFAULT_FILTER_BUCKETS,
                dtype,
            )))
        }
        "lowpass" => {
            let min_hz = get_f32(cfg, "min_cutoff_freq", 8000.0);
            let max_hz = get_f32(cfg, "max_cutoff_freq", 14000.0);
            Some(Box::new(LowPassFilter::new(
                client.clone(),
                min_hz,
                max_hz,
                sample_rate,
                p,
                DEFAULT_FILTER_BUCKETS,
                dtype,
            )))
        }
        "pitch_shift" => {
            let min_st = get_f32(cfg, "min_transpose_semitones", -2.0);
            let max_st = get_f32(cfg, "max_transpose_semitones", 2.0);
            Some(Box::new(PitchShift::new(
                client.clone(),
                sample_rate,
                min_st,
                max_st,
                p,
                dtype,
            )))
        }
        "time_masking" => {
            let num = get_u32(cfg, "num_time_intervals", 2);
            let width = get_u32(cfg, "max_width", 400);
            Some(Box::new(TimeMasking::new(num, width, p)))
        }
        _ => None,
    }
}

fn get_f32(cfg: &AugmentTransformCfg, key: &str, default: f32) -> f32 {
    cfg.params
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(default)
}

fn get_u32(cfg: &AugmentTransformCfg, key: &str, default: u32) -> u32 {
    cfg.params
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|u| u as u32)
        .unwrap_or(default)
}
