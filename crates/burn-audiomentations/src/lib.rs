//! `burn-audiomentations` — waveform augmentation transforms for the
//! pump-noise training loop.
//!
//! ## Transforms
//!
//! Each transform is a `struct` that impls [`Transform<R>`](transform::Transform)
//! for any `cubecl::Runtime`. Transforms consume and return
//! `cubecl::std::tensor::TensorHandle<R>` tensors of shape `(batch, time)` —
//! the same rank-2 layout the `cubek-sinc-filter` bank and the `cubek-stft`
//! launcher use. Multi-channel inputs are expected to be flattened by the
//! caller (the noise-detect training pipeline is mono, so we don't need a
//! channel axis).
//!
//! * [`Gain`] — per-example uniform-in-dB scalar multiply.
//! * [`PolarityInversion`] — per-example sign flip.
//! * [`TimeMasking`] — crossfaded splice with silent tail pad.
//! * [`AddColoredNoise`] — per-example `1/f^α` noise at a target SNR.
//! * [`LowPassFilter`] / [`HighPassFilter`] — mel-space cutoff sampled per
//!   example, quantized into a `cubek-sinc-filter` bucket.
//! * [`PitchShift`] — STFT → phase-vocoder → iSTFT → fractional resample,
//!   per-row nearest-ratio quantized into a precomputed resampler bank.
//! * [`Compose`] — sequence transforms with optional per-batch shuffle.
//!
//! ## Randomness model
//!
//! Host-side reproducibility is cheap and convenient, GPU-side reproducibility
//! is expensive and awkward. We therefore draw every per-example *scalar*
//! parameter (gain, SNR, cutoff, Bernoulli apply mask, splice starts / widths)
//! from a host-side `rand::rngs::StdRng` seeded via [`TransformRng::new`]
//! and uploaded to the device in one shot as `(batch,)` tensors. The only
//! place we use an on-device PRNG is [`AddColoredNoise`]'s white-noise
//! generator, which goes through `cubek_random::seed` /
//! `cubek_random::random_normal` — both are process-global but
//! deterministic once seeded.
//!
//! This keeps the augmentation pipeline bit-reproducible between runs as long
//! as the caller seeds both sources. See [`TransformRng::seed_cubek_random`].
//!
//! ## Bernoulli `p` and "identity for unselected"
//!
//! Each transform is applied to a Bernoulli-selected subset of the batch.
//! Rather than gather/scatter, every transform has an "identity value" for its
//! parameters (gain 0 dB, cutoff = Nyquist for LPF, width 0 for TimeMasking,
//! SNR = ∞ via zeroed noise scale for AddColoredNoise), and unselected samples
//! get that identity value before the single batched kernel runs. This keeps
//! the hot path branch-free and saves us from round-tripping through a
//! selection mask.

pub mod compose;
pub mod filters;
pub mod gain;
pub mod kernels;
pub mod noise;
pub mod pitch_shift;
pub mod polarity;
pub mod time_masking;
pub mod transform;

pub use compose::Compose;
pub use filters::{HighPassFilter, LowPassFilter};
pub use gain::Gain;
pub use noise::AddColoredNoise;
pub use pitch_shift::PitchShift;
pub use polarity::PolarityInversion;
pub use time_masking::TimeMasking;
pub use transform::{Transform, TransformRng};
