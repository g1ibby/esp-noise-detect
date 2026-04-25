//! `LowPassFilter` / `HighPassFilter` — per-example cutoff sampling on top
//! of `cubek-sinc-filter`.
//!
//! * Cutoff `f_c` drawn uniformly in **mel-space** between `min_cutoff_freq`
//!   and `max_cutoff_freq`, then converted back to Hertz. The mel
//!   distribution biases more samples toward perceptually useful low
//!   cutoffs.
//! * The sampled cutoff is quantized into one of `num_buckets` evenly-spaced
//!   (in mel-space) FIR buckets and dispatched via `cubek-sinc-filter`'s
//!   `apply_per_row`. The 1-2% cutoff quantization error is acceptable.
//!
//! For unselected Bernoulli rows, both filters route through a synthetic
//! identity row prepended at bank index 0:
//!
//! * HPF: `cubek-sinc-filter` already defines a cutoff of 0 as the null
//!   filter, which makes `x - lowpass(x)` = `x` — exactly identity.
//! * LPF: the prepended row is a cutoff at Nyquist (normalized 0.5). At
//!   `zeros = 8` and that cutoff the FIR is an extremely narrow low-pass
//!   that collapses to a delta for practical purposes. This keeps the
//!   p=0 case numerically equivalent to the input inside the f32
//!   rounding budget and matches HPF's bucket-0-is-identity convention.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_sinc_filter::{FilterMode, LowPassFilterBank};

use crate::transform::{Transform, TransformRng, bernoulli_mask, sample_uniform_batch};

/// Slaney / HTK-ish mel mapping (2595 * log10(1 + f / 700)).
pub(crate) fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

pub(crate) fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// Build a mel-spaced bucket set — `num_buckets + 1` boundaries in mel,
/// `num_buckets` centers in Hz mapped to normalized cutoff `f/fs`. Using
/// the centers (not boundaries) as the bank's cutoffs gives the minimum
/// quantization error vs. the continuous distribution we're sampling.
pub(crate) fn build_mel_buckets(
    min_hz: f32,
    max_hz: f32,
    num_buckets: u32,
    sample_rate: u32,
) -> Vec<f32> {
    assert!(num_buckets > 0, "num_buckets must be > 0");
    assert!(min_hz > 0.0 && max_hz > min_hz);
    let min_mel = hz_to_mel(min_hz);
    let max_mel = hz_to_mel(max_hz);
    let n = num_buckets as usize;
    (0..n)
        .map(|i| {
            let m = min_mel + (max_mel - min_mel) * (i as f32 + 0.5) / (n as f32);
            mel_to_hz(m) / sample_rate as f32
        })
        .collect()
}

/// Pick the nearest bucket center in mel-space for a Hertz draw.
pub(crate) fn quantize_in_mel(
    freq_hz: f32,
    min_hz: f32,
    max_hz: f32,
    num_buckets: u32,
) -> u32 {
    let min_mel = hz_to_mel(min_hz);
    let max_mel = hz_to_mel(max_hz);
    let mel = hz_to_mel(freq_hz.clamp(min_hz, max_hz));
    let frac = (mel - min_mel) / (max_mel - min_mel);
    let idx = (frac * num_buckets as f32).floor() as i64;
    let clamped = idx.clamp(0, num_buckets as i64 - 1);
    clamped as u32
}

/// Low-pass filter transform with mel-sampled per-row cutoffs.
///
/// Bank construction is host-only and runs at `new` time, so every
/// `apply` launches exactly one `cubek-sinc-filter` kernel.
pub struct LowPassFilter<R: Runtime> {
    pub min_cutoff_freq: f32,
    pub max_cutoff_freq: f32,
    pub sample_rate: u32,
    pub probability: f64,
    pub num_buckets: u32,
    bank: LowPassFilterBank<R>,
}

impl<R: Runtime> LowPassFilter<R> {
    pub fn new(
        client: ComputeClient<R>,
        min_cutoff_freq: f32,
        max_cutoff_freq: f32,
        sample_rate: u32,
        probability: f64,
        num_buckets: u32,
        dtype: StorageType,
    ) -> Self {
        assert!(
            min_cutoff_freq < max_cutoff_freq,
            "min_cutoff_freq ({min_cutoff_freq}) must be < max_cutoff_freq ({max_cutoff_freq})",
        );
        // Prepend a cutoff-0.5 (Nyquist) row. At this cutoff the windowed
        // sinc collapses to a Kronecker delta — `sinc(π*n) = δ[n]` for
        // integer `n` — so the resulting filter row is exactly the
        // identity after DC normalization. This is the "unselected Bernoulli
        // row" slot, routed to by `apply` below.
        let mut cutoffs = vec![0.5f32];
        cutoffs.extend(build_mel_buckets(
            min_cutoff_freq,
            max_cutoff_freq,
            num_buckets,
            sample_rate,
        ));
        let bank = LowPassFilterBank::new(client, &cutoffs, 8, dtype);
        Self {
            min_cutoff_freq,
            max_cutoff_freq,
            sample_rate,
            probability,
            num_buckets,
            bank,
        }
    }
}

impl<R: Runtime> Transform<R> for LowPassFilter<R> {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(samples.shape().len(), 2, "LowPassFilter expects (batch, time)");
        let batch = samples.shape()[0];

        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let cutoffs_hz_mel = sample_uniform_batch(
            batch,
            hz_to_mel(self.min_cutoff_freq),
            hz_to_mel(self.max_cutoff_freq),
            rng.host(),
        );
        // Bucket 0 is the synthetic identity (cutoff = Nyquist); mel-bucket
        // centers start at index 1.
        let indices: Vec<u32> = cutoffs_hz_mel
            .iter()
            .zip(mask.iter())
            .map(|(mel, m)| {
                if *m > 0.5 {
                    let hz = mel_to_hz(*mel);
                    1 + quantize_in_mel(
                        hz,
                        self.min_cutoff_freq,
                        self.max_cutoff_freq,
                        self.num_buckets,
                    )
                } else {
                    0
                }
            })
            .collect();

        dispatch_bank(&self.bank, samples, &indices, FilterMode::LowPass)
    }
}

/// High-pass filter transform. Uses cutoff=0 (identity) as the no-op
/// bucket — `cubek-sinc-filter` documents cutoff-0 high-pass as exactly
/// returning the input.
pub struct HighPassFilter<R: Runtime> {
    pub min_cutoff_freq: f32,
    pub max_cutoff_freq: f32,
    pub sample_rate: u32,
    pub probability: f64,
    pub num_buckets: u32,
    bank: LowPassFilterBank<R>,
}

impl<R: Runtime> HighPassFilter<R> {
    pub fn new(
        client: ComputeClient<R>,
        min_cutoff_freq: f32,
        max_cutoff_freq: f32,
        sample_rate: u32,
        probability: f64,
        num_buckets: u32,
        dtype: StorageType,
    ) -> Self {
        assert!(
            min_cutoff_freq < max_cutoff_freq,
            "min_cutoff_freq ({min_cutoff_freq}) must be < max_cutoff_freq ({max_cutoff_freq})",
        );
        // Prepend a 0-cutoff row so bucket 0 is the exact identity high-pass
        // used for unselected rows. The mel-bucket centers follow.
        let mut cutoffs = vec![0.0f32];
        cutoffs.extend(build_mel_buckets(
            min_cutoff_freq,
            max_cutoff_freq,
            num_buckets,
            sample_rate,
        ));
        let bank = LowPassFilterBank::new(client, &cutoffs, 8, dtype);
        Self {
            min_cutoff_freq,
            max_cutoff_freq,
            sample_rate,
            probability,
            num_buckets,
            bank,
        }
    }
}

impl<R: Runtime> Transform<R> for HighPassFilter<R> {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(samples.shape().len(), 2, "HighPassFilter expects (batch, time)");
        let batch = samples.shape()[0];

        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let cutoffs_hz_mel = sample_uniform_batch(
            batch,
            hz_to_mel(self.min_cutoff_freq),
            hz_to_mel(self.max_cutoff_freq),
            rng.host(),
        );
        // Bucket 0 is the identity row; the "real" buckets start at index 1.
        let indices: Vec<u32> = cutoffs_hz_mel
            .iter()
            .zip(mask.iter())
            .map(|(mel, m)| {
                if *m > 0.5 {
                    let hz = mel_to_hz(*mel);
                    1 + quantize_in_mel(
                        hz,
                        self.min_cutoff_freq,
                        self.max_cutoff_freq,
                        self.num_buckets,
                    )
                } else {
                    0
                }
            })
            .collect();

        dispatch_bank(&self.bank, samples, &indices, FilterMode::HighPass)
    }
}

fn dispatch_bank<R: Runtime>(
    bank: &LowPassFilterBank<R>,
    samples: TensorHandle<R>,
    indices: &[u32],
    mode: FilterMode,
) -> TensorHandle<R> {
    let client = <R as Runtime>::client(&Default::default());
    let idx_handle = client.create_from_slice(u32::as_bytes(indices));
    let idx = TensorHandle::<R>::new_contiguous(
        vec![indices.len()],
        idx_handle,
        u32::as_type_native_unchecked().storage_type(),
    );
    bank.apply_per_row(samples, idx, mode)
}
