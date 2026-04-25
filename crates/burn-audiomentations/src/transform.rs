//! Core trait and seeded RNG wrapper shared by every transform.
//!
//! Transforms consume a `(batch, time)` f32 tensor handle and return one of
//! the same shape. Parameter sampling is deterministic given the state of
//! [`TransformRng`], so identical seeds reproduce identical augmentations
//! run-to-run (the `cubek_random` state is process-global — see
//! [`TransformRng::seed_cubek_random`] for the caveat).

use cubecl::Runtime;
use cubecl::std::tensor::TensorHandle;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Host-side seeded PRNG used by every transform for parameter sampling.
///
/// Wraps a `rand::rngs::StdRng` so every transform can draw deterministic
/// per-batch parameters. For [`crate::AddColoredNoise`] we also forward a
/// seed into `cubek_random`'s process-global PRNG via
/// [`TransformRng::seed_cubek_random`] — that re-seed happens per batch and
/// is sensitive to interleaving with other callers of `cubek_random::seed`.
pub struct TransformRng {
    rng: StdRng,
    cubek_counter: u64,
}

impl TransformRng {
    /// Seed both the host RNG and the counter that feeds
    /// `cubek_random::seed` from a single u64.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            cubek_counter: seed,
        }
    }

    /// Borrow the host RNG for per-batch scalar sampling.
    pub fn host(&mut self) -> &mut StdRng {
        &mut self.rng
    }

    /// Advance the cubek-random seed and push it into the process-global
    /// state. `cubek_random::seed` holds a `Mutex<Option<StdRng>>` under
    /// the hood, so this call is safe between launches but **not**
    /// concurrent-safe across threads. The training loop drives
    /// augmentation from a single thread, so this isn't a constraint in
    /// practice.
    pub fn seed_cubek_random(&mut self) -> u64 {
        self.cubek_counter = self.cubek_counter.wrapping_add(0x9E37_79B9_7F4A_7C15);
        cubek_random::seed(self.cubek_counter);
        self.cubek_counter
    }
}

/// A single waveform augmentation.
///
/// The runtime type `R` is threaded through so every transform can launch
/// its own kernels on the caller's `ComputeClient`. Input and output shape
/// are both `(batch, time)`; returning the same handle is allowed (several
/// transforms do this under `probability == 0` or `rate == 1.0`).
pub trait Transform<R: Runtime>: Send + Sync {
    /// Apply the transform to `samples`, possibly in place, using `rng` for
    /// any parameter sampling.
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R>;
}

/// Draw a (batch,) Bernoulli mask on the host. Returns 0.0 / 1.0 floats so
/// it can be uploaded straight into kernels that multiply against it.
pub(crate) fn bernoulli_mask(batch: usize, p: f64, rng: &mut StdRng) -> Vec<f32> {
    use rand::Rng;
    assert!((0.0..=1.0).contains(&p), "p out of [0, 1]: {p}");
    (0..batch)
        .map(|_| if rng.random::<f64>() < p { 1.0 } else { 0.0 })
        .collect()
}

/// Host-side uniform draw used by every transform with a parameter range.
pub(crate) fn sample_uniform_batch(batch: usize, low: f32, high: f32, rng: &mut StdRng) -> Vec<f32> {
    use rand::Rng;
    assert!(
        high >= low,
        "sample_uniform_batch: high ({high}) must be >= low ({low})",
    );
    if (high - low).abs() < f32::EPSILON {
        return vec![low; batch];
    }
    (0..batch).map(|_| rng.random_range(low..high)).collect()
}
