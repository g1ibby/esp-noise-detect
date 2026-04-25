//! `Compose` — sequence of waveform transforms with optional shuffle.
//!
//! Stored transforms are applied in order; when `shuffle == true` the order
//! is permuted per `apply` call using the host RNG.
//!
//! The stored type is `Box<dyn Transform<R>>` so callers can mix transforms
//! of different concrete types in one pipeline.

use cubecl::Runtime;
use cubecl::std::tensor::TensorHandle;

use crate::transform::{Transform, TransformRng};

pub struct Compose<R: Runtime> {
    pub transforms: Vec<Box<dyn Transform<R>>>,
    pub shuffle: bool,
}

impl<R: Runtime> Compose<R> {
    pub fn new(transforms: Vec<Box<dyn Transform<R>>>) -> Self {
        Self {
            transforms,
            shuffle: false,
        }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
}

impl<R: Runtime> Transform<R> for Compose<R> {
    fn apply(&self, mut samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        if self.shuffle {
            // Fisher–Yates over indices — avoids cloning the boxed trait
            // objects while still giving deterministic per-batch ordering.
            use rand::Rng;
            let mut order: Vec<usize> = (0..self.transforms.len()).collect();
            let host = rng.host();
            for i in (1..order.len()).rev() {
                let j = host.random_range(0..=i);
                order.swap(i, j);
            }
            for &i in &order {
                samples = self.transforms[i].apply(samples, rng);
            }
        } else {
            for t in &self.transforms {
                samples = t.apply(samples, rng);
            }
        }
        samples
    }
}
