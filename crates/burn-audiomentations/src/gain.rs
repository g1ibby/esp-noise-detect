//! `Gain` — uniform-in-dB amplitude scaling.
//!
//! Each batch row gets its own gain drawn uniformly in
//! `[min_gain_in_db, max_gain_in_db]` (converted to a linear ratio
//! `10^(dB/20)`). Unselected rows (Bernoulli `probability`) get gain
//! 0 dB = ratio 1.0 — the single kernel runs over the whole batch either
//! way.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::kernels::{elemwise_launch_dims, per_example_scale_kernel};
use crate::transform::{Transform, TransformRng, bernoulli_mask, sample_uniform_batch};

/// Per-example uniform-in-dB scalar multiply.
///
/// * `min_gain_in_db` / `max_gain_in_db` — range used when the sample is
///   selected by the Bernoulli `probability`. May be equal (constant gain).
/// * `probability` — Bernoulli `p` for whether this transform fires on a
///   given batch row. `1.0` = always.
pub struct Gain {
    pub min_gain_in_db: f32,
    pub max_gain_in_db: f32,
    pub probability: f64,
}

impl Gain {
    pub fn new(min_gain_in_db: f32, max_gain_in_db: f32, probability: f64) -> Self {
        assert!(
            min_gain_in_db <= max_gain_in_db,
            "min_gain_in_db ({min_gain_in_db}) > max_gain_in_db ({max_gain_in_db})",
        );
        Self {
            min_gain_in_db,
            max_gain_in_db,
            probability,
        }
    }
}

impl<R: Runtime> Transform<R> for Gain {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(
            samples.shape().len(),
            2,
            "Gain expects (batch, time), got rank {}",
            samples.shape().len(),
        );
        let batch = samples.shape()[0];
        let time = samples.shape()[1];
        assert!(time > 0 && batch > 0);

        // Draw gains in dB per selected row; unselected rows get 0 dB.
        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let dbs = sample_uniform_batch(batch, self.min_gain_in_db, self.max_gain_in_db, rng.host());
        let scales: Vec<f32> = dbs
            .iter()
            .zip(mask.iter())
            .map(|(db, m)| if *m > 0.5 { 10f32.powf(db / 20.0) } else { 1.0 })
            .collect();

        let client = <R as Runtime>::client(&Default::default());
        let dtype = samples.dtype;
        let scale_handle = client.create_from_slice(f32::as_bytes(&scales));
        let scale = TensorHandle::<R>::new_contiguous(vec![batch], scale_handle, dtype);

        let num_elems = batch * time;
        let (cube_count, cube_dim) = elemwise_launch_dims(&client, num_elems, 256);

        per_example_scale_kernel::launch::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            samples.clone().binding().into_tensor_arg(),
            scale.binding().into_tensor_arg(),
        );

        samples
    }
}
