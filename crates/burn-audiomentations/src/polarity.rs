//! `PolarityInversion` — flip the sign of selected batch rows.
//!
//! Multiplies samples by -1 on rows where the Bernoulli `p` draw succeeds.
//! Shares the per-example scale kernel with [`crate::Gain`].

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::kernels::{elemwise_launch_dims, per_example_scale_kernel};
use crate::transform::{Transform, TransformRng, bernoulli_mask};

/// Per-example sign flip with probability `probability`.
pub struct PolarityInversion {
    pub probability: f64,
}

impl PolarityInversion {
    pub fn new(probability: f64) -> Self {
        Self { probability }
    }
}

impl<R: Runtime> Transform<R> for PolarityInversion {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(
            samples.shape().len(),
            2,
            "PolarityInversion expects (batch, time), got rank {}",
            samples.shape().len(),
        );
        let batch = samples.shape()[0];
        let time = samples.shape()[1];
        assert!(batch > 0 && time > 0);

        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let scales: Vec<f32> = mask
            .iter()
            .map(|m| if *m > 0.5 { -1.0 } else { 1.0 })
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
