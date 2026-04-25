//! Shared CubeCL kernels used by more than one transform.
//!
//! Anything single-transform-specific lives inside that transform's module;
//! kernels re-used across Gain / PolarityInversion / mixing paths land here
//! so we don't duplicate launch boilerplate.

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

/// `(cube_count, cube_dim)` for a one-thread-per-element launch.
///
/// Delegates to [`cubecl::calculate_cube_count_elemwise`], which spreads
/// the work across X/Y/Z to stay within the backend's per-axis
/// `max_cube_count`. The returned layout may overshoot `num_elems`;
/// kernels guard with `terminate!()` on `ABSOLUTE_POS >= len`.
pub(crate) fn elemwise_launch_dims<R: Runtime>(
    client: &ComputeClient<R>,
    num_elems: usize,
    threads_per_cube: u32,
) -> (CubeCount, CubeDim) {
    assert!(threads_per_cube > 0, "threads_per_cube must be > 0");
    let cube_dim = CubeDim::new_1d(threads_per_cube);
    let cube_count = calculate_cube_count_elemwise(client, num_elems, cube_dim);
    (cube_count, cube_dim)
}

/// `samples[b, t] *= scale[b]`.
///
/// Used by both [`crate::Gain`] and [`crate::PolarityInversion`]. They
/// differ only in how `scale` is populated (dB-converted amplitudes vs.
/// ±1). Out-of-range threads terminate; the kernel operates in place on
/// a `(batch, time)` rank-2 tensor with row-major layout.
#[cube(launch)]
pub(crate) fn per_example_scale_kernel<F: Float>(
    samples: &mut Tensor<F>,
    scale: &Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= samples.len() {
        terminate!();
    }
    let time = samples.shape(1);
    let b = pos / time;
    samples[pos] = samples[pos] * scale[b];
}

/// `output[b, t] = x[b, t] + y[b, t] * scale[b]`.
///
/// Shared between [`crate::AddColoredNoise`] and any future additive mixer.
/// Kept runtime-generic (f32 only in practice) so we can reuse it without
/// special-casing the Float parameter at the call site.
#[cube(launch)]
pub(crate) fn add_with_scale_kernel<F: Float>(
    x: &Tensor<F>,
    y: &Tensor<F>,
    output: &mut Tensor<F>,
    scale: &Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let time = output.shape(1);
    let b = pos / time;
    output[pos] = x[pos] + y[pos] * scale[b];
}

/// Per-row sum of squares reduction: `out[b] = Σ_t x[b, t]^2`.
///
/// Simple one-thread-per-row reduction — the batch sizes we deal with
/// (≤ 64) and the row widths (≤ 32768) fit comfortably, and the noise
/// path hits this twice per call (signal RMS + noise RMS). If the training
/// loop ever flags this as a bottleneck we can drop in a proper
/// cooperative reduction; today the one-thread-per-row variant is fine.
#[cube(launch)]
pub(crate) fn per_row_sum_sq_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
) {
    let b = ABSOLUTE_POS;
    if b >= out.len() {
        terminate!();
    }
    let time = x.shape(1);
    let mut acc = F::new(0.0);
    let mut t: u32 = 0;
    while (t as usize) < time {
        let v = x[b * time + t as usize];
        acc += v * v;
        t += 1u32;
    }
    out[b] = acc;
}
