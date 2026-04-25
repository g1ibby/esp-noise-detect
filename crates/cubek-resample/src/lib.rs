//! `cubek-resample` — rational-ratio sinc-windowed FIR resampler.
//!
//! ## Algorithm in 30 seconds
//!
//! Given input at `old_sr` and output at `new_sr`, divide by their GCD to
//! get the smallest rational representation. Build `new_sr` polyphase FIR
//! kernels — each of length `2*width + old_sr` — that implement
//! `sinc(pi * sr * (k/old_sr - i/new_sr)) * cos(·)²` for output phase `i`.
//! Replicate-pad the input on both ends, then for every output sample
//! `t = j*new_sr + i` the result is `Σ_k padded[j*old_sr + k] * kernel[i, k]`.
//! The conv + interleave happens in one [`cubecl`] kernel (one thread per
//! output sample). Kernel tables are built host-side in [`kernels`] once
//! and uploaded at [`Resampler::new`] time — never rebuilt per batch.
//!
//! ## Quick usage
//!
//! ```ignore
//! use cubecl::prelude::*;
//! use cubecl::std::tensor::TensorHandle;
//! use cubek_resample::Resampler;
//!
//! let client = <R as Runtime>::client(&Default::default());
//! let dtype = f32::as_type_native_unchecked().storage_type();
//!
//! // One-time build. Upload cost ≈ new_sr * (2*width + old_sr) f32s.
//! let resampler: Resampler<R> =
//!     Resampler::new(client.clone(), 32_000, 44_100, 24, 0.945, dtype);
//!
//! // Forward: signal is `(batch, time)`.
//! let y = resampler.apply(signal, None);
//! ```
//!
//! ## What's in this crate
//!
//! * [`Resampler`] — polyphase FIR resampler over `cubecl::Runtime`.
//! * [`kernels`] — host-side kernel-bank builder.
//! * [`fast_shifts`] — integer-ratio pitch-shift enumeration. Pure host
//!   code, no dependency on the GPU path.

mod kernels;
mod resample;

pub mod fast_shifts;

pub use resample::Resampler;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

/// `(cube_count, cube_dim)` for a one-thread-per-element launch.
///
/// Wraps [`cubecl::calculate_cube_count_elemwise`] so the cube-count
/// selection uses each backend's real `max_cube_count`.
pub(crate) fn elemwise_launch_dims<R: Runtime>(
    client: &ComputeClient<R>,
    num_elems: usize,
    threads_per_cube: u32,
) -> (CubeCount, CubeDim) {
    assert!(threads_per_cube > 0);
    let cube_dim = CubeDim::new_1d(threads_per_cube);
    let cube_count = calculate_cube_count_elemwise(client, num_elems, cube_dim);
    (cube_count, cube_dim)
}
