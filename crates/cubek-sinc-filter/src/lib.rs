//! `cubek-sinc-filter` — windowed-sinc FIR low-pass / high-pass filter.
//!
//! Implements the Julius `LowPassFilters` algorithm. High-pass is
//! `x - lowpass(x)`, matching `torch_audiomentations.HighPassFilter`.
//!
//! Downstream consumers (`burn-audiomentations::LowPassFilter` /
//! `HighPassFilter`) sample per-example cutoffs uniformly in mel-space,
//! quantize into bucket indices, and dispatch
//! [`LowPassFilterBank::apply_per_row`] once per batch.
//!
//! ## Algorithm in 30 seconds
//!
//! Given a set of normalized cutoffs `c_0, ..., c_{F-1}` (each `f_c / f_s`,
//! in `[0, 0.5]`), derive a shared `half_size = floor(zeros / min_c / 2)`
//! and build one Hann-windowed sinc FIR per cutoff, DC-normalized to sum 1.
//! At apply time, each output sample is a strided dot product between the
//! selected row of the weight table and a replicate-padded window of the
//! input. Low-pass emits the dot product directly; high-pass emits
//! `center_sample - dot_product` so a zero cutoff degrades to identity
//! rather than all-stop.
//!
//! ## What's in this crate
//!
//! * [`LowPassFilterBank`] — bank of FIR filters over `cubecl::Runtime`.
//! * [`FilterMode`] — select low-pass or high-pass at apply time.
//! * [`half_size_from`] — compute the Julius-convention filter half-length
//!   without building a bank. Useful for sizing work buffers before bank
//!   construction.
//!
//! ## Quick usage
//!
//! ```ignore
//! use cubecl::prelude::*;
//! use cubecl::std::tensor::TensorHandle;
//! use cubek_sinc_filter::{FilterMode, LowPassFilterBank};
//!
//! let client = <R as Runtime>::client(&Default::default());
//! let dtype = f32::as_type_native_unchecked().storage_type();
//!
//! // 8 buckets spanning 150 Hz – 7500 Hz at 32 kHz.
//! let cutoffs = [150.0, 300.0, 600.0, 1200.0, 2400.0, 4000.0, 6000.0, 7500.0]
//!     .map(|f| f / 32_000.0);
//! let bank: LowPassFilterBank<R> = LowPassFilterBank::new(
//!     client.clone(),
//!     &cutoffs,
//!     8,       // zeros — Julius default
//!     dtype,
//! );
//!
//! let y = bank.apply_single(signal, 3, FilterMode::LowPass);
//! // or: bank.apply_per_row(signal, indices, FilterMode::HighPass)
//! ```
//!
//! ## Scope and known non-features
//!
//! * Only f32. Other dtypes are not wired into the kernel.
//! * `stride == 1`, `pad == True`. The Julius decimating stride is not
//!   exposed; the augmentation layer doesn't need it.
//! * Direct convolution for short filters; FFT-conv for long filters
//!   (crossover at `half_size > 32`).

mod fft_conv;
mod filter_bank;
mod filters;

pub use filter_bank::{FilterMode, LowPassFilterBank};
pub use filters::half_size_from;

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
