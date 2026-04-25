//! `cubek-phasevocoder` — phase-locked time-stretch for STFT spectrograms.
//!
//! Operates on real/imaginary tensor pairs as produced by
//! `cubek-fft::rfft` / `cubek-stft::stft`.
//!
//! ## Input / output layout
//!
//! Input :  `(batch, n_freq, n_frames)` real and imaginary tensors.
//! Output:  `(batch, n_freq, ceil(n_frames / rate))` real and imaginary.
//!
//! Layout is `n_freq` before `n_frames`.
//! `cubek_stft::stft` returns `(batch, n_frames, n_freq)` instead; callers
//! coming from the STFT path must permute the last two axes before handing
//! spectra in.
//!
//! ## Algorithm
//!
//! For each output frame `t`:
//!
//! * `idx0 = floor(t * rate)`, `idx1 = idx0 + 1`, `alpha = t*rate - idx0`.
//! * Input frames past `n_frames` are treated as zero.
//! * `mag[t] = alpha * |input[idx1]| + (1 - alpha) * |input[idx0]|`.
//! * Phase accumulator starts at `angle(input[0])` and advances by
//!   `wrap(angle(input[idx1]) - angle(input[idx0]) - phase_advance) + phase_advance`
//!   after each emitted frame (phase unwrap is
//!   `x - 2π · round(x / 2π)`).
//! * Output frame at `t` is `polar(mag[t], phase_acc[t])`.
//!
//! Each `(batch, freq)` pair is one kernel thread that walks sequentially
//! along the output frame axis. The `phase_acc` update has a data dependency
//! across frames, so we accept sequential inner iteration in exchange for
//! fully parallel outer iteration over `batch * n_freq` threads — plenty of
//! parallelism for our n_freq (≥ 257 at n_fft ≥ 512) and typical batch sizes.
//!
//! ## Rate = 1.0 fast path
//!
//! When `rate == 1.0` the input is returned unchanged. The algorithm above
//! is only exact at `rate=1.0` when the spectrogram's true frame-to-frame
//! phase advance matches `phase_advance` (which is never the case for
//! arbitrary data), so the short-circuit avoids spurious distortion.
//!
//! ## Quick usage
//!
//! ```ignore
//! use cubecl::prelude::*;
//! use cubecl::std::tensor::TensorHandle;
//! use cubek_phasevocoder::phase_vocoder;
//!
//! let (re_stretched, im_stretched) = phase_vocoder(
//!     spec_re,          // (B, n_freq, n_frames)
//!     spec_im,
//!     phase_advance,    // (n_freq,) — caller builds as linspace(0, pi*hop, n_freq)
//!     1.3,              // rate: speed-up factor
//!     f32::as_type_native_unchecked().storage_type(),
//! );
//! ```

mod phase_vocoder;

pub use phase_vocoder::phase_vocoder;

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
