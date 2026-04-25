//! `cubek-stft` — short-time Fourier transform kernels on top of
//! `cubek-fft`.
//!
//! Runtime-generic over `cubecl::Runtime`, mirroring `cubek-fft`'s
//! public API shape. Expected consumers:
//!
//! * `burn-audiomentations::PitchShift` — STFT → phase-vocoder → iSTFT.
//! * The noise-detect mel front-end — STFT → `|·|²` → mel filterbank.
//!
//! ## Quick usage
//!
//! ```ignore
//! use cubecl::prelude::*;
//! use cubecl::std::tensor::TensorHandle;
//! use cubek_stft::{stft, istft, window::hann_window_periodic};
//!
//! let dtype = f32::as_type_native_unchecked().storage_type();
//! let signal = TensorHandle::<R>::new_contiguous(vec![1, 8192], signal_handle, dtype);
//! let w = TensorHandle::<R>::new_contiguous(vec![1024], window_handle, dtype);
//!
//! let (re, im) = stft(signal, w.clone(), 1024, 256, dtype);
//! let reconstructed = istft(re, im, w, 256, dtype);
//! ```

mod istft;
mod stft;
pub mod window;

pub use istft::istft;
pub use stft::stft;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

/// `(cube_count, cube_dim)` for a one-thread-per-element launch.
///
/// Wraps [`cubecl::calculate_cube_count_elemwise`] so the cube-count
/// selection uses each backend's real `max_cube_count` (via
/// `cubecl-runtime`'s `cube_count_spread`) instead of a hand-rolled
/// 65535-wgpu-worst-case tiling.
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

/// Upper bound on `n_fft` for the wgpu-Metal path. `cubek-fft` stages
/// the signal into per-cube shared memory; above this bound it silently
/// produces wrong output. The STFT launcher refuses values past this so
/// callers get a clean panic instead of garbage spectrograms.
pub const MAX_N_FFT: usize = 4096;

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn max_n_fft_is_power_of_two() {
        assert!(MAX_N_FFT.is_power_of_two());
    }
}
