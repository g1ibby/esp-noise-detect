//! Inverse real-valued FFT — intra-cube-parallel radix-2 Cooley-Tukey.
//!
//! Takes a half-complex spectrum `(batch..., n_fft / 2 + 1)` and writes a
//! real-valued signal of length `n_fft`. Each cube handles one window with
//! `threads_per_cube` units running the butterfly stages in lockstep.
//!
//! The Hermitian mirror of the spectrum is filled inside the cube after
//! the half-spectrum is loaded but before the butterfly stages run.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::fft::{
    FftMode,
    fft_parallel::{bit_reverse, fft_butterfly_parallel, window_batch_offset},
    rfft_large::irfft_large_launch,
};

/// See [`MAX_UNITS_PER_CUBE`](super::rfft). Duplicated here so the two
/// kernels can evolve independently.
const MAX_UNITS_PER_CUBE: usize = 256;

/// Mirror of [`super::rfft::SHARED_MEM_CAP`]. Above this we dispatch to the
/// packed-real + complex-FFT inverse path.
const SHARED_MEM_CAP: usize = 4096;

/// Inverse Real-valued Fast Fourier Transform.
pub fn irfft<R: Runtime>(
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
    dim: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    assert!(
        spectrum_re.shape() == spectrum_im.shape(),
        "Spectrum's real and imaginary parts should be the same shape, got {:?} and {:?}",
        spectrum_re.shape(),
        spectrum_im.shape()
    );

    let client = <R as Runtime>::client(&Default::default());

    let mut signal_shape = spectrum_re.shape().clone();
    signal_shape[dim] = (spectrum_re.shape()[dim] - 1) * 2;
    let num_elems = signal_shape.iter().product::<usize>();
    let signal = TensorHandle::new_contiguous(
        signal_shape.clone(),
        client.empty(num_elems * dtype.size()),
        dtype,
    );

    irfft_launch::<R>(
        &client,
        spectrum_re.binding(),
        spectrum_im.binding(),
        signal.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    signal
}

/// Launches the IRFFT kernel.
pub fn irfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let n_fft = signal.shape[dim];
    assert!(n_fft.is_power_of_two(), "IRFFT requires power-of-2 length");
    assert!(n_fft >= 2, "IRFFT requires n_fft >= 2");

    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();
    if count == 0 {
        return Ok(());
    }

    if n_fft > SHARED_MEM_CAP {
        return irfft_large_launch::<R>(client, spectrum_re, spectrum_im, signal, dim, dtype);
    }

    let log2_n = n_fft.trailing_zeros() as usize;
    let threads_per_cube = (n_fft / 2).min(MAX_UNITS_PER_CUBE).max(1);

    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, count, CubeDim::new_single());

    irfft_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        signal.into_tensor_arg(),
        count as u32,
        n_fft,
        log2_n,
        threads_per_cube,
        dim,
    );
    Ok(())
}

#[cube(launch)]
fn irfft_kernel<F: Float>(
    spectrum_re: &Tensor<F>,
    spectrum_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    num_windows: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
) {
    let window_index = CUBE_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let re_off = window_batch_offset(spectrum_re, window_index, dim);
    let im_off = window_batch_offset(spectrum_im, window_index, dim);
    let sig_off = window_batch_offset(signal, window_index, dim);
    let re_stride = spectrum_re.stride(dim);
    let im_stride = spectrum_im.stride(dim);
    let sig_stride = signal.stride(dim);

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    let n_freq = comptime![n_fft / 2 + 1];

    // Load the half-spectrum and fill the Hermitian mirror at
    // bit-reversed destinations in one pass. DC (k=0) and Nyquist
    // (k=n_fft/2) have no conjugate partner so they're loaded directly.
    let mut k = UNIT_POS as usize;
    while k < n_fft {
        let dst = bit_reverse(k, log2_n);
        if k < n_freq {
            shared_re[dst] = spectrum_re[re_off + k * re_stride];
            shared_im[dst] = spectrum_im[im_off + k * im_stride];
        } else {
            let src_bin = n_fft - k;
            shared_re[dst] = spectrum_re[re_off + src_bin * re_stride];
            shared_im[dst] = -spectrum_im[im_off + src_bin * im_stride];
        }
        k += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        FftMode::Inverse,
    );

    // Normalise by n_fft and write the real output. A Hermitian-symmetric
    // input produces an ideally-real output; only the real part is kept.
    let scale = F::new(1.0) / F::cast_from(n_fft);
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        signal[sig_off + i * sig_stride] = shared_re[i] * scale;
        i += threads_per_cube;
    }
}
