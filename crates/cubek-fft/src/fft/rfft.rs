//! Real-valued FFT — intra-cube-parallel radix-2 Cooley-Tukey.
//!
//! One cube per window, `threads_per_cube` units doing one butterfly per
//! unit per stage (or several if `threads_per_cube < n_fft / 2`).
//!
//! Layout choices:
//! * A full N-point complex FFT is run on the real input (imaginary part
//!   zero). The first `n_fft / 2 + 1` bins are written out as the Hermitian
//!   half-spectrum; the redundant mirrored bins are discarded.
//! * Dispatch is spread across X/Y/Z via
//!   [`cubecl::calculate_cube_count_elemwise`], so batched shapes above
//!   wgpu's 65 535 X-axis cap work without per-ratio fan-out on the caller.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::fft::{
    FftMode,
    fft_parallel::{bit_reverse, fft_butterfly_parallel, window_batch_offset},
    rfft_large::rfft_large_launch,
};

/// Cap for the per-cube unit count. 256 keeps us within wgpu's default
/// `max_compute_invocations_per_workgroup` while giving 8 Apple-Silicon
/// simdgroups per cube. The actual launch dim is `min(n_fft / 2, 256)`.
const MAX_UNITS_PER_CUBE: usize = 256;

/// Size at which the single-pass shared-memory kernel stops fitting in
/// threadgroup memory on Apple Silicon (two `f32[n_fft]` buffers must fit
/// in ~32 KB). Above this we dispatch to the packed-real + complex-FFT
/// path in `rfft_large`.
const SHARED_MEM_CAP: usize = 4096;

/// Real-valued Fast Fourier Transform.
///
/// Allocates the half-spectrum output tensors then dispatches the kernel.
pub fn rfft<R: Runtime>(
    signal: TensorHandle<R>,
    dim: usize,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    assert!(
        dim < signal.shape().len(),
        "dim must be between 0 and {}",
        signal.shape().len()
    );
    assert!(
        signal.shape()[dim].is_power_of_two(),
        "RFFT requires power-of-2 length"
    );
    let client = <R as Runtime>::client(&Default::default());

    let mut spectrum_shape = signal.shape().clone();
    spectrum_shape[dim] = signal.shape()[dim] / 2 + 1;

    let spectrum_re = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    let spectrum_im = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    rfft_launch::<R>(
        &client,
        signal.binding(),
        spectrum_re.clone().binding(),
        spectrum_im.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    (spectrum_re, spectrum_im)
}

/// Launches the RFFT kernel.
pub fn rfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorBinding<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    dim: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let n_fft = signal.shape[dim];
    assert!(n_fft.is_power_of_two(), "RFFT requires power-of-2 length");
    assert!(n_fft >= 2, "RFFT requires n_fft >= 2");

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
        return rfft_large_launch::<R>(client, signal, spectrum_re, spectrum_im, dim, dtype);
    }

    let log2_n = n_fft.trailing_zeros() as usize;
    let threads_per_cube = (n_fft / 2).min(MAX_UNITS_PER_CUBE).max(1);

    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    // One cube per window, spread across X/Y/Z so we don't hit wgpu's
    // 65 535 X-axis dispatch cap at training-loop batch sizes.
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, count, CubeDim::new_single());

    rfft_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        signal.into_tensor_arg(),
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        count as u32,
        n_fft,
        log2_n,
        threads_per_cube,
        dim,
    );
    Ok(())
}

#[cube(launch)]
fn rfft_kernel<F: Float>(
    signal: &Tensor<F>,
    spectrum_re: &mut Tensor<F>,
    spectrum_im: &mut Tensor<F>,
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

    let sig_off = window_batch_offset(signal, window_index, dim);
    let re_off = window_batch_offset(spectrum_re, window_index, dim);
    let im_off = window_batch_offset(spectrum_im, window_index, dim);
    let sig_stride = signal.stride(dim);
    let re_stride = spectrum_re.stride(dim);
    let im_stride = spectrum_im.stride(dim);

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    // Load signal at bit-reversed shared-memory indices, so subsequent
    // butterfly stages can operate directly without a separate permute.
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        shared_re[j] = signal[sig_off + i * sig_stride];
        shared_im[j] = F::new(0.0);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        FftMode::Forward,
    );

    // Write the half-complex spectrum (n_fft/2 + 1 bins).
    let n_freq = comptime![n_fft / 2 + 1];
    let mut k = UNIT_POS as usize;
    while k < n_freq {
        spectrum_re[re_off + k * re_stride] = shared_re[k];
        spectrum_im[im_off + k * im_stride] = shared_im[k];
        k += threads_per_cube;
    }
}
