//! FFT-convolution fast path for long filters.
//!
//! Simplified variant of Julius's `fft_conv1d`: a single filter per batch row
//! (not a `[D, C, K]` weight tensor), `stride=1`, `pad=True`-equivalent
//! replicate padding, and symmetric filters (so linear cross-correlation and
//! convolution coincide — we don't flip).
//!
//! Approach: single-shot FFT per batch row. Pick `n_fft = next_pow2(time +
//! filter_len − 1)`, replicate-pad the input into an `(batch, n_fft)` buffer
//! with zero-padding on the tail, `rfft`, pointwise-multiply by the selected
//! per-row filter spectrum, `irfft`, and slice out the `time`-length valid
//! window. For our training-loop shape `(128, 32000)` at the 40 Hz / 32 kHz
//! cutoff (`filter_len=6057`) this is `n_fft=65536` — 2 forward + 1 inverse
//! batched FFT per call. We tried Julius-style overlap-add with
//! `block_size=5*kernel`: at this shape and K it fans out to 2 FFTs of 32768
//! per row, ~same flops, more bookkeeping, no measured win. Single-shot is
//! simpler and equally fast on the wgpu-Metal backend.
//!
//! The filter spectra are built lazily at first `apply_*` call (we need
//! `time` to size `n_fft`, and the bank constructor doesn't see `time`). We
//! cache the result keyed on `n_fft`; if a caller later passes a different
//! `time`, we rebuild. The filter spectra are a small allocation
//! (`n_cutoffs * (n_fft/2 + 1)` f32 × 2 for re/im), typically ≤ 10 MB, so
//! recomputing on a shape change is cheap.
//!
//! Threshold: we route through this module when `half_size > FFT_THRESHOLD`.
//! 32 is Julius's crossover and matches what our microbench shows — below
//! that, the direct conv kernel wins on the wgpu-Metal backend because the
//! OLA fixed cost (pad + 2 FFTs + multiply + slice) exceeds the O(time · K)
//! direct convolution for short K.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::elemwise_launch_dims;

/// Switch to FFT-conv when `half_size > FFT_THRESHOLD`. Matches Julius's
/// `half_size > 32` crossover.
/// Measured on the training-loop shape `(128, 32000)` f32 at
/// `half_size=63` (filter_len=127), FFT-conv and direct conv cost
/// ~10 ms each — within run-to-run noise; at `half_size=3028`
/// (filter_len=6057) FFT-conv is ~15 ms vs ~257 ms direct (≈17×). So
/// 32 works: nothing is clearly slower on FFT, the 40 Hz highpass is
/// hugely faster.
pub(crate) const FFT_THRESHOLD: u32 = 32;

/// Smallest power-of-two `n_fft` that fits the full linear convolution
/// of the *replicate-padded* signal (length `time + 2*half_size`) with the
/// filter (length `filter_len = 2*half_size + 1`). The FFT would alias if
/// `n_fft < (time + 2*half_size) + filter_len - 1 = time + 4*half_size`.
/// `rfft` requires a power-of-2 length.
pub(crate) fn fft_n_for(time: usize, filter_len: usize) -> usize {
    let half_size = (filter_len - 1) / 2;
    (time + 2 * half_size + filter_len - 1)
        .next_power_of_two()
        .max(2)
}

/// Cached FFT-conv context: per-bucket filter spectra padded to `n_fft`.
///
/// Cheap to clone — the tensor handles are Arc-backed so cloning does not
/// duplicate GPU memory. The struct is re-created only when `time` (hence
/// `n_fft`) changes.
#[derive(Clone)]
pub(crate) struct FftConvSpectra<R: Runtime> {
    pub n_fft: usize,
    pub n_freq: usize,
    /// Per-bucket filter spectrum real part. Shape `(n_cutoffs, n_freq)`.
    pub filter_re: TensorHandle<R>,
    /// Per-bucket filter spectrum imaginary part. Shape `(n_cutoffs, n_freq)`.
    pub filter_im: TensorHandle<R>,
}

impl<R: Runtime> FftConvSpectra<R> {
    /// Zero-pad each filter row to `n_fft` and FFT it on-device via
    /// `cubek-fft`.
    pub fn build(
        client: &ComputeClient<R>,
        weights_host: &[f32],
        n_cutoffs: usize,
        filter_len: usize,
        n_fft: usize,
        dtype: StorageType,
    ) -> Self {
        assert!(n_fft.is_power_of_two(), "n_fft must be power of 2");
        assert!(
            filter_len <= n_fft,
            "filter_len {filter_len} exceeds n_fft {n_fft}",
        );
        assert_eq!(weights_host.len(), n_cutoffs * filter_len);

        // Host-side zero-pad the `(n_cutoffs, filter_len)` bank to
        // `(n_cutoffs, n_fft)` then upload once. Done on host to avoid an
        // extra launch + allocation on device for the cold-cache path.
        let mut padded = vec![0.0f32; n_cutoffs * n_fft];
        for row in 0..n_cutoffs {
            let src = &weights_host[row * filter_len..(row + 1) * filter_len];
            let dst = &mut padded[row * n_fft..row * n_fft + filter_len];
            dst.copy_from_slice(src);
        }

        let handle = client.create_from_slice(f32::as_bytes(&padded));
        let filters_padded = TensorHandle::<R>::new_contiguous(
            vec![n_cutoffs, n_fft],
            handle,
            dtype,
        );

        let (filter_re, filter_im) = cubek_fft::rfft::<R>(filters_padded, 1, dtype);
        let n_freq = n_fft / 2 + 1;

        Self {
            n_fft,
            n_freq,
            filter_re,
            filter_im,
        }
    }
}

/// Replicate-pad the `(batch, time)` signal into `(batch, n_fft)`:
///
/// * `[0, half_size)`                       → `signal[b, 0]`        (left replicate)
/// * `[half_size, half_size+time)`          → `signal[b, i-half]`   (body)
/// * `[half_size+time, 2*half_size+time)`   → `signal[b, time-1]`   (right replicate)
/// * `[2*half_size+time, n_fft)`            → `0`                   (zero tail)
///
/// Matches `F.pad(signal, (half_size, half_size), mode='replicate')` followed
/// by zero-padding out to `n_fft`.
pub(crate) fn launch_pad_replicate<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorHandle<R>,
    padded: TensorHandle<R>,
    half_size: u32,
    n_fft: u32,
) {
    let batch = padded.shape()[0];
    let n_fft_usize = padded.shape()[1];
    let num_elems = batch * n_fft_usize;
    let (cube_count, cube_dim) = elemwise_launch_dims(client, num_elems, 256);

    pad_replicate_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        signal.binding().into_tensor_arg(),
        padded.binding().into_tensor_arg(),
        half_size,
        n_fft,
    );
}

#[cube(launch)]
fn pad_replicate_kernel<F: Float>(
    signal: &Tensor<F>,
    padded: &mut Tensor<F>,
    half_size: u32,
    n_fft: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= padded.len() {
        terminate!();
    }

    let n_fft_u = n_fft as usize;
    let b = pos / n_fft_u;
    let i = pos - b * n_fft_u;
    let half = half_size as usize;
    let time = signal.shape(1);
    let sig_base = b * time;
    let last = time - 1;

    let mut value = F::new(0.0);
    if i < half {
        value = signal[sig_base];
    } else if i < half + time {
        value = signal[sig_base + (i - half)];
    } else if i < half + time + half {
        value = signal[sig_base + last];
    }
    padded[pos] = value;
}

/// Pointwise multiply `(batch, n_freq)` spectrum by a per-row selected
/// filter spectrum from `(n_cutoffs, n_freq)`. Output is complex
/// `(batch, n_freq)`.
pub(crate) fn launch_pointwise_multiply_per_row<R: Runtime>(
    client: &ComputeClient<R>,
    sig_re: TensorHandle<R>,
    sig_im: TensorHandle<R>,
    filter_re: TensorHandle<R>,
    filter_im: TensorHandle<R>,
    indices: TensorHandle<R>,
    out_re: TensorHandle<R>,
    out_im: TensorHandle<R>,
) {
    let batch = out_re.shape()[0];
    let n_freq = out_re.shape()[1];
    let num_elems = batch * n_freq;
    let (cube_count, cube_dim) = elemwise_launch_dims(client, num_elems, 256);

    pointwise_multiply_per_row_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        sig_re.binding().into_tensor_arg(),
        sig_im.binding().into_tensor_arg(),
        filter_re.binding().into_tensor_arg(),
        filter_im.binding().into_tensor_arg(),
        indices.binding().into_tensor_arg(),
        out_re.binding().into_tensor_arg(),
        out_im.binding().into_tensor_arg(),
        n_freq as u32,
    );
}

#[cube(launch)]
fn pointwise_multiply_per_row_kernel<F: Float>(
    sig_re: &Tensor<F>,
    sig_im: &Tensor<F>,
    filter_re: &Tensor<F>,
    filter_im: &Tensor<F>,
    indices: &Tensor<u32>,
    out_re: &mut Tensor<F>,
    out_im: &mut Tensor<F>,
    n_freq: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= out_re.len() {
        terminate!();
    }

    let n_freq_u = n_freq as usize;
    let b = pos / n_freq_u;
    let k = pos - b * n_freq_u;
    let cutoff_idx = indices[b] as usize;
    let filter_base = cutoff_idx * n_freq_u;

    let a_re = sig_re[pos];
    let a_im = sig_im[pos];
    let b_re = filter_re[filter_base + k];
    let b_im = filter_im[filter_base + k];

    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
    out_re[pos] = a_re * b_re - a_im * b_im;
    out_im[pos] = a_re * b_im + a_im * b_re;
}

/// Pointwise multiply with a single filter index (the `apply_single`
/// variant). Kept separate so we don't allocate a tiny (batch,) indices
/// tensor just to encode a constant.
pub(crate) fn launch_pointwise_multiply_single<R: Runtime>(
    client: &ComputeClient<R>,
    sig_re: TensorHandle<R>,
    sig_im: TensorHandle<R>,
    filter_re: TensorHandle<R>,
    filter_im: TensorHandle<R>,
    cutoff_idx: u32,
    out_re: TensorHandle<R>,
    out_im: TensorHandle<R>,
) {
    let batch = out_re.shape()[0];
    let n_freq = out_re.shape()[1];
    let num_elems = batch * n_freq;
    let (cube_count, cube_dim) = elemwise_launch_dims(client, num_elems, 256);

    pointwise_multiply_single_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        sig_re.binding().into_tensor_arg(),
        sig_im.binding().into_tensor_arg(),
        filter_re.binding().into_tensor_arg(),
        filter_im.binding().into_tensor_arg(),
        out_re.binding().into_tensor_arg(),
        out_im.binding().into_tensor_arg(),
        cutoff_idx,
        n_freq as u32,
    );
}

#[cube(launch)]
fn pointwise_multiply_single_kernel<F: Float>(
    sig_re: &Tensor<F>,
    sig_im: &Tensor<F>,
    filter_re: &Tensor<F>,
    filter_im: &Tensor<F>,
    out_re: &mut Tensor<F>,
    out_im: &mut Tensor<F>,
    cutoff_idx: u32,
    n_freq: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= out_re.len() {
        terminate!();
    }

    let n_freq_u = n_freq as usize;
    let b = pos / n_freq_u;
    let k = pos - b * n_freq_u;
    let filter_base = (cutoff_idx as usize) * n_freq_u;
    let _ = b;

    let a_re = sig_re[pos];
    let a_im = sig_im[pos];
    let b_re = filter_re[filter_base + k];
    let b_im = filter_im[filter_base + k];

    out_re[pos] = a_re * b_re - a_im * b_im;
    out_im[pos] = a_re * b_im + a_im * b_re;
}

/// Slice the `time`-length valid window out of the `(batch, n_fft)` full
/// linear convolution and write to `output`, optionally subtracting from
/// the original signal for high-pass mode.
///
/// For a signal of length `time` replicate-padded by `half_size` on each
/// side and convolved with a symmetric filter of length `filter_len =
/// 2*half_size + 1`, the valid output of length `time` starts at offset
/// `filter_len - 1 = 2*half_size` in the full linear convolution.
pub(crate) fn launch_extract_valid<R: Runtime>(
    client: &ComputeClient<R>,
    full_conv: TensorHandle<R>,
    signal: TensorHandle<R>,
    output: TensorHandle<R>,
    offset: u32,
    mode: u32,
) {
    let batch = output.shape()[0];
    let time = output.shape()[1];
    let num_elems = batch * time;
    let (cube_count, cube_dim) = elemwise_launch_dims(client, num_elems, 256);

    extract_valid_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        full_conv.binding().into_tensor_arg(),
        signal.binding().into_tensor_arg(),
        output.binding().into_tensor_arg(),
        offset,
        mode,
    );
}

#[cube(launch)]
fn extract_valid_kernel<F: Float>(
    full_conv: &Tensor<F>,
    signal: &Tensor<F>,
    output: &mut Tensor<F>,
    offset: u32,
    mode: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let time = signal.shape(1);
    let n_fft = full_conv.shape(1);
    let b = pos / time;
    let tt = pos - b * time;
    let full_idx = b * n_fft + (offset as usize) + tt;

    let y = full_conv[full_idx];
    let mut out_val = y;
    if mode != 0 {
        out_val = signal[pos] - y;
    }
    output[pos] = out_val;
}
