//! FIR filter bank launcher + CubeCL kernels.
//!
//! Public surface is [`LowPassFilterBank<R>`] — construct once from a set of
//! normalized cutoffs, reuse across batches. Uploading the weight table is
//! the only allocation that runs at construction; every `apply_*` launches
//! one kernel with the pre-uploaded table.
//!
//! Two launchers:
//!
//! * [`LowPassFilterBank::apply_single`] — apply one cutoff (selected by
//!   index into the bank) to every row of `(batch, time)`. Used by the
//!   swept-sine frequency-response tests and by augmentation callers that
//!   have only one cutoff per batch.
//! * [`LowPassFilterBank::apply_per_row`] — apply a per-row cutoff chosen
//!   by an indices tensor of shape `(batch,)`. This is the augmentation
//!   layer's production path: sample cutoffs uniformly in mel-space, quantize
//!   into bucket indices, launch once.
//!
//! Each variant has a `mode: FilterMode` argument selecting low-pass or
//! high-pass. High-pass is computed as `x - lowpass(x)` inside the kernel to
//! avoid a second launch and a temporary buffer (matches
//! `torch_audiomentations.HighPassFilter` semantics).
//!
//! Replicate padding is folded into the inner loop's clamp, mirroring the
//! approach used by `cubek-resample`: we never materialize the padded
//! signal.

use std::sync::Mutex;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::elemwise_launch_dims;
use crate::fft_conv::{
    FFT_THRESHOLD, FftConvSpectra, fft_n_for, launch_extract_valid, launch_pad_replicate,
    launch_pointwise_multiply_per_row, launch_pointwise_multiply_single,
};
use crate::filters::FilterBank;

/// Low-pass or high-pass mode for a single apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// `y = lowpass(x)`.
    LowPass,
    /// `y = x - lowpass(x)` — matches
    /// `torch_audiomentations.HighPassFilter`.
    HighPass,
}

/// Bank of windowed-sinc FIR low-pass filters sharing one filter length.
///
/// Constructed once per `(cutoff_set, zeros)` pair. The weight table is
/// built on the host and uploaded to the runtime at construction; `apply_*`
/// launches a single compute kernel per batch.
///
/// Operates on rank-2 `(batch, time)` f32 tensors. Higher-rank callers
/// flatten before calling.
pub struct LowPassFilterBank<R: Runtime> {
    client: ComputeClient<R>,
    dtype: StorageType,

    n_cutoffs: u32,
    half_size: u32,
    filter_len: u32,

    /// Uploaded `(n_cutoffs, filter_len)` weight table.
    weights: TensorHandle<R>,

    /// Host-side copy of the weight table, retained so the FFT path can
    /// lazily zero-pad and FFT the filters when `time` becomes known. Only
    /// populated when `half_size > FFT_THRESHOLD`; otherwise empty.
    weights_host: Vec<f32>,

    /// Lazily-built cache of filter spectra keyed on the signal's `time`
    /// dimension (`n_fft = next_pow2(time + filter_len − 1)`). A single
    /// slot: if a new `time` comes in, we rebuild. In the training loop
    /// `time` is constant so this is built exactly once.
    fft_spectra: Mutex<Option<FftConvSpectra<R>>>,
}

impl<R: Runtime> LowPassFilterBank<R> {
    /// Build the filter bank on the host and upload it to `client`.
    ///
    /// * `cutoffs` — normalized cutoffs `f_c / f_s ∈ [0, 0.5]`. May include
    ///   0, which produces an all-zero filter (low-pass returns 0, high-pass
    ///   returns the input). The filter length is derived from the smallest
    ///   **positive** cutoff, so every filter in the bank is the same length
    ///   — identical to `julius.LowPassFilters` semantics.
    /// * `zeros` — sinc truncation in zero-crossings. Julius default: 8.
    ///
    /// Panics if `cutoffs` is empty, if any cutoff is outside `[0, 0.5]`,
    /// if `zeros == 0`, or if the derived `half_size` is 0 (cutoffs too
    /// large for `zeros` — Julius's `int(...)` truncation hitting 0).
    pub fn new(
        client: ComputeClient<R>,
        cutoffs: &[f32],
        zeros: u32,
        dtype: StorageType,
    ) -> Self {
        let bank = FilterBank::new(cutoffs, zeros);

        let handle = client.create_from_slice(f32::as_bytes(&bank.weights));
        let weights = TensorHandle::<R>::new_contiguous(
            vec![bank.n_cutoffs as usize, bank.filter_len as usize],
            handle,
            dtype,
        );

        // Retain a host copy of the weight table only when the FFT path is
        // potentially in reach — otherwise we don't need it and would just
        // pin ~`n_cutoffs * filter_len` f32 of RAM for nothing.
        let weights_host = if bank.half_size > FFT_THRESHOLD {
            bank.weights.clone()
        } else {
            Vec::new()
        };

        Self {
            client,
            dtype,
            n_cutoffs: bank.n_cutoffs,
            half_size: bank.half_size,
            filter_len: bank.filter_len,
            weights,
            weights_host,
            fft_spectra: Mutex::new(None),
        }
    }

    /// Whether this bank routes through the FFT-conv path at apply time.
    /// Exposed for bench / regression scripts; the public `apply_*` methods
    /// pick automatically.
    #[doc(hidden)]
    pub fn uses_fft_path(&self) -> bool {
        self.half_size > FFT_THRESHOLD
    }

    /// Get or build the FFT spectra for a given signal `time`. Caches on
    /// the derived `n_fft`; rebuilds if a new `time` implies a different
    /// `n_fft`.
    fn fft_spectra_for(&self, time: usize) -> FftConvSpectra<R> {
        let n_fft = fft_n_for(time, self.filter_len as usize);
        let mut guard = self.fft_spectra.lock().expect("fft_spectra mutex poisoned");
        if let Some(spectra) = guard.as_ref() {
            if spectra.n_fft == n_fft {
                return spectra.clone();
            }
        }
        let spectra = FftConvSpectra::build(
            &self.client,
            &self.weights_host,
            self.n_cutoffs as usize,
            self.filter_len as usize,
            n_fft,
            self.dtype,
        );
        *guard = Some(spectra.clone());
        spectra
    }

    /// Number of cutoffs (buckets) in the bank.
    pub fn n_cutoffs(&self) -> u32 {
        self.n_cutoffs
    }

    /// Half of the shared filter length; taps cover `[-half_size, +half_size]`.
    pub fn half_size(&self) -> u32 {
        self.half_size
    }

    /// Shared filter length in taps (`2 * half_size + 1`).
    pub fn filter_len(&self) -> u32 {
        self.filter_len
    }

    /// Apply a single cutoff to every row of a `(batch, time)` signal.
    ///
    /// `cutoff_idx` selects the row of the precomputed weight table. Output
    /// has the same shape as the input (replicate-padded, stride 1 — matching
    /// `julius.LowPassFilter(pad=True, stride=1)`).
    pub fn apply_single(
        &self,
        signal: TensorHandle<R>,
        cutoff_idx: u32,
        mode: FilterMode,
    ) -> TensorHandle<R> {
        assert_eq!(
            signal.shape().len(),
            2,
            "apply_single expects (batch, time), got rank {}",
            signal.shape().len(),
        );
        assert!(
            cutoff_idx < self.n_cutoffs,
            "cutoff_idx {cutoff_idx} out of range (n_cutoffs={})",
            self.n_cutoffs,
        );
        let batch = signal.shape()[0];
        let time = signal.shape()[1];
        assert!(time > 0, "signal time must be > 0");

        if self.half_size > FFT_THRESHOLD {
            return self.apply_fft_single(signal, cutoff_idx, mode);
        }

        let output = self.alloc_output(batch, time);
        let num_elems = batch * time;
        let (cube_count, cube_dim) = elemwise_launch_dims(&self.client, num_elems, 256);

        sinc_filter_single_kernel::launch::<f32, R>(
            &self.client,
            cube_count,
            cube_dim,
            signal.binding().into_tensor_arg(),
            self.weights.clone().binding().into_tensor_arg(),
            output.clone().binding().into_tensor_arg(),
            cutoff_idx,
            self.half_size,
            self.filter_len,
            mode_flag(mode),
        );

        output
    }

    /// Apply a per-row cutoff to a `(batch, time)` signal.
    ///
    /// `indices` is a `(batch,)` `u32` tensor; `indices[b]` selects the
    /// filter bank row applied to `signal[b]`. All entries must be `< n_cutoffs`.
    /// The kernel does not bounds-check indices; pass garbage at your peril.
    ///
    /// This is the path `burn-audiomentations::LowPassFilter` /
    /// `HighPassFilter` will use in production.
    pub fn apply_per_row(
        &self,
        signal: TensorHandle<R>,
        indices: TensorHandle<R>,
        mode: FilterMode,
    ) -> TensorHandle<R> {
        assert_eq!(
            signal.shape().len(),
            2,
            "apply_per_row expects (batch, time), got rank {}",
            signal.shape().len(),
        );
        assert_eq!(
            indices.shape().len(),
            1,
            "indices must be rank-1",
        );
        let batch = signal.shape()[0];
        let time = signal.shape()[1];
        assert!(time > 0, "signal time must be > 0");
        assert_eq!(
            indices.shape()[0], batch,
            "indices length {} does not match batch {}",
            indices.shape()[0], batch,
        );

        if self.half_size > FFT_THRESHOLD {
            return self.apply_fft_per_row(signal, indices, mode);
        }

        let output = self.alloc_output(batch, time);
        let num_elems = batch * time;
        let (cube_count, cube_dim) = elemwise_launch_dims(&self.client, num_elems, 256);

        sinc_filter_per_row_kernel::launch::<f32, R>(
            &self.client,
            cube_count,
            cube_dim,
            signal.binding().into_tensor_arg(),
            self.weights.clone().binding().into_tensor_arg(),
            indices.binding().into_tensor_arg(),
            output.clone().binding().into_tensor_arg(),
            self.half_size,
            self.filter_len,
            mode_flag(mode),
        );

        output
    }

    /// FFT-conv variant of `apply_single`. Signal is replicate-padded then
    /// zero-padded to `n_fft`; input and per-bucket filter spectra are
    /// multiplied pointwise; `irfft` gives the full linear convolution; the
    /// `time`-length valid window is extracted (and optionally subtracted
    /// from the original for high-pass mode).
    fn apply_fft_single(
        &self,
        signal: TensorHandle<R>,
        cutoff_idx: u32,
        mode: FilterMode,
    ) -> TensorHandle<R> {
        let batch = signal.shape()[0];
        let time = signal.shape()[1];
        let spectra = self.fft_spectra_for(time);
        let n_fft = spectra.n_fft;

        let padded_input = self.alloc_tensor(vec![batch, n_fft], batch * n_fft);
        launch_pad_replicate(
            &self.client,
            signal.clone(),
            padded_input.clone(),
            self.half_size,
            n_fft as u32,
        );

        let (sig_re, sig_im) = cubek_fft::rfft::<R>(padded_input, 1, self.dtype);

        let n_freq = spectra.n_freq;
        let mul_re = self.alloc_tensor(vec![batch, n_freq], batch * n_freq);
        let mul_im = self.alloc_tensor(vec![batch, n_freq], batch * n_freq);
        launch_pointwise_multiply_single(
            &self.client,
            sig_re,
            sig_im,
            spectra.filter_re.clone(),
            spectra.filter_im.clone(),
            cutoff_idx,
            mul_re.clone(),
            mul_im.clone(),
        );

        let full_conv = cubek_fft::irfft::<R>(mul_re, mul_im, 1, self.dtype);

        let output = self.alloc_output(batch, time);
        let offset = self.filter_len - 1;
        launch_extract_valid(
            &self.client,
            full_conv,
            signal,
            output.clone(),
            offset,
            mode_flag(mode),
        );
        output
    }

    /// FFT-conv variant of `apply_per_row`.
    fn apply_fft_per_row(
        &self,
        signal: TensorHandle<R>,
        indices: TensorHandle<R>,
        mode: FilterMode,
    ) -> TensorHandle<R> {
        let batch = signal.shape()[0];
        let time = signal.shape()[1];
        let spectra = self.fft_spectra_for(time);
        let n_fft = spectra.n_fft;

        let padded_input = self.alloc_tensor(vec![batch, n_fft], batch * n_fft);
        launch_pad_replicate(
            &self.client,
            signal.clone(),
            padded_input.clone(),
            self.half_size,
            n_fft as u32,
        );

        let (sig_re, sig_im) = cubek_fft::rfft::<R>(padded_input, 1, self.dtype);

        let n_freq = spectra.n_freq;
        let mul_re = self.alloc_tensor(vec![batch, n_freq], batch * n_freq);
        let mul_im = self.alloc_tensor(vec![batch, n_freq], batch * n_freq);
        launch_pointwise_multiply_per_row(
            &self.client,
            sig_re,
            sig_im,
            spectra.filter_re.clone(),
            spectra.filter_im.clone(),
            indices,
            mul_re.clone(),
            mul_im.clone(),
        );

        let full_conv = cubek_fft::irfft::<R>(mul_re, mul_im, 1, self.dtype);

        let output = self.alloc_output(batch, time);
        let offset = self.filter_len - 1;
        launch_extract_valid(
            &self.client,
            full_conv,
            signal,
            output.clone(),
            offset,
            mode_flag(mode),
        );
        output
    }

    fn alloc_output(&self, batch: usize, time: usize) -> TensorHandle<R> {
        self.alloc_tensor(vec![batch, time], batch * time)
    }

    fn alloc_tensor(&self, shape: Vec<usize>, num_elems: usize) -> TensorHandle<R> {
        TensorHandle::<R>::new_contiguous(
            shape,
            self.client.empty(num_elems * self.dtype.size()),
            self.dtype,
        )
    }
}

/// `0` = low-pass, `1` = high-pass. Runtime u32 flag — the branch is a
/// single predicate the GPU picks once per thread, same across the cube so
/// it costs nothing in practice.
fn mode_flag(mode: FilterMode) -> u32 {
    match mode {
        FilterMode::LowPass => 0,
        FilterMode::HighPass => 1,
    }
}

/// One thread per output sample, single-cutoff variant.
///
/// For output position `t` (flat index into `(batch, time)`):
///
/// * `b = t / time`, `tt = t % time`.
/// * `acc = Σ_{k=0..filter_len} signal[b, clamp(tt + k - half_size, 0, time-1)] * weights[cutoff_idx, k]`.
/// * If `mode == 0`: `output[b, tt] = acc` (low-pass).
/// * If `mode != 0`: `output[b, tt] = signal[b, tt] - acc` (high-pass).
///
/// The clamp reproduces `F.pad(x, ..., mode='replicate')` without
/// materializing the padded tensor — same approach as `cubek-resample`.
#[cube(launch)]
pub(crate) fn sinc_filter_single_kernel<F: Float>(
    signal: &Tensor<F>,
    weights: &Tensor<F>,
    output: &mut Tensor<F>,
    cutoff_idx: u32,
    half_size: u32,
    filter_len: u32,
    mode: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let time = signal.shape(1);
    let b = pos / time;
    let tt = pos - b * time;

    let half_size_u = half_size as usize;
    let filter_len_u = filter_len as usize;

    let signal_row_base = b * time;
    let weight_row_base = (cutoff_idx as usize) * filter_len_u;
    let last = time - 1;

    let mut acc = F::new(0.0);
    let mut k: usize = 0;
    while k < filter_len_u {
        // Unclamped signal index `p = tt + k - half_size`. Branch to avoid
        // unsigned wraparound — same pattern as the resample kernel.
        let mut idx: usize = 0;
        if tt + k >= half_size_u {
            let unclamped = tt + k - half_size_u;
            if unclamped > last {
                idx = last;
            } else {
                idx = unclamped;
            }
        }
        let x = signal[signal_row_base + idx];
        let w = weights[weight_row_base + k];
        acc += x * w;
        k += 1;
    }

    let center = signal[signal_row_base + tt];
    let mut out = acc;
    if mode != 0 {
        out = center - acc;
    }
    output[pos] = out;
}

/// One thread per output sample, per-row cutoff variant.
///
/// Identical to [`sinc_filter_single_kernel`] except that `cutoff_idx` is
/// read from `indices[b]` per row. The indices tensor is rank-1 `(batch,)`.
#[cube(launch)]
pub(crate) fn sinc_filter_per_row_kernel<F: Float>(
    signal: &Tensor<F>,
    weights: &Tensor<F>,
    indices: &Tensor<u32>,
    output: &mut Tensor<F>,
    half_size: u32,
    filter_len: u32,
    mode: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let time = signal.shape(1);
    let b = pos / time;
    let tt = pos - b * time;

    let half_size_u = half_size as usize;
    let filter_len_u = filter_len as usize;

    let signal_row_base = b * time;
    let cutoff_idx = indices[b];
    let weight_row_base = (cutoff_idx as usize) * filter_len_u;
    let last = time - 1;

    let mut acc = F::new(0.0);
    let mut k: usize = 0;
    while k < filter_len_u {
        let mut idx: usize = 0;
        if tt + k >= half_size_u {
            let unclamped = tt + k - half_size_u;
            if unclamped > last {
                idx = last;
            } else {
                idx = unclamped;
            }
        }
        let x = signal[signal_row_base + idx];
        let w = weights[weight_row_base + k];
        acc += x * w;
        k += 1;
    }

    let center = signal[signal_row_base + tt];
    let mut out = acc;
    if mode != 0 {
        out = center - acc;
    }
    output[pos] = out;
}
