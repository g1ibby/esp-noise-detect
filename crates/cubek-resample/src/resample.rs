//! Polyphase FIR resampler launcher + CubeCL kernel.
//!
//! Public surface is [`Resampler<R>`] — construct once with
//! `(old_sr, new_sr, zeros, rolloff)`, reuse across batches. Uploading the
//! kernel bank is the only allocation that runs at construction; every
//! `apply` launches one kernel with the pre-uploaded table.
//!
//! The kernel computes one output sample per thread:
//!
//! ```text
//! t = j * new_sr + i                 // output index within [0, output_len)
//! y[b, t] = Σ_{k=0..kernel_len} read_padded(b, j*old_sr + k) * kernel[i, k]
//! read_padded(b, p) = x[b, clamp(p - width, 0, length - 1)]
//! ```
//!
//! The `read_padded` clamp implements replicate padding without
//! materializing the padded tensor.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::elemwise_launch_dims;
use crate::kernels::KernelBank;

/// Rational-ratio sinc-windowed FIR resampler.
///
/// Constructed once per `(old_sr, new_sr, zeros, rolloff)` tuple. The
/// kernel bank is built on the host and uploaded to the runtime at
/// construction; `apply` only launches a single compute kernel per batch.
///
/// Operates on rank-2 `(batch, time)` f32 tensors. Higher-rank callers
/// flatten before calling.
pub struct Resampler<R: Runtime> {
    client: ComputeClient<R>,
    dtype: StorageType,

    /// GCD-reduced input rate.
    old_sr: u32,
    /// GCD-reduced output rate.
    new_sr: u32,
    /// Half-width of each kernel (zero if identity).
    width: u32,
    /// Kernel length in taps (zero if identity).
    kernel_len: u32,

    /// Uploaded `(new_sr, kernel_len)` kernel table. `None` iff the
    /// resampler is a passthrough.
    kernel_tensor: Option<TensorHandle<R>>,
}

impl<R: Runtime> Resampler<R> {
    /// Build the kernel bank on the host and upload it to `client`.
    ///
    /// * `zeros` — sinc truncation in zero-crossings. Recommended: 24.
    /// * `rolloff` — extra margin factor on the anti-alias cutoff, in
    ///   `(0, 1]`. Recommended: 0.945.
    ///
    /// `old_sr == new_sr` produces a passthrough resampler — `apply` is
    /// defined to return the input handle unchanged in that case.
    pub fn new(
        client: ComputeClient<R>,
        old_sr: u32,
        new_sr: u32,
        zeros: u32,
        rolloff: f32,
        dtype: StorageType,
    ) -> Self {
        let bank = KernelBank::new(old_sr, new_sr, zeros, rolloff);

        let kernel_tensor = if bank.is_identity() {
            None
        } else {
            let handle = client.create_from_slice(f32::as_bytes(&bank.kernels));
            Some(TensorHandle::<R>::new_contiguous(
                vec![bank.new_sr as usize, bank.kernel_len as usize],
                handle,
                dtype,
            ))
        };

        Self {
            client,
            dtype,
            old_sr: bank.old_sr,
            new_sr: bank.new_sr,
            width: bank.width,
            kernel_len: bank.kernel_len,
            kernel_tensor,
        }
    }

    /// GCD-reduced input rate.
    pub fn old_sr(&self) -> u32 {
        self.old_sr
    }

    /// GCD-reduced output rate.
    pub fn new_sr(&self) -> u32 {
        self.new_sr
    }

    /// Default output length for `length` input samples:
    /// `floor(new_sr * length / old_sr)`.
    pub fn default_output_length(&self, length: usize) -> usize {
        if self.old_sr == self.new_sr {
            return length;
        }
        // Use i64 to avoid overflow for long audio — at 32 kHz a single-
        // minute clip already hits `1_920_000 * new_sr` which overflows
        // u32 for new_sr > ~2200.
        ((self.new_sr as i64) * (length as i64) / (self.old_sr as i64)) as usize
    }

    /// Maximum output length for `length` input samples:
    /// `ceil(new_sr * length / old_sr)`.
    pub fn max_output_length(&self, length: usize) -> usize {
        if self.old_sr == self.new_sr {
            return length;
        }
        let num = (self.new_sr as i64) * (length as i64);
        let den = self.old_sr as i64;
        ((num + den - 1) / den) as usize
    }

    /// Resample `signal` of shape `(batch, time)`.
    ///
    /// If `output_length` is `None`, the output is trimmed to
    /// [`default_output_length`] (`floor` mode). Passing
    /// `Some(n)` crops to `n`; `n` must satisfy
    /// `n <= max_output_length(time)`.
    ///
    /// For the identity case (`old_sr == new_sr` after GCD), the input
    /// handle is returned unchanged.
    pub fn apply(
        &self,
        signal: TensorHandle<R>,
        output_length: Option<usize>,
    ) -> TensorHandle<R> {
        assert_eq!(
            signal.shape().len(),
            2,
            "Resampler::apply expects (batch, time), got rank {}",
            signal.shape().len(),
        );
        let batch = signal.shape()[0];
        let length = signal.shape()[1];

        // Identity short-circuit. Keeps upstream code from having to test
        // for `shift == 1.0` cases separately.
        let Some(kernel_tensor) = self.kernel_tensor.as_ref() else {
            assert!(
                output_length.unwrap_or(length) == length,
                "identity resampler cannot produce output_length != time",
            );
            return signal;
        };

        assert!(length > 0, "signal length must be > 0");
        let max_out = self.max_output_length(length);
        let out_len = output_length.unwrap_or_else(|| self.default_output_length(length));
        assert!(
            out_len <= max_out,
            "output_length {out_len} exceeds max {max_out} for time={length}",
        );
        assert!(out_len > 0, "output_length must be > 0");

        let output_shape = vec![batch, out_len];
        let num_elems = batch * out_len;
        let output = TensorHandle::<R>::new_contiguous(
            output_shape,
            self.client.empty(num_elems * self.dtype.size()),
            self.dtype,
        );

        let (cube_count, cube_dim) = elemwise_launch_dims(&self.client, num_elems, 256);

        resample_kernel::launch::<f32, R>(
            &self.client,
            cube_count,
            cube_dim,
            signal.binding().into_tensor_arg(),
            kernel_tensor.clone().binding().into_tensor_arg(),
            output.clone().binding().into_tensor_arg(),
            self.old_sr,
            self.new_sr,
            self.width,
            self.kernel_len,
        );

        output
    }
}

/// One thread per output sample.
///
/// For output position `t` (flat index into `(batch, out_len)`):
///
/// * `b = t / out_len`, `tt = t % out_len`.
/// * `i = tt % new_sr`, `j = tt / new_sr`.
/// * `y[b, tt] = Σ_{k=0..kernel_len} x[b, clamp(j*old_sr + k - width, 0, L-1)] * kernel[i, k]`.
///
/// `width` is `u32` so we avoid signed subtraction in the kernel: WGSL
/// wraps `u32` subtraction, making a naive `base + k - width` unsound
/// when the result would be negative. Instead we branch explicitly.
#[cube(launch)]
pub(crate) fn resample_kernel<F: Float>(
    signal: &Tensor<F>,
    kernels: &Tensor<F>,
    output: &mut Tensor<F>,
    old_sr: u32,
    new_sr: u32,
    width: u32,
    kernel_len: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let out_len = output.shape(1);
    let b = pos / out_len;
    let tt = pos - b * out_len;

    let new_sr_u = new_sr as usize;
    let old_sr_u = old_sr as usize;
    let width_u = width as usize;
    let kernel_len_u = kernel_len as usize;

    let i = tt % new_sr_u;
    let j = tt / new_sr_u;

    let length = signal.shape(1);
    let signal_row_base = b * length;
    let kernel_row_base = i * kernel_len_u;

    // p_base = j*old_sr - width. We keep `p` in signed-like form by
    // adding `width` as an offset that we subtract once per tap. To avoid
    // signed arithmetic in the kernel (WGSL i32 support is spotty on old
    // wgpu backends), we branch on whether the unclamped index is in-
    // range, below zero, or past L-1.
    let base = j * old_sr_u; // == p_base + width in the comment above
    let last = length - 1;

    let mut acc = F::new(0.0);
    let mut k: usize = 0;
    while k < kernel_len_u {
        // Unclamped padded-index p = base + k - width. Split cases so we
        // don't rely on signed subtraction wraparound. Default `idx = 0`
        // is the left-replicate value; only the in-range and right-edge
        // branches overwrite it. The cube macro translates this mutation
        // pattern more reliably than if-in-expression form.
        let mut idx: usize = 0;
        if base + k >= width_u {
            let unclamped = base + k - width_u;
            if unclamped > last {
                idx = last;
            } else {
                idx = unclamped;
            }
        }
        let x = signal[signal_row_base + idx];
        let w = kernels[kernel_row_base + k];
        acc += x * w;
        k += 1;
    }

    output[pos] = acc;
}
