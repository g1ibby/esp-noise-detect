//! Phase-locked phase vocoder kernel + host launcher.
//!
//! Accumulator semantics:
//!
//! * `phase_acc[0] = angle(input[..., 0])`
//! * `phase_acc[t] = phase_acc[t-1] + wrap(angle_1[t-1] - angle_0[t-1] - pa) + pa`
//!
//! where `angle_{0,1}[t]` are the angles of the input frames at indices
//! `idx0[t]` and `idx0[t] + 1`, and `pa` is the per-bin expected phase
//! advance (typically `pi * hop * f_bin / (n_fft/2)` for a standard STFT).
//!
//! ## wgpu-Metal sharp edges
//!
//! * `atan2(0, 0) = NaN` instead of the IEEE convention `0`.
//!   The zero-pad branch must gate the `atan2` call behind the bounds
//!   check or one poisoned frame drives `phase_acc` to NaN for the rest
//!   of the thread.
//! * `cos` / `sin` precision degrades at large arguments while the CPU
//!   reference stays accurate. We wrap `phase_acc` into `(-π, π]` after
//!   every update so the two paths compute trig on comparable inputs.

use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::elemwise_launch_dims;

/// Time-stretch a complex spectrogram by `rate` without modifying pitch.
///
/// * `spectrum_re`, `spectrum_im` — `(batch, n_freq, n_frames)`.
/// * `phase_advance` — `(n_freq,)`. Typically
///   `linspace(0, PI * hop, n_freq)`, but the caller builds it — we just
///   consume it bin-by-bin.
/// * `rate` — speed-up factor. `rate > 1.0` produces a shorter output
///   (time-compressed), `rate < 1.0` a longer one (time-stretched).
///   Must be > 0.
/// * `dtype` — storage type of every float tensor. Only `f32` is wired up
///   inside the kernel today.
///
/// Returns `(real, imag)` of shape `(batch, n_freq, ceil(n_frames / rate))`.
///
/// At `rate == 1.0` the input handles are returned unchanged.
pub fn phase_vocoder<R: Runtime>(
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
    phase_advance: TensorHandle<R>,
    rate: f32,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    assert_eq!(
        spectrum_re.shape(),
        spectrum_im.shape(),
        "re/im shape mismatch",
    );
    assert_eq!(
        spectrum_re.shape().len(),
        3,
        "phase_vocoder expects spectrum of shape [batch, n_freq, n_frames]",
    );
    assert_eq!(
        phase_advance.shape().len(),
        1,
        "phase_vocoder expects phase_advance of shape [n_freq]",
    );
    let shape = spectrum_re.shape().clone();
    let batch = shape[0];
    let n_freq = shape[1];
    let n_in = shape[2];
    assert_eq!(
        phase_advance.shape()[0],
        n_freq,
        "phase_advance length {} does not match n_freq {}",
        phase_advance.shape()[0],
        n_freq,
    );
    assert!(rate > 0.0, "rate must be > 0, got {rate}");
    assert!(n_in > 0, "n_frames must be > 0");

    // The kernel is only exact at rate=1.0 when the spectrogram's true
    // inter-frame phase advance matches `phase_advance`, which is not
    // generally the case. Return the input unchanged.
    if rate == 1.0 {
        return (spectrum_re, spectrum_im);
    }

    let n_out = ((n_in as f64) / (rate as f64)).ceil() as usize;
    assert!(n_out > 0);

    let client = <R as Runtime>::client(&Default::default());

    let out_shape = vec![batch, n_freq, n_out];
    let num_elems = batch * n_freq * n_out;
    let out_re = TensorHandle::<R>::new_contiguous(
        out_shape.clone(),
        client.empty(num_elems * dtype.size()),
        dtype,
    );
    let out_im = TensorHandle::<R>::new_contiguous(
        out_shape,
        client.empty(num_elems * dtype.size()),
        dtype,
    );

    // One thread per (batch, freq) pair. Inner loop walks t sequentially.
    let total_threads = batch * n_freq;
    let (cube_count, cube_dim) = elemwise_launch_dims(&client, total_threads, 256);

    phase_vocoder_kernel::launch::<f32, R>(
        &client,
        cube_count,
        cube_dim,
        spectrum_re.binding().into_tensor_arg(),
        spectrum_im.binding().into_tensor_arg(),
        phase_advance.binding().into_tensor_arg(),
        out_re.clone().binding().into_tensor_arg(),
        out_im.clone().binding().into_tensor_arg(),
        rate,
        n_out as u32,
    );

    (out_re, out_im)
}

/// One thread per `(batch, freq)` pair.
///
/// Reads:
///   * `input_re / input_im`: `(B, n_freq, n_in)` row-major contiguous.
///   * `phase_advance`: `(n_freq,)`.
///
/// Writes:
///   * `output_re / output_im`: `(B, n_freq, n_out)` row-major contiguous.
///
/// Scalars:
///   * `rate`: time-stretch factor.
///   * `n_out`: number of output frames. Passed in explicitly rather than
///     derived from `output_re.shape(2)` so there is exactly one source of
///     truth for the loop bound — the host-computed value that also sized
///     the output allocation.
#[cube(launch)]
pub(crate) fn phase_vocoder_kernel<F: Float>(
    input_re: &Tensor<F>,
    input_im: &Tensor<F>,
    phase_advance: &Tensor<F>,
    output_re: &mut Tensor<F>,
    output_im: &mut Tensor<F>,
    rate: f32,
    n_out: u32,
) {
    let pos = ABSOLUTE_POS;
    let n_freq = input_re.shape(1);
    let batch = input_re.shape(0);
    let total = batch * n_freq;
    if pos >= total {
        terminate!();
    }
    let b = pos / n_freq;
    let f = pos - b * n_freq;

    let n_in = input_re.shape(2);
    let n_out_u = n_out as usize;

    let row_base_in = (b * n_freq + f) * n_in;
    let row_base_out = (b * n_freq + f) * n_out_u;

    let pa = phase_advance[f];
    let two_pi = F::new(2.0 * PI);
    let rate_f = F::cast_from(rate);

    // phase_acc[0] = angle(input[b, f, 0])
    let re0 = input_re[row_base_in];
    let im0 = input_im[row_base_in];
    let mut phase_acc = im0.atan2(re0);

    let mut t: usize = 0;
    while t < n_out_u {
        // time_step = t * rate, idx0 = floor(time_step), alpha = frac.
        // We take the floor explicitly before casting to u32: depending
        // on which `u32::cast_from(f32)` lowering the backend picks, a
        // raw cast can round to nearest rather than truncate, which would
        // drop and duplicate frames around every half-integer crossing
        // of `t * rate`. Flooring first forces floor-and-cast regardless.
        let time_step = F::cast_from(t) * rate_f;
        let idx0_f = time_step.floor();
        let idx0_u = u32::cast_from(idx0_f);
        let idx0 = idx0_u as usize;
        let idx1 = idx0 + 1;
        let alpha = time_step - idx0_f;

        // Zero-pad past n_in. Written as mutation rather than `if`
        // expressions because the cube macro translates statement form
        // more reliably than rvalue-position `if`s.
        let mut c0_re = F::new(0.0);
        let mut c0_im = F::new(0.0);
        if idx0 < n_in {
            c0_re = input_re[row_base_in + idx0];
            c0_im = input_im[row_base_in + idx0];
        }
        let mut c1_re = F::new(0.0);
        let mut c1_im = F::new(0.0);
        if idx1 < n_in {
            c1_re = input_re[row_base_in + idx1];
            c1_im = input_im[row_base_in + idx1];
        }

        let norm_0 = (c0_re * c0_re + c0_im * c0_im).sqrt();
        let norm_1 = (c1_re * c1_re + c1_im * c1_im).sqrt();
        let mag = alpha * norm_1 + (F::new(1.0) - alpha) * norm_0;

        // Emit at the *current* phase_acc, then advance it.
        output_re[row_base_out + t] = mag * phase_acc.cos();
        output_im[row_base_out + t] = mag * phase_acc.sin();

        // Compute angles conditionally. On wgpu-Metal, `atan2(0, 0)`
        // returns NaN instead of 0, so calling it on the zero-padded
        // branch poisons `phase_acc` for every subsequent iteration.
        // Gate behind bounds checks to force angle = 0 out of range.
        let mut angle_0 = F::new(0.0);
        if idx0 < n_in {
            angle_0 = c0_im.atan2(c0_re);
        }
        let mut angle_1 = F::new(0.0);
        if idx1 < n_in {
            angle_1 = c1_im.atan2(c1_re);
        }
        let mut delta = angle_1 - angle_0 - pa;
        // Phase unwrap: map to (-pi, pi] by subtracting multiples of 2pi.
        delta = delta - two_pi * (delta / two_pi).round();
        phase_acc = phase_acc + delta + pa;
        // Wrap `phase_acc` into (-pi, pi]. Mathematically redundant, but
        // without it the accumulator grows to ~n_out * pa and wgpu-Metal's
        // cos/sin lose precision at large arguments, diverging from the CPU
        // reference.
        phase_acc = phase_acc - two_pi * (phase_acc / two_pi).round();

        t += 1;
    }
}
