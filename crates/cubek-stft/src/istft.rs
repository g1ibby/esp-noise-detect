//! Inverse short-time Fourier transform: per-frame IRFFT + weighted
//! overlap-add.
//!
//! Input is `(batch, n_frames, n_freq)` real / imaginary tensors as
//! produced by [`crate::stft`]. We run `cubek-fft::irfft` on the last axis
//! to recover `(batch, n_frames, n_fft)` frames, then overlap-add with the
//! analysis window and divide each output sample by the local sum of
//! `window^2` — the standard weighted overlap-add (Griffin-Lim form,
//! window / window² correction).
//!
//! The divisor is recomputed inline on-device: for a COLA-compliant
//! Hann / `n_fft/4` setup it is constant over the interior samples, but we
//! don't want to hardcode that because the crate is window-agnostic. The
//! eps floor on the divisor avoids `0/0` at samples covered by a zero
//! window (e.g. the symmetric Hann endpoints).
//!
//! Output length: `(n_frames - 1) * hop + n_fft`. That matches what the
//! forward STFT would have framed from — we do not trim the tail.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_fft::irfft;

use crate::elemwise_launch_dims;

/// Inverse STFT.
///
/// * `spectrum_re`, `spectrum_im` — `(batch, n_frames, n_freq)`.
/// * `window` — `(n_fft,)`, must match what forward STFT used for
///   reconstruction to be exact.
/// * `hop` — frame step in samples.
/// * `dtype` — storage type. Only `f32` is currently supported.
///
/// Returns a waveform of shape `(batch, (n_frames - 1) * hop + n_fft)`.
pub fn istft<R: Runtime>(
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
    window: TensorHandle<R>,
    hop: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    assert_eq!(
        spectrum_re.shape(),
        spectrum_im.shape(),
        "re/im shape mismatch"
    );
    assert_eq!(
        spectrum_re.shape().len(),
        3,
        "istft expects spectrum of shape [batch, n_frames, n_freq]"
    );
    assert_eq!(
        window.shape().len(),
        1,
        "istft expects window of shape [n_fft]"
    );
    assert!(hop > 0, "hop must be > 0");

    let n_freq = spectrum_re.shape()[2];
    let n_fft = (n_freq - 1) * 2;
    assert_eq!(
        window.shape()[0],
        n_fft,
        "window length {} inconsistent with spectrum n_fft {}",
        window.shape()[0],
        n_fft,
    );

    // Per-frame inverse FFT: (B, n_frames, n_freq) -> (B, n_frames, n_fft).
    let frames = irfft(spectrum_re, spectrum_im, 2, dtype);

    let client = <R as Runtime>::client(&Default::default());
    let batch = frames.shape()[0];
    let n_frames = frames.shape()[1];
    let t_out = (n_frames - 1) * hop + n_fft;

    let signal = TensorHandle::<R>::new_contiguous(
        vec![batch, t_out],
        client.empty(batch * t_out * dtype.size()),
        dtype,
    );

    let num_elems = batch * t_out;
    let (cube_count, cube_dim) = elemwise_launch_dims(&client, num_elems, 256);

    overlap_add_kernel::launch::<f32, R>(
        &client,
        cube_count,
        cube_dim,
        frames.binding().into_tensor_arg(),
        window.binding().into_tensor_arg(),
        signal.clone().binding().into_tensor_arg(),
        hop as u32,
        n_fft as u32,
    );

    signal
}

/// One thread per output sample.
///
/// For output sample `(b, t)`:
///   `num = Σ_f frames[b, f, t - f*hop] * window[t - f*hop]`
///   `den = Σ_f window[t - f*hop]^2`
///   `signal[b, t] = num / max(den, eps)`
///
/// where `f` ranges over frames that cover `t`: `f*hop <= t < f*hop + n_fft`.
#[cube(launch)]
pub(crate) fn overlap_add_kernel<F: Float>(
    frames: &Tensor<F>,
    window: &Tensor<F>,
    signal: &mut Tensor<F>,
    hop: u32,
    n_fft: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= signal.len() {
        terminate!();
    }
    let t_out = signal.shape(1);
    let b = pos / t_out;
    let t = pos - b * t_out;

    let n_frames = frames.shape(1);
    let n_fft_u = n_fft as usize;
    let hop_u = hop as usize;

    // f_min: smallest f with f*hop + n_fft > t, i.e. f > (t - n_fft) / hop.
    // f_max_exclusive: one past the largest f with f*hop <= t, clamped to
    // n_frames. Written as mutation rather than `if` expressions because
    // the cube macro only translates the statement form cleanly.
    let mut f_min: usize = 0;
    if t >= n_fft_u {
        f_min = (t - n_fft_u) / hop_u + 1;
    }
    let mut f_max_exclusive = t / hop_u + 1;
    if f_max_exclusive > n_frames {
        f_max_exclusive = n_frames;
    }

    let mut num = F::new(0.0);
    let mut den = F::new(0.0);
    let frame_base = b * n_frames * n_fft_u;
    for f in f_min..f_max_exclusive {
        let i = t - f * hop_u;
        let w = window[i];
        num += frames[frame_base + f * n_fft_u + i] * w;
        den += w * w;
    }

    let eps = F::new(1e-10);
    if den > eps {
        signal[pos] = num / den;
    } else {
        signal[pos] = F::new(0.0);
    }
}
