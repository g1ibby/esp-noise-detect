//! Short-time Fourier transform: frame + window + per-frame RFFT.
//!
//! Layered on top of `cubek-fft::rfft`. The only kernel we author here is a
//! framing + window-multiply step that takes a `(B, T)` signal plus a
//! `(n_fft,)` window and writes a `(B, n_frames, n_fft)` framed tensor. The
//! RFFT then runs on the last axis and produces real / imaginary tensors of
//! shape `(B, n_frames, n_freq)` with `n_freq = n_fft / 2 + 1`.
//!
//! Shape / convention decisions:
//!
//! * Input is restricted to 2-D `(batch, time)`. Higher-rank inputs can be
//!   flattened by the caller. Keeping the rank fixed avoids having to
//!   propagate a `dim` parameter through the framing kernel.
//! * Output puts `n_frames` before `n_freq`. That is the natural layout
//!   after framing + RFFT-on-last-axis and avoids an otherwise redundant
//!   transpose. `torchaudio.stft` uses `(n_freq, n_frames)` but our
//!   downstream mel / phase-vocoder steps do not care about the order and
//!   will handle permutation if they need the torchaudio layout.
//!
//! Guarded invariants (runtime `assert!`):
//!
//! * `n_fft` is a power of two.
//! * `n_fft <= MAX_N_FFT` — wgpu-Metal single-cube shared-memory
//!   ceiling. `cubek-fft` silently produces wrong output above it.
//! * `hop > 0` and `signal.shape()[1] >= n_fft`.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_fft::rfft;

use crate::{MAX_N_FFT, elemwise_launch_dims};

/// Forward short-time Fourier transform.
///
/// * `signal` — `(batch, time)`.
/// * `window` — `(n_fft,)` analysis window. See [`crate::window`] for
///   built-in Hann variants.
/// * `n_fft` — FFT length; must be a power of two and `<= MAX_N_FFT`.
/// * `hop` — frame step in samples.
/// * `dtype` — storage type of every tensor. Only `f32` is currently
///   supported by the inner kernels.
///
/// Returns `(real, imag)`, each of shape `(batch, n_frames, n_freq)` where
/// `n_frames = (time - n_fft) / hop + 1` and `n_freq = n_fft / 2 + 1`.
pub fn stft<R: Runtime>(
    signal: TensorHandle<R>,
    window: TensorHandle<R>,
    n_fft: usize,
    hop: usize,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    assert_eq!(
        signal.shape().len(),
        2,
        "stft expects signal of shape [batch, time]",
    );
    assert_eq!(
        window.shape().len(),
        1,
        "stft expects window of shape [n_fft]",
    );
    assert_eq!(
        window.shape()[0],
        n_fft,
        "window length {} does not match n_fft {}",
        window.shape()[0],
        n_fft,
    );
    assert!(
        n_fft.is_power_of_two(),
        "n_fft ({n_fft}) must be a power of two"
    );
    assert!(
        n_fft <= MAX_N_FFT,
        "n_fft ({n_fft}) exceeds single-cube bound MAX_N_FFT = {MAX_N_FFT}"
    );
    assert!(hop > 0, "hop must be > 0");
    let time = signal.shape()[1];
    assert!(
        time >= n_fft,
        "signal length {time} shorter than n_fft {n_fft}"
    );

    let client = <R as Runtime>::client(&Default::default());
    let batch = signal.shape()[0];
    let n_frames = (time - n_fft) / hop + 1;

    let frames = TensorHandle::<R>::new_contiguous(
        vec![batch, n_frames, n_fft],
        client.empty(batch * n_frames * n_fft * dtype.size()),
        dtype,
    );

    let num_elems = batch * n_frames * n_fft;
    let (cube_count, cube_dim) = elemwise_launch_dims(&client, num_elems, 256);

    frame_and_window_kernel::launch::<f32, R>(
        &client,
        cube_count,
        cube_dim,
        signal.binding().into_tensor_arg(),
        window.binding().into_tensor_arg(),
        frames.clone().binding().into_tensor_arg(),
        hop as u32,
    );

    rfft(frames, 2, dtype)
}

/// One thread per output element: `frames[b, f, i] = signal[b, f*hop + i] * window[i]`.
///
/// Assumes contiguous row-major layout for both `signal` and `frames`. The
/// window is 1-D so stride questions don't arise. Reads are gathered from
/// `signal` (strided in time by `hop` across frames) but our launcher
/// allocates everything contiguously, so the stride is 1 along time.
#[cube(launch)]
pub(crate) fn frame_and_window_kernel<F: Float>(
    signal: &Tensor<F>,
    window: &Tensor<F>,
    frames: &mut Tensor<F>,
    hop: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= frames.len() {
        terminate!();
    }
    let n_fft = frames.shape(2);
    let n_frames = frames.shape(1);

    let i = pos % n_fft;
    let f = (pos / n_fft) % n_frames;
    let b = pos / (n_fft * n_frames);

    let hop_u = hop as usize;
    let t = f * hop_u + i;

    let time_len = signal.shape(1);
    let signal_idx = b * time_len + t;
    frames[pos] = signal[signal_idx] * window[i];
}
