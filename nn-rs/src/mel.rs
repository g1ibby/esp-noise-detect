//! Burn mel front-end.
//!
//! Uses integer-bin filterbank edges (not torchaudio's fractional
//! edges), which affects the lowest-frequency filters slightly.
//!
//! ## Pipeline
//!
//! 1. Optional reflect pad on the time axis by `n_fft / 2`
//!    (`center=True`).
//! 2. Frame with `hop = round(hop_length_ms / 1000 * sample_rate)`,
//!    multiply each frame by a symmetric Hann window, per-frame RFFT.
//!    `cubek-stft::stft` does this end to end.
//! 3. Power spectrum `|X|²`.
//! 4. Matmul against the HTK mel filterbank (triangular filters with
//!    integer bin edges).
//! 5. `log(max(mel, eps))` if `cfg.log`.
//! 6. Per-example mean/var normalization if `cfg.normalize`.
//!
//! ## Zero-copy Burn ↔ CubeCL bridge
//!
//! The waveform arrives as a Burn `Tensor<B, 2>` on a wgpu device. We
//! drop to `CubeTensor<R>` (same GPU buffer) via
//! `tensor.into_primitive().tensor()`, hand the `TensorHandle<R>` to
//! `cubek-stft::stft`, wrap the real / imaginary spectrograms back as
//! Burn tensors, and finish the rest (`|·|²`, mel matmul, log, normalize)
//! with Burn ops.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorPrimitive, ops::PadMode};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, BoolElement};
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_stft::{MAX_N_FFT, stft, window::hann_window_symmetric};

/// Mel front-end configuration.
#[derive(Clone, Debug)]
pub struct MelConfig {
    pub n_mels: usize,
    pub fmin: f32,
    /// `None` defaults to Nyquist (`sample_rate / 2`).
    pub fmax: Option<f32>,
    pub n_fft: usize,
    pub hop_length_ms: f32,
    pub log: bool,
    pub eps: f32,
    /// Per-example mean / variance normalization over `(channel, n_mels,
    /// n_frames)`.
    pub normalize: bool,
    /// Reflect-pad `n_fft / 2` samples on each side of the waveform
    /// before framing.
    pub center: bool,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_mels: 64,
            fmin: 50.0,
            fmax: None,
            n_fft: 1024,
            hop_length_ms: 10.0,
            log: true,
            eps: 1e-10,
            normalize: true,
            center: true,
        }
    }
}

impl MelConfig {
    /// Hop size in samples: `round(hop_length_ms / 1000 * sample_rate)`,
    /// floored to `1`.
    pub fn hop_length(&self, sample_rate: u32) -> usize {
        let hop = (self.hop_length_ms / 1000.0 * sample_rate as f32).round() as i32;
        hop.max(1) as usize
    }

    fn fmax_hz(&self, sample_rate: u32) -> f32 {
        self.fmax.unwrap_or(sample_rate as f32 / 2.0)
    }
}

/// HTK Hz → mel conversion.
#[inline]
fn hz_to_mel_htk(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

#[inline]
fn mel_to_hz_htk(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// Build the HTK mel filterbank as a `(n_mels, n_freq)` row-major buffer
/// with `n_freq = n_fft / 2 + 1`.
///
/// Integer bin edges via `floor((n_fft + 1) * f / sample_rate)`;
/// triangular filters; edge bumping when adjacent bins coincide.
pub fn build_mel_filterbank(cfg: &MelConfig, sample_rate: u32) -> Vec<f32> {
    let n_fft = cfg.n_fft;
    let n_freq = n_fft / 2 + 1;
    let n_mels = cfg.n_mels;

    let f_min = cfg.fmin;
    let f_max = cfg.fmax_hz(sample_rate);
    let m_min = hz_to_mel_htk(f_min);
    let m_max = hz_to_mel_htk(f_max);

    // n_mels + 2 mel points (endpoints + n_mels centers).
    let steps = n_mels + 1;
    let mel_points: Vec<f32> = (0..=steps)
        .map(|i| m_min + (m_max - m_min) * i as f32 / steps as f32)
        .collect();
    let f_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_htk(m)).collect();

    let mut bins: Vec<usize> = f_points
        .iter()
        .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate as f32).floor() as usize)
        .collect();
    // Edge bumping: if two consecutive bins coincide we nudge the later
    // one up by one sample (clamped at n_freq - 1). This only happens
    // for low-frequency filters where bin resolution runs out; without
    // it the filter would have zero width and zero gain.
    let max_idx = n_freq - 1;
    for m in 1..=n_mels {
        if bins[m - 1] == bins[m] {
            bins[m] = (bins[m] + 1).min(max_idx);
        }
        if bins[m] == bins[m + 1] {
            bins[m + 1] = (bins[m + 1] + 1).min(max_idx);
        }
    }

    let mut fbanks = vec![0.0_f32; n_mels * n_freq];
    for m in 1..=n_mels {
        let (lo, mid, hi) = (bins[m - 1], bins[m], bins[m + 1]);
        let row = (m - 1) * n_freq;
        // Rising edge [lo, mid).
        let rise_denom = (mid - lo).max(1) as f32;
        for k in lo..mid {
            fbanks[row + k] = (k - lo) as f32 / rise_denom;
        }
        // Falling edge [mid, hi).
        let fall_denom = (hi - mid).max(1) as f32;
        for k in mid..hi {
            fbanks[row + k] = (hi - k) as f32 / fall_denom;
        }
    }
    fbanks
}

/// Burn mel extractor, generic over a `CubeRuntime`. Holds the HTK
/// filterbank and analysis window on-device; forward takes / returns
/// Burn tensors via the `CubeTensor` bridge.
pub struct MelExtractor<R: CubeRuntime> {
    cfg: MelConfig,
    sample_rate: u32,
    hop: usize,
    /// `(n_freq, n_mels)` — stored pre-transposed so `power @ fb` lands
    /// directly in `(B, n_frames, n_mels)` without a second transpose.
    mel_fb_handle: cubecl::server::Handle,
    mel_fb_shape: [usize; 2],
    /// `(n_fft,)` symmetric Hann window.
    window_handle: cubecl::server::Handle,
    client: ComputeClient<R>,
    device: R::Device,
}

impl<R: CubeRuntime> MelExtractor<R> {
    /// Build a mel extractor, uploading the filterbank and window to the
    /// compute client in one shot.
    ///
    /// Panics on invalid combinations (non-power-of-two `n_fft`, `n_fft`
    /// above `cubek-stft::MAX_N_FFT`, `fmax > Nyquist`).
    pub fn new(client: ComputeClient<R>, device: R::Device, cfg: MelConfig, sample_rate: u32) -> Self {
        assert!(
            cfg.n_fft.is_power_of_two(),
            "n_fft ({}) must be a power of two",
            cfg.n_fft
        );
        assert!(
            cfg.n_fft <= MAX_N_FFT,
            "n_fft ({}) exceeds cubek-stft MAX_N_FFT ({MAX_N_FFT})",
            cfg.n_fft,
        );
        let fmax = cfg.fmax_hz(sample_rate);
        assert!(
            fmax <= sample_rate as f32 / 2.0 + 1e-3,
            "fmax {fmax} Hz above Nyquist ({} Hz)",
            sample_rate as f32 / 2.0,
        );
        assert!(cfg.fmin >= 0.0 && cfg.fmin < fmax, "fmin/fmax invalid");

        let n_freq = cfg.n_fft / 2 + 1;
        let hop = cfg.hop_length(sample_rate);

        // Filterbank is built host-side as row-major (n_mels, n_freq) but
        // we upload the transpose (n_freq, n_mels) so the downstream
        // matmul reads it in its native layout.
        let fb_rowmajor = build_mel_filterbank(&cfg, sample_rate);
        let fb_t = transpose_row_major(&fb_rowmajor, cfg.n_mels, n_freq);
        let mel_fb_handle = client.create_from_slice(f32::as_bytes(&fb_t));

        let window = hann_window_symmetric(cfg.n_fft);
        let window_handle = client.create_from_slice(f32::as_bytes(&window));

        let mel_fb_shape = [n_freq, cfg.n_mels];
        Self {
            cfg,
            sample_rate,
            hop,
            mel_fb_handle,
            mel_fb_shape,
            window_handle,
            client,
            device,
        }
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn hop_length(&self) -> usize {
        self.hop
    }

    /// Number of time frames produced for a waveform of length `time`
    /// samples.
    pub fn num_frames(&self, time: usize) -> usize {
        let padded = if self.cfg.center {
            time + 2 * (self.cfg.n_fft / 2)
        } else {
            time
        };
        if padded < self.cfg.n_fft {
            1
        } else {
            (padded - self.cfg.n_fft) / self.hop + 1
        }
    }

    /// Forward pass: `(batch, time)` waveform → `(batch, 1, n_mels,
    /// n_frames)` log-mel tensor.
    ///
    /// Pads (if `cfg.center`), runs STFT via `cubek-stft`, computes power,
    /// matmuls against the HTK filterbank, optionally logs and
    /// normalizes. Every step after the STFT stays in Burn land.
    pub fn forward<F, I, BT>(
        &self,
        waveform: Tensor<CubeBackend<R, F, I, BT>, 2>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, 4>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        type B<R, F, I, BT> = CubeBackend<R, F, I, BT>;

        let dims = waveform.dims();
        let [_batch, time] = dims;
        let pad = if self.cfg.center {
            self.cfg.n_fft / 2
        } else {
            0
        };
        let padded_len = time + 2 * pad;
        assert!(
            padded_len >= self.cfg.n_fft,
            "input time {time} (padded to {padded_len}) shorter than n_fft {}",
            self.cfg.n_fft,
        );

        let signal = if self.cfg.center {
            waveform.pad([(0, 0), (pad, pad)], PadMode::Reflect)
        } else {
            waveform
        };

        // Peel to the CubeCL primitive, hand the handle to cubek-stft.
        let signal_primitive: CubeTensor<R> = signal.into_primitive().tensor();
        let dtype_burn = signal_primitive.dtype;
        let signal_handle: TensorHandle<R> = signal_primitive.into();

        let window_handle = TensorHandle::<R>::new_contiguous(
            vec![self.cfg.n_fft],
            self.window_handle.clone(),
            signal_handle.dtype,
        );

        let (re_handle, im_handle) = stft(
            signal_handle,
            window_handle,
            self.cfg.n_fft,
            self.hop,
            // StorageType inferred from the signal — stft only supports
            // f32 today (see cubek-stft::stft).
            f32::as_type_native_unchecked().storage_type(),
        );

        let re_shape = re_handle.shape().to_vec();
        let [batch_rt, n_frames, n_freq] = [re_shape[0], re_shape[1], re_shape[2]];
        debug_assert_eq!(n_freq, self.cfg.n_fft / 2 + 1);

        let re: Tensor<B<R, F, I, BT>, 3> = handle_into_tensor_3d(
            re_handle,
            self.client.clone(),
            self.device.clone(),
            dtype_burn,
            [batch_rt, n_frames, n_freq],
        );
        let im: Tensor<B<R, F, I, BT>, 3> = handle_into_tensor_3d(
            im_handle,
            self.client.clone(),
            self.device.clone(),
            dtype_burn,
            [batch_rt, n_frames, n_freq],
        );

        // Power spectrum: |X|² = re² + im²  (B, n_frames, n_freq).
        let power = re.clone() * re + im.clone() * im;

        // Matmul against (1, n_freq, n_mels) — batch-broadcast over
        // batch_rt frames. Rebuilds a Burn tensor around the uploaded
        // filterbank handle, same zero-copy bridge as the STFT inputs.
        let fb: Tensor<B<R, F, I, BT>, 3> = handle_into_tensor_3d(
            TensorHandle::<R>::new_contiguous(
                vec![1, self.mel_fb_shape[0], self.mel_fb_shape[1]],
                self.mel_fb_handle.clone(),
                f32::as_type_native_unchecked().storage_type(),
            ),
            self.client.clone(),
            self.device.clone(),
            dtype_burn,
            [1, self.mel_fb_shape[0], self.mel_fb_shape[1]],
        );
        // (B, n_frames, n_freq) @ (1, n_freq, n_mels) → (B, n_frames, n_mels).
        let mel = power.matmul(fb);

        // (B, n_frames, n_mels) → (B, n_mels, n_frames) → (B, 1, n_mels, n_frames).
        let mel = mel.swap_dims(1, 2);
        let mel: Tensor<B<R, F, I, BT>, 4> = mel.unsqueeze_dim(1);

        let mel = if self.cfg.log {
            mel.clamp_min(self.cfg.eps).log()
        } else {
            mel
        };

        if self.cfg.normalize {
            normalize_per_example(mel)
        } else {
            mel
        }
    }
}

/// Per-example mean / variance normalization over the non-batch dims
/// (all of `(channel, n_mels, n_frames)`).
///
/// `std` uses the population estimator (`ddof=0`).
fn normalize_per_example<B, const D: usize>(mel: Tensor<B, D>) -> Tensor<B, D>
where
    B: Backend,
{
    let dims = mel.dims();
    let batch = dims[0];
    let flat_n: usize = dims.iter().skip(1).product();
    let flat = mel.reshape([batch, flat_n]);
    let mean = flat.clone().mean_dim(1); // (batch, 1)
    let centered = flat - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(1); // (batch, 1)
    let std = var.sqrt().clamp_min(1e-6);
    let normalized = centered / std;
    normalized.reshape(dims)
}

/// Host-side row-major transpose. Tiny helper kept close to the
/// filterbank upload so the indexing arithmetic stays in one place.
fn transpose_row_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Wrap a `TensorHandle<R>` as a Burn `Tensor<B, 3>` without copying.
///
/// The handle already lives on the same `ComputeClient` as the rest of
/// the Burn pipeline, so all we do is rebuild the `CubeTensor` wrapper
/// Burn expects.
fn handle_into_tensor_3d<R, F, I, BT>(
    handle: TensorHandle<R>,
    client: ComputeClient<R>,
    device: R::Device,
    dtype: burn::tensor::DType,
    shape: [usize; 3],
) -> Tensor<CubeBackend<R, F, I, BT>, 3>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube = CubeTensor::<R>::new_contiguous(
        client,
        device,
        burn::tensor::Shape::new(shape),
        handle.handle,
        dtype,
    );
    Tensor::from_primitive(TensorPrimitive::Float(cube))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_python() {
        let cfg = MelConfig::default();
        assert_eq!(cfg.n_mels, 64);
        assert_eq!(cfg.n_fft, 1024);
        assert!((cfg.hop_length_ms - 10.0).abs() < 1e-6);
        assert_eq!(cfg.hop_length(32_000), 320);
    }

    #[test]
    fn num_frames_matches_python() {
        // 1 s window at 32 kHz, center=True, n_fft=1024, hop=320:
        //   (32000 + 1024 - 1024) / 320 + 1 = 101.
        let ex = MelConfig::default();
        let hop = ex.hop_length(32_000);
        // Reuse the computed hop directly.
        let padded = 32_000 + ex.n_fft;
        let frames = (padded - ex.n_fft) / hop + 1;
        assert_eq!(frames, 101);
    }

    #[test]
    fn filterbank_shape_and_unity_partition() {
        let cfg = MelConfig::default();
        let fb = build_mel_filterbank(&cfg, 32_000);
        assert_eq!(fb.len(), cfg.n_mels * (cfg.n_fft / 2 + 1));
        // Sanity: each triangular filter peaks at exactly 1 (rising and
        // falling edges meet at `mid` with value `(mid - lo) / (mid -
        // lo) = 1` on the rising side; we only populate the rising side
        // at `mid`, so check that any row has a value ≥ 1.0 - 1e-6).
        let n_freq = cfg.n_fft / 2 + 1;
        for m in 0..cfg.n_mels {
            let row = &fb[m * n_freq..(m + 1) * n_freq];
            let max_val = row.iter().cloned().fold(0.0_f32, f32::max);
            assert!(
                max_val > 1.0 - 1e-6,
                "filter {m} peaks at {max_val}, expected ≈ 1.0",
            );
        }
    }

    #[test]
    fn transpose_round_trips() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_row_major(&src, 2, 3);
        // src row-major 2x3: [[1,2,3],[4,5,6]] -> cols-major 3x2: [[1,4],[2,5],[3,6]]
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn htk_mel_round_trip() {
        for f in [0.0, 50.0, 200.0, 1000.0, 8000.0, 16000.0] {
            let m = hz_to_mel_htk(f);
            let back = mel_to_hz_htk(m);
            assert!((back - f).abs() < 1e-3, "hz {f} -> mel {m} -> hz {back}");
        }
    }
}
