#![allow(clippy::needless_range_loop)]

use core::f32::consts::PI;
use libm::{cosf, floorf, log10f, logf, powf, roundf, sqrtf};
use microfft::real::rfft_1024;
/// Compute in-place log-mel spectrogram with per-example normalization.
///
/// - wave_i16: mono PCM i16 at sample rate `sr`
/// - sr: sample rate (expected 32000)
/// - n_fft: FFT size (only 1024 supported here)
/// - hop_length: hop in samples (e.g., 320 for 10 ms @ 32 kHz)
/// - fmin_hz..fmax_hz: mel band edges
/// - log_eps: epsilon added before log to prevent -inf
/// - center: reflect-pad by n_fft/2 on both sides when true
/// - out[M][T]: destination buffer (M mels by T frames)
pub fn compute_log_mel_inplace<const M: usize, const T: usize>(
    wave_i16: &[i16],
    sr: usize,
    n_fft: usize,
    hop_length: usize,
    fmin_hz: f32,
    fmax_hz: f32,
    log_eps: f32,
    center: bool,
    out: &mut [[f32; T]; M],
) {
    assert!(n_fft == 1024, "only n_fft=1024 supported in MCU frontend");
    let n_fft_usize: usize = 1024;

    // Guard: compiled for a specific maximum mel bands
    const MAX_MELS: usize = 64;
    debug_assert!(M <= MAX_MELS, "M={} exceeds MAX_MELS={}", M, MAX_MELS);

    // Precompute Hann window for 1024
    let mut hann = [0f32; 1024];
    for n in 0..n_fft_usize {
        let v = 0.5f32 - 0.5f32 * cosf(2.0 * PI * (n as f32) / ((n_fft_usize - 1) as f32));
        hann[n] = v;
    }

    // PCM i16 -> f32 [-1,1]
    // We avoid allocating a separate buffer; convert on the fly via accessor.
    let len = wave_i16.len();

    // Compute frames count consistent with Python reference (center reflect padding)
    let pad = if center { n_fft_usize / 2 } else { 0 };
    let total = len + 2 * pad;
    let frames = if total >= n_fft_usize {
        1 + (total - n_fft_usize) / hop_length
    } else {
        1
    };
    assert!(frames == T, "frame count mismatch: {} != {}", frames, T);

    // Prepare mel bin edges using HTK mel scale
    // Avoid generic const expr in local array; use max M (64) and slice as needed
    let mut bins_storage = [0usize; MAX_MELS + 2];
    let m_min = hz_to_mel_htk(fmin_hz);
    let m_max = hz_to_mel_htk(fmax_hz);
    // Linspace over mels
    for i in 0..(M + 2) {
        let t = i as f32 / ((M + 1) as f32);
        let mel = m_min + t * (m_max - m_min);
        let hz = mel_to_hz_htk(mel);
        let bin = floorf(((n_fft_usize + 1) as f32) * (hz / (sr as f32))) as isize;
        // We'll clamp to the available FFT output bins later, after we know the exact spectrum length.
        let clamped = if bin < 0 { 0 } else { bin };
        bins_storage[i] = clamped as usize;
    }

    // Work buffers
    let mut frame = [0f32; 1024];
    let mut power = [0f32; 513];

    // Compute mel energies per frame
    for t in 0..T {
        // Gather frame with reflect padding
        let base: isize = (t * hop_length) as isize - (pad as isize);
        for n in 0..n_fft_usize {
            let idx = base + (n as isize);
            let s = reflect_get_i16(wave_i16, idx);
            let x = (s as f32) / 32768.0; // i16 -> [-1,1)
            frame[n] = x * hann[n];
        }

        // RFFT 1024
        let spec = rfft_1024(&mut frame);
        let spec_len = spec.len(); // microfft may return 512 bins for 1024-point RFFT
        let last_valid = if spec_len > 0 { spec_len - 1 } else { 0 };
        for k in 0..spec_len {
            let c = spec[k];
            let re = c.re;
            let im = c.im;
            power[k] = re * re + im * im;
        }

        // Clamp mel bins to available spectrum range and fix degeneracies now that we know last_valid
        // Note: do this once per frame is cheap and safe; alternatively we could precompute after first frame.
        let mut b0 = [0usize; MAX_MELS + 2];
        for i in 0..(M + 2) {
            let mut v = bins_storage[i];
            if v > last_valid {
                v = last_valid;
            }
            b0[i] = v;
        }
        for mm in 1..(M + 1) {
            if b0[mm - 1] == b0[mm] {
                b0[mm] = core::cmp::min(b0[mm] + 1, last_valid);
            }
            if b0[mm] == b0[mm + 1] {
                b0[mm + 1] = core::cmp::min(b0[mm + 1] + 1, last_valid);
            }
        }

        // Triangular mel filters
        for m in 0..M {
            let f_m_minus = b0[m];
            let f_m = b0[m + 1];
            let f_m_plus = b0[m + 2];

            let mut acc = 0.0f32;
            if f_m_minus < f_m {
                let denom = (f_m - f_m_minus) as f32;
                for k in f_m_minus..f_m {
                    let w = (k - f_m_minus) as f32 / denom.max(1e-6);
                    acc += w * power[k];
                }
            }
            if f_m < f_m_plus {
                let denom = (f_m_plus - f_m) as f32;
                for k in f_m..f_m_plus {
                    let w = (f_m_plus - k) as f32 / denom.max(1e-6);
                    acc += w * power[k];
                }
            }
            out[m][t] = acc;
        }
    }

    // Log scaling and per-example normalization
    let eps = log_eps.max(1e-12);
    // Compute mean/std over all M*T
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    for m in 0..M {
        for t in 0..T {
            // log
            let v = logf((out[m][t]).max(eps));
            out[m][t] = v;
            sum += v;
            sumsq += v * v;
        }
    }
    let count = (M * T) as f32;
    let mean = sum / count;
    let var = (sumsq / count) - mean * mean;
    let std = sqrtf(var.max(0.0)).max(1e-6);
    for m in 0..M {
        for t in 0..T {
            out[m][t] = (out[m][t] - mean) / std;
        }
    }
}

/// Quantize mel (M,T) into NHWC Buffer4D layout (1, M, T, 1)
pub fn quantize_nhwc_i8<const M: usize, const T: usize>(
    mel: &[[f32; T]; M],
    scale: f32,
    zero_point: i8,
) -> [[[[i8; 1]; T]; M]; 1] {
    let mut out = [[[[0i8; 1]; T]; M]; 1];
    let inv_scale = if scale > 0.0 { 1.0f32 / scale } else { 1.0 };
    let zp = zero_point as i32;
    for m in 0..M {
        for t in 0..T {
            let v = mel[m][t];
            let q = roundf(v * inv_scale + (zp as f32)) as i32;
            let q = q.clamp(-128, 127) as i8;
            out[0][m][t][0] = q;
        }
    }
    out
}

#[inline]
fn reflect_get_i16(x: &[i16], idx: isize) -> i16 {
    // Reflect pad indexing over [0, len)
    let n = x.len() as isize;
    if n == 0 {
        return 0;
    }
    let mut i = idx;
    if i < 0 {
        i = -i; // reflect across -0.5
    }
    let period = 2 * n;
    let mut i_mod = i % period;
    if i_mod < 0 {
        i_mod += period;
    }
    let j = if i_mod < n { i_mod } else { period - 1 - i_mod };
    x[j as usize]
}

#[inline]
fn hz_to_mel_htk(f: f32) -> f32 {
    2595.0 * log10f(1.0 + f / 700.0)
}

#[inline]
fn mel_to_hz_htk(m: f32) -> f32 {
    700.0 * (powf(10.0, m / 2595.0) - 1.0)
}
