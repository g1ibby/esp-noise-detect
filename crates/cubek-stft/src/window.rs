//! Host-side window builders.
//!
//! STFT and iSTFT take the analysis window as a caller-supplied
//! `TensorHandle` so that the library does not force a particular window
//! choice. These helpers cover the two Hann shapes we actually use:
//!
//! * **Symmetric** (`np.hanning`) — used by the mel spectrogram path.
//! * **Periodic** (`torch.hann_window(..., periodic=True)`) — satisfies
//!   COLA (constant overlap-add) at `hop = n_fft / 4`, required for
//!   exact STFT round-trip reconstruction.
//!
//! The helpers return `Vec<f32>` rather than a `TensorHandle` so that
//! callers decide which runtime / client to upload to.

use core::f32::consts::PI;

/// Symmetric Hann window of length `n`, matching `numpy.hanning(n)`.
///
/// `w[i] = 0.5 * (1 - cos(2*pi*i / (n - 1)))`, with zeros at both ends
/// for `n >= 2`.
pub fn hann_window_symmetric(n: usize) -> Vec<f32> {
    assert!(n > 0, "window length must be > 0");
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
        .collect()
}

/// Periodic Hann window of length `n`, matching
/// `torch.hann_window(n, periodic=True)`.
///
/// `w[i] = 0.5 * (1 - cos(2*pi*i / n))`. Satisfies COLA with
/// `hop = n / 4`.
pub fn hann_window_periodic(n: usize) -> Vec<f32> {
    assert!(n > 0, "window length must be > 0");
    let denom = n as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_hann_endpoints_are_zero() {
        let w = hann_window_symmetric(1024);
        assert!(w[0].abs() < 1e-7);
        assert!(w[1023].abs() < 1e-7);
    }

    #[test]
    fn periodic_hann_has_nonzero_tail() {
        // Periodic Hann starts at zero but its last sample is nonzero.
        let w = hann_window_periodic(1024);
        assert!(w[0].abs() < 1e-7);
        assert!(w[1023].abs() > 1e-6);
    }

    #[test]
    fn periodic_hann_cola_at_quarter_hop() {
        // Sum of window^2 across frames at hop = n/4 is constant away
        // from the boundaries — the reason we pick this as the COLA
        // window in round-trip tests.
        let n = 256usize;
        let hop = n / 4;
        let w = hann_window_periodic(n);
        let n_frames = 32;
        let t_out = (n_frames - 1) * hop + n;
        let mut acc = vec![0.0f32; t_out];
        for f in 0..n_frames {
            for i in 0..n {
                acc[f * hop + i] += w[i] * w[i];
            }
        }
        // Only check the interior (away from frame ramp-in / ramp-out).
        let center = acc[t_out / 2];
        for &v in &acc[n..t_out - n] {
            assert!((v - center).abs() / center.max(1e-6) < 1e-5);
        }
    }
}
