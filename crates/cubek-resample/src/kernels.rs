//! Host-side builder for the polyphase sinc/cos² kernel bank.
//!
//! Pure-Rust module with no GPU dependencies so the table can be
//! unit-tested without spinning up a runtime.
//!
//! ## Formulae
//!
//! Let `gcd = gcd(old_sr, new_sr)`, `O = old_sr / gcd`, `N = new_sr / gcd`.
//! Let `sr = min(O, N) * rolloff` and `width = ceil(zeros * O / sr)`.
//! The kernel length is `kernel_len = 2*width + O`. For each output
//! phase `i ∈ [0, N)` and kernel tap `k ∈ [-width, width + O)`:
//!
//! * `t = (-i/N + k/O) * sr`
//! * `t = clamp(t, -zeros, zeros) * pi`
//! * `w = cos(t / (2*zeros))^2`           // Hann-like window in mapped-t
//! * `kernel[i, k + width] = sinc(t) * w`, where `sinc(0) = 1`
//! * Normalize: `kernel[i, :] /= sum(kernel[i, :])` so a DC input is
//!   preserved bit-for-bit (up to float rounding).
//!
//! ## Edge cases handled
//!
//! * `old_sr == new_sr` (after GCD) — the bank is empty and the [`Resampler`]
//!   short-circuits to a noop.
//! * `sinc(0) = 1` — special-cased to avoid `sin(0)/0` NaN.

use core::f32::consts::PI;

/// Precomputed FIR bank.
///
/// `kernels` is `new_sr * kernel_len` f32 values in row-major layout; row
/// `i` is the kernel for output phase `i`. `kernel_len = 2 * width + old_sr`.
#[derive(Debug, Clone)]
pub(crate) struct KernelBank {
    pub old_sr: u32,
    pub new_sr: u32,
    pub width: u32,
    pub kernel_len: u32,
    /// Row-major `(new_sr, kernel_len)` kernel table.
    pub kernels: Vec<f32>,
}

impl KernelBank {
    /// Build the kernel table for `old_sr → new_sr` resampling.
    ///
    /// Panics if either rate is zero. `zeros` must be > 0; `rolloff` must be
    /// in `(0, 1]`. Recommended defaults: `zeros=24, rolloff=0.945`.
    pub fn new(old_sr: u32, new_sr: u32, zeros: u32, rolloff: f32) -> Self {
        assert!(old_sr > 0 && new_sr > 0, "sample rates must be positive");
        assert!(zeros > 0, "zeros must be > 0");
        assert!(
            rolloff > 0.0 && rolloff <= 1.0,
            "rolloff must be in (0, 1], got {rolloff}",
        );

        let g = gcd(old_sr, new_sr);
        let old_sr = old_sr / g;
        let new_sr = new_sr / g;

        if old_sr == new_sr {
            // Passthrough: no kernel needed. width=0 so kernel_len=old_sr.
            // We still record the reduced rates so the Resampler can detect
            // the noop path.
            return Self {
                old_sr,
                new_sr,
                width: 0,
                kernel_len: 0,
                kernels: Vec::new(),
            };
        }

        // `sr` is the filter cutoff in reduced-rate units.
        let sr_int = old_sr.min(new_sr);
        let sr = sr_int as f32 * rolloff;

        // width = ceil(zeros * old_sr / sr). Computed in f64 for safety
        // against borderline floor/ceil at large old_sr.
        let width = ((zeros as f64 * old_sr as f64) / sr as f64).ceil() as u32;
        let kernel_len = 2 * width + old_sr;

        let mut out = vec![0.0f32; (new_sr as usize) * (kernel_len as usize)];

        let zeros_f = zeros as f32;
        let half_zeros_inv = 1.0 / (zeros_f * 2.0);
        for i in 0..new_sr {
            let row_base = (i as usize) * (kernel_len as usize);
            let mut sum = 0.0f64;
            // k ranges over [-width, width + old_sr). In our row layout
            // that is tap index 0..kernel_len, offset by width.
            for k_idx in 0..kernel_len {
                let k = (k_idx as i64) - (width as i64);
                // t = (-i/new_sr + k/old_sr) * sr
                let t_raw = (-(i as f32) / new_sr as f32
                    + (k as f32) / old_sr as f32)
                    * sr;
                let t = t_raw.clamp(-zeros_f, zeros_f) * PI;
                let w = (t * half_zeros_inv).cos();
                let w2 = w * w;
                let s = if t == 0.0 { 1.0 } else { t.sin() / t };
                let v = s * w2;
                out[row_base + k_idx as usize] = v;
                sum += v as f64;
            }
            // Normalize the row so DC is preserved. We normalize in f64
            // before casting back: for large `width` the tail taps are
            // tiny and summing f32 loses significant digits.
            let inv = (1.0 / sum) as f32;
            for k_idx in 0..kernel_len {
                out[row_base + k_idx as usize] *= inv;
            }
        }

        Self {
            old_sr,
            new_sr,
            width,
            kernel_len,
            kernels: out,
        }
    }

    /// True iff the bank represents the identity transform (old_sr ==
    /// new_sr after GCD reduction).
    pub fn is_identity(&self) -> bool {
        self.kernel_len == 0
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_handles_standard_cases() {
        assert_eq!(gcd(32_000, 44_100), 100);
        assert_eq!(gcd(48_000, 16_000), 16_000);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    #[test]
    fn identity_when_rates_equal_after_gcd() {
        let bank = KernelBank::new(16_000, 16_000, 24, 0.945);
        assert!(bank.is_identity());
        assert_eq!(bank.kernel_len, 0);
        assert_eq!(bank.kernels.len(), 0);
    }

    #[test]
    fn rates_are_reduced_by_gcd() {
        let bank = KernelBank::new(32_000, 48_000, 24, 0.945);
        assert_eq!(bank.old_sr, 2);
        assert_eq!(bank.new_sr, 3);
    }

    #[test]
    fn each_row_sums_to_one() {
        // DC-preservation: each row sums to 1.
        let bank = KernelBank::new(4, 5, 24, 0.945);
        assert_eq!(bank.kernels.len(), 5 * bank.kernel_len as usize);
        for i in 0..bank.new_sr as usize {
            let row = &bank.kernels
                [i * bank.kernel_len as usize..(i + 1) * bank.kernel_len as usize];
            let s: f32 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row {i} sums to {s}, expected 1",
            );
        }
    }

    #[test]
    fn width_scales_with_zeros() {
        let b8 = KernelBank::new(3, 2, 8, 0.945);
        let b24 = KernelBank::new(3, 2, 24, 0.945);
        assert!(b24.width > b8.width);
    }
}
