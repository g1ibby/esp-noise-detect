//! Host-side builder for the windowed-sinc FIR filter bank.
//!
//! Implements the `julius.LowPassFilters.__init__` algorithm. Pure-Rust with
//! no GPU dependencies so the bank can be unit-tested against saved Julius
//! fixtures without spinning up a runtime.
//!
//! ## Formulae (as in Julius)
//!
//! Given a set of normalized cutoffs `c_0, ..., c_{F-1}` with `c_f = f_c / f_s`
//! and `c_f ∈ [0, 0.5]`:
//!
//! * `half_size = floor(zeros / min_positive(c) / 2)`
//! * `filter_len = 2 * half_size + 1`
//! * Hann window over `n ∈ [0, filter_len)` with `periodic=False`:
//!   `w[n] = 0.5 * (1 - cos(2π n / (filter_len - 1)))`
//! * For each cutoff `c`:
//!   * If `c == 0`: `filter_[k] = 0` (Julius documents a cutoff of 0 as the
//!     null filter. With high-pass as `x - lowpass(x)`, a high-pass cutoff of
//!     0 is then identity. We preserve that convention.)
//!   * Otherwise, for tap `k ∈ [0, filter_len)` mapped to `t = k - half_size`:
//!     `filter_[k] = 2 * c * w[k] * sinc(2 * c * π * t)` with `sinc(0) = 1`,
//!     then normalize to sum 1.0.
//!
//! The half-size derivation uses the *smallest* positive cutoff so that every
//! filter in the bank has the same length — that's what Julius does when a
//! list of cutoffs is passed to `LowPassFilters`. It is also what makes a
//! grouped batched convolution on the GPU possible.

/// Precomputed windowed-sinc FIR filter bank.
///
/// `weights` is `n_cutoffs * filter_len` f32 values in row-major layout;
/// row `i` is the filter for cutoff `cutoffs[i]`. `filter_len = 2 * half_size + 1`.
#[derive(Debug, Clone)]
pub(crate) struct FilterBank {
    pub n_cutoffs: u32,
    pub half_size: u32,
    pub filter_len: u32,
    /// Row-major `(n_cutoffs, filter_len)` weights. For a cutoff of exactly 0
    /// the row is all zeros (Julius convention — high-pass at 0 = identity).
    pub weights: Vec<f32>,
}

impl FilterBank {
    /// Build the filter bank for a set of normalized cutoffs.
    ///
    /// * `cutoffs` — each entry `f_c / f_s`, in `[0, 0.5]`.
    /// * `zeros` — number of zero crossings of the sinc to retain (receptive
    ///   field control). Julius's default is 8. Larger `zeros` means a
    ///   sharper transition band but longer filter.
    ///
    /// Panics on empty `cutoffs`, cutoffs outside `[0, 0.5]`, `zeros == 0`,
    /// or when every cutoff is zero (the filter length would be ill-defined).
    pub fn new(cutoffs: &[f32], zeros: u32) -> Self {
        assert!(!cutoffs.is_empty(), "cutoffs must not be empty");
        assert!(zeros > 0, "zeros must be > 0");
        for &c in cutoffs {
            assert!(
                (0.0..=0.5).contains(&c),
                "cutoff {c} out of [0, 0.5]; express as f_c / f_s",
            );
        }

        // `half_size = floor(zeros / min_positive(c) / 2)` — Julius derives
        // the kernel length from the smallest positive cutoff so every filter
        // in the bank has the same length and a grouped convolution becomes
        // well-defined. If every cutoff is zero we have no way to choose a
        // length; reject loudly.
        let min_positive = cutoffs
            .iter()
            .copied()
            .filter(|&c| c > 0.0)
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_positive.is_finite(),
            "at least one cutoff must be > 0",
        );
        let half_size = (zeros as f32 / min_positive / 2.0).floor() as u32;
        assert!(
            half_size > 0,
            "derived half_size is 0 (zeros={zeros}, min_cutoff={min_positive}); \
             increase zeros or lower the minimum cutoff",
        );
        let filter_len = 2 * half_size + 1;

        // Hann window with `periodic=False` over `filter_len` samples. We
        // compute it once and re-use across rows. Double precision during
        // construction to match Julius's float64-default behaviour; we cast
        // to f32 after normalization.
        let filter_len_f = filter_len as f64;
        let mut hann = vec![0.0f64; filter_len as usize];
        for n in 0..filter_len as usize {
            hann[n] = 0.5 * (1.0 - (2.0 * core::f64::consts::PI * n as f64 / (filter_len_f - 1.0)).cos());
        }

        let mut weights = vec![0.0f32; (cutoffs.len()) * (filter_len as usize)];
        for (row, &c) in cutoffs.iter().enumerate() {
            let row_base = row * filter_len as usize;
            if c == 0.0 {
                // All-zeros row. DC-normalization is undefined here (sum == 0),
                // so skip it and rely on the consumer's high-pass=identity /
                // low-pass=null convention.
                continue;
            }
            let c_f64 = c as f64;
            let mut sum = 0.0f64;
            for k in 0..filter_len as usize {
                let t = (k as i64 - half_size as i64) as f64;
                let x = 2.0 * c_f64 * core::f64::consts::PI * t;
                let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
                let v = 2.0 * c_f64 * hann[k] * sinc;
                weights[row_base + k] = v as f32;
                sum += v;
            }
            let inv = (1.0 / sum) as f32;
            for k in 0..filter_len as usize {
                weights[row_base + k] *= inv;
            }
        }

        Self {
            n_cutoffs: cutoffs.len() as u32,
            half_size,
            filter_len,
            weights,
        }
    }
}

/// Compute Julius's `half_size` for a given minimum positive cutoff / zeros.
/// Exposed for callers that want to know the filter length before building
/// the bank (e.g. to size intermediate buffers).
pub fn half_size_from(min_cutoff: f32, zeros: u32) -> u32 {
    assert!(min_cutoff > 0.0, "min_cutoff must be > 0");
    assert!(zeros > 0, "zeros must be > 0");
    (zeros as f32 / min_cutoff / 2.0).floor() as u32
}

// Keep sinc accessible for tests even though the builder computes it inline.
#[cfg(test)]
pub(crate) fn sinc(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { x.sin() / x }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_min_cutoff_pick_half_size() {
        // Matches Julius: int(8 / 0.125 / 2) = 32.
        assert_eq!(half_size_from(0.125, 8), 32);
        // zeros=24 at min cutoff 0.25 -> int(48) = 48.
        assert_eq!(half_size_from(0.25, 24), 48);
    }

    #[test]
    fn rows_sum_to_one_for_positive_cutoffs() {
        let bank = FilterBank::new(&[0.05, 0.1, 0.25], 8);
        // half_size = int(8 / 0.05 / 2) = 80, filter_len = 161.
        assert_eq!(bank.half_size, 80);
        assert_eq!(bank.filter_len, 161);
        for row in 0..bank.n_cutoffs as usize {
            let r = &bank.weights
                [row * bank.filter_len as usize..(row + 1) * bank.filter_len as usize];
            let s: f32 = r.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "row {row} sums to {s}, expected 1.0",
            );
        }
    }

    #[test]
    fn zero_cutoff_row_is_all_zeros() {
        let bank = FilterBank::new(&[0.0, 0.1], 8);
        // half_size comes from the positive cutoff 0.1.
        let row = &bank.weights[0..bank.filter_len as usize];
        assert!(row.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn matches_julius_formula_at_a_handful_of_taps() {
        // Compute a reference against the exact Julius formula.
        let cutoff = 0.1f64;
        let zeros = 8u32;
        let half_size = (zeros as f64 / cutoff / 2.0).floor() as u32;
        let filter_len = 2 * half_size + 1;
        let mut reference = vec![0.0f64; filter_len as usize];
        let mut sum = 0.0f64;
        for k in 0..filter_len as usize {
            let t = (k as i64 - half_size as i64) as f64;
            let hann = 0.5
                * (1.0
                    - (2.0 * core::f64::consts::PI * k as f64 / (filter_len as f64 - 1.0)).cos());
            let v = 2.0 * cutoff * hann * sinc(2.0 * cutoff * core::f64::consts::PI * t);
            reference[k] = v;
            sum += v;
        }
        for v in reference.iter_mut() {
            *v /= sum;
        }

        let bank = FilterBank::new(&[cutoff as f32], zeros);
        assert_eq!(bank.filter_len as usize, reference.len());
        for (k, (got, want)) in bank.weights.iter().zip(reference.iter()).enumerate() {
            assert!(
                (*got as f64 - *want).abs() < 1e-6,
                "tap {k}: got {got}, want {want}",
            );
        }
    }

    #[test]
    fn bank_is_symmetric_per_row() {
        // Windowed-sinc with a cosine window is symmetric by construction;
        // a regression here usually means a sign-flip somewhere.
        let bank = FilterBank::new(&[0.1, 0.2], 8);
        for row in 0..bank.n_cutoffs as usize {
            let r = &bank.weights
                [row * bank.filter_len as usize..(row + 1) * bank.filter_len as usize];
            for k in 0..bank.half_size as usize {
                let lhs = r[k];
                let rhs = r[r.len() - 1 - k];
                assert!(
                    (lhs - rhs).abs() < 1e-7,
                    "row {row} tap {k} asymmetric: {lhs} vs {rhs}",
                );
            }
        }
    }
}
