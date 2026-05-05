//! Per-tensor observers: percentile/min-max for weights, KL-histogram for
//! activations.
//!
//! Both observers implement the same [`Observer`] trait so the
//! calibration runner ([`super::runner`]) can drive them uniformly.
//! The observers themselves only know how to consume a stream of
//! `&[f32]` slices and how to render a [`super::QuantConfig`]; the
//! decision of *what* tensor to feed them with lives in the runner.
//!
//! The implementation is byte-for-byte equivalent (within FP noise)
//! to the relevant subset of esp-ppq:
//!
//! * [`PercentileObserver`] ≡ `range.py::TorchPercentileObserver`
//!   per-tensor symmetric path. Used for weights on S3 because
//!   `EspdlQuantizer.create_default_quant_config` leaves the default
//!   observer at `"percentile"` for S3 Conv/Gemm weights.
//! * [`MinMaxObserver`] ≡ `range.py::TorchMinMaxObserver` per-tensor
//!   symmetric path. Kept for tests and for targets/configs that
//!   explicitly request min-max.
//! * [`KlHistObserver`] ≡ `range.py::TorchHistObserver` per-tensor
//!   symmetric path. Two-phase (Phase 1 collects min/max, Phase 2
//!   collects the 4096-bin abs-histogram); render derives the scale
//!   via KL-divergence search over candidate truncation thresholds.

use super::{
    QuantConfig,
    pow2::{Pow2Policy, pow2_round},
};

/// Minimum representable scale, mirroring esp-ppq's
/// `OBSERVER_MIN_SCALE = 1e-8` (`esp_ppq/core/common.py:11`). Both
/// observers clamp here to avoid emitting a literal zero scale on
/// pathological inputs.
const OBSERVER_MIN_SCALE: f32 = 1e-8;

/// `(quant_min, quant_max)` for a signed symmetric integer of width
/// `bits`. Mirrors `EspdlQuantizer.py::_quant_range` (lines 24-27).
pub fn qmin_qmax(bits: u8) -> (i64, i64) {
    let half: i64 = 1 << (bits - 1);
    (-half, half - 1)
}

/// KL histogram bin count for `bits`-wide symmetric quantization.
/// Mirrors `EspdlQuantizer.py::_kl_hist_bins` (lines 30-32):
/// `32 * 2^(bits-1)` → 4096 for INT8, 1 048 576 for INT16.
pub fn kl_hist_bins(bits: u8) -> usize {
    32 * (1usize << (bits - 1))
}

/// What every observer must do once the runner has streamed the
/// tensor's scalars at it.
pub trait Observer {
    /// Feed Phase-1 data (min/max collection). For [`MinMaxObserver`]
    /// this is the only phase. For [`KlHistObserver`] this is the
    /// first of two passes (records the dynamic range so Phase 2 can
    /// build the histogram).
    fn observe_minmax(&mut self, values: &[f32]);

    /// After Phase 1 closes, materialize whatever state Phase 2
    /// needs (e.g. the histogram bucket scale). [`MinMaxObserver`]
    /// uses the call to derive its final scale.
    fn finalize_phase1(&mut self);

    /// Feed Phase-2 data (histogram collection). [`MinMaxObserver`]
    /// ignores this branch; [`KlHistObserver`] increments per-bucket
    /// counts.
    fn observe_hist(&mut self, values: &[f32]);

    /// Final per-tensor config. Panics if the observer never saw any
    /// data — that almost certainly means a bug in the runner.
    fn render(&self, num_bits: u8) -> QuantConfig;
}

// -----------------------------------------------------------------------------
// Percentile (S3 weights)
// -----------------------------------------------------------------------------

/// Percentile used by esp-ppq when no manual override is present.
/// Mirrors `OBSERVER_PERCENTILE = 0.9999` (`core/common.py:23`).
const OBSERVER_PERCENTILE: f32 = 0.9999;

/// Symmetric per-tensor percentile observer.
///
/// Despite esp-ppq's comment saying this observer is designed for
/// activations, the ESP-DL S3 quantizer leaves Conv/Gemm weights on
/// the default `"percentile"` observer. For weights this is a single
/// observe call over the whole tensor:
///
/// ```text
/// min_idx = int(numel * (1 - percentile)) + 1
/// max_idx = min(int(numel * percentile), numel - 1) + 1
/// min = kthvalue(flat, min_idx)
/// max = kthvalue(flat, max_idx)
/// scale = minmax_to_scale_offset(min, max) with ROUND_UP pow-2 snap
/// ```
#[derive(Debug, Clone)]
pub struct PercentileObserver {
    percentile: f32,
    pairs: Vec<(f32, f32)>,
    pow2_scale: Option<f32>,
}

impl PercentileObserver {
    pub fn new() -> Self {
        Self {
            percentile: OBSERVER_PERCENTILE,
            pairs: Vec::new(),
            pow2_scale: None,
        }
    }

    fn observe_percentile(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let numel = values.len();
        let mut sorted = values.to_vec();
        sorted.sort_by(f32::total_cmp);

        let min_idx = (numel as f32 * (1.0 - self.percentile)) as usize;
        let min_idx = min_idx.min(numel - 1);
        let max_idx = (numel as f32 * self.percentile) as usize;
        let max_idx = max_idx.min(numel - 1);

        self.pairs.push((sorted[min_idx], sorted[max_idx]));
    }

    /// Float scale once `finalize_phase1` has run.
    pub fn pow2_scale(&self) -> Option<f32> {
        self.pow2_scale
    }
}

impl Default for PercentileObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for PercentileObserver {
    fn observe_minmax(&mut self, values: &[f32]) {
        self.observe_percentile(values);
    }

    fn finalize_phase1(&mut self) {
        assert!(
            !self.pairs.is_empty(),
            "PercentileObserver::finalize_phase1: observer saw no data"
        );
        let n = self.pairs.len() as f32;
        let (sum_min, sum_max) = self
            .pairs
            .iter()
            .fold((0.0_f32, 0.0_f32), |(acc_min, acc_max), (min, max)| {
                (acc_min + *min, acc_max + *max)
            });
        self.pow2_scale = Some(scale_from_minmax(sum_min / n, sum_max / n, /*bits*/ 8));
    }

    fn observe_hist(&mut self, _values: &[f32]) {
        // No-op: percentile is single-phase.
    }

    fn render(&self, num_bits: u8) -> QuantConfig {
        assert!(
            !self.pairs.is_empty(),
            "PercentileObserver::render: observer saw no data"
        );
        let n = self.pairs.len() as f32;
        let (sum_min, sum_max) = self
            .pairs
            .iter()
            .fold((0.0_f32, 0.0_f32), |(acc_min, acc_max), (min, max)| {
                (acc_min + *min, acc_max + *max)
            });
        let scale = scale_from_minmax(sum_min / n, sum_max / n, num_bits);
        QuantConfig::from_pow2_scale(scale, num_bits)
    }
}

// -----------------------------------------------------------------------------
// MinMax (weights)
// -----------------------------------------------------------------------------

/// Symmetric per-tensor min-max observer.
///
/// Used for *weights* only on S3 (per-tensor; per-channel is the
/// P4-only path we explicitly skip). After Phase 1 closes,
/// [`Self::finalize_phase1`] derives the float scale via the
/// `range = 2 · max(|min|, |max|); scale = range / (qmax-qmin)`
/// formula at `range.py:74-76`, then snaps it to a power of two via
/// `ROUND_UP` (`range.py:88-89`).
#[derive(Debug, Default, Clone)]
pub struct MinMaxObserver {
    seen: bool,
    min: f32,
    max: f32,
    pow2_scale: Option<f32>,
}

impl MinMaxObserver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a scalar slice (typically a Conv/Linear weight tensor
    /// flattened row-major, but the observer doesn't care). Updates
    /// the running min/max.
    fn update(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let (mut lo, mut hi) = if self.seen {
            (self.min, self.max)
        } else {
            (f32::INFINITY, f32::NEG_INFINITY)
        };
        for &v in values {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        self.min = lo;
        self.max = hi;
        self.seen = true;
    }

    /// Float scale once `finalize_phase1` has run.
    pub fn pow2_scale(&self) -> Option<f32> {
        self.pow2_scale
    }
}

impl Observer for MinMaxObserver {
    fn observe_minmax(&mut self, values: &[f32]) {
        self.update(values);
    }

    fn finalize_phase1(&mut self) {
        assert!(
            self.seen,
            "MinMaxObserver::finalize_phase1: observer saw no data"
        );
        self.pow2_scale = Some(scale_from_minmax(self.min, self.max, /*bits*/ 8));
    }

    fn observe_hist(&mut self, _values: &[f32]) {
        // No-op: weights are single-pass.
    }

    fn render(&self, num_bits: u8) -> QuantConfig {
        // Re-derive the scale using the requested bit width — the
        // weight observer is reusable across bit widths because the
        // (qmax - qmin) factor is the only width-dependent term.
        assert!(self.seen, "MinMaxObserver::render: observer saw no data");
        let scale = scale_from_minmax(self.min, self.max, num_bits);
        QuantConfig::from_pow2_scale(scale, num_bits)
    }
}

/// Derive a power-of-two symmetric per-tensor scale from a `(min,
/// max)` range and a bit width. Mirrors `range.py::minmax_to_scale_offset`
/// + `ppq_round_to_power_of_2(_, ROUND_UP)`:
///
/// ```text
///   range = 2 · max(|min|, |max|)
///   scale = range / (qmax - qmin)
///   scale = max(scale, OBSERVER_MIN_SCALE)
///   scale = pow2_round(scale, ROUND_UP)
/// ```
///
/// Note `min/max` are *clamped* against zero first (range.py:57-60):
/// a non-negative tensor pretends `min = 0`, a non-positive tensor
/// pretends `max = 0`. The clamp shifts which side of 0 the symmetric
/// `2 · max(|·|, |·|)` reads from — for an all-positive tensor like a
/// post-Relu activation it makes the negative side contribute 0 to
/// the range (so `range = 2 · max`).
fn scale_from_minmax(min_val: f32, max_val: f32, bits: u8) -> f32 {
    let min_clamped = min_val.min(0.0); // range.py:57-58 (`if min_val > 0: min_val = 0`)
    let max_clamped = max_val.max(0.0); // range.py:59-60 (`if max_val < 0: max_val = 0`)
    let range = 2.0 * min_clamped.abs().max(max_clamped.abs());
    let (qmin, qmax) = qmin_qmax(bits);
    let denom = (qmax - qmin) as f32;
    let raw = range / denom;
    let raw = raw.max(OBSERVER_MIN_SCALE);
    pow2_round(raw, Pow2Policy::Up)
}

// -----------------------------------------------------------------------------
// KL histogram (activations)
// -----------------------------------------------------------------------------

/// Symmetric per-tensor KL-histogram observer.
///
/// Phase 1 collects min/max via the same code path as
/// [`MinMaxObserver`]; [`Self::finalize_phase1`] freezes
/// `hist_scale = max(|min|, |max|) / hist_bins`. Phase 2 increments a
/// 4096-bin histogram of `|value|`. [`Self::render`] then runs the
/// KL-divergence search at `range.py:224-318`:
///
/// 1. Zero out the bottom `0.2 %` bins, plant a `1` at the boundary
///    (TensorRT's noise-suppression hack — line 281).
/// 2. For each candidate truncation
///    `bin_range ∈ {Q, 2Q, …, hist_bins + Q − 1}` (step `Q =
///    2^(bits−1)`), build the truncated reference distribution `P`
///    and the re-quantized distribution `Q`, normalize, compute KL
///    divergence.
/// 3. Pick the `bin_range*` minimizing KL.
/// 4. `scale = bin_range* · hist_scale / Q`, then pow-2 snap with
///    `ROUND_HALF_UP` (line 317).
#[derive(Debug, Clone)]
pub struct KlHistObserver {
    minmax: MinMaxObserver,
    /// Number of histogram bins. esp-ppq pins this at 4096 for INT8
    /// (`_kl_hist_bins(8) = 4096`); we follow the same rule keyed
    /// off the activation bit width.
    hist_bins: usize,
    /// Histogram bucket size, computed from min/max once Phase 1
    /// finalizes. `None` until then.
    hist_scale: Option<f32>,
    /// Per-bucket counts (i64 to mirror esp-ppq's `int32` accumulator
    /// without overflow worries on long calibration runs).
    histogram: Vec<i64>,
}

impl KlHistObserver {
    /// Construct an observer for an activation that will be quantized
    /// with `num_bits` bits. The histogram size is derived from
    /// [`kl_hist_bins`].
    pub fn new(num_bits: u8) -> Self {
        let hist_bins = kl_hist_bins(num_bits);
        Self {
            minmax: MinMaxObserver::new(),
            hist_bins,
            hist_scale: None,
            histogram: vec![0; hist_bins],
        }
    }
}

impl Observer for KlHistObserver {
    fn observe_minmax(&mut self, values: &[f32]) {
        self.minmax.observe_minmax(values);
    }

    fn finalize_phase1(&mut self) {
        assert!(
            self.minmax.seen,
            "KlHistObserver::finalize_phase1: Phase 1 saw no data"
        );
        // hist_range = max(|min|, |max|) for symmetric quant
        // (range.py:331-334).
        let hist_range = self.minmax.min.abs().max(self.minmax.max.abs());
        let scale = hist_range / self.hist_bins as f32;
        // Defensive: a degenerate (all-zero) activation makes the
        // histogram step zero. We still produce a valid scale via the
        // `OBSERVER_MIN_SCALE` floor at render time, so the histogram
        // step here can be 0 — `observe_hist` is a no-op for an
        // empty range.
        self.hist_scale = Some(scale);
    }

    fn observe_hist(&mut self, values: &[f32]) {
        let hist_scale = self
            .hist_scale
            .expect("KlHistObserver::observe_hist called before finalize_phase1");
        if hist_scale <= 0.0 || self.hist_bins == 0 {
            return;
        }
        // range.py:215 — `torch.histc(abs(value), bins=hist_bins,
        //                              min=0, max=hist_scale * hist_bins)`.
        // torch.histc bucketing: bin = floor((v - min) / (max - min) * bins),
        // with the upper edge inclusive. We replicate exactly: for
        // an absolute value `v` falling in `[0, hist_scale * hist_bins]`,
        // the bin index is `floor(v / hist_scale)`, except `v ==
        // hist_scale * hist_bins` lands in the last bucket.
        let upper = hist_scale * self.hist_bins as f32;
        let last = self.hist_bins - 1;
        for &v in values {
            let absv = v.abs();
            if absv > upper {
                continue; // out of range — torch.histc drops these
            }
            let raw = (absv / hist_scale).floor() as isize;
            let idx = if raw >= self.hist_bins as isize {
                last
            } else if raw < 0 {
                continue;
            } else {
                raw as usize
            };
            self.histogram[idx] += 1;
        }
    }

    fn render(&self, num_bits: u8) -> QuantConfig {
        let hist_scale = self
            .hist_scale
            .expect("KlHistObserver::render called before finalize_phase1");

        // Degenerate range — all values were 0. Emit the smallest
        // representable scale so downstream code does not divide by
        // zero. esp-ppq does the same via `OBSERVER_MIN_SCALE`.
        if hist_scale <= 0.0 || self.histogram.iter().sum::<i64>() == 0 {
            let raw = OBSERVER_MIN_SCALE;
            return QuantConfig::from_pow2_scale(pow2_round(raw, Pow2Policy::HalfUp), num_bits);
        }

        let scale = kl_search_scale(&self.histogram, hist_scale, num_bits);
        let scale = scale.max(OBSERVER_MIN_SCALE);
        let scale = pow2_round(scale, Pow2Policy::HalfUp);
        QuantConfig::from_pow2_scale(scale, num_bits)
    }
}

/// KL-divergence threshold search — the body of
/// `range.py::TorchHistObserver::hist_to_scale_offset` (lines
/// 224-306).
///
/// Inputs are the 4096-bin histogram (or whatever
/// `kl_hist_bins(num_bits)` produced) and the bucket size. Returns
/// the *raw* (pre-pow2-snap) scale.
///
/// Algorithm (mirrors range.py:280-306):
///
/// ```text
/// histogram[: int(0.002 * hist_bins)] = 0
/// histogram[int(0.002 * hist_bins)] = 1
/// quant_bins = 2^(num_bits - 1)
/// for bin_range in [quant_bins, 2*quant_bins, …, hist_bins + quant_bins - 1]:
///     P = histogram[:bin_range], with overflow merged into P[-1]
///     Q = quantize(histogram[:bin_range]) re-expanded back to bin_range
///     normalize, compute KL(P || Q), record (bin_range, KL).
/// pick bin_range minimizing KL.
/// scale = bin_range * hist_scale / quant_bins
/// ```
fn kl_search_scale(histogram_in: &[i64], hist_scale: f32, num_bits: u8) -> f32 {
    let hist_bins = histogram_in.len();
    let mut histogram: Vec<f32> = histogram_in.iter().map(|&x| x as f32).collect();

    // Bottom-bin noise suppression (range.py:281-282). The "0.002 *
    // hist_bins" position becomes a single count; everything below is
    // zeroed.
    let cut = (0.002 * hist_bins as f32) as usize;
    if cut < hist_bins {
        for v in histogram.iter_mut().take(cut) {
            *v = 0.0;
        }
        histogram[cut] = 1.0;
    }

    let quant_bins = 1usize << (num_bits - 1);
    let hist_sum: f64 = histogram.iter().map(|&x| x as f64).sum();
    debug_assert!(hist_sum > 0.0);

    // Precompute the cumulative sum so the "overflow merged into
    // P[-1]" tail can be added in O(1).
    let mut suffix_total = vec![0.0f64; hist_bins + 1];
    for i in (0..hist_bins).rev() {
        suffix_total[i] = suffix_total[i + 1] + histogram[i] as f64;
    }

    let mut best_kl = f64::INFINITY;
    let mut best_bin_range = quant_bins;
    let mut bin_range = quant_bins;
    while bin_range < hist_bins + quant_bins {
        // bin_range is the truncation length. esp-ppq's loop limit
        // is `range(quant_bins, hist_bins + quant_bins - 1, quant_bins)`,
        // which excludes `hist_bins + quant_bins - 1` — Python ranges
        // are end-exclusive. We match that with `<` here.
        let len = bin_range.min(hist_bins);

        // P-distribution: histogram[:bin_range], with the tail past
        // bin_range merged into the last bin (range.py:286-289).
        let mut p_hist = vec![0.0f32; bin_range];
        p_hist[..len].copy_from_slice(&histogram[..len]);
        if bin_range >= hist_bins {
            // Tail is empty — nothing to merge.
        } else {
            let tail = suffix_total[bin_range] as f32;
            p_hist[bin_range - 1] += tail;
        }
        // Normalize — divide by the *full* hist_sum (not p_hist's
        // own sum); range.py:289 uses `hist_sum`.
        for v in &mut p_hist {
            *v = ((*v as f64) / hist_sum) as f32;
        }

        // Q-distribution: re-quantize histogram[:bin_range] into
        // quant_bins, then expand back. range.py:291-301.
        let expand_ratio = bin_range / quant_bins;
        debug_assert!(expand_ratio >= 1);
        let mut q_hist = vec![0.0f32; bin_range];

        // For each of the `quant_bins` super-buckets:
        for qb in 0..quant_bins {
            let start = qb * expand_ratio;
            let end = start + expand_ratio;
            // Slice from the *truncated* histogram (range.py:292
            // copies `histogram[:bin_range]` first).
            let mut sum = 0.0f32;
            let mut positive_cnt = 0.0f32;
            for k in start..end {
                let v = if k < len { histogram[k] } else { 0.0 };
                sum += v;
                if v > 0.0 {
                    positive_cnt += 1.0;
                }
            }
            if positive_cnt == 0.0 {
                // range.py:295-296: positive_cnt[positive_cnt==0]=1.
                // Combined with `q_hist = q_hist * positive_map` the
                // result is zero anyway. We just leave the slice as
                // zeros.
                continue;
            }
            let avg = sum / positive_cnt;
            for k in start..end {
                let original = if k < len { histogram[k] } else { 0.0 };
                if original > 0.0 {
                    q_hist[k] = avg;
                }
            }
        }
        // Normalize q_hist by its own sum (range.py:300).
        let q_sum: f64 = q_hist.iter().map(|&x| x as f64).sum();
        if q_sum > 0.0 {
            for v in &mut q_hist {
                *v = ((*v as f64) / q_sum) as f32;
            }
        }

        // KL(P || Q) — esp-ppq's `torch_KL_divergence` uses base-10
        // logs with an epsilon on both distributions, over every bin.
        let kl = kl_divergence(&p_hist, &q_hist);
        if kl < best_kl {
            best_kl = kl;
            best_bin_range = bin_range;
        }

        bin_range += quant_bins;
    }

    // range.py:306 — `scale = (best_bin_range / hist_bins) * hist_scale * (hist_bins / quant_bins)`
    //              = best_bin_range * hist_scale / quant_bins.
    (best_bin_range as f32) * hist_scale / (quant_bins as f32)
}

/// Discrete `KL(p || q)` matching esp-ppq's
/// `quantization/measure/statistic.py::torch_KL_divergence`:
/// `dot(p, log10(p + 1e-30) - log10(q + 1e-30))`.
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    const EPS: f64 = 1e-30;
    let mut kl = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pi = pi as f64;
        let qi = qi as f64;
        kl += pi * ((pi + EPS).log10() - (qi + EPS).log10());
    }
    kl
}

// -----------------------------------------------------------------------------
// Bias (passive)
// -----------------------------------------------------------------------------

/// Passive bias config: `scale_bias = scale_input · scale_weight`,
/// stored as a wider-int (20-bit on INT8 path, 40-bit on INT16) with
/// an exponent. Mirrors esp-ppq's `_apply_bit_width(bias_config,
/// bias_bits)` (`EspdlQuantizer.py:141`) plus the "passive" derivation
/// rule from `passive.py` (the bias never gets its own observer; its
/// scale falls out of the active configs of its consumers).
///
/// Returns `None` if either input is missing — that situation does
/// not arise on a sound graph but the runner stays defensive.
pub fn derive_bias_config(
    input: Option<&QuantConfig>,
    weight: Option<&QuantConfig>,
    bias_bits: u8,
) -> Option<QuantConfig> {
    let input = input?;
    let weight = weight?;
    let scale = input.scale * weight.scale;
    // Bias is already pow-2 (product of two pow-2 scales is pow-2),
    // so re-deriving the exponent is exact. We do not snap again —
    // esp-ppq doesn't either; the passive scale is taken verbatim.
    Some(QuantConfig::from_pow2_scale(scale, bias_bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_observer_symmetric_int8() {
        // Hand-derived oracle. Range = 2 * max(|-3.0|, |2.0|) = 6.0,
        // qmax-qmin = 255, raw = 6/255 ≈ 0.0235, pow2_up → 0.03125.
        let mut obs = MinMaxObserver::new();
        obs.observe_minmax(&[-3.0, 1.0, 2.0, -0.5]);
        obs.finalize_phase1();
        let cfg = obs.render(8);
        assert_eq!(cfg.scale, 0.031_25); // 2^-5
        assert_eq!(cfg.exponent, -5);
        assert_eq!(cfg.zero_point, 0);
        assert_eq!(cfg.num_bits, 8);
    }

    #[test]
    fn percentile_observer_drops_extreme_tail_like_esp_ppq() {
        // 0.9999 percentile over 51_200 values drops the largest few
        // values just like TorchPercentileObserver. This pins the
        // production Conv-4 failure mode: min-max would choose 2^-9,
        // percentile clips the tail and chooses 2^-10.
        let mut values = vec![0.0_f32; 51_200];
        values[0] = -0.109_550_68;
        values[51_194] = 0.098_768_96;
        for (i, v) in values[51_195..].iter_mut().enumerate() {
            *v = 0.124_65 + i as f32 * 1e-6;
        }
        let mut obs = PercentileObserver::new();
        obs.observe_minmax(&values);
        obs.finalize_phase1();
        let cfg = obs.render(8);
        assert_eq!(cfg.scale, 0.000_976_562_5);
        assert_eq!(cfg.exponent, -10);
    }

    #[test]
    fn minmax_observer_clamps_one_sided_range() {
        // All-positive tensor → min becomes 0 in the clamp at
        // range.py:57-58. range = 2 * 4 = 8, raw = 8/255 ≈ 0.0314,
        // pow2_up → 0.0625.
        let mut obs = MinMaxObserver::new();
        obs.observe_minmax(&[0.5, 1.0, 4.0, 2.0]);
        obs.finalize_phase1();
        let cfg = obs.render(8);
        assert_eq!(cfg.scale, 0.062_5);
    }

    #[test]
    fn kl_observer_degenerate_input() {
        // All zeros → emits the OBSERVER_MIN_SCALE-floored scale.
        let mut obs = KlHistObserver::new(8);
        obs.observe_minmax(&[0.0; 256]);
        obs.finalize_phase1();
        for _ in 0..4 {
            obs.observe_hist(&[0.0; 256]);
        }
        let cfg = obs.render(8);
        assert!(cfg.scale > 0.0);
        assert!(cfg.scale.is_finite());
    }

    #[test]
    fn kl_observer_uniform_input_picks_full_range() {
        // Synthetic uniform input over [-1, 1] → KL search should
        // pick a bin_range close to hist_bins (no truncation needed
        // when the distribution fills the histogram).
        let n = 4096usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f32) / (n as f32 - 1.0);
            samples.push(-1.0 + 2.0 * t);
        }
        let mut obs = KlHistObserver::new(8);
        obs.observe_minmax(&samples);
        obs.finalize_phase1();
        // One pass for Phase 2 is enough — uniform input.
        obs.observe_hist(&samples);
        let cfg = obs.render(8);
        // Activations on [-1,1] → expected scale ≈ 1/128 = 2^-7.
        // The KL search may pull this in by one bin (truncating the
        // tail), so we accept 2^-7 or 2^-8.
        assert!(
            cfg.scale == (1.0_f32 / 128.0) || cfg.scale == (1.0_f32 / 256.0),
            "unexpected KL scale {} (expected 2^-7 or 2^-8)",
            cfg.scale,
        );
        assert_eq!(cfg.zero_point, 0);
        assert_eq!(cfg.num_bits, 8);
    }

    #[test]
    fn bias_config_is_product_of_input_and_weight() {
        let input = QuantConfig::from_pow2_scale(0.0625, 8); // 2^-4
        let weight = QuantConfig::from_pow2_scale(0.015_625, 8); // 2^-6
        let bias = derive_bias_config(Some(&input), Some(&weight), 20).unwrap();
        assert_eq!(bias.scale, 0.0625 * 0.015_625); // 2^-10
        assert_eq!(bias.exponent, -10);
        assert_eq!(bias.num_bits, 20);
    }

    #[test]
    fn qmin_qmax_int8_int16_int20() {
        assert_eq!(qmin_qmax(8), (-128, 127));
        assert_eq!(qmin_qmax(16), (-32_768, 32_767));
        assert_eq!(qmin_qmax(20), (-524_288, 524_287));
    }

    #[test]
    fn kl_hist_bins_int8_int16() {
        assert_eq!(kl_hist_bins(8), 4096);
        assert_eq!(kl_hist_bins(16), 1_048_576);
    }
}
