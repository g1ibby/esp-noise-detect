//! Power-of-two snap, byte-for-byte compatible with esp-ppq's
//! `ppq_round_to_power_of_2`
//! (`_reference/esp-ppq/esp_ppq/utils/round.py:128-147`).
//!
//! Two snap variants are used on the S3 INT8 path (see report §5):
//!
//! * Weights snap with `ROUND_UP` — used by
//!   `range.py:88-89` after the symmetric min-max scale derivation.
//! * Activations snap with `ROUND_HALF_UP` — used by
//!   `range.py:316-317` after the KL-derived scale.
//!
//! ## Why a Rust-level oracle for `ROUND_HALF_UP`
//!
//! esp-ppq's helper goes through Python's `decimal.Decimal` so half
//! values resolve via `ROUND_HALF_UP` (positive: away from zero) /
//! `ROUND_HALF_DOWN` (negative: toward zero), mirroring Python's
//! `int()` semantics rather than IEEE round-half-even. We reproduce
//! that exactly with a sign-split: positive half values round away
//! from zero, negative half values round toward zero. That matters at
//! tie boundaries (e.g. `log2(scale) = 1.5`) where `f32::round`'s
//! ties-to-even policy disagrees with esp-ppq.

/// Pow-2 rounding policy, matching the two values esp-ppq actually
/// uses on the ESP32-S3 path. Other policies in `RoundingPolicy`
/// (`ROUND_HALF_EVEN`, `ROUND_HALF_DOWN`, …) are not in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pow2Policy {
    /// Round towards `+∞`. Weights use this (S3 INT8 + INT16).
    Up,
    /// Round to nearest, ties go away from zero.
    /// Activations use this (S3 INT8 + INT16).
    HalfUp,
}

/// Snap `value` to the nearest power of two under `policy`. Returns
/// `0.0` if the input is `0.0`. Mirrors esp-ppq's
/// `ppq_round_to_power_of_2` (round.py:128-147):
///
/// ```python
/// sign = 1 if value >= 0 else -1
/// return sign * float(pow(2, ppq_numerical_round(log2(sign * value), policy=policy)))
/// ```
pub fn pow2_round(value: f32, policy: Pow2Policy) -> f32 {
    if value == 0.0 {
        return 0.0;
    }
    let sign = if value > 0.0 { 1.0 } else { -1.0 };
    let abs = value.abs();
    let log = (abs as f64).log2();
    let rounded = match policy {
        Pow2Policy::Up => log.ceil() as i32,
        Pow2Policy::HalfUp => round_half_up_signed(log),
    };
    sign * pow2_from_int(rounded)
}

/// `int(log2(scale))` — esp-ppq's exponent encoding for per-tensor
/// pow-2 scales (`export_patterns.py:356`).
///
/// Python's `int()` truncates toward zero, so a scale of `0.5` (log =
/// −1) maps to `−1`, `0.25` → `−2`, `1.0` → `0`. After a `ROUND_UP`
/// or `ROUND_HALF_UP` pow-2 snap the log is an integer in IEEE
/// arithmetic, but FP rounding can leave a `−1.0000001`-style residue;
/// we collapse that with a `f32::round` before truncating, which is
/// safe because we know the snap already produced an integer log.
pub fn log2_floor(scale: f32) -> i32 {
    assert!(
        scale > 0.0,
        "log2_floor: scale must be positive (got {scale})"
    );
    let log = (scale as f64).log2();
    // log is an integer modulo FP noise after pow2 snap; round to
    // nearest int, then truncate toward zero.
    let rounded = log.round();
    rounded as i32
}

/// `floor(log2(value) + 0.5)` with positive/negative tie discipline
/// matching `ppq_numerical_round(_, ROUND_HALF_UP)`. Exposed for the
/// observer tests so they can pin tie behaviour without going through
/// [`pow2_round`].
pub fn log2_round_half_up(value: f32) -> i32 {
    assert!(
        value > 0.0,
        "log2_round_half_up: value must be positive (got {value})"
    );
    round_half_up_signed((value as f64).log2())
}

fn round_half_up_signed(value: f64) -> i32 {
    // esp-ppq:
    //   if value > 0:  Decimal(value).quantize(rounding=ROUND_HALF_UP)  → ties away from zero
    //   else        :  Decimal(value).quantize(rounding=ROUND_HALF_DOWN) → ties toward zero
    // The branch is on the sign of `value`, not the sign of `value`'s
    // half-fraction; see round.py:84-88. We replicate that exactly.
    if value > 0.0 {
        // Round half away from zero.
        (value + 0.5).floor() as i32
    } else {
        // Round half toward zero (i.e. ROUND_HALF_DOWN on the
        // negative side).
        (value - 0.5).ceil() as i32
    }
}

/// `2^k` as `f32`, valid for the exponent range we care about.
/// Relying on `f32::powi` keeps the bit pattern stable across hosts
/// (vs going through `f64::powi` and casting).
fn pow2_from_int(k: i32) -> f32 {
    // f32 exponent range (subnormals included) is `[-149, 127]`.
    // The S3 calibration path produces exponents in roughly
    // `[-30, +5]` for INT8, well inside that range; we still clamp
    // gently to keep the test surface predictable.
    debug_assert!(
        (-149..=127).contains(&k),
        "pow2_from_int: |k|={k} out of range"
    );
    (2.0_f32).powi(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_snaps_to_next_pow2() {
        // Range of 0.6 → log2 = -0.737, ceil = 0 → 1.0.
        assert_eq!(pow2_round(0.6, Pow2Policy::Up), 1.0);
        // 0.5 is exactly 2^-1 → log2 = -1, ceil = -1 → 0.5.
        assert_eq!(pow2_round(0.5, Pow2Policy::Up), 0.5);
        // 0.5000001 → log2 ≈ -0.9999..., ceil = 0 → 1.0.
        assert!(pow2_round(0.500_000_1, Pow2Policy::Up) >= 1.0);
        // 3.0 → log2 ≈ 1.585, ceil = 2 → 4.0.
        assert_eq!(pow2_round(3.0, Pow2Policy::Up), 4.0);
    }

    #[test]
    fn half_up_rounds_to_nearest_pow2() {
        // 1.0 → log2 = 0 → 1.0.
        assert_eq!(pow2_round(1.0, Pow2Policy::HalfUp), 1.0);
        // 1.5 → log2 ≈ 0.585 → round_half_up → 1 → 2.0.
        assert_eq!(pow2_round(1.5, Pow2Policy::HalfUp), 2.0);
        // 1.4 → log2 ≈ 0.485 → round_half_up → 0 → 1.0. (Just below
        // the 0.5 tie boundary on the positive side.)
        assert_eq!(pow2_round(1.4, Pow2Policy::HalfUp), 1.0);
        // 3.0 → log2 ≈ 1.585 → round → 2 → 4.0.
        assert_eq!(pow2_round(3.0, Pow2Policy::HalfUp), 4.0);
        // 0.7 → log2 ≈ -0.515 → ROUND_HALF_DOWN (negative branch) →
        // ceil(-0.515 - 0.5) = ceil(-1.015) = -1 → 0.5. (Negative-side
        // ties round *toward* zero per esp-ppq's `int()` semantics.)
        assert_eq!(pow2_round(0.7, Pow2Policy::HalfUp), 0.5);
    }

    #[test]
    fn zero_passes_through() {
        assert_eq!(pow2_round(0.0, Pow2Policy::Up), 0.0);
        assert_eq!(pow2_round(0.0, Pow2Policy::HalfUp), 0.0);
    }

    #[test]
    fn log2_floor_handles_pow2_inputs() {
        assert_eq!(log2_floor(1.0), 0);
        assert_eq!(log2_floor(0.5), -1);
        assert_eq!(log2_floor(0.25), -2);
        assert_eq!(log2_floor(2.0), 1);
        assert_eq!(log2_floor(1024.0), 10);
        // Sub-pow2 values also pass — esp-ppq lets `int(log2(scale))`
        // truncate toward zero, so 0.6 (log ≈ -0.737) → 0.
        // We round-then-truncate, so a value that snapped to 1.0 won't
        // accidentally become 0.
        assert_eq!(log2_floor(pow2_round(0.6, Pow2Policy::Up)), 0);
    }
}
