//! Step 4 — `MinMaxObserver` parity test.
//!
//! Feeds a known weight tensor through [`MinMaxObserver`] and asserts
//! the observed `(scale, exponent)` against a hand-computed oracle
//! derived from
//! `_reference/esp-ppq/esp_ppq/quantization/observer/range.py:74-89`.
//!
//! No external artifacts, no production model — the oracle is
//! computed inline from the formula in the report (§5).

use burn_espdl_export::{MinMaxObserver, Observer};

/// Hand-computed oracle:
///
/// ```text
///   range = 2 * max(|min|, |max|)
///   raw   = range / (qmax - qmin)
///   scale = pow2_round_up(raw)
///   exp   = int(log2(scale))
/// ```
///
/// For symmetric INT8 (`qmax-qmin = 255`).
fn oracle_int8(values: &[f32]) -> (f32, i32) {
    let mut lo = values
        .iter()
        .copied()
        .fold(f32::INFINITY, |a, b| if b < a { b } else { a });
    let mut hi = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    if lo > 0.0 {
        lo = 0.0;
    }
    if hi < 0.0 {
        hi = 0.0;
    }
    let range = 2.0 * lo.abs().max(hi.abs());
    let raw = range / 255.0;
    // round-up pow-2 snap.
    let snapped = if raw == 0.0 {
        0.0
    } else {
        let log = (raw as f64).log2().ceil() as i32;
        2.0_f32.powi(log)
    };
    (snapped, (snapped as f64).log2().round() as i32)
}

#[test]
fn matches_oracle_on_symmetric_weight() {
    // Mixed positive/negative weights.
    let weights = vec![-0.4_f32, -0.1, 0.0, 0.05, 0.2, 0.3];
    let mut obs = MinMaxObserver::new();
    obs.observe_minmax(&weights);
    obs.finalize_phase1();
    let cfg = obs.render(8);
    let (oracle_scale, oracle_exp) = oracle_int8(&weights);
    assert_eq!(cfg.scale, oracle_scale, "scale mismatch (cfg={cfg:?})");
    assert_eq!(cfg.exponent, oracle_exp, "exponent mismatch (cfg={cfg:?})");
    assert_eq!(cfg.zero_point, 0);
    assert_eq!(cfg.num_bits, 8);
}

#[test]
fn matches_oracle_on_all_positive_weight() {
    // After clamping, min becomes 0; range = 2 * 0.5 = 1.0.
    let weights = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
    let mut obs = MinMaxObserver::new();
    obs.observe_minmax(&weights);
    obs.finalize_phase1();
    let cfg = obs.render(8);
    let (oracle_scale, oracle_exp) = oracle_int8(&weights);
    assert_eq!(cfg.scale, oracle_scale);
    assert_eq!(cfg.exponent, oracle_exp);
}

#[test]
fn matches_oracle_on_all_negative_weight() {
    // After clamping, max becomes 0; range = 2 * 0.7 = 1.4.
    let weights = vec![-0.7_f32, -0.5, -0.3, -0.1];
    let mut obs = MinMaxObserver::new();
    obs.observe_minmax(&weights);
    obs.finalize_phase1();
    let cfg = obs.render(8);
    let (oracle_scale, oracle_exp) = oracle_int8(&weights);
    assert_eq!(cfg.scale, oracle_scale);
    assert_eq!(cfg.exponent, oracle_exp);
}

#[test]
fn streamed_observation_matches_one_shot() {
    // Feed in chunks; observed range should equal a single-pass call.
    let weights = vec![-0.3_f32, 0.6, 0.1, -0.05, 0.4, -0.2];
    let mut a = MinMaxObserver::new();
    a.observe_minmax(&weights);
    a.finalize_phase1();
    let cfg_oneshot = a.render(8);

    let mut b = MinMaxObserver::new();
    b.observe_minmax(&weights[..2]);
    b.observe_minmax(&weights[2..4]);
    b.observe_minmax(&weights[4..]);
    b.finalize_phase1();
    let cfg_streamed = b.render(8);

    assert_eq!(cfg_oneshot.scale, cfg_streamed.scale);
    assert_eq!(cfg_oneshot.exponent, cfg_streamed.exponent);
}

#[test]
fn renders_for_int16_too() {
    // Same weights as the symmetric INT8 test, but bit width 16.
    // qmax-qmin = 65535, range stays the same; the smaller divisor
    // shrinks the scale → smaller exponent.
    let weights = vec![-0.4_f32, -0.1, 0.0, 0.05, 0.2, 0.3];
    let mut obs = MinMaxObserver::new();
    obs.observe_minmax(&weights);
    obs.finalize_phase1();
    let cfg8 = obs.render(8);
    let cfg16 = obs.render(16);
    // 8 → 16 widens by 8 bits → scale halves 8x.
    assert!(cfg16.scale < cfg8.scale, "INT16 scale should be smaller");
    assert!(cfg16.exponent < cfg8.exponent);
}
