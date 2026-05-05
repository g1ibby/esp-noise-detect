//! Step 4 — `KlHistObserver` parity test.
//!
//! Constructs synthetic activations whose distribution shape forces
//! known answers out of the KL search, and asserts the observed
//! `(scale, exponent)` against the predicted boundary.
//!
//! Why no esp-ppq oracle JSON committed yet: generating one requires
//! running esp-ppq's `RuntimeCalibrationPass` against a tiny inline
//! ONNX graph offline (Docker), which is the Step-4 follow-up the
//! task spec leaves open. This test pins the *behavioural*
//! invariants — uniform → no truncation, two-bump → KL truncates the
//! tail bump — so a regression in the KL search surfaces on
//! `cargo test` without external setup.

use burn_espdl_export::{KlHistObserver, Observer};

/// Uniform input on `[-r, r]` should make the KL search pick a
/// `bin_range` close to `hist_bins`: there's no tail to truncate.
/// Resulting scale ≈ `r / quant_bins`, snapped to the next pow-2
/// `≥ r / 128` for INT8.
#[test]
fn uniform_input_picks_full_range() {
    let n = 8192usize;
    let r = 1.0_f32;
    let samples: Vec<f32> = (0..n)
        .map(|i| -r + 2.0 * r * (i as f32) / (n as f32 - 1.0))
        .collect();

    let mut obs = KlHistObserver::new(8);
    obs.observe_minmax(&samples);
    obs.finalize_phase1();
    obs.observe_hist(&samples);
    let cfg = obs.render(8);

    // Quant bins for INT8 = 128. raw scale ≈ 1/128 = 2^-7.
    // KL might pull the bin_range in by one step (e.g. truncate the
    // last bucket) → 2^-8. Either is acceptable.
    let allowed: [f32; 2] = [1.0_f32 / 128.0, 1.0_f32 / 256.0];
    assert!(
        allowed.contains(&cfg.scale),
        "uniform input KL scale {} not in {:?}",
        cfg.scale,
        allowed,
    );
}

/// Heavy concentration near zero with a sparse outlier far in the
/// tail. KL should truncate the tail (bin_range < hist_bins) so the
/// resulting scale is much smaller than the raw min/max scale.
#[test]
fn long_tailed_input_truncates_tail() {
    let mut samples: Vec<f32> = Vec::with_capacity(10_001);
    // 10 000 small samples in [-0.05, 0.05].
    for i in 0..10_000 {
        let t = (i as f32) / 9_999.0;
        samples.push(-0.05 + 0.1 * t);
    }
    // One huge outlier at +5.0.
    samples.push(5.0);

    let mut obs_minmax = burn_espdl_export::MinMaxObserver::new();
    obs_minmax.observe_minmax(&samples);
    obs_minmax.finalize_phase1();
    let minmax_cfg = obs_minmax.render(8);

    let mut obs = KlHistObserver::new(8);
    obs.observe_minmax(&samples);
    obs.finalize_phase1();
    obs.observe_hist(&samples);
    let kl_cfg = obs.render(8);

    // KL should pick a much smaller scale than min-max — the outlier
    // is suppressed because it carries a single count.
    assert!(
        kl_cfg.scale < minmax_cfg.scale,
        "KL scale {} should be smaller than min-max scale {}",
        kl_cfg.scale,
        minmax_cfg.scale,
    );
}

/// All-zero input degenerates: the histogram is empty, the
/// observer must still emit a valid (positive, finite, pow-2) scale
/// rather than NaN/0.
#[test]
fn all_zero_input_emits_positive_scale() {
    let samples = vec![0.0_f32; 4096];
    let mut obs = KlHistObserver::new(8);
    obs.observe_minmax(&samples);
    obs.finalize_phase1();
    obs.observe_hist(&samples);
    let cfg = obs.render(8);
    assert!(cfg.scale > 0.0);
    assert!(cfg.scale.is_finite());
    assert_eq!(cfg.zero_point, 0);
    assert_eq!(cfg.num_bits, 8);
}

/// Symmetry property: feeding `x` and feeding `-x` produces the same
/// scale (the observer takes `|value|` before histogramming).
#[test]
fn observer_is_sign_invariant() {
    let n = 4096usize;
    let pos: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let neg: Vec<f32> = pos.iter().map(|x| -x).collect();

    let mut a = KlHistObserver::new(8);
    a.observe_minmax(&pos);
    a.finalize_phase1();
    a.observe_hist(&pos);
    let cfg_a = a.render(8);

    let mut b = KlHistObserver::new(8);
    b.observe_minmax(&neg);
    b.finalize_phase1();
    b.observe_hist(&neg);
    let cfg_b = b.render(8);

    assert_eq!(cfg_a.scale, cfg_b.scale, "scale should be sign-invariant");
    assert_eq!(cfg_a.exponent, cfg_b.exponent);
}
