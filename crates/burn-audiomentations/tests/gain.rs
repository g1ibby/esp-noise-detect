//! Unit tests for `burn_audiomentations::Gain`.
//!
//! Covers: probability = 1.0 (everything scaled), probability = 0.0
//! (identity), constant-gain range (zero variance across batch), and
//! per-example parameter determinism across repeated calls with the same
//! seed.

mod common;

use burn_audiomentations::{Gain, Transform, TransformRng};
use common::{client, max_abs_diff, read_tensor, rms, synth_reals, upload_2d, Runtime};

fn build_batch(batch: usize, time: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * time];
    for b in 0..batch {
        out[b * time..(b + 1) * time].copy_from_slice(&synth_reals(time, 100 + b as u64));
    }
    out
}

#[test]
fn probability_zero_is_identity() {
    let client = client();
    let batch = 4;
    let time = 512;
    let sig = build_batch(batch, time);

    let g = Gain::new(-12.0, 12.0, 0.0);
    let mut rng = TransformRng::new(42);
    let out = <Gain as Transform<Runtime>>::apply(&g, upload_2d(&client, &sig, batch, time), &mut rng);
    let out_host = read_tensor(&client, out);

    let err = max_abs_diff(&out_host, &sig);
    assert!(err < 1e-6, "p=0 should be identity, got max-abs-diff = {err}");
}

#[test]
fn probability_one_scales_every_row() {
    let client = client();
    let batch = 6;
    let time = 256;
    let sig = build_batch(batch, time);

    let g = Gain::new(-6.0, 6.0, 1.0);
    let mut rng = TransformRng::new(7);
    let out = <Gain as Transform<Runtime>>::apply(&g, upload_2d(&client, &sig, batch, time), &mut rng);
    let out_host = read_tensor(&client, out);

    // Each row should be a uniform scaling of the original. Compare row
    // RMS ratios and confirm all land in the expected amplitude range
    // (2^(±6/20) ≈ 0.501 – 1.995).
    for b in 0..batch {
        let r_in = rms(&sig[b * time..(b + 1) * time]);
        let r_out = rms(&out_host[b * time..(b + 1) * time]);
        let ratio = r_out / r_in;
        assert!(
            (0.49..=2.01).contains(&ratio),
            "row {b}: out/in RMS ratio {ratio} out of expected [0.49, 2.01]",
        );
    }
}

#[test]
fn constant_gain_produces_deterministic_scaling() {
    let client = client();
    let batch = 3;
    let time = 128;
    let sig = build_batch(batch, time);

    let g = Gain::new(6.0, 6.0, 1.0); // constant +6 dB
    let mut rng = TransformRng::new(0);
    let out = <Gain as Transform<Runtime>>::apply(&g, upload_2d(&client, &sig, batch, time), &mut rng);
    let out_host = read_tensor(&client, out);

    let expected_ratio = 10f32.powf(6.0 / 20.0);
    for i in 0..sig.len() {
        let got = out_host[i];
        let want = sig[i] * expected_ratio;
        assert!(
            (got - want).abs() < 1e-5,
            "idx {i}: got {got}, want {want}",
        );
    }
}

#[test]
fn same_seed_same_output() {
    let client = client();
    let batch = 5;
    let time = 200;
    let sig = build_batch(batch, time);

    let g = Gain::new(-12.0, 12.0, 1.0);
    let mut rng_a = TransformRng::new(2024);
    let out_a = <Gain as Transform<Runtime>>::apply(
        &g,
        upload_2d(&client, &sig, batch, time),
        &mut rng_a,
    );
    let a = read_tensor(&client, out_a);

    let mut rng_b = TransformRng::new(2024);
    let out_b = <Gain as Transform<Runtime>>::apply(
        &g,
        upload_2d(&client, &sig, batch, time),
        &mut rng_b,
    );
    let b = read_tensor(&client, out_b);

    assert_eq!(a, b, "same seed must produce identical output");
}
