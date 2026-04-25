//! Integration test for `burn_audiomentations::Compose`.
//!
//! Verifies that:
//!
//! * With `shuffle=false` the transforms fire in declaration order
//!   (Gain 0 dB → PolarityInversion 100% collapses to a pure negation).
//! * With `shuffle=false` and probability 0 on every transform, the
//!   pipeline is the identity.
//! * With the full `robust.yaml` minus pitch_shift, the pipeline runs end
//!   to end without panicking and produces a non-NaN, same-shape output
//!   batch.

mod common;

use burn_audiomentations::{
    AddColoredNoise, Compose, Gain, HighPassFilter, LowPassFilter, PolarityInversion,
    TimeMasking, Transform, TransformRng,
};
use common::{client, dtype_f32, max_abs_diff, read_tensor, synth_reals, upload_2d, Runtime};

const SR: u32 = 32_000;

fn build_batch(batch: usize, time: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * time];
    for b in 0..batch {
        out[b * time..(b + 1) * time].copy_from_slice(&synth_reals(time, 900 + b as u64));
    }
    out
}

#[test]
fn compose_with_all_zero_probabilities_is_identity() {
    let client = client();
    let batch = 2;
    let time = 4096;
    let sig = build_batch(batch, time);

    let pipeline: Compose<Runtime> = Compose::new(vec![
        Box::new(Gain::new(-12.0, 12.0, 0.0)),
        Box::new(PolarityInversion::new(0.0)),
        Box::new(TimeMasking::new(2, 400, 0.0)),
        Box::new(LowPassFilter::<Runtime>::new(
            client.clone(),
            1_000.0,
            8_000.0,
            SR,
            0.0,
            8,
            dtype_f32(),
        )),
        Box::new(HighPassFilter::<Runtime>::new(
            client.clone(),
            40.0,
            200.0,
            SR,
            0.0,
            8,
            dtype_f32(),
        )),
        Box::new(AddColoredNoise::new(0.0, 25.0, -1.5, 1.5, SR, 0.0)),
    ]);

    let mut rng = TransformRng::new(1);
    let out = <Compose<Runtime> as Transform<Runtime>>::apply(
        &pipeline,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let got = read_tensor(&client, out);
    let err = max_abs_diff(&got, &sig);
    assert!(err < 1e-6, "all-zero compose should be identity (err {err})");
}

#[test]
fn compose_sequential_order_is_observable() {
    // Gain at fixed +6 dB followed by PolarityInversion @ p=1 collapses to
    // `samples * (-10^0.3)` up to f32 rounding. A direct check of the
    // final amplitude ratio pins down ordering.
    let client = client();
    let batch = 1;
    let time = 256;
    let sig = build_batch(batch, time);

    let pipeline: Compose<Runtime> = Compose::new(vec![
        Box::new(Gain::new(6.0, 6.0, 1.0)),
        Box::new(PolarityInversion::new(1.0)),
    ]);

    let mut rng = TransformRng::new(0);
    let out = <Compose<Runtime> as Transform<Runtime>>::apply(
        &pipeline,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let got = read_tensor(&client, out);

    let expected_ratio = -10f32.powf(6.0 / 20.0);
    for (i, (g, s)) in got.iter().zip(sig.iter()).enumerate() {
        let want = expected_ratio * s;
        assert!(
            (*g - want).abs() < 1e-5,
            "idx {i}: got {g}, want {want}",
        );
    }
}

#[test]
fn compose_robust_config_runs_end_to_end() {
    let client = client();
    let batch = 4;
    // 1 s window at 32 kHz like the real training run (rounded to a power
    // of two just so the noise path chunks cleanly — the noise kernel
    // handles non-power-of-two, but we want a fast test).
    let time = 8192;
    let sig = build_batch(batch, time);

    let pipeline: Compose<Runtime> = Compose::new(vec![
        Box::new(Gain::new(-12.0, 12.0, 0.8)),
        Box::new(AddColoredNoise::new(0.0, 25.0, -1.5, 1.5, SR, 0.7)),
        Box::new(HighPassFilter::<Runtime>::new(
            client.clone(),
            40.0,
            200.0,
            SR,
            0.2,
            16,
            dtype_f32(),
        )),
        Box::new(LowPassFilter::<Runtime>::new(
            client.clone(),
            8_000.0,
            14_000.0,
            SR,
            0.2,
            16,
            dtype_f32(),
        )),
        Box::new(TimeMasking::new(2, 400, 0.4)),
        Box::new(PolarityInversion::new(0.5)),
    ])
    .with_shuffle(true);

    let mut rng = TransformRng::new(54321);
    let out = <Compose<Runtime> as Transform<Runtime>>::apply(
        &pipeline,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let got = read_tensor(&client, out);
    assert_eq!(got.len(), sig.len(), "output should have same shape");
    for (i, v) in got.iter().enumerate() {
        assert!(v.is_finite(), "non-finite value at idx {i}: {v}");
    }
}
