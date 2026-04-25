//! Unit tests for `burn_audiomentations::PolarityInversion`.

mod common;

use burn_audiomentations::{PolarityInversion, Transform, TransformRng};
use common::{client, max_abs_diff, read_tensor, synth_reals, upload_2d, Runtime};

fn build_batch(batch: usize, time: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * time];
    for b in 0..batch {
        out[b * time..(b + 1) * time].copy_from_slice(&synth_reals(time, 300 + b as u64));
    }
    out
}

#[test]
fn probability_one_flips_every_row() {
    let client = client();
    let batch = 4;
    let time = 128;
    let sig = build_batch(batch, time);

    let p = PolarityInversion::new(1.0);
    let mut rng = TransformRng::new(1);
    let out = <PolarityInversion as Transform<Runtime>>::apply(
        &p,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let negated: Vec<f32> = sig.iter().map(|v| -v).collect();
    let err = max_abs_diff(&out_host, &negated);
    assert!(err < 1e-6);
}

#[test]
fn probability_zero_is_identity() {
    let client = client();
    let batch = 3;
    let time = 64;
    let sig = build_batch(batch, time);

    let p = PolarityInversion::new(0.0);
    let mut rng = TransformRng::new(9);
    let out = <PolarityInversion as Transform<Runtime>>::apply(
        &p,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let err = max_abs_diff(&out_host, &sig);
    assert!(err < 1e-6);
}

#[test]
fn mixed_probability_rows_either_flip_or_identity() {
    let client = client();
    let batch = 16;
    let time = 64;
    let sig = build_batch(batch, time);

    let p = PolarityInversion::new(0.5);
    let mut rng = TransformRng::new(2025);
    let out = <PolarityInversion as Transform<Runtime>>::apply(
        &p,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);

    // Every row must be either identity or a full negation. No partial
    // flips or scaling allowed.
    for b in 0..batch {
        let inp = &sig[b * time..(b + 1) * time];
        let got = &out_host[b * time..(b + 1) * time];
        let neg: Vec<f32> = inp.iter().map(|v| -v).collect();
        let d_ident = max_abs_diff(got, inp);
        let d_neg = max_abs_diff(got, &neg);
        assert!(
            d_ident < 1e-6 || d_neg < 1e-6,
            "row {b}: output is neither identity (err {d_ident}) nor negation (err {d_neg})",
        );
    }
}
