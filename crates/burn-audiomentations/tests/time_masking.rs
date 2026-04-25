//! Unit tests for `burn_audiomentations::TimeMasking`.
//!
//! The transform's semantics are easiest to pin down with two targeted
//! probes:
//!
//! * `probability = 0.0` → identity. (All intervals collapse to length 0.)
//! * `probability = 1.0` with a small deterministic seed and small
//!   `max_width_samples` → at least one splice actually fires, and the
//!   output both (a) has exactly as many samples as the input and (b)
//!   leaves a trailing zero-padded region.
//!
//! We deliberately avoid a tight parity test here: SpliceOut's behaviour
//! depends on the interleaving of multiple RNG draws, and reproducing
//! that sequence is not worth the complexity. The algorithmic-shape tests
//! below are the guardrails we actually need.

mod common;

use burn_audiomentations::{TimeMasking, Transform, TransformRng};
use common::{client, max_abs_diff, read_tensor, synth_reals, upload_2d, Runtime};

fn build_batch(batch: usize, time: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * time];
    for b in 0..batch {
        out[b * time..(b + 1) * time].copy_from_slice(&synth_reals(time, 500 + b as u64));
    }
    out
}

#[test]
fn probability_zero_is_identity() {
    let client = client();
    let batch = 2;
    let time = 1024;
    let sig = build_batch(batch, time);

    let tm = TimeMasking::new(2, 400, 0.0);
    let mut rng = TransformRng::new(7);
    let out = <TimeMasking as Transform<Runtime>>::apply(
        &tm,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let err = max_abs_diff(&out_host, &sig);
    assert!(err < 1e-6, "p=0 should be identity, got {err}");
}

#[test]
fn output_length_matches_input() {
    let client = client();
    let batch = 4;
    let time = 2048;
    let sig = build_batch(batch, time);

    let tm = TimeMasking::new(2, 400, 1.0);
    let mut rng = TransformRng::new(12345);
    let out = <TimeMasking as Transform<Runtime>>::apply(
        &tm,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    assert_eq!(
        out_host.len(),
        batch * time,
        "output should preserve shape (batch * time)",
    );
}

#[test]
fn output_has_a_zeroed_tail_when_intervals_fire() {
    // With p=1.0 and large max_width, the cumulative splice across 2
    // intervals strips at least `max_width` samples in expectation; the
    // tail must contain a zero-padded region of at least 1 sample.
    let client = client();
    let batch = 2;
    let time = 8192;
    let sig = build_batch(batch, time);

    let tm = TimeMasking::new(2, 1024, 1.0);
    let mut rng = TransformRng::new(55);
    let out = <TimeMasking as Transform<Runtime>>::apply(
        &tm,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);

    for b in 0..batch {
        // Scan from the tail: at least one sample should be exactly zero
        // (zero-pad comes from the kernel's `else` branch, no f32 noise).
        let row = &out_host[b * time..(b + 1) * time];
        let tail_zero_count = row.iter().rev().take_while(|v| **v == 0.0).count();
        assert!(
            tail_zero_count > 0,
            "row {b} expected a zero-padded tail, but last sample = {}",
            row[time - 1],
        );
    }
}

#[test]
fn leading_samples_preserved_up_to_first_splice() {
    // The splice kernel writes `input[t]` verbatim for t < start of the
    // first interval. If the first interval's start is > 0, at least
    // sample 0 and maybe more must match the input exactly.
    //
    // We rely on the host RNG producing non-zero start indices on the
    // overwhelming majority of draws — `start ∈ [0, time - length)` —
    // but we're tolerant: if start == 0 happens by chance we simply skip
    // the assertion by falling through and relying on the structural
    // tests above.
    let client = client();
    let batch = 8;
    let time = 1024;
    let sig = build_batch(batch, time);

    let tm = TimeMasking::new(1, 100, 1.0);
    let mut rng = TransformRng::new(31337);
    let out = <TimeMasking as Transform<Runtime>>::apply(
        &tm,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);

    let mut rows_with_preserved_prefix = 0;
    for b in 0..batch {
        let inp = &sig[b * time..(b + 1) * time];
        let got = &out_host[b * time..(b + 1) * time];
        if (got[0] - inp[0]).abs() < 1e-6 {
            rows_with_preserved_prefix += 1;
        }
    }
    // With batch=8 and start drawn uniformly from [0, 1024 - len), the
    // overwhelming majority of rows will have start > 0. Demand at least
    // half match — this is a sanity test, not a parity test.
    assert!(
        rows_with_preserved_prefix >= batch / 2,
        "expected ≥ {} rows to preserve sample 0, got {rows_with_preserved_prefix}",
        batch / 2,
    );
}
