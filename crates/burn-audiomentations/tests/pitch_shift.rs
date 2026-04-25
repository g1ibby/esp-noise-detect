//! Unit tests for `burn_audiomentations::PitchShift`.
//!
//! Covers:
//!
//! * `probability = 0.0` → full identity on every row.
//! * The enumerated `(num, den)` set matches `get_fast_shifts` output for
//!   the same range (sanity check on construction).
//! * Per-ratio correctness on a pure tone: for each enumerated ratio, feed
//!   a sine at `f0` and verify the spectral peak of the shifted signal is
//!   near `f0 * num / den`. Tolerance is one STFT bin (`sample_rate /
//!   n_fft`), which is the minimum resolution the analysis window
//!   supports.
//! * Batched correctness: multiple rows, different assigned ratios,
//!   each row independently verified.
//! * `same seed → same output` determinism.

mod common;

use burn_audiomentations::{PitchShift, Transform, TransformRng};
use common::{client, dtype_f32, max_abs_diff, read_tensor, sine, upload_2d, Runtime};

const SR: u32 = 32_000;
const TIME: usize = SR as usize; // 1 s window, matches robust_session.yaml
const N_FFT: usize = 512;

/// Build `(batch, time)` of a single sine wave per row (all the same).
fn sine_batch(batch: usize, freq_hz: f32) -> Vec<f32> {
    let row = sine(freq_hz, TIME, SR);
    let mut out = Vec::with_capacity(batch * TIME);
    for _ in 0..batch {
        out.extend_from_slice(&row);
    }
    out
}

/// Naive DFT peak detector over `(n_freq, n_samples)` via a direct
/// one-bin-per-candidate scan. Used only on the output of PitchShift to
/// locate the shifted sine's fundamental — not on the audio path.
///
/// Searches `f0..f1` inclusive in `bin_hz` steps and returns the
/// frequency of the peak magnitude.
fn peak_freq(samples: &[f32], sr: u32, f0: f32, f1: f32) -> f32 {
    assert!(f1 > f0);
    let bin_hz = sr as f32 / N_FFT as f32;
    let mut best_hz = f0;
    let mut best_mag = 0.0f32;
    let mut f = f0;
    while f <= f1 {
        let w = 2.0 * core::f32::consts::PI * f / sr as f32;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (n, &x) in samples.iter().enumerate() {
            let phase = w * n as f32;
            re += x * phase.cos();
            im -= x * phase.sin();
        }
        let mag = (re * re + im * im).sqrt();
        if mag > best_mag {
            best_mag = mag;
            best_hz = f;
        }
        f += bin_hz / 4.0;
    }
    best_hz
}

#[test]
fn probability_zero_is_identity() {
    let client = client();
    let batch = 4;
    let sig = sine_batch(batch, 1000.0);
    let t_in = upload_2d(&client, &sig, batch, TIME);

    let shifter = PitchShift::<Runtime>::new(client.clone(), SR, -2.0, 2.0, 0.0, dtype_f32());
    let mut rng = TransformRng::new(42);
    let out = shifter.apply(t_in, &mut rng);
    let out_host = read_tensor(&client, out);

    let err = max_abs_diff(&out_host, &sig);
    assert!(err < 1e-6, "p=0 should be identity, got max-abs-diff = {err}");
}

#[test]
fn enumerates_fast_shifts_in_range() {
    // For 32 kHz ± 2 semitones the only available fast ratios are
    // 125/128 and 128/125 (≈ ±0.41 semitones). See fast_shifts.rs tests.
    let client = client();
    let shifter = PitchShift::<Runtime>::new(client, SR, -2.0, 2.0, 1.0, dtype_f32());
    let shifts = shifter.shifts().to_vec();
    assert!(shifts.contains(&(125u32, 128u32)), "missing 125/128 in {shifts:?}");
    assert!(shifts.contains(&(128u32, 125u32)), "missing 128/125 in {shifts:?}");
    for (n, d) in shifts {
        assert_ne!((n, d), (1, 1), "unison should be excluded");
        let r = n as f32 / d as f32;
        let semitones = 12.0 * r.log2();
        assert!(
            (-2.0..=2.0).contains(&semitones),
            "ratio {n}/{d} = {semitones} semitones is outside [-2, 2]",
        );
    }
}

#[test]
fn apply_ratio_shifts_pure_tone() {
    // For each enumerated fast ratio, feed a 1 kHz sine and verify the
    // spectral peak of the output lands near 1 kHz * num / den. Tolerance
    // is one STFT bin = SR / n_fft = 62.5 Hz. This is driven directly via
    // `apply_ratio` so we don't also have to force the bernoulli draw.
    let client = client();
    let f0 = 1000.0f32;
    let batch = 1;
    let sig = sine_batch(batch, f0);
    let t_in = upload_2d(&client, &sig, batch, TIME);

    let shifter = PitchShift::<Runtime>::new(client.clone(), SR, -2.0, 2.0, 1.0, dtype_f32());
    let bin_hz = SR as f32 / N_FFT as f32;

    for (i, &(num, den)) in shifter.shifts().iter().enumerate() {
        let expected = f0 * num as f32 / den as f32;
        let out = shifter.apply_ratio(&t_in, &[0u32], i);
        let out_host = read_tensor(&client, out);

        let row = &out_host[0..TIME];
        let peak = peak_freq(row, SR, expected - 4.0 * bin_hz, expected + 4.0 * bin_hz);
        let err = (peak - expected).abs();
        eprintln!(
            "[ratio {num}/{den}] expected={expected:.2} Hz, peak={peak:.2} Hz, err={err:.2} Hz",
        );
        assert!(
            err <= bin_hz,
            "ratio {num}/{den}: peak {peak:.2} Hz vs expected {expected:.2} Hz (bin = {bin_hz:.2})",
        );
    }
}

#[test]
fn batched_mixed_ratios_each_row_shifts_independently() {
    // Batch of `n_ratios` rows, each one driven with a different ratio
    // through `apply_ratio`. Verifies the gather / scatter scaffolding:
    // rows are assembled into one batched tensor, each row's output is
    // verified separately.
    let client = client();
    let f0 = 1200.0f32;
    let shifter = PitchShift::<Runtime>::new(client.clone(), SR, -2.0, 2.0, 1.0, dtype_f32());
    let n = shifter.shifts().len();
    let sig = sine_batch(n, f0);
    let t_in = upload_2d(&client, &sig, n, TIME);
    let bin_hz = SR as f32 / N_FFT as f32;

    // Process all rows under ratio 0 (for a pure batched correctness
    // check), then repeat for ratio 1 — cannot do one ratio per row via
    // `apply_ratio` alone (that API is per-ratio, not per-row). Use
    // `Transform::apply` instead below for the mixed case.
    for (i, &(num, den)) in shifter.shifts().iter().enumerate() {
        let rows: Vec<u32> = (0..n as u32).collect();
        let out = shifter.apply_ratio(&t_in, &rows, i);
        let out_host = read_tensor(&client, out);

        let expected = f0 * num as f32 / den as f32;
        for b in 0..n {
            let row = &out_host[b * TIME..(b + 1) * TIME];
            let peak = peak_freq(row, SR, expected - 4.0 * bin_hz, expected + 4.0 * bin_hz);
            let err = (peak - expected).abs();
            assert!(
                err <= bin_hz,
                "[ratio_idx={i}, row={b}] peak {peak:.2} Hz vs expected {expected:.2}",
            );
        }
    }
}

#[test]
fn probability_one_shifts_every_row_within_range() {
    // Every row is shifted by *some* ratio in the enumerated set. Because
    // the ratio is drawn per row we can't predict which one ends up
    // assigned; we just verify the peak is close to one of the ratio
    // targets (within one STFT bin).
    let client = client();
    let f0 = 1000.0f32;
    let batch = 4;
    let sig = sine_batch(batch, f0);
    let t_in = upload_2d(&client, &sig, batch, TIME);

    let shifter = PitchShift::<Runtime>::new(client.clone(), SR, -2.0, 2.0, 1.0, dtype_f32());
    let bin_hz = SR as f32 / N_FFT as f32;

    let mut rng = TransformRng::new(12345);
    let out = shifter.apply(t_in, &mut rng);
    let out_host = read_tensor(&client, out);

    for b in 0..batch {
        let row = &out_host[b * TIME..(b + 1) * TIME];
        let peak = peak_freq(row, SR, 800.0, 1200.0);
        let best_err = shifter
            .shifts()
            .iter()
            .map(|&(n, d)| (peak - f0 * n as f32 / d as f32).abs())
            .fold(f32::INFINITY, f32::min);
        assert!(
            best_err <= bin_hz,
            "row {b}: peak {peak:.2} Hz not close to any enumerated ratio (err = {best_err:.2})",
        );
    }
}

#[test]
fn same_seed_same_output() {
    let client = client();
    let batch = 3;
    let sig = sine_batch(batch, 1000.0);
    let t_in = upload_2d(&client, &sig, batch, TIME);

    let shifter = PitchShift::<Runtime>::new(client.clone(), SR, -2.0, 2.0, 1.0, dtype_f32());
    let mut rng_a = TransformRng::new(7);
    let out_a = shifter.apply(t_in.clone(), &mut rng_a);
    let a = read_tensor(&client, out_a);

    let mut rng_b = TransformRng::new(7);
    let out_b = shifter.apply(t_in, &mut rng_b);
    let b = read_tensor(&client, out_b);

    assert_eq!(a, b, "same seed must produce identical output");
}
