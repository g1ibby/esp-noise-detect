//! GPU kernel vs. a pure-Rust reimplementation of the same algorithm.
//!
//! This is the cheapest integration test we have — no Python dep, no FFT in
//! the critical path — and it locks the GPU kernel to Julius-style semantics
//! without being a full Julius parity test (`fixtures.rs` does that).
//!
//! The only tolerance allowance here is f32 reduction ordering — the two
//! paths compute the same dot product but the GPU may reassociate it via
//! `fma`.

mod common;

use common::{
    client, dtype_f32, lowpass_cpu, max_abs_diff, peak_abs, read_tensor, synth_reals, upload_2d,
    upload_indices, Runtime,
};
use cubek_sinc_filter::{FilterMode, LowPassFilterBank};

#[test]
fn single_cutoff_matches_cpu_reference() {
    let client = client();
    let dtype = dtype_f32();

    // Normalized cutoffs — 32 kHz sample rate, cutoffs in Hz mapped down.
    let cutoffs: Vec<f32> = [150.0f32, 600.0, 2400.0, 7500.0]
        .iter()
        .map(|f| f / 32_000.0)
        .collect();
    let zeros = 8;
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &cutoffs, zeros, dtype);

    let batch = 2usize;
    let time = 512usize;
    let mut signal = vec![0.0f32; batch * time];
    for b in 0..batch {
        let r = synth_reals(time, 42 + b as u64);
        signal[b * time..(b + 1) * time].copy_from_slice(&r);
    }
    let signal_t = upload_2d(&client, &signal, batch, time);

    let cutoff_idx = 2u32; // 2400 Hz
    let out_t = bank.apply_single(signal_t, cutoff_idx, FilterMode::LowPass);
    let actual = read_tensor(&client, out_t);

    let indices = vec![cutoff_idx; batch];
    let expected = lowpass_cpu(&signal, batch, time, &cutoffs, zeros, &indices, false);

    let err = max_abs_diff(&actual, &expected);
    let peak = peak_abs(&expected);
    eprintln!(
        "[single/lp] batch={batch} time={time} cutoff_idx={cutoff_idx} err={err:.3e} peak={peak:.3e}",
    );
    // Pure-arithmetic reimplementation of the same algorithm; disagreement
    // is bounded by f32 reduction ordering. Julius-parity fixtures carry
    // their own (looser) tolerance.
    assert!(err < 1e-5, "single-cutoff diverged: {err:.3e}");
}

#[test]
fn single_cutoff_highpass_matches_cpu_reference() {
    let client = client();
    let dtype = dtype_f32();

    let cutoffs: Vec<f32> = [40.0f32, 200.0].iter().map(|f| f / 32_000.0).collect();
    let zeros = 8;
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &cutoffs, zeros, dtype);

    let batch = 1usize;
    let time = 512usize;
    let signal = synth_reals(time, 7);
    let signal_t = upload_2d(&client, &signal, batch, time);

    let out_t = bank.apply_single(signal_t, 0, FilterMode::HighPass);
    let actual = read_tensor(&client, out_t);

    let expected = lowpass_cpu(&signal, batch, time, &cutoffs, zeros, &[0], true);

    let err = max_abs_diff(&actual, &expected);
    eprintln!("[single/hp] time={time} cutoff_idx=0 err={err:.3e}");
    assert!(err < 1e-5, "single-cutoff hp diverged: {err:.3e}");
}

#[test]
fn per_row_cutoffs_match_cpu_reference() {
    let client = client();
    let dtype = dtype_f32();

    // 8 buckets covering the torch_audiomentations default LPF range
    // (150 Hz – 7500 Hz at 32 kHz).
    let cutoffs: Vec<f32> = [150.0f32, 300.0, 600.0, 1200.0, 2400.0, 4000.0, 6000.0, 7500.0]
        .iter()
        .map(|f| f / 32_000.0)
        .collect();
    let zeros = 8;
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &cutoffs, zeros, dtype);

    let batch = 6usize;
    let time = 384usize;
    let mut signal = vec![0.0f32; batch * time];
    for b in 0..batch {
        let r = synth_reals(time, 100 + b as u64);
        signal[b * time..(b + 1) * time].copy_from_slice(&r);
    }

    // Spread rows across buckets so we exercise index lookups, not a
    // single-bucket fast path.
    let indices: Vec<u32> = vec![0, 3, 7, 1, 5, 2];
    let signal_t = upload_2d(&client, &signal, batch, time);
    let idx_t = upload_indices(&client, &indices);

    let out_t = bank.apply_per_row(signal_t, idx_t, FilterMode::LowPass);
    let actual = read_tensor(&client, out_t);

    let expected = lowpass_cpu(&signal, batch, time, &cutoffs, zeros, &indices, false);

    let err = max_abs_diff(&actual, &expected);
    let peak = peak_abs(&expected);
    eprintln!(
        "[per_row/lp] batch={batch} time={time} err={err:.3e} peak={peak:.3e}",
    );
    assert!(err < 1e-5, "per-row lp diverged: {err:.3e}");
}

#[test]
fn per_row_cutoffs_highpass_match_cpu_reference() {
    let client = client();
    let dtype = dtype_f32();

    // HighPassFilter defaults: 20 Hz – 2400 Hz. Quantize into 4 buckets.
    let cutoffs: Vec<f32> = [20.0f32, 120.0, 600.0, 2400.0]
        .iter()
        .map(|f| f / 32_000.0)
        .collect();
    let zeros = 8;
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &cutoffs, zeros, dtype);

    let batch = 4usize;
    let time = 384usize;
    let mut signal = vec![0.0f32; batch * time];
    for b in 0..batch {
        let r = synth_reals(time, 200 + b as u64);
        signal[b * time..(b + 1) * time].copy_from_slice(&r);
    }

    let indices: Vec<u32> = vec![0, 2, 3, 1];
    let signal_t = upload_2d(&client, &signal, batch, time);
    let idx_t = upload_indices(&client, &indices);

    let out_t = bank.apply_per_row(signal_t, idx_t, FilterMode::HighPass);
    let actual = read_tensor(&client, out_t);

    let expected = lowpass_cpu(&signal, batch, time, &cutoffs, zeros, &indices, true);

    let err = max_abs_diff(&actual, &expected);
    eprintln!("[per_row/hp] batch={batch} time={time} err={err:.3e}");
    assert!(err < 1e-5, "per-row hp diverged: {err:.3e}");
}

#[test]
fn zero_cutoff_lowpass_zeros_input_highpass_passes_through() {
    // Julius documents a cutoff of 0 as the null filter. Low-pass must
    // output zero; high-pass (x - lowpass(x)) must output the input
    // unchanged. Important convention because augmentation callers may
    // emit 0 at the edge of their mel-space uniform sampling.
    let client = client();
    let dtype = dtype_f32();

    let cutoffs = [0.0f32, 0.1f32];
    let zeros = 8;
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &cutoffs, zeros, dtype);

    let batch = 1usize;
    let time = 128usize;
    let signal = synth_reals(time, 1234);
    let signal_t = upload_2d(&client, &signal, batch, time);

    // Low-pass @ 0 -> zero output.
    let lp = bank.apply_single(signal_t, 0, FilterMode::LowPass);
    let lp_out = read_tensor(&client, lp);
    assert_eq!(lp_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max), 0.0);

    // High-pass @ 0 -> identity.
    let signal_t2 = upload_2d(&client, &signal, batch, time);
    let hp = bank.apply_single(signal_t2, 0, FilterMode::HighPass);
    let hp_out = read_tensor(&client, hp);
    let err = max_abs_diff(&hp_out, &signal);
    assert!(
        err < 1e-6,
        "highpass@0 should be identity, max-abs-diff={err:.3e}",
    );
}
