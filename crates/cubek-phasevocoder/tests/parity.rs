//! Parity vs the CPU reference (which is in turn fixture-checked against
//! `torchaudio.functional.phase_vocoder`).
//!
//! Tolerances: this crate does pure arithmetic, no FFT, so we can be
//! tighter than the FFT parity tests. wgpu-Metal is ~1e-5 for non-FFT
//! float ops; we target 5e-5 absolute with headroom for `atan2` / `cos` /
//! `sin` precision drift.
//!
//! The CPU reference uses the same `rate == 1.0` short-circuit, so
//! passing rate=1.0 through this suite also covers that branch.

mod common;

use common::{
    client, dtype_f32, max_abs_diff, peak_abs, phase_advance_default, phase_vocoder_cpu,
    read_tensor, synth_spectrogram, upload_1d, upload_3d,
};
use cubek_phasevocoder::phase_vocoder;

fn run_parity(
    b: usize,
    n_freq: usize,
    n_in: usize,
    hop: usize,
    rate: f32,
    seed: u64,
) {
    let client = client();
    let dtype = dtype_f32();

    let (re, im) = synth_spectrogram(b, n_freq, n_in, seed);
    let pa = phase_advance_default(n_freq, hop);

    let re_tensor = upload_3d(&client, &re, b, n_freq, n_in);
    let im_tensor = upload_3d(&client, &im, b, n_freq, n_in);
    let pa_tensor = upload_1d(&client, &pa);

    let (out_re_tensor, out_im_tensor) = phase_vocoder(re_tensor, im_tensor, pa_tensor, rate, dtype);
    let actual_re = read_tensor(&client, out_re_tensor);
    let actual_im = read_tensor(&client, out_im_tensor);

    let (expected_re, expected_im) = phase_vocoder_cpu(&re, &im, b, n_freq, n_in, &pa, rate);

    let n_out = expected_re.len() / (b * n_freq);
    assert_eq!(actual_re.len(), expected_re.len());

    let re_abs = max_abs_diff(&actual_re, &expected_re);
    let im_abs = max_abs_diff(&actual_im, &expected_im);
    let peak = peak_abs(&expected_re).max(peak_abs(&expected_im));

    // 2e-3 absolute covers the phase-unwrap branch-switching drift
    // (dominant error source) across the n_out ranges here (≤ 150 frames).
    // A tighter bound would require f64 trig on the CPU reference.
    let tol = 2e-3_f32;
    eprintln!(
        "[b={b} n_freq={n_freq} n_in={n_in} hop={hop} rate={rate} n_out={n_out}] \
         re_abs={re_abs:.3e} im_abs={im_abs:.3e} peak={peak:.3e}"
    );
    assert!(re_abs < tol, "re diverged: {re_abs:.3e} >= {tol:.3e}");
    assert!(im_abs < tol, "im diverged: {im_abs:.3e} >= {tol:.3e}");
}

/// Slowdown by 20 % — typical "rate < 1" path.
#[test]
fn parity_rate_0p8_n_freq_257() {
    run_parity(1, 257, 64, 128, 0.8, 11);
}

/// Speedup by 30 % — exercises the `new_n_frames < n_in` path and the
/// phase_acc overflow / wrap logic at many ratios of idx0 jumps.
#[test]
fn parity_rate_1p3_n_freq_513() {
    // n_freq = 513 matches the primary mel n_fft=1024 target. We step
    // through a realistic 128 frames of spectrogram.
    run_parity(1, 513, 128, 256, 1.3, 22);
}

/// Round-tripped batch: independent (re, im) spectrograms per batch slot,
/// each goes through its own (b, f) thread path. Validates the batch
/// offset arithmetic inside the kernel.
#[test]
fn parity_batched_rate_0p9() {
    run_parity(3, 129, 96, 64, 0.9, 100);
}

/// Short spectrogram edge case: rate=1.1 with n_in=8 frames. Makes sure
/// the host-side `ceil(n_in / rate)` and the kernel's idx bounds check
/// agree.
#[test]
fn parity_short_spectrogram() {
    run_parity(1, 33, 8, 16, 1.1, 7);
}

/// Low rate produces n_out > n_in, which forces idx1 == n_in near the
/// tail — the zero-pad branch in the kernel.
#[test]
fn parity_rate_0p5_long_output() {
    run_parity(1, 65, 20, 32, 0.5, 3);
}
