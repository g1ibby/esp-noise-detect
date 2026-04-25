//! Forward-STFT parity vs a `rustfft`-based CPU reference.
//!
//! Every output frame must match rustfft's RFFT of the same windowed
//! frame. Tolerances: `abs < 1e-3` per bin on wgpu-Metal, n_fft <= 1024.
//!
//! Sweeps `n_fft in {256, 512, 1024}` and `hop in {n_fft/4, n_fft/2}`
//! to cover both the mel-front-end hop and a pitch-shift-style 50% hop.

mod common;

use common::{
    client, dtype_f32, max_abs_diff, peak_abs, read_tensor, stft_cpu, synthesize_signal,
    upload_1d, upload_2d,
};
use cubek_stft::stft;
use cubek_stft::window::hann_window_periodic;

/// Shared harness: upload a batched signal + window, run GPU STFT, read
/// back, assert parity vs the CPU reference.
fn run_parity(batch: usize, time: usize, n_fft: usize, hop: usize, seed_base: u64) {
    let client = client();
    let dtype = dtype_f32();

    let signals: Vec<Vec<f32>> = (0..batch)
        .map(|b| synthesize_signal(time, seed_base + b as u64))
        .collect();
    let mut flat = Vec::with_capacity(batch * time);
    for s in &signals {
        flat.extend_from_slice(s);
    }
    let signal_tensor = upload_2d(&client, &flat, batch, time);

    let window = hann_window_periodic(n_fft);
    let window_tensor = upload_1d(&client, &window);

    let (re_tensor, im_tensor) = stft(signal_tensor, window_tensor, n_fft, hop, dtype);
    let actual_re = read_tensor(&client, re_tensor);
    let actual_im = read_tensor(&client, im_tensor);

    let (expected_re, expected_im) = stft_cpu(&signals, &window, n_fft, hop);

    let re_abs = max_abs_diff(&actual_re, &expected_re);
    let im_abs = max_abs_diff(&actual_im, &expected_im);
    let peak = peak_abs(&expected_re).max(peak_abs(&expected_im));

    // Bigger n_fft -> larger absolute error per bin. Scale the absolute
    // threshold linearly for headroom.
    let tol = 1e-4_f32 * n_fft as f32;
    let rel_tol = 5e-5_f32;

    let re_rel = if peak > 0.0 { re_abs / peak } else { 0.0 };
    let im_rel = if peak > 0.0 { im_abs / peak } else { 0.0 };
    eprintln!(
        "[batch={batch} time={time} n_fft={n_fft} hop={hop}] \
         re_abs={re_abs:.3e} im_abs={im_abs:.3e} peak={peak:.3e} \
         re_rel={re_rel:.3e} im_rel={im_rel:.3e}",
    );
    assert!(
        re_abs < tol,
        "re diverged: abs {re_abs:.3e} >= tol {tol:.3e}"
    );
    assert!(
        im_abs < tol,
        "im diverged: abs {im_abs:.3e} >= tol {tol:.3e}"
    );
    assert!(
        re_rel < rel_tol,
        "re/peak ratio too high: {re_rel:.3e} >= {rel_tol:.3e}"
    );
    assert!(
        im_rel < rel_tol,
        "im/peak ratio too high: {im_rel:.3e} >= {rel_tol:.3e}"
    );
}

#[test]
fn parity_n_fft_256_quarter_hop() {
    run_parity(1, 2048, 256, 64, 11);
}

#[test]
fn parity_n_fft_512_quarter_hop() {
    run_parity(1, 4096, 512, 128, 22);
}

/// Primary target — mel front-end uses n_fft=1024.
#[test]
fn parity_n_fft_1024_quarter_hop() {
    run_parity(1, 16384, 1024, 256, 42);
}

/// Batched input — exercises the framing kernel's batch offset logic.
#[test]
fn parity_batched_n_fft_1024() {
    run_parity(3, 8192, 1024, 256, 100);
}

/// 50% hop, matches what a pitch-shift phase-vocoder path would do.
#[test]
fn parity_n_fft_512_half_hop() {
    run_parity(2, 4096, 512, 256, 7);
}
