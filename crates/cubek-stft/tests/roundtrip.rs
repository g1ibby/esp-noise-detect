//! iSTFT(STFT(x)) ≈ x round-trip under a COLA-compliant Hann / hop setup.
//!
//! Standard constant-overlap-add condition: periodic Hann window with
//! `hop = n_fft / 4` is COLA-safe, and the `Σ window^2` divisor baked
//! into `overlap_add_kernel` turns the forward/inverse pair into an
//! identity over the interior samples.
//!
//! We compare against the original waveform only on the *interior* region
//! `[n_fft, T - n_fft]` — the outer samples see fewer frames contributing
//! and therefore suffer ramp-in / ramp-out bias even with a COLA window.
//! This matches how librosa / torchaudio document their STFT round-trip
//! guarantees.

mod common;

use common::{
    client, dtype_f32, max_abs_diff, peak_abs, read_tensor, synthesize_signal, upload_1d,
    upload_2d,
};
use cubek_stft::window::hann_window_periodic;
use cubek_stft::{istft, stft};

fn run_roundtrip(batch: usize, time: usize, n_fft: usize, hop: usize, seed_base: u64) {
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

    let (re, im) = stft(signal_tensor, window_tensor.clone(), n_fft, hop, dtype);
    // cubek-fft's rfft path internally clones TensorHandles, so the window
    // we pass to istft must be a fresh upload — pass the handle we saved.
    let reconstructed_tensor = istft(re, im, window_tensor, hop, dtype);
    let reconstructed = read_tensor(&client, reconstructed_tensor);

    // Reconstructed shape is (batch, (n_frames - 1)*hop + n_fft). For our
    // framing that equals (time - (time - n_fft) % hop). When hop divides
    // (time - n_fft), reconstructed length == time.
    assert_eq!((time - n_fft) % hop, 0, "test setup: hop must divide time - n_fft");
    let t_out = (((time - n_fft) / hop) * hop) + n_fft;
    assert_eq!(t_out, time);
    assert_eq!(reconstructed.len(), batch * t_out);

    // Interior: skip one full window at either end where overlap-add is
    // ramping up / down.
    let interior_start = n_fft;
    let interior_end = t_out - n_fft;
    assert!(interior_end > interior_start);

    let mut max_err = 0.0_f32;
    let mut peak = 0.0_f32;
    for b in 0..batch {
        let row = &reconstructed[b * t_out..(b + 1) * t_out];
        let orig = &signals[b];
        let err = max_abs_diff(
            &row[interior_start..interior_end],
            &orig[interior_start..interior_end],
        );
        peak = peak.max(peak_abs(&orig[interior_start..interior_end]));
        max_err = max_err.max(err);
    }

    // STFT + iSTFT compound error; allow headroom over single-pass tolerance.
    let tol = 5e-4_f32;
    eprintln!(
        "[batch={batch} time={time} n_fft={n_fft} hop={hop}] \
         max_err={max_err:.3e} peak={peak:.3e} tol={tol:.3e}",
    );
    assert!(max_err < tol, "round-trip interior error {max_err:.3e} >= {tol:.3e}");
}

#[test]
fn roundtrip_n_fft_256_quarter_hop() {
    // (time - n_fft) = 2048 - 256 = 1792, divisible by hop=64.
    run_roundtrip(1, 2048, 256, 64, 11);
}

#[test]
fn roundtrip_n_fft_1024_quarter_hop() {
    // (time - n_fft) = 8192 - 1024 = 7168, divisible by hop=256.
    run_roundtrip(1, 8192, 1024, 256, 42);
}

#[test]
fn roundtrip_batched_n_fft_512() {
    // (time - n_fft) = 4096 - 512 = 3584, divisible by hop=128.
    run_roundtrip(2, 4096, 512, 128, 77);
}
