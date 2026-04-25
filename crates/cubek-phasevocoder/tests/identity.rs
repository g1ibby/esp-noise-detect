//! `rate = 1.0` fast-path + pure-tone frequency-preservation check.
//!
//! Two distinct properties are covered here:
//!
//! 1. **Fast-path identity.** When `rate == 1.0` the input is returned
//!    unchanged. A random spectrogram must pass through byte-exact.
//!
//! 2. **Pure-tone preservation.** For a signal whose (complex) spectrogram
//!    is zero everywhere except one bin, and whose inter-frame phase
//!    advance exactly matches `phase_advance[bin]`, the phase vocoder at
//!    any rate must leave that bin's magnitude unchanged and concentrate
//!    the output in the same bin.

mod common;

use core::f32::consts::PI;

use common::{
    client, dtype_f32, phase_advance_default, read_tensor, synth_spectrogram, upload_1d, upload_3d,
};
use cubek_phasevocoder::phase_vocoder;

#[test]
fn rate_one_returns_input_unchanged() {
    let client = client();
    let dtype = dtype_f32();

    let b = 2usize;
    let n_freq = 65usize;
    let n_in = 32usize;
    let hop = 16usize;
    let (re, im) = synth_spectrogram(b, n_freq, n_in, 0xDEADBEEF);
    let pa = phase_advance_default(n_freq, hop);

    let (out_re, out_im) = phase_vocoder(
        upload_3d(&client, &re, b, n_freq, n_in),
        upload_3d(&client, &im, b, n_freq, n_in),
        upload_1d(&client, &pa),
        1.0,
        dtype,
    );
    let actual_re = read_tensor(&client, out_re);
    let actual_im = read_tensor(&client, out_im);

    // Short-circuit: output handles are the input handles, so the
    // readback must match the original byte-for-byte (no kernel ran).
    assert_eq!(actual_re, re, "rate=1 fast path must not touch the data");
    assert_eq!(actual_im, im, "rate=1 fast path must not touch the data");
}

/// Construct a spectrogram whose single non-zero bin (`bin = k0`) has the
/// exact inter-frame phase advance the vocoder expects. The vocoder
/// should leave its magnitude untouched; magnitude must match on every
/// output frame up to float precision.
#[test]
fn pure_tone_magnitude_preserved_at_rate_1p5() {
    let client = client();
    let dtype = dtype_f32();

    let n_freq = 129usize;
    let n_in = 32usize;
    let hop = 64usize;
    let k0 = 20usize; // pick a mid-band bin
    let rate = 1.5_f32;

    let pa = phase_advance_default(n_freq, hop);
    // For the k0 bin, the phase advances by exactly pa[k0] per frame.
    // Magnitude stays at 1.0 for a unit pure tone.
    let mut re = vec![0.0f32; n_freq * n_in];
    let mut im = vec![0.0f32; n_freq * n_in];
    let advance = pa[k0];
    for t in 0..n_in {
        let phase = advance * t as f32;
        re[k0 * n_in + t] = phase.cos();
        im[k0 * n_in + t] = phase.sin();
    }

    let (out_re_tensor, out_im_tensor) = phase_vocoder(
        upload_3d(&client, &re, 1, n_freq, n_in),
        upload_3d(&client, &im, 1, n_freq, n_in),
        upload_1d(&client, &pa),
        rate,
        dtype,
    );
    let out_re = read_tensor(&client, out_re_tensor);
    let out_im = read_tensor(&client, out_im_tensor);

    let n_out = ((n_in as f64) / (rate as f64)).ceil() as usize;
    assert_eq!(out_re.len(), n_freq * n_out);

    // Magnitude check on the pure-tone bin.
    let mut max_mag_err = 0.0_f32;
    for t in 0..n_out {
        let r = out_re[k0 * n_out + t];
        let i = out_im[k0 * n_out + t];
        let mag = (r * r + i * i).sqrt();
        max_mag_err = max_mag_err.max((mag - 1.0).abs());
    }
    eprintln!("pure-tone magnitude max err = {max_mag_err:.3e}");
    // Allow some slack at the output's very last sample where the input
    // has no idx1 = n_in neighbour (zero-pad) — the linear interpolation
    // pulls the magnitude toward 0. We check only interior samples to
    // isolate the phase-preservation property.
    for t in 0..n_out - 1 {
        let r = out_re[k0 * n_out + t];
        let i = out_im[k0 * n_out + t];
        let mag = (r * r + i * i).sqrt();
        assert!(
            (mag - 1.0).abs() < 1e-4,
            "bin {k0} frame {t}: mag={mag} deviates from 1.0 (rate={rate})",
        );
    }

    // The other bins started at zero and stay at zero — we never
    // accumulated phase there either way, so output magnitude must also
    // be zero (to within float noise).
    for k in 0..n_freq {
        if k == k0 {
            continue;
        }
        let mut max_other = 0.0_f32;
        for t in 0..n_out {
            let r = out_re[k * n_out + t];
            let i = out_im[k * n_out + t];
            max_other = max_other.max((r * r + i * i).sqrt());
        }
        assert!(
            max_other < 1e-5,
            "spurious energy in bin {k}: max_mag={max_other:.3e}",
        );
    }
}

/// Spot-check that `phase_advance_default` matches the expected
/// `linspace(0, pi * hop, n_freq)` formula.
#[test]
fn phase_advance_default_matches_linspace_formula() {
    let n_freq = 17usize;
    let hop = 32usize;
    let pa = phase_advance_default(n_freq, hop);
    assert_eq!(pa.len(), n_freq);
    assert!((pa[0] - 0.0).abs() < 1e-7);
    assert!((pa[n_freq - 1] - PI * hop as f32).abs() < 1e-5);
    // Evenly spaced.
    let step = pa[1] - pa[0];
    for k in 1..n_freq {
        assert!((pa[k] - pa[k - 1] - step).abs() < 1e-5);
    }
}
