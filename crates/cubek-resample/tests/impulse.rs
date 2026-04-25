//! Impulse-response sanity tests for the resampler.
//!
//! Resampling a Kronecker delta should produce a band-limited sinc-like
//! shape, which the host-side kernel bank describes directly: for an
//! input impulse at index `p`, the output at sample `t = j*N + i` is
//! `kernel[i, p + width - j*O]` (zero out of range). Checking that the GPU
//! path matches this closed form is the cheapest end-to-end test we
//! have — no external dependency, no FFT in the critical path.

mod common;

use common::{
    build_kernel_bank, client, dtype_f32, max_abs_diff, peak_abs, read_tensor, upload_2d,
};
use cubek_resample::Resampler;

fn run_impulse(old_sr: u32, new_sr: u32, impulse_pos: usize, time: usize) {
    let client = client();
    let dtype = dtype_f32();

    // Build kernel bank host-side to derive the reference output.
    let (kernels, old_sr_r, new_sr_r, width, kernel_len) =
        build_kernel_bank(old_sr, new_sr, 24, 0.945);

    // Input: single impulse in a row of `time` samples.
    let mut signal = vec![0.0f32; time];
    signal[impulse_pos] = 1.0;
    let signal_t = upload_2d(&client, &signal, 1, time);

    // Run through the GPU Resampler.
    let resampler = Resampler::<common::Runtime>::new(
        client.clone(),
        old_sr,
        new_sr,
        24,
        0.945,
        dtype,
    );
    let out_t = resampler.apply(signal_t, None);
    let actual = read_tensor(&client, out_t);

    // Reference: output[tt] = kernel[i, impulse_pos + width - j*old_sr]
    // when the kernel tap index is in-range, else the nearest clamped
    // replicate-pad contribution (for indices that reach past the edges,
    // the replicate fills the remainder with the impulse's zero-padding —
    // i.e. zero — so those contributions vanish). The one subtlety is
    // that taps left of the input still see x[0]=0, and taps right past
    // `time-1` see x[last]=0 as long as the impulse isn't at the edge.
    // We pick impulse_pos in the interior so this is clean.
    let default_len = ((new_sr_r as i64) * (time as i64) / (old_sr_r as i64)) as usize;
    let mut expected = vec![0.0f32; default_len];
    for tt in 0..default_len {
        let i = tt % new_sr_r as usize;
        let j = tt / new_sr_r as usize;
        let base = j * old_sr_r as usize;
        let krow = &kernels[(i * kernel_len as usize)..((i + 1) * kernel_len as usize)];
        // Tap k contributes kernel[i, k] if the padded-index equals
        // impulse_pos, i.e. `base + k - width == impulse_pos`.
        // => k = impulse_pos + width - base.
        let kk = impulse_pos as i64 + width as i64 - base as i64;
        if kk >= 0 && (kk as usize) < kernel_len as usize {
            expected[tt] = krow[kk as usize];
        }
    }

    let err = max_abs_diff(&actual, &expected);
    let peak = peak_abs(&expected);
    eprintln!(
        "[impulse old={old_sr_r} new={new_sr_r} time={time} pos={impulse_pos}] \
         err={err:.3e} peak={peak:.3e} out_len={default_len}",
    );
    // Pure-arithmetic path: tight tolerance.
    assert!(err < 1e-6, "impulse response diverged: {err:.3e}");
}

#[test]
fn impulse_upsample_small_ratio() {
    // 2 -> 3 after GCD: width=~38, kernel_len=~78. Output interleaves
    // three phases per input sample.
    run_impulse(2, 3, 64, 128);
}

#[test]
fn impulse_downsample_small_ratio() {
    run_impulse(3, 2, 64, 128);
}

#[test]
fn impulse_4_to_5() {
    // Typical small-ratio case.
    run_impulse(4, 5, 128, 256);
}

#[test]
fn impulse_5_to_4() {
    run_impulse(5, 4, 128, 256);
}

#[test]
fn impulse_16000_to_44100() {
    // Realistic audio ratio. After GCD reduction: (160, 441).
    run_impulse(16_000, 44_100, 512, 1024);
}
