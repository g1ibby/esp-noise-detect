//! Swept-sine frequency-response check.
//!
//! Pass-band should be flat, stop-band attenuated by the expected amount. We build a few pure
//! tones spanning both sides of the cutoff, filter them, and measure the
//! post-filter RMS against the clean RMS of the input tone. A windowed-sinc
//! Hann-tapered FIR with `zeros=8` has a ~6 dB point at the cutoff and a
//! stop-band floor well below -20 dB once we're ~2x the cutoff frequency;
//! both of those are easy to assert without relying on a fragile absolute
//! threshold.

mod common;

use common::{client, dtype_f32, read_tensor, upload_2d, Runtime};
use cubek_sinc_filter::{FilterMode, LowPassFilterBank};

fn pure_tone(n: usize, sample_rate: f32, freq: f32) -> Vec<f32> {
    let w = 2.0 * core::f32::consts::PI * freq / sample_rate;
    (0..n).map(|i| (w * i as f32).cos()).collect()
}

fn rms_interior(x: &[f32], skip: usize) -> f32 {
    // Skip the outer `skip` samples on each side to avoid the transient
    // region of the FIR response (the filter has a "turn-on" tail equal to
    // the half-size plus whatever replicate-padding does at the boundaries).
    assert!(x.len() > 2 * skip);
    let body = &x[skip..x.len() - skip];
    let mean_sq: f32 = body.iter().map(|v| v * v).sum::<f32>() / body.len() as f32;
    mean_sq.sqrt()
}

#[test]
fn lowpass_passband_near_unity_stopband_attenuated() {
    let sample_rate = 32_000.0f32;
    let cutoff_hz = 2_000.0f32;
    let cutoff = cutoff_hz / sample_rate;
    let zeros = 8u32;

    let client = client();
    let dtype = dtype_f32();
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &[cutoff], zeros, dtype);
    let half_size = bank.half_size() as usize;

    let time = 8192usize; // Plenty long so the skip region costs little.
    let skip = half_size + 32;

    // Test tones at 0.25x (well below), 2x (well above) the cutoff.
    //
    // At the cutoff itself a Hann-windowed sinc with zeros=8 is about
    // -6 dB, which makes a tight absolute threshold fiddly. We measure
    // at 0.25x / 2x and assert ratios instead. The 2x stop-band floor is
    // already <-30 dB at zeros=8 based on Julius's design.
    let passband_tone = pure_tone(time, sample_rate, 0.25 * cutoff_hz);
    let stopband_tone = pure_tone(time, sample_rate, 2.0 * cutoff_hz);

    let passband_t = upload_2d(&client, &passband_tone, 1, time);
    let stopband_t = upload_2d(&client, &stopband_tone, 1, time);

    let passband_out_t = bank.apply_single(passband_t, 0, FilterMode::LowPass);
    let stopband_out_t = bank.apply_single(stopband_t, 0, FilterMode::LowPass);
    let passband_out = read_tensor(&client, passband_out_t);
    let stopband_out = read_tensor(&client, stopband_out_t);

    let passband_rms = rms_interior(&passband_out, skip);
    let stopband_rms = rms_interior(&stopband_out, skip);
    // Pure cosine has RMS = 1/sqrt(2).
    let input_rms = (0.5f32).sqrt();
    let passband_gain_db = 20.0 * (passband_rms / input_rms).log10();
    let stopband_gain_db = 20.0 * (stopband_rms / input_rms).log10();

    eprintln!(
        "[lp 2 kHz/32 kHz zeros={zeros}] passband@500Hz={passband_gain_db:.2} dB \
         stopband@4kHz={stopband_gain_db:.2} dB (half_size={half_size})",
    );
    assert!(
        passband_gain_db > -1.0,
        "passband attenuated too much: {passband_gain_db:.2} dB",
    );
    assert!(
        stopband_gain_db < -30.0,
        "stopband attenuation insufficient: {stopband_gain_db:.2} dB",
    );
}

#[test]
fn highpass_is_complement_of_lowpass() {
    // x - lowpass(x) should pass high frequencies and attenuate low ones.
    // We reuse the same setup as the low-pass test, swap the mode, and
    // assert the opposite of the gain pattern.
    let sample_rate = 32_000.0f32;
    let cutoff_hz = 2_000.0f32;
    let cutoff = cutoff_hz / sample_rate;
    let zeros = 8u32;

    let client = client();
    let dtype = dtype_f32();
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &[cutoff], zeros, dtype);
    let half_size = bank.half_size() as usize;

    let time = 8192usize;
    let skip = half_size + 32;

    let low_tone = pure_tone(time, sample_rate, 0.25 * cutoff_hz);
    let high_tone = pure_tone(time, sample_rate, 2.0 * cutoff_hz);

    let low_t = upload_2d(&client, &low_tone, 1, time);
    let high_t = upload_2d(&client, &high_tone, 1, time);

    let low_out_t = bank.apply_single(low_t, 0, FilterMode::HighPass);
    let high_out_t = bank.apply_single(high_t, 0, FilterMode::HighPass);
    let low_out = read_tensor(&client, low_out_t);
    let high_out = read_tensor(&client, high_out_t);

    let low_rms = rms_interior(&low_out, skip);
    let high_rms = rms_interior(&high_out, skip);
    let input_rms = (0.5f32).sqrt();

    let low_gain_db = 20.0 * (low_rms / input_rms).log10();
    let high_gain_db = 20.0 * (high_rms / input_rms).log10();

    eprintln!(
        "[hp 2 kHz/32 kHz zeros={zeros}] low@500Hz={low_gain_db:.2} dB \
         high@4kHz={high_gain_db:.2} dB",
    );
    assert!(
        low_gain_db < -30.0,
        "hp stopband attenuation insufficient at 0.25*cutoff: {low_gain_db:.2} dB",
    );
    assert!(
        high_gain_db > -1.0,
        "hp passband attenuated too much at 2*cutoff: {high_gain_db:.2} dB",
    );
}

#[test]
fn dc_is_preserved_by_lowpass() {
    // The DC-normalization in the filter builder guarantees that a constant
    // input passes through a low-pass unchanged. It's the cheapest sanity
    // check on the normalization.
    let sample_rate = 32_000.0f32;
    let cutoff_hz = 800.0f32;
    let cutoff = cutoff_hz / sample_rate;
    let zeros = 8u32;

    let client = client();
    let dtype = dtype_f32();
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &[cutoff], zeros, dtype);

    let time = 2048usize;
    let signal = vec![0.7f32; time];
    let signal_t = upload_2d(&client, &signal, 1, time);
    let out_t = bank.apply_single(signal_t, 0, FilterMode::LowPass);
    let out = read_tensor(&client, out_t);

    // Replicate-pad at both edges means DC is preserved everywhere, not
    // just in the interior. f32 dot-product rounding bounds the error at a
    // handful of ULPs × filter_len.
    let peak = out
        .iter()
        .map(|v| (v - 0.7).abs())
        .fold(0.0f32, f32::max);
    assert!(peak < 1e-5, "DC drifted under low-pass: peak diff {peak:.3e}");
}
