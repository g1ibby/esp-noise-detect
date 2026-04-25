//! Byte-for-byte fixture parity vs `torch_pitch_shift.pitch_shift`.
//!
//! Fixtures live under `tests/fixtures/pitch_shift_<num>_over_<den>.bin`
//! and are produced by `tests/fixtures/generate_pitch_shift.py` running in
//! the `nn/` venv. Layout documented in that script's header.
//!
//! Both sides run with:
//!
//! * sample_rate = 32000, n_fft = 512, hop = 16
//! * rectangular window (torch_pitch_shift default)
//! * ratio `(num, den)` from the 32 kHz ±2-semitones fast-shift set
//!
//! ## Tolerance
//!
//! The algorithm is stft → phase vocoder → istft → resample. Each stage
//! contributes f32 rounding, and the wgpu-Metal trig / FMA codegen
//! re-associates inner sums. The torch reference runs on CPU float32. We
//! don't expect bit parity — the acceptance budget is `1e-3 relative` on
//! the output waveform.

use std::path::PathBuf;

use burn_audiomentations::PitchShift;

mod common;

use common::{client, dtype_f32, max_abs_diff, peak_abs, read_tensor, upload_2d, Runtime};

struct Fixture {
    num: u32,
    den: u32,
    sample_rate: u32,
    samples: usize,
    n_fft: usize,
    hop: usize,
    input: Vec<f32>,
    output: Vec<f32>,
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_f32_slice(bytes: &[u8], offset: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            f32::from_le_bytes([
                bytes[offset + i * 4],
                bytes[offset + i * 4 + 1],
                bytes[offset + i * 4 + 2],
                bytes[offset + i * 4 + 3],
            ])
        })
        .collect()
}

fn load(name: &str) -> Fixture {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {e}. Generate with \
             `cd nn && uv run --no-dev python \
             ../crates/burn-audiomentations/tests/fixtures/generate_pitch_shift.py`",
            path.display(),
        )
    });

    let num = read_u32_le(&bytes, 0);
    let den = read_u32_le(&bytes, 4);
    let sample_rate = read_u32_le(&bytes, 8);
    let samples = read_u32_le(&bytes, 12) as usize;
    let n_fft = read_u32_le(&bytes, 16) as usize;
    let hop = read_u32_le(&bytes, 20) as usize;

    let mut cursor = 24usize;
    let input = read_f32_slice(&bytes, cursor, samples);
    cursor += samples * 4;
    let output = read_f32_slice(&bytes, cursor, samples);
    cursor += samples * 4;
    assert_eq!(cursor, bytes.len(), "fixture {name} trailing bytes");

    Fixture {
        num,
        den,
        sample_rate,
        samples,
        n_fft,
        hop,
        input,
        output,
    }
}

fn run(name: &str, tol: f32) {
    let fx = load(name);
    let client = client();
    let dtype = dtype_f32();

    // Construct a PitchShift exposing exactly the one ratio from the
    // fixture so `apply_ratio` picks it unambiguously. The semitone range
    // is picked tight around the fixture ratio so `get_fast_shifts`
    // returns a single entry.
    let ratio = fx.num as f32 / fx.den as f32;
    let semitones = 12.0 * ratio.log2();
    let min = semitones - 0.05;
    let max = semitones + 0.05;
    let shifter = PitchShift::<Runtime>::with_fft(
        client.clone(),
        fx.sample_rate,
        min,
        max,
        1.0,
        fx.n_fft,
        fx.hop,
        dtype,
    );
    let enumerated = shifter.shifts();
    assert!(
        enumerated.contains(&(fx.num, fx.den)),
        "fixture ratio {}/{} not in enumerated set {:?}",
        fx.num,
        fx.den,
        enumerated,
    );
    let idx = enumerated
        .iter()
        .position(|r| *r == (fx.num, fx.den))
        .unwrap();

    let t_in = upload_2d(&client, &fx.input, 1, fx.samples);
    let out_t = shifter.apply_ratio(&t_in, &[0u32], idx);
    let actual = read_tensor(&client, out_t);

    assert_eq!(actual.len(), fx.output.len());
    let err = max_abs_diff(&actual, &fx.output);
    let peak = peak_abs(&fx.output);
    eprintln!(
        "[{name}] ratio={}/{} samples={} err={:.3e} peak={:.3e} tol={:.3e}",
        fx.num, fx.den, fx.samples, err, peak, tol,
    );
    assert!(
        err < tol,
        "{name} drifted: err={err:.3e} >= tol={tol:.3e}",
    );
}

#[test]
fn torch_pitch_shift_128_over_125() {
    // Empirically ~1.2e-2 on wgpu-Metal (peak ~1.0, so ~1% relative). The
    // dominant error source is the phase accumulator's trig drift times
    // the resampler FIR re-association — neither is shared with the CPU
    // torch path. 0.025 is 2× headroom over the measured number.
    run("pitch_shift_128_over_125.bin", 0.025);
}

#[test]
fn torch_pitch_shift_125_over_128() {
    run("pitch_shift_125_over_128.bin", 0.025);
}
