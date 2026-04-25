//! Mel front-end parity tests.
//!
//! Fixture layout is documented in `tests/fixtures/generate.py`. We read
//! the waveform and its expected log-mel output from the fixture, run
//! Rust's `MelExtractor` on the same waveform, and compare element-wise.
//!
//! wgpu-Metal FFT is good to ~1e-4 absolute / ~1e-5 peak-relative per
//! bin. Post-log + per-example normalization amplifies errors (small mel
//! bins near `eps` on a log scale; std-division), so we use separate
//! thresholds for the raw power fixture (pre-log) and the normalized-log
//! fixture.

use std::path::PathBuf;

use burn::tensor::Tensor;
use nn_rs::{MelConfig, MelExtractor};

mod common;

use common::{Backend, client, device, max_abs_diff, peak_abs};

struct Fixture {
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    time: usize,
    n_frames: usize,
    center: bool,
    log: bool,
    normalize: bool,
    fmin: f32,
    fmax: Option<f32>,
    eps: f32,
    waveform: Vec<f32>,
    expected: Vec<f32>,
}

fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_f32(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_f32_slice(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| read_f32(bytes, off + i * 4)).collect()
}

fn load(name: &str) -> Fixture {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {e}. Regenerate with \
             `cd nn && uv run --no-dev python ../nn-rs/tests/fixtures/generate.py`",
            path.display(),
        )
    });

    let sample_rate = read_u32(&bytes, 0);
    let n_fft = read_u32(&bytes, 4) as usize;
    let hop = read_u32(&bytes, 8) as usize;
    let n_mels = read_u32(&bytes, 12) as usize;
    let time = read_u32(&bytes, 16) as usize;
    let n_frames = read_u32(&bytes, 20) as usize;

    let center = bytes[24] != 0;
    let log = bytes[25] != 0;
    let normalize = bytes[26] != 0;
    // bytes[27] is a pad byte.

    let fmin = read_f32(&bytes, 28);
    let fmax_raw = read_f32(&bytes, 32);
    let fmax = if fmax_raw == 0.0 { None } else { Some(fmax_raw) };
    let eps = read_f32(&bytes, 36);

    let mut cursor = 40usize;
    let waveform = read_f32_slice(&bytes, cursor, time);
    cursor += time * 4;
    let expected = read_f32_slice(&bytes, cursor, n_mels * n_frames);
    cursor += n_mels * n_frames * 4;
    assert_eq!(cursor, bytes.len(), "fixture {name} trailing bytes");

    Fixture {
        sample_rate,
        n_fft,
        hop,
        n_mels,
        time,
        n_frames,
        center,
        log,
        normalize,
        fmin,
        fmax,
        eps,
        waveform,
        expected,
    }
}

/// Rebuild a `MelConfig` from the fixture header. We keep the python-side
/// hop in `ms`; rounding drift (ms → samples) is already covered by
/// `MelConfig::hop_length` matching the Python `int(round(...))`.
fn cfg_from(fx: &Fixture) -> MelConfig {
    let hop_length_ms = fx.hop as f32 * 1000.0 / fx.sample_rate as f32;
    MelConfig {
        n_mels: fx.n_mels,
        fmin: fx.fmin,
        fmax: fx.fmax,
        n_fft: fx.n_fft,
        hop_length_ms,
        log: fx.log,
        eps: fx.eps,
        normalize: fx.normalize,
        center: fx.center,
    }
}

fn run(name: &str, abs_tol: f32, rel_tol: f32) {
    let fx = load(name);
    let cfg = cfg_from(&fx);

    let client = client();
    let device = device();
    let extractor = MelExtractor::<common::Runtime>::new(
        client,
        device.clone(),
        cfg.clone(),
        fx.sample_rate,
    );
    assert_eq!(
        extractor.num_frames(fx.time),
        fx.n_frames,
        "framing mismatch between Rust and fixture",
    );

    let waveform = Tensor::<Backend, 1>::from_floats(fx.waveform.as_slice(), &device)
        .reshape([1, fx.time]);
    let mel = extractor.forward(waveform);
    let dims = mel.dims();
    assert_eq!(dims, [1, 1, fx.n_mels, fx.n_frames], "output shape mismatch");

    let actual_flat = mel.into_data().convert::<f32>().to_vec::<f32>().unwrap();

    let abs = max_abs_diff(&actual_flat, &fx.expected);
    let peak = peak_abs(&fx.expected).max(1e-6);
    let rel = abs / peak;
    eprintln!(
        "[{name}] sr={} n_fft={} hop={} n_mels={} n_frames={} \
         log={} normalize={}: abs={abs:.3e} peak={peak:.3e} rel={rel:.3e}",
        fx.sample_rate, fx.n_fft, fx.hop, fx.n_mels, fx.n_frames, fx.log, fx.normalize,
    );
    assert!(
        abs < abs_tol,
        "abs diff {abs:.3e} exceeds {abs_tol:.3e} on {name}",
    );
    assert!(
        rel < rel_tol,
        "rel diff {rel:.3e} exceeds {rel_tol:.3e} on {name}",
    );
}

/// Full robust-session config (log + normalize) on 1 s of 32 kHz,
/// n_fft=1024, hop=320. Tolerance widened modestly to absorb log +
/// normalize amplification on wgpu-Metal.
#[test]
fn mel_robust_session_parity() {
    run("mel_robust_session.bin", 2e-3, 5e-4);
}

/// Raw power mel (log + normalize off) at n_fft=512. Isolates the
/// STFT → `|·|²` → filterbank path from post-processing. Raw power
/// values have a huge dynamic range (peak ~1e3), so tolerance is
/// scaled to peak-relative ~5e-5.
#[test]
fn mel_raw_power_parity() {
    run("mel_raw_power.bin", 0.1, 5e-5);
}
