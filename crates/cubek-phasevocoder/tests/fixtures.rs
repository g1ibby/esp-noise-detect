//! Byte-for-byte fixture parity vs `torchaudio.functional.phase_vocoder`.
//!
//! The fixtures under `tests/fixtures/*.bin` are produced by
//! `tests/fixtures/generate.py` running against torchaudio in the `nn/`
//! venv. Their layout is documented in the script header.
//!
//! Why fixtures on top of the CPU-reference parity tests:
//!
//! * `tests/parity.rs` compares the kernel against a Rust reimplementation
//!   of the algorithm. If *both* sides share a misunderstanding of what
//!   torchaudio does, they agree with each other and lie about correctness.
//! * These fixtures pin the ground truth to the actual torchaudio output,
//!   closing that loop.
//!
//! Tolerance is set by the fact that our kernel uses f32 arithmetic for
//! `time_step = t * rate` and `idx0 = floor(time_step)`. torchaudio does
//! the same when its complex tensor is `complex64` (real f32), so
//! theoretically the two paths see identical `idx0` / `alpha` at every
//! step. Remaining error budget is trig-function precision drift, same
//! shape as in `parity.rs`.

use std::path::PathBuf;

use cubek_phasevocoder::phase_vocoder;

mod common;

use common::{client, dtype_f32, max_abs_diff, peak_abs, read_tensor, upload_1d, upload_3d};

struct Fixture {
    batch: usize,
    n_freq: usize,
    n_in: usize,
    n_out: usize,
    hop: usize,
    rate: f32,
    phase_advance: Vec<f32>,
    input_re: Vec<f32>,
    input_im: Vec<f32>,
    output_re: Vec<f32>,
    output_im: Vec<f32>,
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_f32_slice(bytes: &[u8], offset: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| read_f32_le(bytes, offset + i * 4)).collect()
}

fn load(name: &str) -> Fixture {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {e}. Generate with \
             `cd nn && uv run --no-dev python ../crates/cubek-phasevocoder/tests/fixtures/generate.py`",
            path.display(),
        )
    });

    // Header: batch, n_freq, n_in, n_out, hop, rate (f32), then the
    // variable-length payload.
    let batch = read_u32_le(&bytes, 0) as usize;
    let n_freq = read_u32_le(&bytes, 4) as usize;
    let n_in = read_u32_le(&bytes, 8) as usize;
    let n_out = read_u32_le(&bytes, 12) as usize;
    let hop = read_u32_le(&bytes, 16) as usize;
    let rate = read_f32_le(&bytes, 20);

    let mut cursor = 24usize;
    let phase_advance = read_f32_slice(&bytes, cursor, n_freq);
    cursor += n_freq * 4;
    let in_len = batch * n_freq * n_in;
    let input_re = read_f32_slice(&bytes, cursor, in_len);
    cursor += in_len * 4;
    let input_im = read_f32_slice(&bytes, cursor, in_len);
    cursor += in_len * 4;
    let out_len = batch * n_freq * n_out;
    let output_re = read_f32_slice(&bytes, cursor, out_len);
    cursor += out_len * 4;
    let output_im = read_f32_slice(&bytes, cursor, out_len);
    cursor += out_len * 4;
    assert_eq!(cursor, bytes.len(), "fixture {name} trailing bytes");

    Fixture {
        batch,
        n_freq,
        n_in,
        n_out,
        hop,
        rate,
        phase_advance,
        input_re,
        input_im,
        output_re,
        output_im,
    }
}

fn run(name: &str) {
    let fx = load(name);
    let client = client();
    let dtype = dtype_f32();

    let (out_re_t, out_im_t) = phase_vocoder(
        upload_3d(&client, &fx.input_re, fx.batch, fx.n_freq, fx.n_in),
        upload_3d(&client, &fx.input_im, fx.batch, fx.n_freq, fx.n_in),
        upload_1d(&client, &fx.phase_advance),
        fx.rate,
        dtype,
    );
    let actual_re = read_tensor(&client, out_re_t);
    let actual_im = read_tensor(&client, out_im_t);

    assert_eq!(actual_re.len(), fx.output_re.len());
    let re_abs = max_abs_diff(&actual_re, &fx.output_re);
    let im_abs = max_abs_diff(&actual_im, &fx.output_im);
    let peak = peak_abs(&fx.output_re).max(peak_abs(&fx.output_im));

    // Same budget as `parity.rs` — trig precision over `n_out` iterations.
    let tol = 2e-3_f32;
    eprintln!(
        "[{name}] batch={} n_freq={} n_in={} n_out={} hop={} rate={}: \
         re_abs={re_abs:.3e} im_abs={im_abs:.3e} peak={peak:.3e}",
        fx.batch, fx.n_freq, fx.n_in, fx.n_out, fx.hop, fx.rate,
    );
    assert!(re_abs < tol, "re drifted: {re_abs:.3e} >= {tol:.3e}");
    assert!(im_abs < tol, "im drifted: {im_abs:.3e} >= {tol:.3e}");
}

#[test]
fn torchaudio_small_rate_1p3() {
    run("small_rate_1p3.bin");
}

#[test]
fn torchaudio_mid_rate_0p8() {
    run("mid_rate_0p8.bin");
}

#[test]
fn torchaudio_batched_rate_1p1() {
    run("batched_rate_1p1.bin");
}
