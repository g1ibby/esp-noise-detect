//! Fixture parity tests against a reference resampler.
//!
//! The fixtures under `tests/fixtures/*.bin` are produced by
//! `tests/fixtures/generate.py`. Layout is documented in that script.
//!
//! These fixtures pin ground truth to the reference implementation's
//! actual output, complementing the `resample_cpu` comparison in
//! `batched.rs` (which compares two independent Rust implementations
//! that could share a misunderstanding).
//!
//! Tolerance: the GPU path may re-associate the inner dot product (e.g.
//! via FMA), bounding disagreement to a few ULPs per tap x kernel_len.

use std::path::PathBuf;

use cubek_resample::Resampler;

mod common;

use common::{client, dtype_f32, max_abs_diff, peak_abs, read_tensor, upload_2d, Runtime};

struct Fixture {
    batch: usize,
    time: usize,
    out_len: usize,
    old_sr: u32,
    new_sr: u32,
    zeros: u32,
    rolloff: f32,
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
             `cd nn && uv run --no-dev python ../crates/cubek-resample/tests/fixtures/generate.py`",
            path.display(),
        )
    });

    let batch = read_u32_le(&bytes, 0) as usize;
    let time = read_u32_le(&bytes, 4) as usize;
    let out_len = read_u32_le(&bytes, 8) as usize;
    let old_sr = read_u32_le(&bytes, 12);
    let new_sr = read_u32_le(&bytes, 16);
    let zeros = read_u32_le(&bytes, 20);
    let rolloff = read_f32_le(&bytes, 24);

    let mut cursor = 28usize;
    let in_len = batch * time;
    let input = read_f32_slice(&bytes, cursor, in_len);
    cursor += in_len * 4;
    let out_flat = batch * out_len;
    let output = read_f32_slice(&bytes, cursor, out_flat);
    cursor += out_flat * 4;
    assert_eq!(cursor, bytes.len(), "fixture {name} trailing bytes");

    Fixture {
        batch,
        time,
        out_len,
        old_sr,
        new_sr,
        zeros,
        rolloff,
        input,
        output,
    }
}

fn run(name: &str, tol: f32) {
    let fx = load(name);
    let client = client();
    let dtype = dtype_f32();

    let resampler = Resampler::<Runtime>::new(
        client.clone(),
        fx.old_sr,
        fx.new_sr,
        fx.zeros,
        fx.rolloff,
        dtype,
    );
    let signal_t = upload_2d(&client, &fx.input, fx.batch, fx.time);
    let out_t = resampler.apply(signal_t, Some(fx.out_len));
    let actual = read_tensor(&client, out_t);

    assert_eq!(actual.len(), fx.output.len());
    let err = max_abs_diff(&actual, &fx.output);
    let peak = peak_abs(&fx.output);
    eprintln!(
        "[{name}] batch={} time={} out_len={} {}/{}: err={err:.3e} peak={peak:.3e} tol={tol:.3e}",
        fx.batch, fx.time, fx.out_len, fx.old_sr, fx.new_sr,
    );
    assert!(err < tol, "{name} drifted: {err:.3e} >= {tol:.3e}");
}

#[test]
fn julius_upsample_4_to_5() {
    run("upsample_4_to_5.bin", 5e-5);
}

#[test]
fn julius_downsample_5_to_4() {
    run("downsample_5_to_4.bin", 5e-5);
}

#[test]
fn julius_batched_3_to_2() {
    run("batched_3_to_2.bin", 5e-5);
}

#[test]
fn julius_audio_16k_to_44k1() {
    // Realistic audio ratio with a larger kernel; f32 dot-product error
    // scales with kernel_len, so tolerance is wider.
    run("audio_16k_to_44k1.bin", 1e-4);
}
