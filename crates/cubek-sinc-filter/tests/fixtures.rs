//! Byte-for-byte fixture parity vs. `julius.lowpass_filter`.
//!
//! The fixtures under `tests/fixtures/*.bin` are produced by
//! `tests/fixtures/generate.py` against Julius in the `nn/` venv. Their
//! layout is documented in the script header.
//!
//! Why fixtures on top of the `lowpass_cpu` reference in `cpu_parity.rs`:
//!
//! * `cpu_parity.rs` compares the GPU kernel against a Rust reimplementation
//!   of the same algorithm. If both paths share a misunderstanding of what
//!   Julius does, they agree and lie about correctness.
//! * These fixtures pin the ground truth to Julius's actual PyTorch output,
//!   closing that loop.
//!
//! Tolerance: the GPU and Julius paths produce the same arithmetic sequence
//! up to f32 ordering. Metal's `fma`-heavy codegen re-associates the inner
//! dot product, which bounds the disagreement by a few ULPs per tap. A
//! windowed-sinc with DC-normalized weights keeps the accumulator's running
//! magnitude O(peak signal) regardless of filter length, so the error does
//! **not** grow with `filter_len` — measured errors stay at ~1e-6 even at
//! 12801 taps (HP@20 Hz / 32 kHz). 5e-6 is a single common budget with
//! enough headroom to absorb minor backend / driver drift.

use std::path::PathBuf;

use cubek_sinc_filter::{FilterMode, LowPassFilterBank};

mod common;

use common::{client, dtype_f32, max_abs_diff, peak_abs, read_tensor, upload_2d, upload_indices, Runtime};

struct Fixture {
    batch: usize,
    time: usize,
    cutoffs: Vec<f32>,
    zeros: u32,
    mode: FilterMode,
    indices: Vec<u32>,
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

fn load(name: &str) -> Fixture {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {e}. Generate with \
             `cd nn && uv run --no-dev python ../crates/cubek-sinc-filter/tests/fixtures/generate.py`",
            path.display(),
        )
    });

    let batch = read_u32_le(&bytes, 0) as usize;
    let time = read_u32_le(&bytes, 4) as usize;
    let n_cutoffs = read_u32_le(&bytes, 8) as usize;
    let zeros = read_u32_le(&bytes, 12);
    let mode_raw = read_u32_le(&bytes, 16);
    let mode = match mode_raw {
        0 => FilterMode::LowPass,
        1 => FilterMode::HighPass,
        _ => panic!("unknown mode {mode_raw} in fixture {name}"),
    };

    let mut cursor = 20usize;
    let cutoffs: Vec<f32> = (0..n_cutoffs)
        .map(|i| read_f32_le(&bytes, cursor + i * 4))
        .collect();
    cursor += n_cutoffs * 4;

    let indices: Vec<u32> = (0..batch)
        .map(|i| read_u32_le(&bytes, cursor + i * 4))
        .collect();
    cursor += batch * 4;

    let in_len = batch * time;
    let input: Vec<f32> = (0..in_len)
        .map(|i| read_f32_le(&bytes, cursor + i * 4))
        .collect();
    cursor += in_len * 4;

    let out_len = batch * time;
    let output: Vec<f32> = (0..out_len)
        .map(|i| read_f32_le(&bytes, cursor + i * 4))
        .collect();
    cursor += out_len * 4;
    assert_eq!(cursor, bytes.len(), "fixture {name} trailing bytes");

    Fixture { batch, time, cutoffs, zeros, mode, indices, input, output }
}

fn run_per_row(name: &str, tol: f32) {
    let fx = load(name);
    let client = client();
    let dtype = dtype_f32();

    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &fx.cutoffs, fx.zeros, dtype);
    let signal_t = upload_2d(&client, &fx.input, fx.batch, fx.time);
    let idx_t = upload_indices(&client, &fx.indices);
    let out_t = bank.apply_per_row(signal_t, idx_t, fx.mode);
    let actual = read_tensor(&client, out_t);

    assert_eq!(actual.len(), fx.output.len());
    let err = max_abs_diff(&actual, &fx.output);
    let peak = peak_abs(&fx.output);
    eprintln!(
        "[{name}] batch={} time={} zeros={} err={err:.3e} peak={peak:.3e} tol={tol:.3e}",
        fx.batch, fx.time, fx.zeros,
    );
    assert!(err < tol, "{name} drifted: {err:.3e} >= {tol:.3e}");
}

// Measured errors on wgpu-Metal (M-series, spike-time) are 1e-7 to 1e-6
// range across all four fixtures — see the `eprintln!` output in
// `run_per_row`. The shared 5e-6 budget leaves ~5× headroom.
const FIXTURE_TOL: f32 = 5e-6;

#[test]
fn julius_lp_single_cutoff() {
    run_per_row("lp_single_cutoff.bin", FIXTURE_TOL);
}

#[test]
fn julius_lp_batched_mixed_cutoffs() {
    run_per_row("lp_batched_mixed_cutoffs.bin", FIXTURE_TOL);
}

#[test]
fn julius_hp_single_cutoff() {
    run_per_row("hp_single_cutoff.bin", FIXTURE_TOL);
}

#[test]
fn julius_hp_batched_mixed_cutoffs() {
    run_per_row("hp_batched_mixed_cutoffs.bin", FIXTURE_TOL);
}
