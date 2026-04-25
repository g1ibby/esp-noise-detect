// Rust compiles this module separately into each `tests/*.rs` binary and
// warns on any helper an individual binary doesn't call. Silencing is
// simpler than splitting the helpers across multiple files.
#![allow(dead_code)]

//! Test helpers shared by the impulse / batched / fixture suites.
//!
//! Includes a pure-Rust CPU reference (`resample_cpu`) for test paths
//! that don't need a Python fixture.

use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;

// `Runtime` is a per-binary type alias for cubecl's TestRuntime — the
// concrete backend is picked at `cargo test` time via one of the crate's
// `test-*` features.
pub type Runtime = TestRuntime;

pub fn client() -> cubecl::client::ComputeClient<Runtime> {
    <Runtime as cubecl::Runtime>::client(&<Runtime as cubecl::Runtime>::Device::default())
}

pub fn dtype_f32() -> StorageType {
    f32::as_type_native_unchecked().storage_type()
}

pub fn upload_2d(
    client: &cubecl::client::ComputeClient<Runtime>,
    data: &[f32],
    batch: usize,
    time: usize,
) -> TensorHandle<Runtime> {
    assert_eq!(data.len(), batch * time);
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::<Runtime>::new_contiguous(vec![batch, time], handle, dtype_f32())
}

pub fn read_tensor(
    client: &cubecl::client::ComputeClient<Runtime>,
    tensor: TensorHandle<Runtime>,
) -> Vec<f32> {
    let bytes = client.read_one_unchecked_tensor(tensor.into_copy_descriptor());
    f32::from_bytes(&bytes).to_vec()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

pub fn peak_abs(a: &[f32]) -> f32 {
    a.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
}

/// Deterministic pseudo-random f32 in [-1, 1). Byte-compatible with the
/// same generator used in sibling crate test suites.
pub fn synth_reals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s as i64 as f32) / (i64::MAX as f32)
        })
        .collect()
}

/// Host-side kernel bank builder.
///
/// Duplicated from `src/kernels.rs` so the test reference is computed
/// independently from the GPU code path.
pub fn build_kernel_bank(old_sr: u32, new_sr: u32, zeros: u32, rolloff: f32) -> (Vec<f32>, u32, u32, u32, u32) {
    let g = {
        let (mut a, mut b) = (old_sr, new_sr);
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    };
    let old_sr = old_sr / g;
    let new_sr = new_sr / g;
    if old_sr == new_sr {
        return (Vec::new(), old_sr, new_sr, 0, 0);
    }

    let sr_int = old_sr.min(new_sr);
    let sr = sr_int as f32 * rolloff;
    let width = ((zeros as f64 * old_sr as f64) / sr as f64).ceil() as u32;
    let kernel_len = 2 * width + old_sr;
    let mut out = vec![0.0f32; new_sr as usize * kernel_len as usize];
    let zeros_f = zeros as f32;
    for i in 0..new_sr {
        let base = (i as usize) * (kernel_len as usize);
        let mut sum = 0.0f64;
        for k_idx in 0..kernel_len {
            let k = (k_idx as i64) - (width as i64);
            let t_raw = (-(i as f32) / new_sr as f32 + (k as f32) / old_sr as f32) * sr;
            let t = t_raw.clamp(-zeros_f, zeros_f) * PI;
            let w = (t / (zeros_f * 2.0)).cos();
            let w2 = w * w;
            let s = if t == 0.0 { 1.0 } else { t.sin() / t };
            let v = s * w2;
            out[base + k_idx as usize] = v;
            sum += v as f64;
        }
        let inv = (1.0 / sum) as f32;
        for k_idx in 0..kernel_len {
            out[base + k_idx as usize] *= inv;
        }
    }
    (out, old_sr, new_sr, width, kernel_len)
}

/// Pure-host reference of the full resample path: build kernel bank, pad
/// with replicate, strided conv, interleave output, trim to
/// `output_length`.
pub fn resample_cpu(
    signal: &[f32],
    batch: usize,
    time: usize,
    old_sr: u32,
    new_sr: u32,
    zeros: u32,
    rolloff: f32,
    output_length: Option<usize>,
) -> Vec<f32> {
    assert_eq!(signal.len(), batch * time);
    let (kernels, old_sr_r, new_sr_r, width, kernel_len) =
        build_kernel_bank(old_sr, new_sr, zeros, rolloff);
    if old_sr_r == new_sr_r {
        return signal.to_vec();
    }
    let default_len =
        ((new_sr_r as i64) * (time as i64) / (old_sr_r as i64)) as usize;
    let out_len = output_length.unwrap_or(default_len);
    let mut out = vec![0.0f32; batch * out_len];

    for b in 0..batch {
        let row = &signal[b * time..(b + 1) * time];
        let last = time - 1;
        for tt in 0..out_len {
            let i = tt % new_sr_r as usize;
            let j = tt / new_sr_r as usize;
            let krow = &kernels
                [(i * kernel_len as usize)..((i + 1) * kernel_len as usize)];
            let base = j * old_sr_r as usize;
            let mut acc = 0.0f32;
            for k in 0..kernel_len as usize {
                let idx = if base + k < width as usize {
                    0
                } else {
                    let u = base + k - width as usize;
                    if u > last { last } else { u }
                };
                acc += row[idx] * krow[k];
            }
            out[b * out_len + tt] = acc;
        }
    }
    out
}
