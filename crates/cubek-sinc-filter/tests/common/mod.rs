// Rust compiles this module separately into each `tests/*.rs` binary and
// warns on any helper an individual binary doesn't call. Silencing is
// simpler than splitting the helpers across multiple files.
#![allow(dead_code)]

//! Test helpers shared by the cpu-parity / swept-sine / julius-fixture
//! suites.
//!
//! Every test exercises the selected GPU backend, and we include a pure-Rust
//! CPU reference (`lowpass_cpu`) for code paths that don't need a Python
//! fixture.

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

pub fn upload_indices(
    client: &cubecl::client::ComputeClient<Runtime>,
    data: &[u32],
) -> TensorHandle<Runtime> {
    let handle = client.create_from_slice(u32::as_bytes(data));
    TensorHandle::<Runtime>::new_contiguous(
        vec![data.len()],
        handle,
        u32::as_type_native_unchecked().storage_type(),
    )
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

/// Deterministic pseudo-random f32 in [-1, 1). Same generator as in the
/// cubek-stft / cubek-phasevocoder / cubek-resample test commons, kept
/// byte-compatible so seeded signals look identical across crates.
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

/// Host-side windowed-sinc FIR builder, independent of `src/filters.rs` so
/// the tests don't just validate "the same code against itself". Matches
/// `julius.lowpass.LowPassFilters.__init__` in the reference tree.
pub fn build_filter_bank(cutoffs: &[f32], zeros: u32) -> (Vec<f32>, u32, u32) {
    assert!(!cutoffs.is_empty());
    let min_positive = cutoffs
        .iter()
        .copied()
        .filter(|&c| c > 0.0)
        .fold(f32::INFINITY, f32::min);
    assert!(min_positive.is_finite());
    let half_size = (zeros as f32 / min_positive / 2.0).floor() as u32;
    let filter_len = 2 * half_size + 1;

    let filter_len_f = filter_len as f64;
    let mut hann = vec![0.0f64; filter_len as usize];
    for n in 0..filter_len as usize {
        hann[n] = 0.5
            * (1.0 - (2.0 * core::f64::consts::PI * n as f64 / (filter_len_f - 1.0)).cos());
    }

    let mut weights = vec![0.0f32; cutoffs.len() * filter_len as usize];
    for (row, &c) in cutoffs.iter().enumerate() {
        let base = row * filter_len as usize;
        if c == 0.0 {
            continue;
        }
        let mut sum = 0.0f64;
        for k in 0..filter_len as usize {
            let t = (k as i64 - half_size as i64) as f64;
            let x = 2.0 * c as f64 * core::f64::consts::PI * t;
            let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
            let v = 2.0 * c as f64 * hann[k] * sinc;
            weights[base + k] = v as f32;
            sum += v;
        }
        let inv = (1.0 / sum) as f32;
        for k in 0..filter_len as usize {
            weights[base + k] *= inv;
        }
    }

    (weights, half_size, filter_len)
}

/// Pure-host reference of the full sinc-filter path: build filter bank, apply
/// the selected row to each batch row via replicate-padded FIR convolution
/// with `stride=1, pad=True`. Matches `julius.lowpass_filter` semantics so
/// we can compare against it without a Python roundtrip.
///
/// `indices[b]` picks the bank row for row `b`. If `highpass` is true, the
/// reference returns `x - lowpass(x)` per Julius-style wrapping.
pub fn lowpass_cpu(
    signal: &[f32],
    batch: usize,
    time: usize,
    cutoffs: &[f32],
    zeros: u32,
    indices: &[u32],
    highpass: bool,
) -> Vec<f32> {
    assert_eq!(signal.len(), batch * time);
    assert_eq!(indices.len(), batch);

    let (weights, half_size, filter_len) = build_filter_bank(cutoffs, zeros);
    let half_size_u = half_size as usize;
    let filter_len_u = filter_len as usize;
    let last = time - 1;

    let mut out = vec![0.0f32; batch * time];
    for b in 0..batch {
        let row = &signal[b * time..(b + 1) * time];
        let cutoff_idx = indices[b] as usize;
        let w = &weights[cutoff_idx * filter_len_u..(cutoff_idx + 1) * filter_len_u];
        for tt in 0..time {
            let mut acc = 0.0f32;
            for k in 0..filter_len_u {
                let idx = if tt + k < half_size_u {
                    0
                } else {
                    let u = tt + k - half_size_u;
                    if u > last { last } else { u }
                };
                acc += row[idx] * w[k];
            }
            let y = if highpass { row[tt] - acc } else { acc };
            out[b * time + tt] = y;
        }
    }
    out
}
