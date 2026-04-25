#![allow(dead_code)]

//! Test helpers shared across every test binary in this crate.
//!
//! Mirrors the common modules in the other `cubek-*` crates: we fix the
//! wgpu runtime, provide `upload_2d` / `read_tensor` helpers with the same
//! signatures, and reuse the LCG `synth_reals` generator so seeded signals
//! look identical across the workspace.

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

pub fn rms(a: &[f32]) -> f32 {
    let sq: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum();
    ((sq / a.len() as f64).sqrt()) as f32
}

/// Deterministic pseudo-random f32 in [-1, 1). Byte-compatible with the
/// other `cubek-*` test commons in this workspace.
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

/// Sine wave at `freq_hz` with `num_samples` samples at sample rate `sr`.
pub fn sine(freq_hz: f32, num_samples: usize, sr: u32) -> Vec<f32> {
    let w = 2.0 * core::f32::consts::PI * freq_hz / sr as f32;
    (0..num_samples).map(|n| (w * n as f32).sin()).collect()
}
