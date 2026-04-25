#![allow(dead_code)]

//! Shared test plumbing for the nn-rs parity tests.
//!
//! Mirrors the `common/mod.rs` pattern used by the cubek-* crates: pin
//! the wgpu runtime, provide the upload / read helpers every test leans
//! on, reuse the same `max_abs_diff` signature so tolerance reports
//! read the same across the workspace.

use burn_cubecl::CubeBackend;
use cubecl::TestRuntime;

// `Runtime` is a per-binary type alias for cubecl's TestRuntime — the
// concrete backend is picked at `cargo test` time via one of the crate's
// `test-*` features.
pub type Runtime = TestRuntime;
pub type Backend = CubeBackend<Runtime, f32, i32, u32>;

pub fn device() -> <Runtime as cubecl::Runtime>::Device {
    <Runtime as cubecl::Runtime>::Device::default()
}

pub fn client() -> cubecl::client::ComputeClient<Runtime> {
    <Runtime as cubecl::Runtime>::client(&device())
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

pub fn peak_abs(a: &[f32]) -> f32 {
    a.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
}
