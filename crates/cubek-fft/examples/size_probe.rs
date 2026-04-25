//! Quick command-line probe for the large-`n_fft` path.
//!
//! On wgpu-Metal the old intra-cube shared-memory kernel silently returned
//! zeros for `n_fft >= 8192` because its `SharedMemory` allocation spilled
//! past the 32 KB Apple-Silicon threadgroup cap. An all-ones input DFTs to
//! `n_fft` in the DC bin and 0 elsewhere, so printing the DC bin is a
//! cheap health check after a cubek-fft change.
//!
//! Run: `cargo run --example size_probe -p cubek-fft --release`
//!
//! A correct implementation prints `DC = <n_fft>.00` on every row.

use cubecl::{
    CubeElement, Runtime, TestRuntime, frontend::CubePrimitive, std::tensor::TensorHandle,
};
use cubek_fft::rfft;

// Pick the backend at `cargo build` time via one of the crate's `test-*`
// features — same machinery the test suites use.
type R = TestRuntime;

fn probe(client: &cubecl::client::ComputeClient<R>, dtype: cubecl::prelude::StorageType, n_fft: usize) {
    let sig = vec![1.0f32; n_fft];
    let handle = client.create_from_slice(f32::as_bytes(&sig));
    let signal =
        TensorHandle::<R>::new_contiguous(vec![1, n_fft], handle, dtype);
    let (re, _im) = rfft::<R>(signal, 1, dtype);
    let bytes = client.read_one(re.handle).expect("read");
    let data = f32::from_bytes(&bytes);
    println!("  n_fft = {:>6}   DC bin = {:>10.2}", n_fft, data[0]);
}

fn main() {
    let client = <R as Runtime>::client(&<R as Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    println!("rfft DC-bin probe (all-ones input -> DC should equal n_fft):");
    for &n in &[1024usize, 2048, 4096, 8192, 16384] {
        probe(&client, dtype, n);
    }
}
