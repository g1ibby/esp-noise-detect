// Rust compiles this module separately into each `tests/*.rs` binary and
// warns on any helper that particular binary doesn't call. Silencing is
// simpler than splitting the helpers across two files.
#![allow(dead_code)]

//! Test helpers shared by the parity and round-trip suites.
//!
//! GPU output is validated against a `rustfft`-based CPU reference, so a
//! separate CPU cubecl runtime test is unnecessary.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

// `Runtime` is a per-binary type alias for cubecl's TestRuntime — the
// concrete backend is picked at `cargo test` time via one of the crate's
// `test-*` features. See README "Running tests".
pub type Runtime = TestRuntime;

pub fn client() -> cubecl::client::ComputeClient<Runtime> {
    <Runtime as cubecl::Runtime>::client(&<Runtime as cubecl::Runtime>::Device::default())
}

pub fn dtype_f32() -> StorageType {
    f32::as_type_native_unchecked().storage_type()
}

pub fn upload_1d(
    client: &cubecl::client::ComputeClient<Runtime>,
    data: &[f32],
) -> TensorHandle<Runtime> {
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::<Runtime>::new_contiguous(vec![data.len()], handle, dtype_f32())
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
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

pub fn peak_abs(a: &[f32]) -> f32 {
    a.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
}

/// Deterministic pseudo-random f32 in [-1, 1). Avoids a `rand` dependency
/// just to seed test waveforms — we want bit-reproducible inputs across
/// runs.
pub fn synthesize_signal(n: usize, seed: u64) -> Vec<f32> {
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

/// Host RFFT via rustfft, returning only the Hermitian half (`n/2 + 1` bins)
/// to match `cubek-fft::rfft` output.
pub fn rustfft_rfft(signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = signal.len();
    let mut buf: Vec<Complex32> = signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let bins = n / 2 + 1;
    let re = buf[..bins].iter().map(|c| c.re).collect();
    let im = buf[..bins].iter().map(|c| c.im).collect();
    (re, im)
}

/// Full CPU STFT reference: frame with `hop`, multiply by `window`, RFFT
/// each frame. Output layout matches our GPU STFT:
/// `(batch, n_frames, n_freq)` flattened row-major.
pub fn stft_cpu(
    signal_batch: &[Vec<f32>],
    window: &[f32],
    n_fft: usize,
    hop: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(window.len(), n_fft);
    let batch = signal_batch.len();
    let time = signal_batch[0].len();
    assert!(time >= n_fft);
    let n_frames = (time - n_fft) / hop + 1;
    let n_freq = n_fft / 2 + 1;

    let mut re_out = vec![0.0f32; batch * n_frames * n_freq];
    let mut im_out = vec![0.0f32; batch * n_frames * n_freq];

    for (b, signal) in signal_batch.iter().enumerate() {
        for f in 0..n_frames {
            let start = f * hop;
            let framed: Vec<f32> = (0..n_fft)
                .map(|i| signal[start + i] * window[i])
                .collect();
            let (re, im) = rustfft_rfft(&framed);
            let base = (b * n_frames + f) * n_freq;
            re_out[base..base + n_freq].copy_from_slice(&re);
            im_out[base..base + n_freq].copy_from_slice(&im);
        }
    }
    (re_out, im_out)
}
