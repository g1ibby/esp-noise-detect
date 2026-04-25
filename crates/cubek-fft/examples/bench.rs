//! `cubek-fft` micro-benchmark. Times the raw `rfft` / `irfft` kernels on
//! wgpu-Metal at the shapes the training loop hits: the "colored-noise"
//! shape `(1024, 1024)` and the "mel" shape `(16384, 512)`. Drains the
//! queue with `cubecl::future::block_on(client.sync())` between iterations
//! so the numbers reflect real device time.
//!
//! Run: `cargo run --example bench -p cubek-fft --release`

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;
use cubek_fft::{irfft, rfft};

// Pick the backend at `cargo build` time via one of the crate's `test-*`
// features — same machinery the test suites use.
type R = TestRuntime;

const WARMUP: usize = 3;
const ITERS: usize = 25;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 * p).ceil() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn summarize(label: &str, samples_ms: &mut Vec<f64>) {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = samples_ms[0];
    let med_ms = samples_ms[samples_ms.len() / 2];
    let p99_ms = percentile(samples_ms, 0.99);
    println!(
        "  {:<36}  min={:>7.2} ms   median={:>7.2} ms   p99={:>7.2} ms",
        label, min_ms, med_ms, p99_ms,
    );
}

fn upload_2d(
    client: &cubecl::client::ComputeClient<R>,
    batch: usize,
    n: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    let data = vec![0.5f32; batch * n];
    let handle = client.create_from_slice(f32::as_bytes(&data));
    TensorHandle::<R>::new_contiguous(vec![batch, n], handle, dtype)
}

fn upload_2d_spec(
    client: &cubecl::client::ComputeClient<R>,
    batch: usize,
    n_freq: usize,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    let data = vec![0.5f32; batch * n_freq];
    let re_handle = client.create_from_slice(f32::as_bytes(&data));
    let im_handle = client.create_from_slice(f32::as_bytes(&data));
    let re = TensorHandle::<R>::new_contiguous(vec![batch, n_freq], re_handle, dtype);
    let im = TensorHandle::<R>::new_contiguous(vec![batch, n_freq], im_handle, dtype);
    (re, im)
}

fn bench_rfft(client: &cubecl::client::ComputeClient<R>, dtype: StorageType, batch: usize, n: usize) {
    // warmup
    for _ in 0..WARMUP {
        let sig = upload_2d(client, batch, n, dtype);
        let (re, im) = rfft::<R>(sig, 1, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop((re, im));
    }

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let sig = upload_2d(client, batch, n, dtype);
        let t0 = Instant::now();
        let (re, im) = rfft::<R>(sig, 1, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        samples_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop((re, im));
    }
    summarize(&format!("rfft  (batch={}, n_fft={})", batch, n), &mut samples_ms);
}

fn bench_irfft(client: &cubecl::client::ComputeClient<R>, dtype: StorageType, batch: usize, n: usize) {
    let n_freq = n / 2 + 1;
    for _ in 0..WARMUP {
        let (re, im) = upload_2d_spec(client, batch, n_freq, dtype);
        let sig = irfft::<R>(re, im, 1, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(sig);
    }

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let (re, im) = upload_2d_spec(client, batch, n_freq, dtype);
        let t0 = Instant::now();
        let sig = irfft::<R>(re, im, 1, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        samples_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(sig);
    }
    summarize(&format!("irfft (batch={}, n_fft={})", batch, n), &mut samples_ms);
}

fn main() {
    let client = <R as cubecl::Runtime>::client(&<R as cubecl::Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    println!("cubek-fft microbench — warmup={} iters={}", WARMUP, ITERS);
    println!();

    bench_rfft(&client, dtype, 1024, 1024);
    bench_irfft(&client, dtype, 1024, 1024);
    bench_rfft(&client, dtype, 16384, 512);
    bench_irfft(&client, dtype, 16384, 512);
    // Large-n_fft path (packed-real + single-cfft at 8192, packed-real +
    // four-step at 16384).
    bench_rfft(&client, dtype, 128, 8192);
    bench_irfft(&client, dtype, 128, 8192);
    bench_rfft(&client, dtype, 128, 16384);
    bench_irfft(&client, dtype, 128, 16384);
}
