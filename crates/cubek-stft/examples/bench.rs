//! Microbenchmark comparing the two FFT call patterns used by the training
//! loop:
//!
//! 1. **Mel-style STFT** — `stft(signal=(128, padded), window, n_fft=1024,
//!    hop=320)` forward only, matching `nn-rs/src/mel.rs`. This is the "fast"
//!    case the training loop observes at ~92 ms per step.
//! 2. **Colored-noise-style FFT** — `rfft(noise=(128, 8, 4096), dim=2)`
//!    followed by `irfft(...)`. This is the raw `cubek-fft` pattern used
//!    inside `burn_audiomentations::AddColoredNoise::apply`, before the
//!    surrounding RMS / SNR glue. Each cube is single-threaded
//!    (`CubeDim::new_single()`), so the total wall time is dominated by how
//!    many cubes the backend can schedule in parallel and how many serial
//!    butterfly iterations each cube runs.
//!
//! Launch-shape summary:
//!
//!   Mel STFT forward:      (128, 101, 1024) → 12,928 cubes × ~10,240 serial ops
//!   Colored rfft forward:  (128,   8, 4096) →  1,024 cubes × ~49,152 serial ops
//!   Colored irfft:         (128,   8, 4096) →  1,024 cubes × ~49,152 serial ops
//!
//! So the colored-noise path has ~12.6× fewer cubes AND ~4.8× more serial
//! work per cube — a textbook GPU-underutilisation setup on Apple Silicon
//! where the scheduler wants thousands of concurrent threads.
//!
//! Run:
//! ```
//! cargo run --example bench -p cubek-stft --release
//! ```

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;
use cubek_fft::{irfft, rfft};
use cubek_stft::stft;
use cubek_stft::window::hann_window_symmetric;

const BATCH: usize = 128;
const WARMUP: usize = 3;
const ITERS: usize = 25;

// Pick the backend at `cargo build` time via one of the crate's `test-*`
// features — same machinery the test suites use.
type R = TestRuntime;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 * p).ceil() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn summarize(label: &str, warmup_ms: f64, samples_ms: &mut Vec<f64>) {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = samples_ms[0];
    let med_ms = samples_ms[samples_ms.len() / 2];
    let p99_ms = percentile(samples_ms, 0.99);
    println!(
        "  {:<44} warmup_mean={:>7.2} ms   min={:>7.2} ms   median={:>7.2} ms   p99={:>7.2} ms",
        label, warmup_ms, min_ms, med_ms, p99_ms,
    );
}

fn upload_2d(
    client: &cubecl::client::ComputeClient<R>,
    data: &[f32],
    batch: usize,
    time: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::<R>::new_contiguous(vec![batch, time], handle, dtype)
}

fn upload_3d(
    client: &cubecl::client::ComputeClient<R>,
    data: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::<R>::new_contiguous(vec![d0, d1, d2], handle, dtype)
}

fn upload_1d(
    client: &cubecl::client::ComputeClient<R>,
    data: &[f32],
    dtype: StorageType,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::<R>::new_contiguous(vec![data.len()], handle, dtype)
}

fn bench_mel_stft_forward(client: &cubecl::client::ComputeClient<R>, dtype: StorageType) {
    // Match mel.rs: n_fft=1024, hop=320, center=True -> pad n_fft/2 each side.
    let n_fft = 1024usize;
    let hop = 320usize;
    let time = 32_000usize;
    let padded = time + 2 * (n_fft / 2);
    let n_frames = (padded - n_fft) / hop + 1;

    let sig_host = vec![1.0f32; BATCH * padded];
    let window_host = hann_window_symmetric(n_fft);

    println!(
        "  (Mel STFT forward)  n_fft={}, hop={}, input=({}, {}), n_frames={}, cubes={}",
        n_fft,
        hop,
        BATCH,
        padded,
        n_frames,
        BATCH * n_frames,
    );

    let warmup_start = Instant::now();
    for _ in 0..WARMUP {
        let sig = upload_2d(client, &sig_host, BATCH, padded, dtype);
        let win = upload_1d(client, &window_host, dtype);
        let (re, im) = stft(sig, win, n_fft, hop, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop((re, im));
    }
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1e3 / WARMUP as f64;

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let sig = upload_2d(client, &sig_host, BATCH, padded, dtype);
        let win = upload_1d(client, &window_host, dtype);

        let t0 = Instant::now();
        let (re, im) = stft(sig, win, n_fft, hop, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_ms.push(dt);
        drop((re, im));
    }
    summarize("mel_stft_fwd (1024, hop=320)", warmup_ms, &mut samples_ms);
}

fn bench_colored_rfft(client: &cubecl::client::ComputeClient<R>, dtype: StorageType) {
    // Match noise.rs: rfft on (batch, n_chunks=8, chunk=4096) along dim=2.
    let n_chunks = 8usize;
    let chunk = 4096usize;
    let noise_host = vec![0.5f32; BATCH * n_chunks * chunk];

    println!(
        "  (Colored rfft)      n_fft={}, shape=({}, {}, {}), cubes={}",
        chunk,
        BATCH,
        n_chunks,
        chunk,
        BATCH * n_chunks,
    );

    let warmup_start = Instant::now();
    for _ in 0..WARMUP {
        let t = upload_3d(client, &noise_host, BATCH, n_chunks, chunk, dtype);
        let (re, im) = rfft(t, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop((re, im));
    }
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1e3 / WARMUP as f64;

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = upload_3d(client, &noise_host, BATCH, n_chunks, chunk, dtype);

        let t0 = Instant::now();
        let (re, im) = rfft(t, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_ms.push(dt);
        drop((re, im));
    }
    summarize("colored_rfft_fwd (4096 x 128*8)", warmup_ms, &mut samples_ms);
}

fn bench_colored_rfft_plus_irfft(client: &cubecl::client::ComputeClient<R>, dtype: StorageType) {
    // rfft + irfft back-to-back, same shape as noise.rs.
    let n_chunks = 8usize;
    let chunk = 4096usize;
    let noise_host = vec![0.5f32; BATCH * n_chunks * chunk];

    let warmup_start = Instant::now();
    for _ in 0..WARMUP {
        let t = upload_3d(client, &noise_host, BATCH, n_chunks, chunk, dtype);
        let (re, im) = rfft(t, 2, dtype);
        let out = irfft(re, im, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(out);
    }
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1e3 / WARMUP as f64;

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = upload_3d(client, &noise_host, BATCH, n_chunks, chunk, dtype);

        let t0 = Instant::now();
        let (re, im) = rfft(t, 2, dtype);
        let out = irfft(re, im, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_ms.push(dt);
        drop(out);
    }
    summarize("colored_rfft+irfft (4096 x 128*8)", warmup_ms, &mut samples_ms);
}

fn bench_mel_shape_rfft(client: &cubecl::client::ComputeClient<R>, dtype: StorageType) {
    // Flat rfft at "mel shape": (batch, n_frames=101, 1024) — isolates the
    // FFT cost from the framing + window kernel so we can compare like for
    // like against the colored-rfft bench above.
    let n_frames = 101usize;
    let n_fft = 1024usize;
    let data_host = vec![0.5f32; BATCH * n_frames * n_fft];

    println!(
        "  (Mel-shape rfft)    n_fft={}, shape=({}, {}, {}), cubes={}",
        n_fft,
        BATCH,
        n_frames,
        n_fft,
        BATCH * n_frames,
    );

    let warmup_start = Instant::now();
    for _ in 0..WARMUP {
        let t = upload_3d(client, &data_host, BATCH, n_frames, n_fft, dtype);
        let (re, im) = rfft(t, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop((re, im));
    }
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1e3 / WARMUP as f64;

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = upload_3d(client, &data_host, BATCH, n_frames, n_fft, dtype);

        let t0 = Instant::now();
        let (re, im) = rfft(t, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_ms.push(dt);
        drop((re, im));
    }
    summarize("mel_shape_rfft (1024 x 128*101)", warmup_ms, &mut samples_ms);
}

fn main() {
    let client = <R as cubecl::Runtime>::client(&<R as cubecl::Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    println!(
        "cubek-stft microbench — batch={}, warmup={}, iters={}",
        BATCH, WARMUP, ITERS,
    );
    println!();

    bench_mel_stft_forward(&client, dtype);
    bench_mel_shape_rfft(&client, dtype);
    bench_colored_rfft(&client, dtype);
    bench_colored_rfft_plus_irfft(&client, dtype);
}
