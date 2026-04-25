//! End-to-end benchmark for `burn_audiomentations::AddColoredNoise` at the
//! exact shape the Burn training loop uses: `(128, 32000)` f32, sr=32 kHz.
//!
//! Why this bench exists: the training loop reports `aug_ms ≈ 328 ms` per log
//! line on steps that include colored noise, or ~470 ms per fire once you
//! factor out the 0.7 Bernoulli probability. The cubek-stft microbench (see
//! `crates/cubek-stft/examples/bench.rs`) accounts for ~200 ms of that as
//! raw rfft + irfft work. This bench isolates the remainder: RMS reductions,
//! host readback of the two sum-of-squares tensors, and the add kernel.
//!
//! We force `probability=1.0` so every iteration exercises the full fire
//! path — otherwise ~30% of calls short-circuit into identity and skew the
//! distribution.
//!
//! Run:
//! ```
//! cargo run --example bench_colored_noise -p burn-audiomentations --release
//! ```

use std::time::Instant;

use burn_audiomentations::{AddColoredNoise, Transform, TransformRng};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;

const BATCH: usize = 128;
const TIME: usize = 32_000;
const SR: u32 = 32_000;
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

fn main() {
    let client = <R as cubecl::Runtime>::client(&<R as cubecl::Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    println!(
        "AddColoredNoise bench — shape=({}, {}), sr={}, warmup={}, iters={}",
        BATCH, TIME, SR, WARMUP, ITERS,
    );
    println!();

    // Mirror nn-rs/configs/aug_colored_noise.yaml: SNR 0..25 dB, f_decay
    // -1.5..1.5. probability=1.0 so every call goes through the full path.
    let t = AddColoredNoise::new(0.0, 25.0, -1.5, 1.5, SR, 1.0);
    let sig_host = vec![0.1f32; BATCH * TIME];

    let mut rng = TransformRng::new(42);

    // Warm-up: first call pays shader JIT for every unique pipeline in
    // noise.rs (random_normal, rfft, colored_mask_kernel, irfft,
    // per_row_sum_sq, add_with_scale). Subsequent calls should hit the
    // shader cache.
    let warmup_start = Instant::now();
    for _ in 0..WARMUP {
        let sig = upload_2d(&client, &sig_host, BATCH, TIME, dtype);
        let out = <AddColoredNoise as Transform<R>>::apply(&t, sig, &mut rng);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(out);
    }
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1e3 / WARMUP as f64;

    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let sig = upload_2d(&client, &sig_host, BATCH, TIME, dtype);

        let t0 = Instant::now();
        let out = <AddColoredNoise as Transform<R>>::apply(&t, sig, &mut rng);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_ms.push(dt);
        drop(out);
    }
    summarize("apply()  (prob=1.0)", warmup_ms, &mut samples_ms);

    // Also measure apply() *without* the outer client.sync() — i.e. the
    // cost up to and including the host readback that happens inside
    // apply() itself (noise_sumsq + signal_sumsq). If this is close to the
    // total, the final add kernel is cheap and the blocking readback is
    // most of the tail.
    let mut samples_noblock_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let sig = upload_2d(&client, &sig_host, BATCH, TIME, dtype);

        let t0 = Instant::now();
        let out = <AddColoredNoise as Transform<R>>::apply(&t, sig, &mut rng);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_noblock_ms.push(dt);
        drop(out);
    }
    // Drain the queue before the next bench so we aren't measuring stale
    // pending work.
    cubecl::future::block_on(client.sync()).unwrap();
    summarize("apply()  (no final sync)", 0.0, &mut samples_noblock_ms);
}
