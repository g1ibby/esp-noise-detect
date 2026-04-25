//! Stage-by-stage reproduction of `AddColoredNoise::apply` so we can measure
//! each GPU phase independently. Every stage ends with a `client.sync()` so
//! its wall time is attributed to it and not deferred into the next stage.
//!
//! Mirrors `src/noise.rs` closely enough that numbers here are directly
//! comparable to the end-to-end bench in `bench_colored_noise.rs`.
//!
//! Run:
//! ```
//! cargo run --example bench_colored_noise_stages -p burn-audiomentations --release
//! ```

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;
use cubek_fft::{irfft, rfft};
use cubek_random::random_normal;

const BATCH: usize = 128;
const TIME: usize = 32_000;
const CHUNK: usize = 4096;
const WARMUP: usize = 3;
const ITERS: usize = 25;

// Pick the backend at `cargo build` time via one of the crate's `test-*`
// features — same machinery the test suites use.
type R = TestRuntime;

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
        "  {:<44} min={:>7.2} ms   median={:>7.2} ms   p99={:>7.2} ms",
        label, min_ms, med_ms, p99_ms,
    );
}

fn main() {
    let client = <R as cubecl::Runtime>::client(&<R as cubecl::Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    let n_chunks = TIME.div_ceil(CHUNK);
    let padded = n_chunks * CHUNK;

    println!(
        "AddColoredNoise stage bench — batch={}, time={}, chunk={}, n_chunks={}, padded={}",
        BATCH, TIME, CHUNK, n_chunks, padded,
    );
    println!("(warmup={} iters={} per stage)", WARMUP, ITERS);
    println!();

    // (Signal buffer is not needed here — stages 2/4/6 re-seed noise only.
    // The end-to-end bench mixes shaped noise back into a real signal.)

    // -------- Stage 1: allocate noise buffer + random_normal fill --------
    let mut s1 = Vec::with_capacity(ITERS);
    for i in 0..(WARMUP + ITERS) {
        let t0 = Instant::now();
        let noise = TensorHandle::<R>::new_contiguous(
            vec![BATCH, n_chunks, CHUNK],
            client.empty(BATCH * n_chunks * CHUNK * dtype.size()),
            dtype,
        );
        cubek_random::seed(0xdeadbeef + i as u64);
        random_normal::<R>(&client, 0.0, 1.0, noise.binding(), dtype)
            .expect("random_normal launch failed");
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= WARMUP {
            s1.push(dt);
        }
    }
    summarize("stage1_random_normal", &mut s1);

    // -------- Stage 2: rfft of noise (dim=2) --------
    let mut s2 = Vec::with_capacity(ITERS);
    for i in 0..(WARMUP + ITERS) {
        let noise = TensorHandle::<R>::new_contiguous(
            vec![BATCH, n_chunks, CHUNK],
            client.empty(BATCH * n_chunks * CHUNK * dtype.size()),
            dtype,
        );
        cubek_random::seed(0xdeadbeef + i as u64);
        random_normal::<R>(&client, 0.0, 1.0, noise.clone().binding(), dtype).unwrap();
        cubecl::future::block_on(client.sync()).unwrap();

        let t0 = Instant::now();
        let (re, im) = rfft(noise, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= WARMUP {
            s2.push(dt);
        }
        drop((re, im));
    }
    summarize("stage2_rfft", &mut s2);

    // -------- Stage 3: rfft + mask (we don't invoke the same macro here —
    //          measuring rfft alone isolates the FFT cost; the mask kernel
    //          is cheap element-wise work and lumped into stage 3).
    //          To keep the bench self-contained and avoid re-implementing
    //          colored_mask_kernel outside burn-audiomentations, we skip
    //          the mask here; the noise.rs microbench in
    //          bench_colored_noise.rs includes it end-to-end.

    // -------- Stage 4: irfft --------
    let mut s4 = Vec::with_capacity(ITERS);
    for i in 0..(WARMUP + ITERS) {
        let noise = TensorHandle::<R>::new_contiguous(
            vec![BATCH, n_chunks, CHUNK],
            client.empty(BATCH * n_chunks * CHUNK * dtype.size()),
            dtype,
        );
        cubek_random::seed(0xdeadbeef + i as u64);
        random_normal::<R>(&client, 0.0, 1.0, noise.clone().binding(), dtype).unwrap();
        let (re, im) = rfft(noise, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();

        let t0 = Instant::now();
        let shaped = irfft(re, im, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= WARMUP {
            s4.push(dt);
        }
        drop(shaped);
    }
    summarize("stage4_irfft", &mut s4);

    // -------- Stage 5: host readback of 2×(128,) f32 tensors --------
    // The implicit sync the readback triggers is the suspected big cost in
    // the training loop, but the readback itself (128 f32 values = 512 B)
    // should be trivial once the queue is already drained.
    let mut s5 = Vec::with_capacity(ITERS);
    for _ in 0..(WARMUP + ITERS) {
        let t_handle = TensorHandle::<R>::new_contiguous(
            vec![BATCH],
            client.create_from_slice(f32::as_bytes(&vec![0.0f32; BATCH])),
            dtype,
        );
        cubecl::future::block_on(client.sync()).unwrap();

        let t0 = Instant::now();
        let _bytes = client.read_one_unchecked_tensor(t_handle.into_copy_descriptor());
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        s5.push(dt);
    }
    // Drop warmups
    s5.drain(0..WARMUP);
    summarize("stage5_readback_128f32_only (queue empty)", &mut s5);

    // -------- Stage 6: readback of (128,) AFTER a big pipelined workload --
    // Reproduces the apply() path: kick off rfft + irfft on noise, then
    // read a (128,) tensor without syncing first. This should be similar in
    // wall time to stage 2 + stage 4 combined, since the readback blocks
    // until the queue drains.
    let mut s6 = Vec::with_capacity(ITERS);
    for i in 0..(WARMUP + ITERS) {
        let noise = TensorHandle::<R>::new_contiguous(
            vec![BATCH, n_chunks, CHUNK],
            client.empty(BATCH * n_chunks * CHUNK * dtype.size()),
            dtype,
        );
        cubek_random::seed(0xbeef + i as u64);
        random_normal::<R>(&client, 0.0, 1.0, noise.clone().binding(), dtype).unwrap();

        let fake_sumsq = TensorHandle::<R>::new_contiguous(
            vec![BATCH],
            client.create_from_slice(f32::as_bytes(&vec![0.0f32; BATCH])),
            dtype,
        );
        cubecl::future::block_on(client.sync()).unwrap();

        let t0 = Instant::now();
        let (re, im) = rfft(noise, 2, dtype);
        let shaped = irfft(re, im, 2, dtype);
        // Now block on a tiny readback — its sync waits for shaped.
        let _bytes = client.read_one_unchecked_tensor(fake_sumsq.into_copy_descriptor());
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= WARMUP {
            s6.push(dt);
        }
        drop(shaped);
    }
    summarize("stage6_rfft+irfft+readback (implicit sync)", &mut s6);

    // -------- Stage 7: per-row sum-sq on (128, 32000) signal --------
    // We don't have a public handle to per_row_sum_sq_kernel from outside
    // burn-audiomentations; skip this rather than duplicate it. The end-to-
    // end bench captures its contribution.
}
