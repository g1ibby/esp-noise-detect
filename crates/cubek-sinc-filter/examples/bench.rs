//! Reproducibility benchmark for the low-pass / high-pass bank under the
//! exact shape the Burn training loop uses: `(128, 32000)` f32 at sr=32 kHz.
//!
//! Why an `examples/bench.rs` and not a Criterion bench: Criterion measures
//! CPU time and expects closures with sub-millisecond steady-state cost,
//! neither of which is a good fit for a wgpu kernel where most of the time
//! is spent on the GPU and where we want to drain the queue (via
//! `cubecl::future::block_on(client.sync())`) between iterations so that we
//! see real device time rather than queue-submit throughput.
//!
//! The bench reproduces the asymmetry observed in the training loop:
//!
//!   highpass cutoffs 40..200 Hz  -> filter_len 6401 taps  (slow)
//!   lowpass  cutoffs 8000..14000 Hz -> filter_len 33 taps (trivial)
//!
//! Reported per case: `half_size`, `filter_len`, warm-up time, and
//! min/median/p99 over `ITERS` iterations of one `apply_per_row` launch
//! followed by `client.sync()`.
//!
//! Run:
//! ```
//! cargo run --example bench -p cubek-sinc-filter --release
//! ```

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;
use cubek_sinc_filter::{FilterMode, LowPassFilterBank};

const BATCH: usize = 128;
const TIME: usize = 32_000;
const SAMPLE_RATE: u32 = 32_000;
const NUM_BUCKETS: u32 = 32;
const ZEROS: u32 = 8;
const WARMUP: usize = 3;
const ITERS: usize = 25;

// Pick the backend at `cargo build` time via one of the crate's `test-*`
// features — same machinery the test suites use.
type R = TestRuntime;

// Mirror burn-audiomentations::filters — we don't depend on that crate here
// to keep the bench self-contained and to avoid getting cross-wired with
// the rng / transform plumbing.
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

fn build_mel_buckets(min_hz: f32, max_hz: f32, n: u32, sr: u32) -> Vec<f32> {
    let min_mel = hz_to_mel(min_hz);
    let max_mel = hz_to_mel(max_hz);
    (0..n)
        .map(|i| {
            let m = min_mel + (max_mel - min_mel) * (i as f32 + 0.5) / (n as f32);
            mel_to_hz(m) / sr as f32
        })
        .collect()
}

struct Case {
    label: &'static str,
    min_hz: f32,
    max_hz: f32,
    mode: FilterMode,
}

fn main() {
    let cases = [
        Case { label: "highpass_low_cutoff  (40..200 Hz)",   min_hz: 40.0,    max_hz: 200.0,   mode: FilterMode::HighPass },
        Case { label: "highpass_mid_cutoff  (2000..5000 Hz)",min_hz: 2000.0,  max_hz: 5000.0,  mode: FilterMode::HighPass },
        Case { label: "lowpass_high_cutoff  (8000..14000 Hz)",min_hz: 8000.0, max_hz: 14000.0, mode: FilterMode::LowPass },
        Case { label: "lowpass_mid_cutoff   (2000..5000 Hz)",min_hz: 2000.0,  max_hz: 5000.0,  mode: FilterMode::LowPass },
    ];

    let client = <R as cubecl::Runtime>::client(&<R as cubecl::Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let u32_dtype = u32::as_type_native_unchecked().storage_type();

    // One signal, reused. Ones are fine — kernel cost is cutoff/tap driven.
    let sig_host = vec![1.0f32; BATCH * TIME];

    println!(
        "shape=({}, {})   sample_rate={} Hz   num_buckets={}   zeros={}   warmup={}   iters={}",
        BATCH, TIME, SAMPLE_RATE, NUM_BUCKETS, ZEROS, WARMUP, ITERS,
    );
    println!();

    for case in &cases {
        // Match burn-audiomentations: prepend the "no-op" row (0.5 for
        // lowpass Nyquist-identity, 0.0 for highpass identity) then append
        // the mel-spaced bucket centers.
        let no_op_cutoff = match case.mode {
            FilterMode::LowPass => 0.5f32,
            FilterMode::HighPass => 0.0f32,
        };
        let mut cutoffs = vec![no_op_cutoff];
        cutoffs.extend(build_mel_buckets(
            case.min_hz,
            case.max_hz,
            NUM_BUCKETS,
            SAMPLE_RATE,
        ));

        let bank = LowPassFilterBank::<R>::new(client.clone(), &cutoffs, ZEROS, dtype);

        // Pick a spread of buckets for the per-row indices so we're not
        // benchmarking a single row repeatedly.
        let mut indices: Vec<u32> = (0..BATCH)
            .map(|b| 1 + (b as u32 % NUM_BUCKETS))
            .collect();
        // Sprinkle a couple of bucket-0 (no-op) entries to look realistic.
        indices[0] = 0;
        indices[BATCH / 2] = 0;

        // Warm up: first call includes shader JIT.
        let warmup_start = Instant::now();
        for _ in 0..WARMUP {
            let sig = upload_2d(&client, &sig_host, BATCH, TIME, dtype);
            let idx = upload_1d_u32(&client, &indices, u32_dtype);
            let out = bank.apply_per_row(sig, idx, case.mode);
            // Force the kernel to actually run and drain before we move on.
            cubecl::future::block_on(client.sync()).unwrap();
            drop(out);
        }
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1e3 / WARMUP as f64;

        // Timed iterations.
        let mut samples_ms = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let sig = upload_2d(&client, &sig_host, BATCH, TIME, dtype);
            let idx = upload_1d_u32(&client, &indices, u32_dtype);

            let t0 = Instant::now();
            let out = bank.apply_per_row(sig, idx, case.mode);
            cubecl::future::block_on(client.sync()).unwrap();
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            samples_ms.push(dt);
            drop(out);
        }
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_ms = samples_ms[0];
        let med_ms = samples_ms[samples_ms.len() / 2];
        let p99_ms = samples_ms[((samples_ms.len() as f64 * 0.99).ceil() as usize)
            .min(samples_ms.len() - 1)];

        println!("{}", case.label);
        println!(
            "  half_size={:>5}   filter_len={:>5}   warmup_mean={:>7.2} ms   min={:>7.2} ms   median={:>7.2} ms   p99={:>7.2} ms",
            bank.half_size(),
            bank.filter_len(),
            warmup_ms,
            min_ms,
            med_ms,
            p99_ms,
        );
        println!();
    }
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

fn upload_1d_u32(
    client: &cubecl::client::ComputeClient<R>,
    data: &[u32],
    dtype: StorageType,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(u32::as_bytes(data));
    TensorHandle::<R>::new_contiguous(vec![data.len()], handle, dtype)
}
