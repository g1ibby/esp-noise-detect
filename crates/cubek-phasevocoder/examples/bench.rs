//! Reproducibility benchmark for the full pitch-shift pipeline at shapes
//! the Burn training loop hits: `(batch, 32000)` f32 at sr=32 kHz.
//!
//! Why `examples/` and not Criterion: Criterion measures CPU time and
//! expects closures with sub-millisecond steady-state cost — neither fits
//! a wgpu compute kernel where most of the time is on the GPU and we want
//! to drain the queue with `cubecl::future::block_on(client.sync())`
//! between iterations so we measure real device time rather than
//! queue-submit throughput.
//!
//! Isolates per-stage cost so we can tell which kernel dominates:
//!
//!   STFT framing + RFFT          (cubek-stft)
//!   Phase-vocoder time-stretch   (cubek-phasevocoder)
//!   iSTFT = iRFFT + overlap-add  (cubek-stft)
//!   Resample                     (cubek-resample)
//!   End-to-end = all of the above + transposes + reflect-pad
//!   (mirrors burn-audiomentations::PitchShift::apply_ratio).
//!
//! We also benchmark the raw `rfft` / `irfft` kernels from cubek-fft with
//! the exact shape cubek-stft hands them, so we can attribute time inside
//! STFT / iSTFT to the FFT vs the elementwise wrappers.
//!
//! Two batch sizes: 128 (full batch) and 16 (typical per-ratio subgroup).
//!
//! Run:
//! ```
//! cargo run --example bench -p cubek-phasevocoder --release
//! ```

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;

use cubek_fft::{irfft, rfft};
use cubek_phasevocoder::phase_vocoder;
use cubek_resample::Resampler;
use cubek_stft::{istft, stft};

// --- shape knobs — mirror the training-loop case we're debugging ---------
const TIME: usize = 32_000;
const SAMPLE_RATE: u32 = 32_000;
const N_FFT: usize = 512;
const HOP: usize = N_FFT / 32; // 16
// 4/5 ratio (~+3.86 semitones). Keeps the sample-rate math simple
// (new_sr = 32000 * 5 / 4 = 40000) and exercises the same kernels as a
// more precise ratio — kernel cost is ratio-independent at similar n_out.
const SHIFT_NUM: u32 = 5;
const SHIFT_DEN: u32 = 4;

const WARMUP: usize = 3;
const ITERS: usize = 15;

const BATCHES: &[usize] = &[16, 128];

// Pick the backend at `cargo build` time via one of the crate's `test-*`
// features — same machinery the test suites use.
type R = TestRuntime;

fn main() {
    let client = <R as cubecl::Runtime>::client(&<R as cubecl::Runtime>::Device::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    let pad = N_FFT / 2;
    let padded_time = TIME + 2 * pad;
    let n_frames = (padded_time - N_FFT) / HOP + 1;
    let n_freq = N_FFT / 2 + 1;
    let rate = SHIFT_DEN as f32 / SHIFT_NUM as f32;
    let n_out_frames = ((n_frames as f64) / (rate as f64)).ceil() as usize;
    let istft_out_len = (n_out_frames - 1) * HOP + N_FFT;
    let new_sr = (SAMPLE_RATE as u64 * SHIFT_DEN as u64 / SHIFT_NUM as u64) as u32;

    println!(
        "Shared shape params  sample_rate={} Hz  n_fft={}  hop={}  shift={}/{}  rate={:.6}",
        SAMPLE_RATE, N_FFT, HOP, SHIFT_NUM, SHIFT_DEN, rate,
    );
    println!(
        "Per-window counts    padded_time={}  n_frames={}  n_freq={}  n_out_frames={}  istft_out={}  new_sr={}",
        padded_time, n_frames, n_freq, n_out_frames, istft_out_len, new_sr,
    );
    println!("warmup={}  iters={}", WARMUP, ITERS);
    println!();

    // Resampler is batch-independent — build once.
    let resampler = Resampler::<R>::new(client.clone(), SAMPLE_RATE, new_sr, 24, 0.945, dtype);

    // PA + window are also batch-independent.
    let window_host = vec![1.0f32; N_FFT]; // rectangular window
    let pa_host: Vec<f32> = (0..n_freq)
        .map(|k| {
            if n_freq == 1 {
                0.0
            } else {
                core::f32::consts::PI * HOP as f32 * k as f32 / (n_freq - 1) as f32
            }
        })
        .collect();

    for &batch in BATCHES {
        println!("======================================");
        println!("== batch = {}   (tensor shape = ({}, {}))", batch, batch, TIME);
        println!("======================================");
        println!(
            "  derived launch-grids:\n    stft rfft:           cube_count = {:>8} (one thread per cube — wgpu caps X at 65535)\n    istft irfft:         cube_count = {:>8} (ditto)\n    phase_vocoder:       threads     = {:>8}\n    stft frame+window:   threads     = {:>8}\n    istft overlap-add:   threads     = {:>8}",
            batch * n_frames,
            batch * n_out_frames,
            batch * n_freq,
            batch * n_frames * N_FFT,
            batch * istft_out_len,
        );
        println!();
        run_for_batch(
            &client,
            dtype,
            &resampler,
            &window_host,
            &pa_host,
            batch,
            padded_time,
            n_frames,
            n_freq,
            rate,
            n_out_frames,
        );
        println!();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_for_batch(
    client: &cubecl::client::ComputeClient<R>,
    dtype: StorageType,
    resampler: &Resampler<R>,
    window_host: &[f32],
    pa_host: &[f32],
    batch: usize,
    padded_time: usize,
    n_frames: usize,
    n_freq: usize,
    rate: f32,
    n_out_frames: usize,
) {
    let sig_host = vec![1.0f32; batch * TIME];

    let upload_window = |client: &cubecl::client::ComputeClient<R>| -> TensorHandle<R> {
        let h = client.create_from_slice(f32::as_bytes(window_host));
        TensorHandle::<R>::new_contiguous(vec![N_FFT], h, dtype)
    };
    let upload_pa = |client: &cubecl::client::ComputeClient<R>| -> TensorHandle<R> {
        let h = client.create_from_slice(f32::as_bytes(pa_host));
        TensorHandle::<R>::new_contiguous(vec![n_freq], h, dtype)
    };
    let upload_padded = |client: &cubecl::client::ComputeClient<R>| -> TensorHandle<R> {
        let data = vec![1.0f32; batch * padded_time];
        let h = client.create_from_slice(f32::as_bytes(&data));
        TensorHandle::<R>::new_contiguous(vec![batch, padded_time], h, dtype)
    };
    let upload_frames = |client: &cubecl::client::ComputeClient<R>| -> TensorHandle<R> {
        let data = vec![1.0f32; batch * n_frames * N_FFT];
        let h = client.create_from_slice(f32::as_bytes(&data));
        TensorHandle::<R>::new_contiguous(vec![batch, n_frames, N_FFT], h, dtype)
    };
    // Spectrum (batch, n_frames, n_freq) — shape RFFT of `frames` produces.
    let upload_spec = |client: &cubecl::client::ComputeClient<R>, nf: usize| -> TensorHandle<R> {
        let data = vec![1.0f32; batch * nf * n_freq];
        let h = client.create_from_slice(f32::as_bytes(&data));
        TensorHandle::<R>::new_contiguous(vec![batch, nf, n_freq], h, dtype)
    };
    // Spectrum (batch, n_freq, n_frames) — phase-vocoder input layout.
    let upload_spec_pv =
        |client: &cubecl::client::ComputeClient<R>, nf: usize| -> TensorHandle<R> {
            let data = vec![1.0f32; batch * n_freq * nf];
            let h = client.create_from_slice(f32::as_bytes(&data));
            TensorHandle::<R>::new_contiguous(vec![batch, n_freq, nf], h, dtype)
        };
    let istft_out_len_local = (n_out_frames - 1) * HOP + N_FFT;
    let upload_istft_out = |client: &cubecl::client::ComputeClient<R>| -> TensorHandle<R> {
        let data = vec![1.0f32; batch * istft_out_len_local];
        let h = client.create_from_slice(f32::as_bytes(&data));
        TensorHandle::<R>::new_contiguous(vec![batch, istft_out_len_local], h, dtype)
    };

    // Pre-reflect-pad on host. Using an all-ones signal, so this doesn't
    // matter numerically — just realistic padding shape.
    let mut padded_host = vec![0.0f32; batch * padded_time];
    let pad = N_FFT / 2;
    for b in 0..batch {
        for t in 0..padded_time {
            let src = if t < pad {
                pad - t
            } else if t - pad < TIME {
                t - pad
            } else {
                let j = t - pad;
                (TIME - 1) + (TIME - 1) - j
            };
            padded_host[b * padded_time + t] = sig_host[b * TIME + src];
        }
    }

    println!("  [per-stage]");

    bench("stft         (full: frame+window + rfft)", WARMUP, ITERS, || {
        let padded = upload_padded(client);
        let win = upload_window(client);
        let (re, im) = stft::<R>(padded, win, N_FFT, HOP, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(re);
        drop(im);
    });
    bench("  rfft only  (pre-framed input)", WARMUP, ITERS, || {
        let frames = upload_frames(client);
        let (re, im) = rfft::<R>(frames, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(re);
        drop(im);
    });

    bench("phase_vocoder", WARMUP, ITERS, || {
        let re = upload_spec_pv(client, n_frames);
        let im = upload_spec_pv(client, n_frames);
        let pa = upload_pa(client);
        let (re2, im2) = phase_vocoder::<R>(re, im, pa, rate, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(re2);
        drop(im2);
    });

    bench("istft        (full: irfft + overlap-add)", WARMUP, ITERS, || {
        let re = upload_spec(client, n_out_frames);
        let im = upload_spec(client, n_out_frames);
        let win = upload_window(client);
        let sig = istft::<R>(re, im, win, HOP, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(sig);
    });
    bench("  irfft only", WARMUP, ITERS, || {
        let re = upload_spec(client, n_out_frames);
        let im = upload_spec(client, n_out_frames);
        let sig = irfft::<R>(re, im, 2, dtype);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(sig);
    });

    bench("resample     (on iSTFT-shaped output)", WARMUP, ITERS, || {
        let sig = upload_istft_out(client);
        let out = resampler.apply(sig, None);
        cubecl::future::block_on(client.sync()).unwrap();
        drop(out);
    });

    // --- end-to-end ------------------------------------------------------
    println!();
    println!("  [end-to-end]");
    bench(
        "pitch_shift  (end-to-end PitchShift::apply_ratio)",
        WARMUP,
        ITERS,
        || {
            let win = upload_window(client);
            let pa = upload_pa(client);
            let padded_handle = client.create_from_slice(f32::as_bytes(&padded_host));
            let padded = TensorHandle::<R>::new_contiguous(
                vec![batch, padded_time],
                padded_handle,
                dtype,
            );

            let (re, im) = stft::<R>(padded, win.clone(), N_FFT, HOP, dtype);
            let re_t = transpose_last_two(client, re, dtype);
            let im_t = transpose_last_two(client, im, dtype);
            let (pv_re, pv_im) = phase_vocoder::<R>(re_t, im_t, pa, rate, dtype);
            let pv_re_b = transpose_last_two(client, pv_re, dtype);
            let pv_im_b = transpose_last_two(client, pv_im, dtype);
            let signal = istft::<R>(pv_re_b, pv_im_b, win, HOP, dtype);
            let out = resampler.apply(signal, None);
            cubecl::future::block_on(client.sync()).unwrap();
            drop(out);
        },
    );
}

fn bench(label: &str, warmup: usize, iters: usize, mut f: impl FnMut()) {
    let w0 = Instant::now();
    for _ in 0..warmup {
        f();
    }
    let warmup_ms = w0.elapsed().as_secs_f64() * 1e3 / warmup as f64;

    let mut samples_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = samples_ms[0];
    let med_ms = samples_ms[samples_ms.len() / 2];
    let p99_idx = ((samples_ms.len() as f64 * 0.99).ceil() as usize)
        .min(samples_ms.len() - 1);
    let p99_ms = samples_ms[p99_idx];

    println!(
        "    {:<52}  warm_avg={:>8.2} ms   min={:>8.2}   med={:>8.2}   p99={:>8.2}",
        label, warmup_ms, min_ms, med_ms, p99_ms,
    );
}

// ---------------------------------------------------------------------------
// Local transpose-last-two kernel (self-contained to avoid pulling in
// `burn-audiomentations`).
// ---------------------------------------------------------------------------

fn transpose_last_two(
    client: &cubecl::client::ComputeClient<R>,
    input: TensorHandle<R>,
    dtype: StorageType,
) -> TensorHandle<R> {
    let shape = input.shape().clone();
    assert_eq!(shape.len(), 3);
    let num_elems: usize = shape.iter().product();
    let out_shape = vec![shape[0], shape[2], shape[1]];
    let out = TensorHandle::<R>::new_contiguous(
        out_shape,
        client.empty(num_elems * dtype.size()),
        dtype,
    );
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, num_elems, cube_dim);
    transpose_last_two_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        input.binding().into_tensor_arg(),
        out.clone().binding().into_tensor_arg(),
    );
    out
}

#[cube(launch)]
fn transpose_last_two_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let out_d1 = output.shape(1);
    let out_d2 = output.shape(2);
    let b = pos / (out_d1 * out_d2);
    let rem = pos - b * out_d1 * out_d2;
    let y = rem / out_d2;
    let x = rem - y * out_d2;

    let in_idx = b * out_d1 * out_d2 + x * out_d1 + y;
    output[pos] = input[in_idx];
}
