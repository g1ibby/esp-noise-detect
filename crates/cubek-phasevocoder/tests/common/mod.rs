// Rust compiles this module separately into each `tests/*.rs` binary and
// warns on any helper that particular binary doesn't call. Silencing is
// simpler than splitting the helpers across multiple files.
#![allow(dead_code)]

//! Test helpers shared by the phase-vocoder parity, identity, and fixture
//! suites.
//!
//! Tests exercise the GPU backend and compare output against a CPU reference
//! (`phase_vocoder_cpu`), which is itself validated against torchaudio via
//! the fixture files under `tests/fixtures`.

use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::TestRuntime;

/// Backend selected at compile time via one of the `test-*` features.
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

pub fn upload_3d(
    client: &cubecl::client::ComputeClient<Runtime>,
    data: &[f32],
    b: usize,
    n_freq: usize,
    n_frames: usize,
) -> TensorHandle<Runtime> {
    assert_eq!(data.len(), b * n_freq * n_frames);
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::<Runtime>::new_contiguous(vec![b, n_freq, n_frames], handle, dtype_f32())
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

/// Deterministic pseudo-random f32 in [-1, 1). Same generator as in the
/// cubek-stft test common module, kept byte-compatible so both crates see
/// the same waveforms for a given seed.
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

/// Build a random complex spectrogram `(B, n_freq, n_frames)` as two flat
/// row-major f32 vectors. Returns (re, im).
pub fn synth_spectrogram(
    b: usize,
    n_freq: usize,
    n_frames: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let n = b * n_freq * n_frames;
    let re = synth_reals(n, seed);
    let im = synth_reals(n, seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    (re, im)
}

/// Standard phase-advance vector: `linspace(0, pi * hop, n_freq)`.
///
/// Callers typically build this once per (n_freq, hop) and re-use across
/// batches.
pub fn phase_advance_default(n_freq: usize, hop: usize) -> Vec<f32> {
    if n_freq == 1 {
        return vec![0.0];
    }
    let stop = PI * hop as f32;
    let step = stop / (n_freq - 1) as f32;
    (0..n_freq).map(|k| step * k as f32).collect()
}

/// CPU reference implementation of the phase vocoder algorithm.
///
/// Input / output layout `(B, n_freq, n_frames)` row-major, matching the
/// GPU kernel's convention. Returns `(re, im)` of length
/// `B * n_freq * ceil(n_in / rate)`.
///
/// Each operation mirrors the corresponding GPU kernel line 1:1 so this
/// serves as the ground-truth for parity tests.
pub fn phase_vocoder_cpu(
    re: &[f32],
    im: &[f32],
    b: usize,
    n_freq: usize,
    n_in: usize,
    phase_advance: &[f32],
    rate: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(re.len(), b * n_freq * n_in);
    assert_eq!(im.len(), re.len());
    assert_eq!(phase_advance.len(), n_freq);
    assert!(rate > 0.0);

    if rate == 1.0 {
        return (re.to_vec(), im.to_vec());
    }

    let n_out = ((n_in as f64) / (rate as f64)).ceil() as usize;
    let mut out_re = vec![0.0f32; b * n_freq * n_out];
    let mut out_im = vec![0.0f32; b * n_freq * n_out];

    let two_pi = 2.0 * PI;

    for bi in 0..b {
        for f in 0..n_freq {
            let row_in = (bi * n_freq + f) * n_in;
            let row_out = (bi * n_freq + f) * n_out;
            let pa = phase_advance[f];

            let mut phase_acc = im[row_in].atan2(re[row_in]);

            for t in 0..n_out {
                // Use f32 arithmetic to match the kernel exactly.
                let ts = (t as f32) * rate;
                let idx0_f = ts.floor();
                let idx0 = idx0_f as usize;
                let idx1 = idx0 + 1;
                let alpha = ts - idx0_f;

                let (c0r, c0i) = if idx0 < n_in {
                    (re[row_in + idx0], im[row_in + idx0])
                } else {
                    (0.0, 0.0)
                };
                let (c1r, c1i) = if idx1 < n_in {
                    (re[row_in + idx1], im[row_in + idx1])
                } else {
                    (0.0, 0.0)
                };

                let norm_0 = (c0r * c0r + c0i * c0i).sqrt();
                let norm_1 = (c1r * c1r + c1i * c1i).sqrt();
                let mag = alpha * norm_1 + (1.0 - alpha) * norm_0;

                out_re[row_out + t] = mag * phase_acc.cos();
                out_im[row_out + t] = mag * phase_acc.sin();

                // Mirror the kernel's bounds-gated atan2 to stay
                // numerically aligned (see kernel comment on atan2(0,0)).
                let angle_0 = if idx0 < n_in {
                    c0i.atan2(c0r)
                } else {
                    0.0
                };
                let angle_1 = if idx1 < n_in {
                    c1i.atan2(c1r)
                } else {
                    0.0
                };
                let mut delta = angle_1 - angle_0 - pa;
                delta -= two_pi * (delta / two_pi).round();
                phase_acc += delta + pa;
                // Wrap to (-pi, pi] to match kernel (see kernel comment).
                phase_acc -= two_pi * (phase_acc / two_pi).round();
            }
        }
    }

    (out_re, out_im)
}
