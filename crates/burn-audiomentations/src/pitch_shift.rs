//! `PitchShift` — rational-ratio pitch shift via STFT → phase-vocoder →
//! iSTFT → fractional resample.
//!
//! We precompute one [`Resampler`] per ratio at construction time and pick
//! from them by host-side group index, rather than rebuilding a resampler
//! per batch.
//!
//! ## Algorithm per selected row
//!
//! 1. Reflect-pad by `n_fft / 2` on each end (centered STFT convention).
//! 2. [`stft`] → `(n, n_frames, n_freq)`.
//! 3. Transpose last two dims to `(n, n_freq, n_frames)` for
//!    [`phase_vocoder`].
//! 4. Time-stretch at `rate = 1 / shift` (i.e. `den / num`).
//! 5. Transpose back to `(n, n_frames_new, n_freq)`.
//! 6. [`istft`] → `(n, signal_len)`.
//! 7. Trim `n_fft / 2` from each end (centered iSTFT trim).
//! 8. [`Resampler::apply`] from `sample_rate` to `sample_rate * den / num`
//!    — the pitch-raising / lowering step.
//! 9. Crop / zero-pad to the original `time` length.
//!
//! Unselected rows (Bernoulli `probability == 0` case, or a row that
//! randomly got sampled out) skip the whole pipeline — the output row is
//! bit-for-bit identical to the input row.
//!
//! ## Ratio quantization
//!
//! We pick the set of fast ratios once at construction by calling
//! [`get_fast_shifts`] with a condition that filters to the augmentation's
//! semitone range. At runtime the per-row `semitones` draw is mapped to
//! the **nearest** `(num, den)` in that set. The ±~0.4 semitone
//! quantization error is acceptable for audio augmentation, and it enables
//! the one-resampler-per-ratio trick.
//!
//! If the enumerated set is empty for a given `(sample_rate, semitone range)`
//! the constructor panics — widen the range or pick a sample rate with more
//! prime factors.
//!
//! ## Grouping model
//!
//! At apply time we bucket batch rows by their assigned ratio index, gather
//! each bucket into a contiguous `(n_group, time)` tensor, run the pipeline,
//! and scatter the result back into the output buffer. Rows with no group
//! (unselected by Bernoulli) keep their input values via an initial copy.
//!
//! This keeps every kernel launch batched over `n_group` rows (so the GPU
//! stays saturated on the common case where one ratio dominates), at the
//! cost of a host-side `HashMap`-like loop and two extra gather / scatter
//! kernels per group. For typical batches (8–64 rows, 2–6 ratios in the
//! set) that overhead is a rounding error next to the STFT / iSTFT work.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_phasevocoder::phase_vocoder;
use cubek_resample::Resampler;
use cubek_resample::fast_shifts::get_fast_shifts;
use cubek_stft::{istft, stft};

use crate::kernels::elemwise_launch_dims;
use crate::transform::{Transform, TransformRng, bernoulli_mask, sample_uniform_batch};

/// Default FFT length. For 32 kHz the natural `sample_rate // 64 = 500` is
/// rounded to the nearest power of two (512) for `cubek-fft`'s radix-2
/// stages.
pub const DEFAULT_N_FFT: usize = 512;

/// Default hop length: `n_fft // 32`.
pub const DEFAULT_HOP: usize = DEFAULT_N_FFT / 32;

/// Pitch shift by a semitone amount drawn per batch row.
///
/// * `min_semitones` / `max_semitones` — inclusive semitone range. The
///   draw is uniform in semitones and then quantized to the nearest
///   enumerated ratio.
/// * `probability` — Bernoulli per-row apply probability. `0.0` returns
///   the input unchanged.
pub struct PitchShift<R: Runtime> {
    pub sample_rate: u32,
    pub min_semitones: f32,
    pub max_semitones: f32,
    pub probability: f64,
    pub n_fft: usize,
    pub hop: usize,

    /// Enumerated `(num, den)` ratios inside the semitone range.
    shifts: Vec<(u32, u32)>,
    /// One resampler per entry in `shifts`, `(sample_rate, sample_rate * den / num)`.
    resamplers: Vec<Resampler<R>>,
    /// `(n_fft,)` analysis window, uploaded once.
    window: TensorHandle<R>,
    /// `(n_freq,)` per-bin expected phase advance, uploaded once.
    phase_advance: TensorHandle<R>,

    dtype: StorageType,
    client: ComputeClient<R>,
}

impl<R: Runtime> PitchShift<R> {
    /// Construct with default STFT parameters (`n_fft = 512`, `hop = 16`).
    pub fn new(
        client: ComputeClient<R>,
        sample_rate: u32,
        min_semitones: f32,
        max_semitones: f32,
        probability: f64,
        dtype: StorageType,
    ) -> Self {
        Self::with_fft(
            client,
            sample_rate,
            min_semitones,
            max_semitones,
            probability,
            DEFAULT_N_FFT,
            DEFAULT_HOP,
            dtype,
        )
    }

    /// Same as [`Self::new`] but lets the caller override `n_fft` and `hop`.
    pub fn with_fft(
        client: ComputeClient<R>,
        sample_rate: u32,
        min_semitones: f32,
        max_semitones: f32,
        probability: f64,
        n_fft: usize,
        hop: usize,
        dtype: StorageType,
    ) -> Self {
        assert!(
            min_semitones <= max_semitones,
            "min_semitones ({min_semitones}) > max_semitones ({max_semitones})",
        );
        assert!(n_fft >= 2 && n_fft.is_power_of_two(), "n_fft must be a power of two ≥ 2");
        assert!(hop > 0 && hop <= n_fft, "hop must be in (0, n_fft]");
        assert!(sample_rate > 0);

        // Ratio filter. 1e-6 slack so exact-boundary ratios like (1, 2)
        // at -12 semitones get picked up.
        let min_ratio = 2f32.powf(min_semitones / 12.0) - 1e-6;
        let max_ratio = 2f32.powf(max_semitones / 12.0) + 1e-6;
        let shifts = get_fast_shifts(sample_rate, |n, d| {
            let r = n as f32 / d as f32;
            n != d && r >= min_ratio && r <= max_ratio
        });
        assert!(
            !shifts.is_empty(),
            "no fast pitch shifts for sample_rate={sample_rate} in [{min_semitones}, {max_semitones}] semitones",
        );

        let resamplers: Vec<Resampler<R>> = shifts
            .iter()
            .map(|&(num, den)| {
                // new_sr = sample_rate / shift = sample_rate * den / num.
                // Valid because `num` divides `sample_rate` by the way
                // `get_fast_shifts` constructs its products.
                let new_sr = (sample_rate as u64 * den as u64 / num as u64) as u32;
                Resampler::new(client.clone(), sample_rate, new_sr, 24, 0.945, dtype)
            })
            .collect();

        // Rectangular window (all ones). WOLA
        // reconstruction still works because the denominator `Σ w²` is
        // constant (= n_fft / hop) over the interior.
        let window_vec = vec![1.0f32; n_fft];
        let window = TensorHandle::<R>::new_contiguous(
            vec![n_fft],
            client.create_from_slice(f32::as_bytes(&window_vec)),
            dtype,
        );

        // phase_advance[k] = π * hop * k / (n_freq - 1).
        let n_freq = n_fft / 2 + 1;
        let pa: Vec<f32> = (0..n_freq)
            .map(|k| {
                if n_freq == 1 {
                    0.0
                } else {
                    core::f32::consts::PI * hop as f32 * k as f32 / (n_freq - 1) as f32
                }
            })
            .collect();
        let phase_advance = TensorHandle::<R>::new_contiguous(
            vec![n_freq],
            client.create_from_slice(f32::as_bytes(&pa)),
            dtype,
        );

        Self {
            sample_rate,
            min_semitones,
            max_semitones,
            probability,
            n_fft,
            hop,
            shifts,
            resamplers,
            window,
            phase_advance,
            dtype,
            client,
        }
    }

    /// The `(num, den)` ratios this instance will pick from. Exposed so
    /// parity tests can bypass the `semitones → nearest ratio` sampling
    /// step and exercise each ratio directly.
    pub fn shifts(&self) -> &[(u32, u32)] {
        &self.shifts
    }

    /// Apply the full pipeline to a single ratio index across `rows` —
    /// used both by the Bernoulli path (via [`Transform::apply`]) and by
    /// tests that want to drive a specific ratio.
    pub fn apply_ratio(
        &self,
        samples: &TensorHandle<R>,
        rows: &[u32],
        ratio_idx: usize,
    ) -> TensorHandle<R> {
        assert!(ratio_idx < self.shifts.len());
        let time = samples.shape()[1];
        let n = rows.len();
        assert!(n > 0, "apply_ratio called with no rows");

        let resampler = &self.resamplers[ratio_idx];
        let (num, den) = self.shifts[ratio_idx];
        // `rate = 1 / shift = den / num`. `phase_vocoder` stretches time
        // by 1/rate, so for shift>1 (higher pitch) we pass rate<1 here
        // and get a longer spectrogram, which the downsample step then
        // compresses back to pitch-raised audio of approximately the
        // original length.
        let rate = den as f32 / num as f32;

        let idx_tensor = TensorHandle::<R>::new_contiguous(
            vec![n],
            self.client.create_from_slice(u32::as_bytes(rows)),
            u32::as_type_native_unchecked().storage_type(),
        );

        // 1. Gather rows.
        let gathered = alloc_2d(&self.client, n, time, self.dtype);
        let (cc, cd) = elemwise_launch(&self.client, n * time);
        gather_rows_kernel::launch::<f32, R>(
            &self.client,
            cc,
            cd,
            samples.clone().binding().into_tensor_arg(),
            idx_tensor.clone().binding().into_tensor_arg(),
            gathered.clone().binding().into_tensor_arg(),
        );

        // 2. Reflect-pad by n_fft/2 on each side.
        let pad = self.n_fft / 2;
        let padded_time = time + 2 * pad;
        let padded = alloc_2d(&self.client, n, padded_time, self.dtype);
        let (cc, cd) = elemwise_launch(&self.client, n * padded_time);
        reflect_pad_kernel::launch::<f32, R>(
            &self.client,
            cc,
            cd,
            gathered.binding().into_tensor_arg(),
            padded.clone().binding().into_tensor_arg(),
            pad as u32,
        );

        // 3. STFT — re / im shape: `(n, n_frames, n_freq)`.
        let (re, im) = stft(padded, self.window.clone(), self.n_fft, self.hop, self.dtype);

        // 4. Transpose last two dims → (n, n_freq, n_frames) for phase vocoder.
        let re_t = transpose_last_two(&self.client, re, self.dtype);
        let im_t = transpose_last_two(&self.client, im, self.dtype);

        // 5. Phase vocoder — output: (n, n_freq, n_frames_new).
        let (pv_re, pv_im) =
            phase_vocoder(re_t, im_t, self.phase_advance.clone(), rate, self.dtype);

        // 6. Transpose back to (n, n_frames_new, n_freq) for istft.
        let pv_re_b = transpose_last_two(&self.client, pv_re, self.dtype);
        let pv_im_b = transpose_last_two(&self.client, pv_im, self.dtype);

        // 7. iSTFT — output length = (n_frames_new - 1) * hop + n_fft.
        let signal = istft(pv_re_b, pv_im_b, self.window.clone(), self.hop, self.dtype);
        let sig_len = signal.shape()[1];

        // 8. Trim pad samples on each side. The centered STFT convention's
        // canonical output length is `(n_frames_new - 1) * hop`, which at
        // hop = n_fft/32 is `sig_len - n_fft`. We drop `pad` from each end
        // which is exactly that (pad = n_fft/2, two sides).
        let trimmed_time = sig_len - 2 * pad;
        let trimmed = alloc_2d(&self.client, n, trimmed_time, self.dtype);
        let (cc, cd) = elemwise_launch(&self.client, n * trimmed_time);
        slice_time_kernel::launch::<f32, R>(
            &self.client,
            cc,
            cd,
            signal.binding().into_tensor_arg(),
            trimmed.clone().binding().into_tensor_arg(),
            pad as u32,
        );

        // 9. Resample.
        let resampled = resampler.apply(trimmed, None);

        // 10. Crop/pad to original `time` length.
        if resampled.shape()[1] == time {
            resampled
        } else {
            let final_out = alloc_2d(&self.client, n, time, self.dtype);
            let (cc, cd) = elemwise_launch(&self.client, n * time);
            crop_or_pad_kernel::launch::<f32, R>(
                &self.client,
                cc,
                cd,
                resampled.binding().into_tensor_arg(),
                final_out.clone().binding().into_tensor_arg(),
            );
            final_out
        }
    }
}

impl<R: Runtime> Transform<R> for PitchShift<R> {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(samples.shape().len(), 2, "PitchShift expects (batch, time)");
        let batch = samples.shape()[0];
        let time = samples.shape()[1];
        assert!(batch > 0 && time > self.n_fft);

        // Per-row decisions happen on the host.
        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let semitones =
            sample_uniform_batch(batch, self.min_semitones, self.max_semitones, rng.host());

        // Group rows by nearest ratio.
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); self.shifts.len()];
        for (row, (m, s)) in mask.iter().zip(semitones.iter()).enumerate() {
            if *m < 0.5 {
                continue;
            }
            let target = 2f32.powf(*s / 12.0);
            let mut best = 0usize;
            let mut best_err = f32::INFINITY;
            for (i, &(n, d)) in self.shifts.iter().enumerate() {
                let err = (n as f32 / d as f32 - target).abs();
                if err < best_err {
                    best_err = err;
                    best = i;
                }
            }
            groups[best].push(row as u32);
        }

        // Short-circuit: every row skipped. Return input unchanged.
        if groups.iter().all(|g| g.is_empty()) {
            return samples;
        }

        // Allocate an output buffer and seed it with a copy of the input.
        let out = alloc_2d(&self.client, batch, time, self.dtype);
        let identity_indices: Vec<u32> = (0..batch as u32).collect();
        let identity_t = TensorHandle::<R>::new_contiguous(
            vec![batch],
            self.client.create_from_slice(u32::as_bytes(&identity_indices)),
            u32::as_type_native_unchecked().storage_type(),
        );
        let (cc, cd) = elemwise_launch(&self.client, batch * time);
        gather_rows_kernel::launch::<f32, R>(
            &self.client,
            cc,
            cd,
            samples.clone().binding().into_tensor_arg(),
            identity_t.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        );

        // Process each non-empty group.
        for (g_idx, rows) in groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let processed = self.apply_ratio(&samples, rows, g_idx);
            let n = rows.len();

            // Scatter `processed[g, :]` into `out[rows[g], :]`.
            let idx_tensor = TensorHandle::<R>::new_contiguous(
                vec![n],
                self.client.create_from_slice(u32::as_bytes(rows)),
                u32::as_type_native_unchecked().storage_type(),
            );
            let (cc, cd) = elemwise_launch(&self.client, n * time);
            scatter_rows_kernel::launch::<f32, R>(
                &self.client,
                cc,
                cd,
                processed.binding().into_tensor_arg(),
                idx_tensor.binding().into_tensor_arg(),
                out.clone().binding().into_tensor_arg(),
            );
        }

        out
    }
}

// --- helper launches ---------------------------------------------------------

fn alloc_2d<R: Runtime>(
    client: &ComputeClient<R>,
    rows: usize,
    cols: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    TensorHandle::<R>::new_contiguous(
        vec![rows, cols],
        client.empty(rows * cols * dtype.size()),
        dtype,
    )
}

/// `(cube_count, cube_dim)` for a one-thread-per-element launch.
///
/// Thin wrapper around the shared [`crate::kernels::elemwise_launch_dims`]
/// helper so every elementwise kernel in this module routes through
/// CubeCL's hardware-aware cube-count spread.
fn elemwise_launch<R: Runtime>(
    client: &ComputeClient<R>,
    num_elems: usize,
) -> (CubeCount, CubeDim) {
    elemwise_launch_dims(client, num_elems, 256)
}

fn transpose_last_two<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorHandle<R>,
    dtype: StorageType,
) -> TensorHandle<R> {
    let shape = input.shape().clone();
    assert_eq!(shape.len(), 3, "transpose_last_two expects rank-3 input");
    let num_elems: usize = shape.iter().product();
    let out_shape = vec![shape[0], shape[2], shape[1]];
    let out = TensorHandle::<R>::new_contiguous(
        out_shape,
        client.empty(num_elems * dtype.size()),
        dtype,
    );
    let (cc, cd) = elemwise_launch(client, num_elems);
    transpose_last_two_kernel::launch::<f32, R>(
        client,
        cc,
        cd,
        input.binding().into_tensor_arg(),
        out.clone().binding().into_tensor_arg(),
    );
    out
}

// --- kernels ----------------------------------------------------------------

/// `output[i, t] = input[indices[i], t]`.
#[cube(launch)]
pub(crate) fn gather_rows_kernel<F: Float>(
    input: &Tensor<F>,
    indices: &Tensor<u32>,
    output: &mut Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let t_out = output.shape(1);
    let b_out = pos / t_out;
    let t = pos - b_out * t_out;

    let t_in = input.shape(1);
    let src = indices[b_out] as usize;
    output[pos] = input[src * t_in + t];
}

/// `output[indices[i], t] = input[i, t]`.
///
/// Unlike the gather path we write by `indices[i]`, so if two input rows
/// point at the same output row the last writer wins. Our caller never
/// lets that happen (each batch row appears in at most one group).
#[cube(launch)]
pub(crate) fn scatter_rows_kernel<F: Float>(
    input: &Tensor<F>,
    indices: &Tensor<u32>,
    output: &mut Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= input.len() {
        terminate!();
    }
    let t_in = input.shape(1);
    let b_in = pos / t_in;
    let t = pos - b_in * t_in;

    let t_out = output.shape(1);
    let dst = indices[b_in] as usize;
    output[dst * t_out + t] = input[pos];
}

/// Reflect-pad along the last axis of a `(batch, time)` tensor.
///
/// Reflection excludes the boundary sample itself, so for input
/// `[a, b, c, d, e]` with `pad = 2` the output is `[c, b, a, b, c, d, e, d, c]`.
#[cube(launch)]
pub(crate) fn reflect_pad_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    pad: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let t_out = output.shape(1);
    let b = pos / t_out;
    let t = pos - b * t_out;

    let t_in = input.shape(1);
    let pad_u = pad as usize;
    let last = t_in - 1;

    // Three non-overlapping cases; written as mutations rather than an
    // if-in-expression because the cube macro translates statement form
    // more reliably.
    #[allow(unused_assignments)]
    let mut src: usize = 0;
    if t < pad_u {
        src = pad_u - t;
    } else if t - pad_u < t_in {
        src = t - pad_u;
    } else {
        let j = t - pad_u;
        src = last + last - j;
    }
    output[pos] = input[b * t_in + src];
}

/// `output[b, y, x] = input[b, x, y]` — transpose the last two dims of a
/// rank-3 tensor.
#[cube(launch)]
pub(crate) fn transpose_last_two_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
) {
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

    // input shape: (out_d0, out_d2, out_d1) = (out_d0, in_d1, in_d2)
    // where in_d1 = out_d2 and in_d2 = out_d1.
    let in_idx = b * out_d1 * out_d2 + x * out_d1 + y;
    output[pos] = input[in_idx];
}

/// `output[b, t] = input[b, t + start]`. Output length is taken from
/// `output.shape(1)`; the kernel reads `start..start + output.shape(1)`
/// from the corresponding row of `input`.
#[cube(launch)]
pub(crate) fn slice_time_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    start: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let t_out = output.shape(1);
    let b = pos / t_out;
    let t = pos - b * t_out;
    let t_in = input.shape(1);
    output[pos] = input[b * t_in + t + start as usize];
}

/// Copy input into output, cropping if output is shorter and zero-padding
/// if output is longer.
#[cube(launch)]
pub(crate) fn crop_or_pad_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let t_out = output.shape(1);
    let b = pos / t_out;
    let t = pos - b * t_out;
    let t_in = input.shape(1);

    let mut v = F::new(0.0);
    if t < t_in {
        v = input[b * t_in + t];
    }
    output[pos] = v;
}
