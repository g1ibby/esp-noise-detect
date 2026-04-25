//! `AddColoredNoise` — `1/f^α` noise at a target SNR.
//!
//! Algorithm per selected batch row `b`:
//!
//! 1. Draw `num_samples` white-noise samples `η ~ N(0, 1)` on device.
//! 2. Compute `η̂ = rfft(η)`, multiply bin `k` by `1 / linspace(1,
//!    √(fs/2), n_freq)[k]^(f_decay[b])`, and take `irfft(η̂)`.
//! 3. RMS-normalize the shaped noise to unit RMS.
//! 4. Scale by `signal_rms[b] / 10^(snr_in_db[b] / 20)` and add to the
//!    signal.
//!
//! ## Chunking
//!
//! `cubek-fft` caps `n_fft` at 4096 on wgpu-Metal.
//! The pump-noise window is 32000 samples. We satisfy the cap by splitting
//! the noise buffer into `n_chunks` power-of-two chunks of size `chunk`,
//! shaping each chunk independently, and concatenating (truncated to the
//! exact `num_samples`). Every chunk uses a different white-noise draw, so
//! the result has no obvious periodicity and is still 1/f^α in
//! expectation.
//!
//! The chunk size is chosen as `max_pow2_le(num_samples) <= 4096` at
//! construction time. For the 32000-sample input this is 4096, giving 8
//! chunks. If the noise generator is ever invoked on a waveform ≤ 4096
//! samples, `n_chunks == 1` and behaviour reduces to the non-tiled case.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_fft::{irfft, rfft};
use cubek_random::random_normal;

use crate::kernels::{add_with_scale_kernel, elemwise_launch_dims, per_row_sum_sq_kernel};
use crate::transform::{Transform, TransformRng, bernoulli_mask, sample_uniform_batch};

/// Max per-chunk FFT length (wgpu-Metal radix-2 limit). Copied from
/// `cubek-stft::MAX_N_FFT` to avoid a dependency on the STFT crate.
const MAX_CHUNK: usize = 4096;

pub struct AddColoredNoise {
    pub min_snr_in_db: f32,
    pub max_snr_in_db: f32,
    pub min_f_decay: f32,
    pub max_f_decay: f32,
    pub sample_rate: u32,
    pub probability: f64,
}

impl AddColoredNoise {
    pub fn new(
        min_snr_in_db: f32,
        max_snr_in_db: f32,
        min_f_decay: f32,
        max_f_decay: f32,
        sample_rate: u32,
        probability: f64,
    ) -> Self {
        assert!(min_snr_in_db <= max_snr_in_db);
        assert!(min_f_decay <= max_f_decay);
        assert!(sample_rate > 0);
        Self {
            min_snr_in_db,
            max_snr_in_db,
            min_f_decay,
            max_f_decay,
            sample_rate,
            probability,
        }
    }
}

impl<R: Runtime> Transform<R> for AddColoredNoise {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(samples.shape().len(), 2, "AddColoredNoise expects (batch, time)");
        let batch = samples.shape()[0];
        let num_samples = samples.shape()[1];
        assert!(batch > 0 && num_samples > 0);

        // 1. Host-side parameter sampling.
        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let snr_db =
            sample_uniform_batch(batch, self.min_snr_in_db, self.max_snr_in_db, rng.host());
        let f_decay =
            sample_uniform_batch(batch, self.min_f_decay, self.max_f_decay, rng.host());

        // 2. Chunk layout. We want the largest power-of-two chunk size that
        //    both fits the 4096-sample FFT cap and is ≤ num_samples.
        let chunk = {
            let max_pow2_le_n = 1usize << (num_samples.ilog2() as usize);
            max_pow2_le_n.min(MAX_CHUNK)
        };
        let n_chunks = num_samples.div_ceil(chunk);
        let padded = n_chunks * chunk;

        let client = <R as Runtime>::client(&Default::default());
        let dtype = samples.dtype;

        // 3. Generate white noise directly into a (batch, n_chunks, chunk)
        //    buffer. cubek-random's process-global seed is advanced per
        //    batch for reproducibility (see TransformRng::seed_cubek_random).
        rng.seed_cubek_random();
        let noise = TensorHandle::<R>::new_contiguous(
            vec![batch, n_chunks, chunk],
            client.empty(batch * n_chunks * chunk * dtype.size()),
            dtype,
        );
        random_normal::<R>(&client, 0.0, 1.0, noise.clone().binding(), dtype)
            .expect("cubek-random launch failed");

        // 4. rfft along the chunk axis, apply 1/f^α mask, irfft back.
        let (spec_re, spec_im) = rfft(noise, 2, dtype);
        let n_freq = spec_re.shape()[2];

        // Upload (batch,) f_decay for the mask kernel.
        let f_decay_handle = client.create_from_slice(f32::as_bytes(&f_decay));
        let f_decay_t =
            TensorHandle::<R>::new_contiguous(vec![batch], f_decay_handle, dtype);

        // `linspace(1, sqrt(sample_rate/2), n_freq)` frequency envelope.
        let fs_sqrt = (self.sample_rate as f32 / 2.0).sqrt();
        let lin_vec: Vec<f32> = (0..n_freq)
            .map(|k| {
                if n_freq == 1 {
                    1.0
                } else {
                    1.0 + (fs_sqrt - 1.0) * (k as f32) / (n_freq as f32 - 1.0)
                }
            })
            .collect();
        let lin_handle = client.create_from_slice(f32::as_bytes(&lin_vec));
        let lin = TensorHandle::<R>::new_contiguous(vec![n_freq], lin_handle, dtype);

        let spec_elems = batch * n_chunks * n_freq;
        let (cube_count, cube_dim) = elemwise_launch_dims(&client, spec_elems, 256);
        colored_mask_kernel::launch::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            spec_re.clone().binding().into_tensor_arg(),
            spec_im.clone().binding().into_tensor_arg(),
            lin.binding().into_tensor_arg(),
            f_decay_t.binding().into_tensor_arg(),
        );

        // irfft is applied per-row along the last dim.
        let shaped = irfft(spec_re, spec_im, 2, dtype);
        // Reshape (batch, n_chunks, chunk) → (batch, padded) conceptually;
        // since the storage is already contiguous we just re-label.
        let shaped_2d = TensorHandle::<R>::new_contiguous(
            vec![batch, padded],
            shaped.handle,
            dtype,
        );
        // `padded >= num_samples` always; if padded > num_samples we'll
        // just ignore the trailing samples in the mix kernel by reading
        // from `[0, num_samples)`.

        // 5. Per-row RMS of the shaped noise and of the signal.
        let noise_sumsq = TensorHandle::<R>::new_contiguous(
            vec![batch],
            client.empty(batch * dtype.size()),
            dtype,
        );
        let signal_sumsq = TensorHandle::<R>::new_contiguous(
            vec![batch],
            client.empty(batch * dtype.size()),
            dtype,
        );
        // One thread per batch row. Routed through the shared elemwise
        // launcher so we inherit CubeCL's per-backend `max_cube_count`
        // spread instead of hitting the per-axis cap at large batches.
        let row_launch = || elemwise_launch_dims(&client, batch, 64);
        let (row_cc, row_cd) = row_launch();
        per_row_sum_sq_kernel::launch::<f32, R>(
            &client,
            row_cc,
            row_cd,
            shaped_2d.clone().binding().into_tensor_arg(),
            noise_sumsq.clone().binding().into_tensor_arg(),
        );
        let (row_cc, row_cd) = row_launch();
        per_row_sum_sq_kernel::launch::<f32, R>(
            &client,
            row_cc,
            row_cd,
            samples.clone().binding().into_tensor_arg(),
            signal_sumsq.clone().binding().into_tensor_arg(),
        );

        // 6. Combine RMS / SNR / mask into a per-row mix scale.
        //    scale[b] = mask[b] * rms(signal[b]) / rms(noise[b]) / 10^(snr/20)
        //    We read the two reductions back to the host, fuse the math
        //    there (it's `batch` f32 values — nothing), then re-upload as
        //    one (batch,) tensor to feed `add_with_scale_kernel`.
        let noise_sumsq_host = client
            .read_one_unchecked_tensor(noise_sumsq.into_copy_descriptor())
            .to_vec();
        let signal_sumsq_host = client
            .read_one_unchecked_tensor(signal_sumsq.into_copy_descriptor())
            .to_vec();
        let noise_sumsq_f = f32::from_bytes(&noise_sumsq_host);
        let signal_sumsq_f = f32::from_bytes(&signal_sumsq_host);

        let scales: Vec<f32> = (0..batch)
            .map(|b| {
                if mask[b] < 0.5 {
                    return 0.0;
                }
                let n_rms = (noise_sumsq_f[b] / padded as f32).sqrt().max(1e-12);
                let s_rms = (signal_sumsq_f[b] / num_samples as f32).sqrt();
                (s_rms / n_rms) / 10f32.powf(snr_db[b] / 20.0)
            })
            .collect();

        // 7. Add shaped noise into the signal.
        //    We reinterpret `shaped_2d` as a (batch, num_samples) slice by
        //    creating a view at the same offset — the storage is already
        //    contiguous and row-major and the trailing `padded - num_samples`
        //    columns are never read by the add kernel.
        let noise_signal_view = TensorHandle::<R>::new_contiguous(
            vec![batch, padded],
            shaped_2d.handle.clone(),
            dtype,
        );

        let scale_handle = client.create_from_slice(f32::as_bytes(&scales));
        let scale_t = TensorHandle::<R>::new_contiguous(vec![batch], scale_handle, dtype);

        let out = TensorHandle::<R>::new_contiguous(
            vec![batch, num_samples],
            client.empty(batch * num_samples * dtype.size()),
            dtype,
        );
        let add_elems = batch * num_samples;
        let (add_cube_count, add_cube_dim) = elemwise_launch_dims(&client, add_elems, 256);
        add_with_scale_kernel::launch::<f32, R>(
            &client,
            add_cube_count,
            add_cube_dim,
            samples.binding().into_tensor_arg(),
            noise_signal_view.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
            scale_t.binding().into_tensor_arg(),
        );
        out
    }
}

/// `spec[b, c, k] *= 1 / lin[k]^f_decay[b]` — one thread per spectrum bin.
///
/// The 1/f^α envelope is applied identically to the real and imaginary
/// parts (it's a real-valued magnitude mask), preserving the phase of
/// each white-noise bin.
#[cube(launch)]
pub(crate) fn colored_mask_kernel<F: Float>(
    spec_re: &mut Tensor<F>,
    spec_im: &mut Tensor<F>,
    lin: &Tensor<F>,
    f_decay: &Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= spec_re.len() {
        terminate!();
    }
    let n_freq = spec_re.shape(2);
    let n_chunks = spec_re.shape(1);
    let k = pos % n_freq;
    let b = pos / (n_chunks * n_freq);

    let mask = F::new(1.0) / F::powf(lin[k], f_decay[b]);
    spec_re[pos] = spec_re[pos] * mask;
    spec_im[pos] = spec_im[pos] * mask;
}
