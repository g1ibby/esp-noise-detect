//! `TimeMasking` — crossfaded splice-out with silent tail pad.
//!
//! Each selected batch row has `num_time_intervals` segments of random
//! width spliced out. A splice at `(start, length)` on the current working
//! array replaces `sample[start..start+length/2]` with a Hann-weighted
//! crossfade between the left and right halves of the cut region, deletes
//! `length/2` samples, and zero-pads the tail so the final length is
//! preserved. Intervals are applied sequentially (each sees the result of
//! the previous), matching the Python reference.
//!
//! ## Kernel strategy
//!
//! One kernel launch per interval, alternating between a front and a back
//! scratch buffer. Each launch reads `(start, length)` for every row from a
//! small `(batch, 2)` u32 tensor uploaded from the host and writes one
//! output sample per thread. Length-0 intervals (unselected rows, or the
//! Python "randint low=0 hi=..." producing 0) collapse to a straight copy.
//!
//! The sequential launch model is chosen over a single fused kernel because
//! SpliceOut's per-interval cumulative shift is much easier to express as
//! "apply splice i to input, produce output" than as a fused index
//! remapping. `num_time_intervals` is ≤ 2 for our `robust.yaml` config, so
//! the extra launches cost nothing meaningful.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use rand::Rng;

use crate::kernels::elemwise_launch_dims;
use crate::transform::{Transform, TransformRng, bernoulli_mask};

/// SpliceOut-equivalent time masking.
///
/// * `num_time_intervals` — number of splice operations applied per row.
/// * `max_width_samples` — upper bound on `length`, exclusive. Each
///   interval samples `length ∈ [0, max_width_samples)` and then rounds up
///   to the nearest even number (Hann crossfade wants a matched pair of
///   halves).
/// * `probability` — Bernoulli per-row apply probability.
pub struct TimeMasking {
    pub num_time_intervals: u32,
    pub max_width_samples: u32,
    pub probability: f64,
}

impl TimeMasking {
    pub fn new(num_time_intervals: u32, max_width_samples: u32, probability: f64) -> Self {
        assert!(num_time_intervals > 0, "num_time_intervals must be > 0");
        assert!(max_width_samples > 0, "max_width_samples must be > 0");
        Self {
            num_time_intervals,
            max_width_samples,
            probability,
        }
    }
}

impl<R: Runtime> Transform<R> for TimeMasking {
    fn apply(&self, samples: TensorHandle<R>, rng: &mut TransformRng) -> TensorHandle<R> {
        assert_eq!(samples.shape().len(), 2, "TimeMasking expects (batch, time)");
        let batch = samples.shape()[0];
        let time = samples.shape()[1];
        assert!(batch > 0 && time > 0);

        let mask = bernoulli_mask(batch, self.probability, rng.host());
        let client = <R as Runtime>::client(&Default::default());
        let dtype = samples.dtype;

        // Ping-pong buffers — one kernel launch per interval.
        let mut current = samples;
        let mut scratch: Option<TensorHandle<R>> = None;

        for _interval in 0..self.num_time_intervals {
            // Per-row sample (start, length). Length is clamped to even
            // because the crossfade needs matched halves; rows where
            // mask==0 get length=0 (identity pass-through in the kernel).
            let mut params: Vec<u32> = Vec::with_capacity(batch * 2);
            let host = rng.host();
            for b in 0..batch {
                let selected = mask[b] > 0.5;
                let raw_len: u32 = if selected && self.max_width_samples > 1 {
                    host.random_range(0..self.max_width_samples)
                } else {
                    0
                };
                // Even, and bounded so start + length <= time - 1 is always feasible.
                let length = (raw_len & !1).min(time as u32 - 2);
                let start = if length == 0 {
                    0
                } else {
                    host.random_range(0..(time as u32 - length))
                };
                params.push(start);
                params.push(length);
            }

            let param_handle = client.create_from_slice(u32::as_bytes(&params));
            let params_t = TensorHandle::<R>::new_contiguous(
                vec![batch, 2],
                param_handle,
                u32::as_type_native_unchecked().storage_type(),
            );

            // Lazily allocate the scratch buffer on first iteration.
            let out = match scratch.take() {
                Some(buf) => buf,
                None => TensorHandle::<R>::new_contiguous(
                    vec![batch, time],
                    client.empty(batch * time * dtype.size()),
                    dtype,
                ),
            };

            let num_elems = batch * time;
            let (cube_count, cube_dim) = elemwise_launch_dims(&client, num_elems, 256);

            splice_out_kernel::launch::<f32, R>(
                &client,
                cube_count,
                cube_dim,
                current.clone().binding().into_tensor_arg(),
                params_t.binding().into_tensor_arg(),
                out.clone().binding().into_tensor_arg(),
            );

            // out becomes the new current; the old current becomes the
            // scratch buffer for the next iteration.
            scratch = Some(current);
            current = out;
        }

        current
    }
}

/// Per output `(b, t')`:
///
/// Let `(s, L)` = `params[b]`, `T = input.shape(1)`.
///
/// * `t' < s` → `out = in[b, t']`
/// * `s <= t' < s + L/2` → crossfade: `k = t' - s`, `out = hann_r[k] *
///   in[b, s+k] + hann_l[k] * in[b, s + L/2 + k]`, where `hann_l[k] =
///   0.5 * (1 - cos(2π * k / (L - 1)))` and `hann_r[k] = hann_l[L/2 + k]`
///   (a standard Hann window of length `L` cut in half).
/// * `s + L/2 <= t' < T - L/2` → `out = in[b, t' + L/2]`
/// * `t' >= T - L/2` → `out = 0` (tail padding)
///
/// `L == 0` collapses to an unconditional copy: the first branch fires for
/// `t' < s` (with `s == 0`, never), second branch `t' < 0 + 0` never,
/// third branch `t' < T - 0` always — so `out = in[b, t']`. ✓
#[cube(launch)]
pub(crate) fn splice_out_kernel<F: Float>(
    input: &Tensor<F>,
    params: &Tensor<u32>,
    output: &mut Tensor<F>,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let time = input.shape(1);
    let b = pos / time;
    let t_prime = pos - b * time;

    let start = params[b * 2] as usize;
    let length = params[b * 2 + 1] as usize;
    let half = length / 2;
    let row_base = b * time;

    let mut out_val = F::new(0.0);

    if t_prime < start {
        // Pre-splice: copy directly.
        out_val = input[row_base + t_prime];
    } else if t_prime < start + half {
        // Crossfade region. Hann window `0.5 * (1 - cos(2π n / (L - 1)))`,
        // split at the midpoint into left (n = k) and right (n = half + k)
        // slopes. `length` is even and >= 2 here, so `length - 1 >= 1`.
        let k = t_prime - start;
        let denom = F::cast_from(length as u32 - 1u32);
        let two_pi = F::new(2.0 * core::f32::consts::PI);
        let hann_l = F::new(0.5) * (F::new(1.0) - F::cos(two_pi * F::cast_from(k as u32) / denom));
        let hann_r = F::new(0.5)
            * (F::new(1.0) - F::cos(two_pi * F::cast_from((half + k) as u32) / denom));
        let fading_out = input[row_base + start + k];
        let fading_in = input[row_base + start + half + k];
        out_val = hann_r * fading_out + hann_l * fading_in;
    } else if t_prime + half < time {
        // Post-splice: skip forward by half — the deleted right half of
        // the spliced region is gone, everything past it shifts left.
        out_val = input[row_base + t_prime + half];
    }
    // Else: tail padding, leave out_val at 0.

    output[pos] = out_val;
}
