//! Intra-cube-parallel radix-2 FFT primitives used by both `rfft_kernel`
//! and `irfft_kernel`.
//!
//! Each cube processes one FFT window with `CUBE_DIM` units (one butterfly
//! per unit per stage, or several if `CUBE_DIM < n_fft / 2`). Inputs and
//! outputs live in per-cube `SharedMemory<F>`; callers are responsible for
//! loading the window (with bit-reversed indices) before calling
//! [`fft_butterfly_parallel`], and for draining the shared memory back out
//! to global tensors after.
//!
//! Invariants that callers must preserve:
//! * `n_fft` is a power of two, and `log2_n = trailing_zeros(n_fft)`.
//! * `threads_per_cube` equals the actual `CUBE_DIM` at launch time.
//! * A `sync_cube()` follows the bit-reversed load **before** this is called.

use core::f32::consts::PI;

use cubecl::prelude::*;

use crate::fft::FftMode;

/// Reverse the lowest `log2_n` bits of `i`.
#[cube]
pub(crate) fn bit_reverse(i: usize, #[comptime] log2_n: usize) -> usize {
    let mut j = 0usize;
    let mut x = i;
    #[unroll]
    for _ in 0..log2_n {
        j = (j << 1usize) | (x & 1usize);
        x = x >> 1usize;
    }
    j
}

/// Parallel radix-2 butterfly stages on an already-bit-reversed window in
/// shared memory. Each unit processes butterflies at indices
/// `UNIT_POS, UNIT_POS + threads_per_cube, …`, synchronising between
/// stages. `threads_per_cube` must equal `CUBE_DIM` at launch time.
#[cube]
pub(crate) fn fft_butterfly_parallel<F: Float>(
    shared_re: &mut SharedMemory<F>,
    shared_im: &mut SharedMemory<F>,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] fft_mode: FftMode,
) {
    let num_butterflies = comptime![n_fft / 2];
    let sign = F::new(fft_mode.sign());
    let two_pi = F::new(2.0 * PI);

    let mut s = 0usize;
    while s < log2_n {
        let half_m = 1usize << s;
        let m = half_m << 1usize;

        let mut b = UNIT_POS as usize;
        while b < num_butterflies {
            let group = b / half_m;
            let j = b - group * half_m;
            let i0 = group * m + j;
            let i1 = i0 + half_m;

            let theta: F = sign * two_pi * F::cast_from(j) / F::cast_from(m);
            let w_re = theta.cos();
            let w_im = theta.sin();

            let ar = shared_re[i0];
            let ai = shared_im[i0];
            let br = shared_re[i1];
            let bi = shared_im[i1];

            let tr = w_re * br - w_im * bi;
            let ti = w_re * bi + w_im * br;

            shared_re[i0] = ar + tr;
            shared_im[i0] = ai + ti;
            shared_re[i1] = ar - tr;
            shared_im[i1] = ai - ti;

            b += threads_per_cube;
        }
        sync_cube();
        s += 1usize;
    }
}

/// Element offset of `window_index` inside `tensor`, skipping dimension
/// `dim`. For a `(batch, n_fft)` tensor with `dim = 1`, this returns the
/// offset of the start of the given window. `dim` is comptime; all other
/// inputs are runtime.
#[cube]
pub(crate) fn window_batch_offset<F: Numeric>(
    tensor: &Tensor<F>,
    window_index: usize,
    #[comptime] dim: usize,
) -> usize {
    let rank = tensor.rank();
    let mut offset = 0usize;
    let mut temp = window_index;
    let mut i = 0usize;
    while i < rank {
        if i != dim {
            let size = tensor.shape(i);
            let stride = tensor.stride(i);
            let coord = temp % size;
            offset += coord * stride;
            temp = temp / size;
        }
        i += 1usize;
    }
    offset
}
