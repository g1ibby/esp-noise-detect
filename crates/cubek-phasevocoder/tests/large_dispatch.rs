//! Regression test for the 1-D `CubeCount` overflow in
//! `phase_vocoder_kernel`.
//!
//! The kernel launches one thread per `(batch, freq)` pair: before the fix
//! the launcher used `CubeCount::new_1d((batch * n_freq) / 256)`. wgpu
//! rejects the dispatch once that value exceeds 65535.
//!
//! Picks `batch = 8192`, `n_freq = 2049`, `n_frames_in = 1`, `rate = 2.0`.
//! `batch * n_freq = 16_785_408 = 65568 * 256` — just past the 65535 cube
//! cap along X. With the fix the launcher spreads into 2-D via
//! `calculate_cube_count_elemwise`.
//!
//! Correctness check: with `n_in = 1` and `rate = 2.0`, `n_out = ceil(1/2)
//! = 1`. At `t = 0`, `time_step = 0`, `idx0 = 0`, `alpha = 0`, `idx1 = 1`
//! is out of range, so `mag = |input[0]|`, `phase_acc =
//! atan2(im[0], re[0])`, and the output bin is exactly `input[0]`.

mod common;

use common::{client, dtype_f32, phase_advance_default, read_tensor, upload_1d, upload_3d, Runtime};
use cubek_phasevocoder::phase_vocoder;

const BATCH: usize = 8192;
const N_FREQ: usize = 2049;
const N_IN: usize = 1;
const RATE: f32 = 2.0;

#[test]
fn phase_vocoder_over_1d_dispatch_cap() {
    let client = client();

    // Deterministic but non-trivial input: re[k] = 1 + k mod 7 as f32,
    // im[k] = (k mod 5) as f32. Using small integers keeps the expected
    // output exactly representable so we don't have to dial the tolerance.
    let len = BATCH * N_FREQ * N_IN;
    let re: Vec<f32> = (0..len).map(|k| 1.0 + (k % 7) as f32).collect();
    let im: Vec<f32> = (0..len).map(|k| (k % 5) as f32).collect();

    let pa = phase_advance_default(N_FREQ, 1);

    let (out_re, out_im) = phase_vocoder::<Runtime>(
        upload_3d(&client, &re, BATCH, N_FREQ, N_IN),
        upload_3d(&client, &im, BATCH, N_FREQ, N_IN),
        upload_1d(&client, &pa),
        RATE,
        dtype_f32(),
    );
    let out_re_host = read_tensor(&client, out_re);
    let out_im_host = read_tensor(&client, out_im);

    let n_out: usize = ((N_IN as f64) / (RATE as f64)).ceil() as usize;
    assert_eq!(n_out, 1);
    assert_eq!(out_re_host.len(), BATCH * N_FREQ * n_out);

    // At t=0 with idx1 out of range, the kernel should emit exactly the
    // input sample. Spot-check a few indices straddling the 65535-cube
    // boundary (cube #65535 starts at flat thread 65535*256 = 16_776_960).
    for &k in &[0usize, 1, 16_776_960, len - 1] {
        let got_re = out_re_host[k];
        let got_im = out_im_host[k];
        let want_re = re[k];
        let want_im = im[k];
        assert!(
            (got_re - want_re).abs() < 1e-4,
            "idx {k}: got_re {got_re}, want {want_re}",
        );
        assert!(
            (got_im - want_im).abs() < 1e-4,
            "idx {k}: got_im {got_im}, want {want_im}",
        );
    }
}
