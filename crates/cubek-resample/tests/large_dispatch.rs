//! Regression test for the 1-D `CubeCount` overflow in the resample
//! kernel.
//!
//! `Resampler::apply` launched `CubeCount::new_1d(num_elems / 256)` where
//! `num_elems = batch * out_len`. At `out_len > 16_777_216` wgpu rejected
//! the dispatch. This test upsamples 1 → 2 on a signal long enough that
//! `out_len = 16_778_000 > 16_777_216`.
//!
//! Correctness check: a DC signal (all-ones) is preserved by every
//! DC-normalized polyphase FIR — the kernels in `KernelBank` are built to
//! sum to 1. Spot-check a few output samples around the 65535-cube
//! boundary, excluding the replicate-padded edges where numerical precision
//! is a little looser.

mod common;

use common::{client, dtype_f32, read_tensor, upload_2d, Runtime};
use cubek_resample::Resampler;

const BATCH: usize = 1;
const OLD_SR: u32 = 1;
const NEW_SR: u32 = 2;
const TIME: usize = 8_389_000; // out_len = 2 * TIME = 16_778_000

#[test]
fn resample_over_1d_dispatch_cap() {
    let client = client();

    let sig = vec![1.0f32; BATCH * TIME];
    let resampler = Resampler::<Runtime>::new(client.clone(), OLD_SR, NEW_SR, 24, 0.945, dtype_f32());
    let out = resampler.apply(upload_2d(&client, &sig, BATCH, TIME), None);

    let out_shape = out.shape().clone();
    let out_host = read_tensor(&client, out);
    let out_len = out_shape[1];
    assert_eq!(out_len, 2 * TIME);
    assert_eq!(out_host.len(), BATCH * out_len);
    assert!(
        out_len > 16_777_216,
        "test must push past the 1-D dispatch cap; out_len = {out_len}",
    );

    // DC passthrough: every interior output sample should be ~1.0. Stay
    // away from both edges (replicate padding rings) and probe a range
    // that straddles the 65535-cube boundary (starts at flat index
    // 65535*256 = 16_776_960).
    let margin = 1024;
    for &i in &[margin, 16_776_000, 16_777_000, out_len - margin - 1] {
        let got = out_host[i];
        assert!(
            (got - 1.0).abs() < 1e-3,
            "idx {i}: got {got}, want ~1.0 (DC passthrough)",
        );
    }
}
