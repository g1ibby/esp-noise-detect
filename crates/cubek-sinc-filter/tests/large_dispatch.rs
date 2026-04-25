//! Regression test for the 1-D `CubeCount` overflow in the sinc-filter
//! kernels.
//!
//! `LowPassFilterBank::apply_single` and `apply_per_row` launched
//! `CubeCount::new_1d((batch * time) / 256)`. wgpu rejects that at
//! `batch * time > 16_777_216`. This test drives `apply_single` at
//! `batch = 1`, `time = 16_777_472` — exactly 65537 cubes, 2 past the 1-D
//! limit.
//!
//! Correctness check: a DC input (all-ones) passes through a low-pass
//! filter with the bank's DC-normalized weights unchanged. Spot-check a
//! handful of output samples that straddle the 65535-cube boundary.

mod common;

use common::{client, dtype_f32, read_tensor, upload_2d, Runtime};
use cubek_sinc_filter::{FilterMode, LowPassFilterBank};

const BATCH: usize = 1;
const TIME: usize = 16_777_472;

#[test]
fn sinc_filter_over_1d_dispatch_cap() {
    let client = client();

    let sig = vec![1.0f32; BATCH * TIME];
    let bank = LowPassFilterBank::<Runtime>::new(client.clone(), &[0.25f32], 8, dtype_f32());

    let out = bank.apply_single(
        upload_2d(&client, &sig, BATCH, TIME),
        0,
        FilterMode::LowPass,
    );
    let out_host = read_tensor(&client, out);
    assert_eq!(out_host.len(), BATCH * TIME);

    // DC passthrough: interior samples should be ~1.0. Stay away from the
    // replicate-padded edges and probe around the 65535-cube boundary
    // (flat index 16_776_960).
    let margin = 256;
    for &i in &[margin, 16_776_000, 16_777_000, TIME - margin - 1] {
        let got = out_host[i];
        assert!(
            (got - 1.0).abs() < 1e-3,
            "idx {i}: got {got}, want ~1.0 (DC passthrough)",
        );
    }
}
