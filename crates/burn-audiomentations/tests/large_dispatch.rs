//! Regression test for the 1-D `CubeCount` overflow that crashed training
//! at `batch_size >= 64`.
//!
//! The elementwise launchers used to compute `CubeCount::new_1d(
//! num_elems.div_ceil(256))`. wgpu's guaranteed per-axis limit on
//! `maxComputeWorkgroupsPerDimension` is 65535, so any call with
//! `num_elems > 65535 * 256 = 16_777_216` was rejected by the driver.
//!
//! This test drives `Gain` at `batch * time = 16_777_472` (exactly 65537
//! cubes, 2 past the 1-D limit). With the fix in `kernels.rs` it spreads
//! across X/Y via `calculate_cube_count_elemwise`; without the fix the
//! launch would fail validation at the wgpu layer.

mod common;

use burn_audiomentations::{Gain, Transform, TransformRng};
use common::{client, read_tensor, upload_2d, Runtime};

/// `num_elems = 16_777_472 = 65537 * 256`. Minimal-over-threshold size so
/// the test allocates ~67 MB per tensor rather than a full GB.
const BATCH: usize = 1;
const TIME: usize = 16_777_472;

#[test]
fn gain_over_1d_dispatch_cap() {
    let client = client();

    // Constant input → constant expected output = input * 10^(6/20).
    let sig = vec![1.0f32; BATCH * TIME];
    let expected = 10f32.powf(6.0 / 20.0);

    let g = Gain::new(6.0, 6.0, 1.0);
    let mut rng = TransformRng::new(0);
    let out = <Gain as Transform<Runtime>>::apply(
        &g,
        upload_2d(&client, &sig, BATCH, TIME),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);

    assert_eq!(out_host.len(), BATCH * TIME);
    // Spot-check the endpoints plus one interior sample well past the
    // 65535-cube boundary. Failing only the tail would indicate an axis
    // overshoot; failing the head would indicate the launch never ran.
    for &i in &[0usize, 1, 16_776_960, TIME - 1] {
        let got = out_host[i];
        assert!(
            (got - expected).abs() < 1e-5,
            "idx {i}: got {got}, want {expected}",
        );
    }
}
