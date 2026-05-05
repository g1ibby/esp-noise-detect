//! Step 3 — FP32 forward-parity test.
//!
//! Build the inline `MiniNet` fixture with non-trivial parameters
//! (random init + perturbed BN stats), lower it into the IR, and
//! verify the IR forward executor reproduces the model's own
//! `forward()` output to ≤ 1e-4 (pre-fold/pre-fuse) and ≤ 1e-3
//! (post-fold/post-fuse — BN folding changes accumulation order in
//! FP32, costing about 1 OoM of precision).
//!
//! No production model, no checkpoint loading, no `/tmp/...`
//! artifacts. The test is fully self-contained: it constructs the
//! model in memory, runs it on a synthetic input, and compares.

use burn::backend::NdArray;
use burn::tensor::{Tensor as BurnTensor, backend::Backend};
use burn_espdl_export::ir::{self, fold_batchnorm, fuse_relu};

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::{MiniNetConfig, perturb_bn_stats};

type B = NdArray;

const INPUT_SHAPE: [usize; 4] = [1, 1, 16, 16];

#[test]
fn pre_fold_ir_matches_fixture_forward() {
    let device: <B as Backend>::Device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0xc0ffee_u64);

    let input = sample_input(&device, 0xa5a5_5a5a_u64);

    // Reference: the model's own forward.
    let reference = model.forward(input.clone());
    let ref_v: Vec<f32> = reference.into_data().convert::<f32>().into_vec().unwrap();

    // IR forward, pre-fold/pre-fuse: BN and Relu are still standalone
    // layers, executed by the IR executor branch-by-branch.
    let graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    let logits = ir::forward(&graph, input, &device);
    let v: Vec<f32> = logits.into_data().convert::<f32>().into_vec().unwrap();

    let max_diff = max_abs_diff(&v, &ref_v);
    // Same op set, same operand order — should be bit-identical, but
    // we allow a tiny FP epsilon so the test is robust to platform
    // libm jitter.
    assert!(
        max_diff <= 1e-4,
        "pre-fold IR vs MiniNet forward: max-abs-diff {max_diff} > 1e-4 \
         (got {v:?}, ref {ref_v:?})",
    );
}

#[test]
fn post_fold_post_fuse_ir_matches_fixture_forward() {
    let device: <B as Backend>::Device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0xdeadbeef_u64);

    let input = sample_input(&device, 0x1234_5678_u64);

    let reference = model.forward(input.clone());
    let ref_v: Vec<f32> = reference.into_data().convert::<f32>().into_vec().unwrap();

    let mut graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);
    let logits = ir::forward(&graph, input, &device);
    let v: Vec<f32> = logits.into_data().convert::<f32>().into_vec().unwrap();

    let max_diff = max_abs_diff(&v, &ref_v);
    // BN folding folds γ/√(σ²+ε) into the Conv weight; FP32
    // accumulation order changes, so we relax to 1e-3 (still well
    // under the 1e-2 the task's Step-7 acceptance budget allows).
    assert!(
        max_diff <= 1e-3,
        "post-fold/post-fuse IR vs MiniNet forward: max-abs-diff {max_diff} > 1e-3 \
         (got {v:?}, ref {ref_v:?})",
    );
}

#[test]
fn ir_post_fold_is_invariant_across_inputs() {
    // Sanity: parity should hold for any input the executor sees,
    // not just one. Run a small batch through the same fixture.
    let device: <B as Backend>::Device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0x4242_4242_u64);

    let mut graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);

    for seed in 0..8u64 {
        let input = sample_input(&device, seed.wrapping_mul(0x9E37_79B9));
        let ref_v: Vec<f32> = model
            .forward(input.clone())
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();
        let v: Vec<f32> = ir::forward(&graph, input, &device)
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();
        let d = max_abs_diff(&v, &ref_v);
        assert!(d <= 1e-3, "seed {seed}: diff {d} > 1e-3");
    }
}

/// Deterministic synthetic input drawn from N(0, 1). Uses an
/// in-test LCG → Box-Muller pair so the input is reproducible across
/// platforms without relying on Burn's global RNG seed.
fn sample_input<B: Backend>(device: &B::Device, seed: u64) -> BurnTensor<B, 4> {
    let n: usize = INPUT_SHAPE.iter().product();
    let mut state = seed.max(1);
    let mut data = Vec::with_capacity(n);
    // Box-Muller via a pair of LCG draws — keeps the tensor
    // deterministically Gaussian-ish without pulling in `rand`.
    while data.len() < n {
        let u1 = lcg_unit(&mut state).max(1e-9);
        let u2 = lcg_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f32::consts::PI * u2;
        data.push(r * theta.cos());
        if data.len() < n {
            data.push(r * theta.sin());
        }
    }
    BurnTensor::<B, 4>::from_data(burn::tensor::TensorData::new(data, INPUT_SHAPE), device)
}

fn lcg_unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 32) as u32 as f32) / (u32::MAX as f32)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
