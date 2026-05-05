//! Step 3 — BurnGraph extraction topology test.
//!
//! Build the inline `MiniNet` fixture (see `tests/common/fixture_model.rs`),
//! lower it into the IR, and assert:
//!
//! 1. Pre-fold/pre-fuse the IR has the un-rewritten shape:
//!    `(Conv → BN → Relu) × 2 × n_blocks` then `ReduceMean → Linear`.
//! 2. After `fold_batchnorm` + `fuse_relu` the IR collapses to
//!    `Conv × 2*n_blocks` then `ReduceMean → Gemm`, with bias and a
//!    fused activation on every Conv. That post-rewrite shape is
//!    what the writer in Step 5 expects to see.
//!
//! The test depends on no production model and no external artifact —
//! `MiniNet` is defined in test code with random initialization.

use burn::backend::NdArray;
use burn_espdl_export::ir::{BurnGraph, Layer, fold_batchnorm, fuse_relu};

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::MiniNetConfig;

type B = NdArray;

#[test]
fn extracted_pre_fold_topology_matches_fixture() {
    let device = Default::default();
    let cfg = MiniNetConfig::default(); // channels=[4,8], n_classes=2
    let n_blocks = cfg.channels.len();
    let model = cfg.init::<B>(&device);
    let graph = mininet_to_burn_graph(&model, [1, 1, 16, 16]);

    // n_blocks × (Conv, BN, Relu, Conv, BN, Relu) + ReduceMean + Linear.
    let expected_per_block = 6;
    assert_eq!(
        graph.layers.len(),
        n_blocks * expected_per_block + 2,
        "pre-fold layer count"
    );

    let expected_pre: Vec<&'static str> = {
        let mut v = Vec::with_capacity(n_blocks * expected_per_block + 2);
        for _ in 0..n_blocks {
            v.extend_from_slice(&[
                "Conv",
                "BatchNormalization",
                "Relu",
                "Conv",
                "BatchNormalization",
                "Relu",
            ]);
        }
        v.push("ReduceMean");
        v.push("Gemm");
        v
    };
    assert_eq!(graph.op_sequence(), expected_pre);

    assert_eq!(graph.input_name, "input");
    assert_eq!(graph.input_shape, [1, 1, 16, 16]);
    assert_eq!(graph.output_name, "logits");

    // Spot-check Conv shapes against MiniNet's construction
    // (in_ch=1 → 4 → 4 (refine) → 8 → 8 (refine), all 3×3, stride 2
    // on the first block's down conv, 1 elsewhere).
    let conv_layers: Vec<&Layer> = graph
        .layers
        .iter()
        .filter(|l| matches!(l, Layer::Conv2d { .. }))
        .collect();
    assert_eq!(conv_layers.len(), n_blocks * 2);
    assert_conv_shape(conv_layers[0], &[4, 1, 3, 3], [2, 2], [1, 1, 1, 1]);
    assert_conv_shape(conv_layers[1], &[4, 4, 3, 3], [1, 1], [1, 1, 1, 1]);
    assert_conv_shape(conv_layers[2], &[8, 4, 3, 3], [1, 1], [1, 1, 1, 1]);
    assert_conv_shape(conv_layers[3], &[8, 8, 3, 3], [1, 1], [1, 1, 1, 1]);

    // No bias on Convs pre-fold (MiniNet disables Conv bias —
    // BatchNorm carries the affine offset until the BN-fold).
    for c in conv_layers {
        match c {
            Layer::Conv2d { bias, .. } => {
                assert!(bias.is_none(), "Conv bias must be None pre-fold")
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn post_fold_post_fuse_topology_matches_writer_expectation() {
    let device = Default::default();
    let cfg = MiniNetConfig::default();
    let n_blocks = cfg.channels.len();
    let model = cfg.init::<B>(&device);
    let mut graph = mininet_to_burn_graph(&model, [1, 1, 16, 16]);

    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);

    let mut expected_post: Vec<&'static str> = vec!["Conv"; n_blocks * 2];
    expected_post.push("ReduceMean");
    expected_post.push("Gemm");
    assert_eq!(
        graph.op_sequence(),
        expected_post,
        "post-fold/post-fuse op sequence"
    );

    // Every Conv must now have a bias (from BN-fold) and a fused Relu.
    let conv_layers: Vec<&Layer> = graph
        .layers
        .iter()
        .filter(|l| matches!(l, Layer::Conv2d { .. }))
        .collect();
    assert_eq!(conv_layers.len(), n_blocks * 2);
    for c in conv_layers {
        match c {
            Layer::Conv2d {
                activation, bias, ..
            } => {
                assert!(bias.is_some(), "Conv must carry a bias after BN fold");
                assert!(activation.is_some(), "Conv must carry a fused activation");
            }
            _ => unreachable!(),
        }
    }

    // ReduceMean: NCHW axes [2, 3] (Step 5 rewrites to NHWC), keepdims=false.
    let reduce = graph
        .layers
        .iter()
        .find(|l| matches!(l, Layer::ReduceMean { .. }))
        .expect("ReduceMean must exist");
    if let Layer::ReduceMean { axes, keepdims, .. } = reduce {
        let expected: Vec<i64> = vec![2, 3];
        assert_eq!(axes, &expected, "axes pre-NHWC rewrite");
        assert!(!keepdims, "global-avg-pool collapses H/W");
    }

    // Linear ("Gemm"): weight `[d_in, d_out] = [last_channel, n_classes]`.
    let lin = graph.layers.last().expect("non-empty graph");
    match lin {
        Layer::Linear {
            weight,
            bias,
            output,
            ..
        } => {
            assert_eq!(weight.shape, vec![8, 2]);
            assert!(bias.is_some(), "Burn Linear default has bias");
            assert_eq!(output, "logits");
        }
        _ => panic!("last layer should be Linear"),
    }

    assert_chain_well_formed(&graph);
}

fn assert_conv_shape(
    layer: &Layer,
    weight_shape: &[usize],
    stride: [usize; 2],
    padding: [usize; 4],
) {
    match layer {
        Layer::Conv2d {
            weight,
            stride: s,
            padding: p,
            ..
        } => {
            assert_eq!(weight.shape, weight_shape, "weight shape mismatch");
            assert_eq!(*s, stride, "stride mismatch");
            assert_eq!(*p, padding, "padding mismatch");
        }
        _ => panic!("expected Conv2d"),
    }
}

/// Defensive: every layer's input must be either the graph input or a
/// previous layer's output. Catches ordering / renaming bugs in
/// either rewrite pass.
fn assert_chain_well_formed(graph: &BurnGraph) {
    let mut produced: std::collections::HashSet<String> = Default::default();
    produced.insert(graph.input_name.clone());
    for layer in &graph.layers {
        let i = layer.input();
        assert!(
            produced.contains(i),
            "layer {} consumes unknown tensor {}",
            layer.op_type(),
            i,
        );
        produced.insert(layer.output().to_string());
    }
    assert!(
        produced.contains(&graph.output_name),
        "graph output {} is never produced",
        graph.output_name,
    );
}
