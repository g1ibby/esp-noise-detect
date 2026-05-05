//! Lower the [`super::fixture_model::MiniNet`] fixture into the IR.
//!
//! Lives next to the fixture model (test code only). The library
//! itself stays model-agnostic — it ships per-Burn-primitive
//! extractors (`conv2d_layer`, `batchnorm_layer`, `linear_layer`),
//! and any caller (test or downstream binary) is responsible for
//! threading them into a `BurnGraph` for whatever model they have.

#![allow(dead_code)]

use burn::tensor::backend::Backend;
use burn_espdl_export::ir::{
    BurnGraph, Layer,
    extract::{batchnorm_layer, conv2d_layer, linear_layer},
};

use super::fixture_model::MiniNet;

/// Lower `model` into a `BurnGraph` with stable, descriptive tensor
/// names so test failures are easy to read.
pub fn mininet_to_burn_graph<B: Backend>(model: &MiniNet<B>, input_shape: [usize; 4]) -> BurnGraph {
    let mut layers: Vec<Layer> = Vec::with_capacity(model.blocks.len() * 6 + 2);
    let mut current = "input".to_string();

    for (b, block) in model.blocks.iter().enumerate() {
        emit_conv_bn_relu(
            &mut layers,
            &mut current,
            b,
            "down",
            &block.conv_down,
            &block.bn_down,
        );
        emit_conv_bn_relu(
            &mut layers,
            &mut current,
            b,
            "refine",
            &block.conv_refine,
            &block.bn_refine,
        );
    }

    // mean_dim(3).mean_dim(2).reshape([B, C]) collapses into a single
    // ReduceMean(axes=[2,3], keepdims=false). Axes are NCHW (the
    // NHWC rewrite is Step 5's responsibility).
    let pool_out = format!("act_{}", layers.len());
    layers.push(Layer::ReduceMean {
        input: core::mem::replace(&mut current, pool_out.clone()),
        output: pool_out,
        axes: vec![2, 3],
        keepdims: false,
    });

    let head_in = current;
    let head_out = "logits".to_string();
    layers.push(linear_layer(&model.head, head_in, head_out.clone()));

    BurnGraph {
        input_name: "input".to_string(),
        input_shape,
        output_name: head_out,
        layers,
    }
}

fn emit_conv_bn_relu<B: Backend>(
    layers: &mut Vec<Layer>,
    current: &mut String,
    block_idx: usize,
    role: &str,
    conv: &burn::nn::conv::Conv2d<B>,
    bn: &burn::nn::BatchNorm<B>,
) {
    let conv_in = core::mem::replace(current, format!("act_{}_{}_conv", block_idx, role));
    let conv_out = current.clone();
    layers.push(conv2d_layer(conv, conv_in, conv_out));

    let bn_in = core::mem::replace(current, format!("act_{}_{}_bn", block_idx, role));
    let bn_out = current.clone();
    layers.push(batchnorm_layer(bn, bn_in, bn_out));

    let relu_in = core::mem::replace(current, format!("act_{}_{}_relu", block_idx, role));
    let relu_out = current.clone();
    layers.push(Layer::Relu {
        input: relu_in,
        output: relu_out,
    });
}
