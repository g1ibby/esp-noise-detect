//! Production-model wiring for native ESP-DL export.
//!
//! `burn-espdl-export` intentionally stays model-agnostic: it exposes
//! primitive extractors and a `BurnGraph` writer, but it does not know
//! about `TinyConv` or checkpoint records. This module is the narrow
//! `nn-rs` side of that boundary.

use crate::model::{TinyConv, TinyConvBlock};
use burn::tensor::backend::Backend;
use burn_espdl_export::ir::{
    BurnGraph, Layer,
    extract::{batchnorm_layer, conv2d_layer, linear_layer},
};

/// Lower the production `TinyConv` topology into the exporter IR.
///
/// The graph starts as the pre-fold form (`Conv -> BN -> Relu`) so the
/// shared exporter passes can fold BatchNorm and fuse Relu exactly like
/// the fixture tests do.
pub fn tinyconv_to_burn_graph<B: Backend>(
    model: &TinyConv<B>,
    input_shape: [usize; 4],
) -> BurnGraph {
    let mut layers: Vec<Layer> = Vec::with_capacity(model.blocks.len() * 6 + 2);
    let mut current = "input".to_string();

    for (block_idx, block) in model.blocks.iter().enumerate() {
        emit_conv_bn_relu(&mut layers, &mut current, block_idx, "down", block);
        emit_conv_bn_relu(&mut layers, &mut current, block_idx, "refine", block);
    }

    let pool_out = format!("act_{}", layers.len());
    layers.push(Layer::ReduceMean {
        input: core::mem::replace(&mut current, pool_out.clone()),
        output: pool_out,
        axes: vec![2, 3],
        keepdims: false,
    });

    let logits = "logits".to_string();
    layers.push(linear_layer(
        &model.head,
        core::mem::replace(&mut current, logits.clone()),
        logits.clone(),
    ));

    BurnGraph {
        input_name: "input".to_string(),
        input_shape,
        output_name: logits,
        layers,
    }
}

fn emit_conv_bn_relu<B: Backend>(
    layers: &mut Vec<Layer>,
    current: &mut String,
    block_idx: usize,
    role: &str,
    block: &TinyConvBlock<B>,
) {
    let conv = if role == "down" {
        &block.conv_down
    } else {
        &block.conv_refine
    };
    let bn = if role == "down" {
        &block.bn_down
    } else {
        &block.bn_refine
    };

    let conv_out = format!("act_{block_idx}_{role}_conv");
    layers.push(conv2d_layer(
        conv,
        core::mem::replace(current, conv_out.clone()),
        conv_out,
    ));

    let bn_out = format!("act_{block_idx}_{role}_bn");
    layers.push(batchnorm_layer(
        bn,
        core::mem::replace(current, bn_out.clone()),
        bn_out,
    ));

    let relu_out = format!("act_{block_idx}_{role}_relu");
    layers.push(Layer::Relu {
        input: core::mem::replace(current, relu_out.clone()),
        output: relu_out,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn_espdl_export::{fold_batchnorm, fuse_relu};

    #[test]
    fn tinyconv_lowering_matches_export_op_shape() {
        let device = Default::default();
        let model = crate::model::TinyConvConfig {
            channels: vec![4, 8],
            dropout: 0.0,
            n_classes: 2,
        }
        .init::<NdArray>(&device);
        let mut graph = tinyconv_to_burn_graph(&model, [1, 1, 16, 16]);

        assert_eq!(graph.layers.len(), 14);
        assert!(matches!(graph.layers[0], Layer::Conv2d { .. }));
        assert!(matches!(graph.layers[1], Layer::BatchNorm2d { .. }));
        assert!(matches!(graph.layers[2], Layer::Relu { .. }));
        assert!(matches!(graph.layers[12], Layer::ReduceMean { .. }));
        assert!(matches!(graph.layers[13], Layer::Linear { .. }));

        fold_batchnorm(&mut graph);
        fuse_relu(&mut graph);

        assert_eq!(graph.layers.len(), 6);
        assert!(graph.layers[..4].iter().all(|l| matches!(
            l,
            Layer::Conv2d {
                activation: Some(_),
                ..
            }
        )));
        assert!(matches!(graph.layers[4], Layer::ReduceMean { .. }));
        assert!(matches!(graph.layers[5], Layer::Linear { .. }));
    }
}
