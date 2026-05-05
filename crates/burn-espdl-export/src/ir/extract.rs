//! Model-agnostic helpers for lowering Burn modules into the IR.
//!
//! The crate is scoped to the op set TinyConv currently uses
//! (`Conv2d`, `BatchNorm`, `ReLU`, mean-reduce, `Linear`) — **not** to
//! TinyConv itself. Extraction therefore exposes per-primitive
//! conversions that any caller can assemble into a [`BurnGraph`]:
//!
//! ```ignore
//! use burn_espdl_export::ir::{BurnGraph, Layer, extract};
//!
//! let mut layers = vec![];
//! layers.push(extract::conv2d_layer(&model.conv, "input", "act_0"));
//! layers.push(extract::batchnorm_layer(&model.bn, "act_0", "act_1"));
//! layers.push(Layer::Relu { input: "act_1".into(), output: "act_2".into() });
//! layers.push(extract::linear_layer(&model.head, "act_2", "logits"));
//! let graph = BurnGraph { /* … */ layers };
//! ```
//!
//! Adding a new model is "construct the same `Vec<Layer>` for it".
//! Adding a new layer kind (Conv1D, Sigmoid, MaxPool, …) is one new
//! `Layer` variant, one new helper here, and one writer arm in
//! Step 5; the extraction layer doesn't need to know about the
//! topology it's being plugged into. Adding a new chip (ESP32-P4 et
//! al.) reuses everything in this module — the chip-specific bits
//! (per-channel weight quant, alignment) live downstream in Steps
//! 4-5, not here.
//!
//! When in doubt, bias toward "thinner helper, more freedom for the
//! caller" — the alternative drags model-specific shape choices
//! (where to thread tensor names, when to fold `mean_dim` pairs,
//! etc.) into a place that wants to stay reusable.

use burn::nn::BatchNorm as BurnBatchNorm;
use burn::nn::Linear as BurnLinear;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d as BurnConv2d;
use burn::tensor::backend::Backend;

use super::{Layer, Tensor};

/// Lower a Burn `Conv2d` to a [`Layer::Conv2d`] with no fused
/// activation. The caller picks the input/output tensor names so the
/// extractor doesn't have to know how layers are named in the
/// surrounding graph.
pub fn conv2d_layer<B: Backend>(
    conv: &BurnConv2d<B>,
    input: impl Into<String>,
    output: impl Into<String>,
) -> Layer {
    let weight = burn_tensor_to_ir::<B, 4>(conv.weight.val(), conv.weight.dims());
    let bias = conv
        .bias
        .as_ref()
        .map(|p| burn_tensor_to_ir::<B, 1>(p.val(), p.dims()));
    Layer::Conv2d {
        input: input.into(),
        output: output.into(),
        weight,
        bias,
        stride: conv.stride,
        padding: padding_to_array(&conv.padding),
        dilation: conv.dilation,
        groups: conv.groups,
        activation: None,
    }
}

/// Lower a Burn `BatchNorm` to a [`Layer::BatchNorm2d`]. The
/// pre-fold IR keeps BN as its own node; [`super::fold_batchnorm`]
/// rewrites the pair `Conv2d → BatchNorm2d` into a single
/// `Conv2d` with bias.
pub fn batchnorm_layer<B: Backend>(
    bn: &BurnBatchNorm<B>,
    input: impl Into<String>,
    output: impl Into<String>,
) -> Layer {
    Layer::BatchNorm2d {
        input: input.into(),
        output: output.into(),
        gamma: burn_tensor_to_ir::<B, 1>(bn.gamma.val(), bn.gamma.dims()),
        beta: burn_tensor_to_ir::<B, 1>(bn.beta.val(), bn.beta.dims()),
        running_mean: tensor_from_burn_1d::<B>(bn.running_mean.value_sync()),
        running_var: tensor_from_burn_1d::<B>(bn.running_var.value_sync()),
        epsilon: bn.epsilon,
    }
}

/// Lower a Burn `Linear` to a [`Layer::Linear`]. Burn stores
/// `[d_in, d_out]` (its `Row` layout, transposed wrt PyTorch's
/// `[d_out, d_in]`); the IR keeps that convention and lets Step 5
/// apply the transpose + 4-D unsqueeze + blocked pack as part of
/// the Gemm writer.
pub fn linear_layer<B: Backend>(
    linear: &BurnLinear<B>,
    input: impl Into<String>,
    output: impl Into<String>,
) -> Layer {
    let weight = burn_tensor_to_ir::<B, 2>(linear.weight.val(), linear.weight.dims());
    let bias = linear
        .bias
        .as_ref()
        .map(|p| burn_tensor_to_ir::<B, 1>(p.val(), p.dims()));
    Layer::Linear {
        input: input.into(),
        output: output.into(),
        weight,
        bias,
    }
}

fn burn_tensor_to_ir<B: Backend, const D: usize>(
    t: burn::tensor::Tensor<B, D>,
    dims: [usize; D],
) -> Tensor {
    let data = burn_to_vec_f32(t);
    Tensor::new(data, dims.to_vec())
}

fn tensor_from_burn_1d<B: Backend>(t: burn::tensor::Tensor<B, 1>) -> Tensor {
    let dims = t.dims();
    let data = burn_to_vec_f32(t);
    Tensor::new(data, dims.to_vec())
}

fn burn_to_vec_f32<B: Backend, const D: usize>(t: burn::tensor::Tensor<B, D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("Burn tensor → Vec<f32> conversion")
}

/// `PaddingConfig2d` → `[top, left, bottom, right]`. We only support
/// `Explicit` and `Valid` for now; `Same` would require knowing the
/// input shape per-layer, which is not on the IR.
fn padding_to_array(p: &PaddingConfig2d) -> [usize; 4] {
    match p {
        PaddingConfig2d::Valid => [0, 0, 0, 0],
        PaddingConfig2d::Explicit(top, left, bottom, right) => [*top, *left, *bottom, *right],
        PaddingConfig2d::Same => panic!(
            "burn-espdl-export: PaddingConfig2d::Same is not supported (resolve at the call \
             site by computing the explicit padding for the layer's input shape)",
        ),
    }
}
