//! Run a [`BurnGraph`] forward on a host backend.
//!
//! Step 4 (calibration) needs to feed the graph the same `.npy` windows
//! esp-ppq feeds and observe every activation. Doing that through the
//! original `TinyConv::forward` would tie us to (a) the pre-fold layer
//! shapes and (b) Burn's nested `Module` API for hooks, neither of
//! which is great. Instead, this module re-runs the IR layer-by-layer
//! through `burn::tensor` ops so the calibration loop can see exactly
//! the layers and tensor names the writer is going to emit.
//!
//! Tested in `tests/forward_parity_fp32.rs` against the FP32 logits
//! the production model produces for an all-zero input
//! (`/tmp/nn-rs-robust-cuda/export/logits_rust.json`).

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d as BurnConv2d, Conv2dConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor as BurnTensor, TensorData};

use super::{Activation, BurnGraph, Layer, Tensor};

/// Run the graph on `input` (an `[N, C, H, W]` rank-4 tensor) and
/// return the final logits as a rank-2 `[N, n_classes]` tensor. Panics
/// if the IR is not yet in the post-fold/post-fuse shape that
/// [`super::fold_batchnorm`] + [`super::fuse_relu`] produce, because
/// Step 3's parity tests only care about that representation. Pre-fold
/// IRs *can* be executed too, but the runner falls back to the same
/// op set; it doesn't try to execute a standalone `BatchNorm2d`.
pub fn forward<B: Backend>(
    graph: &BurnGraph,
    input: BurnTensor<B, 4>,
    device: &B::Device,
) -> BurnTensor<B, 2> {
    forward_with_hook(graph, input, device, &mut |_, _| {})
}

/// Like [`forward`], but invokes `hook(name, values)` once per
/// activation produced — first for the graph's named input, then for
/// every layer's output. Used by Step 4 calibration to stream
/// activations through KL observers without keeping an
/// activation-by-tensor map alive between phases.
///
/// `name` is the tensor's IR name (matches `Layer::output()` /
/// `BurnGraph::input_name`); `values` is the row-major flattened
/// activation slice, materialised on the host.
pub fn forward_with_hook<B: Backend>(
    graph: &BurnGraph,
    input: BurnTensor<B, 4>,
    device: &B::Device,
    hook: &mut dyn FnMut(&str, &[f32]),
) -> BurnTensor<B, 2> {
    // Observe the graph input. Cloning the Burn tensor here is the
    // cheapest way to get a host-side `Vec<f32>` without consuming
    // the tensor — `into_data` would move out.
    let input_v = burn_to_vec(input.clone());
    hook(&graph.input_name, &input_v);
    let mut act = Act::Rank4(input);
    for layer in &graph.layers {
        act = run_layer(layer, act, device);
        let v = act_to_vec(&act);
        hook(layer.output(), &v);
    }
    match act {
        Act::Rank2(t) => t,
        Act::Rank4(t) => {
            // Defensive: if the IR was pre-collapse the final activation
            // could still be 4-D. Squeeze trailing 1s so callers always
            // see the logits shape they expect.
            let dims = t.dims();
            assert_eq!(
                dims[2], 1,
                "BurnGraph::forward: trailing 4-D output has H={} != 1",
                dims[2]
            );
            assert_eq!(
                dims[3], 1,
                "BurnGraph::forward: trailing 4-D output has W={} != 1",
                dims[3]
            );
            t.reshape([dims[0], dims[1]])
        }
    }
}

/// Like [`forward_with_hook`], but run through esp-ppq-style fake
/// quantization while observing the pre-output-quantized tensor values.
///
/// During PPQ calibration, parameter configs have already been rendered
/// before activation observers run. The calibration hook observes raw
/// op inputs/outputs, while the executor may feed fake-quantized
/// tensors downstream. This helper mirrors that split: observe raw
/// values, then fake-quantize parameters and any activation with a
/// supplied scale before it becomes a later op's input.
pub fn forward_with_fake_quant_hook<B, FA, FP>(
    graph: &BurnGraph,
    input: BurnTensor<B, 4>,
    device: &B::Device,
    activation_scale: &FA,
    parameter_scale: &FP,
    hook: &mut dyn FnMut(&str, &[f32]),
) -> BurnTensor<B, 2>
where
    B: Backend,
    FA: Fn(&str) -> Option<(f32, u8)>,
    FP: Fn(&str) -> Option<(f32, u8)>,
{
    let input_v = burn_to_vec(input.clone());
    hook(&graph.input_name, &input_v);
    let mut act = Act::Rank4(fake_quant_activation(
        &graph.input_name,
        input,
        activation_scale,
        device,
    ));
    for layer in &graph.layers {
        act = run_layer_fake_quant(layer, act, device, parameter_scale);
        let v = act_to_vec(&act);
        hook(layer.output(), &v);
        act = fake_quant_act(layer.output(), act, activation_scale, device);
    }
    match act {
        Act::Rank2(t) => t,
        Act::Rank4(t) => {
            let dims = t.dims();
            assert_eq!(
                dims[2], 1,
                "BurnGraph::forward: trailing 4-D output has H={} != 1",
                dims[2]
            );
            assert_eq!(
                dims[3], 1,
                "BurnGraph::forward: trailing 4-D output has W={} != 1",
                dims[3]
            );
            t.reshape([dims[0], dims[1]])
        }
    }
}

fn burn_to_vec<B: Backend, const D: usize>(t: BurnTensor<B, D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("Burn tensor → Vec<f32> conversion (calibration hook)")
}

fn act_to_vec<B: Backend>(act: &Act<B>) -> Vec<f32> {
    match act {
        Act::Rank4(t) => burn_to_vec(t.clone()),
        Act::Rank2(t) => burn_to_vec(t.clone()),
    }
}

/// Intermediate activation. The IR mixes ranks (Convs are 4-D, Linear
/// is 2-D) so an enum keeps the rank discipline that Burn's typed
/// tensors enforce.
enum Act<B: Backend> {
    Rank4(BurnTensor<B, 4>),
    Rank2(BurnTensor<B, 2>),
}

fn run_layer<B: Backend>(layer: &Layer, input: Act<B>, device: &B::Device) -> Act<B> {
    match layer {
        Layer::Conv2d {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            activation,
            ..
        } => {
            let x = input.expect_rank4("Conv2d");
            let conv = build_burn_conv2d::<B>(
                weight,
                bias.as_ref(),
                *stride,
                *padding,
                *dilation,
                *groups,
                device,
            );
            let mut y = conv.forward(x);
            if let Some(act) = activation {
                y = apply_activation(y, *act);
            }
            Act::Rank4(y)
        }
        Layer::BatchNorm2d {
            gamma,
            beta,
            running_mean,
            running_var,
            epsilon,
            ..
        } => {
            // Manual inference-time BN: y = γ · (x − μ) / √(σ² + ε) + β.
            // Avoiding burn's BatchNorm module here keeps the runner's
            // surface minimal (no need to construct a RunningState).
            let x = input.expect_rank4("BatchNorm2d");
            Act::Rank4(apply_batchnorm(
                x,
                gamma,
                beta,
                running_mean,
                running_var,
                *epsilon,
                device,
            ))
        }
        Layer::Relu { .. } => {
            let x = input.expect_rank4("Relu");
            Act::Rank4(burn::tensor::activation::relu(x))
        }
        Layer::ReduceMean { axes, keepdims, .. } => {
            let x = input.expect_rank4("ReduceMean");
            apply_reducemean(x, axes, *keepdims)
        }
        Layer::Linear { weight, bias, .. } => {
            let x = input.expect_rank2("Linear");
            Act::Rank2(apply_linear(x, weight, bias.as_ref(), device))
        }
    }
}

fn run_layer_fake_quant<B, FP>(
    layer: &Layer,
    input: Act<B>,
    device: &B::Device,
    parameter_scale: &FP,
) -> Act<B>
where
    B: Backend,
    FP: Fn(&str) -> Option<(f32, u8)>,
{
    match layer {
        Layer::Conv2d {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            activation,
            output,
            ..
        } => {
            let x = input.expect_rank4("Conv2d");
            let weight_key = format!("{output}.weight");
            let weight_data = parameter_scale(&weight_key)
                .map(|(scale, bits)| fake_quant_vec(&weight.data, scale, bits))
                .unwrap_or_else(|| weight.data.clone());
            let quant_weight = Tensor {
                data: weight_data,
                shape: weight.shape.clone(),
            };
            let conv = build_burn_conv2d::<B>(
                &quant_weight,
                bias.as_ref(),
                *stride,
                *padding,
                *dilation,
                *groups,
                device,
            );
            let mut y = conv.forward(x);
            if let Some(act) = activation {
                y = apply_activation(y, *act);
            }
            Act::Rank4(y)
        }
        Layer::BatchNorm2d { .. } => run_layer(layer, input, device),
        Layer::Relu { .. } => run_layer(layer, input, device),
        Layer::ReduceMean { .. } => run_layer(layer, input, device),
        Layer::Linear {
            weight,
            bias,
            output,
            ..
        } => {
            let x = input.expect_rank2("Linear");
            let weight_key = format!("{output}.weight");
            let weight_data = parameter_scale(&weight_key)
                .map(|(scale, bits)| fake_quant_vec(&weight.data, scale, bits))
                .unwrap_or_else(|| weight.data.clone());
            let quant_weight = Tensor {
                data: weight_data,
                shape: weight.shape.clone(),
            };
            Act::Rank2(apply_linear(x, &quant_weight, bias.as_ref(), device))
        }
    }
}

impl<B: Backend> Act<B> {
    fn expect_rank4(self, op: &str) -> BurnTensor<B, 4> {
        match self {
            Self::Rank4(t) => t,
            Self::Rank2(_) => panic!("{op}: expected rank-4 input, got rank-2"),
        }
    }
    fn expect_rank2(self, op: &str) -> BurnTensor<B, 2> {
        match self {
            Self::Rank2(t) => t,
            Self::Rank4(_) => panic!("{op}: expected rank-2 input, got rank-4"),
        }
    }
}

fn fake_quant_act<B, FA>(
    name: &str,
    act: Act<B>,
    activation_scale: &FA,
    device: &B::Device,
) -> Act<B>
where
    B: Backend,
    FA: Fn(&str) -> Option<(f32, u8)>,
{
    match act {
        Act::Rank4(t) => Act::Rank4(fake_quant_activation(name, t, activation_scale, device)),
        Act::Rank2(t) => Act::Rank2(fake_quant_activation(name, t, activation_scale, device)),
    }
}

fn fake_quant_activation<B, FA, const D: usize>(
    name: &str,
    t: BurnTensor<B, D>,
    activation_scale: &FA,
    device: &B::Device,
) -> BurnTensor<B, D>
where
    B: Backend,
    FA: Fn(&str) -> Option<(f32, u8)>,
{
    let Some((scale, bits)) = activation_scale(name) else {
        return t;
    };
    let dims = t.dims();
    let data = fake_quant_vec(&burn_to_vec(t), scale, bits);
    BurnTensor::<B, D>::from_data(TensorData::new(data, dims), device)
}

fn fake_quant_vec(values: &[f32], scale: f32, bits: u8) -> Vec<f32> {
    let (qmin, qmax) = signed_qrange(bits);
    values
        .iter()
        .map(|&v| {
            let q = (v / scale).round_ties_even() as i64;
            q.clamp(qmin, qmax) as f32 * scale
        })
        .collect()
}

fn signed_qrange(bits: u8) -> (i64, i64) {
    assert!(
        (1..=62).contains(&bits),
        "signed_qrange: unsupported bit width {bits}"
    );
    let qmax = (1_i64 << (bits - 1)) - 1;
    let qmin = -(1_i64 << (bits - 1));
    (qmin, qmax)
}

fn apply_activation<B: Backend>(t: BurnTensor<B, 4>, act: Activation) -> BurnTensor<B, 4> {
    match act {
        Activation::Relu => burn::tensor::activation::relu(t),
    }
}

fn apply_batchnorm<B: Backend>(
    x: BurnTensor<B, 4>,
    gamma: &Tensor,
    beta: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f64,
    device: &B::Device,
) -> BurnTensor<B, 4> {
    // All four parameters are `[C]`; broadcast as `[1, C, 1, 1]`.
    let c = gamma.shape[0];
    let g = vec1d_as_4d::<B>(&gamma.data, c, device);
    let b = vec1d_as_4d::<B>(&beta.data, c, device);
    let m = vec1d_as_4d::<B>(&mean.data, c, device);
    let inv_sigma: Vec<f32> = var
        .data
        .iter()
        .map(|v| 1.0 / (v + epsilon as f32).sqrt())
        .collect();
    let s = vec1d_as_4d::<B>(&inv_sigma, c, device);
    (x - m).mul(s).mul(g).add(b)
}

fn vec1d_as_4d<B: Backend>(data: &[f32], c: usize, device: &B::Device) -> BurnTensor<B, 4> {
    let td = TensorData::new(data.to_vec(), [1, c, 1, 1]);
    BurnTensor::<B, 4>::from_data(td, device)
}

fn apply_reducemean<B: Backend>(x: BurnTensor<B, 4>, axes: &[i64], keepdims: bool) -> Act<B> {
    // For TinyConv: axes = [2, 3] (NCHW H, W), keepdims = false.
    // We handle the general single-/double-axis case the model needs;
    // anything else panics so it surfaces the day a new model arrives.
    assert!(!axes.is_empty(), "ReduceMean: empty axes");
    let mut sorted: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
    sorted.sort_by(|a, b| b.cmp(a)); // reduce highest dim first to keep lower indices stable
    let mut tmp = x;
    for axis in &sorted {
        tmp = tmp.mean_dim(*axis);
    }
    if keepdims {
        Act::Rank4(tmp)
    } else {
        // The dropped dims must each be size 1 after `mean_dim` (which
        // keeps the dim). Squeeze them by reshape.
        let dims = tmp.dims();
        // For TinyConv (axes=[2,3], keepdims=false) the result is [N, C].
        assert!(
            sorted.iter().all(|&a| dims[a] == 1),
            "ReduceMean: expected size-1 reduced dims, got {:?} with axes {:?}",
            dims,
            sorted,
        );
        // Build the squeezed-shape vector.
        let mut new_shape: Vec<usize> = Vec::with_capacity(dims.len() - sorted.len());
        for (i, d) in dims.iter().enumerate() {
            if !sorted.contains(&i) {
                new_shape.push(*d);
            }
        }
        // We only support collapse-to-2D in this path; anything else
        // would need to keep a generic-rank tensor alive through the
        // executor enum.
        assert_eq!(
            new_shape.len(),
            2,
            "ReduceMean(keepdims=false): only collapse-to-2D supported, got shape {:?}",
            new_shape,
        );
        let n = new_shape[0];
        let c = new_shape[1];
        Act::Rank2(tmp.reshape([n, c]))
    }
}

fn apply_linear<B: Backend>(
    x: BurnTensor<B, 2>,
    weight: &Tensor,
    bias: Option<&Tensor>,
    device: &B::Device,
) -> BurnTensor<B, 2> {
    let d_in = weight.shape[0];
    let d_out = weight.shape[1];
    let w =
        BurnTensor::<B, 2>::from_data(TensorData::new(weight.data.clone(), [d_in, d_out]), device);
    let mut y = x.matmul(w);
    if let Some(b) = bias {
        let bt =
            BurnTensor::<B, 1>::from_data(TensorData::new(b.data.clone(), [b.shape[0]]), device);
        // Broadcast `[d_out]` over the leading batch dim.
        y = y + bt.reshape([1, d_out]);
    }
    y
}

/// Build a `burn::nn::Conv2d` whose weight (and optional bias) match
/// the IR parameters exactly. We reconstruct on every call rather than
/// caching, because the IR holds raw `Vec<f32>` and the BurnGraph is
/// allowed to outlive the device. For a 6-Conv graph on the host this
/// is well below 1 ms total.
fn build_burn_conv2d<B: Backend>(
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [usize; 2],
    padding: [usize; 4],
    dilation: [usize; 2],
    groups: usize,
    device: &B::Device,
) -> BurnConv2d<B> {
    let out_c = weight.shape[0];
    let in_c_per_g = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];

    let cfg = Conv2dConfig::new([in_c_per_g * groups, out_c], [kh, kw])
        .with_stride(stride)
        .with_padding(PaddingConfig2d::Explicit(
            padding[0], padding[1], padding[2], padding[3],
        ))
        .with_dilation(dilation)
        .with_groups(groups)
        .with_bias(bias.is_some());
    let mut conv = cfg.init::<B>(device);

    let weight_data = TensorData::new(weight.data.clone(), [out_c, in_c_per_g, kh, kw]);
    conv.weight = conv
        .weight
        .map(|_| BurnTensor::<B, 4>::from_data(weight_data, device));

    if let Some(b) = bias {
        let bd = TensorData::new(b.data.clone(), [b.shape[0]]);
        conv.bias = conv
            .bias
            .map(|p| p.map(|_| BurnTensor::<B, 1>::from_data(bd, device)));
    }

    conv
}
