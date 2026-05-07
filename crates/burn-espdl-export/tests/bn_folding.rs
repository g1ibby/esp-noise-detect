//! Step 3 — BN-folding sanity test.
//!
//! Build a tiny `Conv2d → BatchNorm` block, lower it into a
//! `BurnGraph`, fold the BN, and assert the folded graph reproduces
//! the unfolded forward output to ≤ 1e-5 max-abs-diff (the bar set in
//! `BURN_TO_ESPDL_TASK.md` for Step 3).
//!
//! TinyConv is intentionally *not* used here: keeping the fixture
//! tiny isolates the math of the fold from the rest of the IR.

use burn::backend::NdArray;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNormConfig, Initializer, PaddingConfig2d};
use burn::tensor::{Distribution, Tensor as BurnTensor, TensorData, backend::{Backend, BackendTypes}};
use burn_espdl_export::ir::{Activation, BurnGraph, Layer, Tensor as IrTensor, fold_batchnorm};

type B = NdArray;

#[test]
fn bn_fold_matches_unfolded_forward() {
    let device: <B as BackendTypes>::Device = Default::default();

    // Hand-built two-layer block: Conv2d (with bias) → BatchNorm2d.
    let conv = Conv2dConfig::new([3, 4], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(true)
        .with_initializer(Initializer::Normal {
            mean: 0.0,
            std: 0.5,
        })
        .init::<B>(&device);
    let mut bn = BatchNormConfig::new(4).init::<B>(&device);

    // Replace running stats / γ / β with non-trivial values — Burn's
    // defaults (0/1/1/0) would let a buggy fold pass.
    // RunningState::update stages the new tensor in a per-thread map;
    // BatchNorm::forward calls .value() which reads the canonical
    // (un-synced) value, so we have to force a sync before either
    // path can read the new running stats.
    // (See _reference/burn/.../module/param/running.rs.)
    bn.running_mean.update(BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![0.1f32, -0.2, 0.4, 0.3], [4]),
        &device,
    ));
    let _ = bn.running_mean.value_sync();
    bn.running_var.update(BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![1.5f32, 0.7, 1.2, 2.0], [4]),
        &device,
    ));
    let _ = bn.running_var.value_sync();
    bn.gamma = bn.gamma.map(|_| {
        BurnTensor::<B, 1>::from_data(TensorData::new(vec![1.3f32, 0.8, 1.0, 1.5], [4]), &device)
    });
    bn.beta = bn.beta.map(|_| {
        BurnTensor::<B, 1>::from_data(
            TensorData::new(vec![-0.1f32, 0.05, 0.2, -0.3], [4]),
            &device,
        )
    });

    let input = BurnTensor::<B, 4>::random([2, 3, 5, 7], Distribution::Normal(0.0, 1.0), &device);

    // Reference forward. NdArray is a non-autodiff backend, so
    // `BatchNorm::forward` automatically takes the inference branch
    // and reads running stats — that's the regime PTQ assumes
    // (see _reference/burn/.../norm/batch.rs:103-106).
    let unfolded = bn.forward(conv.forward(input.clone()));

    // Lower into a 2-layer BurnGraph and fold.
    let mut graph = BurnGraph {
        input_name: "in".into(),
        input_shape: [2, 3, 5, 7],
        output_name: "out".into(),
        layers: vec![
            Layer::Conv2d {
                input: "in".into(),
                output: "conv_out".into(),
                weight: tensor_from_burn_4d(conv.weight.val()),
                bias: conv.bias.as_ref().map(|p| tensor_from_burn_1d(p.val())),
                stride: [1, 1],
                padding: [1, 1, 1, 1],
                dilation: [1, 1],
                groups: 1,
                activation: None,
            },
            Layer::BatchNorm2d {
                input: "conv_out".into(),
                output: "out".into(),
                gamma: tensor_from_burn_1d(bn.gamma.val()),
                beta: tensor_from_burn_1d(bn.beta.val()),
                running_mean: tensor_from_burn_1d(bn.running_mean.value_sync()),
                running_var: tensor_from_burn_1d(bn.running_var.value_sync()),
                epsilon: 1e-5,
            },
        ],
    };
    fold_batchnorm(&mut graph);

    // Post-fold: 1 layer (Conv with bias, output renamed to "out").
    assert_eq!(graph.layers.len(), 1, "fold should remove the BN layer");
    match &graph.layers[0] {
        Layer::Conv2d {
            activation,
            output,
            bias,
            ..
        } => {
            assert!(
                activation.is_none(),
                "fold must not introduce an activation"
            );
            assert_eq!(output, "out", "Conv must inherit BN's output name");
            assert!(bias.is_some(), "fold must write a bias");
        }
        _ => panic!("first layer should be Conv after fold"),
    }

    // The IR forward returns rank-2 (collapse semantics for ReduceMean).
    // For a Conv-only post-fold IR, the executor's defensive squeeze
    // would only fire when H=W=1; here our conv output is [2,4,5,7], so
    // we deliberately hit the assertion. Run the Conv layer directly
    // through the same code path (rebuild & forward) to compare 4-D
    // outputs.
    let folded_conv = match &graph.layers[0] {
        Layer::Conv2d { .. } => rebuild_and_forward::<B>(&graph.layers[0], input, &device),
        _ => unreachable!(),
    };

    let max_diff = max_abs_diff_4d::<B>(&unfolded, &folded_conv);
    assert!(
        max_diff <= 1e-5,
        "BN fold parity: max-abs-diff {max_diff} > 1e-5"
    );
}

#[test]
fn fold_no_op_when_no_bn() {
    let device: <B as BackendTypes>::Device = Default::default();
    let conv = Conv2dConfig::new([1, 2], [1, 1])
        .with_bias(false)
        .init::<B>(&device);
    let mut graph = BurnGraph {
        input_name: "in".into(),
        input_shape: [1, 1, 1, 1],
        output_name: "out".into(),
        layers: vec![Layer::Conv2d {
            input: "in".into(),
            output: "out".into(),
            weight: tensor_from_burn_4d(conv.weight.val()),
            bias: None,
            stride: [1, 1],
            padding: [0, 0, 0, 0],
            dilation: [1, 1],
            groups: 1,
            activation: Some(Activation::Relu),
        }],
    };
    let before = graph.layers.len();
    fold_batchnorm(&mut graph);
    assert_eq!(graph.layers.len(), before);
}

/// Rebuild a `burn::nn::Conv2d` from an IR layer's parameters and run
/// `forward` on `input`. Mirrors what `ir::forward` does internally
/// for Conv2d, but skips the rank-coercion logic so the test can
/// inspect the rank-4 output directly.
fn rebuild_and_forward<B: Backend>(
    layer: &Layer,
    input: BurnTensor<B, 4>,
    device: &B::Device,
) -> BurnTensor<B, 4> {
    let (weight, bias, stride, padding, dilation, groups) = match layer {
        Layer::Conv2d {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => (weight, bias.as_ref(), *stride, *padding, *dilation, *groups),
        _ => panic!("rebuild_and_forward: not a Conv2d layer"),
    };
    let out_c = weight.shape[0];
    let in_c_per_g = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];

    let mut conv = Conv2dConfig::new([in_c_per_g * groups, out_c], [kh, kw])
        .with_stride(stride)
        .with_padding(PaddingConfig2d::Explicit(
            padding[0], padding[1], padding[2], padding[3],
        ))
        .with_dilation(dilation)
        .with_groups(groups)
        .with_bias(bias.is_some())
        .init::<B>(device);

    let wd = TensorData::new(weight.data.clone(), [out_c, in_c_per_g, kh, kw]);
    conv.weight = conv
        .weight
        .map(|_| BurnTensor::<B, 4>::from_data(wd, device));
    if let Some(b) = bias {
        let bd = TensorData::new(b.data.clone(), [b.shape[0]]);
        conv.bias = conv
            .bias
            .map(|p| p.map(|_| BurnTensor::<B, 1>::from_data(bd, device)));
    }
    conv.forward(input)
}

fn tensor_from_burn_4d<B: Backend>(t: BurnTensor<B, 4>) -> IrTensor {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    IrTensor::new(data, dims.to_vec())
}

fn tensor_from_burn_1d<B: Backend>(t: BurnTensor<B, 1>) -> IrTensor {
    let dims = t.dims();
    let data = t.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    IrTensor::new(data, dims.to_vec())
}

fn max_abs_diff_4d<B: Backend>(a: &BurnTensor<B, 4>, b: &BurnTensor<B, 4>) -> f32 {
    let av = a
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let bv = b
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
