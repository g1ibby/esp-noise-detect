//! Self-contained fixture CNN for the integration tests.
//!
//! `MiniNet` is a tiny Burn `Module` defined inline here — it is **not**
//! pulled from any production crate, and the test suite does **not**
//! load any pre-trained checkpoint or external artifact (no `/tmp/...`,
//! no production safetensors, no nn-rs imports). Tests build the
//! model in-memory, optionally drive a few synthetic training steps if
//! they want non-trivial weights, and assert against the model's own
//! forward pass.
//!
//! Architecture mirrors *the structural shape* (not the size) of the
//! real-world target the exporter has to handle:
//!
//! ```text
//!   input [N,1,H,W]
//!     → Conv2d(no bias) → BatchNorm → ReLU         // block 0 ("down")
//!     → Conv2d(no bias) → BatchNorm → ReLU         // block 0 ("refine")
//!     → Conv2d(no bias) → BatchNorm → ReLU         // block 1 ("down")
//!     → Conv2d(no bias) → BatchNorm → ReLU         // block 1 ("refine")
//!     → mean over (2, 3) → reshape to [N, C]       // global-avg-pool
//!     → Linear                                      // head
//!     → logits [N, n_classes]
//! ```
//!
//! That covers every op kind in the Step-3 scope (Conv2d / BatchNorm /
//! ReLU / mean-reduce / Linear) without any TinyConv-specific shape
//! choices.

#![allow(dead_code)]

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Hyper-parameters for [`MiniNet`]. Defaults pick small numbers so
/// the test suite stays fast (≪ 100 ms on CPU).
#[derive(Clone, Debug)]
pub struct MiniNetConfig {
    /// Output channels per block (length determines block count).
    pub channels: Vec<usize>,
    /// Number of output classes for the Linear head.
    pub n_classes: usize,
}

impl Default for MiniNetConfig {
    fn default() -> Self {
        Self {
            channels: vec![4, 8],
            n_classes: 2,
        }
    }
}

impl MiniNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniNet<B> {
        assert!(
            !self.channels.is_empty(),
            "MiniNetConfig.channels must have at least one entry",
        );
        let mut blocks = Vec::with_capacity(self.channels.len());
        let mut in_ch = 1usize;
        for (i, &out_ch) in self.channels.iter().enumerate() {
            // Stride 2 on every block except the last — simple
            // "downsample-most" pattern; nothing TinyConv-specific.
            let stride = if i + 1 < self.channels.len() { 2 } else { 1 };
            blocks.push(MiniBlock::new(in_ch, out_ch, stride, device));
            in_ch = out_ch;
        }
        let head = LinearConfig::new(in_ch, self.n_classes).init(device);
        let mut model = MiniNet { blocks, head };
        seed_parameters(&mut model, device, 0x4d49_4e49_4e45_54_u64);
        model
    }
}

/// One `Conv → BN → Relu`, twice. Mirrors the "down + refine" pattern
/// used by the real model so the test exercises the BN-fold +
/// Relu-fuse rewrite the same way the production model would.
#[derive(Module, Debug)]
pub struct MiniBlock<B: Backend> {
    pub conv_down: Conv2d<B>,
    pub bn_down: BatchNorm<B>,
    pub conv_refine: Conv2d<B>,
    pub bn_refine: BatchNorm<B>,
    relu: Relu,
}

impl<B: Backend> MiniBlock<B> {
    fn new(in_ch: usize, out_ch: usize, stride: usize, device: &B::Device) -> Self {
        let conv_down = Conv2dConfig::new([in_ch, out_ch], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init(device);
        let bn_down = BatchNormConfig::new(out_ch).init(device);
        let conv_refine = Conv2dConfig::new([out_ch, out_ch], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init(device);
        let bn_refine = BatchNormConfig::new(out_ch).init(device);
        Self {
            conv_down,
            bn_down,
            conv_refine,
            bn_refine,
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_down.forward(x);
        let x = self.bn_down.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv_refine.forward(x);
        let x = self.bn_refine.forward(x);
        self.relu.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MiniNet<B: Backend> {
    pub blocks: Vec<MiniBlock<B>>,
    pub head: Linear<B>,
}

impl<B: Backend> MiniNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut y = x;
        for block in &self.blocks {
            y = block.forward(y);
        }
        // Global-avg-pool: mean over H (dim 2) and W (dim 3), then
        // squeeze to rank-2.
        let y = y.mean_dim(3).mean_dim(2);
        let [b, c, _, _] = y.dims();
        let y = y.reshape([b, c]);
        self.head.forward(y)
    }
}

/// Replace BN running stats with non-trivial values so the BN-fold
/// rewrite is genuinely exercised. With Burn's defaults
/// (`mean = 0`, `var = 1`, `gamma = 1`, `beta = 0`) the BN forward is
/// a no-op and a buggy fold would silently pass.
///
/// `seed` lets each test pick its own values without colliding with
/// neighbouring fixtures.
pub fn perturb_bn_stats<B: Backend>(model: &mut MiniNet<B>, device: &B::Device, seed: u64) {
    let mut state = seed.max(1);
    for block in &mut model.blocks {
        perturb_one(&mut block.bn_down, &mut state, device);
        perturb_one(&mut block.bn_refine, &mut state, device);
    }
}

/// Replace Burn's backend-global random initialization with deterministic
/// fixture weights. This keeps oracle generation independent of test
/// execution order while still giving every layer non-trivial parameters.
fn seed_parameters<B: Backend>(model: &mut MiniNet<B>, device: &B::Device, seed: u64) {
    let mut state = seed.max(1);
    for block in &mut model.blocks {
        seed_conv(&mut block.conv_down, &mut state, device);
        seed_conv(&mut block.conv_refine, &mut state, device);
    }
    seed_linear(&mut model.head, &mut state, device);
}

fn seed_conv<B: Backend>(conv: &mut Conv2d<B>, state: &mut u64, device: &B::Device) {
    let dims = conv.weight.dims();
    let n = dims.iter().product::<usize>();
    let data = sample_vec(state, n, -0.25, 0.25);
    conv.weight = conv
        .weight
        .clone()
        .map(|_| Tensor::<B, 4>::from_data(TensorData::new(data, dims), device));

    if let Some(bias) = conv.bias.take() {
        let dims = bias.dims();
        let data = sample_vec(state, dims[0], -0.05, 0.05);
        conv.bias =
            Some(bias.map(|_| Tensor::<B, 1>::from_data(TensorData::new(data, dims), device)));
    }
}

fn seed_linear<B: Backend>(linear: &mut Linear<B>, state: &mut u64, device: &B::Device) {
    let dims = linear.weight.dims();
    let n = dims.iter().product::<usize>();
    let data = sample_vec(state, n, -0.2, 0.2);
    linear.weight = linear
        .weight
        .clone()
        .map(|_| Tensor::<B, 2>::from_data(TensorData::new(data, dims), device));

    if let Some(bias) = linear.bias.take() {
        let dims = bias.dims();
        let data = sample_vec(state, dims[0], -0.05, 0.05);
        linear.bias =
            Some(bias.map(|_| Tensor::<B, 1>::from_data(TensorData::new(data, dims), device)));
    }
}

fn perturb_one<B: Backend>(bn: &mut BatchNorm<B>, state: &mut u64, device: &B::Device) {
    let c = bn.gamma.dims()[0];

    let mean = sample_vec(state, c, -0.4, 0.4);
    let var = sample_vec(state, c, 0.5, 1.8); // strictly positive
    let gamma = sample_vec(state, c, 0.7, 1.4);
    let beta = sample_vec(state, c, -0.3, 0.3);

    bn.gamma = bn
        .gamma
        .clone()
        .map(|_| Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(gamma, [c]), device));
    bn.beta = bn
        .beta
        .clone()
        .map(|_| Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(beta, [c]), device));
    bn.running_mean.update(Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(mean, [c]),
        device,
    ));
    // `update` stages into a per-thread map; `value()` (which
    // BatchNorm::forward calls) reads the canonical un-synced value,
    // so force a sync before any forward runs.
    let _ = bn.running_mean.value_sync();
    bn.running_var.update(Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(var, [c]),
        device,
    ));
    let _ = bn.running_var.value_sync();
}

/// Tiny LCG → uniform-in-`[lo, hi]` vector. Deterministic, seeded,
/// dependency-free.
fn sample_vec(state: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((*state >> 32) as u32 as f32) / (u32::MAX as f32);
        out.push(lo + (hi - lo) * u);
    }
    out
}
