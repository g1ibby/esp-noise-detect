//! TinyConv in Burn.
//!
//! Default config: `channels = [16, 32, 64]`, `dropout = 0.0`.
//! For each channel the block is:
//!
//! ```text
//! Conv2d(k=5, stride=s_i, pad=2, bias=False) -> BN -> ReLU
//! Conv2d(k=3, stride=1, pad=1, bias=False)   -> BN -> ReLU
//! ```
//!
//! where `s_i = 2` for all but the last channel (downsample-early).
//! Then `GlobalAvgPool2d` (spatial mean over H and W) → optional
//! dropout → `Linear(last_channel, n_classes=2)`.
//!
//! Weight layout conventions (important for the ONNX export bridge):
//!
//! * Conv2d: `[out, in/groups, kH, kW]` — same as PyTorch when
//!   `groups=1`.
//! * BatchNorm2d: `gamma` / `beta` / `running_mean` / `running_var`,
//!   each `[C]`. Momentum convention matches PyTorch (default 0.1).
//! * Linear (default `Row` layout): `[d_in, d_out]` — transposed
//!   relative to PyTorch's `[d_out, d_in]`. The ONNX bridge
//!   transposes on export.

use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

/// TinyConv hyper-parameters. `n_classes` is baked in so the model
/// config is fully self-contained.
#[derive(Clone, Debug)]
pub struct TinyConvConfig {
    pub channels: Vec<usize>,
    pub dropout: f64,
    pub n_classes: usize,
}

impl Default for TinyConvConfig {
    fn default() -> Self {
        Self {
            channels: vec![16, 32, 64],
            dropout: 0.0,
            n_classes: 2,
        }
    }
}

impl TinyConvConfig {
    /// At least two channel entries are required (the downsample-early
    /// stride pattern requires `len - 1 >= 1`).
    fn validate(&self) {
        assert!(
            self.channels.len() >= 2,
            "TinyConvConfig.channels must have at least 2 entries (got {})",
            self.channels.len(),
        );
    }

    /// Build the model on `device` with random (Kaiming-uniform)
    /// initialization.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TinyConv<B> {
        self.validate();
        let n_blocks = self.channels.len();
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut in_ch = 1usize;
        for (i, &out_ch) in self.channels.iter().enumerate() {
            let stride = if i + 1 < n_blocks { 2 } else { 1 };
            blocks.push(TinyConvBlock::new(in_ch, out_ch, stride, device));
            in_ch = out_ch;
        }

        let dropout = if self.dropout > 0.0 {
            Some(DropoutConfig::new(self.dropout).init())
        } else {
            None
        };
        let head = LinearConfig::new(in_ch, self.n_classes).init(device);

        TinyConv {
            blocks,
            dropout,
            head,
        }
    }
}

/// One `conv_bn_relu` pair — the downsample conv (k=5, stride=s) plus
/// the refinement conv (k=3, stride=1). Both convs have `bias=false`.
#[derive(Module, Debug)]
pub struct TinyConvBlock<B: Backend> {
    pub conv_down: Conv2d<B>,
    pub bn_down: BatchNorm<B>,
    pub conv_refine: Conv2d<B>,
    pub bn_refine: BatchNorm<B>,
    relu: Relu,
}

impl<B: Backend> TinyConvBlock<B> {
    fn new(in_ch: usize, out_ch: usize, stride: usize, device: &B::Device) -> Self {
        // Padding `k // 2` on both sides for odd kernels.
        let pad_5 = 5 / 2;
        let pad_3 = 3 / 2;
        let conv_down = Conv2dConfig::new([in_ch, out_ch], [5, 5])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(pad_5, pad_5, pad_5, pad_5))
            .with_bias(false)
            .init(device);
        let bn_down = BatchNormConfig::new(out_ch).init(device);
        let conv_refine = Conv2dConfig::new([out_ch, out_ch], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(pad_3, pad_3, pad_3, pad_3))
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

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_down.forward(x);
        let x = self.bn_down.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv_refine.forward(x);
        let x = self.bn_refine.forward(x);
        self.relu.forward(x)
    }
}

/// TinyConv — compact 2D CNN for log-mel inputs.
///
/// * input: `(B, 1, n_mels, T)` — e.g. `(B, 1, 64, 101)` for the
///   32 kHz / 1 s / n_fft=1024 setup.
/// * output: `(B, n_classes)` logits.
#[derive(Module, Debug)]
pub struct TinyConv<B: Backend> {
    pub blocks: Vec<TinyConvBlock<B>>,
    pub dropout: Option<Dropout>,
    pub head: Linear<B>,
}

impl<B: Backend> TinyConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut y = x;
        for block in &self.blocks {
            y = block.forward(y);
        }
        // GlobalAvgPool2d: mean over H (dim 2) and W (dim 3).
        // `mean_dim` keeps the dim; squeeze both at the end.
        let y = y.mean_dim(3).mean_dim(2);
        let [b, c, _, _] = y.dims();
        let y = y.reshape([b, c]);
        let y = if let Some(drop) = &self.dropout {
            drop.forward(y)
        } else {
            y
        };
        self.head.forward(y)
    }

    /// Forward with an `after_block` callback fired after each block.
    /// Produces bit-identical output to [`TinyConv::forward`] — same
    /// ops, same order — but lets the profiling path insert
    /// `client.sync()` plus a timer read between blocks. The hot path
    /// should call `forward`, not this.
    pub fn forward_with_probes<F>(&self, x: Tensor<B, 4>, mut after_block: F) -> Tensor<B, 2>
    where
        F: FnMut(usize),
    {
        let mut y = x;
        for (i, block) in self.blocks.iter().enumerate() {
            y = block.forward(y);
            after_block(i);
        }
        let y = y.mean_dim(3).mean_dim(2);
        let [b, c, _, _] = y.dims();
        let y = y.reshape([b, c]);
        let y = if let Some(drop) = &self.dropout {
            drop.forward(y)
        } else {
            y
        };
        self.head.forward(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_python() {
        let cfg = TinyConvConfig::default();
        assert_eq!(cfg.channels, vec![16, 32, 64]);
        assert!((cfg.dropout - 0.0).abs() < 1e-12);
        assert_eq!(cfg.n_classes, 2);
    }

    #[test]
    #[should_panic = "at least 2 entries"]
    fn rejects_too_few_channels() {
        let cfg = TinyConvConfig {
            channels: vec![8],
            dropout: 0.0,
            n_classes: 2,
        };
        cfg.validate();
    }
}
