//! [`Layer`] enum and [`BurnGraph`] container — the host-side IR that
//! Step 3 produces from a [`nn_rs::TinyConv`] and that Steps 4–5
//! consume.
//!
//! Design notes:
//!
//! * Tensors stay as plain `Vec<f32>` plus a row-major shape. We do
//!   not keep a `burn::Tensor` here because the IR has to outlive the
//!   Burn device that produced it (Step 4 reuses it across multiple
//!   forward passes against potentially different backends).
//! * Activation tensors flow as opaque `String` ids; the extractor
//!   names them after the producing layer. The names line up with
//!   esp-ppq's `model.json` keys after the same mapping the
//!   `safetensors` exporter applies (see `nn-rs/src/bin/export_weights.rs`),
//!   but Step 3 itself does not need that mapping — Steps 4 and 5 will
//!   convert if/when needed.
//! * The IR is layout-agnostic in the NCHW sense: extraction emits
//!   shapes/axes in the original `[N, C, H, W]` form. The NCHW→NHWC
//!   layout pass lives in Step 5 (per esp-ppq's
//!   `ResetConvLayoutPattern`) — it is intentionally separated from
//!   the BN-fold / Relu-fuse rewrites that live here.

use core::fmt;

/// Activation fused into a [`Layer::Conv2d`] by [`super::fuse_relu`].
/// The only activation ESPDL S3 cares about for TinyConv is `Relu`;
/// the field is kept as an enum so future fusions (Relu6, Sigmoid, …)
/// can be added without churn in the writer signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// ReLU. Maps to `attribute("activation") = "Relu"` in the
    /// FlatBuffers `Conv` node.
    Relu,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relu => write!(f, "Relu"),
        }
    }
}

/// A floating-point parameter tensor stored row-major.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Row-major contents. Length must equal `shape.iter().product()`.
    pub data: Vec<f32>,
    /// Logical dimensions, in declaration order
    /// (e.g. `[out_c, in_c/groups, kH, kW]` for a Conv weight).
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Wrap a `Vec<f32>` plus its shape. Panics if length mismatches.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "Tensor::new: shape {:?} requires {} elements, got {}",
            shape,
            n,
            data.len(),
        );
        Self { data, shape }
    }

    /// Total number of scalars.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Per-layer payload. Variants are limited to the ops the current
/// `TinyConv` needs (after BN folding + Relu fusion) plus the two
/// transient nodes (`BatchNorm2d`, `Relu`) that exist only on the
/// pre-fold path.
///
/// Each layer carries the names of its input(s) and its output, so a
/// [`BurnGraph`] is just a topologically-ordered `Vec<Layer>`.
#[derive(Debug, Clone)]
pub enum Layer {
    /// 2-D convolution. `weight.shape == [out_c, in_c/groups, kH, kW]`
    /// (Burn's `Conv2d` storage order, identical to PyTorch).
    Conv2d {
        /// Producing tensor name on the input side.
        input: String,
        /// Output activation name.
        output: String,
        /// `[out_c, in_c/groups, kH, kW]`.
        weight: Tensor,
        /// `[out_c]`. Pre-fold this is `None` for TinyConv (BN owns the
        /// affine bias); BN fold writes a value here.
        bias: Option<Tensor>,
        /// `[stride_h, stride_w]`.
        stride: [usize; 2],
        /// Explicit padding `[top, left, bottom, right]` — same
        /// convention as `PaddingConfig2d::Explicit`.
        padding: [usize; 4],
        /// `[dilation_h, dilation_w]`.
        dilation: [usize; 2],
        /// Group count (1 for TinyConv).
        groups: usize,
        /// Optional fused activation. `None` pre-fuse; Relu after
        /// [`super::fuse_relu`] runs.
        activation: Option<Activation>,
    },
    /// PyTorch-style 2-D batch normalization. Always exists on the
    /// pre-fold IR; the BN-fold pass removes it.
    BatchNorm2d {
        input: String,
        output: String,
        /// `[C]`. The learnable scale γ.
        gamma: Tensor,
        /// `[C]`. The learnable bias β.
        beta: Tensor,
        /// `[C]`. The running mean (inference-time).
        running_mean: Tensor,
        /// `[C]`. The running variance (inference-time).
        running_var: Tensor,
        /// Numerical-stability ε (Burn default = 1e-5).
        epsilon: f64,
    },
    /// Standalone ReLU. Exists pre-fuse; [`super::fuse_relu`] absorbs
    /// the trailing one onto the upstream `Conv2d`.
    Relu { input: String, output: String },
    /// Mean reduction over the listed `axes`. Used as the global-avg
    /// pool for TinyConv with `axes=[2,3]` and `keepdims=false`. The
    /// axes are kept in NCHW order — Step 5's layout pass rewrites
    /// them to NHWC.
    ReduceMean {
        input: String,
        output: String,
        axes: Vec<i64>,
        keepdims: bool,
    },
    /// `Y = X · W (+ b)`. `weight.shape == [d_in, d_out]` (Burn's
    /// `Linear::Row` layout — transposed wrt PyTorch's
    /// `[d_out, d_in]`). The transpose to PyTorch order, the
    /// `unsqueeze` to a 4-D Conv-shaped pack, and the blocked HWCN
    /// transform all happen in Step 5.
    Linear {
        input: String,
        output: String,
        /// `[d_in, d_out]`.
        weight: Tensor,
        /// `[d_out]`.
        bias: Option<Tensor>,
    },
}

impl Layer {
    /// Op-type label, mirroring the strings esp-ppq writes into
    /// `dl::Node::op_type`.
    pub fn op_type(&self) -> &'static str {
        match self {
            Self::Conv2d { .. } => "Conv",
            Self::BatchNorm2d { .. } => "BatchNormalization",
            Self::Relu { .. } => "Relu",
            Self::ReduceMean { .. } => "ReduceMean",
            Self::Linear { .. } => "Gemm",
        }
    }

    /// Producing tensor name on the input side.
    pub fn input(&self) -> &str {
        match self {
            Self::Conv2d { input, .. }
            | Self::BatchNorm2d { input, .. }
            | Self::Relu { input, .. }
            | Self::ReduceMean { input, .. }
            | Self::Linear { input, .. } => input,
        }
    }

    /// Output activation name.
    pub fn output(&self) -> &str {
        match self {
            Self::Conv2d { output, .. }
            | Self::BatchNorm2d { output, .. }
            | Self::Relu { output, .. }
            | Self::ReduceMean { output, .. }
            | Self::Linear { output, .. } => output,
        }
    }

    /// Mutable view of the output name. Used by the BN-fold pass to
    /// rewire downstream consumers when the BN node disappears.
    pub fn output_mut(&mut self) -> &mut String {
        match self {
            Self::Conv2d { output, .. }
            | Self::BatchNorm2d { output, .. }
            | Self::Relu { output, .. }
            | Self::ReduceMean { output, .. }
            | Self::Linear { output, .. } => output,
        }
    }

    /// Mutable view of the input name. Used by both rewrite passes to
    /// re-thread tensor names after a node is removed.
    pub fn input_mut(&mut self) -> &mut String {
        match self {
            Self::Conv2d { input, .. }
            | Self::BatchNorm2d { input, .. }
            | Self::Relu { input, .. }
            | Self::ReduceMean { input, .. }
            | Self::Linear { input, .. } => input,
        }
    }
}

/// Topologically-ordered IR — produced by [`super::extract::from_tinyconv`],
/// rewritten in place by [`super::fold_batchnorm`] and
/// [`super::fuse_relu`], consumed by Steps 4–5.
#[derive(Debug, Clone)]
pub struct BurnGraph {
    /// Name of the graph's single input tensor (always `"input"` for
    /// TinyConv).
    pub input_name: String,
    /// `[N, C, H, W]` shape of the input tensor in NCHW order.
    pub input_shape: [usize; 4],
    /// Name of the graph's single output tensor (always `"logits"` for
    /// TinyConv).
    pub output_name: String,
    /// Layers in execution order.
    pub layers: Vec<Layer>,
}

impl BurnGraph {
    /// Sequence of `op_type()` labels in execution order. Useful for
    /// quick assertions in tests against `model.info`.
    pub fn op_sequence(&self) -> Vec<&'static str> {
        self.layers.iter().map(Layer::op_type).collect()
    }
}
