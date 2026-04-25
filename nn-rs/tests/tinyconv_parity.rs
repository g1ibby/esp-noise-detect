//! TinyConv forward-parity test.
//!
//! Loads weights, input, and expected output from a fixture produced by
//! `tests/fixtures/generate_tinyconv.py`, injects the weights into a
//! Burn `TinyConv`, runs forward, and asserts element-wise agreement.
//!
//! Tolerance: `1e-4` absolute. Raw logits on this seed range in roughly
//! `[-2, 2]`, so the bound doubles as ~1e-4 relative.

use std::path::PathBuf;

use burn::module::{Param, RunningState};
use burn::prelude::*;
use burn::tensor::TensorData;
use nn_rs::model::{TinyConv, TinyConvBlock, TinyConvConfig};

mod common;

use common::{Backend, device, max_abs_diff};

const MAGIC: u32 = 0x564E_4354; // "TCNV" little-endian
const VERSION: u32 = 1;

struct Fixture {
    channels: Vec<usize>,
    n_classes: usize,
    n_mels: usize,
    n_frames: usize,
    batch: usize,
    bn_eps: f32,
    weights: Vec<BlockWeights>,
    head_weight: Vec<f32>, // [n_classes, last_channel] row-major, PyTorch layout
    head_bias: Vec<f32>,
    input: Vec<f32>,
    expected: Vec<f32>,
}

struct BlockWeights {
    in_ch: usize,
    out_ch: usize,
    conv_down: Vec<f32>, // [out_ch, in_ch, 5, 5]
    bn_down_gamma: Vec<f32>,
    bn_down_beta: Vec<f32>,
    bn_down_rmean: Vec<f32>,
    bn_down_rvar: Vec<f32>,
    conv_refine: Vec<f32>, // [out_ch, out_ch, 3, 3]
    bn_refine_gamma: Vec<f32>,
    bn_refine_beta: Vec<f32>,
    bn_refine_rmean: Vec<f32>,
    bn_refine_rvar: Vec<f32>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.bytes[self.cursor..self.cursor + 4].try_into().unwrap());
        self.cursor += 4;
        v
    }

    fn f32_one(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.bytes[self.cursor..self.cursor + 4].try_into().unwrap());
        self.cursor += 4;
        v
    }

    fn f32_slice(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.f32_one());
        }
        out
    }
}

fn load_fixture(name: &str) -> Fixture {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {e}. Regenerate with \
             `cd nn && uv run --no-dev python ../nn-rs/tests/fixtures/generate_tinyconv.py`",
            path.display(),
        )
    });

    let mut r = Reader {
        bytes: &bytes,
        cursor: 0,
    };
    let magic = r.u32();
    assert_eq!(magic, MAGIC, "bad fixture magic");
    let version = r.u32();
    assert_eq!(version, VERSION, "unsupported fixture version");
    let n_classes = r.u32() as usize;
    let n_mels = r.u32() as usize;
    let n_frames = r.u32() as usize;
    let batch = r.u32() as usize;
    let n_blocks = r.u32() as usize;
    let bn_eps = r.f32_one();
    let channels: Vec<usize> = (0..n_blocks).map(|_| r.u32() as usize).collect();

    let mut weights = Vec::with_capacity(n_blocks);
    let mut in_ch = 1usize;
    for &out_ch in &channels {
        let conv_down = r.f32_slice(out_ch * in_ch * 5 * 5);
        let bn_down_gamma = r.f32_slice(out_ch);
        let bn_down_beta = r.f32_slice(out_ch);
        let bn_down_rmean = r.f32_slice(out_ch);
        let bn_down_rvar = r.f32_slice(out_ch);
        let conv_refine = r.f32_slice(out_ch * out_ch * 3 * 3);
        let bn_refine_gamma = r.f32_slice(out_ch);
        let bn_refine_beta = r.f32_slice(out_ch);
        let bn_refine_rmean = r.f32_slice(out_ch);
        let bn_refine_rvar = r.f32_slice(out_ch);

        weights.push(BlockWeights {
            in_ch,
            out_ch,
            conv_down,
            bn_down_gamma,
            bn_down_beta,
            bn_down_rmean,
            bn_down_rvar,
            conv_refine,
            bn_refine_gamma,
            bn_refine_beta,
            bn_refine_rmean,
            bn_refine_rvar,
        });
        in_ch = out_ch;
    }
    let last_channel = *channels.last().unwrap();
    let head_weight = r.f32_slice(n_classes * last_channel);
    let head_bias = r.f32_slice(n_classes);

    let input_n = batch * n_mels * n_frames;
    let input = r.f32_slice(input_n);
    let expected = r.f32_slice(batch * n_classes);

    assert_eq!(r.cursor, bytes.len(), "trailing bytes in fixture");

    Fixture {
        channels,
        n_classes,
        n_mels,
        n_frames,
        batch,
        bn_eps,
        weights,
        head_weight,
        head_bias,
        input,
        expected,
    }
}

/// Build a Burn `Tensor<B, D>` from a flat row-major `Vec<f32>` with an
/// explicit shape. Thin helper around `TensorData::new` + `Tensor::from_data`
/// so the injection code below reads linearly.
fn tensor_from<const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Tensor<Backend, D> {
    Tensor::<Backend, D>::from_data(TensorData::new(data, shape), device)
}

/// Replace a block's parameters with values pulled from the fixture.
///
/// Mirrors Burn's `Param::from_tensor` / `RunningState::new` pattern
/// used in the reference `Linear` tests. We mark the new parameters as
/// not requiring gradients — the parity test forwards only, there's no
/// backward pass, and this avoids autodiff bookkeeping on the borrowed
/// tensors.
fn inject_block(
    block: &mut TinyConvBlock<Backend>,
    w: BlockWeights,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) {
    // Conv2d weights: PyTorch `[out, in, k, k]` = Burn `[out, in/g, kH, kW]`.
    let conv_down = tensor_from(w.conv_down, [w.out_ch, w.in_ch, 5, 5], device);
    block.conv_down.weight = Param::from_tensor(conv_down.set_require_grad(false));

    block.bn_down.gamma =
        Param::from_tensor(tensor_from(w.bn_down_gamma, [w.out_ch], device).set_require_grad(false));
    block.bn_down.beta =
        Param::from_tensor(tensor_from(w.bn_down_beta, [w.out_ch], device).set_require_grad(false));
    block.bn_down.running_mean =
        RunningState::new(tensor_from(w.bn_down_rmean, [w.out_ch], device));
    block.bn_down.running_var =
        RunningState::new(tensor_from(w.bn_down_rvar, [w.out_ch], device));

    let conv_refine = tensor_from(w.conv_refine, [w.out_ch, w.out_ch, 3, 3], device);
    block.conv_refine.weight = Param::from_tensor(conv_refine.set_require_grad(false));

    block.bn_refine.gamma = Param::from_tensor(
        tensor_from(w.bn_refine_gamma, [w.out_ch], device).set_require_grad(false),
    );
    block.bn_refine.beta = Param::from_tensor(
        tensor_from(w.bn_refine_beta, [w.out_ch], device).set_require_grad(false),
    );
    block.bn_refine.running_mean =
        RunningState::new(tensor_from(w.bn_refine_rmean, [w.out_ch], device));
    block.bn_refine.running_var =
        RunningState::new(tensor_from(w.bn_refine_rvar, [w.out_ch], device));
}

#[test]
fn tinyconv_forward_parity() {
    let fx = load_fixture("tinyconv_forward.bin");
    assert_eq!(fx.channels, vec![16, 32, 64]);

    let device = device();
    let cfg = TinyConvConfig {
        channels: fx.channels.clone(),
        dropout: 0.0,
        n_classes: fx.n_classes,
    };
    let mut model: TinyConv<Backend> = cfg.init(&device);

    // Inject per-block weights. `fx.weights` is consumed (drain) so we
    // can move each `BlockWeights` into `inject_block` without clones.
    for (block, w) in model.blocks.iter_mut().zip(fx.weights.into_iter()) {
        inject_block(block, w, &device);
    }

    // Head: PyTorch stores `Linear.weight` as `[n_classes, d_in]`; Burn's
    // default `LinearLayout::Row` expects `[d_in, n_classes]`. Transpose
    // on the way in. (Per the `burn_nn::Linear` docs in
    // `_reference/burn/crates/burn-nn/src/modules/linear.rs:45`.)
    let last_ch = *fx.channels.last().unwrap();
    let head_w_t = transpose_row_major(&fx.head_weight, fx.n_classes, last_ch);
    let head_w = tensor_from(head_w_t, [last_ch, fx.n_classes], &device);
    model.head.weight = Param::from_tensor(head_w.set_require_grad(false));
    let head_b = tensor_from(fx.head_bias, [fx.n_classes], &device);
    model.head.bias = Some(Param::from_tensor(head_b.set_require_grad(false)));

    let input = tensor_from(
        fx.input,
        [fx.batch, 1, fx.n_mels, fx.n_frames],
        &device,
    );

    let out = model.forward(input);
    let out_dims = out.dims();
    assert_eq!(
        out_dims,
        [fx.batch, fx.n_classes],
        "output shape mismatch",
    );

    let actual = out.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    let abs = max_abs_diff(&actual, &fx.expected);
    eprintln!(
        "[tinyconv_forward] channels={:?} bn_eps={:.1e} batch={} abs_diff={abs:.3e}",
        fx.channels, fx.bn_eps, fx.batch,
    );
    // 1e-4 absolute. TinyConv is a short chain of matmul+BN —
    // empirically well inside this bound on wgpu-Metal.
    assert!(
        abs < 1e-4,
        "TinyConv forward diverged: abs_diff {abs:.3e} exceeds 1e-4",
    );
}

fn transpose_row_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(src.len(), rows * cols);
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}
