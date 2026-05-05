//! Fold every `BatchNorm2d` into the preceding `Conv2d`.
//!
//! Standard PTQ pre-processing: at inference time a 2-D BatchNorm is a
//! per-channel affine transform with constant parameters, so we can
//! rewrite
//!
//! ```text
//!   y = γ · ((Conv(x) − μ) / √(σ² + ε)) + β
//! ```
//!
//! into a single Conv whose weights and bias are the original Conv's
//! weights scaled by `γ / √(σ² + ε)` and a fresh bias of
//! `β − γ μ / √(σ² + ε)` (plus the original Conv bias if it had one).
//!
//! The fold removes the BN node from the graph, threading the Conv's
//! output name forward to whatever consumed the BN's output. This
//! mirrors esp-ppq's behaviour at the ONNX boundary: the golden
//! `model.info` graph has zero `BatchNormalization` nodes — they have
//! all been absorbed into the Convs upstream.

use super::{BurnGraph, Layer, Tensor};

/// In-place rewrite: every `(Conv2d → BatchNorm2d)` pair becomes a
/// single `Conv2d` with the BN baked into its weights and bias. BN
/// nodes whose immediate predecessor is *not* a `Conv2d` are left
/// alone (TinyConv never produces such a pair, but keeping the pass
/// total avoids surprises if the IR ever sees a different model).
pub fn fold_batchnorm(graph: &mut BurnGraph) {
    let mut i = 0;
    while i + 1 < graph.layers.len() {
        let can_fold = match (&graph.layers[i], &graph.layers[i + 1]) {
            (
                Layer::Conv2d { output, weight, .. },
                Layer::BatchNorm2d {
                    input: bn_in,
                    gamma,
                    ..
                },
            ) if output == bn_in => {
                // Conv weight `[out_c, in_c/g, kH, kW]`; γ is `[out_c]`.
                let out_c = weight.shape[0];
                gamma.shape == vec![out_c]
            }
            _ => false,
        };
        if !can_fold {
            i += 1;
            continue;
        }

        // Take ownership so we can mutate independently of the borrow.
        let bn = graph.layers.remove(i + 1);
        let (bn_output, gamma, beta, mean, var, eps) = match bn {
            Layer::BatchNorm2d {
                output,
                gamma,
                beta,
                running_mean,
                running_var,
                epsilon,
                ..
            } => (output, gamma, beta, running_mean, running_var, epsilon),
            _ => unreachable!(),
        };

        // Apply the fold to the Conv.
        let scale = bn_scale(&gamma, &var, eps); // γ / √(σ² + ε)
        let new_bias = bn_bias(&beta, &mean, &scale); //  β − μ · scale (+ existing conv bias scaled)
        match &mut graph.layers[i] {
            Layer::Conv2d {
                weight,
                bias,
                output,
                ..
            } => {
                scale_conv_weight(weight, &scale);
                let folded_bias = match bias.take() {
                    None => new_bias,
                    Some(prev) => add_biases(&prev, &scale, &new_bias),
                };
                *bias = Some(folded_bias);
                // Re-thread downstream consumers from the BN's output to
                // the Conv's output. We accomplish this by *renaming*
                // the Conv's output to whatever the BN was producing:
                // any later layer that consumed `bn_output` keeps
                // working with no rewrites of its own.
                *output = bn_output;
            }
            _ => unreachable!(),
        }

        // Don't advance `i`: the next layer (was at `i + 2`, now `i + 1`)
        // could itself be a BN if some future model stacks them.
    }
}

/// `γ / √(σ² + ε)`. All three inputs are `[C]`.
fn bn_scale(gamma: &Tensor, var: &Tensor, eps: f64) -> Vec<f32> {
    debug_assert_eq!(gamma.shape, var.shape);
    let eps32 = eps as f32;
    gamma
        .data
        .iter()
        .zip(var.data.iter())
        .map(|(g, v)| g / (v + eps32).sqrt())
        .collect()
}

/// `β − μ · scale`. All three are `[C]`.
fn bn_bias(beta: &Tensor, mean: &Tensor, scale: &[f32]) -> Tensor {
    debug_assert_eq!(beta.shape, mean.shape);
    debug_assert_eq!(beta.numel(), scale.len());
    let data = beta
        .data
        .iter()
        .zip(mean.data.iter())
        .zip(scale.iter())
        .map(|((b, m), s)| b - m * s)
        .collect();
    Tensor::new(data, beta.shape.clone())
}

/// Multiply the Conv weight per output channel by `scale[c]`.
/// Weight is `[out_c, in_c/groups, kH, kW]`, row-major.
fn scale_conv_weight(weight: &mut Tensor, scale: &[f32]) {
    let out_c = weight.shape[0];
    debug_assert_eq!(out_c, scale.len());
    let per_channel = weight.shape.iter().skip(1).product::<usize>();
    debug_assert_eq!(weight.numel(), out_c * per_channel);
    for c in 0..out_c {
        let s = scale[c];
        let start = c * per_channel;
        let end = start + per_channel;
        for v in &mut weight.data[start..end] {
            *v *= s;
        }
    }
}

/// `prev_bias[c] · scale[c] + new_bias[c]`. Used when the original
/// Conv already had a bias *and* a BN follows (TinyConv's blocks
/// disable Conv bias, so this branch is dormant for the production
/// model — it's here for correctness, not for any current call site).
fn add_biases(prev: &Tensor, scale: &[f32], new_bias: &Tensor) -> Tensor {
    debug_assert_eq!(prev.shape, new_bias.shape);
    debug_assert_eq!(prev.numel(), scale.len());
    let data = prev
        .data
        .iter()
        .zip(scale.iter())
        .zip(new_bias.data.iter())
        .map(|((p, s), n)| p * s + n)
        .collect();
    Tensor::new(data, prev.shape.clone())
}
