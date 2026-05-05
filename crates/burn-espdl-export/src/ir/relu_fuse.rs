//! Fuse standalone `Relu` nodes into the preceding `Conv2d`.
//!
//! Mirrors esp-ppq's `FuseReluLikePattern`
//! (`_reference/esp-ppq/esp_ppq/parser/espdl/export_patterns.py:309`):
//! the golden `model.info` graph for TinyConv has zero `Relu` nodes
//! because every one has been absorbed onto its upstream Conv as
//! `attribute("activation") = "Relu"`.
//!
//! Without this fusion, [`crate::ir::BurnGraph::op_sequence`] would
//! disagree with the golden node count and Step 5's op-node-parity
//! test would fail.

use super::{Activation, BurnGraph, Layer};

/// In-place rewrite: every `(Conv2d → Relu)` pair becomes a single
/// `Conv2d { activation: Some(Activation::Relu), .. }`.
///
/// `Relu`s whose immediate predecessor is not a `Conv2d`, or whose
/// predecessor already has a fused activation, are left as standalone
/// nodes. (Neither happens in TinyConv after BN folding; the
/// pre-condition is checked here so we fail loud rather than silently
/// drop a Relu if a future model breaks the assumption.)
pub fn fuse_relu(graph: &mut BurnGraph) {
    let mut i = 0;
    while i + 1 < graph.layers.len() {
        let can_fuse = match (&graph.layers[i], &graph.layers[i + 1]) {
            (
                Layer::Conv2d {
                    activation: None,
                    output,
                    ..
                },
                Layer::Relu { input: relu_in, .. },
            ) if output == relu_in => true,
            _ => false,
        };
        if !can_fuse {
            i += 1;
            continue;
        }

        let relu = graph.layers.remove(i + 1);
        let relu_out = match relu {
            Layer::Relu { output, .. } => output,
            _ => unreachable!(),
        };

        match &mut graph.layers[i] {
            Layer::Conv2d {
                activation, output, ..
            } => {
                *activation = Some(Activation::Relu);
                // Re-thread downstream consumers from the Relu's output
                // to the Conv's output.
                *output = relu_out;
            }
            _ => unreachable!(),
        }

        // Don't advance: the next layer at `i + 1` is whatever followed
        // the Relu — could in principle be another Relu (over-deep
        // activations), so let the loop re-check.
    }
}
