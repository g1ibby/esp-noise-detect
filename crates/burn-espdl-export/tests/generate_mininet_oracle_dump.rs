//! Ignored integration test that dumps the Step-4 `MiniNet` fixture as
//! plain JSON for the offline esp-ppq oracle generator.
//!
//! This is test-fixture generation machinery, not a library example.
//! It builds the same graph as `calibration_parity.rs` and writes enough
//! data for `tests/fixtures/generate/mininet_oracle.py` to build ONNX
//! and calibration `.npy` files inside the esp-ppq Docker image.

use std::env;
use std::fs;
use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn_espdl_export::{BurnGraph, IrTensor, Layer, fold_batchnorm, fuse_relu};

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::{MiniNetConfig, perturb_bn_stats};

type B = NdArray;

const INPUT_SHAPE: [usize; 4] = [1, 1, 16, 16];
const MODEL_SEED: u64 = 0xfeed_face_u64;
const CALIB_SEED: u64 = 0x9e37_79b9_u64;
const CALIB_WINDOWS: usize = 32;

#[test]
#[ignore = "offline fixture generator; set MININET_ORACLE_DUMP or uses /tmp/mininet-oracle/dump.json"]
fn dump_mininet_oracle() {
    let out = env::var_os("MININET_ORACLE_DUMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/mininet-oracle/dump.json"));

    let device = Default::default();
    let graph = build_graph(&device);
    let windows = calibration_windows(CALIB_WINDOWS, CALIB_SEED);
    let json = render_dump(&graph, &windows);

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    fs::write(&out, json).expect("write dump json");
    eprintln!("wrote {}", out.display());
}

fn build_graph(device: &<B as Backend>::Device) -> BurnGraph {
    let mut model = MiniNetConfig::default().init::<B>(device);
    perturb_bn_stats(&mut model, device, MODEL_SEED);
    let mut graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);
    graph
}

/// Same deterministic synthetic windows as `tests/calibration_parity.rs`.
fn calibration_windows(n_windows: usize, seed: u64) -> Vec<Vec<f32>> {
    let n_per = INPUT_SHAPE.iter().product::<usize>();
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(n_windows);
    for _ in 0..n_windows {
        let mut window = Vec::with_capacity(n_per);
        while window.len() < n_per {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u1 = ((state >> 32) as u32 as f32 / u32::MAX as f32).max(1e-9);
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u2 = (state >> 32) as u32 as f32 / u32::MAX as f32;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * core::f32::consts::PI * u2;
            window.push(r * theta.cos());
            if window.len() < n_per {
                window.push(r * theta.sin());
            }
        }
        out.push(window);
    }
    out
}

fn render_dump(graph: &BurnGraph, windows: &[Vec<f32>]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    push_kv_array_usize(&mut s, "input_shape", &graph.input_shape, 2, true);
    push_kv_str(&mut s, "input_name", &graph.input_name, 2, true);
    push_kv_str(&mut s, "output_name", &graph.output_name, 2, true);
    s.push_str("  \"layers\": [\n");
    for (idx, layer) in graph.layers.iter().enumerate() {
        render_layer(&mut s, layer, 4);
        if idx + 1 != graph.layers.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ],\n");
    s.push_str("  \"calibration_windows\": [\n");
    for (idx, window) in windows.iter().enumerate() {
        push_indent(&mut s, 4);
        push_f32_array(&mut s, window);
        if idx + 1 != windows.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

fn render_layer(s: &mut String, layer: &Layer, indent: usize) {
    push_indent(s, indent);
    s.push_str("{\n");
    match layer {
        Layer::Conv2d {
            input,
            output,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            activation,
        } => {
            push_kv_str(s, "type", "Conv2d", indent + 2, true);
            push_kv_str(s, "input", input, indent + 2, true);
            push_kv_str(s, "output", output, indent + 2, true);
            push_tensor(s, "weight", weight, indent + 2, true);
            if let Some(bias) = bias {
                push_tensor(s, "bias", bias, indent + 2, true);
            } else {
                push_kv_null(s, "bias", indent + 2, true);
            }
            push_kv_array_usize(s, "stride", stride, indent + 2, true);
            push_kv_array_usize(s, "padding", padding, indent + 2, true);
            push_kv_array_usize(s, "dilation", dilation, indent + 2, true);
            push_kv_usize(s, "groups", *groups, indent + 2, true);
            let activation = activation.map(|a| a.to_string());
            match activation {
                Some(a) => push_kv_str(s, "activation", &a, indent + 2, false),
                None => push_kv_null(s, "activation", indent + 2, false),
            }
        }
        Layer::ReduceMean {
            input,
            output,
            axes,
            keepdims,
        } => {
            push_kv_str(s, "type", "ReduceMean", indent + 2, true);
            push_kv_str(s, "input", input, indent + 2, true);
            push_kv_str(s, "output", output, indent + 2, true);
            push_kv_array_i64(s, "axes", axes, indent + 2, true);
            push_kv_bool(s, "keepdims", *keepdims, indent + 2, false);
        }
        Layer::Linear {
            input,
            output,
            weight,
            bias,
        } => {
            push_kv_str(s, "type", "Linear", indent + 2, true);
            push_kv_str(s, "input", input, indent + 2, true);
            push_kv_str(s, "output", output, indent + 2, true);
            push_tensor(s, "weight", weight, indent + 2, true);
            if let Some(bias) = bias {
                push_tensor(s, "bias", bias, indent + 2, false);
            } else {
                push_kv_null(s, "bias", indent + 2, false);
            }
        }
        Layer::BatchNorm2d { .. } | Layer::Relu { .. } => {
            panic!("dump_mininet_oracle expects post-fold/post-fuse graph");
        }
    }
    s.push('\n');
    push_indent(s, indent);
    s.push('}');
}

fn push_tensor(s: &mut String, key: &str, tensor: &IrTensor, indent: usize, comma: bool) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(": {\"shape\": ");
    push_usize_array(s, &tensor.shape);
    s.push_str(", \"data\": ");
    push_f32_array(s, &tensor.data);
    s.push('}');
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_kv_str(s: &mut String, key: &str, value: &str, indent: usize, comma: bool) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(": ");
    push_json_string(s, value);
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_kv_null(s: &mut String, key: &str, indent: usize, comma: bool) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(": null");
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_kv_bool(s: &mut String, key: &str, value: bool, indent: usize, comma: bool) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(if value { ": true" } else { ": false" });
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_kv_usize(s: &mut String, key: &str, value: usize, indent: usize, comma: bool) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(": ");
    s.push_str(&value.to_string());
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_kv_array_usize<const N: usize>(
    s: &mut String,
    key: &str,
    value: &[usize; N],
    indent: usize,
    comma: bool,
) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(": ");
    push_usize_array(s, value);
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_kv_array_i64(s: &mut String, key: &str, value: &[i64], indent: usize, comma: bool) {
    push_indent(s, indent);
    push_json_string(s, key);
    s.push_str(": ");
    s.push('[');
    for (idx, v) in value.iter().enumerate() {
        if idx > 0 {
            s.push_str(", ");
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
    if comma {
        s.push(',');
    }
    s.push('\n');
}

fn push_usize_array(s: &mut String, value: &[usize]) {
    s.push('[');
    for (idx, v) in value.iter().enumerate() {
        if idx > 0 {
            s.push_str(", ");
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
}

fn push_f32_array(s: &mut String, value: &[f32]) {
    s.push('[');
    for (idx, v) in value.iter().enumerate() {
        if idx > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{v:.9e}"));
    }
    s.push(']');
}

fn push_json_string(s: &mut String, value: &str) {
    s.push('"');
    for ch in value.chars() {
        match ch {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c => s.push(c),
        }
    }
    s.push('"');
}

fn push_indent(s: &mut String, n: usize) {
    for _ in 0..n {
        s.push(' ');
    }
}
