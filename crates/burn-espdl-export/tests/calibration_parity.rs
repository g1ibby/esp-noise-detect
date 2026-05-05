//! Step 4 — calibration parity test (the important one per task spec).
//!
//! Runs the two-phase calibration over the inline `MiniNet` fixture
//! with deterministic synthetic windows, then asserts:
//!
//! 1. **Structural completeness**: every quantizable tensor has an
//!    entry — graph input, every layer output, every Conv/Linear
//!    weight, every Conv/Linear bias.
//! 2. **Determinism**: same seed → same scales / exponents.
//! 3. **Pow-2 invariant**: every scale is exactly `2^exp` for the
//!    emitted exponent (this is what the on-device decoder expects;
//!    a mismatch would silently corrupt inference).
//! 4. **Bias passivity**: every Conv/Linear bias scale equals
//!    `scale_input · scale_weight` of the same layer.
//! 5. **esp-ppq parity** (when the offline-generated fixture
//!    `tests/fixtures/mininet_calib.json` is present): per-tensor
//!    `(scale, exponent)` matches within the Step-1 tolerance for
//!    quant-config parity (exact match — `int(log2(scale))` is
//!    integer; scale is exactly `2^exp`).
//!
//! The fixture in (5) must be regenerated offline by passing the same
//! `MiniNet` definition through esp-ppq (Docker). Until that fixture
//! lands the test prints a "skipping" notice and runs (1)–(4) only,
//! mirroring the pattern already in `tests/common/mod.rs::read_golden`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn_espdl_export::{
    BurnGraph, CalibrationConfig, TensorRole, calibrate, fold_batchnorm, fuse_relu,
};

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::{MiniNetConfig, perturb_bn_stats};

type B = NdArray;

const INPUT_SHAPE: [usize; 4] = [1, 1, 16, 16];

fn build_graph(device: &<B as Backend>::Device, seed: u64) -> BurnGraph {
    let mut model = MiniNetConfig::default().init::<B>(device);
    perturb_bn_stats(&mut model, device, seed);
    let mut graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);
    graph
}

/// Deterministic synthetic calibration windows: 32 inputs drawn
/// from N(0, 1) via Box-Muller from a seeded LCG. Mirrors the
/// generator used in `tests/forward_parity_fp32.rs`.
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

#[test]
fn structural_completeness_int8() {
    let device = Default::default();
    let graph = build_graph(&device, 0xfeed_face_u64);
    let windows = calibration_windows(32, 0x9e37_79b9_u64);

    let table = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration");

    // Every named activation gets an entry.
    let mut expected_act: Vec<&str> = vec![graph.input_name.as_str()];
    expected_act.extend(graph.layers.iter().map(|l| l.output()));
    for name in expected_act {
        let entry = table
            .get(name)
            .unwrap_or_else(|| panic!("activation {name} missing from ScaleTable"));
        assert_eq!(
            entry.role,
            TensorRole::Activation,
            "{name} should be an activation"
        );
        assert_eq!(entry.config.num_bits, 8);
    }

    // Every Conv2d / Linear has weight + bias entries.
    for layer in &graph.layers {
        match layer {
            burn_espdl_export::Layer::Conv2d { output, bias, .. } => {
                let w_key = format!("{output}.weight");
                let entry = table
                    .get(&w_key)
                    .unwrap_or_else(|| panic!("{w_key} missing"));
                assert_eq!(entry.role, TensorRole::Weight);
                assert_eq!(entry.config.num_bits, 8);
                if bias.is_some() {
                    let b_key = format!("{output}.bias");
                    let bias_entry = table
                        .get(&b_key)
                        .unwrap_or_else(|| panic!("{b_key} missing"));
                    assert_eq!(bias_entry.role, TensorRole::Bias);
                    assert_eq!(bias_entry.config.num_bits, 20, "INT8 bias bit width");
                }
            }
            burn_espdl_export::Layer::Linear { output, bias, .. } => {
                let w_key = format!("{output}.weight");
                let entry = table
                    .get(&w_key)
                    .unwrap_or_else(|| panic!("{w_key} missing"));
                assert_eq!(entry.role, TensorRole::Weight);
                if bias.is_some() {
                    let b_key = format!("{output}.bias");
                    let bias_entry = table
                        .get(&b_key)
                        .unwrap_or_else(|| panic!("{b_key} missing"));
                    assert_eq!(bias_entry.role, TensorRole::Bias);
                    assert_eq!(bias_entry.config.num_bits, 20);
                }
            }
            _ => {}
        }
    }
}

#[test]
fn deterministic_across_runs() {
    let device = Default::default();
    let graph = build_graph(&device, 0xfeed_face_u64);
    let windows = calibration_windows(32, 0x9e37_79b9_u64);

    let a = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration A");
    let b = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration B");

    assert_eq!(a.len(), b.len(), "ScaleTable len mismatch");
    for ((ka, va), (kb, vb)) in a.iter().zip(b.iter()) {
        assert_eq!(ka, kb, "key order mismatch");
        assert_eq!(va.role, vb.role);
        assert_eq!(va.config.scale, vb.config.scale, "scale at {ka}");
        assert_eq!(va.config.exponent, vb.config.exponent, "exp at {ka}");
        assert_eq!(va.config.num_bits, vb.config.num_bits);
    }
}

#[test]
fn every_scale_is_a_power_of_two() {
    let device = Default::default();
    let graph = build_graph(&device, 0xfeed_face_u64);
    let windows = calibration_windows(16, 0xc0ffee_u64);
    let table = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration");
    for (name, entry) in table.iter() {
        let s = entry.config.scale;
        let e = entry.config.exponent;
        assert!(s > 0.0, "scale at {name} should be positive (got {s})");
        // 2^exp must round-trip to scale exactly.
        let reconstructed = (2.0_f32).powi(e);
        assert_eq!(
            reconstructed, s,
            "scale {s} at {name} not exactly 2^{e} (got 2^{e} = {reconstructed})",
        );
    }
}

#[test]
fn bias_scale_equals_scale_input_times_scale_weight() {
    let device = Default::default();
    let graph = build_graph(&device, 0xfeed_face_u64);
    let windows = calibration_windows(32, 0x1234_5678_u64);
    let table = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration");

    for layer in &graph.layers {
        let (input_name, output_name, has_bias) = match layer {
            burn_espdl_export::Layer::Conv2d {
                input,
                output,
                bias,
                ..
            } => (input.clone(), output.clone(), bias.is_some()),
            burn_espdl_export::Layer::Linear {
                input,
                output,
                bias,
                ..
            } => (input.clone(), output.clone(), bias.is_some()),
            _ => continue,
        };
        if !has_bias {
            continue;
        }
        let s_in = table.get(&input_name).expect("input scale").config.scale;
        let s_w = table
            .get(&format!("{output_name}.weight"))
            .expect("weight scale")
            .config
            .scale;
        let s_b = table
            .get(&format!("{output_name}.bias"))
            .expect("bias scale")
            .config
            .scale;
        // scale_in * scale_w is exact in f32 when both are pow-2.
        assert_eq!(
            s_b,
            s_in * s_w,
            "bias scale at {output_name} ({s_b}) ≠ scale_in × scale_w ({s_in} × {s_w} = {})",
            s_in * s_w,
        );
    }
}

/// Per-tensor `(scale, exponent)` parity against esp-ppq's own
/// `model.json`, regenerated offline by passing the same `MiniNet`
/// fixture through esp-ppq.
///
/// Skipped when the fixture file is absent — generating it requires
/// running the esp-ppq Docker pipeline against a one-off ONNX export
/// of `MiniNet`, which is the agent's offline follow-up. The test
/// fails *only* when the fixture exists and a tensor's
/// `(scale, exponent)` disagrees with esp-ppq's.
#[test]
fn matches_esp_ppq_oracle_when_present() {
    let fixture = fixture_path("mininet_calib.json");
    let bytes = match std::fs::read(&fixture) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skipping esp-ppq parity check: oracle fixture not present at {} ({e}). \
                 Regenerate offline by running esp-ppq against the in-test MiniNet.",
                fixture.display(),
            );
            return;
        }
    };
    let oracle: BTreeMap<String, OracleEntry> =
        serde_lite::from_str(&String::from_utf8(bytes).expect("fixture is utf-8"));

    let device = Default::default();
    let graph = build_graph(&device, 0xfeed_face_u64);
    let windows = calibration_windows(32, 0x9e37_79b9_u64);
    let table = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration");

    let mut diffs: Vec<String> = vec![];
    for (name, expected) in &oracle {
        match table.get(name) {
            Some(got) => {
                if got.config.exponent != expected.exponent {
                    diffs.push(format!(
                        "  {name}: exp got {} expected {}",
                        got.config.exponent, expected.exponent,
                    ));
                }
                if (got.config.scale - expected.scale).abs() > 0.0 {
                    // Both should be pow-2 → exact equality. A
                    // tolerance line lives here only to make the
                    // diff message clearer if the assumption ever
                    // breaks.
                    diffs.push(format!(
                        "  {name}: scale got {} expected {}",
                        got.config.scale, expected.scale,
                    ));
                }
            }
            None => diffs.push(format!("  {name}: missing from new table")),
        }
    }
    if !diffs.is_empty() {
        panic!(
            "calibration parity vs esp-ppq oracle failed ({} divergences):\n{}",
            diffs.len(),
            diffs.join("\n"),
        );
    }
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Tiny JSON oracle row. Layout matches what an offline esp-ppq run
/// would dump (`scale: f32, exponent: i32`); we keep the schema
/// minimal so the agent does not commit incidental fields like
/// `dominator` or `state` that are esp-ppq internals.
#[derive(Debug)]
struct OracleEntry {
    scale: f32,
    exponent: i32,
}

// Minimal handwritten JSON parser scoped to the oracle file's
// expected shape:
//
// ```json
// {
//   "<tensor_name>": {"scale": 0.03125, "exponent": -5},
//   ...
// }
// ```
//
// We avoid a `serde_json` dev-dependency to keep the crate's
// `[dev-dependencies]` strictly empty of new crates (the
// architectural-non-negotiables checklist enforces this via
// `cargo tree`).
mod serde_lite {
    use super::OracleEntry;
    use std::collections::BTreeMap;

    pub fn from_str(s: &str) -> BTreeMap<String, OracleEntry> {
        let s = s.trim();
        let s = s.strip_prefix('{').expect("oracle: top-level object");
        let s = s.strip_suffix('}').expect("oracle: top-level object close");
        let mut out = BTreeMap::new();
        // Walk until we hit the closing brace; entries look like
        // `"<name>": {"scale": <f>, "exponent": <i>}` separated by
        // commas. Object values may contain commas, so we count
        // braces.
        let mut bytes = s.as_bytes();
        loop {
            bytes = skip_whitespace(bytes);
            if bytes.is_empty() {
                break;
            }
            let (key, rest) = read_string(bytes);
            let rest = skip_whitespace(rest);
            assert_eq!(rest.first(), Some(&b':'), "oracle: expected ':' after key");
            let rest = skip_whitespace(&rest[1..]);
            let (entry, rest) = read_object(rest);
            out.insert(key, entry);
            let rest = skip_whitespace(rest);
            if rest.is_empty() {
                break;
            }
            assert_eq!(
                rest.first(),
                Some(&b','),
                "oracle: expected ',' between entries"
            );
            bytes = &rest[1..];
        }
        out
    }

    fn skip_whitespace(s: &[u8]) -> &[u8] {
        let mut i = 0;
        while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        &s[i..]
    }

    fn read_string(s: &[u8]) -> (String, &[u8]) {
        assert_eq!(s.first(), Some(&b'"'), "oracle: expected string start");
        let s = &s[1..];
        let end = s
            .iter()
            .position(|&b| b == b'"')
            .expect("oracle: unterminated string");
        let key = std::str::from_utf8(&s[..end])
            .expect("oracle: utf-8 key")
            .to_string();
        (key, &s[end + 1..])
    }

    fn read_object(s: &[u8]) -> (OracleEntry, &[u8]) {
        assert_eq!(s.first(), Some(&b'{'), "oracle: expected object start");
        // Find matching close brace (no nesting expected for our
        // shape, but be defensive).
        let mut depth = 1;
        let mut i = 1;
        while i < s.len() && depth > 0 {
            match s[i] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        let body = std::str::from_utf8(&s[1..i - 1]).expect("oracle: utf-8 body");
        let mut scale: Option<f32> = None;
        let mut exponent: Option<i32> = None;
        for part in body.split(',') {
            let (k, v) = part.split_once(':').expect("oracle: 'k: v'");
            let k = k.trim().trim_matches('"');
            let v = v.trim();
            match k {
                "scale" => scale = Some(v.parse().expect("oracle: scale")),
                "exponent" => exponent = Some(v.parse().expect("oracle: exponent")),
                _ => {}
            }
        }
        (
            OracleEntry {
                scale: scale.expect("oracle: missing scale"),
                exponent: exponent.expect("oracle: missing exponent"),
            },
            &s[i..],
        )
    }
}
