//! Step 5 — quantized payload parity against esp-ppq.

use burn::backend::NdArray;
use burn_espdl_export::{
    CalibrationConfig, EspdlFile, ExportConfig, calibrate, fold_batchnorm, fuse_relu, write_graph,
};

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::{MiniNetConfig, perturb_bn_stats};

type B = NdArray;
const INPUT_SHAPE: [usize; 4] = [1, 1, 16, 16];
const ESPDL_FIXTURE: &[u8] = include_bytes!("fixtures/mininet.espdl");

#[test]
fn exported_weight_payloads_match_esp_ppq_fixture() {
    let device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0xfeed_face_u64);
    let mut graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);
    let windows = calibration_windows(32, 0x9e37_79b9_u64);
    let scales = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration");
    let bytes = write_graph(&graph, &scales, ExportConfig::esp32s3_int8()).expect("export");
    let actual = EspdlFile::parse(&bytes).expect("parse exported espdl");
    let expected = EspdlFile::parse(ESPDL_FIXTURE).expect("parse esp-ppq fixture");
    let actual_graph = actual.model().graph().expect("actual graph");
    let expected_graph = expected.model().graph().expect("expected graph");
    let actual_initializers = actual_graph.initializer().expect("actual initializers");
    let expected_initializers = expected_graph.initializer().expect("expected initializers");

    for i in 0..expected_initializers.len() {
        let expected_tensor = expected_initializers.get(i);
        let name = expected_tensor.name().expect("fixture tensor name");
        if !(name.ends_with(".weight") || name.ends_with(".bias")) {
            continue;
        }
        let actual_tensor = find_initializer(&actual_initializers, name);
        assert_eq!(
            actual_tensor.data_type(),
            expected_tensor.data_type(),
            "data_type for {name}"
        );
        assert_i64_vec(
            actual_tensor.dims(),
            expected_tensor.dims(),
            &format!("dims for {name}"),
        );
        assert_i64_vec(
            actual_tensor.exponents(),
            expected_tensor.exponents(),
            &format!("exponents for {name}"),
        );

        let expected_values = raw_int_values(&expected_tensor);
        let actual_values = raw_int_values(&actual_tensor);
        assert_lsb_close(name, &actual_values, &expected_values);
    }
}

fn find_initializer<'a>(
    initializers: &flatbuffers::Vector<
        'a,
        flatbuffers::ForwardsUOffset<burn_espdl_export::dl::Tensor<'a>>,
    >,
    name: &str,
) -> burn_espdl_export::dl::Tensor<'a> {
    (0..initializers.len())
        .map(|i| initializers.get(i))
        .find(|t| t.name() == Some(name))
        .unwrap_or_else(|| panic!("missing initializer {name}"))
}

fn assert_i64_vec(
    actual: Option<flatbuffers::Vector<'_, i64>>,
    expected: Option<flatbuffers::Vector<'_, i64>>,
    label: &str,
) {
    let actual = actual
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect::<Vec<_>>())
        .unwrap_or_default();
    let expected = expected
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect::<Vec<_>>())
        .unwrap_or_default();
    assert_eq!(actual, expected, "{label}");
}

fn raw_int_values(t: &burn_espdl_export::dl::Tensor<'_>) -> Vec<i64> {
    let raw = t.raw_data().expect("raw_data");
    let dims_numel: usize = t
        .dims()
        .map(|v| (0..v.len()).map(|i| v.get(i) as usize).product())
        .unwrap_or(0);
    let mut bytes = Vec::with_capacity(raw.len() * 16);
    for i in 0..raw.len() {
        bytes.extend(raw.get(i).bytes().iter());
    }
    match t.data_type() {
        burn_espdl_export::dl::TensorDataType::INT8 => bytes
            .into_iter()
            .take(dims_numel)
            .map(|b| (b as i8) as i64)
            .collect(),
        burn_espdl_export::dl::TensorDataType::INT32 => bytes
            .chunks_exact(4)
            .take(dims_numel)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
            .collect(),
        other => panic!(
            "unsupported parity tensor type {other:?} for {:?}",
            t.name()
        ),
    }
}

fn assert_lsb_close(name: &str, actual: &[i64], expected: &[i64]) {
    assert_eq!(actual.len(), expected.len(), "payload length for {name}");
    let bad = actual
        .iter()
        .zip(expected)
        .filter(|(a, e)| (*a - *e).abs() > 1)
        .count();
    let max_bad = actual.len() / 100;
    assert!(
        bad <= max_bad,
        "{name}: {bad}/{} values differ by more than 1 LSB (allowed {max_bad})",
        actual.len()
    );
}

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
