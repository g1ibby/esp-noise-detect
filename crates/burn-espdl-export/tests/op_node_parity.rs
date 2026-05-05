//! Step 5 — exported FlatBuffers graph structure tests.

use burn::backend::NdArray;
use burn_espdl_export::{
    CalibrationConfig, EspdlFile, ExportConfig, calibrate, fold_batchnorm, fuse_relu, write_graph,
};
use std::collections::BTreeMap;

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::{MiniNetConfig, perturb_bn_stats};

type B = NdArray;
const INPUT_SHAPE: [usize; 4] = [1, 1, 16, 16];
const ESPDL_FIXTURE: &[u8] = include_bytes!("fixtures/mininet.espdl");

fn build_export() -> Vec<u8> {
    let device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0xfeed_face_u64);
    let mut graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    fold_batchnorm(&mut graph);
    fuse_relu(&mut graph);
    let windows = calibration_windows(32, 0x9e37_79b9_u64);
    let scales = calibrate::<B>(&graph, &windows, CalibrationConfig::esp32s3_int8(), &device)
        .expect("calibration");
    write_graph(&graph, &scales, ExportConfig::esp32s3_int8()).expect("export")
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

#[test]
fn emitted_graph_has_step5_node_sequence() {
    let bytes = build_export();
    assert_eq!(&bytes[0..4], b"EDL2");
    let parsed = EspdlFile::parse(&bytes).expect("parse exported espdl");
    let fixture = EspdlFile::parse(ESPDL_FIXTURE).expect("parse esp-ppq fixture");
    let graph = parsed.model().graph().expect("graph");
    let fixture_graph = fixture.model().graph().expect("fixture graph");
    let nodes = graph.node().expect("nodes");
    let fixture_nodes = fixture_graph.node().expect("fixture nodes");
    let ops: Vec<_> = (0..nodes.len())
        .map(|i| nodes.get(i).op_type().unwrap().to_string())
        .collect();
    let fixture_ops: Vec<_> = (0..fixture_nodes.len())
        .map(|i| fixture_nodes.get(i).op_type().unwrap().to_string())
        .collect();
    assert_eq!(ops, fixture_ops);
    assert!(!ops.iter().any(|op| op == "Relu"));
}

#[test]
fn emitted_nodes_match_esp_ppq_fixture() {
    let bytes = build_export();
    let actual = EspdlFile::parse(&bytes).expect("parse exported espdl");
    let expected = EspdlFile::parse(ESPDL_FIXTURE).expect("parse esp-ppq fixture");
    let actual_graph = actual.model().graph().expect("actual graph");
    let expected_graph = expected.model().graph().expect("expected graph");
    let actual_nodes = actual_graph.node().expect("actual nodes");
    let expected_nodes = expected_graph.node().expect("expected nodes");

    assert_eq!(actual_nodes.len(), expected_nodes.len(), "node count");
    for i in 0..expected_nodes.len() {
        let actual = actual_nodes.get(i);
        let expected = expected_nodes.get(i);
        assert_eq!(actual.name(), expected.name(), "node {i} name");
        assert_eq!(actual.op_type(), expected.op_type(), "node {i} op_type");
        assert_str_vec(
            actual.input(),
            expected.input(),
            &format!("node {i} inputs"),
        );
        assert_str_vec(
            actual.output(),
            expected.output(),
            &format!("node {i} outputs"),
        );
        assert_eq!(
            attr_map(actual.attribute()),
            attr_map(expected.attribute()),
            "node {i} attributes"
        );
    }
}

#[test]
fn value_infos_match_esp_ppq_fixture() {
    let bytes = build_export();
    let actual = EspdlFile::parse(&bytes).expect("parse exported espdl");
    let expected = EspdlFile::parse(ESPDL_FIXTURE).expect("parse esp-ppq fixture");
    let actual_graph = actual.model().graph().expect("actual graph");
    let expected_graph = expected.model().graph().expect("expected graph");

    assert_value_info_vec(actual_graph.input(), expected_graph.input(), "graph inputs");
    assert_value_info_vec(
        actual_graph.output(),
        expected_graph.output(),
        "graph outputs",
    );
}

#[test]
fn initializers_match_fixture_metadata() {
    let bytes = build_export();
    let actual = EspdlFile::parse(&bytes).expect("parse exported espdl");
    let expected = EspdlFile::parse(ESPDL_FIXTURE).expect("parse esp-ppq fixture");
    let actual_graph = actual.model().graph().expect("actual graph");
    let expected_graph = expected.model().graph().expect("expected graph");
    let actual_initializers = actual_graph.initializer().expect("actual initializers");
    let expected_initializers = expected_graph.initializer().expect("expected initializers");

    for i in 0..expected_initializers.len() {
        let expected = expected_initializers.get(i);
        let name = expected.name().expect("fixture initializer name");
        let actual = find_initializer(&actual_initializers, name);
        assert_eq!(
            actual.data_type(),
            expected.data_type(),
            "data_type for {name}"
        );
        assert_eq!(
            actual.doc_string(),
            expected.doc_string(),
            "doc_string for {name}"
        );
        assert_i64_vec(actual.dims(), expected.dims(), &format!("dims for {name}"));
        assert_i64_vec(
            actual.exponents(),
            expected.exponents(),
            &format!("exponents for {name}"),
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
enum AttrValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
}

fn attr_map(
    attrs: Option<
        flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<burn_espdl_export::dl::Attribute<'_>>>,
    >,
) -> BTreeMap<String, AttrValue> {
    let mut out = BTreeMap::new();
    if let Some(attrs) = attrs {
        for i in 0..attrs.len() {
            let attr = attrs.get(i);
            let name = attr.name().unwrap().to_string();
            let value = match attr.attr_type() {
                burn_espdl_export::dl::AttributeType::INT => AttrValue::Int(attr.i().unwrap().i()),
                burn_espdl_export::dl::AttributeType::FLOAT => {
                    AttrValue::Float(attr.f().unwrap().f())
                }
                burn_espdl_export::dl::AttributeType::STRING => {
                    let s = attr.s().unwrap();
                    AttrValue::String(
                        String::from_utf8((0..s.len()).map(|j| s.get(j)).collect()).unwrap(),
                    )
                }
                burn_espdl_export::dl::AttributeType::INTS => {
                    let ints = attr.ints().unwrap();
                    AttrValue::Ints((0..ints.len()).map(|j| ints.get(j)).collect())
                }
                other => panic!("unsupported attr type {other:?} for {name}"),
            };
            out.insert(name, value);
        }
    }
    out
}

fn assert_str_vec(
    actual: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<&str>>>,
    expected: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<&str>>>,
    label: &str,
) {
    let actual = actual
        .map(|v| {
            (0..v.len())
                .map(|i| v.get(i).to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let expected = expected
        .map(|v| {
            (0..v.len())
                .map(|i| v.get(i).to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    assert_eq!(actual, expected, "{label}");
}

fn assert_value_info_vec(
    actual: Option<
        flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<burn_espdl_export::dl::ValueInfo<'_>>>,
    >,
    expected: Option<
        flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<burn_espdl_export::dl::ValueInfo<'_>>>,
    >,
    label: &str,
) {
    let actual = actual.expect("actual value info");
    let expected = expected.expect("expected value info");
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for i in 0..expected.len() {
        let a = actual.get(i);
        let e = expected.get(i);
        assert_eq!(a.name(), e.name(), "{label} {i} name");
        assert_eq!(a.doc_string(), e.doc_string(), "{label} {i} doc_string");
        assert_i64_vec(
            a.exponents(),
            e.exponents(),
            &format!("{label} {i} exponents"),
        );
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
