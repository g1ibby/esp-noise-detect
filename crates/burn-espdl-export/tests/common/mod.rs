//! Test-only helpers shared across the `.espdl` round-trip tests.
//!
//! Two responsibilities:
//!
//! 1. Locate the golden `.espdl` artifact produced by the legacy esp-ppq
//!    pipeline (`/tmp/nn-rs-robust-cuda/export/model.espdl`). Tests skip
//!    cleanly when it is missing instead of failing — CI is expected to
//!    have it mounted.
//! 2. Provide a structural-equality assertion over [`dl::Model`] that
//!    walks every field the device-side loader consumes. Byte-equality
//!    is intentionally **not** required because FlatBuffers does not
//!    promise stable field/vtable ordering.

// Each integration-test binary picks the subset of these helpers it
// uses; from any single test's perspective the rest look unused.
#![allow(dead_code)]

use burn_espdl_export::dl;

pub mod fixture_lowering;
pub mod fixture_model;

pub const GOLDEN_PATH: &str = "/tmp/nn-rs-robust-cuda/export/model.espdl";

/// Read the golden artifact, or return `None` (with a printed warning)
/// if it is not present.
pub fn read_golden() -> Option<Vec<u8>> {
    match std::fs::read(GOLDEN_PATH) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!(
                "skipping: golden artifact not available at {GOLDEN_PATH} ({e}). \
                 Re-run the existing burn_to_espdl pipeline to produce it."
            );
            None
        }
    }
}

/// Assert two parsed models are structurally equal across every field
/// the device-side loader (`edgedl`) consumes.
pub fn assert_models_structurally_equal(a: &dl::Model<'_>, b: &dl::Model<'_>) {
    assert_eq!(a.ir_version(), b.ir_version(), "ir_version");
    assert_eq!(a.producer_name(), b.producer_name(), "producer_name");
    assert_eq!(
        a.producer_version(),
        b.producer_version(),
        "producer_version"
    );
    assert_eq!(a.domain(), b.domain(), "domain");
    assert_eq!(a.model_version(), b.model_version(), "model_version");
    assert_eq!(a.doc_string(), b.doc_string(), "doc_string");

    match (a.graph(), b.graph()) {
        (Some(ga), Some(gb)) => assert_graphs_equal(&ga, &gb),
        (None, None) => {}
        _ => panic!("graph presence mismatch"),
    }

    assert_opt_vec(
        a.opset_import(),
        b.opset_import(),
        |x, y| {
            assert_eq!(x.domain(), y.domain(), "opset domain");
            assert_eq!(x.version(), y.version(), "opset version");
        },
        "opset_import",
    );

    assert_opt_vec(
        a.metadata_props(),
        b.metadata_props(),
        |x, y| {
            assert_eq!(x.key(), y.key());
            assert_eq!(x.value(), y.value());
        },
        "metadata_props",
    );
}

fn assert_graphs_equal(a: &dl::Graph<'_>, b: &dl::Graph<'_>) {
    assert_eq!(a.name(), b.name(), "graph.name");
    assert_eq!(a.doc_string(), b.doc_string(), "graph.doc_string");

    assert_opt_vec(a.node(), b.node(), assert_nodes_equal, "graph.node");
    assert_opt_vec(
        a.initializer(),
        b.initializer(),
        assert_tensors_equal,
        "graph.initializer",
    );
    assert_opt_vec(
        a.input(),
        b.input(),
        assert_value_infos_equal,
        "graph.input",
    );
    assert_opt_vec(
        a.output(),
        b.output(),
        assert_value_infos_equal,
        "graph.output",
    );
    assert_opt_vec(
        a.value_info(),
        b.value_info(),
        assert_value_infos_equal,
        "graph.value_info",
    );
    assert_opt_vec(
        a.test_inputs_value(),
        b.test_inputs_value(),
        assert_tensors_equal,
        "graph.test_inputs_value",
    );
    assert_opt_vec(
        a.test_outputs_value(),
        b.test_outputs_value(),
        assert_tensors_equal,
        "graph.test_outputs_value",
    );
}

fn assert_nodes_equal(a: &dl::Node<'_>, b: &dl::Node<'_>) {
    assert_eq!(a.name(), b.name(), "node.name");
    assert_eq!(a.op_type(), b.op_type(), "node.op_type");
    assert_eq!(a.domain(), b.domain(), "node.domain");
    assert_str_vec(a.input(), b.input(), "node.input");
    assert_str_vec(a.output(), b.output(), "node.output");
    assert_opt_vec(
        a.attribute(),
        b.attribute(),
        assert_attributes_equal,
        "node.attribute",
    );
}

fn assert_attributes_equal(a: &dl::Attribute<'_>, b: &dl::Attribute<'_>) {
    assert_eq!(a.name(), b.name(), "attr.name");
    assert_eq!(a.attr_type(), b.attr_type(), "attr.attr_type");
    assert_eq!(a.f().map(|s| s.f()), b.f().map(|s| s.f()), "attr.f");
    assert_eq!(a.i().map(|s| s.i()), b.i().map(|s| s.i()), "attr.i");

    let s_a: Option<Vec<u8>> = a.s().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    let s_b: Option<Vec<u8>> = b.s().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    assert_eq!(s_a, s_b, "attr.s");

    match (a.t(), b.t()) {
        (Some(ta), Some(tb)) => assert_tensors_equal(&ta, &tb),
        (None, None) => {}
        _ => panic!("attr.t presence mismatch ({:?})", a.name()),
    }

    let ints_a: Option<Vec<i64>> = a.ints().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    let ints_b: Option<Vec<i64>> = b.ints().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    assert_eq!(ints_a, ints_b, "attr.ints");

    let floats_a: Option<Vec<f32>> = a.floats().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    let floats_b: Option<Vec<f32>> = b.floats().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    assert_eq!(floats_a, floats_b, "attr.floats");
}

fn assert_tensors_equal(a: &dl::Tensor<'_>, b: &dl::Tensor<'_>) {
    assert_eq!(a.name(), b.name(), "tensor.name");
    assert_eq!(a.doc_string(), b.doc_string(), "tensor.doc_string");
    assert_eq!(a.data_type(), b.data_type(), "tensor.data_type");
    assert_eq!(a.data_location(), b.data_location(), "tensor.data_location");

    let dims_a: Option<Vec<i64>> = a.dims().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    let dims_b: Option<Vec<i64>> = b.dims().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    assert_eq!(dims_a, dims_b, "tensor.dims");

    let exp_a: Option<Vec<i64>> = a
        .exponents()
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    let exp_b: Option<Vec<i64>> = b
        .exponents()
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    assert_eq!(exp_a, exp_b, "tensor.exponents");

    let raw_a = a.raw_data().map(flatten_aligned);
    let raw_b = b.raw_data().map(flatten_aligned);
    assert_eq!(raw_a, raw_b, "tensor.raw_data");

    macro_rules! cmp_typed {
        ($field:ident, $ty:ty) => {{
            let av: Option<Vec<$ty>> = a.$field().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
            let bv: Option<Vec<$ty>> = b.$field().map(|v| (0..v.len()).map(|i| v.get(i)).collect());
            assert_eq!(av, bv, concat!("tensor.", stringify!($field)));
        }};
    }
    cmp_typed!(float_data, f32);
    cmp_typed!(int32_data, i32);
    cmp_typed!(int64_data, i64);
    cmp_typed!(double_data, f64);
    cmp_typed!(uint64_data, u64);
}

fn assert_value_infos_equal(a: &dl::ValueInfo<'_>, b: &dl::ValueInfo<'_>) {
    assert_eq!(a.name(), b.name(), "value_info.name");
    let exp_a: Option<Vec<i64>> = a
        .exponents()
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    let exp_b: Option<Vec<i64>> = b
        .exponents()
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect());
    assert_eq!(exp_a, exp_b, "value_info.exponents");

    match (a.value_info_type(), b.value_info_type()) {
        (Some(ta), Some(tb)) => assert_eq!(
            ta.value_type(),
            tb.value_type(),
            "value_info.type discriminator"
        ),
        (None, None) => {}
        _ => panic!("value_info.type presence mismatch ({:?})", a.name()),
    }
}

fn flatten_aligned<'a>(v: flatbuffers::Vector<'a, dl::AlignedBytes>) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 16);
    for i in 0..v.len() {
        out.extend_from_slice(&v.get(i).0);
    }
    out
}

fn assert_str_vec(
    a: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<&str>>>,
    b: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<&str>>>,
    label: &str,
) {
    match (a, b) {
        (Some(av), Some(bv)) => {
            assert_eq!(av.len(), bv.len(), "{label} length");
            for i in 0..av.len() {
                assert_eq!(av.get(i), bv.get(i), "{label}[{i}]");
            }
        }
        (None, None) => {}
        _ => panic!("{label} presence mismatch"),
    }
}

fn assert_opt_vec<'a, T, F>(
    a: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<T>>>,
    b: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<T>>>,
    mut cmp: F,
    label: &str,
) where
    T: flatbuffers::Verifiable + flatbuffers::Follow<'a, Inner = T>,
    F: FnMut(&T, &T),
{
    match (a, b) {
        (Some(av), Some(bv)) => {
            assert_eq!(av.len(), bv.len(), "{label} length");
            for i in 0..av.len() {
                cmp(&av.get(i), &bv.get(i));
            }
        }
        (None, None) => {}
        _ => panic!("{label} presence mismatch"),
    }
}
