//! Step-1 gating test.
//!
//! Reads the production golden `.espdl` (produced by esp-ppq from the
//! TinyConv checkpoint), parses every table the writer must populate
//! using the copied `Dl_generated.rs` bindings, then re-serializes the
//! whole thing back into a fresh FlatBuffers buffer using only the
//! public builder API. The test asserts **structural equality** across
//! every field the device-side loader consumes — byte-equality is not
//! required since FlatBuffers ordering is implementation-defined.
//!
//! If this test fails, the pre-existing `Dl.fbs` schema is missing
//! something the S3 INT8 path needs and Step 1 is not actually green.
//!
//! The test is skipped (with a printed warning) if the golden artifact
//! is not present at the expected path. CI is expected to run with the
//! artifact mounted at `/tmp/nn-rs-robust-cuda/export/model.espdl`.

use std::path::Path;

use burn_espdl_export::{EspdlContainer, dl};
use flatbuffers::{FlatBufferBuilder, WIPOffset};

const GOLDEN_PATH: &str = "/tmp/nn-rs-robust-cuda/export/model.espdl";

#[test]
fn golden_espdl_roundtrips_through_dl_generated() {
    let bytes = match std::fs::read(GOLDEN_PATH) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skipping: golden artifact not available at {GOLDEN_PATH} ({e}). \
                 Re-run the existing burn_to_espdl pipeline to produce it."
            );
            return;
        }
    };

    let container = EspdlContainer::parse(&bytes).expect("EDL2 header parses");
    let original = flatbuffers::root::<dl::Model>(container.payload).expect("model parses");

    // Re-serialize through the public builder API.
    let mut builder = FlatBufferBuilder::with_capacity(container.payload.len() * 2);
    let new_root = clone_model(&mut builder, &original);
    builder.finish(new_root, None);
    let reserialized = builder.finished_data();

    let cloned = flatbuffers::root::<dl::Model>(reserialized).expect("clone parses");

    assert_models_structurally_equal(&original, &cloned);

    // And the EDL2 container framing round-trips cleanly too.
    let repacked = EspdlContainer::pack(reserialized);
    let reparsed = EspdlContainer::parse(&repacked).expect("repacked parses");
    assert_eq!(reparsed.payload, reserialized);
}

// ---------------------------------------------------------------------
// Cloners — walk the read API and rebuild via the writer API. These are
// intentionally exhaustive so any new schema field that lands later
// would force a compile error here.
// ---------------------------------------------------------------------

fn clone_model<'a>(b: &mut FlatBufferBuilder<'a>, m: &dl::Model<'_>) -> WIPOffset<dl::Model<'a>> {
    let producer_name = m.producer_name().map(|s| b.create_string(s));
    let producer_version = m.producer_version().map(|s| b.create_string(s));
    let domain = m.domain().map(|s| b.create_string(s));
    let doc_string = m.doc_string().map(|s| b.create_string(s));

    let opset_import = m.opset_import().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_opset_id(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });

    let metadata_props = m.metadata_props().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_kv(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });

    let functions = m.functions().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_function(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });

    let graph = m.graph().map(|g| clone_graph(b, &g));

    let args = dl::ModelArgs {
        ir_version: m.ir_version(),
        opset_import,
        producer_name,
        producer_version,
        domain,
        model_version: m.model_version(),
        doc_string,
        graph,
        metadata_props,
        functions,
    };
    dl::Model::create(b, &args)
}

fn clone_graph<'a>(b: &mut FlatBufferBuilder<'a>, g: &dl::Graph<'_>) -> WIPOffset<dl::Graph<'a>> {
    let name = g.name().map(|s| b.create_string(s));
    let doc_string = g.doc_string().map(|s| b.create_string(s));

    let node = g.node().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_node(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });
    let initializer = g.initializer().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_tensor(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });
    let input = g.input().map(|v| {
        let cloned: Vec<_> = (0..v.len())
            .map(|i| clone_value_info(b, &v.get(i)))
            .collect();
        b.create_vector(&cloned)
    });
    let output = g.output().map(|v| {
        let cloned: Vec<_> = (0..v.len())
            .map(|i| clone_value_info(b, &v.get(i)))
            .collect();
        b.create_vector(&cloned)
    });
    let value_info = g.value_info().map(|v| {
        let cloned: Vec<_> = (0..v.len())
            .map(|i| clone_value_info(b, &v.get(i)))
            .collect();
        b.create_vector(&cloned)
    });
    let quantization_annotation = g.quantization_annotation().map(|v| {
        let cloned: Vec<_> = (0..v.len())
            .map(|i| clone_tensor_annotation(b, &v.get(i)))
            .collect();
        b.create_vector(&cloned)
    });
    let test_inputs_value = g.test_inputs_value().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_tensor(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });
    let test_outputs_value = g.test_outputs_value().map(|v| {
        let cloned: Vec<_> = (0..v.len()).map(|i| clone_tensor(b, &v.get(i))).collect();
        b.create_vector(&cloned)
    });

    let args = dl::GraphArgs {
        node,
        name,
        initializer,
        doc_string,
        input,
        output,
        value_info,
        quantization_annotation,
        test_inputs_value,
        test_outputs_value,
    };
    dl::Graph::create(b, &args)
}

fn clone_node<'a>(b: &mut FlatBufferBuilder<'a>, n: &dl::Node<'_>) -> WIPOffset<dl::Node<'a>> {
    let name = n.name().map(|s| b.create_string(s));
    let op_type = n.op_type().map(|s| b.create_string(s));
    let domain = n.domain().map(|s| b.create_string(s));
    let doc_string = n.doc_string().map(|s| b.create_string(s));
    let input = n.input().map(|v| {
        let strs: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&strs)
    });
    let output = n.output().map(|v| {
        let strs: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&strs)
    });
    let attribute = n.attribute().map(|v| {
        let attrs: Vec<_> = (0..v.len())
            .map(|i| clone_attribute(b, &v.get(i)))
            .collect();
        b.create_vector(&attrs)
    });

    let args = dl::NodeArgs {
        input,
        output,
        name,
        op_type,
        domain,
        attribute,
        doc_string,
    };
    dl::Node::create(b, &args)
}

fn clone_tensor<'a>(
    b: &mut FlatBufferBuilder<'a>,
    t: &dl::Tensor<'_>,
) -> WIPOffset<dl::Tensor<'a>> {
    let name = t.name().map(|s| b.create_string(s));
    let doc_string = t.doc_string().map(|s| b.create_string(s));
    let dims = t.dims().map(|v| {
        let owned: Vec<i64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let exponents = t.exponents().map(|v| {
        let owned: Vec<i64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let float_data = t.float_data().map(|v| {
        let owned: Vec<f32> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let int32_data = t.int32_data().map(|v| {
        let owned: Vec<i32> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let int64_data = t.int64_data().map(|v| {
        let owned: Vec<i64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let double_data = t.double_data().map(|v| {
        let owned: Vec<f64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let uint64_data = t.uint64_data().map(|v| {
        let owned: Vec<u64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let string_data = t.string_data().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&owned)
    });
    let raw_data = t.raw_data().map(|v| {
        // Each element is the 16-byte AlignedBytes struct; copy verbatim.
        let owned: Vec<dl::AlignedBytes> = (0..v.len()).map(|i| *v.get(i)).collect();
        b.create_vector(&owned)
    });
    let external_data = t.external_data().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| clone_kv(b, &v.get(i))).collect();
        b.create_vector(&owned)
    });

    let args = dl::TensorArgs {
        dims,
        data_type: t.data_type(),
        float_data,
        int32_data,
        string_data,
        int64_data,
        name,
        doc_string,
        raw_data,
        external_data,
        data_location: t.data_location(),
        double_data,
        uint64_data,
        exponents,
    };
    dl::Tensor::create(b, &args)
}

fn clone_attribute<'a>(
    b: &mut FlatBufferBuilder<'a>,
    a: &dl::Attribute<'_>,
) -> WIPOffset<dl::Attribute<'a>> {
    let name = a.name().map(|s| b.create_string(s));
    let ref_attr_name = a.ref_attr_name().map(|s| b.create_string(s));
    let doc_string = a.doc_string().map(|s| b.create_string(s));

    // The schema represents single-int / single-float as inline structs.
    // The bindings copy them by value — collect them first, then move the
    // copies into the writer.
    let f_val = a.f().copied();
    let i_val = a.i().copied();

    let s = a.s().map(|v| {
        let owned: Vec<u8> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let t = a.t().map(|t| clone_tensor(b, &t));
    let g = a.g().map(|g| clone_graph(b, &g));
    let tp = a.tp().map(|t| clone_type_info(b, &t));
    let floats = a.floats().map(|v| {
        let owned: Vec<f32> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let ints = a.ints().map(|v| {
        let owned: Vec<i64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let strings = a.strings().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&owned)
    });
    let tensors = a.tensors().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| clone_tensor(b, &v.get(i))).collect();
        b.create_vector(&owned)
    });
    let graphs = a.graphs().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| clone_graph(b, &v.get(i))).collect();
        b.create_vector(&owned)
    });
    let type_protos = a.type_protos().map(|v| {
        let owned: Vec<_> = (0..v.len())
            .map(|i| clone_type_info(b, &v.get(i)))
            .collect();
        b.create_vector(&owned)
    });

    let args = dl::AttributeArgs {
        name,
        ref_attr_name,
        doc_string,
        attr_type: a.attr_type(),
        f: f_val.as_ref(),
        i: i_val.as_ref(),
        s,
        t,
        g,
        tp,
        floats,
        ints,
        strings,
        tensors,
        graphs,
        type_protos,
    };
    dl::Attribute::create(b, &args)
}

fn clone_value_info<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &dl::ValueInfo<'_>,
) -> WIPOffset<dl::ValueInfo<'a>> {
    let name = v.name().map(|s| b.create_string(s));
    let doc_string = v.doc_string().map(|s| b.create_string(s));
    let value_info_type = v.value_info_type().map(|t| clone_type_info(b, &t));
    let exponents = v.exponents().map(|v| {
        let owned: Vec<i64> = (0..v.len()).map(|i| v.get(i)).collect();
        b.create_vector(&owned)
    });
    let args = dl::ValueInfoArgs {
        name,
        value_info_type,
        doc_string,
        exponents,
    };
    dl::ValueInfo::create(b, &args)
}

fn clone_type_info<'a>(
    b: &mut FlatBufferBuilder<'a>,
    t: &dl::TypeInfo<'_>,
) -> WIPOffset<dl::TypeInfo<'a>> {
    let denotation = t.denotation().map(|s| b.create_string(s));

    let (value_type, value_offset) = match t.value_type() {
        dl::TypeInfoValue::tensor_type => {
            let inner = t.value_as_tensor_type().expect("union discriminator");
            let cloned = clone_tensor_type_and_shape(b, &inner);
            (
                dl::TypeInfoValue::tensor_type,
                Some(cloned.as_union_value()),
            )
        }
        dl::TypeInfoValue::sequence_type => {
            let inner = t.value_as_sequence_type().expect("union discriminator");
            let elem = inner.elem_type().map(|e| clone_type_info(b, &e));
            let cloned = dl::SequenceType::create(b, &dl::SequenceTypeArgs { elem_type: elem });
            (
                dl::TypeInfoValue::sequence_type,
                Some(cloned.as_union_value()),
            )
        }
        dl::TypeInfoValue::map_type => {
            let inner = t.value_as_map_type().expect("union discriminator");
            let value_type = inner.value_type().map(|e| clone_type_info(b, &e));
            let cloned = dl::MapType::create(
                b,
                &dl::MapTypeArgs {
                    key_type: inner.key_type(),
                    value_type,
                },
            );
            (dl::TypeInfoValue::map_type, Some(cloned.as_union_value()))
        }
        dl::TypeInfoValue::optional_type => {
            let inner = t.value_as_optional_type().expect("union discriminator");
            let elem = inner.elem_type().map(|e| clone_type_info(b, &e));
            let cloned = dl::OptionalType::create(b, &dl::OptionalTypeArgs { elem_type: elem });
            (
                dl::TypeInfoValue::optional_type,
                Some(cloned.as_union_value()),
            )
        }
        other => (other, None),
    };

    let args = dl::TypeInfoArgs {
        value_type,
        value: value_offset,
        denotation,
    };
    dl::TypeInfo::create(b, &args)
}

fn clone_tensor_type_and_shape<'a>(
    b: &mut FlatBufferBuilder<'a>,
    t: &dl::TensorTypeAndShape<'_>,
) -> WIPOffset<dl::TensorTypeAndShape<'a>> {
    let shape = t.shape().map(|s| {
        let dims = s.dim().map(|v| {
            let cloned: Vec<_> = (0..v.len())
                .map(|i| clone_dimension(b, &v.get(i)))
                .collect();
            b.create_vector(&cloned)
        });
        dl::TensorShape::create(b, &dl::TensorShapeArgs { dim: dims })
    });
    let args = dl::TensorTypeAndShapeArgs {
        elem_type: t.elem_type(),
        shape,
    };
    dl::TensorTypeAndShape::create(b, &args)
}

fn clone_dimension<'a>(
    b: &mut FlatBufferBuilder<'a>,
    d: &dl::Dimension<'_>,
) -> WIPOffset<dl::Dimension<'a>> {
    let denotation = d.denotation().map(|s| b.create_string(s));
    let value = d.value().map(|v| {
        let dim_param = v.dim_param().map(|s| b.create_string(s));
        dl::DimensionValue::create(
            b,
            &dl::DimensionValueArgs {
                dim_type: v.dim_type(),
                dim_value: v.dim_value(),
                dim_param,
            },
        )
    });
    dl::Dimension::create(b, &dl::DimensionArgs { value, denotation })
}

fn clone_opset_id<'a>(
    b: &mut FlatBufferBuilder<'a>,
    o: &dl::OperatorSetId<'_>,
) -> WIPOffset<dl::OperatorSetId<'a>> {
    let domain = o.domain().map(|s| b.create_string(s));
    dl::OperatorSetId::create(
        b,
        &dl::OperatorSetIdArgs {
            domain,
            version: o.version(),
        },
    )
}

fn clone_kv<'a>(
    b: &mut FlatBufferBuilder<'a>,
    e: &dl::StringStringEntry<'_>,
) -> WIPOffset<dl::StringStringEntry<'a>> {
    let key = e.key().map(|s| b.create_string(s));
    let value = e.value().map(|s| b.create_string(s));
    dl::StringStringEntry::create(b, &dl::StringStringEntryArgs { key, value })
}

fn clone_tensor_annotation<'a>(
    b: &mut FlatBufferBuilder<'a>,
    a: &dl::TensorAnnotation<'_>,
) -> WIPOffset<dl::TensorAnnotation<'a>> {
    let tensor_name = a.tensor_name().map(|s| b.create_string(s));
    let kvs = a.quant_parameter_tensor_names().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| clone_kv(b, &v.get(i))).collect();
        b.create_vector(&owned)
    });
    dl::TensorAnnotation::create(
        b,
        &dl::TensorAnnotationArgs {
            tensor_name,
            quant_parameter_tensor_names: kvs,
        },
    )
}

fn clone_function<'a>(
    b: &mut FlatBufferBuilder<'a>,
    f: &dl::Function<'_>,
) -> WIPOffset<dl::Function<'a>> {
    let name = f.name().map(|s| b.create_string(s));
    let doc_string = f.doc_string().map(|s| b.create_string(s));
    let domain = f.domain().map(|s| b.create_string(s));
    let input = f.input().map(|v| {
        let strs: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&strs)
    });
    let output = f.output().map(|v| {
        let strs: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&strs)
    });
    let attribute = f.attribute().map(|v| {
        let strs: Vec<_> = (0..v.len()).map(|i| b.create_string(v.get(i))).collect();
        b.create_vector(&strs)
    });
    let attribute_proto = f.attribute_proto().map(|v| {
        let owned: Vec<_> = (0..v.len())
            .map(|i| clone_attribute(b, &v.get(i)))
            .collect();
        b.create_vector(&owned)
    });
    let node = f.node().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| clone_node(b, &v.get(i))).collect();
        b.create_vector(&owned)
    });
    let opset_import = f.opset_import().map(|v| {
        let owned: Vec<_> = (0..v.len()).map(|i| clone_opset_id(b, &v.get(i))).collect();
        b.create_vector(&owned)
    });
    dl::Function::create(
        b,
        &dl::FunctionArgs {
            name,
            input,
            output,
            attribute,
            attribute_proto,
            node,
            doc_string,
            opset_import,
            domain,
        },
    )
}

// ---------------------------------------------------------------------
// Structural-equality assertions. These intentionally compare the
// fields the device-side loader (edgedl) reads — anything more strict
// would force byte-equality that FlatBuffers does not promise.
// ---------------------------------------------------------------------

fn assert_models_structurally_equal(a: &dl::Model<'_>, b: &dl::Model<'_>) {
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
    assert_eq!(a.f().map(|s| s.f()), b.f().map(|s| s.f()), "attr.f",);
    assert_eq!(a.i().map(|s| s.i()), b.i().map(|s| s.i()), "attr.i",);
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

#[allow(dead_code)]
fn _golden_path_exists() -> bool {
    Path::new(GOLDEN_PATH).exists()
}
