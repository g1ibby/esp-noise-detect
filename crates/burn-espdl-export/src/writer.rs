//! Minimal `.espdl` writer.
//!
//! At Step 2 of the porting plan the writer is *structural only*: given an
//! existing [`dl::Model`] reader (parsed from another `.espdl` file or
//! constructed via the FlatBuffers builder API) it re-serializes the model
//! through the public builder API and wraps the result in the 16-byte
//! `EDL2` container. There is no quantization, no graph rewriting, and no
//! op-level helpers yet — those land in Step 5.
//!
//! The clone helpers (`clone_model`, `clone_graph`, …) walk every field
//! the device-side loader consumes. They are exposed publicly so that
//! later steps can build new models by mutating a freshly cloned skeleton
//! before [`finish_model`] is called.

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::container::EspdlContainer;
use crate::dl_generated::dl;

/// Re-serialize a parsed model and wrap it in an `EDL2` container.
///
/// The returned bytes are a syntactically complete `.espdl` file: a 16-byte
/// `EDL2` header followed by the FlatBuffers payload. Re-encoding goes
/// through the public FlatBuffers builder API, so byte-equality with the
/// original is **not** guaranteed (FlatBuffers does not promise stable
/// field/vtable ordering) but structural equality is.
pub fn write_model(model: &dl::Model<'_>) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::with_capacity(1024);
    let root = clone_model(&mut builder, model);
    finish_model(builder, root)
}

/// Build an `.espdl` file containing an empty graph and nothing else.
///
/// Used as a smoke test for the container framing: the on-device loader is
/// not expected to run an empty graph, but the resulting bytes must still
/// satisfy the `EDL2` header layout and parse cleanly as a [`dl::Model`].
pub fn write_empty() -> Vec<u8> {
    let mut builder = FlatBufferBuilder::with_capacity(64);
    let graph = dl::Graph::create(&mut builder, &dl::GraphArgs::default());
    let root = dl::Model::create(
        &mut builder,
        &dl::ModelArgs {
            graph: Some(graph),
            ..dl::ModelArgs::default()
        },
    );
    finish_model(builder, root)
}

/// Finish a FlatBuffers builder rooted at `Model` and wrap the resulting
/// payload in an `EDL2` container. Callers that build a [`dl::Model`]
/// from scratch (rather than cloning one) drop in here.
pub fn finish_model<'a>(
    mut builder: FlatBufferBuilder<'a>,
    root: WIPOffset<dl::Model<'a>>,
) -> Vec<u8> {
    builder.finish(root, None);
    EspdlContainer::pack(builder.finished_data())
}

// ---------------------------------------------------------------------
// Clone helpers — walk a `dl::Model` reader and emit it back through the
// builder. Every clone here is exhaustive over the schema fields the
// device-side loader consumes; adding a new field without extending the
// matching cloner would be caught by the round-trip test.
// ---------------------------------------------------------------------

/// Clone a [`dl::Model`] through the public builder API.
pub fn clone_model<'a>(
    b: &mut FlatBufferBuilder<'a>,
    m: &dl::Model<'_>,
) -> WIPOffset<dl::Model<'a>> {
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

    dl::Model::create(
        b,
        &dl::ModelArgs {
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
        },
    )
}

/// Clone a [`dl::Graph`] through the public builder API.
pub fn clone_graph<'a>(
    b: &mut FlatBufferBuilder<'a>,
    g: &dl::Graph<'_>,
) -> WIPOffset<dl::Graph<'a>> {
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

    dl::Graph::create(
        b,
        &dl::GraphArgs {
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
        },
    )
}

/// Clone a [`dl::Node`] through the public builder API.
pub fn clone_node<'a>(b: &mut FlatBufferBuilder<'a>, n: &dl::Node<'_>) -> WIPOffset<dl::Node<'a>> {
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

    dl::Node::create(
        b,
        &dl::NodeArgs {
            input,
            output,
            name,
            op_type,
            domain,
            attribute,
            doc_string,
        },
    )
}

/// Clone a [`dl::Tensor`] through the public builder API.
pub fn clone_tensor<'a>(
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

    dl::Tensor::create(
        b,
        &dl::TensorArgs {
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
        },
    )
}

/// Clone a [`dl::Attribute`] through the public builder API.
pub fn clone_attribute<'a>(
    b: &mut FlatBufferBuilder<'a>,
    a: &dl::Attribute<'_>,
) -> WIPOffset<dl::Attribute<'a>> {
    let name = a.name().map(|s| b.create_string(s));
    let ref_attr_name = a.ref_attr_name().map(|s| b.create_string(s));
    let doc_string = a.doc_string().map(|s| b.create_string(s));

    // f / i are inline structs returned by-reference from the reader.
    // Copy their values out so the borrow on the source ends before we
    // hand the writer any &mut FlatBufferBuilder.
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

    dl::Attribute::create(
        b,
        &dl::AttributeArgs {
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
        },
    )
}

/// Clone a [`dl::ValueInfo`] through the public builder API.
pub fn clone_value_info<'a>(
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
    dl::ValueInfo::create(
        b,
        &dl::ValueInfoArgs {
            name,
            value_info_type,
            doc_string,
            exponents,
        },
    )
}

/// Clone a [`dl::TypeInfo`] through the public builder API. Walks the
/// `TypeInfoValue` union and rebuilds whichever variant is set.
pub fn clone_type_info<'a>(
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

    dl::TypeInfo::create(
        b,
        &dl::TypeInfoArgs {
            value_type,
            value: value_offset,
            denotation,
        },
    )
}

/// Clone a [`dl::TensorTypeAndShape`] through the public builder API.
pub fn clone_tensor_type_and_shape<'a>(
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
    dl::TensorTypeAndShape::create(
        b,
        &dl::TensorTypeAndShapeArgs {
            elem_type: t.elem_type(),
            shape,
        },
    )
}

/// Clone a [`dl::Dimension`] through the public builder API.
pub fn clone_dimension<'a>(
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

/// Clone a [`dl::OperatorSetId`] through the public builder API.
pub fn clone_opset_id<'a>(
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

/// Clone a [`dl::StringStringEntry`] through the public builder API.
pub fn clone_kv<'a>(
    b: &mut FlatBufferBuilder<'a>,
    e: &dl::StringStringEntry<'_>,
) -> WIPOffset<dl::StringStringEntry<'a>> {
    let key = e.key().map(|s| b.create_string(s));
    let value = e.value().map(|s| b.create_string(s));
    dl::StringStringEntry::create(b, &dl::StringStringEntryArgs { key, value })
}

/// Clone a [`dl::TensorAnnotation`] through the public builder API.
pub fn clone_tensor_annotation<'a>(
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

/// Clone a [`dl::Function`] through the public builder API.
pub fn clone_function<'a>(
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
