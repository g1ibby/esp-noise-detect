//! Step 5 graph exporter.
//!
//! Consumes the post-fold/post-fuse [`crate::ir::BurnGraph`] and a
//! Step 4 [`crate::ScaleTable`], then writes a complete ESP-DL
//! FlatBuffers model wrapped in the existing `EDL2` container.

use std::collections::BTreeMap;

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::dl;
use crate::ir::{BurnGraph, Layer, Tensor};
use crate::layout::{self, PackedTensor, annotation};
use crate::quant;
use crate::writer;
use crate::{QuantConfig, ScaleTable, TensorRole};

/// Step 5 export configuration.
#[derive(Debug, Clone, Copy)]
pub struct ExportConfig {
    pub num_bits: u8,
}

impl ExportConfig {
    pub fn esp32s3_int8() -> Self {
        Self { num_bits: 8 }
    }

    pub fn esp32s3_int16() -> Self {
        Self { num_bits: 16 }
    }

    fn weight_data_type(self) -> dl::TensorDataType {
        match self.num_bits {
            8 => dl::TensorDataType::INT8,
            16 => dl::TensorDataType::INT16,
            other => panic!("unsupported ESP-DL export bit width {other}"),
        }
    }

    fn activation_data_type(self) -> dl::TensorDataType {
        self.weight_data_type()
    }

    fn quant_type(self) -> &'static str {
        match self.num_bits {
            8 => "S8",
            16 => "S16",
            other => panic!("unsupported ESP-DL export bit width {other}"),
        }
    }
}

/// Failure modes from [`write_graph`].
#[derive(Debug, Clone)]
pub enum ExportError {
    UnsupportedPreStep5Layer(&'static str),
    MissingScale { name: String, role: TensorRole },
    Shape(String),
}

impl core::fmt::Display for ExportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedPreStep5Layer(op) => {
                write!(f, "export: graph still contains pre-Step-5 layer {op}")
            }
            Self::MissingScale { name, role } => {
                write!(f, "export: missing {role:?} quant config for {name}")
            }
            Self::Shape(msg) => write!(f, "export: shape error: {msg}"),
        }
    }
}

impl std::error::Error for ExportError {}

#[derive(Debug, Clone, Copy)]
enum ActShape {
    Rank4Nchw([usize; 4]),
    Rank2([usize; 2]),
}

impl ActShape {
    fn exported_shape(self) -> Vec<usize> {
        match self {
            Self::Rank4Nchw(s) => layout::nchw_to_nhwc(s).to_vec(),
            Self::Rank2(s) => s.to_vec(),
        }
    }
}

struct ExportState {
    shapes: BTreeMap<String, ActShape>,
}

/// Write a full `.espdl` file for `graph`.
pub fn write_graph(
    graph: &BurnGraph,
    scales: &ScaleTable,
    cfg: ExportConfig,
) -> Result<Vec<u8>, ExportError> {
    let mut state = ExportState {
        shapes: BTreeMap::new(),
    };
    state.shapes.insert(
        graph.input_name.clone(),
        ActShape::Rank4Nchw(graph.input_shape),
    );

    let mut builder = FlatBufferBuilder::with_capacity(16 * 1024);
    let mut nodes = Vec::with_capacity(graph.layers.len());
    let mut initializers = Vec::new();
    let mut conv_value_infos = Vec::new();
    let mut reduce_mean_value_infos = Vec::new();
    let mut linear_value_infos = Vec::new();
    let mut shape_initializer_idx = 0_usize;

    for layer in &graph.layers {
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
                let input_shape = rank4(&state, input, "Conv")?;
                let output_shape =
                    conv_output_shape(input_shape, weight, *stride, *padding, *dilation)?;
                state
                    .shapes
                    .insert(output.clone(), ActShape::Rank4Nchw(output_shape));

                let weight_name = format!("{output}.weight");
                let weight_cfg = scale(scales, &weight_name, TensorRole::Weight)?;
                let packed_weight = match cfg.num_bits {
                    8 => PackedBytes::I8(layout::pack_quantized_conv(
                        quant::quantize_i8(&weight.data, weight_cfg),
                        weight,
                        cfg.num_bits,
                    )),
                    16 => PackedBytes::I16(layout::pack_quantized_conv(
                        quant::quantize_i16(&weight.data, weight_cfg),
                        weight,
                        cfg.num_bits,
                    )),
                    _ => unreachable!("validated by ExportConfig"),
                };
                initializers.push(make_packed_initializer(
                    &mut builder,
                    &weight_name,
                    cfg.weight_data_type(),
                    packed_weight,
                    weight_cfg.exponent,
                ));

                let mut node_inputs = vec![input.clone(), weight_name];
                if let Some(bias) = bias {
                    let bias_name = format!("{output}.bias");
                    let bias_cfg = scale(scales, &bias_name, TensorRole::Bias)?;
                    let bias_values = quant::quantize_i32(&bias.data, bias_cfg);
                    initializers.push(make_i32_initializer(
                        &mut builder,
                        &bias_name,
                        &[bias_values.len()],
                        &bias_values,
                        bias_cfg.exponent,
                        annotation::UNKNOWN,
                    ));
                    node_inputs.push(bias_name);
                }

                let mut attrs = vec![
                    attr_ints(
                        &mut builder,
                        "strides",
                        &[stride[0] as i64, stride[1] as i64],
                    ),
                    attr_ints(
                        &mut builder,
                        "pads",
                        &[
                            padding[0] as i64,
                            padding[1] as i64,
                            padding[2] as i64,
                            padding[3] as i64,
                        ],
                    ),
                    attr_ints(
                        &mut builder,
                        "dilations",
                        &[dilation[0] as i64, dilation[1] as i64],
                    ),
                    attr_int(&mut builder, "group", *groups as i64),
                    attr_string(&mut builder, "quant_type", cfg.quant_type()),
                ];
                if let Some(act) = activation {
                    attrs.push(attr_string(&mut builder, "activation", &act.to_string()));
                }

                let node_name = format!("{output}.conv");
                nodes.push(make_node(
                    &mut builder,
                    "Conv",
                    &node_inputs,
                    core::slice::from_ref(output),
                    &node_name,
                    &attrs,
                ));
                conv_value_infos.push(make_activation_value_info(
                    &mut builder,
                    output,
                    state.shapes[output],
                    cfg.activation_data_type(),
                    scale(scales, output, TensorRole::Activation)?.exponent,
                ));
            }
            Layer::ReduceMean {
                input,
                output,
                axes,
                keepdims,
            } => {
                let input_shape = rank4(&state, input, "ReduceMean")?;
                let axes_nhwc = layout::reduce_axes_nchw_to_nhwc(axes, 4);
                let output_shape = if *keepdims {
                    let mut s = layout::nchw_to_nhwc(input_shape);
                    for &axis in &axes_nhwc {
                        s[axis as usize] = 1;
                    }
                    ActShape::Rank4Nchw([s[0], s[3], s[1], s[2]])
                } else {
                    ActShape::Rank2([input_shape[0], input_shape[1]])
                };
                state.shapes.insert(output.clone(), output_shape);

                let axes_name = format!("PPQ_Variable_{shape_initializer_idx}");
                shape_initializer_idx += 1;
                initializers.push(make_i64_initializer(
                    &mut builder,
                    &axes_name,
                    &[axes_nhwc.len()],
                    &axes_nhwc,
                    0,
                    annotation::UNKNOWN,
                ));
                let attrs = vec![
                    attr_int(&mut builder, "keepdims", i64::from(*keepdims)),
                    attr_string(&mut builder, "quant_type", cfg.quant_type()),
                ];
                let node_name = format!("{output}.reduce_mean");
                nodes.push(make_node(
                    &mut builder,
                    "ReduceMean",
                    &[input.clone(), axes_name],
                    core::slice::from_ref(output),
                    &node_name,
                    &attrs,
                ));
                reduce_mean_value_infos.push(make_activation_value_info(
                    &mut builder,
                    output,
                    output_shape,
                    cfg.activation_data_type(),
                    scale(scales, output, TensorRole::Activation)?.exponent,
                ));
            }
            Layer::Linear {
                input,
                output,
                weight,
                bias,
            } => {
                let input_shape = rank2(&state, input, "Gemm")?;
                let dout = *weight.shape.get(1).ok_or_else(|| {
                    ExportError::Shape(format!("Linear {output} weight is not [in,out]"))
                })?;
                state
                    .shapes
                    .insert(output.clone(), ActShape::Rank2([input_shape[0], dout]));

                let weight_name = format!("{output}.weight");
                let weight_cfg = scale(scales, &weight_name, TensorRole::Weight)?;
                let packed_weight = match cfg.num_bits {
                    8 => PackedBytes::I8(layout::pack_quantized_linear(
                        quant::quantize_i8(&weight.data, weight_cfg),
                        weight,
                        cfg.num_bits,
                    )),
                    16 => PackedBytes::I16(layout::pack_quantized_linear(
                        quant::quantize_i16(&weight.data, weight_cfg),
                        weight,
                        cfg.num_bits,
                    )),
                    _ => unreachable!("validated by ExportConfig"),
                };
                initializers.push(make_packed_initializer(
                    &mut builder,
                    &weight_name,
                    cfg.weight_data_type(),
                    packed_weight,
                    weight_cfg.exponent,
                ));

                let mut node_inputs = vec![input.clone(), weight_name];
                if let Some(bias) = bias {
                    let bias_name = format!("{output}.bias");
                    let bias_cfg = scale(scales, &bias_name, TensorRole::Bias)?;
                    let bias_values = quant::quantize_i32(&bias.data, bias_cfg);
                    initializers.push(make_i32_initializer(
                        &mut builder,
                        &bias_name,
                        &[bias_values.len()],
                        &bias_values,
                        bias_cfg.exponent,
                        annotation::UNKNOWN,
                    ));
                    node_inputs.push(bias_name);
                }
                let attrs = vec![
                    attr_string(&mut builder, "activation", "Linear"),
                    attr_float(&mut builder, "alpha", 1.0),
                    attr_float(&mut builder, "beta", 1.0),
                    attr_string(&mut builder, "quant_type", cfg.quant_type()),
                    attr_int(&mut builder, "transB", 0),
                ];
                let node_name = format!("{output}.gemm");
                nodes.push(make_node(
                    &mut builder,
                    "Gemm",
                    &node_inputs,
                    core::slice::from_ref(output),
                    &node_name,
                    &attrs,
                ));
                linear_value_infos.push(make_activation_value_info(
                    &mut builder,
                    output,
                    state.shapes[output],
                    cfg.activation_data_type(),
                    scale(scales, output, TensorRole::Activation)?.exponent,
                ));
            }
            Layer::BatchNorm2d { .. } => {
                return Err(ExportError::UnsupportedPreStep5Layer("BatchNormalization"));
            }
            Layer::Relu { .. } => return Err(ExportError::UnsupportedPreStep5Layer("Relu")),
        }
    }

    let input_shape = state.shapes[&graph.input_name];
    let output_shape = *state
        .shapes
        .get(&graph.output_name)
        .ok_or_else(|| ExportError::Shape(format!("missing output {}", graph.output_name)))?;
    let input_vi = make_activation_value_info(
        &mut builder,
        &graph.input_name,
        input_shape,
        cfg.activation_data_type(),
        scale(scales, &graph.input_name, TensorRole::Activation)?.exponent,
    );
    let output_vi = make_activation_value_info(
        &mut builder,
        &graph.output_name,
        output_shape,
        cfg.activation_data_type(),
        scale(scales, &graph.output_name, TensorRole::Activation)?.exponent,
    );

    let mut value_infos = Vec::with_capacity(
        linear_value_infos.len() + reduce_mean_value_infos.len() + conv_value_infos.len() + 1,
    );
    value_infos.extend(linear_value_infos);
    value_infos.extend(reduce_mean_value_infos);
    value_infos.extend(conv_value_infos);
    value_infos.push(input_vi);

    let node_vec = builder.create_vector(&nodes);
    let init_vec = builder.create_vector(&initializers);
    let input_vec = builder.create_vector(&[input_vi]);
    let output_vec = builder.create_vector(&[output_vi]);
    let value_info_vec = builder.create_vector(&value_infos);
    let graph_name = builder.create_string("burn-espdl-export");
    let graph_root = dl::Graph::create(
        &mut builder,
        &dl::GraphArgs {
            node: Some(node_vec),
            name: Some(graph_name),
            initializer: Some(init_vec),
            input: Some(input_vec),
            output: Some(output_vec),
            value_info: Some(value_info_vec),
            ..dl::GraphArgs::default()
        },
    );
    let producer = builder.create_string("burn-espdl-export");
    let model = dl::Model::create(
        &mut builder,
        &dl::ModelArgs {
            graph: Some(graph_root),
            producer_name: Some(producer),
            ..dl::ModelArgs::default()
        },
    );

    Ok(writer::finish_model(builder, model))
}

fn scale(scales: &ScaleTable, name: &str, role: TensorRole) -> Result<QuantConfig, ExportError> {
    let entry = scales.get(name).ok_or_else(|| ExportError::MissingScale {
        name: name.to_string(),
        role,
    })?;
    if entry.role != role {
        return Err(ExportError::MissingScale {
            name: name.to_string(),
            role,
        });
    }
    Ok(entry.config)
}

fn rank4(state: &ExportState, input: &str, op: &str) -> Result<[usize; 4], ExportError> {
    match state.shapes.get(input) {
        Some(ActShape::Rank4Nchw(shape)) => Ok(*shape),
        other => Err(ExportError::Shape(format!(
            "{op} expected rank-4 input {input}, got {other:?}"
        ))),
    }
}

fn rank2(state: &ExportState, input: &str, op: &str) -> Result<[usize; 2], ExportError> {
    match state.shapes.get(input) {
        Some(ActShape::Rank2(shape)) => Ok(*shape),
        other => Err(ExportError::Shape(format!(
            "{op} expected rank-2 input {input}, got {other:?}"
        ))),
    }
}

fn conv_output_shape(
    input: [usize; 4],
    weight: &Tensor,
    stride: [usize; 2],
    padding: [usize; 4],
    dilation: [usize; 2],
) -> Result<[usize; 4], ExportError> {
    if weight.shape.len() != 4 {
        return Err(ExportError::Shape(
            "Conv weight must be [N,C,H,W]".to_string(),
        ));
    }
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let oh = (input[2] + padding[0] + padding[2] - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
    let ow = (input[3] + padding[1] + padding[3] - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
    Ok([input[0], weight.shape[0], oh, ow])
}

enum PackedBytes {
    I8(PackedTensor<i8>),
    I16(PackedTensor<i16>),
}

fn make_packed_initializer<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    data_type: dl::TensorDataType,
    packed: PackedBytes,
    exponent: i32,
) -> WIPOffset<dl::Tensor<'a>> {
    match packed {
        PackedBytes::I8(p) => make_raw_initializer(
            b,
            name,
            data_type,
            &p.shape,
            &i8_to_bytes(&p.values),
            exponent,
            p.annotation,
        ),
        PackedBytes::I16(p) => make_raw_initializer(
            b,
            name,
            data_type,
            &p.shape,
            &i16_to_bytes(&p.values),
            exponent,
            p.annotation,
        ),
    }
}

fn make_i32_initializer<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    shape: &[usize],
    values: &[i32],
    exponent: i32,
    doc_string: &str,
) -> WIPOffset<dl::Tensor<'a>> {
    make_raw_initializer(
        b,
        name,
        dl::TensorDataType::INT32,
        shape,
        &i32_to_bytes(values),
        exponent,
        doc_string,
    )
}

fn make_i64_initializer<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    shape: &[usize],
    values: &[i64],
    exponent: i32,
    doc_string: &str,
) -> WIPOffset<dl::Tensor<'a>> {
    make_raw_initializer(
        b,
        name,
        dl::TensorDataType::INT64,
        shape,
        &i64_to_bytes(values),
        exponent,
        doc_string,
    )
}

fn make_raw_initializer<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    data_type: dl::TensorDataType,
    shape: &[usize],
    bytes: &[u8],
    exponent: i32,
    doc_string: &str,
) -> WIPOffset<dl::Tensor<'a>> {
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let chunks = aligned_chunks(bytes);
    let name = b.create_string(name);
    let doc_string = b.create_string(&format!("layout ==> {doc_string}"));
    let dims = b.create_vector(&dims);
    let raw_data = b.create_vector(&chunks);
    let exponents = b.create_vector(&[exponent as i64]);
    dl::Tensor::create(
        b,
        &dl::TensorArgs {
            dims: Some(dims),
            data_type,
            name: Some(name),
            doc_string: Some(doc_string),
            raw_data: Some(raw_data),
            exponents: Some(exponents),
            ..dl::TensorArgs::default()
        },
    )
}

fn make_activation_value_info<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    shape: ActShape,
    elem_type: dl::TensorDataType,
    exponent: i32,
) -> WIPOffset<dl::ValueInfo<'a>> {
    make_value_info(b, name, elem_type, &shape.exported_shape(), exponent)
}

fn make_value_info<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    elem_type: dl::TensorDataType,
    shape: &[usize],
    exponent: i32,
) -> WIPOffset<dl::ValueInfo<'a>> {
    let dims: Vec<_> = shape
        .iter()
        .map(|&dim| {
            let value = dl::DimensionValue::create(
                b,
                &dl::DimensionValueArgs {
                    dim_type: dl::DimensionValueType::VALUE,
                    dim_value: dim as i64,
                    dim_param: None,
                },
            );
            dl::Dimension::create(
                b,
                &dl::DimensionArgs {
                    value: Some(value),
                    denotation: None,
                },
            )
        })
        .collect();
    let dim_vec = b.create_vector(&dims);
    let shape = dl::TensorShape::create(b, &dl::TensorShapeArgs { dim: Some(dim_vec) });
    let tensor_type = dl::TensorTypeAndShape::create(
        b,
        &dl::TensorTypeAndShapeArgs {
            elem_type,
            shape: Some(shape),
        },
    );
    let type_info = dl::TypeInfo::create(
        b,
        &dl::TypeInfoArgs {
            value_type: dl::TypeInfoValue::tensor_type,
            value: Some(tensor_type.as_union_value()),
            denotation: None,
        },
    );
    let name = b.create_string(name);
    let exponents = b.create_vector(&[exponent as i64]);
    dl::ValueInfo::create(
        b,
        &dl::ValueInfoArgs {
            name: Some(name),
            value_info_type: Some(type_info),
            doc_string: None,
            exponents: Some(exponents),
        },
    )
}

fn make_node<'a>(
    b: &mut FlatBufferBuilder<'a>,
    op_type: &str,
    inputs: &[String],
    outputs: &[String],
    name: &str,
    attrs: &[WIPOffset<dl::Attribute<'a>>],
) -> WIPOffset<dl::Node<'a>> {
    let input_offsets: Vec<_> = inputs.iter().map(|s| b.create_string(s)).collect();
    let output_offsets: Vec<_> = outputs.iter().map(|s| b.create_string(s)).collect();
    let input = b.create_vector(&input_offsets);
    let output = b.create_vector(&output_offsets);
    let name = b.create_string(name);
    let op_type = b.create_string(op_type);
    let attribute = if attrs.is_empty() {
        None
    } else {
        Some(b.create_vector(attrs))
    };
    dl::Node::create(
        b,
        &dl::NodeArgs {
            input: Some(input),
            output: Some(output),
            name: Some(name),
            op_type: Some(op_type),
            attribute,
            ..dl::NodeArgs::default()
        },
    )
}

fn attr_int<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    value: i64,
) -> WIPOffset<dl::Attribute<'a>> {
    let name = b.create_string(name);
    let i = dl::AttributeI::new(value);
    dl::Attribute::create(
        b,
        &dl::AttributeArgs {
            name: Some(name),
            attr_type: dl::AttributeType::INT,
            i: Some(&i),
            ..dl::AttributeArgs::default()
        },
    )
}

fn attr_float<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    value: f32,
) -> WIPOffset<dl::Attribute<'a>> {
    let name = b.create_string(name);
    let f = dl::AttributeF::new(value);
    dl::Attribute::create(
        b,
        &dl::AttributeArgs {
            name: Some(name),
            attr_type: dl::AttributeType::FLOAT,
            f: Some(&f),
            ..dl::AttributeArgs::default()
        },
    )
}

fn attr_ints<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    values: &[i64],
) -> WIPOffset<dl::Attribute<'a>> {
    let name = b.create_string(name);
    let ints = b.create_vector(values);
    dl::Attribute::create(
        b,
        &dl::AttributeArgs {
            name: Some(name),
            attr_type: dl::AttributeType::INTS,
            ints: Some(ints),
            ..dl::AttributeArgs::default()
        },
    )
}

fn attr_string<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    value: &str,
) -> WIPOffset<dl::Attribute<'a>> {
    let name = b.create_string(name);
    let s = b.create_vector(value.as_bytes());
    dl::Attribute::create(
        b,
        &dl::AttributeArgs {
            name: Some(name),
            attr_type: dl::AttributeType::STRING,
            s: Some(s),
            ..dl::AttributeArgs::default()
        },
    )
}

fn aligned_chunks(bytes: &[u8]) -> Vec<dl::AlignedBytes> {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(16).max(1));
    for chunk in bytes.chunks(16) {
        let mut block = [0_u8; 16];
        block[..chunk.len()].copy_from_slice(chunk);
        out.push(dl::AlignedBytes::new(&block));
    }
    if out.is_empty() {
        out.push(dl::AlignedBytes::new(&[0_u8; 16]));
    }
    out
}

fn i8_to_bytes(values: &[i8]) -> Vec<u8> {
    values.iter().map(|&v| v as u8).collect()
}

fn i16_to_bytes(values: &[i16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_to_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_to_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_chunks_are_zero_padded() {
        let chunks = aligned_chunks(&[1, 2, 3]);
        assert_eq!(chunks.len(), 1);
        let bytes: Vec<u8> = chunks[0].bytes().iter().collect();
        assert_eq!(&bytes[0..4], &[1, 2, 3, 0]);
    }
}
