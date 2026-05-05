//! High-level ESP-DL export orchestration.
//!
//! Callers own model-specific lowering and calibration data collection.
//! This module only coordinates the generic exporter steps over an
//! existing [`BurnGraph`]: graph rewrites, calibration, FlatBuffers
//! writing, and sidecar text artifacts.

use std::path::Path;

use burn::tensor::backend::Backend;

use crate::{
    BurnGraph, CalibrationConfig, CalibrationError, ExportConfig, ExportError, ScaleTable,
    TensorRole, calibrate, fold_batchnorm, fuse_relu, write_graph,
};

/// Reusable ESP-DL exporter.
///
/// The ergonomic public constructor intentionally exposes only the
/// verified ESP32-S3 INT8 path. The lower-level config types still have
/// experimental INT16 plumbing, but INT16 is not advertised here until
/// it has fixture-backed export parity.
#[derive(Debug, Clone)]
pub struct EspdlExporter {
    options: ExportOptions,
}

impl EspdlExporter {
    /// ESP32-S3 INT8 exporter with BN folding and ReLU fusion enabled.
    pub fn esp32s3_int8() -> Self {
        Self {
            options: ExportOptions::esp32s3_int8(),
        }
    }

    /// Override exporter options.
    pub fn with_options(mut self, options: ExportOptions) -> Self {
        self.options = options;
        self
    }

    /// Export `graph` using caller-provided flat calibration windows.
    ///
    /// `graph` is cloned before graph rewrites, so the caller-owned IR is
    /// not mutated.
    pub fn export_graph<B: Backend>(
        &self,
        graph: &BurnGraph,
        windows: &[Vec<f32>],
        device: &B::Device,
    ) -> Result<ExportArtifacts, EspdlExportError> {
        let mut graph = graph.clone();
        if self.options.fold_batchnorm {
            fold_batchnorm(&mut graph);
        }
        if self.options.fuse_relu {
            fuse_relu(&mut graph);
        }

        let scales = calibrate::<B>(&graph, windows, self.options.calibration, device)?;
        let model_bytes = write_graph(&graph, &scales, self.options.export)?;
        Ok(ExportArtifacts {
            model_bytes,
            quant_json: render_quant_json(&scales),
            model_info: render_model_info(&graph),
        })
    }
}

/// Options for the high-level exporter.
#[derive(Debug, Clone, Copy)]
pub struct ExportOptions {
    pub(crate) calibration: CalibrationConfig,
    pub(crate) export: ExportConfig,
    pub fold_batchnorm: bool,
    pub fuse_relu: bool,
}

impl ExportOptions {
    /// Verified ESP32-S3 INT8 defaults.
    pub fn esp32s3_int8() -> Self {
        Self {
            calibration: CalibrationConfig::esp32s3_int8(),
            export: ExportConfig::esp32s3_int8(),
            fold_batchnorm: true,
            fuse_relu: true,
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self::esp32s3_int8()
    }
}

/// Files produced by an ESP-DL export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportArtifacts {
    pub model_bytes: Vec<u8>,
    pub quant_json: String,
    pub model_info: String,
}

impl ExportArtifacts {
    /// Write `model.espdl`, `model.json`, and `model.info` into `dir`.
    pub fn write_to_dir(&self, dir: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_to_model_path(dir.as_ref().join("model.espdl"))
    }

    /// Write the ESP-DL model at an explicit path and sidecars next to it.
    ///
    /// Sidecars use the same stem with `.json` and `.info` extensions.
    pub fn write_to_model_path(&self, model_path: impl AsRef<Path>) -> std::io::Result<()> {
        let model_path = model_path.as_ref();
        if let Some(parent) = model_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(model_path, &self.model_bytes)?;
        std::fs::write(model_path.with_extension("json"), &self.quant_json)?;
        std::fs::write(model_path.with_extension("info"), &self.model_info)?;
        Ok(())
    }
}

/// High-level export failure.
#[derive(Debug)]
pub enum EspdlExportError {
    Calibration(CalibrationError),
    Export(ExportError),
}

impl core::fmt::Display for EspdlExportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Calibration(e) => write!(f, "{e}"),
            Self::Export(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EspdlExportError {}

impl From<CalibrationError> for EspdlExportError {
    fn from(value: CalibrationError) -> Self {
        Self::Calibration(value)
    }
}

impl From<ExportError> for EspdlExportError {
    fn from(value: ExportError) -> Self {
        Self::Export(value)
    }
}

/// Render a deterministic quantization summary next to the exported model.
pub fn render_quant_json(scales: &ScaleTable) -> String {
    let mut out = String::from("{\n");
    for (idx, (name, entry)) in scales.iter().enumerate() {
        let comma = if idx + 1 == scales.len() { "" } else { "," };
        let role = match entry.role {
            TensorRole::Activation => "activation",
            TensorRole::Weight => "weight",
            TensorRole::Bias => "bias",
        };
        out.push_str(&format!(
            "  {:?}: {{ \"role\": {:?}, \"scale\": {:.9e}, \"zero_point\": {}, \"exponent\": {}, \"num_bits\": {} }}{}\n",
            name,
            role,
            entry.config.scale,
            entry.config.zero_point,
            entry.config.exponent,
            entry.config.num_bits,
            comma,
        ));
    }
    out.push_str("}\n");
    out
}

/// Render a compact graph dump similar in spirit to esp-ppq's `model.info`.
pub fn render_model_info(graph: &BurnGraph) -> String {
    let mut out = String::new();
    out.push_str("producer: burn-espdl-export\n");
    out.push_str(&format!(
        "input: {} {:?}\n",
        graph.input_name, graph.input_shape
    ));
    out.push_str(&format!("output: {}\n", graph.output_name));
    out.push_str("nodes:\n");
    for (idx, layer) in graph.layers.iter().enumerate() {
        out.push_str(&format!(
            "  {idx:02}: {:<10} {} -> {}\n",
            layer.op_type(),
            layer.input(),
            layer.output()
        ));
    }
    out
}
