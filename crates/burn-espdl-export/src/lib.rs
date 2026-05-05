//! Native Burn -> ESP-DL `.espdl` exporter (work in progress).
//!
//! This crate replaces the existing 3-stage `burn_to_espdl.sh` pipeline
//! (Burn -> safetensors -> PyTorch + esp-ppq Docker) with a single Rust
//! binary. The high-level API exposes the verified ESP32-S3 INT8 path;
//! lower-level modules also contain experimental INT16 plumbing.
//!
//! Step 2 of the porting plan ships:
//!   * the FlatBuffers schema (`schema/Dl.fbs`) and pre-generated Rust
//!     bindings ([`dl`]) copied verbatim from `edgedl/macros/`,
//!   * a minimal `.espdl` container reader/writer pair that handles the
//!     `EDL2` 16-byte header used by esp-ppq's `helper.py::save`
//!     ([`container`], [`reader::EspdlFile`]),
//!   * a structural [`writer`] that re-serializes any [`dl::Model`]
//!     through the FlatBuffers builder API and wraps the result in the
//!     `EDL2` container.
//!
//! Quantization, BurnGraph extraction, and op-level writers land in
//! later steps — at this point nothing in the public API knows what a
//! Burn module looks like.

#![allow(clippy::too_many_arguments)]

#[allow(
    clippy::needless_lifetimes,
    clippy::extra_unused_lifetimes,
    clippy::missing_safety_doc,
    clippy::derivable_impls,
    dead_code,
    unused_imports,
    unsafe_op_in_unsafe_fn,
    mismatched_lifetime_syntaxes
)]
#[rustfmt::skip]
mod dl_generated;

pub use dl_generated::dl;

pub mod calib;
pub mod container;
pub mod export;
pub mod exporter;
pub mod ir;
pub mod layout;
pub mod quant;
pub mod reader;
pub mod writer;

pub use calib::{
    CalibrationConfig, CalibrationError, KlHistObserver, MinMaxObserver, Observer,
    PercentileObserver, QuantConfig, ScaleTable, TensorRole, calibrate,
};
pub use container::{EspdlContainer, EspdlContainerError};
pub use export::{ExportConfig, ExportError, write_graph};
pub use exporter::{
    EspdlExportError, EspdlExporter, ExportArtifacts, ExportOptions, render_model_info,
    render_quant_json,
};
pub use ir::{Activation, BurnGraph, Layer, Tensor as IrTensor, fold_batchnorm, fuse_relu};
pub use reader::{EspdlFile, EspdlReadError};
pub use writer::{write_empty, write_model};
