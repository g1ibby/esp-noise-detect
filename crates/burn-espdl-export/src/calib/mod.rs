//! Calibration & observers — Step 4 deliverable.
//!
//! This module ports the ESP32-S3 PTQ calibration pipeline from
//! `_reference/esp-ppq/esp_ppq/quantization/observer/range.py` and the
//! two-phase driver from
//! `_reference/esp-ppq/esp_ppq/quantization/optim/calibration.py` into
//! pure Rust on top of the [`crate::ir::BurnGraph`] IR built in
//! Step 3.
//!
//! Pipeline shape (mirrors the S3 INT8 path of esp-ppq):
//!
//! ```text
//!   Phase 1 (Detecting Minmax):
//!     for each calibration window:
//!     → freeze weight scales (0.9999 percentile, then ROUND_UP
//!        pow-2 snap on the symmetric min/max formula).
//!       BurnGraph::forward(window) with fake-quantized weights
//!         → record min/max for every activation.
//!
//!   Phase 2 (Collating Hist):
//!     for each calibration window:
//!       BurnGraph::forward(window) with fake-quantized weights
//!         → accumulate a 4096-bin histogram of |activation|
//!           over [0, hist_scale * 4096].
//!     → derive activation scales via KL-divergence search,
//!        ROUND_HALF_UP pow-2 snap.
//!
//!   Bias is passive: `scale_bias = scale_input * scale_weight`,
//!   computed after both phases finish, with a 20-bit (INT8 path) or
//!   40-bit (INT16 path) range stored as INT32 with an exponent.
//! ```
//!
//! Public surface (re-exported from [`crate::calib`]):
//!
//! * [`pow2`]: `ROUND_UP` / `ROUND_HALF_UP` power-of-two snap.
//! * [`Observer`], [`PercentileObserver`], [`MinMaxObserver`],
//!   [`KlHistObserver`]: the observer kinds used by the S3 path.
//! * [`QuantConfig`]: per-tensor `(scale, exponent, num_bits, …)`
//!   payload, mirroring the relevant subset of esp-ppq's
//!   `TensorQuantizationConfig`.
//! * [`ScaleTable`]: the keyed map of all per-tensor configs the
//!   calibration pass produces.
//! * [`calibrate`]: the two-phase driver — feeds calibration windows
//!   into the IR forward executor, attaches observers, derives
//!   scales, and emits a [`ScaleTable`].

mod observer;
mod pow2;
mod runner;

pub use observer::{
    KlHistObserver, MinMaxObserver, Observer, PercentileObserver, derive_bias_config, kl_hist_bins,
    qmin_qmax,
};
pub use pow2::{Pow2Policy, log2_floor, log2_round_half_up, pow2_round};
pub use runner::{CalibrationConfig, CalibrationError, ScaleTable, TensorRole, calibrate};

/// Per-tensor quantization configuration emitted by every observer.
///
/// Fields chosen to mirror what
/// `esp_ppq/parser/espdl/export_patterns.py::QuantVariableToIntPattern`
/// reads back when it serializes a tensor:
///
/// * `scale` — the float value used to map int → float.
/// * `zero_point` — always `0` on S3 (symmetric); kept to make the
///   field obvious for downstream consumers.
/// * `exponent` — `int(log2(scale))`. Stored alongside the scale
///   because the device-side decoder reads the exponent, not the
///   scale, from the FlatBuffers tensor.
/// * `num_bits` — 8 (S3 INT8 activations + weights), 16 (S3 INT16),
///   or 20/40 (passive bias on the int8/int16 path respectively).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantConfig {
    /// Float scale (signed, but always positive in practice on S3).
    pub scale: f32,
    /// Always `0` for symmetric quantization.
    pub zero_point: i32,
    /// `int(log2(scale))`.
    pub exponent: i32,
    /// Bit width of the integer representation.
    pub num_bits: u8,
}

impl QuantConfig {
    /// Construct the config from a derived `scale` and a bit width.
    /// Assumes power-of-two and symmetric: panics if the scale is
    /// non-positive (the observers guarantee that path is unreachable).
    pub fn from_pow2_scale(scale: f32, num_bits: u8) -> Self {
        assert!(
            scale > 0.0,
            "QuantConfig::from_pow2_scale: scale must be positive (got {scale})",
        );
        Self {
            scale,
            zero_point: 0,
            exponent: log2_floor(scale),
            num_bits,
        }
    }
}
