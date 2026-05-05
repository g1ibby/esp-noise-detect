//! Two-phase calibration driver.
//!
//! Mirrors the relevant subset of
//! `_reference/esp-ppq/esp_ppq/quantization/optim/calibration.py::RuntimeCalibrationPass`:
//!
//! * **Phase 1 (Detecting Minmax)** — sweep the calibration loader,
//!   record min/max for every activation; one-shot percentile observe
//!   for every weight read straight off the IR. Render weight scales
//!   first so activation calibration runs with fake-quantized weights,
//!   matching esp-ppq's pass order.
//! * **Phase 2 (Collating Hist)** — sweep the loader again; the
//!   activations' `KlHistObserver` accumulates a 4096-bin histogram.
//!   Render activation scales via the KL search.
//! * **Bias** is passive: `scale_bias = scale_input · scale_weight`,
//!   20 bits for INT8 and 40 bits for INT16, derived after the two
//!   phases finish.
//!
//! The runner does **not** know what model it's calibrating. It only
//! sees a [`crate::ir::BurnGraph`] (post-fold/post-fuse) plus a
//! borrowed list of calibration windows. That keeps the calibration
//! layer model-agnostic, matching the architectural non-negotiables
//! in `BURN_TO_ESPDL_TASK.md`.

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor as BurnTensor, TensorData};

use crate::ir::{BurnGraph, Layer, forward_with_fake_quant_hook};

use super::{
    QuantConfig,
    observer::{KlHistObserver, Observer, PercentileObserver, derive_bias_config},
};

/// Bit-width and target choices for one calibration run.
///
/// Mirrors esp-ppq's `_PlatformQuantConfig` for ESP32-S3:
/// `(num_of_bits, bias_bits)` is `(8, 20)` for INT8 and `(16, 40)`
/// for INT16. The S3 path always uses `ROUND_HALF_UP` for
/// activations and `ROUND_UP` for weights — those are baked into
/// [`super::observer`], not surfaced here.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationConfig {
    /// Activation + weight bit width. 8 (S3 INT8) or 16 (S3 INT16).
    pub num_bits: u8,
    /// Bias bit width. 20 (S3 INT8) or 40 (S3 INT16). Stored as
    /// INT32 with an exponent regardless.
    pub bias_bits: u8,
}

impl CalibrationConfig {
    /// S3 INT8 defaults — the only target Step 4 needs to support
    /// today, kept as a named constructor so the test code can be
    /// noise-free.
    pub fn esp32s3_int8() -> Self {
        Self {
            num_bits: 8,
            bias_bits: 20,
        }
    }

    /// S3 INT16. Not exercised by Step 4 tests yet; the runner
    /// works with any `num_bits ∈ {8, 16}` so it's exposed for
    /// future use.
    pub fn esp32s3_int16() -> Self {
        Self {
            num_bits: 16,
            bias_bits: 40,
        }
    }
}

/// What kind of tensor a [`QuantConfig`] in the [`ScaleTable`]
/// describes. Useful for downstream consumers (the FlatBuffers
/// writer, the JSON exporter) to dispatch on the right field of
/// `Tensor.exponents` / `ValueInfo.exponents`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// Graph input or any layer output. Calibrated via KL.
    Activation,
    /// Conv2d/Linear weight. Calibrated via S3 percentile.
    Weight,
    /// Conv2d/Linear bias. Passive: scale = scale_in × scale_w.
    Bias,
}

/// Failure modes from [`calibrate`].
#[derive(Debug, Clone)]
pub enum CalibrationError {
    /// No calibration windows were supplied. esp-ppq enforces a
    /// minimum of 2 (`calibration.py:154-158`); we mirror that to
    /// reject the obvious mistake early.
    EmptyDataset,
    /// A window's flat length does not match the graph's declared
    /// `[N, C, H, W]` shape.
    ShapeMismatch {
        expected: usize,
        got: usize,
        window_idx: usize,
    },
}

impl core::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDataset => write!(
                f,
                "calibration: at least one calibration window is required"
            ),
            Self::ShapeMismatch {
                expected,
                got,
                window_idx,
            } => {
                write!(
                    f,
                    "calibration: window {window_idx} has length {got}, expected {expected} \
                     for the graph's input shape",
                )
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

/// All per-tensor configs produced by one calibration run.
///
/// Keys are deterministic strings:
///
/// * Activations: the IR tensor name (`graph.input_name` or
///   `Layer::output()`).
/// * Parameters: `<layer.output()>.weight` / `<layer.output()>.bias`.
///   Using the layer's *output* name (not its position) keeps the
///   key stable across IR rewrites that do not change tensor
///   identities (BN-fold, Relu-fuse).
///
/// Ordering: a `BTreeMap` so iteration is alphabetical and the
/// resulting JSON / parity tests have stable diffs.
#[derive(Debug, Clone, Default)]
pub struct ScaleTable {
    pub entries: BTreeMap<String, ScaleEntry>,
}

/// One row of the [`ScaleTable`].
#[derive(Debug, Clone, Copy)]
pub struct ScaleEntry {
    pub config: QuantConfig,
    pub role: TensorRole,
}

impl ScaleTable {
    /// Look up a tensor's quant config by IR name (or `<layer>.weight`
    /// / `<layer>.bias` for parameters).
    pub fn get(&self, key: &str) -> Option<&ScaleEntry> {
        self.entries.get(key)
    }

    /// Number of entries — handy for sanity tests.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate `(name, entry)` pairs in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ScaleEntry)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Run the two-phase calibration over `windows` and emit the
/// per-tensor scales/exponents for every quantizable tensor in
/// `graph`.
///
/// `windows` is a flat host-side iterator: each entry must be a
/// row-major `Vec<f32>` whose length matches `graph.input_shape`'s
/// product. esp-ppq's calibration loop doesn't see batches either —
/// each window is treated independently.
pub fn calibrate<B: Backend>(
    graph: &BurnGraph,
    windows: &[Vec<f32>],
    cfg: CalibrationConfig,
    device: &B::Device,
) -> Result<ScaleTable, CalibrationError> {
    if windows.is_empty() {
        return Err(CalibrationError::EmptyDataset);
    }
    let expected: usize = graph.input_shape.iter().product();
    for (i, w) in windows.iter().enumerate() {
        if w.len() != expected {
            return Err(CalibrationError::ShapeMismatch {
                expected,
                got: w.len(),
                window_idx: i,
            });
        }
    }

    // ---------------------------------------------------------------
    // Build observers
    // ---------------------------------------------------------------
    // Activations: graph input + every layer output.
    let mut act_names: Vec<String> = Vec::with_capacity(graph.layers.len() + 1);
    act_names.push(graph.input_name.clone());
    for l in &graph.layers {
        act_names.push(l.output().to_string());
    }
    let mut act_observers: BTreeMap<String, KlHistObserver> = act_names
        .iter()
        .map(|n| (n.clone(), KlHistObserver::new(cfg.num_bits)))
        .collect();

    // Weights: per Conv2d / Linear, keyed off the producing layer's
    // output name. ESP-DL S3 leaves these on esp-ppq's default
    // percentile observer (0.9999), not min-max; the difference is
    // visible when a tensor's absolute max sits just above a pow-2
    // boundary.
    let mut weight_observers: BTreeMap<String, (PercentileObserver, &Layer)> = BTreeMap::new();
    for l in &graph.layers {
        match l {
            Layer::Conv2d { weight, .. } => {
                let mut obs = PercentileObserver::new();
                obs.observe_minmax(&weight.data);
                weight_observers.insert(format!("{}.weight", l.output()), (obs, l));
            }
            Layer::Linear { weight, .. } => {
                let mut obs = PercentileObserver::new();
                obs.observe_minmax(&weight.data);
                weight_observers.insert(format!("{}.weight", l.output()), (obs, l));
            }
            _ => {}
        }
    }

    // esp-ppq calibrates activations after ParameterQuantizePass. That
    // means calibration forwards see fake-quantized weights, while
    // passive parameters such as biases are still raw FP32 until after
    // activation alignment/calibration completes.
    for (obs, _) in weight_observers.values_mut() {
        obs.finalize_phase1();
    }
    let weight_configs: BTreeMap<String, QuantConfig> = weight_observers
        .iter()
        .map(|(name, (obs, _))| (name.clone(), obs.render(cfg.num_bits)))
        .collect();
    let parameter_scale = |name: &str| {
        weight_configs
            .get(name)
            .map(|config| (config.scale, config.num_bits))
    };
    let no_activation_scale = |_name: &str| None;

    // ---------------------------------------------------------------
    // Phase 1 — min/max sweep over the calibration set
    // ---------------------------------------------------------------
    for window in windows {
        let input = window_to_burn::<B>(window, graph.input_shape, device);
        let _ = forward_with_fake_quant_hook(
            graph,
            input,
            device,
            &no_activation_scale,
            &parameter_scale,
            &mut |name, values| {
                if let Some(obs) = act_observers.get_mut(name) {
                    obs.observe_minmax(values);
                }
            },
        );
    }

    // Finalize Phase 1 for activation observers (computes hist_scale).
    for obs in act_observers.values_mut() {
        obs.finalize_phase1();
    }
    // ---------------------------------------------------------------
    // Phase 2 — histogram sweep (activations only)
    // ---------------------------------------------------------------
    for window in windows {
        let input = window_to_burn::<B>(window, graph.input_shape, device);
        let _ = forward_with_fake_quant_hook(
            graph,
            input,
            device,
            &no_activation_scale,
            &parameter_scale,
            &mut |name, values| {
                if let Some(obs) = act_observers.get_mut(name) {
                    obs.observe_hist(values);
                }
            },
        );
    }

    // ---------------------------------------------------------------
    // Render
    // ---------------------------------------------------------------
    let mut entries: BTreeMap<String, ScaleEntry> = BTreeMap::new();

    // Activations.
    for (name, obs) in &act_observers {
        entries.insert(
            name.clone(),
            ScaleEntry {
                config: obs.render(cfg.num_bits),
                role: TensorRole::Activation,
            },
        );
    }
    // Weights.
    for (name, config) in &weight_configs {
        entries.insert(
            name.clone(),
            ScaleEntry {
                config: *config,
                role: TensorRole::Weight,
            },
        );
    }

    // Biases (passive). Each Conv2d / Linear with a bias gets one.
    // Scale = scale_input(layer) × scale_weight(layer), bit_width =
    // cfg.bias_bits.
    for layer in &graph.layers {
        let (has_bias, layer_input, layer_output) = match layer {
            Layer::Conv2d {
                input,
                output,
                bias,
                ..
            } => (bias.is_some(), input.clone(), output.clone()),
            Layer::Linear {
                input,
                output,
                bias,
                ..
            } => (bias.is_some(), input.clone(), output.clone()),
            _ => continue,
        };
        if !has_bias {
            continue;
        }
        let weight_key = format!("{layer_output}.weight");
        let input_cfg = entries.get(&layer_input).map(|e| &e.config);
        let weight_cfg = entries.get(&weight_key).map(|e| &e.config);
        if let Some(bias_cfg) = derive_bias_config(input_cfg, weight_cfg, cfg.bias_bits) {
            entries.insert(
                format!("{layer_output}.bias"),
                ScaleEntry {
                    config: bias_cfg,
                    role: TensorRole::Bias,
                },
            );
        }
    }

    Ok(ScaleTable { entries })
}

fn window_to_burn<B: Backend>(
    window: &[f32],
    shape: [usize; 4],
    device: &B::Device,
) -> BurnTensor<B, 4> {
    let data = TensorData::new(window.to_vec(), shape);
    BurnTensor::<B, 4>::from_data(data, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dataset_is_an_error() {
        // A "graph" stub: empty layers, but a non-zero input shape.
        let graph = BurnGraph {
            input_name: "input".to_string(),
            input_shape: [1, 1, 1, 1],
            output_name: "logits".to_string(),
            layers: vec![],
        };
        let cfg = CalibrationConfig::esp32s3_int8();
        // We can't pick a backend in unit tests without depending on
        // a real device; but the empty-dataset check fires before
        // any forward runs, so we use a stub by giving a `()` device
        // through a zero-sized backend would still need a Backend
        // type. Instead, we just exercise the error path through
        // a tiny in-process check that doesn't hit the backend.
        // (`calibrate` returns the error before allocating the
        // input tensor.)
        let res = calibrate::<burn::backend::NdArray>(&graph, &[], cfg, &Default::default());
        assert!(matches!(res, Err(CalibrationError::EmptyDataset)));
    }

    #[test]
    fn shape_mismatch_is_caught_per_window() {
        let graph = BurnGraph {
            input_name: "input".to_string(),
            input_shape: [1, 1, 4, 4], // expected 16 elements
            output_name: "logits".to_string(),
            layers: vec![],
        };
        let windows = vec![vec![0.0_f32; 16], vec![0.0_f32; 8]]; // 2nd window wrong
        let cfg = CalibrationConfig::esp32s3_int8();
        let res = calibrate::<burn::backend::NdArray>(&graph, &windows, cfg, &Default::default());
        match res {
            Err(CalibrationError::ShapeMismatch {
                window_idx,
                expected,
                got,
            }) => {
                assert_eq!(window_idx, 1);
                assert_eq!(expected, 16);
                assert_eq!(got, 8);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }
}
