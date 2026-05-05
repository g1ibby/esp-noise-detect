//! High-level exporter API tests.

use std::time::{SystemTime, UNIX_EPOCH};

use burn::backend::NdArray;
use burn_espdl_export::{EspdlExporter, EspdlFile, Layer};

mod common;
use common::fixture_lowering::mininet_to_burn_graph;
use common::fixture_model::{MiniNetConfig, perturb_bn_stats};

type B = NdArray;
const INPUT_SHAPE: [usize; 4] = [1, 1, 16, 16];

#[test]
fn esp32s3_int8_exporter_returns_three_artifacts_without_mutating_graph() {
    let device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0xfeed_face_u64);
    let graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    let original_layer_count = graph.layers.len();
    assert!(
        graph
            .layers
            .iter()
            .any(|layer| matches!(layer, Layer::BatchNorm2d { .. })),
        "fixture should start pre-fold"
    );

    let windows = calibration_windows(32, 0x9e37_79b9_u64);
    let artifacts = EspdlExporter::esp32s3_int8()
        .export_graph::<B>(&graph, &windows, &device)
        .expect("high-level export");

    assert_eq!(&artifacts.model_bytes[..4], b"EDL2");
    assert!(artifacts.quant_json.contains("\"role\": \"activation\""));
    assert!(artifacts.model_info.contains("producer: burn-espdl-export"));
    EspdlFile::parse(&artifacts.model_bytes).expect("parse exported espdl");

    assert_eq!(graph.layers.len(), original_layer_count);
    assert!(
        graph
            .layers
            .iter()
            .any(|layer| matches!(layer, Layer::BatchNorm2d { .. })),
        "exporter must not fold the caller-owned graph"
    );
}

#[test]
fn artifacts_write_expected_filenames() {
    let device = Default::default();
    let mut model = MiniNetConfig::default().init::<B>(&device);
    perturb_bn_stats(&mut model, &device, 0xfeed_face_u64);
    let graph = mininet_to_burn_graph(&model, INPUT_SHAPE);
    let windows = calibration_windows(32, 0x9e37_79b9_u64);
    let artifacts = EspdlExporter::esp32s3_int8()
        .export_graph::<B>(&graph, &windows, &device)
        .expect("high-level export");

    let dir = unique_temp_dir("burn-espdl-exporter-api");
    artifacts.write_to_dir(&dir).expect("write artifacts");
    assert!(dir.join("model.espdl").is_file());
    assert!(dir.join("model.json").is_file());
    assert!(dir.join("model.info").is_file());
}

fn unique_temp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{}-{nanos}", std::process::id()))
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
