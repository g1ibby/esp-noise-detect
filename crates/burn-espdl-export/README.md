# burn-espdl-export

Native Rust exporter that converts a trained Burn model into an ESP-DL
`.espdl` binary for ESP32-S3 inference, replacing the legacy
`Burn -> safetensors -> PyTorch + esp-ppq Docker` pipeline.

This crate is being implemented in seven steps; see
`docs/BURN_TO_ESPDL_TASK.md` for the plan and
`docs/PORT_SCOPE_REPORT.md` for the reference walk-through. The
production-ready `TinyConv` checkpoint wrapper lives in `nn-rs` as
`cargo run -p nn-rs --bin export_espdl`; this crate remains the
model-agnostic exporter library.

## High-level API

Callers provide a model-agnostic `BurnGraph`, flat calibration windows,
and a Burn backend device. The ergonomic API currently exposes the
verified ESP32-S3 INT8 path only; INT16 internals are not advertised
until fixture-backed export parity exists.

```rust
let artifacts = burn_espdl_export::EspdlExporter::esp32s3_int8()
    .export_graph::<B>(&graph, &windows, &device)?;

artifacts.write_to_dir(out_dir)?;
```

This writes:

- `model.espdl`
- `model.json`
- `model.info`

## What ships today

* `schema/Dl.fbs`
  Verbatim copy of `edgedl/macros/Dl.fbs` (the canonical ESP-DL
  FlatBuffers schema). Re-running `flatc` is a manual one-time step if
  the schema ever changes; nothing else in this crate runs it.
* `src/dl_generated.rs`
  Verbatim copy of `edgedl/macros/src/Dl_generated.rs` (the
  pre-generated Rust bindings). This is the file the crate actually
  links against. To regenerate, run `flatc --rust schema/Dl.fbs` from
  the crate root and replace `src/dl_generated.rs`.
* `src/container.rs`
  The 16-byte `EDL2` framing layer (`EspdlContainer`).
* `src/reader.rs`
  Tiny host-side reader (`EspdlFile`) that wraps the container parser
  and FlatBuffers root verification.
* `src/writer.rs` / `src/export.rs`
  Structural model writer: `write_empty()`, `write_model(&dl::Model)`,
  `clone_*` helpers, and the graph writer that emits ESP-DL op nodes,
  quantized weights, layout annotations, and the `EDL2` container.
* `src/ir.rs`, `src/calib.rs`, `src/layout.rs`, `src/quant.rs`
  Model-agnostic BurnGraph extraction helpers, BN/Relu rewrites, PTQ
  calibration, weight packing, and quantization primitives.

## Tests

```
cargo test -p burn-espdl-export
```

End-to-end native-vs-legacy ESP-DL parity is driven from the workspace
root through `xtask`:

```sh
cargo xtask espdl-acceptance \
    --checkpoint runs/robust_session/checkpoints/best.mpk \
    --config nn-rs/configs/robust_session.yaml \
    --manifest ../recordings/manifest.jsonl \
    --out-dir target/espdl-acceptance-fixed-512 \
    --calib-steps 512
```

The `nn-rs` wrapper owns checkpoint loading and calibration-window
collection; this crate remains model-agnostic.

The golden round-trip tests (`flatbuffers_roundtrip`, `golden_roundtrip`)
expect `/tmp/nn-rs-robust-cuda/export/model.espdl` to exist; they skip
cleanly with a printed warning if it is missing. CI is expected to mount
the artifact at that path.
