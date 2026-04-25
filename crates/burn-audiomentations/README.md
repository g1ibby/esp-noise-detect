# burn-audiomentations

Waveform augmentation transforms for Burn / CubeCL audio pipelines.
Assembles `cubek-fft` / `cubek-stft` / `cubek-resample` / `cubek-sinc-filter`
/ `cubek-phasevocoder` into the transforms required by the production
augmentation pipeline (gain, polarity invert, colored noise, time
masking, low-/high-pass filters, pitch shift). Backend-agnostic — every
public type is generic over `R: cubecl::Runtime` and operates on
`cubecl::std::tensor::TensorHandle<R>`.

## Running tests

This crate is backend-agnostic. Pick one of the `test-*` features when
running tests. Use the explicit `--no-default-features --features "std,…"`
form to keep the invocation deterministic across machines:

```sh
cargo test --no-default-features --features "std,test-metal"   # Apple Silicon
cargo test --no-default-features --features "std,test-cuda"    # Linux / NVIDIA (needs CUDA Toolkit)
cargo test --no-default-features --features "std,test-vulkan"  # Linux / AMD / fallback
cargo test --no-default-features --features "std,test-cpu"     # no GPU required, slow but portable
```

Exactly one `test-*` feature must be enabled per run — `cubecl`'s build
script falls back to wgpu if zero or multiple are set, which will not
work on a CUDA-only box. `tests/_backend_guard.rs` carries a
`compile_error!` guard that catches the zero-feature case up front.

For workspace-level test invocations and the `.cargo/config.toml`
aliases (`cargo test-mac`, `cargo test-cuda`, …), see
[`nn-rs/README.md`](../../nn-rs/README.md).
