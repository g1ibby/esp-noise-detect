# nn-rs

Pump-noise classifier training / inference pipeline (Burn / CubeCL port of
the Python `nn/` tree). Three binaries:

- `train` — TinyConv training loop with macro-F1 early stopping.
- `eval` — load a checkpoint, walk a manifest, dump `metrics.json` /
  `calibration.json` / `threshold.json`.
- `export_weights` — dump a trained checkpoint to a PyTorch-compatible
  `.safetensors` blob for the `python/burn_to_onnx.py` bridge.

The same source tree compiles against **two production GPU backends**:

| Backend | OS                 | Activation feature | Cubecl runtime           |
|---------|--------------------|--------------------|--------------------------|
| Metal   | macOS (Apple Silicon) | `metal`         | `cubecl::wgpu::WgpuRuntime` (MSL) |
| CUDA    | Linux / Windows + NVIDIA driver ≥ toolkit min | `cuda` | `cubecl::cuda::CudaRuntime` (native cuBLAS / cuDNN / fusion) |
| Vulkan  | Linux (AMD / fallback) | `vulkan`        | `cubecl::wgpu::WgpuRuntime` (SPIR-V) |
| WebGPU  | (browser / generic) | `webgpu`         | `cubecl::wgpu::WgpuRuntime` (WGSL) |

Backend selection happens at compile time via a Cargo feature. Burn's
`Module` / `AutodiffBackend` trait tree is not dyn-compatible, so we
follow Burn's own examples and pick a backend with `#[cfg(feature = …)]`
in `src/train/runner.rs`.

The four backend features (`cuda`, `metal`, `vulkan`, `webgpu`) are
**mutually exclusive** — pick exactly one per build.

## Prerequisites

### Apple Silicon (Metal)
- macOS with Xcode Command Line Tools (`xcode-select --install`).
- No extra runtime libraries; Metal is part of the OS.

### Linux / NVIDIA (CUDA)
- NVIDIA driver compatible with CUDA Toolkit ≥ 12.0.
- CUDA Toolkit ≥ 12.0 with `nvcc` and `libnvrtc` on `PATH` /
  `LD_LIBRARY_PATH`. Verify with `nvcc --version` and `nvidia-smi`.
- `cubecl-cuda` links `nvrtc` at build time; failures here are usually
  toolkit-not-installed or wrong `LD_LIBRARY_PATH`.

## Build

The `.cargo/config.toml` at the repo root defines aliases. Either form
below works — the raw form is documented for parity with CI scripts that
shouldn't depend on aliases.

```sh
# Mac (Metal)
cargo build-mac
# === cargo build --release -p nn-rs --no-default-features \
#         --features 'std,metal' --bin train

# Linux / NVIDIA (CUDA)
cargo build-cuda
# === cargo build --release -p nn-rs --no-default-features \
#         --features 'std,cuda' --bin train
```

To build all three bins on a backend, use `cargo build --release -p nn-rs
--no-default-features --features 'std,metal' --bins` (or `'std,cuda'`).

## Train

```sh
# Mac
cargo train-mac -- --config nn-rs/configs/robust_session.yaml

# Linux / NVIDIA
cargo train-cuda -- --config nn-rs/configs/robust_session.yaml
```

Raw form (no alias):

```sh
cargo run --release -p nn-rs --no-default-features \
    --features "std,metal" --bin train -- \
    --config nn-rs/configs/robust_session.yaml \
    [--manifest /path/to/voicy/recordings/manifest.jsonl] \
    [--epochs 30] [--batch-size 128] [--artifact-dir runs/robust_session]
```

Swap `metal` for `cuda` on the NVIDIA box.

## Evaluate

```sh
cargo run --release -p nn-rs --no-default-features \
    --features "std,metal" --bin eval -- \
    --config nn-rs/configs/robust_session.yaml \
    --checkpoint runs/robust_session/checkpoints/best \
    --split val
```

## Export weights for the Python ONNX bridge

```sh
cargo run --release -p nn-rs --no-default-features \
    --features "std,metal" --bin export_weights -- \
    --config nn-rs/configs/robust_session.yaml \
    --checkpoint runs/robust_session/checkpoints/best \
    --out runs/robust_session/export/tinyconv.safetensors
```

## Train on RunPod via SkyPilot

If you don't have a local NVIDIA box, `sky.yaml` at the repo root
provisions a spot pod on RunPod (L40S / RTX 4090 / RTXPRO4500 / A100),
builds nn-rs with the `cuda` backend, and runs the three Rust steps
(train → eval → export_weights). The Python ONNX bridge + esp-ppq
Docker quantizer stay on the laptop afterwards, so the pod image
stays minimal and you don't fight nested-Docker on RunPod.

### Prerequisites

```sh
# SkyPilot with the RunPod provisioner.
pip install 'skypilot[runpod]'
# Configure RunPod credentials + an R2 bucket (one-time); see:
# https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/runpod.html
sky check runpod
```

The dataset source path is read from the `RECORDINGS_PATH` environment
variable (declared under `envs:` in `sky.yaml`). Pass it on the CLI or
export it in your shell before launching — see the command block below.
First launch uploads the recordings folder (~8 GB) to R2; subsequent
launches pull server-to-server.

### Launch + download + finish locally

Run every command **from the repo root** — SkyPilot resolves `workdir:`
against the CWD of `sky launch`, not the yaml's location.

```sh
# 1) Provision the pod, build nn-rs with --features cuda, train, eval,
#    export_weights. 3-epoch robust_session takes ~2-3 min on a 4090.
#    Pass RECORDINGS_PATH via --env (or export it in your shell).
sky launch -c nn-rs-pump sky.yaml -y \
    --env RECORDINGS_PATH=/abs/path/to/voicy/recordings

# 2) Pull checkpoint + metrics + safetensors back. SkyPilot 0.12 has no
#    `sky rsync-down`; use plain rsync via the SSH alias SkyPilot
#    registers in ~/.ssh/config.
rsync -avz --progress \
    nn-rs-pump:sky_workdir/runs/robust_session/ \
    /tmp/nn-rs-robust-cuda/

# 3) Finish on the laptop: safetensors → ONNX (onnxslim) → .espdl
#    (esp-ppq PTQ INT8 in Docker). Requires Docker Desktop running.
scripts/burn_to_espdl.sh \
    --checkpoint /tmp/nn-rs-robust-cuda/checkpoints/best.mpk \
    --manifest   /path/to/voicy/recordings/manifest.jsonl \
    --out-dir    /tmp/nn-rs-robust-cuda/export

# 4) Stop billing.
sky down nn-rs-pump -y
```

The final `.espdl` lands at `/tmp/nn-rs-robust-cuda/export/model.espdl`
— copy it into `firmware/models/` and rebuild the firmware binary.

### Re-runs on the same cluster

```sh
# Changed setup: (toolchain / apt packages) — re-run setup idempotently.
sky launch -c nn-rs-pump sky.yaml -y

# Changed only run: (training flags) — skip setup for speed.
sky exec nn-rs-pump sky.yaml
```

### Notes

- `.skyignore` at the repo root overrides `.gitignore` so that
  `_reference/` (a required path dep) is uploaded while `target/`,
  `firmware/`, `host-tools/`, etc. are skipped.
- The R2 bucket (`name: pump-dataset`) is `persistent: true` and is
  shared with `nn/sky.yaml` — `sky down` does not delete it.
- Spot pods can be preempted; a 3-epoch training is short enough to
  tolerate this. Bump `use_spot: false` in `sky.yaml` if you need a
  guaranteed long-running slot.

## Tests

Tests are backend-agnostic: they use `cubecl::TestRuntime`, which
resolves to whichever runtime is selected by the `test-*` feature flag.
Like the production backend features, `test-metal` / `test-vulkan` /
`test-cuda` / `test-cpu` are **mutually exclusive** per `cargo test` run.

```sh
# Apple Silicon
cargo test-mac
# === cargo test --workspace --no-default-features \
#         --features 'std,test-metal'

# Linux / NVIDIA (needs CUDA Toolkit)
cargo test-cuda
# === cargo test --workspace --no-default-features \
#         --features 'std,test-cuda'

# Linux / AMD or fallback (Vulkan via wgpu)
cargo test-vulkan

# No GPU required (slow but portable; not a GPU sanity check)
cargo test-cpu
```

The `cubek-*` and `burn-audiomentations` crates each carry their own
`test-*` features and a `tests/_backend_guard.rs` that fails the build
if zero or none of those features is set — the silent fallback to wgpu
(which `cubecl/build.rs` defaults to) would otherwise mask a CUDA-only
box pretending to test on CUDA when it's actually using wgpu-Metal /
wgpu-Vulkan.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `compile_error: nn-rs requires exactly one GPU backend feature` | Built nn-rs without any backend feature. | Add `--features metal` / `cuda` / `vulkan` / `webgpu`. |
| `compile_error: the cuda / test-cuda features are not supported on macOS` | Tried to build the CUDA path on a Mac. | Use `metal` (or `test-metal`) on macOS. |
| `compile_error: the metal / test-metal features require macOS` | Tried to build the Metal path on Linux. | Use `cuda` on NVIDIA, `vulkan` on AMD / fallback. |
| `compile_error: choose a test backend` | `cargo test` without a `test-*` feature. | Add `--features test-metal` / `test-cuda` / etc. |
| Linux `cannot find -lnvrtc` / `nvrtc not found` | CUDA Toolkit missing or `LD_LIBRARY_PATH` not set. | Install CUDA Toolkit; export `LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`. |
| Runtime panic `CUDA_ERROR_NO_DEVICE` at first kernel launch | Built ok but no NVIDIA GPU visible to the process. | Check `nvidia-smi` works; in Docker pass `--gpus all`; in WSL2 ensure the Windows NVIDIA driver is current. |
| Runtime panic `Failed to create default device` (Metal) | Built `metal` on Apple Silicon but the process has no Metal access (some VMs / sandboxes). | Run on the host or fall back to `vulkan` via MoltenVK. |
| Tests silently fall back to wgpu when expecting CUDA | Two `test-*` features enabled at once → `cubecl/build.rs` selects `test_runtime_default` = wgpu. | Always pass exactly one `test-*` feature. The guard files catch zero; multiples slip through silently. |
