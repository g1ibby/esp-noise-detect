# noise-detect — Electric Pump Audio Classifier

Binary classifier that detects whether audio contains a running electric pump (`pump_on` vs `pump_off`).

**Stack:** PyTorch + Lightning + Hydra. Export path: PyTorch → ONNX → ESP-DL (`.espdl`).

---

## 1. Setup

### Requirements
- Python 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (only required for the `.espdl` export step)

### Install

```bash
cd nn
uv venv .venv
# Training + evaluation only:
uv pip install -e .[dev]

# Add the export extras if you plan to produce .espdl models:
uv pip install -e .[dev,export]
```

> The `export` extra pulls `onnx`, `onnxruntime`, and `onnxslim`, which are required by `noise_detect.export`. Skip it and export will fail at import time.

### Platform notes

| Platform | Accelerator | Default precision |
|----------|-------------|-------------------|
| Apple Silicon | MPS (auto) | `bf16-mixed` |
| CUDA | GPU | `16-mixed` |
| CPU only | CPU | `32` |

---

## 2. Prepare your audio + build the manifest

### Audio requirements
- Format: WAV, 32 kHz, 24-bit, mono
- Location: **any folder you choose** — you point the manifest script at it explicitly

### Filename convention (important)

Labels are inferred from the filename. Each WAV must contain either the token `on` **or** the token `off` (case-insensitive, delimited by non-alphanumerics):

- `pump_on_001.wav` → `pump_on`
- `recording-off-17.wav` → `pump_off`
- `ambient_noise.wav` → **skipped** (no label)
- `turn_on_off_test.wav` → **skipped** (ambiguous)

Files that can't be labeled unambiguously are silently skipped — check the summary output to confirm counts.

### Generate the manifest

`scripts/build_manifest.py` walks the folder, assigns stratified train/val/test splits (80/10/10 by default), and writes `manifest.jsonl` **into the recordings folder itself** (next to your WAVs). Paths inside the manifest are stored relative to that folder.

```bash
# Preview (no file written)
uv run python scripts/build_manifest.py --data-root /path/to/recordings --dry-run

# Write the manifest
uv run python scripts/build_manifest.py --data-root /path/to/recordings
# → /path/to/recordings/manifest.jsonl
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | *(required)* | Folder containing your WAV files (searched recursively) |
| `--output` | `<data-root>/manifest.jsonl` | Override output location |
| `--val-ratio` | `0.1` | Validation split fraction |
| `--test-ratio` | `0.1` | Test split fraction |
| `--seed` | `42` | Shuffle seed for split assignment |

Manifest format:
```json
{"audio_path": "pump_on_001.wav", "label": "pump_on", "split": "train"}
{"audio_path": "subdir/off_042.wav", "label": "pump_off", "split": "val"}
```

---

## 3. Train

Point Hydra at your manifest:

```bash
uv run -m noise_detect.train dataset.manifest_path=/path/to/recordings/manifest.jsonl
```

Checkpoints are written to `runs/<timestamp>/checkpoints/best.ckpt` (Hydra changes into `runs/<timestamp>/` during the job).

### Common overrides

```bash
# More epochs + augmentation
uv run -m noise_detect.train \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  trainer.max_epochs=50 \
  augment=light

# Smaller windows (faster on-device inference)
uv run -m noise_detect.train \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  dataset.window_s=0.64 \
  dataset.hop_s=0.32

# Prebuilt experiment bundle
uv run -m noise_detect.train \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  experiment=light_baseline
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.max_epochs` | 20 | Training epochs |
| `trainer.precision` | `32` | Mixed-precision mode |
| `dataset.window_s` | 1.0 | Window size in seconds |
| `dataset.hop_s` | 0.5 | Hop size in seconds |
| `dataset.sample_rate` | 32000 | Audio sample rate (Hz) |
| `features.mel.n_mels` | 64 | Mel bands |
| `augment` | `none` | Preset: `none`, `light`, `strong` |

---

## 4. Evaluate

> **Note:** `eval` and `export` both need the manifest — pass the same `dataset.manifest_path=...` you used for training. `manifest_path` is not baked into the checkpoint.
>
> Training already runs a post-train eval on the val split (producing the first `metrics.json` / `threshold.json`), so you only need to run `eval` yourself for the test set or after changing a setting.

Calibrate the classification threshold on the **validation** set, then report final metrics on **test** without recalibrating.

```bash
# Step 1 — calibrate on val (writes threshold.json, calibration.json)
uv run -m noise_detect.eval \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  split=val

# Step 2 — report on test
uv run -m noise_detect.eval \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  split=test \
  calibrate=false
```

Outputs are written next to the checkpoint: `metrics.json`, `threshold.json`, `calibration.json`.

Useful extras:

```bash
# Per-window metrics + CSV dump
uv run -m noise_detect.eval \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  split=test \
  calibrate=false \
  window_metrics=true \
  window_preds_csv=predictions.csv
```

---

## 5. Export to `.espdl` (ESP32-S3)

Requires the `export` extras (step 1) and **Docker running**. On first run, the `esp-ppq:cpu` image is built from `nn/docker/esp-ppq-cpu.Dockerfile` (clones esp-ppq at a pinned commit).

The export step re-reads the dataset for PTQ calibration windows, so pass `dataset.manifest_path=` too:

```bash
uv run -m noise_detect.export \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl
```

Output:

```
runs/<timestamp>/export/
├── model.espdl          # ← flash target, copy into firmware/models/
├── model_fp32.onnx
├── model_fp32_slim.onnx
└── calib/               # calibration windows (.npy)
```

### Export options

| Option | Default | Description |
|--------|---------|-------------|
| `espdl.enabled` | `true` | Produce `.espdl` |
| `espdl.target` | `esp32s3` | `esp32s3`, `esp32p4`, or `c` |
| `espdl.num_bits` | `8` | Quantization bits (`8` or `16`) |
| `calib.split` | `train` | Split used for PTQ calibration |
| `calib.num_windows` | `512` | Representative windows for PTQ |

### Deploy to firmware

Copy the resulting `model.espdl` into `firmware/models/` — the `#[edgedl_macros::espdl_model]` macro picks it up at compile time.

---

## Inference (optional, host-side)

Sanity-check a checkpoint against a folder or manifest before exporting:

```bash
# Folder of WAVs
uv run -m noise_detect.infer \
  input_path=/path/to/recordings \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt

# Existing manifest split
uv run -m noise_detect.infer \
  use_manifest=true \
  split=val \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt \
  output_csv=predictions.csv
```

---

## Project structure

```
nn/
├── configs/              # Hydra configs
│   ├── augment/
│   ├── experiment/
│   ├── dataset/manifest.yaml
│   └── export.yaml
├── src/noise_detect/
│   ├── data/             # Dataset + datamodule
│   ├── features/         # Mel-spectrogram
│   ├── models/           # TinyConv, BC-ResNet
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   └── export.py
├── scripts/build_manifest.py
├── docker/               # esp-ppq Dockerfile
├── runs/                 # Checkpoints + logs (git-ignored)
└── docs/EXPORT_AND_QUANTIZATION.md
```

---

## Model I/O (for firmware integration)

- Input: `[1, 64, 101, 1]` INT8 mel-spectrogram (NHWC)
- Output: `[1, 2]` INT8 logits (`pump_off`, `pump_on`)
- Memory: ~170 KB PSRAM + ~126 KB flash
- Latency: ~31 ms per inference on ESP32-S3 (single core)

---

## Tips

- **Calibrate on val, report on test** — don't let the test set leak into threshold selection.
- **Window size** — try `dataset.window_s` in `{0.64, 0.8, 1.0}`; smaller windows = faster on-device inference.
- **Augmentation** — start with `augment=light`; escalate to `strong` if training data is limited.
- **Filename labels** — double-check the manifest summary; anything without a clear `on`/`off` token is skipped.
