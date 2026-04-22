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

**Session timestamp.** If you plan to use the session- or day-aware split (recommended; see below), filenames should embed a 10-digit unix timestamp identifying the recording session, e.g. `xiao_esp32s3_1759581195_c000_off_chunk000.wav`. All chunks from the same recording session share that timestamp, so the splitter can keep a session's files together.

**Label-noise suffixes.** By default the manifest builder **excludes** files containing `_canceled`, `_timeout`, or `_undefined` — those correspond to aborted or truncated sessions and carry significant label noise. Pass `--keep-flagged` if you want them in anyway.

### Generate the manifest

`scripts/build_manifest.py` walks the folder, assigns train/val/test splits (80/10/10 by default), and writes `manifest.jsonl` **into the recordings folder itself** (next to your WAVs). Paths inside the manifest are stored relative to that folder.

```bash
# Preview (no file written). Default: split by whole recording session.
uv run python scripts/build_manifest.py --data-root /path/to/recordings --dry-run

# Write the manifest, keeping all files from the same *day* in one split
# (strongest held-out signal; recommended if you have >= ~10 recording days).
uv run python scripts/build_manifest.py \
  --data-root /path/to/recordings \
  --split-by day
```

**Pick the split strategy with care.** A random per-file shuffle silently leaks recording sessions across splits and produces inflated test accuracy: many chunk files share the same ambient noise and mic placement, so the model just recognizes the session. Use `--split-by session` (default) or `--split-by day` to keep whole sessions/days together.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | *(required)* | Folder containing your WAV files (searched recursively) |
| `--output` | `<data-root>/manifest.jsonl` | Override output location |
| `--split-by` | `session` | `session` (whole session → one split), `day` (whole calendar day → one split), or `random` (legacy, leaks sessions) |
| `--keep-flagged` | off | Keep `_canceled` / `_timeout` / `_undefined` files (default: exclude — they carry label noise) |
| `--val-ratio` | `0.1` | Validation split fraction |
| `--test-ratio` | `0.1` | Test split fraction |
| `--seed` | `42` | Shuffle seed for split assignment |

After building, verify the split is clean (0 session/day overlap):

```bash
uv run python scripts/check_split_integrity.py \
  --manifest /path/to/recordings/manifest.jsonl
```

The script reports path/content-hash duplicates **and** session/day-level leakage.

Manifest format:
```json
{"audio_path": "pump_on_001.wav", "label": "pump_on", "split": "train"}
{"audio_path": "subdir/off_042.wav", "label": "pump_off", "split": "val"}
```

---

## 3. Train

### Recommended command

```bash
uv run -m noise_detect.train \
  experiment=robust_session \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl
```

This bundles what we've validated as best for this task: the `robust` augmentation profile (gain / colored noise / high+low-pass / time masking — tuned to keep a small `[16, 32, 64]`-channel model generalizing well), cosine LR, 30 epochs with early stopping, and `dm.use_cache: false` so waveform augmentation actually runs (the cached-mel path silently skips it). Checkpoints land in `runs/<timestamp>/checkpoints/best.ckpt`.

### Minimal alternative

If you just want a fast baseline run without augmentation:

```bash
uv run -m noise_detect.train dataset.manifest_path=/path/to/recordings/manifest.jsonl
```

### Common overrides

```bash
# Smaller windows (faster on-device inference)
uv run -m noise_detect.train \
  experiment=robust_session \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  dataset.window_s=0.64 \
  dataset.hop_s=0.32

# Override the augmentation preset inside the experiment
uv run -m noise_detect.train \
  experiment=robust_session \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  augment=light

# Longer training
uv run -m noise_detect.train \
  experiment=robust_session \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  trainer.max_epochs=50
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.max_epochs` | 20 | Training epochs (30 under `experiment=robust_session`) |
| `trainer.precision` | `32` | Mixed-precision mode |
| `dataset.window_s` | 1.0 | Window size in seconds |
| `dataset.hop_s` | 0.5 | Hop size in seconds |
| `dataset.sample_rate` | 32000 | Audio sample rate (Hz) |
| `features.mel.n_mels` | 64 | Mel bands |
| `augment` | `none` | Preset: `none`, `light`, `strong`, `robust`, `robust_lite` |
| `experiment` | `null` | `light_baseline` or `robust_session` (recommended) |

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

**Error audit** — print a per-session / per-day accuracy breakdown plus the worst files by FP and FN count. Off by default (keeps normal eval output compact); flip it on when debugging a regression or deciding where to collect more data:

```bash
uv run -m noise_detect.eval \
  checkpoint=runs/<timestamp>/checkpoints/best.ckpt \
  dataset.manifest_path=/path/to/recordings/manifest.jsonl \
  split=test \
  calibrate=false \
  audit=true
```

The audit block looks like this, appended after the metrics output:

```
Error audit (threshold=0.490)
By day (worst-first):
  2025-10-26   5502  0.9869  TP=3042 TN=2388 FP=22 FN=50
  2025-10-24   4113  0.9947  TP=2881 TN=1210 FP=1  FN=21
  ...
Top 10 files by FP count:
   23  xiao_esp32s3_1775956737_c172_off_chunk000.wav
   17  xiao_esp32s3_1761522227_c073_off_chunk000.wav
  ...
```

It requires filenames with the 10-digit session timestamp (windows without one are skipped and counted). Tune `audit_top_n=N` for shorter/longer file lists.

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

- **Never trust a random-per-file split.** Use `--split-by session` (default) or `--split-by day` and run `scripts/check_split_integrity.py` after building. A leaky split will report 98–99 % accuracy on data the model has effectively seen.
- **Calibrate on val, report on test** — don't let the test set leak into threshold selection.
- **Window size** — try `dataset.window_s` in `{0.64, 0.8, 1.0}`; smaller windows = faster on-device inference.
- **Augmentation** — for the pump task, `experiment=robust_session` (which uses `augment=robust_lite`) is what we've validated. Plain `augment=light` is fine too; `augment=robust` is heavier and starts to overwhelm the small `[16,32,64]` model's capacity.
- **Debug regressions with `audit=true`** — the per-session / per-day accuracy breakdown tells you whether errors cluster in one hard session (label noise, specific acoustic condition) or are spread across the whole split (a real generalization gap).
- **Filename labels** — double-check the manifest summary; anything without a clear `on`/`off` token is skipped, and `_canceled` / `_timeout` / `_undefined` are excluded by default.
