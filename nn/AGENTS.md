Scope
- This AGENTS.md governs the `nn/` directory and all its subfolders. It complements the repository’s root guidelines. When instructions conflict, follow direct user/system prompts first, then this file, then root repo conventions.

Goals
- Build a Python + PyTorch audio classifier that detects a working electric pump in WAV audio (binary: pump_on vs pump_off).
- Train on desktop, export compact artifacts for eventual ESP32-S3 deployment (Rust inference out of scope here, but our exports should be MCU-friendly).

Languages, Versions, Tooling
- Python 3.13.
- Use uv for env + installs; prefer `uv pip install -e .[dev]` and `uv run ...` over raw pip/python.
- PyTorch (latest stable 2.x), torchaudio, torchmetrics, lightning.
- Configuration with Hydra + OmegaConf.
- Augmentations with torch-audiomentations.
- Experiment tracking with Aim (optional) and rich console logs.
- Dev tools: ruff, black, mypy, pytest, pre-commit (keep CI fast and local-only; no network in tests).

Platforms & Accelerators
- Primary dev target: macOS on Apple Silicon (M4). Default to MPS accelerator via Lightning (`accelerator=auto`) with CPU fallback.
- CUDA compatibility is required: ensure configs can switch to GPU with `accelerator=gpu`, `devices=1`, `precision=16-mixed` and that code avoids MPS/CUDA-specific assumptions.
- Precision guidance: prefer float32 or bf16-mixed on MPS; use 16-mixed (fp16) or bf16 on CUDA-capable GPUs.
- Avoid ops with known MPS gaps when possible; keep feature extraction CPU-bound (torchaudio/sox_io) then move tensors to device.

Project Layout (planned)
- `src/noise_detect/`
  - `data/` datasets and datamodules (manifest + optional HF Datasets)
  - `features/` audio preprocessing (resample, mel-spec, normalization)
  - `augment/` augmentation policies with torch-audiomentations
  - `models/` TinyConv, BC-ResNet variants, optional teacher (AST/Conformer-lite)
  - `train.py`, `eval.py`, `infer.py`, `export.py`
- `configs/` Hydra configs (dataset/*.yaml, model/*.yaml, train.yaml, export.yaml)
- `scripts/` utilities (e.g., build_manifest.py, split_dataset.py)
- `tests/` unit tests
- `runs/` outputs (git-ignored)

Data & I/O
- Default dataset root: `../recordings/` (preferred) with fallback to `../host-tools/recordings/`.
- WAV expected at 32 kHz, 24-bit. Use `torchaudio.load` with sox_io backend; normalize to float32 in [-1, 1].
- Provide a `manifest.jsonl` with fields: `audio_path` (str), `label` in {pump_on, pump_off}, optional `split` in {train,val,test}, optional `start_s`, `end_s` for window labels.
- Implement sliding-window segmentation for training (default 1.0 s window, 0.5 s hop). Ensure consistent handling at clip boundaries (pad/trim) and mono mixing if multi-channel.

Window Length Guidance
- Treat `train.window_s` as a hyperparameter. Evaluate {0.64, 0.8, 1.0, 1.28} seconds.
- 0.64 s rationale: with ~10 ms hop you get ~64 frames; combined with `n_mels=64`, this yields compact 64x64 Mel patches ideal for MCU memory/latency.
- 1.0 s rationale: more context improves robustness for weak/ambiguous pump signatures.
- For exact frame control, pad/center-crop to target T (e.g., 64) after Mel computation.

Features & Augmentations
- Mel-Spectrogram defaults: n_mels=64, fmin=50 Hz, fmax=SR/2, n_fft≈1024 (32 kHz), hop_length aligned to ~10 ms.
- Augmentations (training only): Gain, AddColoredNoise (SNR range 0–25 dB), Bandpass/High/Low-pass, TimeMasking, limited PitchShift, optional IR/Reverb. Keep validation/test clean.
- If a local `noise/` directory exists, support AddBackgroundNoise.

Models & Loss
- Start with TinyConv/BC-ResNet-8 on Mel-spectrograms. Keep params small (<200k ideal). Provide an option for a larger teacher (AST/Conformer-lite) for distillation.
- Use CrossEntropy (two logits) with optional class weighting or focal loss for imbalance.
- Calibrate threshold on validation to maximize macro-F1; expose threshold in config.

Training, Logging, Metrics
- Use Lightning Trainer (mixed precision optional), gradient clipping, cosine or OneCycle LR.
- TorchMetrics: accuracy, balanced accuracy, macro-F1 (primary), AUROC, AUPRC, confusion matrix.
- Early stopping on macro-F1 or AUPRC; checkpoint best.
- Logging: Rich console; optional Aim logger (off by default in CI). Ensure runs/ is git-ignored.
 - Device toggles: default `accelerator=auto` for MPS on macOS; document CUDA overrides in README and config examples.

Export & Quantization
- Provide `export.py` to emit `torch.export` and ONNX models. Fix input window length (e.g., 1 s) and sample rate for MCU.
- Quantization: integrate PTQ/QAT via torch.ao/torchao. Include a small calibration routine using representative windows.
- Clearly document I/O tensor shapes, normalization, mel parameters, and class order for the Rust MCU consumer.

Testing Guidelines
- Use pytest for unit tests covering: dataset indexing/length, transform output shapes/dtypes, a single forward pass on CPU, a tiny training step (overfit 1–2 batches), CLI argument parsing.
- No network access in tests. Use synthetic or tiny local WAV fixtures.
- Keep tests fast (<1 min full suite on CPU).

Code Style & Quality
- Type hints required in library code. mypy strict-ish on `src/noise_detect/`.
- Format with black; lint with ruff; keep imports clean (isort via ruff). Avoid large functions; prefer small, testable units.
- Avoid global state; prefer dependency injection via Hydra configs.

Typing & Type Checking
- Mandatory typing: all public functions, methods, class attributes, and module-level constants must have explicit type hints. Prefer precise types over `Any`; allow `Any` only with justification and `# type: ignore[reason]` when unavoidable.
- Primary checker: mypy in strict mode for `src/noise_detect/` (tests may be looser). Suggested mypy config: `disallow_untyped_defs = True`, `disallow_incomplete_defs = True`, `warn_return_any = True`, `warn_unused_ignores = True`, `no_implicit_optional = True`, `strict_optional = True`, `warn_redundant_casts = True`, `warn_unreachable = True`, `check_untyped_defs = True`.
- Optional checker: Pyright (fast); keep settings aligned with mypy and run locally or in CI if available.
- Ruff typing rules: enable `ANN` (annotations) and `TCH` (type-checking imports) to enforce annotations presence and move heavy typing-only imports under `if TYPE_CHECKING:`. Treat these as errors in CI.
- Modern typing features: prefer `Self`, `Literal`, `TypedDict`, `NamedTuple`, `TypeAlias`, and `Enum` where appropriate. Use `from __future__ import annotations` where needed to avoid forward-reference issues.
- Structured configs: use typed Hydra Structured Configs (dataclasses) instead of loose dicts for configuration surfaces.
- Runtime performance: guard typing-only imports with `from typing import TYPE_CHECKING` and place under `if TYPE_CHECKING:` blocks.

Security & Reproducibility
- Do not commit real or sensitive audio. Keep `runs/` and large artifacts out of git.
- Set and log seeds; record exact Hydra config + git hash; include manifest checksum in run metadata.

Interoperability with Firmware
- Do not assume availability of `std` on device; keep feature extraction parameters documented for parity with the eventual Rust inference path.
- Provide a small reference Numpy/Python function to compute the exact frontend used during training (mel params, scaling) to aid porting.

Commit Messages & PRs
- Use Conventional Commits (e.g., feat(nn): add TinyConv model).
- Keep PRs focused; include motivation, results (metrics), and how to reproduce (config overrides).
