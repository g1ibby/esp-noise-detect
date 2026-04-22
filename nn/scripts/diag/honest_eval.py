"""Run multiple diagnostic evals against the existing checkpoint:

A. Standard window metrics on current test split (sanity-check).
B. Session-disjoint test: only test windows whose session_ts does NOT also
   appear in the train split (honest, but small).
C. Day-disjoint test: only test windows whose day is uniquely test-only.
D. Per-energy-quartile metrics (uses the pre-normalization log-mel mean
   recovered from the cached mel by inverse-z-score is impossible since cache
   stores normalized mels; instead, we read raw waveforms for the test files
   and compute energy directly).
E. FP audit: file/start_s/day/session for every false positive.

Run: uv run python scripts/diag/honest_eval.py <checkpoint>
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

NN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NN_ROOT / "src"))

from noise_detect.config import DatasetConfig, MelConfig, TinyConvConfig
from noise_detect.data.manifest import load_manifest
from noise_detect.data.mel_cache import (
    cache_path_for,
    ensure_mel_cache,
    load_mel_cache,
    SPLIT_CODE,
)
from noise_detect.models.tinyconv import TinyConv

if len(sys.argv) < 3:
    print("Usage: honest_eval.py <checkpoint> <manifest> [threshold] [channels] [dropout]", file=sys.stderr)
    sys.exit(1)
CKPT = Path(sys.argv[1])
MANIFEST = Path(sys.argv[2])
THRESHOLD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.52
CHANNELS = [int(c) for c in sys.argv[4].split(",")] if len(sys.argv) > 4 else [16, 32, 64]
DROPOUT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

TS_RE = re.compile(r"xiao_esp32s3_(\d{10})_")

ds_cfg = DatasetConfig(
    sample_rate=32000,
    window_s=1.0,
    hop_s=0.5,
    class_names=["pump_off", "pump_on"],
    manifest_path=str(MANIFEST),
)
mel_cfg = MelConfig()  # defaults match the saved hparams
mdl_cfg = TinyConvConfig(channels=CHANNELS, dropout=DROPOUT)


def load_model() -> TinyConv:
    model = TinyConv(mdl_cfg, n_classes=2)
    state = torch.load(str(CKPT), map_location="cpu")
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    sd = {k.split(".", 1)[-1] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    if len(y_true) == 0:
        print(f"--- {name}: EMPTY ---")
        return {}
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / max(n, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    print(
        f"--- {name} (n={n}) ---  "
        f"acc={acc:.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}  "
        f"TP={tp} TN={tn} FP={fp} FN={fn}"
    )
    return {"name": name, "n": n, "acc": acc, "fpr": fpr, "fnr": fnr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def session_ts(name: str) -> Optional[int]:
    m = TS_RE.search(name)
    return int(m.group(1)) if m else None


def session_day(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


# --- Step 1: load mel cache and model (build cache if missing) ---
manifest_items = load_manifest(MANIFEST)
cache_p = ensure_mel_cache(manifest_items, ds_cfg, mel_cfg, MANIFEST, cache_dir=None)
print(f"Loading mel cache: {cache_p}")
cache = load_mel_cache(cache_p)
model = load_model()

mels: torch.Tensor = cache["mels"]            # (N, 1, M, T) fp16
labels: torch.Tensor = cache["labels"]         # (N,) int
splits: torch.Tensor = cache["splits"]         # (N,) int8
file_idx: torch.Tensor = cache["file_idx"]     # (N,) int32
file_list: list[str] = cache["file_list"]
starts: torch.Tensor = cache["starts"]         # (N,) float

print(f"Cache: {tuple(mels.shape)}  splits "
      f"train={int((splits == SPLIT_CODE['train']).sum())} "
      f"val={int((splits == SPLIT_CODE['val']).sum())} "
      f"test={int((splits == SPLIT_CODE['test']).sum())}")

# Build per-window metadata
file_idx_np = file_idx.numpy()
ts_per_window = np.array(
    [session_ts(Path(file_list[i]).name) or 0 for i in file_idx_np],
    dtype=np.int64,
)
day_per_window = np.array(
    [session_day(int(t)) if t else "" for t in ts_per_window]
)
splits_np = splits.numpy()
labels_np = labels.numpy()

# --- Step 2: run inference on entire cache (need predictions for all windows) ---
print("Running inference on full cache...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
all_probs = np.zeros(len(mels), dtype=np.float32)
batch = 1024
with torch.inference_mode():
    for i in range(0, len(mels), batch):
        x = mels[i : i + batch].to(device).float()
        logits = model(x)
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs[i : i + batch] = p

# --- A. Standard test metrics (matches eval.py output) ---
test_mask = splits_np == SPLIT_CODE["test"]
metrics("A. Standard test (window)", labels_np[test_mask], all_probs[test_mask], THRESHOLD)

# --- B. Session-disjoint test: test windows whose ts is NOT in train ---
train_ts = set(ts_per_window[splits_np == SPLIT_CODE["train"]].tolist())
honest_mask = test_mask & np.array([t not in train_ts for t in ts_per_window])
metrics(
    "B. Test \\ train sessions",
    labels_np[honest_mask], all_probs[honest_mask], THRESHOLD,
)

# --- C. Day-disjoint test: test windows whose day appears only in test ---
train_days = set(day_per_window[splits_np == SPLIT_CODE["train"]].tolist())
val_days = set(day_per_window[splits_np == SPLIT_CODE["val"]].tolist())
honest_day_mask = test_mask & np.array(
    [d not in train_days and d not in val_days for d in day_per_window]
)
metrics(
    "C. Test \\ (train+val) days",
    labels_np[honest_day_mask], all_probs[honest_day_mask], THRESHOLD,
)

# --- D. Energy stratification: read raw waveforms for test files and compute RMS ---
print("\nComputing per-window energy from raw waveforms (test split only)...")
import torchaudio  # type: ignore

window_samples = int(ds_cfg.window_s * ds_cfg.sample_rate)
hop_samples = int(ds_cfg.hop_s * ds_cfg.sample_rate)

# Group windows by file_idx
from collections import defaultdict as _dd
file_windows: dict[int, list[int]] = _dd(list)
for win_i in np.where(test_mask)[0]:
    file_windows[int(file_idx_np[win_i])].append(int(win_i))

energies = np.full(len(mels), np.nan, dtype=np.float32)
manifest_dir = MANIFEST.parent
n_files_done = 0
for fi, win_idxs in file_windows.items():
    fpath = file_list[fi]
    p = (manifest_dir / fpath) if not Path(fpath).is_absolute() else Path(fpath)
    try:
        wav, sr = torchaudio.load(str(p))
    except Exception as e:
        print(f"  skip {p}: {e}")
        continue
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[0].numpy().astype(np.float32)
    if sr != ds_cfg.sample_rate:
        # Skip resample — none of these files need it (all 32kHz)
        pass
    for win_i in win_idxs:
        s = int(round(float(starts[win_i].item()) * ds_cfg.sample_rate))
        e = s + window_samples
        seg = wav[s:e]
        if len(seg) < window_samples:
            tmp = np.zeros(window_samples, dtype=np.float32)
            tmp[: len(seg)] = seg
            seg = tmp
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        energies[win_i] = 20 * np.log10(rms + 1e-12)
    n_files_done += 1
    if n_files_done % 50 == 0:
        print(f"  energy: {n_files_done} files done")

valid_mask = test_mask & ~np.isnan(energies)
e_test = energies[valid_mask]
y_test = labels_np[valid_mask]
p_test = all_probs[valid_mask]
print(f"\nEnergy distribution (test windows): "
      f"min={e_test.min():.1f}dB  median={np.median(e_test):.1f}dB  max={e_test.max():.1f}dB")

# Quartile bins
qs = np.quantile(e_test, [0, 0.25, 0.5, 0.75, 1.0])
print(f"Quartile cuts (dBFS): {qs.round(1)}")
for i in range(4):
    lo, hi = qs[i], qs[i + 1]
    if i < 3:
        bin_mask = (e_test >= lo) & (e_test < hi)
    else:
        bin_mask = (e_test >= lo) & (e_test <= hi)
    metrics(
        f"D.Q{i+1} energy [{lo:.1f},{hi:.1f}]",
        y_test[bin_mask], p_test[bin_mask], THRESHOLD,
    )

# Stratify by label too (FP rate by energy on negatives)
neg_mask = y_test == 0
print("\nFP rate on NEGATIVES, by energy quartile:")
e_neg = e_test[neg_mask]
p_neg = p_test[neg_mask]
qs_neg = np.quantile(e_neg, [0, 0.25, 0.5, 0.75, 1.0])
for i in range(4):
    lo, hi = qs_neg[i], qs_neg[i + 1]
    if i < 3:
        bm = (e_neg >= lo) & (e_neg < hi)
    else:
        bm = (e_neg >= lo) & (e_neg <= hi)
    if bm.sum() == 0:
        continue
    fp_count = int((p_neg[bm] >= THRESHOLD).sum())
    print(f"  Q{i+1} energy [{lo:.1f},{hi:.1f}] dBFS  n={int(bm.sum()):>4}  FP={fp_count:>3}  FPR={fp_count/max(int(bm.sum()),1):.3f}")

# --- E. FP audit (cluster by file/session/day) ---
print("\n=== FP audit (test) ===")
fp_idx = np.where(test_mask & (labels_np == 0) & (all_probs >= THRESHOLD))[0]
print(f"Total FPs: {len(fp_idx)}")
fp_file_count = defaultdict(int)
fp_session_count = defaultdict(int)
fp_day_count = defaultdict(int)
for i in fp_idx:
    fp_file_count[file_list[file_idx_np[i]]] += 1
    fp_session_count[int(ts_per_window[i])] += 1
    fp_day_count[day_per_window[i]] += 1

print(f"FPs span {len(fp_file_count)} files, {len(fp_session_count)} sessions, {len(fp_day_count)} days")
print("Top files by FP count:")
for f, c in sorted(fp_file_count.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {c:>3}  {f}")

print("FPs by day:")
for d, c in sorted(fp_day_count.items()):
    print(f"  {d}  {c}")

# Save artifacts
out_dir = NN_ROOT / "runs" / "diag"
out_dir.mkdir(parents=True, exist_ok=True)

# FN audit too
fn_idx = np.where(test_mask & (labels_np == 1) & (all_probs < THRESHOLD))[0]
print(f"\nTotal FNs: {len(fn_idx)}")
fn_file_count = defaultdict(int)
for i in fn_idx:
    fn_file_count[file_list[file_idx_np[i]]] += 1
print(f"FNs span {len(fn_file_count)} files. Top files:")
for f, c in sorted(fn_file_count.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {c:>3}  {f}")

# Dump per-window predictions for the test split
out_csv = out_dir / "test_window_preds.csv"
with out_csv.open("w") as f:
    f.write("file,start_s,label,prob,pred,energy_dbfs,session_ts,day,split\n")
    for i in np.where(splits_np == SPLIT_CODE["test"])[0]:
        f.write(
            f"{file_list[file_idx_np[i]]},{float(starts[i]):.3f},"
            f"{int(labels_np[i])},{float(all_probs[i]):.4f},"
            f"{int(all_probs[i] >= THRESHOLD)},"
            f"{float(energies[i]) if not np.isnan(energies[i]) else ''},"
            f"{int(ts_per_window[i])},{day_per_window[i]},test\n"
        )
print(f"\nWrote per-window predictions -> {out_csv}")
