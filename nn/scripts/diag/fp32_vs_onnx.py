"""Sanity check: the fp32 ONNX export must give identical predictions to
the PyTorch checkpoint on the test set. If they disagree, the export path
is broken and .espdl is suspect.

Also simulates INT8 input quantization (matches the firmware input path:
log-mel -> quantize_by_engine_exp with the model's input exponent) and
measures how many test-window decisions change vs fp32 ONNX. That gives
an upper bound estimate of input-quant-only error; internal layer quant
effects are separate but typically smaller (we saw <1.3% NSR per layer).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

NN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NN_ROOT / "src"))

from noise_detect.config import DatasetConfig, MelConfig, TinyConvConfig
from noise_detect.data.manifest import load_manifest
from noise_detect.data.mel_cache import cache_path_for, ensure_mel_cache, load_mel_cache, SPLIT_CODE
from noise_detect.models.tinyconv import TinyConv

CKPT = Path(sys.argv[1])
ONNX = Path(sys.argv[2])
MANIFEST = Path(sys.argv[3])
CHANNELS = [int(c) for c in sys.argv[4].split(",")] if len(sys.argv) > 4 else [24, 48, 96]
DROPOUT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
INPUT_EXP = int(sys.argv[6]) if len(sys.argv) > 6 else -5  # from model.info exponents: [-5]

ds_cfg = DatasetConfig(sample_rate=32000, window_s=1.0, hop_s=0.5,
                       class_names=["pump_off", "pump_on"], manifest_path=str(MANIFEST))
mel_cfg = MelConfig()
mdl_cfg = TinyConvConfig(channels=CHANNELS, dropout=DROPOUT)

items = load_manifest(MANIFEST)
cache_p = ensure_mel_cache(items, ds_cfg, mel_cfg, MANIFEST, cache_dir=None)
cache = load_mel_cache(cache_p)
mels: torch.Tensor = cache["mels"]
labels = cache["labels"].numpy()
splits = cache["splits"].numpy()
test_mask = splits == SPLIT_CODE["test"]
print(f"Test windows: {test_mask.sum()}")

# --- PyTorch checkpoint ---
model = TinyConv(mdl_cfg, n_classes=2)
state = torch.load(str(CKPT), map_location="cpu")
sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
sd = {k.split(".", 1)[-1] if k.startswith("model.") else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()

# --- Run PyTorch on test ---
test_mels = mels[test_mask].float()
pt_probs = np.zeros(len(test_mels), dtype=np.float32)
with torch.inference_mode():
    for i in range(0, len(test_mels), 1024):
        x = test_mels[i:i+1024]
        logits = model(x)
        pt_probs[i:i+1024] = torch.softmax(logits, dim=-1)[:, 1].numpy()

# --- Run fp32 ONNX on test ---
import onnxruntime as ort  # type: ignore
sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
onnx_probs = np.zeros(len(test_mels), dtype=np.float32)
# ONNX was exported with fixed batch=1; run window-by-window (fast enough for 20k).
test_np_all = test_mels.numpy()
for i in range(len(test_mels)):
    out = sess.run(None, {in_name: test_np_all[i:i+1]})[0]
    e = np.exp(out - out.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    onnx_probs[i] = p[0, 1]
    if i % 2000 == 0 and i > 0:
        print(f"  onnx: {i}/{len(test_mels)}")

# --- Simulate INT8 input quantization (matches firmware quantize_by_engine_exp) ---
scale = 2.0 ** INPUT_EXP
test_np = test_mels.numpy()
quantized = np.clip(np.round(test_np / scale), -128, 127)
dequantized = quantized * scale
test_q = torch.from_numpy(dequantized.astype(np.float32))
q_probs = np.zeros(len(test_q), dtype=np.float32)
with torch.inference_mode():
    for i in range(0, len(test_q), 1024):
        x = test_q[i:i+1024]
        logits = model(x)
        q_probs[i:i+1024] = torch.softmax(logits, dim=-1)[:, 1].numpy()

y = labels[test_mask]
thr = 0.47

def report(name, p):
    yhat = (p >= thr).astype(np.int64)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n
    print(f"  {name:<30} acc={acc:.4f}  FP={fp:3d}  FN={fn:3d}  TP={tp} TN={tn}")

print("\n--- Test-set metrics at threshold 0.47 ---")
report("PyTorch fp32 ckpt", pt_probs)
report("ONNX fp32 export", onnx_probs)
report("fp32 ckpt + INT8 input sim", q_probs)

# Disagreement counts
pt_dec = pt_probs >= thr
onnx_dec = onnx_probs >= thr
q_dec = q_probs >= thr
print("\n--- Decision disagreement vs PyTorch fp32 ---")
print(f"  ONNX fp32      disagrees on {int(np.sum(pt_dec != onnx_dec)):>4} windows / {len(pt_dec)}")
print(f"  INT8-input sim disagrees on {int(np.sum(pt_dec != q_dec)):>4} windows / {len(pt_dec)}")

# Probability-level drift
print("\n--- Probability drift (mean abs delta) ---")
print(f"  ONNX fp32  vs PyTorch: {np.mean(np.abs(pt_probs - onnx_probs)):.6f}")
print(f"  INT8-input vs PyTorch: {np.mean(np.abs(pt_probs - q_probs)):.6f}")

# Distribution of INT8 impact by confidence
print("\n--- INT8-input impact stratified by fp32 confidence ---")
for lo, hi in [(0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.0)]:
    in_band = (pt_probs >= lo) & (pt_probs < hi)
    if in_band.sum() == 0:
        continue
    disagree = int(np.sum((pt_dec != q_dec) & in_band))
    print(f"  pt_prob [{lo:.2f},{hi:.2f}) n={in_band.sum():>5}  flipped={disagree:>3}  ({disagree/max(in_band.sum(),1)*100:.1f}%)")
