"""Recompute test metrics stratified by filename-suffix label-quality flags.

We split test files into three buckets:
  clean    — normal files (no _canceled/_timeout/_undefined suffix)
  canceled — filenames containing _canceled
  timeout  — filenames containing _timeout
  other    — _undefined etc
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
from collections import defaultdict

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/diag/test_window_preds.csv")
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.52

def bucket(name: str) -> str:
    n = name.lower()
    if "_canceled" in n:
        return "canceled"
    if "_timeout" in n:
        return "timeout"
    if "_undefined" in n:
        return "undefined"
    return "clean"

counts = defaultdict(lambda: {"n": 0, "tp":0, "tn":0, "fp":0, "fn":0})

with CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        b = bucket(Path(row["file"]).name)
        y = int(row["label"])
        p = float(row["prob"])
        pred = 1 if p >= THRESHOLD else 0
        c = counts[b]
        c["n"] += 1
        if pred == 1 and y == 1: c["tp"] += 1
        elif pred == 0 and y == 0: c["tn"] += 1
        elif pred == 1 and y == 0: c["fp"] += 1
        elif pred == 0 and y == 1: c["fn"] += 1

print(f"{'bucket':<10} {'n':>6} {'acc':>7} {'FPR':>7} {'FNR':>7}  TP TN FP FN")
for b, c in sorted(counts.items()):
    n = c["n"] or 1
    acc = (c["tp"] + c["tn"]) / n
    fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
    fnr = c["fn"] / max(c["fn"] + c["tp"], 1)
    print(
        f"{b:<10} {c['n']:>6} {acc:>7.4f} {fpr:>7.4f} {fnr:>7.4f}  "
        f"{c['tp']} {c['tn']} {c['fp']} {c['fn']}"
    )
