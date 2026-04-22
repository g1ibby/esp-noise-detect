from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import hydra
from omegaconf import DictConfig, OmegaConf


_AUDIT_SESSION_RE = re.compile(r"_(\d{10})_")


def _audit_key_of(path: str) -> tuple[Optional[int], Optional[str]]:
    """Extract (session_ts, day) from a filename like
    xiao_esp32s3_<ts>_c000_off_chunk000.wav. Returns (None, None) if the
    filename doesn't match — such windows are simply skipped in the audit."""
    name = Path(path).name
    m = _AUDIT_SESSION_RE.search(name)
    if not m:
        return None, None
    ts = int(m.group(1))
    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return ts, day


def _error_audit(
    files: list[str],
    probs: list[float],
    labels: list[int],
    threshold: float,
    top_n: int = 10,
) -> None:
    """Group window-level predictions by session and day; print accuracy per
    group (worst-first) and the top-N worst files by FP and FN counts.

    Meant for debugging regressions — helps see whether a bad run's errors
    cluster on one hard session or are spread across the whole split.
    """
    if not files:
        print("[audit] no window data available (set window_metrics: true)")
        return

    per_session_tp: Dict[int, int] = defaultdict(int)
    per_session_tn: Dict[int, int] = defaultdict(int)
    per_session_fp: Dict[int, int] = defaultdict(int)
    per_session_fn: Dict[int, int] = defaultdict(int)
    per_day_tp: Dict[str, int] = defaultdict(int)
    per_day_tn: Dict[str, int] = defaultdict(int)
    per_day_fp: Dict[str, int] = defaultdict(int)
    per_day_fn: Dict[str, int] = defaultdict(int)
    per_file_fp: Dict[str, int] = defaultdict(int)
    per_file_fn: Dict[str, int] = defaultdict(int)
    skipped_no_ts = 0

    for f, p, y in zip(files, probs, labels):
        pred = 1 if p >= threshold else 0
        ts, day = _audit_key_of(f)
        if ts is None or day is None:
            skipped_no_ts += 1
            continue
        if pred == 1 and y == 1:
            per_session_tp[ts] += 1
            per_day_tp[day] += 1
        elif pred == 0 and y == 0:
            per_session_tn[ts] += 1
            per_day_tn[day] += 1
        elif pred == 1 and y == 0:
            per_session_fp[ts] += 1
            per_day_fp[day] += 1
            per_file_fp[f] += 1
        else:  # pred == 0 and y == 1
            per_session_fn[ts] += 1
            per_day_fn[day] += 1
            per_file_fn[f] += 1

    print("")
    print("=" * 72)
    print(f"Error audit (threshold={threshold:.3f})")
    if skipped_no_ts:
        print(f"  skipped {skipped_no_ts} windows whose filename has no session timestamp")
    print("=" * 72)

    def _group_rows(keys, tp, tn, fp, fn) -> list[tuple]:
        rows = []
        for k in keys:
            n = tp[k] + tn[k] + fp[k] + fn[k]
            acc = (tp[k] + tn[k]) / max(n, 1)
            rows.append((k, n, acc, tp[k], tn[k], fp[k], fn[k]))
        # worst-accuracy first; ties broken by group size descending
        rows.sort(key=lambda r: (r[2], -r[1]))
        return rows

    session_keys = set(per_session_tp) | set(per_session_tn) | set(per_session_fp) | set(per_session_fn)
    day_keys = set(per_day_tp) | set(per_day_tn) | set(per_day_fp) | set(per_day_fn)

    print("\nBy day (worst-first):")
    print(f"  {'day':<12} {'n':>5} {'acc':>7}  TP TN FP FN")
    for k, n, acc, tp_, tn_, fp_, fn_ in _group_rows(day_keys, per_day_tp, per_day_tn, per_day_fp, per_day_fn):
        print(f"  {k:<12} {n:>5} {acc:>7.4f}  {tp_} {tn_} {fp_} {fn_}")

    print("\nBy session (worst-first, up to 15):")
    print(f"  {'session_ts':>12} {'n':>5} {'acc':>7}  TP TN FP FN")
    for k, n, acc, tp_, tn_, fp_, fn_ in _group_rows(session_keys, per_session_tp, per_session_tn, per_session_fp, per_session_fn)[:15]:
        print(f"  {k:>12} {n:>5} {acc:>7.4f}  {tp_} {tn_} {fp_} {fn_}")

    if per_file_fp:
        print(f"\nTop {top_n} files by FP count:")
        for f, c in sorted(per_file_fp.items(), key=lambda kv: -kv[1])[:top_n]:
            print(f"  {c:>4}  {Path(f).name}")
    if per_file_fn:
        print(f"\nTop {top_n} files by FN count:")
        for f, c in sorted(per_file_fn.items(), key=lambda kv: -kv[1])[:top_n]:
            print(f"  {c:>4}  {Path(f).name}")
    print("=" * 72)

from noise_detect.config import (
    DatasetConfig,
    MelConfig,
    TinyConvConfig,
    DataModuleConfig,
    load_augmentation_config,
)
from noise_detect.data.datamodule import PumpAudioDataModule
from noise_detect.features.mel import compute_mel
from noise_detect.models.tinyconv import TinyConv


def _device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_abs(p: str) -> Path:
    from hydra.utils import to_absolute_path

    return Path(to_absolute_path(p)).expanduser().resolve()


def _load_checkpoint(model: TinyConv, ckpt_path: str) -> None:
    p = _to_abs(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    data = torch.load(str(p), map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        state = data["state_dict"]
        new_state = {k.split(".", 1)[-1] if k.startswith("model.") else k: v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    elif isinstance(data, dict):
        model.load_state_dict(data, strict=False)
    else:
        raise RuntimeError("Unsupported checkpoint format")


def _aggregate_probs(probs: torch.Tensor, mode: str) -> float:
    if probs.numel() == 0:
        return 0.0
    if mode == "max":
        return float(probs.max().item())
    return float(probs.mean().item())


def _metrics(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).to(torch.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    # Per-class F1
    def f1(tp: int, fp: int, fn: int) -> float:
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    f1_pos = f1(tp, fp, fn)
    f1_neg = f1(tn, fn, fp)
    f1_macro = 0.5 * (f1_pos + f1_neg)

    # Balanced accuracy
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (tpr + tnr)

    # AUROC/AUPRC via torchmetrics.functional
    try:
        from torchmetrics.functional.classification import binary_auroc, binary_average_precision

        auroc = float(binary_auroc(y_prob, y_true).item())
        auprc = float(binary_average_precision(y_prob, y_true).item())
    except Exception:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(f1_macro),
        "f1_pos": float(f1_pos),
        "f1_neg": float(f1_neg),
        "auroc": auroc,
        "auprc": auprc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def _calibrate_threshold(y_true: torch.Tensor, y_prob: torch.Tensor, mode: str = "macro_f1") -> float:
    best_t = 0.5
    best_v = -1.0
    # Coarse-to-fine search
    for step in [0.01, 0.002, 0.001]:
        start = max(best_t - 0.05, 0.0)
        end = min(best_t + 0.05, 1.0)
        ts = torch.arange(start, end + 1e-9, step)
        for t in ts:
            m = _metrics(y_true, y_prob, float(t))
            v = m.get(mode, float("nan"))
            if v == v and v > best_v:  # check for NaN
                best_v = v
                best_t = float(t)
    return float(best_t)


def evaluate(cfg: DictConfig) -> int:
    # Resolve configs
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    ds = DatasetConfig(**d["dataset"])  # type: ignore[arg-type]
    mel = MelConfig(**d["features"])  # type: ignore[arg-type]
    mdl_cfg = TinyConvConfig(**d["model"])  # type: ignore[arg-type]
    dm_cfg = DataModuleConfig(**d.get("dm", {}))  # type: ignore[arg-type]
    augment_cfg = load_augmentation_config(d.get("augment"))
    eval_seed = int(d.get("seed", 42))
    window_metrics = bool(d.get("window_metrics", False))
    window_preds_csv = d.get("window_preds_csv")
    audit = bool(d.get("audit", False))
    audit_top_n = int(d.get("audit_top_n", 10))

    split = str(d.get("split", "val"))
    aggregate = str(d.get("aggregate", "mean"))
    ckpt = d.get("checkpoint")
    if not isinstance(ckpt, str) or not ckpt:
        raise ValueError("Please provide a checkpoint=... path")

    dev = _device_auto()
    model = TinyConv(mdl_cfg, n_classes=len(ds.class_names)).to(dev)
    _load_checkpoint(model, ckpt)
    model.eval()

    ckpt_dir = _to_abs(ckpt).parent

    dm = PumpAudioDataModule(ds, dm_cfg, augment_cfg, seed=eval_seed, mel_cfg=mel)
    dm.setup()
    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

    # Accumulate per-file probabilities
    probs_by_file: dict[str, list[float]] = {}
    label_by_file: dict[str, int] = {}
    collect_windows = (
        window_metrics
        or audit
        or (isinstance(window_preds_csv, str) and window_preds_csv)
    )
    window_probs: list[float] = []
    window_labels: list[int] = []
    window_files: list[str] = []
    window_starts: list[float] = []
    window_ends: list[float] = []

    with torch.no_grad():
        for batch in loader:  # type: ignore[assignment]
            labels = batch["label"]  # (B,)
            files: list[str] = batch["file"]  # type: ignore[assignment]
            starts = batch.get("start_s")
            ends = batch.get("end_s")
            if "mel" in batch:
                mel_batch = batch["mel"].to(dev)
            else:
                wave = batch["waveform"]  # (B, T) CPU
                mel_batch = compute_mel(wave, sample_rate=ds.sample_rate, cfg=mel).to(dev)
            logits = model(mel_batch)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
            for idx, (f, p, y) in enumerate(zip(files, probs, labels)):
                probs_by_file.setdefault(f, []).append(float(p))
                label_by_file[f] = int(y)
                if collect_windows:
                    window_probs.append(float(p))
                    window_labels.append(int(y))
                    window_files.append(f)
                    if isinstance(starts, torch.Tensor):
                        window_starts.append(float(starts[idx].item()))
                    else:
                        window_starts.append(float(idx))
                    if isinstance(ends, torch.Tensor):
                        window_ends.append(float(ends[idx].item()))
                    else:
                        window_ends.append(float(idx))

    # Aggregate to file-level
    files_sorted = sorted(probs_by_file.keys())
    y_prob = torch.tensor([_aggregate_probs(torch.tensor(probs_by_file[f]), aggregate) for f in files_sorted])
    y_true = torch.tensor([label_by_file[f] for f in files_sorted], dtype=torch.long)

    raw_threshold = d.get("threshold")
    if raw_threshold is None or (isinstance(raw_threshold, str) and raw_threshold.lower() == "null"):
        threshold: float | None = None
    else:
        threshold = float(raw_threshold)

    cal = bool(d.get("calibrate", True))

    if threshold is None and not cal:
        thr_path = ckpt_dir / "threshold.json"
        if thr_path.exists():
            try:
                data = json.loads(thr_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if "best_threshold" in data and isinstance(data["best_threshold"], (int, float)):
                        threshold = float(data["best_threshold"])
                    elif "threshold" in data and isinstance(data["threshold"], (int, float)):
                        threshold = float(data["threshold"])
            except Exception:
                threshold = None

    if threshold is None and cal:
        # threshold currently None -> calibrate
        best_t = _calibrate_threshold(y_true, y_prob, mode="macro_f1")
        calibration = {
            "best_threshold": best_t,
            "split": split,
            "aggregate": aggregate,
            "num_files": len(files_sorted),
        }
        cal_path = Path(d.get("calibration_json", "calibration.json"))
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with cal_path.open("w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        threshold = best_t
        print(f"Calibrated best_threshold={best_t:.3f} -> wrote {cal_path}")
        # Also write next to checkpoint for reuse in infer/export
        thr_path = ckpt_dir / "threshold.json"
        with thr_path.open("w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        print(f"Wrote calibrated threshold next to checkpoint: {thr_path}")

    if threshold is None:
        threshold = 0.5

    metrics_file = _metrics(y_true, y_prob, float(threshold))

    result_payload: dict[str, Any] = {"file": metrics_file}

    if window_metrics and window_probs:
        y_true_win = torch.tensor(window_labels, dtype=torch.long)
        y_prob_win = torch.tensor(window_probs, dtype=torch.float32)
        metrics_window = _metrics(y_true_win, y_prob_win, float(threshold))
        result_payload["window"] = metrics_window
        print("Window metrics: " + json.dumps(metrics_window, indent=2))

    metrics_path = Path(d.get("metrics_json", "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print("File metrics: " + json.dumps(metrics_file, indent=2))
    print(f"Wrote metrics to {metrics_path.resolve()}")

    # Optional CSV
    preds_csv = d.get("preds_csv")
    if isinstance(preds_csv, str) and preds_csv:
        rows = []
        for f, prob, y in zip(files_sorted, y_prob.tolist(), y_true.tolist()):
            pred = 1 if prob >= float(threshold) else 0
            rows.append({"file": f, "prob_on": prob, "label": y, "pred": pred})
        out = Path(preds_csv)
        with out.open("w", newline="", encoding="utf-8") as fw:
            w = csv.DictWriter(fw, fieldnames=["file", "prob_on", "label", "pred"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote predictions CSV to {out}")

    if isinstance(window_preds_csv, str) and window_preds_csv and collect_windows:
        out_w = Path(window_preds_csv)
        out_w.parent.mkdir(parents=True, exist_ok=True)
        with out_w.open("w", newline="", encoding="utf-8") as fw:
            w = csv.DictWriter(
                fw,
                fieldnames=["file", "start_s", "end_s", "prob_on", "label", "pred"],
            )
            w.writeheader()
            for f, s, e, prob, y in zip(
                window_files,
                window_starts,
                window_ends,
                window_probs,
                window_labels,
            ):
                pred = 1 if prob >= float(threshold) else 0
                w.writerow(
                    {
                        "file": f,
                        "start_s": s,
                        "end_s": e,
                        "prob_on": prob,
                        "label": y,
                        "pred": pred,
                    }
                )
        print(f"Wrote window predictions CSV to {out_w}")

    # Opt-in per-session / per-day error audit. Requires window-level data,
    # which we already have when audit or window_metrics was set (see
    # `collect_windows` above).
    if audit:
        if not window_files:
            print("[audit] no window data collected — enable audit/window_metrics")
        else:
            _error_audit(
                window_files,
                window_probs,
                window_labels,
                float(threshold),
                top_n=audit_top_n,
            )

    # Final summary block
    print("")
    print("=" * 72)
    print(f"Evaluation complete (split={split})")
    print("=" * 72)
    print(f"  Checkpoint       : {_to_abs(ckpt)}")
    print(f"  Metrics JSON     : {metrics_path.resolve()}")
    thr_file = ckpt_dir / "threshold.json"
    if thr_file.exists():
        print(f"  Threshold JSON   : {thr_file}")
    cal_file = ckpt_dir / "calibration.json"
    if cal_file.exists():
        print(f"  Calibration JSON : {cal_file}")
    if isinstance(preds_csv, str) and preds_csv:
        print(f"  File preds CSV   : {Path(preds_csv).resolve()}")
    if isinstance(window_preds_csv, str) and window_preds_csv:
        print(f"  Window preds CSV : {Path(window_preds_csv).resolve()}")
    print("=" * 72)

    return 0


@hydra.main(config_path="../../configs", config_name="eval", version_base=None)
def _hydra_entry(cfg: DictConfig) -> int:
    return evaluate(cfg)


def main() -> int:
    return _hydra_entry()


if __name__ == "__main__":
    raise SystemExit(main())
