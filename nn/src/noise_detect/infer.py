from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import Tensor

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hydra
from omegaconf import DictConfig, OmegaConf

from noise_detect.config import (
    DatasetConfig,
    InferenceConfig,
    MelConfig,
    TinyConvConfig,
    resolve_input_path,
)
from noise_detect.features.mel import compute_mel
from noise_detect.features.utils import (
    ensure_float32,
    mix_down_to_mono,
    resample_if_needed,
    seconds_to_samples,
    window_signal,
)
from noise_detect.models.tinyconv import TinyConv
from noise_detect.data.manifest import default_manifest_path, load_manifest


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _iter_wav_files(root_or_file: Path) -> Iterable[Path]:
    if root_or_file.is_file():
        yield root_or_file
        return
    for p in sorted(root_or_file.rglob("*.wav")):
        if p.is_file():
            yield p


def _load_audio(path: Path) -> tuple[Tensor, int]:
    import torchaudio  # type: ignore

    x, sr = torchaudio.load(str(path))
    x = ensure_float32(x)
    x = mix_down_to_mono(x)
    return x, sr


def _aggregate(probs_on: Tensor, mode: str) -> float:
    if probs_on.numel() == 0:
        return 0.0
    if mode == "max":
        return float(probs_on.max().item())
    return float(probs_on.mean().item())


def _load_checkpoint_if_any(model: TinyConv, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    try:
        from hydra.utils import to_absolute_path  # type: ignore

        p = Path(to_absolute_path(ckpt_path)).expanduser()
    except Exception:  # pragma: no cover
        p = Path(ckpt_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    data = torch.load(str(p), map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        state = data["state_dict"]
        # Try to strip common Lightning prefixes like "model."
        new_state = {k.split(".", 1)[-1] if k.startswith("model.") else k: v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    elif isinstance(data, dict):
        model.load_state_dict(data, strict=False)
    else:
        raise RuntimeError("Unsupported checkpoint format")


def _maybe_load_calibrated_threshold(ckpt_path: Optional[str]) -> Optional[float]:
    if not ckpt_path:
        return None
    try:
        from hydra.utils import to_absolute_path  # type: ignore

        p = Path(to_absolute_path(ckpt_path)).expanduser()
    except Exception:  # pragma: no cover
        p = Path(ckpt_path).expanduser()
    thr_file = p.parent / "threshold.json"
    if thr_file.exists():
        try:
            data = json.loads(thr_file.read_text(encoding="utf-8"))
            # Accept either {"best_threshold": ...} or {"threshold": ...}
            if isinstance(data, dict):
                if "best_threshold" in data and isinstance(data["best_threshold"], (int, float)):
                    return float(data["best_threshold"])
                if "threshold" in data and isinstance(data["threshold"], (int, float)):
                    return float(data["threshold"])
        except Exception:
            return None
    return None


def run_inference(cfg: InferenceConfig) -> int:
    dev = _device_from_name(cfg.device)

    ds_cfg = cfg.dataset
    mel_cfg = cfg.mel
    model_cfg = cfg.model

    model = TinyConv(model_cfg, n_classes=len(ds_cfg.class_names)).to(dev)
    _load_checkpoint_if_any(model, cfg.checkpoint)
    model.eval()

    window_samples, hop_samples = seconds_to_samples(
        window_s=ds_cfg.window_s, hop_s=ds_cfg.hop_s, sample_rate=ds_cfg.sample_rate
    )

    # Load calibrated threshold next to checkpoint if threshold is not provided
    threshold = cfg.threshold
    if threshold is None:
        threshold = _maybe_load_calibrated_threshold(cfg.checkpoint) or 0.5

    rows: list[dict[str, object]] = []
    collect_windows = cfg.emit_window_stats or (isinstance(cfg.window_output_csv, str) and cfg.window_output_csv)
    window_rows: list[dict[str, object]] = []

    # Determine file list: manifest or filesystem
    if cfg.use_manifest:
        mpath = None
        if ds_cfg.manifest_path:
            try:
                from hydra.utils import to_absolute_path  # type: ignore
                mpath = Path(to_absolute_path(ds_cfg.manifest_path)).expanduser().resolve()
            except Exception:  # pragma: no cover
                mpath = Path(ds_cfg.manifest_path).expanduser().resolve()
        else:
            mpath = default_manifest_path()
        if mpath is None or not mpath.exists():
            raise FileNotFoundError("Manifest not found; set dataset.manifest_path or provide one at ../recordings/manifest.jsonl")
        items = load_manifest(mpath)
        # Filter by split if requested
        if cfg.split is not None:
            items = [it for it in items if (it.split or "train") == cfg.split]
        file_list = sorted({it.audio_path for it in items})
        iter_files = file_list
    else:
        input_path = resolve_input_path(cfg)
        iter_files = list(_iter_wav_files(input_path))

    with torch.no_grad():
        for wav_path in iter_files:
            x, sr = _load_audio(wav_path)
            if sr != ds_cfg.sample_rate:
                x = resample_if_needed(x, sr, ds_cfg.sample_rate)

            windows = window_signal(x, window_samples=window_samples, hop_samples=hop_samples, pad=True)
            if windows.numel() == 0:
                continue

            probs_on_all: list[float] = []
            total_windows = windows.shape[0]
            duration_s = float(x.shape[0]) / float(ds_cfg.sample_rate)

            for start_idx in range(0, total_windows, cfg.batch_size):
                end_idx = min(start_idx + cfg.batch_size, total_windows)
                batch = windows[start_idx:end_idx]
                mel = compute_mel(batch, sample_rate=ds_cfg.sample_rate, cfg=mel_cfg)  # (B,1,M,T)
                mel = mel.to(dev)
                logits = model(mel)
                probs = torch.softmax(logits, dim=-1)  # (B, 2)
                probs_on = probs[:, 1].detach().cpu()
                for offset, prob in enumerate(probs_on):
                    win_idx = start_idx + offset
                    prob_float = float(prob)
                    probs_on_all.append(prob_float)
                    if collect_windows:
                        start_s = (win_idx * hop_samples) / float(ds_cfg.sample_rate)
                        end_s = start_s + window_samples / float(ds_cfg.sample_rate)
                        end_s = min(end_s, duration_s)
                        window_rows.append(
                            {
                                "file": str(wav_path),
                                "window_index": win_idx,
                                "start_s": start_s,
                                "end_s": end_s,
                                "prob_on": prob_float,
                            }
                        )

            probs_on_tensor = torch.tensor(probs_on_all, dtype=torch.float32)
            agg_prob = _aggregate(probs_on_tensor, cfg.aggregate)
            pred_label = ds_cfg.class_names[1] if agg_prob >= threshold else ds_cfg.class_names[0]

            rows.append(
                {
                    "file": str(wav_path),
                    "n_windows": int(len(probs_on_all)),
                    "prob_on": agg_prob,
                    "pred": pred_label,
                }
            )

            if cfg.emit_window_stats and probs_on_all:
                min_prob = float(min(probs_on_all))
                max_prob = float(max(probs_on_all))
                mean_prob = float(sum(probs_on_all) / len(probs_on_all))
                print(
                    f"{wav_path}: windows={len(probs_on_all)} prob_on(mean={mean_prob:.3f}, min={min_prob:.3f}, max={max_prob:.3f})"
                )

    # Output
    if cfg.output_csv:
        out = Path(cfg.output_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "n_windows", "prob_on", "pred"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote predictions to {out}")
    else:
        for r in rows:
            print(f"{r['file']}: prob_on={r['prob_on']:.3f} pred={r['pred']} windows={r['n_windows']}")

    if cfg.window_output_csv and collect_windows:
        out_w = Path(cfg.window_output_csv).expanduser()
        out_w.parent.mkdir(parents=True, exist_ok=True)
        with out_w.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "window_index", "start_s", "end_s", "prob_on"],
            )
            writer.writeheader()
            for row in window_rows:
                writer.writerow(row)
        print(f"Wrote window probabilities to {out_w}")

    return 0


def _to_infer_config(dc: DictConfig) -> InferenceConfig:
    # Convert Hydra DictConfig to typed dataclasses
    d = OmegaConf.to_container(dc, resolve=True)
    assert isinstance(d, dict)
    ds = DatasetConfig(**d["dataset"])  # type: ignore[arg-type]
    mel = MelConfig(**d["features"])  # type: ignore[arg-type]
    mdl = TinyConvConfig(**d["model"])  # type: ignore[arg-type]
    thr_val = d.get("threshold")
    if thr_val is None or (isinstance(thr_val, str) and thr_val.lower() == "null"):
        threshold = None
    else:
        threshold = float(thr_val)
    return InferenceConfig(
        input_path=d.get("input_path", ""),
        output_csv=d.get("output_csv"),
        checkpoint=d.get("checkpoint"),
        device=d.get("device", "auto"),
        threshold=threshold,
        aggregate=d.get("aggregate", "mean"),
        batch_size=int(d.get("batch_size", 32)),
        use_manifest=bool(d.get("use_manifest", False)),
        split=d.get("split"),
        window_output_csv=d.get("window_output_csv"),
        emit_window_stats=bool(d.get("emit_window_stats", False)),
        dataset=ds,
        mel=mel,
        model=mdl,
    )


@hydra.main(config_path="../../configs", config_name="infer", version_base=None)
def _hydra_entry(cfg: DictConfig) -> int:
    return run_inference(_to_infer_config(cfg))


def main() -> int:
    return _hydra_entry()


if __name__ == "__main__":
    raise SystemExit(main())
