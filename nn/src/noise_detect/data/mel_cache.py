from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

from noise_detect.config import DatasetConfig, MelConfig
from noise_detect.data.manifest import ManifestItem, manifest_checksum
from noise_detect.features.mel import MelExtractor
from noise_detect.features.utils import (
    ensure_float32,
    mix_down_to_mono,
    resample_if_needed,
    seconds_to_samples,
)


CACHE_VERSION = 1
SPLIT_CODE: dict[str, int] = {"train": 0, "val": 1, "test": 2}


def _cfg_payload(ds_cfg: DatasetConfig, mel_cfg: MelConfig) -> dict[str, Any]:
    ds = asdict(ds_cfg)
    ds.pop("manifest_path", None)  # cache is keyed on content, not path
    return {"version": CACHE_VERSION, "ds": ds, "mel": asdict(mel_cfg)}


def _cfg_hash(ds_cfg: DatasetConfig, mel_cfg: MelConfig, manifest_sha: str) -> str:
    payload = _cfg_payload(ds_cfg, mel_cfg)
    payload["manifest_sha"] = manifest_sha
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def default_cache_dir(manifest_path: Path) -> Path:
    return manifest_path.parent / ".mel_cache"


def cache_path_for(
    manifest_path: Path,
    ds_cfg: DatasetConfig,
    mel_cfg: MelConfig,
    cache_dir: Optional[Path] = None,
) -> Path:
    sha = manifest_checksum(manifest_path)
    h = _cfg_hash(ds_cfg, mel_cfg, sha)
    base = cache_dir if cache_dir is not None else default_cache_dir(manifest_path)
    return base / f"mel_cache_{h}.pt"


def _build_cache_tensors(
    items: list[ManifestItem],
    ds_cfg: DatasetConfig,
    mel_cfg: MelConfig,
) -> dict[str, Any]:
    import torchaudio  # type: ignore

    window_samples, hop_samples = seconds_to_samples(
        ds_cfg.window_s, ds_cfg.hop_s, ds_cfg.sample_rate
    )
    class_to_idx = {name: i for i, name in enumerate(ds_cfg.class_names)}

    extractor = MelExtractor(mel_cfg, ds_cfg.sample_rate).eval()

    mel_chunks: list[Tensor] = []
    labels: list[int] = []
    splits: list[int] = []
    file_idx_list: list[int] = []
    starts: list[float] = []
    ends: list[float] = []

    file_list: list[str] = []
    seen_paths: dict[str, int] = {}

    started = time.time()
    n_items = len(items)
    skipped = 0

    with torch.inference_mode():
        for file_i, it in enumerate(items):
            p_str = str(it.audio_path)
            if p_str not in seen_paths:
                seen_paths[p_str] = len(file_list)
                file_list.append(p_str)
            fi = seen_paths[p_str]

            try:
                x, sr = torchaudio.load(p_str)
            except Exception as exc:  # unreadable files are rare; skip
                print(f"[mel-cache] skip {p_str}: {exc}")
                skipped += 1
                continue

            x = ensure_float32(x)
            x = mix_down_to_mono(x)
            if int(sr) != ds_cfg.sample_rate:
                x = resample_if_needed(x, int(sr), ds_cfg.sample_rate)

            total = x.shape[0]
            if total >= window_samples:
                remainder = total - window_samples
                n_windows = 1 + remainder // hop_samples
                if remainder % hop_samples != 0:
                    n_windows += 1
            else:
                n_windows = 1

            windows = x.new_zeros((n_windows, window_samples))
            for w in range(n_windows):
                start = w * hop_samples
                end = start + window_samples
                if end <= total:
                    windows[w] = x[start:end]
                else:
                    part = x[start:]
                    windows[w, : part.shape[0]] = part
                starts.append(float(start / ds_cfg.sample_rate))
                ends.append(float(min(end, total) / ds_cfg.sample_rate))

            mel = extractor(windows)  # (n_windows, 1, n_mels, n_frames)
            mel_chunks.append(mel.to(torch.float16).contiguous())

            label_idx = class_to_idx.get(it.label, 0)
            split_idx = SPLIT_CODE.get(it.split or "train", 0)
            labels.extend([label_idx] * n_windows)
            splits.extend([split_idx] * n_windows)
            file_idx_list.extend([fi] * n_windows)

            if (file_i + 1) % 20 == 0 or file_i + 1 == n_items:
                elapsed = time.time() - started
                total_windows = sum(c.shape[0] for c in mel_chunks)
                print(
                    f"[mel-cache] {file_i + 1}/{n_items} files "
                    f"windows={total_windows} elapsed={elapsed:.1f}s"
                )

    if not mel_chunks:
        raise RuntimeError("Mel cache build produced no windows")

    mels = torch.cat(mel_chunks, dim=0)
    if skipped:
        print(f"[mel-cache] skipped {skipped} unreadable files")

    return {
        "mels": mels,
        "labels": torch.tensor(labels, dtype=torch.long),
        "splits": torch.tensor(splits, dtype=torch.int8),
        "file_idx": torch.tensor(file_idx_list, dtype=torch.int32),
        "file_list": file_list,
        "starts": torch.tensor(starts, dtype=torch.float32),
        "ends": torch.tensor(ends, dtype=torch.float32),
    }


def build_mel_cache(
    items: list[ManifestItem],
    ds_cfg: DatasetConfig,
    mel_cfg: MelConfig,
    cache_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = _build_cache_tensors(items, ds_cfg, mel_cfg)
    data["meta"] = _cfg_payload(ds_cfg, mel_cfg)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(data, str(tmp))
    tmp.replace(cache_path)


def load_mel_cache(cache_path: Path) -> dict[str, Any]:
    return torch.load(str(cache_path), map_location="cpu", weights_only=False)


def ensure_mel_cache(
    items: list[ManifestItem],
    ds_cfg: DatasetConfig,
    mel_cfg: MelConfig,
    manifest_path: Path,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Return the cache path, building it on first use."""
    p = cache_path_for(manifest_path, ds_cfg, mel_cfg, cache_dir=cache_dir)
    if p.exists():
        return p
    print(f"[mel-cache] building cache -> {p}")
    build_mel_cache(items, ds_cfg, mel_cfg, p)
    print(f"[mel-cache] done -> {p}")
    return p
