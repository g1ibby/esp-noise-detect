from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

LabelName = Literal["pump_off", "pump_on"]


@dataclass(frozen=True)
class ManifestItem:
    audio_path: Path
    label: LabelName
    split: Optional[Literal["train", "val", "test"]] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None


def _resolve_audio_path(audio_path: str, base_dir: Path) -> Path:
    p = Path(audio_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_manifest(path: Path) -> list[ManifestItem]:
    """Load and validate manifest.jsonl.

    Required fields per row: audio_path (str), label (pump_off|pump_on)
    Optional fields: split (train|val|test), start_s, end_s.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    base_dir = path.parent

    items: list[ManifestItem] = []
    for row in _iter_jsonl(path):
        apath = row.get("audio_path")
        label = row.get("label")
        split = row.get("split")
        start_s = row.get("start_s")
        end_s = row.get("end_s")
        if not isinstance(apath, str) or not isinstance(label, str):
            continue
        if label not in {"pump_off", "pump_on"}:
            continue
        if split is not None and split not in {"train", "val", "test"}:
            split = None
        audio_path = _resolve_audio_path(apath, base_dir)
        items.append(
            ManifestItem(
                audio_path=audio_path,
                label=label,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
                start_s=float(start_s) if start_s is not None else None,
                end_s=float(end_s) if end_s is not None else None,
            )
        )
    return items


def manifest_checksum(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def default_manifest_path() -> Optional[Path]:
    """Try to find a default manifest path based on repo layout.

    Uses Hydra original working directory if available to keep relative paths stable.
    """
    base = None
    # Prefer Hydra original CWD if available
    try:
        from hydra.core.hydra_config import HydraConfig  # type: ignore

        if HydraConfig.initialized():
            base = Path(HydraConfig.get().runtime.cwd)
    except Exception:
        base = None
    if base is None:
        base_env = os.environ.get("HYDRA_ORIGINAL_CWD")
        if base_env:
            base = Path(base_env)
    if base is None:
        base = Path.cwd()

    candidates = [
        base / "../recordings/manifest.jsonl",
        base / "./manifest.jsonl",
        base / "../host-tools/recordings/manifest.jsonl",
    ]
    for p in candidates:
        p = p.expanduser().resolve()
        if p.exists():
            return p
    return None
