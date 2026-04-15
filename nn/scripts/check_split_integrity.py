from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_manifest(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def file_hash(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def analyze(manifest_path: Path) -> None:
    rows = read_manifest(manifest_path)
    if not rows:
        print("Empty or invalid manifest")
        return
    # Normalize to absolute paths relative to manifest
    root = manifest_path.parent
    def abs_path(p: str) -> Path:
        q = Path(p)
        if not q.is_absolute():
            q = (root / q).resolve()
        return q

    by_split: Dict[str, List[Path]] = defaultdict(list)
    by_label: Dict[str, int] = defaultdict(int)
    all_paths: List[Path] = []
    for r in rows:
        ap = r.get("audio_path")
        if not isinstance(ap, str):
            continue
        p = abs_path(ap)
        sp = r.get("split") or "train"
        lb = r.get("label") or "?"
        by_split[sp].append(p)
        by_label[lb] += 1
        all_paths.append(p)

    # Print counts
    print("Counts by split:")
    for s in ("train", "val", "test"):
        print(f"  - {s}: {len(by_split.get(s, []))}")
    print("Counts by label:")
    for lb, c in by_label.items():
        print(f"  - {lb}: {c}")

    # Overlap by exact absolute path
    inter_tv = set(by_split.get("train", [])) & set(by_split.get("val", []))
    inter_tt = set(by_split.get("train", [])) & set(by_split.get("test", []))
    inter_vt = set(by_split.get("val", [])) & set(by_split.get("test", []))
    print("Path overlaps:")
    print(f"  - train ∩ val: {len(inter_tv)}")
    print(f"  - train ∩ test: {len(inter_tt)}")
    print(f"  - val ∩ test: {len(inter_vt)}")

    # Overlap by content hash (expensive but robust)
    print("Hashing files to detect duplicates across splits (this may take time)...")
    hashes: Dict[Path, str] = {}
    for p in sorted(set(all_paths)):
        if not p.exists():
            continue
        try:
            hashes[p] = file_hash(p)
        except Exception:
            continue
    inv: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for s, paths in by_split.items():
        for p in paths:
            h = hashes.get(p)
            if h:
                inv[h].append((s, p))
    dup_tv = sum(1 for hs, items in inv.items() if {s for s, _ in items} >= {"train", "val"})
    dup_tt = sum(1 for hs, items in inv.items() if {s for s, _ in items} >= {"train", "test"})
    dup_vt = sum(1 for hs, items in inv.items() if {s for s, _ in items} >= {"val", "test"})
    print("Content-duplicate overlaps (by SHA256):")
    print(f"  - train ↔ val: {dup_tv}")
    print(f"  - train ↔ test: {dup_tt}")
    print(f"  - val ↔ test: {dup_vt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check manifest split integrity and duplicates")
    ap.add_argument("--manifest", type=str, default="../recordings/manifest.jsonl", help="Path to manifest.jsonl")
    args = ap.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    analyze(manifest_path)


if __name__ == "__main__":
    main()

