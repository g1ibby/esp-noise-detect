from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SUPPORTED_EXTS: tuple[str, ...] = (".wav",)
MIN_FILE_SIZE_BYTES = 1024
LABEL_ON = "pump_on"
LABEL_OFF = "pump_off"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42

@dataclass
class AudioFile:
    path: Path
    size_bytes: int
    label: Optional[str] = None


def find_recordings_root(explicit_root: Optional[str]) -> Path:
    if explicit_root is None:
        print("Error: --data-root is required (path to folder containing WAV recordings)", file=sys.stderr)
        sys.exit(2)
    return Path(explicit_root).expanduser().resolve()


def split_tokens(filename: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    for ch in filename:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append(''.join(current))
                current.clear()
    if current:
        tokens.append(''.join(current))
    return tokens


def infer_label(path: Path) -> Optional[str]:
    tokens = split_tokens(path.name.lower())
    found_on = any(token == "on" for token in tokens)
    found_off = any(token == "off" for token in tokens)
    if found_on and not found_off:
        return LABEL_ON
    if found_off and not found_on:
        return LABEL_OFF
    return None


def collect_audio_files(root: Path) -> tuple[List[AudioFile], int]:
    audio_files: List[AudioFile] = []
    skipped_other = 0
    for file_path in sorted(root.rglob('*')):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            size_bytes = file_path.stat().st_size
        except OSError:
            skipped_other += 1
            continue
        audio_files.append(AudioFile(path=file_path.resolve(), size_bytes=size_bytes))
    return audio_files, skipped_other


def assign_splits(entries: List[AudioFile], val_ratio: float, test_ratio: float, seed: int) -> Dict[Path, str]:
    rng = random.Random(seed)
    splits: Dict[Path, str] = {}
    grouped: Dict[str, List[AudioFile]] = defaultdict(list)
    for entry in entries:
        if entry.label is None:
            continue
        grouped[entry.label].append(entry)
    for label, group_entries in grouped.items():
        group_entries_sorted = list(group_entries)
        rng.shuffle(group_entries_sorted)
        n_total = len(group_entries_sorted)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_val = min(n_val, n_total)
        n_test = min(n_test, max(n_total - n_val, 0))
        for idx, entry in enumerate(group_entries_sorted):
            if idx < n_val:
                splits[entry.path] = "val"
            elif idx < n_val + n_test:
                splits[entry.path] = "test"
            else:
                splits[entry.path] = "train"
    return splits


def validate_ratios(val_ratio: float, test_ratio: float) -> None:
    for name, value in [("val_ratio", val_ratio), ("test_ratio", test_ratio)]:
        if value < 0 or value >= 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")


def make_relative_path(path: Path, base_dir: Path) -> str:
    try:
        rel = path.relative_to(base_dir)
        return rel.as_posix()
    except ValueError:
        rel = os.path.relpath(path, start=base_dir)
        return Path(rel).as_posix()


def build_manifest(root: Path, output_path: Path, val_ratio: float, test_ratio: float, seed: int, dry_run: bool) -> None:
    if not root.exists():
        print(f"Error: data root {root} does not exist", file=sys.stderr)
        sys.exit(1)
    audio_files, skipped_other_errors = collect_audio_files(root)
    total_found = len(audio_files)
    kept_entries: List[AudioFile] = []
    skipped_undefined = 0
    skipped_small = 0
    skipped_other = skipped_other_errors
    for entry in audio_files:
        if entry.size_bytes < MIN_FILE_SIZE_BYTES:
            skipped_small += 1
            continue
        entry.label = infer_label(entry.path)
        if entry.label is None:
            skipped_undefined += 1
            continue
        kept_entries.append(entry)
    validate_ratios(val_ratio, test_ratio)
    splits = assign_splits(kept_entries, val_ratio, test_ratio, seed)
    counts_by_label = Counter(entry.label for entry in kept_entries)
    counts_by_split = defaultdict(Counter)
    for entry in kept_entries:
        split = splits.get(entry.path, "train")
        counts_by_split[split][entry.label] += 1
    if dry_run:
        print("Dry run: manifest not written")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for entry in kept_entries:
                split = splits.get(entry.path, "train")
                rel_path = make_relative_path(entry.path, output_path.parent)
                record = {
                    "audio_path": rel_path,
                    "label": entry.label,
                    "split": split,
                }
                f.write(json.dumps(record) + "\n")
    print("Manifest build summary:")
    print(f"- Data root: {root}")
    print(f"- Output: {'(dry-run)' if dry_run else output_path}")
    print(f"- Total WAV files scanned: {total_found}")
    print(f"- Kept entries: {len(kept_entries)}")
    for label in (LABEL_ON, LABEL_OFF):
        print(f"  - {label}: {counts_by_label.get(label, 0)}")
    print("- Skipped files:")
    print(f"  - Undefined label: {skipped_undefined}")
    print(f"  - Too small (<1 KiB): {skipped_small}")
    print(f"  - Other errors: {skipped_other}")
    print("- Split distribution:")
    for split_name in ("train", "val", "test"):
        split_counter = counts_by_split.get(split_name, Counter())
        total = sum(split_counter.values())
        print(f"  - {split_name}: {total} (on={split_counter.get(LABEL_ON, 0)}, off={split_counter.get(LABEL_OFF, 0)})")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest.jsonl for pump audio dataset")
    parser.add_argument("--data-root", type=str, required=True, help="Path to recordings root folder (containing WAV files)")
    parser.add_argument("--output", type=str, default=None, help="Output path for manifest (default <data-root>/manifest.jsonl)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio (default 0.1)")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for stratified split (default 42)")
    parser.add_argument("--dry-run", action="store_true", help="Only compute summary; do not write manifest")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    root = find_recordings_root(args.data_root)
    output_path = Path(args.output).expanduser().resolve() if args.output else root / "manifest.jsonl"
    build_manifest(root, output_path, args.val_ratio, args.test_ratio, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
