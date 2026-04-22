from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SUPPORTED_EXTS: tuple[str, ...] = (".wav",)
MIN_FILE_SIZE_BYTES = 1024
LABEL_ON = "pump_on"
LABEL_OFF = "pump_off"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42

# Suffix flags in filenames that indicate aborted/partial recordings.
# These files are usually mis-labeled: "_canceled" recordings are often
# labeled pump_off but contain pump running audio (or vice versa), and
# "_timeout" recordings are cut short and may be mostly silence. Excluding
# them removes label noise from training and eval.
DEFAULT_EXCLUDE_FLAGS: tuple[str, ...] = ("_canceled", "_timeout", "_undefined")

SESSION_TS_RE = re.compile(r"_(\d{10})_")


@dataclass
class AudioFile:
    path: Path
    size_bytes: int
    label: Optional[str] = None
    session_ts: Optional[int] = None
    day: Optional[str] = None
    flag: Optional[str] = None


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


def extract_session_ts(path: Path) -> Optional[int]:
    m = SESSION_TS_RE.search(path.name)
    return int(m.group(1)) if m else None


def session_day(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def detect_flag(path: Path, exclude_flags: tuple[str, ...]) -> Optional[str]:
    name = path.name.lower()
    for flag in exclude_flags:
        if flag in name:
            return flag
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


def assign_splits_by_session(
    entries: List[AudioFile],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_by: str,
) -> Dict[Path, str]:
    """Assign splits so that an entire group (session or day) lands in a single split.

    A single recording session typically contains both pump_on and pump_off
    chunks (captured immediately before/after the pump cycle), so we must
    group across labels — otherwise the same session could end up in train
    for pump_on and val for pump_off, leaking environmental acoustics.

    We pack whole groups greedily into val/test until the target per-label
    file counts are hit, then the rest go to train. Groups are sorted by
    size descending so large sessions don't overshoot small targets.
    """
    rng = random.Random(seed)
    splits: Dict[Path, str] = {}

    def key_of(e: AudioFile) -> Optional[str]:
        if group_by == "session":
            return str(e.session_ts) if e.session_ts is not None else None
        return e.day

    # Build group -> list of entries (across all labels)
    groups: Dict[str, List[AudioFile]] = defaultdict(list)
    for e in entries:
        if e.label is None:
            continue
        k = key_of(e)
        if k is None:
            continue
        groups[k].append(e)

    # Per-label totals for quota computation
    total_per_label: Counter = Counter()
    for gk, group_entries in groups.items():
        for e in group_entries:
            total_per_label[e.label] += 1

    target_val = {lbl: int(total_per_label[lbl] * val_ratio) for lbl in total_per_label}
    target_test = {lbl: int(total_per_label[lbl] * test_ratio) for lbl in total_per_label}

    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)
    # Secondary sort: descending by size so large groups get placed first
    # (prevents a single huge session from overshooting a small quota).
    group_keys.sort(key=lambda k: -len(groups[k]))

    assigned_val: Counter = Counter()
    assigned_test: Counter = Counter()

    def group_label_counts(g_entries: List[AudioFile]) -> Counter:
        c: Counter = Counter()
        for e in g_entries:
            c[e.label] += 1
        return c

    def fits(target: dict, assigned: Counter, lbl_counts: Counter) -> bool:
        # A group fits if placing it wouldn't overshoot more than 25% of the
        # target on any label (prevents deadlock on very large sessions while
        # keeping quotas approximately honored).
        for lbl, n in lbl_counts.items():
            if target.get(lbl, 0) == 0:
                return False
            limit = int(target[lbl] * 1.25)
            if assigned[lbl] + n > limit:
                return False
        return True

    for k in group_keys:
        g_entries = groups[k]
        lbl_counts = group_label_counts(g_entries)

        # Check val/test need: prefer placing where remaining need is largest
        # relative to target.
        val_need = sum(max(target_val.get(l, 0) - assigned_val.get(l, 0), 0) for l in lbl_counts)
        test_need = sum(max(target_test.get(l, 0) - assigned_test.get(l, 0), 0) for l in lbl_counts)

        if val_need > 0 and fits(target_val, assigned_val, lbl_counts) and val_need >= test_need:
            split = "val"
            for l, n in lbl_counts.items():
                assigned_val[l] += n
        elif test_need > 0 and fits(target_test, assigned_test, lbl_counts):
            split = "test"
            for l, n in lbl_counts.items():
                assigned_test[l] += n
        else:
            split = "train"
        for entry in g_entries:
            splits[entry.path] = split

    return splits


def assign_splits_random(entries: List[AudioFile], val_ratio: float, test_ratio: float, seed: int) -> Dict[Path, str]:
    """Legacy random-per-file split. LEAKS sessions — do not use in production."""
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


def build_manifest(
    root: Path,
    output_path: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    dry_run: bool,
    split_by: str,
    exclude_flags: tuple[str, ...],
) -> None:
    if not root.exists():
        print(f"Error: data root {root} does not exist", file=sys.stderr)
        sys.exit(1)
    audio_files, skipped_other_errors = collect_audio_files(root)
    total_found = len(audio_files)
    kept_entries: List[AudioFile] = []
    skipped_undefined = 0
    skipped_small = 0
    skipped_flagged: Counter = Counter()
    skipped_no_ts = 0
    skipped_other = skipped_other_errors
    for entry in audio_files:
        if entry.size_bytes < MIN_FILE_SIZE_BYTES:
            skipped_small += 1
            continue
        flag = detect_flag(entry.path, exclude_flags)
        if flag is not None:
            skipped_flagged[flag] += 1
            continue
        entry.label = infer_label(entry.path)
        if entry.label is None:
            skipped_undefined += 1
            continue
        entry.session_ts = extract_session_ts(entry.path)
        if entry.session_ts is None:
            skipped_no_ts += 1
            if split_by != "random":
                continue
        else:
            entry.day = session_day(entry.session_ts)
        kept_entries.append(entry)
    validate_ratios(val_ratio, test_ratio)

    if split_by == "random":
        splits = assign_splits_random(kept_entries, val_ratio, test_ratio, seed)
    else:
        splits = assign_splits_by_session(kept_entries, val_ratio, test_ratio, seed, split_by)

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

    # Leakage audit
    sessions_per_split = defaultdict(set)
    days_per_split = defaultdict(set)
    for entry in kept_entries:
        split = splits.get(entry.path, "train")
        if entry.session_ts is not None:
            sessions_per_split[split].add(entry.session_ts)
        if entry.day is not None:
            days_per_split[split].add(entry.day)

    print("Manifest build summary:")
    print(f"- Data root: {root}")
    print(f"- Output: {'(dry-run)' if dry_run else output_path}")
    print(f"- Split strategy: {split_by}")
    print(f"- Exclude flags: {exclude_flags}")
    print(f"- Total WAV files scanned: {total_found}")
    print(f"- Kept entries: {len(kept_entries)}")
    for label in (LABEL_ON, LABEL_OFF):
        print(f"  - {label}: {counts_by_label.get(label, 0)}")
    print("- Skipped files:")
    print(f"  - Undefined label: {skipped_undefined}")
    print(f"  - Too small (<1 KiB): {skipped_small}")
    for flag, count in skipped_flagged.items():
        print(f"  - Flagged {flag}: {count}")
    if skipped_no_ts:
        print(f"  - No session_ts (required for group splits): {skipped_no_ts}")
    print(f"  - Other errors: {skipped_other}")
    print("- Split distribution:")
    for split_name in ("train", "val", "test"):
        split_counter = counts_by_split.get(split_name, Counter())
        total = sum(split_counter.values())
        n_sess = len(sessions_per_split.get(split_name, set()))
        n_days = len(days_per_split.get(split_name, set()))
        print(
            f"  - {split_name}: {total} files "
            f"(on={split_counter.get(LABEL_ON, 0)}, off={split_counter.get(LABEL_OFF, 0)}) "
            f"sessions={n_sess} days={n_days}"
        )

    # Leakage checks
    tr, va, te = (
        sessions_per_split["train"],
        sessions_per_split["val"],
        sessions_per_split["test"],
    )
    print("- Session leakage check:")
    print(f"  - train ∩ val: {len(tr & va)}")
    print(f"  - train ∩ test: {len(tr & te)}")
    print(f"  - val ∩ test: {len(va & te)}")
    tr_d, va_d, te_d = (
        days_per_split["train"],
        days_per_split["val"],
        days_per_split["test"],
    )
    print("- Day leakage check:")
    print(f"  - train ∩ val: {len(tr_d & va_d)}")
    print(f"  - train ∩ test: {len(tr_d & te_d)}")
    print(f"  - val ∩ test: {len(va_d & te_d)}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest.jsonl for pump audio dataset")
    parser.add_argument("--data-root", type=str, required=True, help="Path to recordings root folder (containing WAV files)")
    parser.add_argument("--output", type=str, default=None, help="Output path for manifest (default <data-root>/manifest.jsonl)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio (default 0.1)")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for stratified split (default 42)")
    parser.add_argument(
        "--split-by",
        choices=("session", "day", "random"),
        default="session",
        help="Split strategy: 'session' keeps a whole recording session in one split "
        "(default, honest), 'day' keeps a whole calendar day in one split, "
        "'random' shuffles per file (LEGACY, leaks sessions).",
    )
    parser.add_argument(
        "--keep-flagged",
        action="store_true",
        help="Keep files with _canceled/_timeout/_undefined in filename (default: exclude)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only compute summary; do not write manifest")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    root = find_recordings_root(args.data_root)
    output_path = Path(args.output).expanduser().resolve() if args.output else root / "manifest.jsonl"
    exclude_flags: tuple[str, ...] = () if args.keep_flagged else DEFAULT_EXCLUDE_FLAGS
    build_manifest(
        root,
        output_path,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.dry_run,
        args.split_by,
        exclude_flags,
    )


if __name__ == "__main__":
    main()
