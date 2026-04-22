"""Inventory the manifest by recording-session timestamp.

Filename format: xiao_esp32s3_<unix_ts>_c<N>_<label>_chunk<N>[_<flag>].wav
The unix_ts is the moment the device started a session; many files share the
same ts because they're chunks of the same on/off cycle.

Outputs a per-day breakdown and a per-session breakdown so we can decide how
to do honest day/session-stratified splits.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: inventory.py <manifest>", file=sys.stderr)
    sys.exit(1)
MANIFEST = Path(sys.argv[1])

TS_RE = re.compile(r"xiao_esp32s3_(\d{10})_")

per_day = defaultdict(lambda: {"on": 0, "off": 0, "files": 0, "sessions": set()})
per_session = defaultdict(lambda: {"on": 0, "off": 0, "split": defaultdict(int)})
per_split = defaultdict(lambda: {"on": 0, "off": 0, "files": 0})
unknown_ts = 0
total = 0

with MANIFEST.open() as f:
    for line in f:
        total += 1
        rec = json.loads(line)
        name = Path(rec["audio_path"]).name
        m = TS_RE.search(name)
        if not m:
            unknown_ts += 1
            continue
        ts = int(m.group(1))
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        label = rec["label"]
        split = rec.get("split", "train")
        lbl_short = "on" if label == "pump_on" else "off"
        per_day[day]["files"] += 1
        per_day[day][lbl_short] += 1
        per_day[day]["sessions"].add(ts)
        per_session[ts][lbl_short] += 1
        per_session[ts]["split"][split] += 1
        per_split[split][lbl_short] += 1
        per_split[split]["files"] += 1

print(f"Total manifest entries: {total}  unknown_ts: {unknown_ts}")
print(f"Distinct sessions (ts): {len(per_session)}")
print(f"Distinct days: {len(per_day)}")
print()
print("=== Per-day breakdown ===")
print(f"{'date':<12} {'files':>6} {'on':>5} {'off':>5} {'sessions':>9}")
for day in sorted(per_day):
    d = per_day[day]
    print(f"{day:<12} {d['files']:>6} {d['on']:>5} {d['off']:>5} {len(d['sessions']):>9}")

print()
print("=== Per-split breakdown ===")
for s in ("train", "val", "test"):
    d = per_split[s]
    print(f"{s:<6} files={d['files']:>5} on={d['on']:>4} off={d['off']:>4}")

print()
# Check whether any single session ends up across multiple splits — the scary case
multi_split_sessions = [ts for ts, d in per_session.items() if len(d["split"]) > 1]
print(f"Sessions split across multiple splits (LEAKY): {len(multi_split_sessions)} / {len(per_session)}")
if multi_split_sessions:
    sample = multi_split_sessions[:5]
    for ts in sample:
        d = per_session[ts]
        print(f"  ts={ts} on={d['on']} off={d['off']} splits={dict(d['split'])}")

print()
# Check whether any day ends up in only one split (would mean day-leak ≠ file-leak)
day_to_splits = defaultdict(set)
with MANIFEST.open() as f:
    for line in f:
        rec = json.loads(line)
        name = Path(rec["audio_path"]).name
        m = TS_RE.search(name)
        if not m:
            continue
        ts = int(m.group(1))
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        day_to_splits[day].add(rec.get("split", "train"))

multi_day = sum(1 for s in day_to_splits.values() if len(s) > 1)
print(f"Days appearing in >1 split: {multi_day} / {len(day_to_splits)}")
print(f"  -> {multi_day / max(len(day_to_splits), 1) * 100:.1f}% of days are mixed across splits")
