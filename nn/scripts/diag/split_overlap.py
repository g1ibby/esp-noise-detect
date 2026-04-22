"""Compare two manifest splits to see day-level overlap."""
import json, re, sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

OLD = Path(sys.argv[1])
NEW = Path(sys.argv[2])
TS = re.compile(r"_(\d{10})_")

def days_per_split(p):
    out = defaultdict(set)
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            m = TS.search(Path(r["audio_path"]).name)
            if not m:
                continue
            ts = int(m.group(1))
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            out[r.get("split", "train")].add(d)
    return out

old = days_per_split(OLD)
new = days_per_split(NEW)
print(f"OLD: train_days={sorted(old['train'])}")
print(f"OLD: val_days={sorted(old['val'])}")
print(f"OLD: test_days={sorted(old['test'])}")
print()
print(f"NEW: train_days={sorted(new['train'])}")
print(f"NEW: val_days={sorted(new['val'])}")
print(f"NEW: test_days={sorted(new['test'])}")
print()
print("=== Did old model see new TEST days during training? ===")
overlap = new["test"] & old["train"]
print(f"NEW test days that were in OLD train: {sorted(overlap)} (of {len(new['test'])})")
print(f"NEW test days fully unseen by old model: {sorted(new['test'] - old['train'] - old['val'])}")
print()
print("=== Did old model see new VAL days during training? ===")
overlap = new["val"] & old["train"]
print(f"NEW val days that were in OLD train: {sorted(overlap)} (of {len(new['val'])})")
