#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: firmware/scripts/ram_report.sh [--source-esp] [--run|--build] [--release|--debug] [--example NAME] [--offline]

Runs a firmware build (or run) and produces a RAM/linker usage report.

Examples:
  source ~/export-esp.sh && firmware/scripts/ram_report.sh --build --release --example pump_monitor
  firmware/scripts/ram_report.sh --source-esp --build --release --example pump_monitor --offline

Outputs:
  firmware/target/ram_report/<timestamp>/{build.log,*.map,ram_report.json,ram_report.txt}
EOF
}

mode="build"
profile="release"
example="pump_monitor"
source_esp=0
offline=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --run) mode="run"; shift ;;
    --build) mode="build"; shift ;;
    --release) profile="release"; shift ;;
    --debug) profile="debug"; shift ;;
    --example) example="${2:-}"; shift 2 ;;
    --source-esp) source_esp=1; shift ;;
    --offline) offline=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fw_dir="${root_dir}/firmware"

if [[ $source_esp -eq 1 ]]; then
  if [[ -f "${HOME}/export-esp.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/export-esp.sh"
  else
    echo "WARN: --source-esp set but ${HOME}/export-esp.sh not found" >&2
  fi
fi

ts="$(date +%Y%m%d-%H%M%S)"
out_dir="${fw_dir}/target/ram_report/${ts}"
mkdir -p "${out_dir}"

map_path="${out_dir}/${example}.map"
log_path="${out_dir}/build.log"
txt_path="${out_dir}/ram_report.txt"
json_path="${out_dir}/ram_report.json"

abs_map_path="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${map_path}")"

rustflags=(
  "-C" "link-arg=-Wl,-Map=${abs_map_path}"
  "-C" "link-arg=-Wl,--print-gc-sections"
  "-C" "link-arg=-Wl,--print-memory-usage"
)

cargo_cmd=(cargo "${mode}")
if [[ "${profile}" == "release" ]]; then
  cargo_cmd+=(--release)
fi
cargo_cmd+=(--example "${example}")
if [[ $offline -eq 1 ]]; then
  cargo_cmd+=(--offline)
fi

echo "Running: (cd ${fw_dir} && RUSTFLAGS='${rustflags[*]}' ${cargo_cmd[*]})" | tee "${log_path}"
set +e
(cd "${fw_dir}" && env RUSTFLAGS="${rustflags[*]}" "${cargo_cmd[@]}") >>"${log_path}" 2>&1
exit_code=$?
set -e
echo "Exit code: ${exit_code}" >>"${log_path}"

python3 - "${fw_dir}" "${profile}" "${example}" "${log_path}" "${map_path}" "${json_path}" "${txt_path}" <<'PY'
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

fw_dir = Path(sys.argv[1])
profile = sys.argv[2]
example = sys.argv[3]
log_path = Path(sys.argv[4])
map_path = Path(sys.argv[5])
json_path = Path(sys.argv[6])
txt_path = Path(sys.argv[7])

@dataclass
class Region:
    name: str
    origin: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.origin

def find_latest_memory_x() -> Path | None:
    base = fw_dir / "target" / "xtensa-esp32s3-none-elf" / profile / "build"
    if not base.exists():
        return None
    candidates = sorted(base.glob("esp-hal-*/out/memory.x"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def parse_regions(memory_x: Path) -> tuple[Region | None, Region | None]:
    text = memory_x.read_text(errors="ignore")
    def hex_int(s: str) -> int:
        return int(s, 16)

    # Example lines:
    # dram2_seg ( RW )       : ORIGIN = 0x3FCDB700, len = 0x3FCED710 - 0x3FCDB700
    # dram_seg ( RW )        : ORIGIN = 0x3FC88000 , len = ORIGIN(dram2_seg) - 0x3FC88000
    m_dram2 = re.search(r"^\s*dram2_seg.*ORIGIN\s*=\s*(0x[0-9A-Fa-f]+)\s*,\s*len\s*=\s*(0x[0-9A-Fa-f]+)\s*-\s*(0x[0-9A-Fa-f]+)",
                       text, flags=re.M)
    if not m_dram2:
        return None, None
    dram2_origin = hex_int(m_dram2.group(1))
    dram2_end = hex_int(m_dram2.group(2))
    dram2 = Region("dram2_seg", dram2_origin, dram2_end)

    m_dram = re.search(r"^\s*dram_seg.*ORIGIN\s*=\s*(0x[0-9A-Fa-f]+)\s*,\s*len\s*=\s*ORIGIN\(dram2_seg\)\s*-\s*(0x[0-9A-Fa-f]+)",
                       text, flags=re.M)
    if not m_dram:
        return None, dram2
    dram_origin = hex_int(m_dram.group(1))
    # end of dram_seg is start of dram2_seg by definition in memory.x
    dram = Region("dram_seg", dram_origin, dram2_origin)
    return dram, dram2

def parse_overflow_from_log(log_text: str) -> tuple[int, int] | None:
    # ld:stack.x:12 cannot move location counter backwards (from 3fcf832c to 3fcdb700)
    m = re.search(r"cannot move location counter backwards \(from ([0-9A-Fa-f]+) to ([0-9A-Fa-f]+)\)", log_text)
    if not m:
        return None
    return int(m.group(1), 16), int(m.group(2), 16)

def parse_model_from_expansion() -> dict:
    exp = fw_dir / "target" / "edgedl-expansion.rs"
    if not exp.exists():
        return {"ok": False, "error": f"missing {exp}"}
    text = exp.read_text(errors="ignore")
    pat = re.compile(r"static\s+__NN_INIT_\d+(?:_EXPS)?\s*:\s*__NN_ALIGN16\s*<\s*\[(u8|i8);\s*(\d+)usize\]\s*>")
    u8 = 0
    i8 = 0
    for m in pat.finditer(text):
        if m.group(1) == "u8":
            u8 += int(m.group(2))
        else:
            i8 += int(m.group(2))

    arena_size_m = re.search(r"pub\(super\)\s+const\s+ARENA_SIZE\s*:\s*usize\s*=\s*(\d+)usize", text)
    arena_scratch_m = re.search(r"pub\(super\)\s+const\s+ARENA_SCRATCH\s*:\s*usize\s*=\s*(\d+)usize", text)

    arena_size = int(arena_size_m.group(1)) if arena_size_m else None
    arena_scratch = int(arena_scratch_m.group(1)) if arena_scratch_m else None

    return {
        "ok": True,
        "weights_bytes": u8 + i8,
        "weights_u8_bytes": u8,
        "weights_i8_bytes": i8,
        "arena_bytes": (arena_size + arena_scratch) if (arena_size is not None and arena_scratch is not None) else None,
        "arena_planned_bytes": arena_size,
        "arena_scratch_bytes": arena_scratch,
        "expansion_path": str(exp),
    }

def newest_file(glob_pat: str) -> Path | None:
    candidates = list(fw_dir.glob(glob_pat))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def parse_top_bss_from_rlib(rlib: Path) -> dict:
    nm = "xtensa-esp32s3-elf-nm"
    try:
        out = subprocess.check_output([nm, "-S", "--size-sort", "--radix=d", str(rlib)], stderr=subprocess.DEVNULL)
    except Exception as e:
        return {"ok": False, "error": f"failed to run {nm}: {e}", "rlib": str(rlib)}

    items = []
    pat = re.compile(r"^(\d+)\s+(\d+)\s+([BbDdRr])\s+(\S+)")
    for line in out.decode("utf-8", "ignore").splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        addr = int(m.group(1))
        size = int(m.group(2))
        typ = m.group(3)
        name = m.group(4)
        items.append((size, typ, name))

    bss = sum(s for s, t, _ in items if t in "Bb")
    data = sum(s for s, t, _ in items if t in "Dd")
    ro = sum(s for s, t, _ in items if t in "Rr")
    top_bss = [{"size": s, "name": n} for s, t, n in sorted([i for i in items if i[1] in "Bb"], key=lambda x: x[0], reverse=True)[:25]]
    return {
        "ok": True,
        "rlib": str(rlib),
        "bss_bytes": bss,
        "data_bytes": data,
        "rodata_bytes": ro,
        "top_bss": top_bss,
    }

log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""

memory_x = find_latest_memory_x()
dram_seg = None
dram2_seg = None
if memory_x:
    dram_seg, dram2_seg = parse_regions(memory_x)

overflow = None
from_addr = None
to_addr = None
overflow_bytes = None
static_used_bytes = None
if (ov := parse_overflow_from_log(log_text)) is not None:
    from_addr, to_addr = ov
    overflow_bytes = from_addr - to_addr
    overflow = {
        "from": hex(from_addr),
        "to": hex(to_addr),
        "overflow_bytes": overflow_bytes,
    }
    if dram_seg:
        static_used_bytes = from_addr - dram_seg.origin

model = parse_model_from_expansion()

rlib = newest_file(f"target/xtensa-esp32s3-none-elf/{profile}/deps/libfirmware-*.rlib")
rlib_stats = parse_top_bss_from_rlib(rlib) if rlib else {"ok": False, "error": "no libfirmware rlib found"}

report = {
    "example": example,
    "profile": profile,
    "log_path": str(log_path),
    "map_path": str(map_path),
    "memory_x": str(memory_x) if memory_x else None,
    "regions": {
        "dram_seg": None if not dram_seg else {"origin": hex(dram_seg.origin), "end": hex(dram_seg.end), "size_bytes": dram_seg.size},
        "dram2_seg": None if not dram2_seg else {"origin": hex(dram2_seg.origin), "end": hex(dram2_seg.end), "size_bytes": dram2_seg.size},
    },
    "overflow": overflow,
    "static_used_in_dram_seg_bytes": static_used_bytes,
    "model": model,
    "rlib_stats": rlib_stats,
}

json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

def kib(n: int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n/1024:.2f} KiB"

lines = []
lines.append(f"example={example} profile={profile}")
if dram_seg:
    lines.append(f"dram_seg (RWDATA): origin={hex(dram_seg.origin)} end={hex(dram_seg.end)} size={dram_seg.size} ({kib(dram_seg.size)})")
if dram2_seg:
    lines.append(f"dram2_seg: origin={hex(dram2_seg.origin)} end={hex(dram2_seg.end)} size={dram2_seg.size} ({kib(dram2_seg.size)})")
if overflow_bytes is not None:
    lines.append(f"LINK FAIL overflow: {overflow_bytes} bytes ({kib(overflow_bytes)}) (from {hex(from_addr)} to {hex(to_addr)})")
    if static_used_bytes is not None and dram_seg:
        lines.append(f"Static end address={hex(from_addr)}; dram_seg used={static_used_bytes} ({kib(static_used_bytes)}) of {dram_seg.size} ({kib(dram_seg.size)})")
else:
    lines.append("No ld overflow detected in build log.")

if model.get("ok"):
    lines.append(f"Model weights bytes={model.get('weights_bytes')} ({kib(model.get('weights_bytes'))}) (forced into .data on target_os=none)")
    lines.append(f"Model arena bytes={model.get('arena_bytes')} ({kib(model.get('arena_bytes'))}) planned={model.get('arena_planned_bytes')} scratch={model.get('arena_scratch_bytes')}")
else:
    lines.append(f"Model parse: {model.get('error')}")

if rlib_stats.get("ok"):
    lines.append(f"firmware rlib: {rlib_stats['rlib']}")
    lines.append(f"  BSS={rlib_stats['bss_bytes']} ({kib(rlib_stats['bss_bytes'])}) RODATA={rlib_stats['rodata_bytes']} ({kib(rlib_stats['rodata_bytes'])})")
    lines.append("  Top BSS symbols (rlib, may include GC-removed):")
    for ent in rlib_stats["top_bss"][:15]:
        lines.append(f"    {ent['size']:8d}  {ent['name']}")
else:
    lines.append(f"rlib stats: {rlib_stats.get('error')}")

txt_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo
echo "Wrote:"
echo "  ${txt_path}"
echo "  ${json_path}"
echo "  ${log_path}"
echo "  ${map_path}"
