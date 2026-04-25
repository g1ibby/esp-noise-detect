#!/usr/bin/env bash
# End-to-end Burn → safetensors → ONNX → .espdl export.
#
# Step 13 of MIGRATION_PLAN.md. Wraps:
#   1. nn-rs `export_weights`     (Rust, wgpu/Metal)
#   2. nn-rs/python/burn_to_onnx  (PyTorch + onnxslim)
#   3. esp-ppq docker container   (PTQ INT8 → .espdl)
#
# Usage:
#   scripts/burn_to_espdl.sh \
#       --checkpoint /tmp/nn-rs-robust-smoke/checkpoints/best.mpk \
#       --manifest   /path/to/manifest.jsonl \
#       --out-dir    /tmp/nn-rs-robust-smoke/export
#
# Optional flags mirror burn_to_onnx.py's ESPDL knobs:
#   --config <yaml>          default: nn-rs/configs/robust_session.yaml
#   --target <esp32s3|esp32p4|c>   default: esp32s3
#   --num-bits <8|16>        default: 8
#   --calib-windows N        default: 512
#   --calib-steps N          default: 64
#   --calib-split train|val  default: train

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="nn-rs/configs/robust_session.yaml"
CHECKPOINT=""
MANIFEST=""
OUT_DIR=""
TARGET="esp32s3"
NUM_BITS=8
CALIB_WINDOWS=512
CALIB_STEPS=64
CALIB_SPLIT="train"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)         CONFIG="$2"; shift 2 ;;
        --checkpoint)     CHECKPOINT="$2"; shift 2 ;;
        --manifest)       MANIFEST="$2"; shift 2 ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        --target)         TARGET="$2"; shift 2 ;;
        --num-bits)       NUM_BITS="$2"; shift 2 ;;
        --calib-windows)  CALIB_WINDOWS="$2"; shift 2 ;;
        --calib-steps)    CALIB_STEPS="$2"; shift 2 ;;
        --calib-split)    CALIB_SPLIT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$CHECKPOINT" ]] || { echo "missing --checkpoint"; exit 2; }
[[ -n "$MANIFEST" ]]   || { echo "missing --manifest"; exit 2; }
[[ -n "$OUT_DIR" ]]    || { echo "missing --out-dir"; exit 2; }

SAFETENSORS="$OUT_DIR/tinyconv.safetensors"
LOGITS_JSON="$OUT_DIR/logits_rust.json"

mkdir -p "$OUT_DIR"

echo "==[1/3] Burn → safetensors ==========================================="
cargo run --release -p nn-rs --features metal --bin export_weights -- \
    --config     "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out        "$SAFETENSORS" \
    --verify-logits "$LOGITS_JSON"

echo ""
echo "==[2/3] safetensors → ONNX ==========================================="
VENV_PY="$REPO_ROOT/nn/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "nn/.venv not found — create with: cd nn && uv venv .venv && uv pip install -e .[dev,export]"
    echo "Also install safetensors: nn/.venv/bin/python -m uv pip install safetensors"
    exit 2
fi

echo "==[3/3] ONNX → .espdl (via esp-ppq docker) ==========================="
"$VENV_PY" nn-rs/python/burn_to_onnx.py \
    --weights        "$SAFETENSORS" \
    --config         "$CONFIG" \
    --manifest       "$MANIFEST" \
    --out-dir        "$OUT_DIR" \
    --calib-split    "$CALIB_SPLIT" \
    --calib-windows  "$CALIB_WINDOWS" \
    --compare-logits "$LOGITS_JSON" \
    --espdl \
    --espdl-target     "$TARGET" \
    --espdl-num-bits   "$NUM_BITS" \
    --espdl-calib-steps "$CALIB_STEPS"

echo ""
echo "Done. .espdl file: $OUT_DIR/model.espdl"
echo "Copy into firmware/models/ and rebuild the firmware binary."
