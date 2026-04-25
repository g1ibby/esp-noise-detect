"""Burn-to-ONNX export bridge.

Step 12 of ``MIGRATION_PLAN.md``. Consumes the ``.safetensors`` file
emitted by ``nn-rs export_weights`` and produces the artifacts the
existing ``esp-ppq`` Docker container expects:

* ``model_fp32.onnx`` (always)
* ``model_fp32_slim.onnx`` (via ``onnxslim``)
* ``calib/calib_*.npy`` mel windows for PTQ calibration

The layout matches what ``nn/src/noise_detect/export.py`` produces, so
the downstream ``scripts/export_espdl.py`` + Docker step is unchanged.

The bridge reuses ``noise_detect.models.TinyConv`` and the existing
export helpers. The Rust pipeline's single composed YAML
(``nn-rs/configs/robust_session.yaml``) is read directly — no Hydra
composition, no OmegaConf overrides.

Usage::

    uv run --project nn python nn-rs/python/burn_to_onnx.py \\
        --weights /tmp/nn-rs-robust-smoke/export/tinyconv.safetensors \\
        --config  nn-rs/configs/robust_session.yaml \\
        --manifest /abs/path/manifest.jsonl \\
        --out-dir /tmp/nn-rs-robust-smoke/export

If ``--manifest`` is omitted, calibration windows are skipped (useful
when only parity-testing the ONNX export itself).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

# Make ``noise_detect`` importable regardless of cwd.
_THIS = Path(__file__).resolve()
_NN_SRC = _THIS.parents[2] / "nn" / "src"
if str(_NN_SRC) not in sys.path:
    sys.path.insert(0, str(_NN_SRC))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from noise_detect.config import (  # noqa: E402  (import after sys.path tweak)
    CalibConfig,
    DatasetConfig,
    MelConfig,
    OnnxExportConfig,
    TinyConvConfig,
)
from noise_detect.export import (  # noqa: E402
    _compute_mel_shape,
    _export_onnx,
    _save_calibration_windows,
)
from noise_detect.models.tinyconv import TinyConv  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_from_yaml(cfg: dict, manifest_override: str | None) -> DatasetConfig:
    d = cfg.get("dataset") or {}
    return DatasetConfig(
        sample_rate=int(d.get("sample_rate", 32_000)),
        window_s=float(d.get("window_s", 1.0)),
        hop_s=float(d.get("hop_s", 0.5)),
        class_names=list(d.get("class_names", ["pump_off", "pump_on"])),
        manifest_path=manifest_override or d.get("manifest_path"),
    )


def _mel_from_yaml(cfg: dict) -> MelConfig:
    m = cfg.get("features") or cfg.get("mel") or {}
    kw = {
        "n_mels": int(m.get("n_mels", 64)),
        "fmin": float(m.get("fmin", 50.0)),
        "fmax": m.get("fmax"),
        "n_fft": int(m.get("n_fft", 1024)),
        "hop_length_ms": float(m.get("hop_length_ms", 10.0)),
        "power": float(m.get("power", 2.0)),
        "log": bool(m.get("log", True)),
        "eps": float(m.get("eps", 1e-10)),
        "normalize": bool(m.get("normalize", True)),
        "center": bool(m.get("center", True)),
        "pad_mode": str(m.get("pad_mode", "reflect")),
    }
    if kw["fmax"] is not None:
        kw["fmax"] = float(kw["fmax"])
    return MelConfig(**kw)


def _model_from_yaml(cfg: dict) -> TinyConvConfig:
    m = cfg.get("model") or {}
    return TinyConvConfig(
        channels=list(m.get("channels", [16, 32, 64])),
        dropout=float(m.get("dropout", 0.0)),
    )


def _load_weights(model: TinyConv, weights_path: Path) -> None:
    state = load_file(str(weights_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ``num_batches_tracked`` lives in PyTorch BatchNorm but is absent from
    # Burn's running state — it's expected to be reported as missing and
    # defaulted to 0, which is fine for eval.
    missing_non_bn = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing_non_bn:
        raise RuntimeError(
            f"Missing tensors when loading safetensors into TinyConv: {missing_non_bn}"
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected tensors in safetensors: {unexpected}. "
            "The Rust export key-remapping in `nn-rs/src/bin/export_weights.rs` "
            "likely disagrees with the PyTorch TinyConv state dict."
        )


def _run_espdl_docker(
    onnx_path: Path,
    calib_dir: Path,
    out_path: Path,
    n_mels: int,
    n_frames: int,
    target: str,
    num_bits: int,
    calib_steps: int,
    export_test_values: bool,
    error_report: bool,
) -> None:
    """Build and run the ``esp-ppq`` CPU Docker image to quantize ONNX → .espdl.

    Direct port of ``noise_detect.export._run_espdl_docker``. We keep the
    same repo-root mount so ``scripts/export_espdl.py`` and the existing
    Dockerfile in ``nn/docker/`` stay untouched.
    """
    repo_root = _THIS.parents[2]  # workspace root
    nn_root = repo_root / "nn"
    dockerfile = os.environ.get(
        "ESP_PPQ_DOCKERFILE", str(nn_root / "docker" / "esp-ppq-cpu.Dockerfile")
    )
    image = os.environ.get("ESP_PPQ_IMAGE", "esp-ppq:cpu")
    build_context = os.environ.get("ESP_PPQ_BUILD_CONTEXT", str(repo_root))

    print(f"[espdl] building image {image} (this is idempotent — fast after first run)")
    subprocess.run(
        ["docker", "build", "-f", dockerfile, "-t", image, build_context],
        check=True,
    )

    # Two bind mounts: repo root at ``/work`` (so the export script is
    # visible) and the *export root* at ``/out``. The export root is
    # chosen as the common parent of onnx / calib / out so all three
    # paths resolve under one mount — they're typically siblings under
    # ``--out-dir``. Any host layout works; the container only sees the
    # stable ``/work`` and ``/out`` prefixes.
    onnx_abs = onnx_path.resolve()
    calib_abs = calib_dir.resolve()
    out_abs = out_path.resolve()
    common_host = Path(os.path.commonpath([str(onnx_abs), str(calib_abs), str(out_abs)]))
    if common_host.is_file():
        common_host = common_host.parent
    script_container = "/work/" + os.path.relpath(
        str(nn_root / "scripts" / "export_espdl.py"), start=str(repo_root)
    )

    def _to_container(p: Path) -> str:
        return "/out/" + os.path.relpath(str(p), start=str(common_host))

    run_cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo_root}:/work",
        "-v", f"{common_host}:/out",
        "-w", "/work",
        image,
        script_container,
        "--onnx", _to_container(onnx_abs),
        "--out", _to_container(out_abs),
        "--calib-dir", _to_container(calib_abs),
        "--input-shape", "1", "1", str(int(n_mels)), str(int(n_frames)),
        "--target", str(target),
        "--num-bits", str(int(num_bits)),
        "--calib-steps", str(int(calib_steps)),
        "--device", "cpu",
        "--verbose", "1",
    ]
    if export_test_values:
        run_cmd.append("--export-test-values")
    if not error_report:
        run_cmd.append("--no-error-report")

    print("[espdl] running:", " ".join(run_cmd))
    subprocess.run(run_cmd, check=True)


def _run_forward_check(model: TinyConv, mel_shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    n_mels, n_frames = mel_shape
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 1, n_mels, n_frames), generator=g, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    return y.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Burn-to-ONNX export bridge")
    ap.add_argument("--weights", required=True, type=Path,
                    help="Path to the .safetensors file emitted by `nn-rs export_weights`")
    ap.add_argument("--config", required=True, type=Path,
                    help="Composed YAML config (e.g. nn-rs/configs/robust_session.yaml)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for ONNX + calibration windows")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Manifest.jsonl — required if calibration windows are needed")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset (default: 13)")
    ap.add_argument("--no-simplify", action="store_true", help="Skip the onnxslim pass")
    ap.add_argument("--calib-split", default="train", choices=["train", "val"],
                    help="Split to draw calibration windows from")
    ap.add_argument("--calib-windows", type=int, default=512,
                    help="How many calibration windows to dump (0 = skip)")
    ap.add_argument("--dump-logits", type=Path, default=None,
                    help="Optional: save a seed-0 forward pass output as .npy for parity testing")
    ap.add_argument("--compare-logits", type=Path, default=None,
                    help="Optional: path to the JSON file written by `export_weights "
                         "--verify-logits`. Runs the PyTorch model on the same zero input "
                         "and checks the outputs agree.")
    ap.add_argument("--logit-tolerance", type=float, default=1e-4,
                    help="Absolute tolerance for the --compare-logits check")
    ap.add_argument("--espdl", action="store_true",
                    help="Also run the esp-ppq Docker container to quantize the ONNX "
                         "into a .espdl model. Requires docker on PATH.")
    ap.add_argument("--espdl-target", default="esp32s3",
                    choices=["c", "esp32s3", "esp32p4"])
    ap.add_argument("--espdl-num-bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--espdl-calib-steps", type=int, default=64)
    ap.add_argument("--espdl-out", type=Path, default=None,
                    help="Path to the generated .espdl file. Defaults to "
                         "<out-dir>/model.espdl")
    ap.add_argument("--espdl-export-test-values", action="store_true", default=True,
                    help="Embed test input/output tensors in the .espdl")
    ap.add_argument("--espdl-no-error-report", action="store_true")
    args = ap.parse_args()

    yaml_cfg = _load_yaml(args.config)
    dataset_cfg = _dataset_from_yaml(yaml_cfg, args.manifest)
    mel_cfg = _mel_from_yaml(yaml_cfg)
    model_cfg = _model_from_yaml(yaml_cfg)

    n_classes = len(dataset_cfg.class_names)
    model = TinyConv(model_cfg, n_classes=n_classes)
    model.eval()
    _load_weights(model, args.weights)

    n_mels, n_frames = _compute_mel_shape(dataset_cfg, mel_cfg)
    print(f"Mel shape: n_mels={n_mels}, n_frames={n_frames}")

    if args.compare_logits is not None:
        import json

        with open(args.compare_logits, "r", encoding="utf-8") as f:
            rust_payload = json.load(f)
        rust_logits = np.asarray(rust_payload["logits"], dtype=np.float64)
        x = torch.zeros((1, 1, n_mels, n_frames), dtype=torch.float32)
        with torch.no_grad():
            pt_logits = model(x).cpu().numpy().astype(np.float64).ravel()
        diff = np.max(np.abs(pt_logits - rust_logits))
        print(f"PyTorch logits : {pt_logits.tolist()}")
        print(f"Rust logits    : {rust_logits.tolist()}")
        print(f"max |diff|     : {diff:.3e}  (tolerance {args.logit_tolerance:.1e})")
        if diff > args.logit_tolerance:
            raise SystemExit(
                f"parity check failed: max |diff| {diff:.3e} exceeds {args.logit_tolerance:.1e}"
            )
        print("parity check OK — Burn checkpoint and PyTorch safetensors agree.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_cfg = OnnxExportConfig(opset=args.opset, simplify=not args.no_simplify)
    onnx_path = _export_onnx(model, (n_mels, n_frames), args.out_dir / "model_fp32.onnx", onnx_cfg)
    print(f"ONNX written to {onnx_path}")

    if args.dump_logits:
        logits = _run_forward_check(model, (n_mels, n_frames), seed=0)
        np.save(args.dump_logits, logits)
        print(f"Saved seed-0 forward logits to {args.dump_logits} (shape={logits.shape})")

    calib_dir = args.out_dir / "calib"
    existing_calib = calib_dir.is_dir() and any(calib_dir.glob("*.npy"))

    if args.espdl and not existing_calib and args.calib_windows <= 0:
        raise SystemExit(
            "--espdl needs calibration windows for PTQ. Either pass --calib-windows N "
            "with --manifest <path>, or point --out-dir at a directory that already "
            "contains calib/*.npy."
        )
    if args.espdl and args.calib_windows > 0 and not dataset_cfg.manifest_path:
        raise SystemExit("--calib-windows > 0 needs --manifest to build the calibration set.")

    if args.calib_windows > 0:
        if not dataset_cfg.manifest_path:
            print("No manifest — skipping calibration dump. "
                  "Pass --manifest to enable.", file=sys.stderr)
        else:
            calib_cfg = CalibConfig(split=args.calib_split, num_windows=args.calib_windows)
            _save_calibration_windows(
                dataset_cfg, mel_cfg, calib_dir,
                split=calib_cfg.split, num_windows=calib_cfg.num_windows, target_T=n_frames,
            )

    espdl_path: Path | None = None
    if args.espdl:
        espdl_path = (args.espdl_out or (args.out_dir / "model.espdl")).resolve()
        # Use the slimmed ONNX if onnxslim ran; otherwise the fp32 file.
        onnx_for_espdl = onnx_path.resolve()
        _run_espdl_docker(
            onnx_path=onnx_for_espdl,
            calib_dir=calib_dir.resolve(),
            out_path=espdl_path,
            n_mels=n_mels,
            n_frames=n_frames,
            target=args.espdl_target,
            num_bits=args.espdl_num_bits,
            calib_steps=args.espdl_calib_steps,
            export_test_values=args.espdl_export_test_values,
            error_report=not args.espdl_no_error_report,
        )

    out_root = args.out_dir.resolve()
    print("")
    print("=" * 72)
    print("Burn → ONNX" + (" → ESPDL" if args.espdl else "") + " export complete")
    print("=" * 72)
    print(f"  Output directory : {out_root}")
    print(f"  ONNX (fp32)      : {out_root / 'model_fp32.onnx'}")
    if not args.no_simplify:
        print(f"  ONNX (slim)      : {out_root / 'model_fp32_slim.onnx'}")
    if args.calib_windows > 0 and dataset_cfg.manifest_path:
        print(f"  Calibration data : {out_root / 'calib'}")
    if espdl_path is not None:
        print(f"  ESP-DL model     : {espdl_path}   <-- copy into firmware/models/")
    print("=" * 72)
    if espdl_path is None:
        print("To produce a .espdl: re-run with --espdl --manifest <path> "
              "--calib-windows N.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
