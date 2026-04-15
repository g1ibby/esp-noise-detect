from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from noise_detect.config import (
    DatasetConfig,
    EspdlExportConfig,
    ExportAppConfig,
    MelConfig,
    OnnxExportConfig,
    TinyConvConfig,
)
from noise_detect.data.dataset import WindowedAudioDataset
from noise_detect.data.manifest import default_manifest_path, load_manifest
from noise_detect.features.mel import compute_mel
from noise_detect.features.utils import seconds_to_samples
from noise_detect.models.tinyconv import TinyConv


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _abs_path(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    try:
        from hydra.utils import to_absolute_path  # type: ignore

        return Path(to_absolute_path(p)).expanduser().resolve()
    except Exception:  # pragma: no cover
        return Path(p).expanduser().resolve()


def _to_export_config(dc: DictConfig) -> ExportAppConfig:
    d = OmegaConf.to_container(dc, resolve=True)
    assert isinstance(d, dict)
    ds = DatasetConfig(**d["dataset"])  # type: ignore[arg-type]
    mel = MelConfig(**d["features"])  # type: ignore[arg-type]
    mdl = TinyConvConfig(**d["model"])  # type: ignore[arg-type]
    onnx = OnnxExportConfig(**d.get("onnx", {}))  # type: ignore[arg-type]
    esp = EspdlExportConfig(**d.get("espdl", {}))  # type: ignore[arg-type]

    from noise_detect.config import CalibConfig

    calib = CalibConfig(**d.get("calib", {}))  # type: ignore[arg-type]

    return ExportAppConfig(
        checkpoint=d.get("checkpoint"),
        out_dir=d.get("out_dir", "export"),
        dataset=ds,
        mel=mel,
        model=mdl,
        onnx=onnx,
        calib=calib,
        espdl=esp,
    )


def _compute_mel_shape(ds: DatasetConfig, mel: MelConfig) -> Tuple[int, int]:
    window_samples, _ = seconds_to_samples(ds.window_s, ds.hop_s, ds.sample_rate)
    dummy = torch.zeros((window_samples,), dtype=torch.float32)
    mel_t = compute_mel(dummy, sample_rate=ds.sample_rate, cfg=mel)  # (1,1,M,T)
    assert mel_t.dim() == 4 and mel_t.shape[1] == 1
    return int(mel_t.shape[2]), int(mel_t.shape[3])


def _build_model(mdl: TinyConvConfig, n_classes: int) -> TinyConv:
    model = TinyConv(mdl, n_classes=n_classes)
    model.eval()
    return model


def _load_checkpoint_if_any(model: TinyConv, ckpt: Optional[str]) -> None:
    if not ckpt:
        return
    p = _abs_path(ckpt)
    if p is None or not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    state = torch.load(str(p), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        sd = {
            k.split(".", 1)[-1] if k.startswith("model.") else k: v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
    elif isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        raise RuntimeError("Unsupported checkpoint format")


def _center_pad_or_crop(x: torch.Tensor, target_T: int) -> torch.Tensor:
    cur_T = x.shape[-1]
    if cur_T == target_T:
        return x
    if cur_T < target_T:
        pad = target_T - cur_T
        left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(x, (left, right), mode="constant", value=0.0)
    diff = cur_T - target_T
    left = diff // 2
    right = left + target_T
    return x[..., left:right]


def _export_onnx(
    model: TinyConv,
    mel_shape: Tuple[int, int],
    out_path: Path,
    onnx_cfg: OnnxExportConfig,
) -> Path:
    n_mels, n_frames = mel_shape
    example = torch.zeros((1, 1, n_mels, n_frames), dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
        opset_version=int(onnx_cfg.opset),
        dynamic_axes={},
    )
    onnx_path = out_path
    if onnx_cfg.simplify:
        import onnx  # type: ignore
        import onnxslim  # type: ignore

        model_onnx = onnx.load(str(out_path))
        model_slim = onnxslim.slim(model_onnx)
        slim_path = out_path.with_name(out_path.stem + "_slim.onnx")
        onnx.save(model_slim, str(slim_path))
        onnx_path = slim_path
        print(f"Slimmed ONNX with onnxslim -> {onnx_path}")
    return onnx_path


def _save_calibration_windows(
    ds: DatasetConfig,
    mel: MelConfig,
    out_dir: Path,
    split: str,
    num_windows: int,
    target_T: int,
) -> list[Path]:
    if ds.manifest_path:
        manifest = _abs_path(ds.manifest_path)
        if manifest is None or not manifest.exists():
            raise FileNotFoundError(f"manifest.jsonl not found at {ds.manifest_path}")
    else:
        manifest = default_manifest_path()
        if manifest is None:
            raise FileNotFoundError(
                "manifest.jsonl not found; set dataset.manifest_path=/path/to/manifest.jsonl"
            )
    items = [it for it in load_manifest(manifest) if (it.split or "train") == split]
    if not items:
        raise RuntimeError(f"No items found for split={split}")
    ds_win = WindowedAudioDataset(items, ds, split=split)
    out = []
    saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    for idx in range(len(ds_win)):
        batch = ds_win[idx]
        wave = batch["waveform"]
        assert isinstance(wave, torch.Tensor)
        mel_t = compute_mel(
            wave, sample_rate=ds.sample_rate, cfg=mel
        ).contiguous()  # (1,1,M,T)
        mel_t = _center_pad_or_crop(mel_t, target_T)
        npy = mel_t.squeeze(0).squeeze(0).cpu().numpy()
        p = out_dir / f"calib_{saved:04d}.npy"
        np.save(p, npy)
        out.append(p)
        saved += 1
        if saved >= num_windows:
            break
    print(f"Saved {saved} calibration windows to {out_dir}")
    return out


def _run_espdl_docker(
    onnx_path: Path,
    calib_dir: Path,
    out_path: Path,
    espdl_cfg: EspdlExportConfig,
    n_mels: int,
    n_frames: int,
) -> None:
    """Build and run the esp-ppq CPU Docker image to export .espdl."""
    nn_root = Path(__file__).resolve().parents[2]
    repo_root = nn_root.parent
    dockerfile = os.environ.get(
        "ESP_PPQ_DOCKERFILE", str(nn_root / "docker" / "esp-ppq-cpu.Dockerfile")
    )
    image = os.environ.get("ESP_PPQ_IMAGE", "esp-ppq:cpu")
    build_context = os.environ.get("ESP_PPQ_BUILD_CONTEXT", str(repo_root))

    build_cmd = [
        "docker",
        "build",
        "-f",
        dockerfile,
        "-t",
        image,
        build_context,
    ]
    subprocess.run(build_cmd, check=True)

    script_rel = os.path.relpath(
        str(nn_root / "scripts" / "export_espdl.py"), start=str(repo_root)
    )
    onnx_rel = os.path.relpath(str(onnx_path), start=str(repo_root))
    calib_rel = os.path.relpath(str(calib_dir), start=str(repo_root))
    out_rel = os.path.relpath(str(out_path), start=str(repo_root))

    run_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo_root}:/work",
        "-w",
        "/work",
        image,
        script_rel,
        "--onnx",
        onnx_rel,
        "--out",
        out_rel,
        "--calib-dir",
        calib_rel,
        "--input-shape",
        "1",
        "1",
        str(int(n_mels)),
        str(int(n_frames)),
        "--target",
        str(espdl_cfg.target),
        "--num-bits",
        str(int(espdl_cfg.num_bits)),
        "--calib-steps",
        str(int(espdl_cfg.calib_steps)),
        "--device",
        "cpu",
        "--verbose",
        str(int(espdl_cfg.verbose)),
    ]
    if espdl_cfg.export_test_values:
        run_cmd.append("--export-test-values")
    if not espdl_cfg.error_report:
        run_cmd.append("--no-error-report")
    subprocess.run(run_cmd, check=True)


def run_export(cfg: ExportAppConfig) -> int:
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_mels, n_frames = _compute_mel_shape(cfg.dataset, cfg.mel)

    model = _build_model(cfg.model, n_classes=len(cfg.dataset.class_names))
    _load_checkpoint_if_any(model, cfg.checkpoint)

    onnx_path = _export_onnx(
        model, (n_mels, n_frames), out_root / "model_fp32.onnx", cfg.onnx
    )

    calib_dir = out_root / "calib"
    _save_calibration_windows(
        cfg.dataset,
        cfg.mel,
        calib_dir,
        split=cfg.calib.split,
        num_windows=cfg.calib.num_windows,
        target_T=n_frames,
    )

    if cfg.espdl.enabled:
        _run_espdl_docker(
            onnx_path=onnx_path,
            calib_dir=calib_dir,
            out_path=out_root / "model.espdl",
            espdl_cfg=cfg.espdl,
            n_mels=n_mels,
            n_frames=n_frames,
        )

    out_root_abs = out_root.resolve()
    espdl_path = out_root_abs / "model.espdl"
    onnx_fp32 = out_root_abs / "model_fp32.onnx"
    onnx_slim = out_root_abs / "model_fp32_slim.onnx"

    print("")
    print("=" * 72)
    print("Export complete")
    print("=" * 72)
    print(f"  Output directory : {out_root_abs}")
    if cfg.espdl.enabled and espdl_path.exists():
        print(f"  ESP-DL model     : {espdl_path}   <-- copy this into firmware/models/")
    if onnx_fp32.exists():
        print(f"  ONNX (fp32)      : {onnx_fp32}")
    if onnx_slim.exists():
        print(f"  ONNX (slim)      : {onnx_slim}")
    print(f"  Calibration data : {(out_root_abs / 'calib').resolve()}")
    print("=" * 72)
    return 0


@hydra.main(config_path="../../configs", config_name="export", version_base=None)
def _hydra_entry(dc: DictConfig) -> int:
    cfg = _to_export_config(dc)
    return run_export(cfg)


def main() -> int:
    return _hydra_entry()


if __name__ == "__main__":
    raise SystemExit(main())
