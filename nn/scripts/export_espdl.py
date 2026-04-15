#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _parse_shape(arg: List[int | str]) -> Tuple[int, int, int, int]:
    vals = [int(x) for x in arg]
    if len(vals) != 4:
        raise SystemExit("--input-shape must provide exactly 4 integers: N C H W")
    return tuple(vals)  # type: ignore[return-value]


class NpyMelDataset(Dataset):
    def __init__(self, root: Path, n_mels: int, n_frames: int) -> None:
        self.files = sorted((root).glob("*.npy"))
        self.n_mels = int(n_mels)
        self.n_frames = int(n_frames)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        p = self.files[idx]
        x = np.load(p).astype(np.float32)  # (M,T)
        t = torch.from_numpy(x)  # (M,T)
        # expect exact shape; minor mismatch will center-crop or pad with zeros
        if t.shape != (self.n_mels, self.n_frames):
            t = _center_pad_or_crop(t.unsqueeze(0).unsqueeze(0), self.n_frames).squeeze(0).squeeze(0)
        return t.unsqueeze(0).unsqueeze(0)  # (1,1,M,T)


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Export ESP-DL (.espdl) using esp-ppq")
    ap.add_argument("--onnx", required=True, type=str, help="Path to ONNX model")
    ap.add_argument("--out", required=True, type=str, help="Output .espdl file path")
    ap.add_argument("--calib-dir", required=True, type=str, help="Directory with .npy calibration windows")
    ap.add_argument(
        "--input-shape", nargs=4, metavar=("N", "C", "H", "W"), required=True, help="Fixed input shape"
    )
    ap.add_argument("--target", default="esp32s3", choices=["c", "esp32s3", "esp32p4"])
    ap.add_argument("--num-bits", default=8, type=int, choices=[8, 16])
    ap.add_argument("--calib-steps", default=64, type=int)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])  # container is CPU
    ap.add_argument("--verbose", default=1, type=int)
    ap.add_argument("--export-test-values", action="store_true")
    ap.add_argument("--no-error-report", action="store_true")

    args = ap.parse_args()
    input_shape = _parse_shape(args.input_shape)
    _, _, n_mels, n_frames = input_shape

    from esp_ppq.api import espdl_quantize_onnx

    # Build DataLoader over .npy calibration windows
    ds = NpyMelDataset(Path(args.calib_dir), n_mels=n_mels, n_frames=n_frames)
    if len(ds) == 0:
        raise SystemExit(f"No .npy files found under {args.calib_dir}")

    def dl_collate(batch):
        # DataLoader collate: unwrap the single sample from the length-1 list.
        return batch[0].to(torch.float32).to(args.device)

    def ppq_collate(x):
        # esp-ppq calls this again on the tensor produced by the DataLoader.
        # It must be idempotent: if we strip a dim here we break the 4D NCHW shape.
        return x.to(torch.float32).to(args.device)

    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=dl_collate)
    steps = max(1, min(int(args.calib_steps), len(dl)))

    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    espdl_quantize_onnx(
        onnx_import_file=str(Path(args.onnx)),
        espdl_export_file=str(Path(args.out)),
        calib_dataloader=dl,
        calib_steps=steps,
        input_shape=list(input_shape),
        inputs=None,
        target=str(args.target),
        num_of_bits=int(args.num_bits),
        collate_fn=ppq_collate,
        device=str(args.device),
        error_report=not bool(args.no_error_report),
        skip_export=False,
        export_test_values=bool(args.export_test_values),
        verbose=int(args.verbose),
    )

    print(f"[esp-ppq] Exported ESP-DL model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

