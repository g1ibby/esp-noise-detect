#!/usr/bin/env python3
"""Generate tests/fixtures/mininet_calib.json and mininet.espdl oracles.

Inputs are produced by:

    MININET_ORACLE_DUMP=/tmp/mininet-oracle/dump.json \
      cargo test -p burn-espdl-export --test generate_mininet_oracle_dump \
      -- --ignored --nocapture

This script is meant to run inside the esp-ppq Docker image. It builds an
ONNX model directly from the dumped post-fold/post-fuse IR, writes the same
32 calibration windows as `.npy`, runs `espdl_quantize_onnx`, then extracts
the minimal per-tensor `{scale, exponent}` oracle consumed by
`tests/calibration_parity.rs`. When `--espdl-out` is supplied, it also copies
the generated esp-ppq `.espdl` fixture consumed by Step 5 parity tests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from torch.utils.data import DataLoader, Dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate MiniNet esp-ppq calibration oracle")
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--espdl-out", type=Path)
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--num-bits", default=8, type=int)
    args = ap.parse_args()

    payload = json.loads(args.dump.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    onnx_path = args.work_dir / "mininet.onnx"
    espdl_path = args.work_dir / "mininet.espdl"
    calib_dir = args.work_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)

    build_onnx(payload, onnx_path)
    write_calibration_windows(payload, calib_dir)
    run_esp_ppq(payload, onnx_path, espdl_path, calib_dir, args.target, args.num_bits)

    oracle = extract_oracle(payload, espdl_path.with_suffix(".json"))
    args.out.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    if args.espdl_out is not None:
        args.espdl_out.parent.mkdir(parents=True, exist_ok=True)
        args.espdl_out.write_bytes(espdl_path.read_bytes())
        print(f"wrote {args.espdl_out}")
    return 0


def build_onnx(payload: dict[str, Any], out: Path) -> None:
    input_shape = list(payload["input_shape"])
    input_name = payload["input_name"]
    output_name = payload["output_name"]

    nodes = []
    initializers = []
    value_infos = []
    current_shape = input_shape[:]

    for layer in payload["layers"]:
        kind = layer["type"]
        if kind == "Conv2d":
            output = layer["output"]
            raw_output = output + ".conv_raw" if layer.get("activation") == "Relu" else output
            weight_name = output + ".weight"
            bias_name = output + ".bias"
            w = np.asarray(layer["weight"]["data"], dtype=np.float32).reshape(layer["weight"]["shape"])
            b = np.asarray(layer["bias"]["data"], dtype=np.float32).reshape(layer["bias"]["shape"])
            initializers.append(numpy_helper.from_array(w, name=weight_name))
            initializers.append(numpy_helper.from_array(b, name=bias_name))
            pads = [
                int(layer["padding"][0]),
                int(layer["padding"][1]),
                int(layer["padding"][2]),
                int(layer["padding"][3]),
            ]
            strides = [int(v) for v in layer["stride"]]
            dilations = [int(v) for v in layer["dilation"]]
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=[layer["input"], weight_name, bias_name],
                    outputs=[raw_output],
                    name=output + ".conv",
                    pads=pads,
                    strides=strides,
                    dilations=dilations,
                    group=int(layer["groups"]),
                )
            )
            current_shape = conv_out_shape(current_shape, list(w.shape), pads, strides, dilations)
            if raw_output != output:
                nodes.append(helper.make_node("Relu", inputs=[raw_output], outputs=[output], name=output + ".relu"))
            value_infos.append(tensor_value_info(output, current_shape))
        elif kind == "ReduceMean":
            axes = [int(v) for v in layer["axes"]]
            keepdims = 1 if layer["keepdims"] else 0
            nodes.append(
                helper.make_node(
                    "ReduceMean",
                    inputs=[layer["input"]],
                    outputs=[layer["output"]],
                    name=layer["output"] + ".reduce_mean",
                    axes=axes,
                    keepdims=keepdims,
                )
            )
            if keepdims:
                for axis in axes:
                    current_shape[axis] = 1
            else:
                current_shape = [d for idx, d in enumerate(current_shape) if idx not in set(axes)]
            value_infos.append(tensor_value_info(layer["output"], current_shape))
        elif kind == "Linear":
            weight_name = layer["output"] + ".weight"
            bias_name = layer["output"] + ".bias"
            w = np.asarray(layer["weight"]["data"], dtype=np.float32).reshape(layer["weight"]["shape"])
            b = np.asarray(layer["bias"]["data"], dtype=np.float32).reshape(layer["bias"]["shape"])
            initializers.append(numpy_helper.from_array(w, name=weight_name))
            initializers.append(numpy_helper.from_array(b, name=bias_name))
            nodes.append(
                helper.make_node(
                    "Gemm",
                    inputs=[layer["input"], weight_name, bias_name],
                    outputs=[layer["output"]],
                    name=layer["output"] + ".gemm",
                    alpha=1.0,
                    beta=1.0,
                    transB=0,
                )
            )
            current_shape = [current_shape[0], int(w.shape[1])]
            value_infos.append(tensor_value_info(layer["output"], current_shape))
        else:
            raise ValueError(f"unsupported layer type {kind}")

    graph = helper.make_graph(
        nodes,
        "mininet",
        [tensor_value_info(input_name, input_shape)],
        [tensor_value_info(output_name, current_shape)],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="burn-espdl-export-mininet-oracle",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"wrote {out}")


def conv_out_shape(
    input_shape: list[int],
    weight_shape: list[int],
    pads: list[int],
    strides: list[int],
    dilations: list[int],
) -> list[int]:
    n, _, h, w = input_shape
    out_c, _, kh, kw = weight_shape
    oh = ((h + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) // strides[0]) + 1
    ow = ((w + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) // strides[1]) + 1
    return [n, out_c, oh, ow]


def tensor_value_info(name: str, shape: list[int]) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [int(v) for v in shape])


def write_calibration_windows(payload: dict[str, Any], calib_dir: Path) -> None:
    _, _, h, w = payload["input_shape"]
    for idx, window in enumerate(payload["calibration_windows"]):
        arr = np.asarray(window, dtype=np.float32).reshape((h, w))
        np.save(calib_dir / f"calib_{idx:04d}.npy", arr)
    print(f"wrote {len(payload['calibration_windows'])} calibration windows to {calib_dir}")


class NpyWindowDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.files = sorted(root.glob("*.npy"))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        arr = np.load(self.files[idx]).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def run_esp_ppq(
    payload: dict[str, Any],
    onnx_path: Path,
    espdl_path: Path,
    calib_dir: Path,
    target: str,
    num_bits: int,
) -> None:
    from esp_ppq.api import espdl_quantize_onnx

    ds = NpyWindowDataset(calib_dir)
    if len(ds) == 0:
        raise RuntimeError(f"no calibration windows in {calib_dir}")

    def dl_collate(batch: list[torch.Tensor]) -> torch.Tensor:
        return batch[0].to(torch.float32)

    def ppq_collate(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)

    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=dl_collate)
    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=dl,
        calib_steps=len(ds),
        input_shape=[int(v) for v in payload["input_shape"]],
        inputs=None,
        target=target,
        num_of_bits=num_bits,
        collate_fn=ppq_collate,
        device="cpu",
        error_report=False,
        skip_export=False,
        export_config=True,
        export_test_values=False,
        verbose=1,
    )
    print(f"wrote {espdl_path}")


def extract_oracle(payload: dict[str, Any], ppq_json: Path) -> dict[str, dict[str, float | int]]:
    data = json.loads(ppq_json.read_text())
    values = data["values"]
    by_var: dict[str, dict[str, float | int]] = {}

    expected = expected_names(payload)
    for op in data["configs"].values():
        for name, cfg in op.items():
            if name not in expected:
                continue
            value = values.get(str(cfg["hash"]))
            if value is None:
                continue
            scale = scalar(value["scale"])
            by_var[name] = {"scale": scale, "exponent": exponent(scale)}

    # esp-ppq treats Conv/Gemm bias as passive. Its exported JSON may
    # omit a standalone scale, so derive it from the same active configs
    # the Rust runner uses: scale_bias = scale_input * scale_weight.
    for layer in payload["layers"]:
        if layer["type"] not in ("Conv2d", "Linear") or layer["bias"] is None:
            continue
        bias_name = layer["output"] + ".bias"
        if bias_name in by_var:
            continue
        input_name = layer["input"]
        weight_name = layer["output"] + ".weight"
        if input_name in by_var and weight_name in by_var:
            scale = float(by_var[input_name]["scale"]) * float(by_var[weight_name]["scale"])
            by_var[bias_name] = {"scale": scale, "exponent": exponent(scale)}

    missing = sorted(expected - set(by_var))
    if missing:
        raise RuntimeError(
            "esp-ppq oracle is missing expected tensors:\n  " + "\n  ".join(missing)
        )
    return dict(sorted(by_var.items()))


def expected_names(payload: dict[str, Any]) -> set[str]:
    names = {payload["input_name"]}
    for layer in payload["layers"]:
        names.add(layer["output"])
        if layer["type"] in ("Conv2d", "Linear"):
            names.add(layer["output"] + ".weight")
            if layer["bias"] is not None:
                names.add(layer["output"] + ".bias")
    return names


def scalar(value: Any) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected scalar/list-of-one scale, got {value}")
        value = value[0]
    return float(value)


def exponent(scale: float) -> int:
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError(f"invalid scale {scale}")
    return int(math.log2(scale))


if __name__ == "__main__":
    raise SystemExit(main())
