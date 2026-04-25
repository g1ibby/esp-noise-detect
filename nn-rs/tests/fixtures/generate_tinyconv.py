#!/usr/bin/env python3
"""
Generate a TinyConv forward-parity fixture for the Step 9 parity test.

Run from the `nn/` venv:

    cd nn && uv run --no-dev python \
        ../nn-rs/tests/fixtures/generate_tinyconv.py

The Python `TinyConv` (nn/src/noise_detect/models/tinyconv.py) is the
numerical ground truth for the Rust port. We randomize every parameter
of the model — including BatchNorm running mean / variance — so that
eval-mode BN actually exercises its own math (with the default
zero-mean / unit-var running stats the inference path is a near-no-op
and wouldn't catch a broken BN).

Binary layout (little-endian, packed). All floats are f32, all integers
are u32 unless stated otherwise.

    magic        : u32          "TCNV" = 0x564E4354
    version      : u32          = 1
    n_classes    : u32
    n_mels       : u32          H of mel input
    n_frames     : u32          W of mel input
    batch        : u32          number of inputs / outputs
    n_blocks     : u32          len(channels)
    bn_eps       : f32          BatchNorm epsilon
    channels     : u32 × n_blocks

    For each block b in 0..n_blocks with
        in_ch = 1 if b == 0 else channels[b - 1]
        out_ch = channels[b]:
        conv_down.weight : f32 × (out_ch × in_ch × 5 × 5)
        bn_down.gamma    : f32 × out_ch
        bn_down.beta     : f32 × out_ch
        bn_down.r_mean   : f32 × out_ch
        bn_down.r_var    : f32 × out_ch
        conv_refine.weight : f32 × (out_ch × out_ch × 3 × 3)
        bn_refine.gamma  : f32 × out_ch
        bn_refine.beta   : f32 × out_ch
        bn_refine.r_mean : f32 × out_ch
        bn_refine.r_var  : f32 × out_ch

    head.weight  : f32 × (n_classes × last_channel)   PyTorch layout
    head.bias    : f32 × n_classes

    input        : f32 × (batch × 1 × n_mels × n_frames)    contiguous
    output       : f32 × (batch × n_classes)                contiguous
"""

from __future__ import annotations

import pathlib
import struct
from typing import Iterable

import numpy as np
import torch
from torch import nn

from noise_detect.config import TinyConvConfig
from noise_detect.models.tinyconv import TinyConv


HERE = pathlib.Path(__file__).parent
MAGIC = 0x564E_4354  # little-endian "TCNV"


def _as_f32_bytes(t: torch.Tensor | np.ndarray) -> bytes:
    """Flatten to row-major f32, return raw bytes."""

    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    arr = np.ascontiguousarray(t, dtype=np.float32)
    return arr.tobytes()


def _randomize_model(model: TinyConv, rng: torch.Generator) -> None:
    """Replace every parameter / buffer with fresh random values.

    The PyTorch-default init already randomizes Conv / Linear weights
    and biases, but BatchNorm's running_mean (zeros) and running_var
    (ones) make eval-mode BN effectively identity. We overwrite them
    so the parity test actually exercises the BN math.
    """

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Kaiming-uniform is already applied, but we re-sample under
            # the caller's generator so fixtures are reproducible run-to-
            # run regardless of torch's global init order.
            bound = 1.0 / float(np.sqrt(module.weight.numel() / module.weight.shape[0]))
            module.weight.data.uniform_(-bound, bound, generator=rng)
            if getattr(module, "bias", None) is not None:
                module.bias.data.uniform_(-bound, bound, generator=rng)
        elif isinstance(module, nn.BatchNorm2d):
            n = module.num_features
            # gamma ~ N(1, 0.1), beta ~ N(0, 0.1) — close to init but
            # non-default so the affine params actually matter.
            module.weight.data = 1.0 + 0.1 * torch.randn(n, generator=rng)
            module.bias.data = 0.1 * torch.randn(n, generator=rng)
            # running_mean ~ N(0, 0.5), running_var ~ positive around 1.
            module.running_mean = 0.5 * torch.randn(n, generator=rng)
            module.running_var = torch.abs(torch.randn(n, generator=rng)) + 0.5


def _blocks(model: TinyConv) -> list[tuple[nn.Conv2d, nn.BatchNorm2d, nn.Conv2d, nn.BatchNorm2d]]:
    """Group the flat `features` Sequential into (conv5, bn5, conv3, bn3)
    tuples, one per TinyConv block. Mirrors the Rust `TinyConvBlock`
    layout so the fixture reader can consume them in order."""

    # `TinyConv.features` = Sequential(conv_bn_relu_5, conv_bn_relu_3,
    # conv_bn_relu_5, conv_bn_relu_3, ...) where each conv_bn_relu is
    # itself Sequential(Conv2d, BatchNorm2d, ReLU).
    convs: list[nn.Conv2d] = []
    bns: list[nn.BatchNorm2d] = []
    for inner in model.features:
        assert isinstance(inner, nn.Sequential)
        conv, bn, _relu = inner
        assert isinstance(conv, nn.Conv2d)
        assert isinstance(bn, nn.BatchNorm2d)
        convs.append(conv)
        bns.append(bn)
    # Two conv_bn_relu per block (downsample + refine).
    assert len(convs) % 2 == 0
    return [
        (convs[2 * i], bns[2 * i], convs[2 * i + 1], bns[2 * i + 1]) for i in range(len(convs) // 2)
    ]


def _pack(f, *channels: int) -> None:
    for ch in channels:
        f.write(struct.pack("<I", ch))


def _serialize(
    path: pathlib.Path,
    *,
    model: TinyConv,
    cfg: TinyConvConfig,
    n_classes: int,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> None:
    n_mels, n_frames = inputs.shape[2], inputs.shape[3]
    batch = inputs.shape[0]
    bn_eps = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d)).eps

    blocks = _blocks(model)
    assert len(blocks) == len(cfg.channels)

    with path.open("wb") as f:
        f.write(struct.pack("<II", MAGIC, 1))  # magic + version
        _pack(f, n_classes, n_mels, n_frames, batch, len(cfg.channels))
        f.write(struct.pack("<f", float(bn_eps)))
        _pack(f, *cfg.channels)

        for (conv_down, bn_down, conv_refine, bn_refine) in blocks:
            f.write(_as_f32_bytes(conv_down.weight))
            f.write(_as_f32_bytes(bn_down.weight))
            f.write(_as_f32_bytes(bn_down.bias))
            f.write(_as_f32_bytes(bn_down.running_mean))
            f.write(_as_f32_bytes(bn_down.running_var))
            f.write(_as_f32_bytes(conv_refine.weight))
            f.write(_as_f32_bytes(bn_refine.weight))
            f.write(_as_f32_bytes(bn_refine.bias))
            f.write(_as_f32_bytes(bn_refine.running_mean))
            f.write(_as_f32_bytes(bn_refine.running_var))

        assert isinstance(model.head, nn.Linear)
        f.write(_as_f32_bytes(model.head.weight))  # [n_classes, C_last]
        f.write(_as_f32_bytes(model.head.bias))

        f.write(_as_f32_bytes(inputs))
        f.write(_as_f32_bytes(outputs))

    print(
        f"wrote {path.name}: channels={cfg.channels} n_classes={n_classes} "
        f"batch={batch} n_mels={n_mels} n_frames={n_frames} bn_eps={bn_eps}"
    )


def main() -> None:
    torch.set_grad_enabled(False)

    cfg = TinyConvConfig(channels=[16, 32, 64], dropout=0.0)
    n_classes = 2
    batch = 2
    # Robust-session shape: 1 s of 32 kHz → n_fft=1024, hop=320 → 101
    # frames; MelConfig default gives 64 mel bands.
    n_mels = 64
    n_frames = 101

    # Seeded PRNG used for every random draw in this script. Using a
    # torch.Generator means the stream is reproducible independent of
    # torch's global RNG state.
    rng = torch.Generator().manual_seed(0xC0DE_F00D)

    model = TinyConv(cfg, n_classes=n_classes)
    _randomize_model(model, rng)
    model.eval()

    inputs = torch.randn(batch, 1, n_mels, n_frames, generator=rng)
    outputs = model(inputs)

    _serialize(
        HERE / "tinyconv_forward.bin",
        model=model,
        cfg=cfg,
        n_classes=n_classes,
        inputs=inputs,
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
