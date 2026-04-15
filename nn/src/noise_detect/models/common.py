from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


def conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
    padding = (k // 2) if p is None else p
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class GlobalAvgPool2d(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # (B, C, H, W) -> (B, C)
        return x.mean(dim=(2, 3))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

