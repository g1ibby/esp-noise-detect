from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from noise_detect.config import TinyConvConfig
from noise_detect.models.common import GlobalAvgPool2d, conv_bn_relu


class TinyConv(nn.Module):
    """Compact 2D CNN for log-mel inputs.

    Input: (B, 1, n_mels, T)
    Output: logits (B, 2)
    """

    def __init__(self, cfg: TinyConvConfig, n_classes: int = 2) -> None:
        super().__init__()
        chs: Sequence[int] = cfg.channels
        if len(chs) < 2:
            raise ValueError("TinyConvConfig.channels must have at least 2 entries")

        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(chs):
            stride = 2 if i < len(chs) - 1 else 1  # downsample early
            layers.append(conv_bn_relu(in_ch, out_ch, k=5, s=stride))
            layers.append(conv_bn_relu(out_ch, out_ch, k=3, s=1))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(p=cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.dropout(x)
        return self.head(x)

