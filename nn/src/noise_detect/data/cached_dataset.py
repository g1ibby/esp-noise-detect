from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CachedMelDataset(Dataset[dict[str, object]]):
    """Dataset backed by precomputed log-mel tensors in shared memory.

    Items are dicts with:
      - "mel": Tensor (1, n_mels, n_frames) float32
      - "label": int
      - "file": str
      - "start_s": float
      - "end_s": float
    """

    def __init__(
        self,
        mels: Tensor,
        labels: Tensor,
        file_idx: Tensor,
        file_list: list[str],
        starts: Tensor,
        ends: Tensor,
        indices: Iterable[int],
    ) -> None:
        super().__init__()
        self._mels = mels
        self._labels = labels
        self._file_idx = file_idx
        self._file_list = file_list
        self._starts = starts
        self._ends = ends
        self._indices = list(indices)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, object]:  # type: ignore[override]
        i = self._indices[idx]
        mel = self._mels[i].to(torch.float32)  # cached as fp16, train in fp32
        return {
            "mel": mel,
            "label": int(self._labels[i].item()),
            "file": self._file_list[int(self._file_idx[i].item())],
            "start_s": float(self._starts[i].item()),
            "end_s": float(self._ends[i].item()),
        }


def collate_mels(batch: list[dict[str, object]]) -> dict[str, object]:
    mels = torch.stack([b["mel"] for b in batch if isinstance(b["mel"], Tensor)])
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    files = [str(b["file"]) for b in batch]
    starts = torch.tensor([float(b["start_s"]) for b in batch], dtype=torch.float32)
    ends = torch.tensor([float(b["end_s"]) for b in batch], dtype=torch.float32)
    return {"mel": mels, "label": labels, "file": files, "start_s": starts, "end_s": ends}
