from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info

from noise_detect.augment import apply_pipeline, set_pipeline_seed

if TYPE_CHECKING:
    from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
else:  # pragma: no cover - typing only
    BaseWaveformTransform = Any
from noise_detect.config import DatasetConfig
from noise_detect.data.manifest import ManifestItem
from noise_detect.features.utils import (
    ensure_float32,
    mix_down_to_mono,
    resample_if_needed,
    seconds_to_samples,
)


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class WindowIndex:
    file_idx: int
    window_idx: int


class WindowedAudioDataset(Dataset[dict[str, object]]):
    """Dataset producing fixed-length waveform windows and labels.

    Items are dictionaries with keys:
      - "waveform": Tensor (T,) float32
      - "label": int (0 or 1)
      - "file": str (path)
      - "start_s": float
      - "end_s": float
    """

    def __init__(
        self,
        items: Sequence[ManifestItem],
        cfg: DatasetConfig,
        split: SplitName,
        augmentation: Optional[BaseWaveformTransform] = None,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        self.items = list(item for item in items if (item.split or "train") == split)
        self.cfg = cfg
        self.split = split
        self.class_to_idx = {name: i for i, name in enumerate(cfg.class_names)}
        self._augmentation = augmentation
        self._base_seed = int(base_seed)
        self._current_seed: int | None = None

        self.window_samples, self.hop_samples = seconds_to_samples(
            cfg.window_s, cfg.hop_s, cfg.sample_rate
        )

        # Precompute window counts from audio metadata where possible
        self._indices: list[WindowIndex] = []
        for file_idx, it in enumerate(self.items):
            n_windows = self._estimate_num_windows(it)
            if n_windows <= 0:
                n_windows = 1  # ensure at least one window per file with padding later
            for w in range(n_windows):
                self._indices.append(WindowIndex(file_idx=file_idx, window_idx=w))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._indices)

    def _estimate_num_windows(self, it: ManifestItem) -> int:
        # Use torchaudio.info to get frames and samplerate without decoding
        try:
            import torchaudio  # type: ignore

            info = torchaudio.info(str(it.audio_path))
            frames = int(info.num_frames)
            src_sr = int(info.sample_rate)
        except Exception:
            return 1

        if src_sr <= 0 or frames <= 0:
            return 1

        # Estimate frames after resample if needed
        if src_sr != self.cfg.sample_rate:
            frames = int(round(frames * (self.cfg.sample_rate / float(src_sr))))

        if frames < self.window_samples:
            return 1
        remainder = max(frames - self.window_samples, 0)
        n = 1 + remainder // self.hop_samples
        if remainder % self.hop_samples != 0:
            n += 1  # final partial padded window
        return n

    def _load_file(self, path: Path) -> tuple[Tensor, int]:
        import torchaudio  # type: ignore

        x, sr = torchaudio.load(str(path))
        x = ensure_float32(x)
        x = mix_down_to_mono(x)
        return x, int(sr)

    def __getitem__(self, idx: int) -> dict[str, object]:  # type: ignore[override]
        wi = self._indices[idx]
        it = self.items[wi.file_idx]
        x, sr = self._load_file(it.audio_path)
        if sr != self.cfg.sample_rate:
            x = resample_if_needed(x, sr, self.cfg.sample_rate)

        start = wi.window_idx * self.hop_samples
        end = start + self.window_samples

        if end <= x.shape[0]:
            window = x[start:end]
        else:
            window = x.new_zeros((self.window_samples,))
            part = x[start:]
            window[: part.shape[0]] = part

        window = self._maybe_augment(window)

        label_idx = self.class_to_idx.get(it.label, 0)
        return {
            "waveform": window,
            "label": int(label_idx),
            "file": str(it.audio_path),
            "start_s": float(start / self.cfg.sample_rate),
            "end_s": float(min(end, x.shape[0]) / self.cfg.sample_rate),
        }

    def _maybe_augment(self, window: Tensor) -> Tensor:
        if self._augmentation is None:
            return window
        worker = get_worker_info()
        if worker is None:
            if self._current_seed is None:
                set_pipeline_seed(self._augmentation, self._base_seed)
                self._current_seed = self._base_seed
        else:
            if self._current_seed != worker.seed:
                set_pipeline_seed(self._augmentation, worker.seed)
                self._current_seed = worker.seed
        return apply_pipeline(self._augmentation, window, self.cfg.sample_rate)


def collate_windows(batch: list[dict[str, object]]) -> dict[str, object]:
    waveforms = torch.stack([b["waveform"] for b in batch if isinstance(b["waveform"], Tensor)])
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    files = [str(b["file"]) for b in batch]
    start_s = torch.tensor([float(b["start_s"]) for b in batch], dtype=torch.float32)
    end_s = torch.tensor([float(b["end_s"]) for b in batch], dtype=torch.float32)
    return {"waveform": waveforms, "label": labels, "file": files, "start_s": start_s, "end_s": end_s}
