from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

from noise_detect.augment import build_pipeline
from noise_detect.config import AugmentationConfig, DatasetConfig, DataModuleConfig
from noise_detect.data.cached_dataset import CachedMelDataset, collate_mels
from noise_detect.data.manifest import ManifestItem, default_manifest_path, load_manifest
from noise_detect.data.dataset import WindowedAudioDataset, collate_windows
from noise_detect.data.mel_cache import SPLIT_CODE, ensure_mel_cache, load_mel_cache


def _any_augment_enabled(cfg: AugmentationConfig) -> bool:
    return any(p.enabled and p.transforms for p in (cfg.train, cfg.val, cfg.test))


class PumpAudioDataModule(LightningDataModule):
    def __init__(
        self,
        ds_cfg: DatasetConfig,
        dm_cfg: DataModuleConfig,
        augment_cfg: Optional[AugmentationConfig] = None,
        seed: int = 42,
        mel_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.ds_cfg = ds_cfg
        self.dm_cfg = dm_cfg
        self.augment_cfg = augment_cfg or AugmentationConfig()
        self.seed = seed
        self.mel_cfg = mel_cfg  # required when dm_cfg.use_cache is True
        self._manifest_items: list[ManifestItem] | None = None
        self._train: Dataset | None = None
        self._val: Dataset | None = None
        self._test: Dataset | None = None
        self._collate = collate_windows
        self._using_cache = False

    def _manifest_path(self) -> Path:
        if self.ds_cfg.manifest_path is not None:
            try:
                from hydra.utils import to_absolute_path  # type: ignore
            except Exception:  # pragma: no cover
                to_absolute_path = None  # type: ignore
            if to_absolute_path is not None:
                return Path(to_absolute_path(self.ds_cfg.manifest_path)).expanduser().resolve()
            return Path(self.ds_cfg.manifest_path).expanduser().resolve()
        p = default_manifest_path()
        if p is None:
            raise FileNotFoundError("Could not locate manifest.jsonl; set dm.manifest_path")
        return p

    def _manifest(self) -> list[ManifestItem]:
        if self._manifest_items is not None:
            return self._manifest_items
        self._manifest_items = load_manifest(self._manifest_path())
        return self._manifest_items

    # -- setup paths --------------------------------------------------------

    def _setup_cache(self) -> None:
        if self.mel_cfg is None:
            raise ValueError(
                "PumpAudioDataModule was constructed without mel_cfg but dm.use_cache=true; "
                "pass mel_cfg=<MelConfig> or set dm.use_cache=false."
            )
        if _any_augment_enabled(self.augment_cfg):
            print(
                "[datamodule] warning: waveform augmentations are configured but dm.use_cache=true; "
                "they will be skipped. Set dm.use_cache=false to enable them."
            )

        manifest_path = self._manifest_path()
        items = self._manifest()
        cache_dir = Path(self.dm_cfg.cache_dir).expanduser().resolve() if self.dm_cfg.cache_dir else None
        cache_path = ensure_mel_cache(items, self.ds_cfg, self.mel_cfg, manifest_path, cache_dir)
        cache = load_mel_cache(cache_path)

        splits: torch.Tensor = cache["splits"]
        mels = cache["mels"]
        labels = cache["labels"]
        file_idx = cache["file_idx"]
        file_list = cache["file_list"]
        starts = cache["starts"]
        ends = cache["ends"]

        train_idx = (splits == SPLIT_CODE["train"]).nonzero(as_tuple=True)[0].tolist()
        val_idx = (splits == SPLIT_CODE["val"]).nonzero(as_tuple=True)[0].tolist()
        test_idx = (splits == SPLIT_CODE["test"]).nonzero(as_tuple=True)[0].tolist()

        def _mk(indices: list[int]) -> CachedMelDataset:
            return CachedMelDataset(
                mels=mels,
                labels=labels,
                file_idx=file_idx,
                file_list=file_list,
                starts=starts,
                ends=ends,
                indices=indices,
            )

        self._train = _mk(train_idx)
        self._val = _mk(val_idx)
        self._test = _mk(test_idx)
        self._collate = collate_mels
        self._using_cache = True

        print(
            f"[datamodule] cache-backed splits: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"(mels={tuple(mels.shape)})"
        )

    def _setup_waveform(self) -> None:
        items = self._manifest()
        sr = self.ds_cfg.sample_rate
        train_aug = build_pipeline(self.augment_cfg.train, sample_rate=sr)
        val_aug = build_pipeline(self.augment_cfg.val, sample_rate=sr)
        test_aug = build_pipeline(self.augment_cfg.test, sample_rate=sr)
        self._train = WindowedAudioDataset(
            items, self.ds_cfg, split="train", augmentation=train_aug, base_seed=self.seed,
        )
        self._val = WindowedAudioDataset(
            items, self.ds_cfg, split="val", augmentation=val_aug, base_seed=self.seed,
        )
        self._test = WindowedAudioDataset(
            items, self.ds_cfg, split="test", augmentation=test_aug, base_seed=self.seed,
        )
        self._collate = collate_windows
        self._using_cache = False

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if self.dm_cfg.use_cache:
            self._setup_cache()
        else:
            self._setup_waveform()

    @property
    def using_cache(self) -> bool:
        return self._using_cache

    # -- loaders ------------------------------------------------------------

    def _loader(self, ds: Dataset, shuffle: bool, workers: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.dm_cfg.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=self.dm_cfg.pin_memory,
            collate_fn=self._collate,
            persistent_workers=workers > 0,
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train is not None
        return self._loader(self._train, shuffle=True, workers=self.dm_cfg.num_workers)

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val is not None
        return self._loader(self._val, shuffle=False, workers=max(self.dm_cfg.num_workers // 2, 0))

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._test is not None
        return self._loader(self._test, shuffle=False, workers=max(self.dm_cfg.num_workers // 2, 0))
