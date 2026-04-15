from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from lightning import LightningDataModule

from noise_detect.augment import build_pipeline
from noise_detect.config import AugmentationConfig, DatasetConfig, DataModuleConfig
from noise_detect.data.manifest import ManifestItem, default_manifest_path, load_manifest
from noise_detect.data.dataset import WindowedAudioDataset, collate_windows


class PumpAudioDataModule(LightningDataModule):
    def __init__(
        self,
        ds_cfg: DatasetConfig,
        dm_cfg: DataModuleConfig,
        augment_cfg: Optional[AugmentationConfig] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.ds_cfg = ds_cfg
        self.dm_cfg = dm_cfg
        self.augment_cfg = augment_cfg or AugmentationConfig()
        self.seed = seed
        self._manifest_items: list[ManifestItem] | None = None
        self._train = None
        self._val = None
        self._test = None

    def _manifest(self) -> list[ManifestItem]:
        if self._manifest_items is not None:
            return self._manifest_items
        if self.ds_cfg.manifest_path is not None:
            # Resolve relative to original CWD (Hydra may change working dir)
            try:
                from hydra.utils import to_absolute_path  # type: ignore
            except Exception:  # pragma: no cover
                to_absolute_path = None  # type: ignore
            if to_absolute_path is not None:
                mpath = Path(to_absolute_path(self.ds_cfg.manifest_path)).expanduser().resolve()
            else:
                mpath = Path(self.ds_cfg.manifest_path).expanduser().resolve()
        else:
            mpath = default_manifest_path()
            if mpath is None:
                raise FileNotFoundError("Could not locate manifest.jsonl; set dm.manifest_path")
        self._manifest_items = load_manifest(mpath)
        return self._manifest_items

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        items = self._manifest()
        train_aug = build_pipeline(self.augment_cfg.train)
        val_aug = build_pipeline(self.augment_cfg.val)
        test_aug = build_pipeline(self.augment_cfg.test)
        self._train = WindowedAudioDataset(
            items,
            self.ds_cfg,
            split="train",
            augmentation=train_aug,
            base_seed=self.seed,
        )
        self._val = WindowedAudioDataset(
            items,
            self.ds_cfg,
            split="val",
            augmentation=val_aug,
            base_seed=self.seed,
        )
        self._test = WindowedAudioDataset(
            items,
            self.ds_cfg,
            split="test",
            augmentation=test_aug,
            base_seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.dm_cfg.batch_size,
            shuffle=True,
            num_workers=self.dm_cfg.num_workers,
            pin_memory=self.dm_cfg.pin_memory,
            collate_fn=collate_windows,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.dm_cfg.batch_size,
            shuffle=False,
            num_workers=max(self.dm_cfg.num_workers // 2, 0),
            pin_memory=self.dm_cfg.pin_memory,
            collate_fn=collate_windows,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._test is not None
        return DataLoader(
            self._test,
            batch_size=self.dm_cfg.batch_size,
            shuffle=False,
            num_workers=max(self.dm_cfg.num_workers // 2, 0),
            pin_memory=self.dm_cfg.pin_memory,
            collate_fn=collate_windows,
        )
