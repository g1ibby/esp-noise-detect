from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import Tensor, nn

import hydra
from omegaconf import DictConfig, OmegaConf

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassF1Score,
)

from noise_detect.config import (
    TrainAppConfig,
    DatasetConfig,
    MelConfig,
    TinyConvConfig,
    DataModuleConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    load_augmentation_config,
)
from noise_detect.data.datamodule import PumpAudioDataModule
from noise_detect.features.mel import MelExtractor
from noise_detect.models.tinyconv import TinyConv
from noise_detect.eval import evaluate as eval_model


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class LitTinyConv(LightningModule):
    def __init__(
        self,
        ds_cfg: DatasetConfig,
        mel_cfg: MelConfig,
        model_cfg: TinyConvConfig,
        optim_cfg: OptimizerConfig,
        sched_cfg: SchedulerConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "dataset": asdict(ds_cfg),
                "mel": asdict(mel_cfg),
                "model": asdict(model_cfg),
                "optim": asdict(optim_cfg),
                "sched": asdict(sched_cfg),
            }
        )
        self.ds_cfg = ds_cfg
        self.mel_cfg = mel_cfg
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.model = TinyConv(model_cfg, n_classes=len(ds_cfg.class_names))
        # Registered submodule: Lightning moves its filterbank buffers to the
        # training device. Used only when batches carry raw waveforms; batches
        # from the mel cache come as (B, 1, n_mels, n_frames) already.
        self.mel_extractor = MelExtractor(mel_cfg, ds_cfg.sample_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.val_f1_macro = MulticlassF1Score(num_classes=2, average="macro")
        self.val_acc_macro = MulticlassAccuracy(num_classes=2, average="macro")
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _step(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        labels: Tensor = batch["label"]
        if "mel" in batch:
            mel = batch["mel"].to(self.device, non_blocking=True)
        else:
            wave: Tensor = batch["waveform"].to(self.device, non_blocking=True)
            mel = self.mel_extractor(wave)
        logits = self.forward(mel)
        loss = self.criterion(logits, labels.to(self.device))
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, probs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # type: ignore[override]
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:  # type: ignore[override]
        loss, preds, probs = self._step(batch)
        y = batch["label"].to(self.device)
        self.val_f1_macro.update(preds, y)
        self.val_acc_macro.update(preds, y)
        self.val_auroc.update(probs, y)
        self.val_auprc.update(probs, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=False)

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        f1 = self.val_f1_macro.compute()
        acc = self.val_acc_macro.compute()
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        self.log("val_f1_macro", f1, prog_bar=True)
        self.log("val_acc_macro", acc)
        self.log("val_auroc", auroc)
        self.log("val_auprc", auprc)
        self.val_f1_macro.reset()
        self.val_acc_macro.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()

    def configure_optimizers(self):  # type: ignore[override]
        if self.optim_cfg.name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.optim_cfg.lr,
                momentum=self.optim_cfg.momentum,
                weight_decay=self.optim_cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay
            )
        if self.sched_cfg.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.sched_cfg.t_max, eta_min=self.sched_cfg.min_lr
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


def _to_train_config(dc: DictConfig) -> TrainAppConfig:  # type: ignore[name-defined]
    d = OmegaConf.to_container(dc, resolve=True)  # type: ignore[name-defined]
    assert isinstance(d, dict)
    dataset = DatasetConfig(**d["dataset"])  # type: ignore[arg-type]
    mel = MelConfig(**d["features"])  # type: ignore[arg-type]
    model = TinyConvConfig(**d["model"])  # type: ignore[arg-type]
    augment_cfg = load_augmentation_config(d.get("augment"))
    dm = DataModuleConfig(**d["dm"])  # type: ignore[arg-type]
    optim = OptimizerConfig(**d["optim"])  # type: ignore[arg-type]
    sched = SchedulerConfig(**d["sched"])  # type: ignore[arg-type]
    trainer = TrainerConfig(**d["trainer"])  # type: ignore[arg-type]
    return TrainAppConfig(
        dataset=dataset,
        mel=mel,
        model=model,
        augment=augment_cfg,
        dm=dm,
        optim=optim,
        sched=sched,
        trainer=trainer,
    )


def _build_trainer(cfg: TrainerConfig) -> Trainer:
    callbacks = [
        ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="best", dirpath="checkpoints"),
        EarlyStopping(monitor="val_f1_macro", mode="max", patience=8),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip_val,
        deterministic=cfg.deterministic,
        callbacks=callbacks,
        log_every_n_steps=25,
    )
    return trainer


def _run_post_train_eval(best_ckpt: Path, tc: TrainAppConfig) -> None:
    if not best_ckpt.exists():
        print(f"Best checkpoint not found at {best_ckpt}; skipping post-train eval")
        return

    eval_cfg = {
        "dataset": asdict(tc.dataset),
        "features": asdict(tc.mel),
        "model": asdict(tc.model),
        "dm": asdict(tc.dm),
        "checkpoint": str(best_ckpt),
        "split": "val",
        "aggregate": "mean",
        "calibrate": True,
        "window_metrics": True,
        "metrics_json": str(best_ckpt.parent / "metrics.json"),
        "calibration_json": str(best_ckpt.parent / "calibration.json"),
    }

    cfg = OmegaConf.create(eval_cfg)
    print("Running post-train evaluation on val split...")
    eval_model(cfg)  # writes threshold.json, calibration.json, metrics.json


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def _hydra_entry(cfg: DictConfig) -> int:
    tc = _to_train_config(cfg)
    seed_everything(tc.trainer.seed, workers=True)
    dm = PumpAudioDataModule(tc.dataset, tc.dm, tc.augment, tc.trainer.seed, mel_cfg=tc.mel)
    model = LitTinyConv(tc.dataset, tc.mel, tc.model, tc.optim, tc.sched)
    trainer = _build_trainer(tc.trainer)
    trainer.fit(model, datamodule=dm)
    # Print best checkpoint path for convenience
    from lightning.pytorch.callbacks import ModelCheckpoint as _MC

    best_path = None
    for cb in trainer.callbacks:
        if isinstance(cb, _MC):
            best_path = cb.best_model_path
            break
    if best_path:
        best_ckpt_path = Path(best_path).resolve()
        print(f"Best checkpoint: {best_ckpt_path}")
        _run_post_train_eval(best_ckpt_path, tc)

        run_dir = best_ckpt_path.parent.parent  # runs/<timestamp>/
        ckpt_dir = best_ckpt_path.parent
        print("")
        print("=" * 72)
        print("Training complete")
        print("=" * 72)
        print(f"  Run directory    : {run_dir}")
        print(f"  Best checkpoint  : {best_ckpt_path}")
        metrics_file = ckpt_dir / "metrics.json"
        thr_file = ckpt_dir / "threshold.json"
        cal_file = ckpt_dir / "calibration.json"
        if metrics_file.exists():
            print(f"  Metrics JSON     : {metrics_file}")
        if thr_file.exists():
            print(f"  Threshold JSON   : {thr_file}")
        if cal_file.exists():
            print(f"  Calibration JSON : {cal_file}")
        print("")
        print("Next steps:")
        print(f"  Evaluate on test : uv run -m noise_detect.eval \\")
        print(f"                       checkpoint={best_ckpt_path} \\")
        if tc.dataset.manifest_path:
            print(f"                       dataset.manifest_path={tc.dataset.manifest_path} \\")
        print(f"                       split=test calibrate=false")
        print(f"  Export .espdl    : uv run -m noise_detect.export \\")
        print(f"                       checkpoint={best_ckpt_path}" + (" \\" if tc.dataset.manifest_path else ""))
        if tc.dataset.manifest_path:
            print(f"                       dataset.manifest_path={tc.dataset.manifest_path}")
        print("=" * 72)
    return 0


def main() -> int:
    return _hydra_entry()


if __name__ == "__main__":
    raise SystemExit(main())
