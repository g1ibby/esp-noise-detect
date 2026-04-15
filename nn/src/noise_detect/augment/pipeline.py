from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import torch
from torch import Tensor

from noise_detect.config import AugmentationPipelineConfig, AugmentationTransformConfig

from torch_audiomentations import (
    AddBackgroundNoise,
    AddColoredNoise,
    ApplyImpulseResponse,
    BandPassFilter,
    BandStopFilter,
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
    PolarityInversion,
    Shift,
    SpliceOut,
    TimeInversion,
)

if TYPE_CHECKING:
    from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
else:
    BaseWaveformTransform = Any


_TRANSFORM_REGISTRY: dict[str, type[BaseWaveformTransform]] = {
    "gain": Gain,
    "add_colored_noise": AddColoredNoise,
    "add_background_noise": AddBackgroundNoise,
    "bandpass": BandPassFilter,
    "bandstop": BandStopFilter,
    "highpass": HighPassFilter,
    "lowpass": LowPassFilter,
    "pitch_shift": PitchShift,
    "polarity_inversion": PolarityInversion,
    "impulse_response": ApplyImpulseResponse,
    "shift": Shift,
    "time_masking": SpliceOut,
    "time_inversion": TimeInversion,
}


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, list):
            if not value:
                normalized[key] = value
                continue
            first = value[0]
            if isinstance(first, (int, float)) and all(isinstance(v, (int, float)) for v in value):
                normalized[key] = tuple(float(v) for v in value)
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def build_pipeline(cfg: AugmentationPipelineConfig) -> Optional[BaseWaveformTransform]:
    if not cfg.enabled or not cfg.transforms:
        return None

    transforms: list[BaseWaveformTransform] = []
    for item in cfg.transforms:
        transforms.append(_build_transform(item, cfg.same_on_batch))
    if not transforms:
        return None
    return Compose(
        transforms=transforms,
        shuffle=cfg.shuffle,
        p=1.0,
        output_type="tensor",
    )


def _build_transform(
    cfg: AugmentationTransformConfig,
    same_on_batch: bool,
) -> BaseWaveformTransform:
    cls = _TRANSFORM_REGISTRY.get(cfg.type)
    if cls is None:
        raise KeyError(f"Unknown augmentation transform type: {cfg.type}")
    params = _normalize_params(cfg.params)
    if "p" not in params:
        params["p"] = cfg.probability
    try:
        import inspect

        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        sig = None
    if sig is not None:
        if same_on_batch:
            if "mode" in sig.parameters and "mode" not in params:
                params["mode"] = "per_batch"
            if "p_mode" in sig.parameters and "p_mode" not in params:
                params["p_mode"] = "per_batch"
        if "output_type" in sig.parameters and "output_type" not in params:
            params["output_type"] = "tensor"
    return cls(**params)


def apply_pipeline(
    pipeline: Optional[BaseWaveformTransform],
    waveform: Tensor,
    sample_rate: int,
) -> Tensor:
    if pipeline is None:
        return waveform
    if waveform.dim() != 1:
        raise ValueError(f"Expected mono waveform tensor of shape (T,), got {tuple(waveform.shape)}")
    batch = waveform.unsqueeze(0).unsqueeze(0)
    augmented = pipeline(batch, sample_rate=sample_rate)
    if not torch.is_tensor(augmented):
        try:
            augmented = augmented.samples  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise RuntimeError("Augmentation pipeline must return tensor-like output") from exc
    if augmented.shape != batch.shape:
        augmented = augmented.view(batch.shape)
    return augmented.squeeze(0).squeeze(0)


def set_pipeline_seed(pipeline: Optional[BaseWaveformTransform], seed: int) -> None:
    if pipeline is None:
        return
    try:
        pipeline.set_seed(seed)
    except AttributeError:
        return
