from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Self


ParamScalar = float | int | bool | str
ParamList = list[float] | list[int] | list[str]
ParamValue = ParamScalar | ParamList


def _to_plain_obj(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:  # pragma: no cover - OmegaConf optional
        return value
    return value


@dataclass
class DatasetConfig:
    sample_rate: int = 32_000
    window_s: float = 1.0
    hop_s: float = 0.5
    class_names: list[str] = field(default_factory=lambda: ["pump_off", "pump_on"])  # index 0,1
    # Optional path to manifest for training/eval
    manifest_path: str | None = None


@dataclass
class MelConfig:
    n_mels: int = 64
    fmin: float = 50.0
    fmax: Optional[float] = None  # defaults to Nyquist if None
    n_fft: int = 1024
    hop_length_ms: float = 10.0  # ~10 ms hop by default
    power: float = 2.0
    log: bool = True
    eps: float = 1e-10
    normalize: bool = True  # per-example mean-variance norm
    center: bool = True
    pad_mode: str = "reflect"


@dataclass
class TinyConvConfig:
    channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    dropout: float = 0.0


DeviceName = Literal["auto", "cpu", "mps", "cuda"]
AggregateMode = Literal["mean", "max"]


@dataclass
class AugmentationTransformConfig:
    """Configuration for a single waveform augmentation transform."""

    type: Literal[
        "gain",
        "add_colored_noise",
        "add_background_noise",
        "bandpass",
        "bandstop",
        "highpass",
        "lowpass",
        "pitch_shift",
        "time_masking",
        "polarity_inversion",
        "impulse_response",
        "shift",
        "time_inversion",
    ]
    probability: float = 0.5
    params: dict[str, ParamValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = _to_plain_obj(data) or {}
        if "type" not in data:
            raise ValueError("Augmentation transform requires a 'type' field")
        type_value = data["type"]
        probability = float(data.get("probability", 0.5))
        raw_params = _to_plain_obj(data.get("params", {}))
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise TypeError("Augmentation transform 'params' must be a mapping")

        params: dict[str, ParamValue] = {}
        for key, value in raw_params.items():
            if isinstance(value, (int, float, bool, str)):
                params[key] = value
            elif isinstance(value, list):
                if not value:
                    params[key] = value
                    continue
                first = value[0]
                if isinstance(first, (int, float)) and all(isinstance(v, (int, float)) for v in value):
                    params[key] = [float(v) for v in value]
                elif isinstance(first, str) and all(isinstance(v, str) for v in value):
                    params[key] = [str(v) for v in value]
                else:
                    raise TypeError(
                        "Augmentation transform list params must be numeric or string lists"
                    )
            else:
                raise TypeError(
                    "Unsupported parameter type for augmentation transform; "
                    "use scalar or list of scalars"
                )
        return cls(type=type_value, probability=probability, params=params)


@dataclass
class AugmentationPipelineConfig:
    enabled: bool = False
    shuffle: bool = True
    same_on_batch: bool = False
    transforms: list[AugmentationTransformConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = _to_plain_obj(data) or {}
        enabled = bool(data.get("enabled", False))
        shuffle = bool(data.get("shuffle", True))
        same_on_batch = bool(data.get("same_on_batch", False))
        transforms_data = _to_plain_obj(data.get("transforms", []) or [])
        try:
            transforms_iterable = list(transforms_data)
        except TypeError as exc:
            raise TypeError("Augmentation pipeline 'transforms' must be iterable") from exc
        transforms: list[AugmentationTransformConfig] = []
        for entry in transforms_iterable:
            entry_plain = _to_plain_obj(entry)
            if not isinstance(entry_plain, dict):
                raise TypeError("Each augmentation transform must be a mapping")
            transforms.append(AugmentationTransformConfig.from_dict(entry_plain))
        return cls(
            enabled=enabled,
            shuffle=shuffle,
            same_on_batch=same_on_batch,
            transforms=transforms,
        )


@dataclass
class AugmentationConfig:
    train: AugmentationPipelineConfig = field(default_factory=AugmentationPipelineConfig)
    val: AugmentationPipelineConfig = field(default_factory=AugmentationPipelineConfig)
    test: AugmentationPipelineConfig = field(default_factory=AugmentationPipelineConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = _to_plain_obj(data) or {}
        train_cfg = AugmentationPipelineConfig.from_dict(data.get("train", {}) or {})
        val_cfg = AugmentationPipelineConfig.from_dict(data.get("val", {}) or {})
        test_cfg = AugmentationPipelineConfig.from_dict(data.get("test", {}) or {})
        return cls(train=train_cfg, val=val_cfg, test=test_cfg)


@dataclass
class InferenceConfig:
    # IO
    input_path: str = ""  # file or directory (wav)
    output_csv: Optional[str] = None  # optional CSV output
    checkpoint: Optional[str] = None  # PyTorch state_dict or Lightning ckpt
    use_manifest: bool = False  # if true, read file list from manifest
    split: Optional[Literal["train", "val", "test"]] = None
    window_output_csv: Optional[str] = None
    emit_window_stats: bool = False

    # Execution
    device: DeviceName = "auto"
    threshold: float | None = None
    aggregate: AggregateMode = "mean"
    batch_size: int = 32

    # Nested configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    model: TinyConvConfig = field(default_factory=TinyConvConfig)


def resolve_input_path(cfg: InferenceConfig) -> Path:
    # Resolve relative to original CWD (Hydra may change working dir)
    try:
        from hydra.utils import to_absolute_path  # type: ignore

        p = Path(to_absolute_path(cfg.input_path)).expanduser()
    except Exception:  # pragma: no cover
        p = Path(cfg.input_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"input_path does not exist: {p}")
    return p


@dataclass
class DataModuleConfig:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    # Precompute mel-spectrograms once and train from a cached tensor file.
    # Skips waveform decode/resample/mel on every batch; disables waveform-level
    # augmentations (they'd need SpecAugment on the mel side instead).
    use_cache: bool = True
    # Optional override for the cache directory. If None, cache is placed next
    # to the manifest under <manifest_dir>/.mel_cache/.
    cache_dir: Optional[str] = None


@dataclass
class OptimizerConfig:
    name: Literal["adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    momentum: float = 0.9  # for SGD


@dataclass
class SchedulerConfig:
    name: Literal["none", "cosine"] = "none"
    t_max: int = 100  # for cosine
    min_lr: float = 1e-5


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    accelerator: DeviceName = "auto"
    devices: int | str = 1  # use 1 by default to avoid None with MPS
    precision: str = "32"  # "32" | "bf16-mixed" | "16-mixed"
    gradient_clip_val: float = 1.0
    deterministic: bool = True
    seed: int = 42


@dataclass
class TrainAppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    model: TinyConvConfig = field(default_factory=TinyConvConfig)
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)
    dm: DataModuleConfig = field(default_factory=DataModuleConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def load_augmentation_config(data: Optional[dict[str, Any]] = None) -> AugmentationConfig:
    if not data:
        return AugmentationConfig()
    return AugmentationConfig.from_dict(_to_plain_obj(data))


# Export / Quantization configuration


@dataclass
class OnnxExportConfig:
    opset: int = 13
    simplify: bool = True


@dataclass
class CalibConfig:
    split: Literal["train", "val"] = "train"
    num_windows: int = 512
    save_npy: bool = True


@dataclass
class EspdlExportConfig:
    enabled: bool = False
    # Target backend for ESP-DL export: 'esp32s3', 'esp32p4' or 'c'
    target: Literal["c", "esp32s3", "esp32p4"] = "esp32s3"
    num_bits: Literal[8, 16] = 8
    device: Literal["cpu", "cuda"] = "cpu"
    # How many calibration steps to run (capped by available windows)
    calib_steps: int = 64
    # Emit detailed logs from esp-ppq
    verbose: int = 1
    # Also export test input/output tensors into the .espdl package
    export_test_values: bool = True
    # Print error report after calibration
    error_report: bool = True


@dataclass
class ExportAppConfig:
    # IO / target formats
    checkpoint: str | None = None
    out_dir: str = "export"

    # Nested configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    model: TinyConvConfig = field(default_factory=TinyConvConfig)

    # Export specifics
    onnx: OnnxExportConfig = field(default_factory=OnnxExportConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    espdl: EspdlExportConfig = field(default_factory=EspdlExportConfig)
