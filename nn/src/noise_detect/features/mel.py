from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from noise_detect.config import MelConfig


def _make_mel_transform(cfg: MelConfig, sample_rate: int):
    import torchaudio  # type: ignore

    hop_length = int(round((cfg.hop_length_ms / 1000.0) * sample_rate))
    hop_length = max(hop_length, 1)
    f_max: Optional[float] = cfg.fmax if cfg.fmax is not None else sample_rate / 2
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=cfg.n_fft,
        hop_length=hop_length,
        f_min=cfg.fmin,
        f_max=f_max,
        n_mels=cfg.n_mels,
        power=cfg.power,
        center=cfg.center,
        pad_mode=cfg.pad_mode,
        norm=None,
        mel_scale="htk",
    )


class MelExtractor(nn.Module):
    """Reusable log-mel extractor as an nn.Module.

    Wrapping the torchaudio transform in a Module lets Lightning move its
    filterbank/window buffers to the training device once, instead of rebuilding
    on every batch (see features/mel.py:compute_mel).

    Input: (T,) or (B, T). Output: (B, 1, n_mels, n_frames).
    """

    def __init__(self, cfg: MelConfig, sample_rate: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.mel = _make_mel_transform(cfg, sample_rate)

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"MelExtractor expected (T,) or (B, T), got {tuple(waveform.shape)}"
            )
        x = waveform.unsqueeze(1)  # (B, 1, T)
        mel = self.mel(x)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        elif mel.dim() == 4 and mel.shape[1] != 1:
            mel = mel.mean(dim=1, keepdim=True)
        if self.cfg.log:
            mel = log_mel(mel, eps=self.cfg.eps)
        if self.cfg.normalize:
            mel = normalize_per_example(mel)
        return mel


def log_mel(x: Tensor, eps: float) -> Tensor:
    # Power mel -> log mel (natural log). eps to avoid -inf
    return torch.log(x.clamp_min(eps))


def normalize_per_example(x: Tensor) -> Tensor:
    # x: (B, 1, M, T) or (1, M, T)
    dims = tuple(range(1, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def compute_mel(
    waveform_batch: Tensor,  # (B, T) or (T,)
    sample_rate: int,
    cfg: MelConfig,
) -> Tensor:
    """Compute log-mel spectrograms on CPU.

    Returns shape: (B, 1, n_mels, time_frames)
    """
    if waveform_batch.dim() == 1:
        waveform_batch = waveform_batch.unsqueeze(0)
    if waveform_batch.dim() != 2:
        raise ValueError(f"Expected waveform batch of shape (B, T) or (T,), got {tuple(waveform_batch.shape)}")

    # Ensure CPU feature extraction
    if waveform_batch.device.type != "cpu":
        waveform_batch = waveform_batch.cpu()
    mel_tf = _make_mel_transform(cfg, sample_rate)
    # Expect input to MelSpectrogram as (B, C, T)
    x = waveform_batch.unsqueeze(1)  # (B, 1, T)
    mel = mel_tf(x)
    # torchaudio returns (B, C, n_mels, n_frames) when input is (B, C, T)
    if mel.dim() == 3:
        # (B, n_mels, n_frames) -> add channel dim
        mel = mel.unsqueeze(1)
    elif mel.dim() == 4:
        # (B, C, n_mels, n_frames), expect C==1
        if mel.shape[1] != 1:
            # mix channels if not 1
            mel = mel.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unexpected mel shape {tuple(mel.shape)} from torchaudio")
    if cfg.log:
        mel = log_mel(mel, eps=cfg.eps)
    if cfg.normalize:
        mel = normalize_per_example(mel)
    return mel


# Reference NumPy implementation for portability
def compute_mel_numpy(
    waveform: "numpy.ndarray",  # (T,)
    sample_rate: int,
    cfg: MelConfig,
) -> "numpy.ndarray":
    """Reference CPU NumPy mel-spectrogram to aid MCU porting.

    Returns array of shape (1, n_mels, time_frames). Uses HTK mel scale.
    """
    import numpy as np  # local import to avoid hard dependency at module import

    x = waveform.astype(np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected mono waveform (T,), got {x.shape}")

    n_fft = int(cfg.n_fft)
    hop_length = max(int(round((cfg.hop_length_ms / 1000.0) * sample_rate)), 1)
    win_length = n_fft

    def pad_center(y: np.ndarray, size: int) -> np.ndarray:
        pad = (size - y.shape[-1]) // 2
        if pad <= 0:
            return y
        return np.pad(y, (pad, size - y.shape[-1] - pad), mode="reflect")

    if cfg.center:
        pad = (n_fft // 2)
        x = np.pad(x, (pad, pad), mode=cfg.pad_mode)

    n_frames = 1 + (len(x) - n_fft) // hop_length if len(x) >= n_fft else 1
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))

    # Framing
    shape = (n_frames, n_fft)
    strides = (hop_length * x.strides[0], x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()
    # Hann window
    window = np.hanning(win_length).astype(np.float32)
    frames *= window
    # FFT power
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    power_spec = (spec.real**2 + spec.imag**2).astype(np.float32)

    # Mel filter bank (HTK)
    def hz_to_mel_htk(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz_htk(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    f_min = float(cfg.fmin)
    f_max = float(cfg.fmax) if cfg.fmax is not None else sample_rate / 2.0
    m_min = hz_to_mel_htk(np.array([f_min], dtype=np.float32))[0]
    m_max = hz_to_mel_htk(np.array([f_max], dtype=np.float32))[0]
    m_points = np.linspace(m_min, m_max, cfg.n_mels + 2, dtype=np.float32)
    f_points = mel_to_hz_htk(m_points)
    bins = np.floor((n_fft + 1) * f_points / sample_rate).astype(int)

    fbanks = np.zeros((cfg.n_mels, power_spec.shape[-1]), dtype=np.float32)
    for m in range(1, cfg.n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        if f_m_minus == f_m:
            f_m = min(f_m + 1, fbanks.shape[1] - 1)
        if f_m == f_m_plus:
            f_m_plus = min(f_m_plus + 1, fbanks.shape[1] - 1)
        for k in range(f_m_minus, f_m):
            fbanks[m - 1, k] = (k - f_m_minus) / max(f_m - f_m_minus, 1)
        for k in range(f_m, f_m_plus):
            fbanks[m - 1, k] = (f_m_plus - k) / max(f_m_plus - f_m, 1)

    mel = power_spec @ fbanks.T  # (frames, n_mels)
    mel = mel.T[None, ...]  # (1, n_mels, frames)
    if cfg.log:
        mel = np.log(np.maximum(mel, cfg.eps)).astype(np.float32)
    if cfg.normalize:
        mean = mel.mean(axis=(1, 2), keepdims=True)
        std = mel.std(axis=(1, 2), keepdims=True)
        mel = (mel - mean) / np.maximum(std, 1e-6)
    return mel.astype(np.float32)
