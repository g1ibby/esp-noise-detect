from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def mix_down_to_mono(waveform: Tensor) -> Tensor:
    """Mix multi-channel waveform to mono by averaging channels.

    Expects shape (channels, samples) or (samples,).
    Returns shape (samples,).
    """
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform with shape (C, T) or (T,), got {tuple(waveform.shape)}")
    return waveform.mean(dim=0)


def ensure_float32(x: Tensor) -> Tensor:
    if x.dtype != torch.float32:
        return x.to(torch.float32)
    return x


def resample_if_needed(waveform: Tensor, src_sr: int, dst_sr: int) -> Tensor:
    """Resample mono waveform from src_sr to dst_sr if different.

    Expects shape (T,). Returns shape (T').
    """
    if src_sr == dst_sr:
        return waveform
    if waveform.dim() != 1:
        raise ValueError(f"resample_if_needed expects mono waveform (T,), got {tuple(waveform.shape)}")
    # Import locally to avoid global dependency during type checking
    import torchaudio.functional as AF  # type: ignore

    # Resample expects (channels, time)
    x = waveform.unsqueeze(0)
    y = AF.resample(x, src_sr, dst_sr)
    return y.squeeze(0)


def window_signal(
    waveform: Tensor, *, window_samples: int, hop_samples: int, pad: bool = True
) -> Tensor:
    """Split mono waveform into overlapping windows (num_windows, window_samples).

    - If pad=True and signal is shorter than one window, zero-pad to one window.
    - Drops last partial window if pad=False.
    """
    if waveform.dim() != 1:
        raise ValueError(f"window_signal expects mono waveform (T,), got {tuple(waveform.shape)}")
    T = waveform.shape[0]
    if T <= 0:
        return waveform.new_zeros((0, window_samples))

    if T < window_samples:
        if pad:
            out = waveform.new_zeros((1, window_samples))
            out[0, :T] = waveform
            return out
        return waveform.new_zeros((0, window_samples))

    # Compute number of windows (include last if pad and remainder)
    n = 1 + (T - window_samples) // hop_samples
    rem = (T - window_samples) % hop_samples
    if pad and rem > 0:
        n += 1
    windows = waveform.new_zeros((n, window_samples))
    for i in range(n):
        start = i * hop_samples
        end = start + window_samples
        if end <= T:
            windows[i] = waveform[start:end]
        else:
            # pad tail with zeros
            part = waveform[start:]
            windows[i, : part.shape[0]] = part
    return windows


def seconds_to_samples(window_s: float, hop_s: float, sample_rate: int) -> Tuple[int, int]:
    window_samples = int(round(window_s * sample_rate))
    hop_samples = int(round(hop_s * sample_rate))
    return max(window_samples, 1), max(hop_samples, 1)

