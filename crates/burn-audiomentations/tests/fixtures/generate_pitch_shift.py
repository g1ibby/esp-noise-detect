#!/usr/bin/env python3
"""
Generate `torch_pitch_shift.pitch_shift` fixtures for
`burn_audiomentations::PitchShift` parity tests.

Committed output: `pitch_shift_<ratio>.bin`.

Run from the `nn/` venv:

    cd nn && uv run --no-dev python \
        ../crates/burn-audiomentations/tests/fixtures/generate_pitch_shift.py

Each fixture pins:

* sample rate, n_fft, hop_length — we override these to 32000 / 512 / 16 so
  the Rust side (which rounds `sample_rate // 64` to the nearest power of
  two) shares the exact STFT configuration with the reference.
* window = all-ones (torch_pitch_shift's default); both sides therefore
  use a rectangular analysis window.
* input shape (batch=1, channels=1, samples) — a seeded sine plus a
  narrow-band noise to give the phase vocoder something to do.
* `ratio = num/den` as a `Fraction`, taken from the set the Rust side
  enumerates at 32 kHz ± 2 semitones: `125/128` and `128/125`.

Layout of each fixture (little-endian, packed):

    header (4 × u32 + 2 × f32):
        num
        den
        sample_rate
        samples
    n_fft : u32
    hop   : u32
    input  : f32 × samples
    output : f32 × samples
"""

import pathlib
import struct
from fractions import Fraction

import torch
from torch_pitch_shift import pitch_shift

HERE = pathlib.Path(__file__).parent

SR = 32_000
N_FFT = 512
HOP = N_FFT // 32  # 16, matches torch_pitch_shift default for n_fft=512
SAMPLES = 8192     # shorter than a training window; plenty for peak parity


def _synth(seed: int) -> torch.Tensor:
    """A 1-s-ish signal with content across the band the phase vocoder
    actually operates on. Pure tones fold down to single bins and hide
    phase-accumulator bugs; we add two tones plus light noise so the
    vocoder touches multiple bins."""
    gen = torch.Generator().manual_seed(seed)
    n = torch.arange(SAMPLES, dtype=torch.float32)
    sig = (
        0.6 * torch.sin(2 * torch.pi * 1000 * n / SR)
        + 0.3 * torch.sin(2 * torch.pi * 2500 * n / SR)
        + 0.05 * torch.empty(SAMPLES).uniform_(-1, 1, generator=gen)
    )
    return sig


def _write(path: pathlib.Path, *, num: int, den: int):
    ratio = Fraction(num, den)
    sig = _synth(seed=12345 + num * 7 + den)
    x = sig[None, None, :]  # (batch=1, channels=1, samples)

    # Pass explicit n_fft / hop_length and a rectangular window to keep
    # parity with the Rust side.
    window = torch.ones(N_FFT, dtype=torch.float32)
    y = pitch_shift(
        x, ratio, sample_rate=SR, n_fft=N_FFT, hop_length=HOP, window=window
    )
    # y has the same shape as x; drop the batch/channel dims.
    y = y.squeeze(0).squeeze(0).contiguous().float()

    with path.open("wb") as f:
        f.write(struct.pack("<6I", num, den, SR, SAMPLES, N_FFT, HOP))
        f.write(sig.numpy().astype("float32").tobytes())
        f.write(y.numpy().astype("float32").tobytes())

    print(f"wrote {path.name}: ratio={num}/{den} samples={SAMPLES} n_fft={N_FFT} hop={HOP}")


def main():
    _write(HERE / "pitch_shift_128_over_125.bin", num=128, den=125)
    _write(HERE / "pitch_shift_125_over_128.bin", num=125, den=128)


if __name__ == "__main__":
    main()
