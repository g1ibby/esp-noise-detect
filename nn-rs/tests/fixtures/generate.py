#!/usr/bin/env python3
"""
Generate `compute_mel_numpy` fixtures for the Step 7 mel parity test.

Run from the `nn/` venv:

    cd nn && uv run --no-dev python \
        ../nn-rs/tests/fixtures/generate.py

Layout of each fixture (little-endian, packed):

    header (6 × u32):
        sample_rate
        n_fft
        hop           (samples)
        n_mels
        time          (input length per row)
        n_frames
    flags (3 × u8):
        center        (0/1)
        log           (0/1)
        normalize     (0/1)
    padding           (1 × u8, keeps f32 alignment)
    fmin              (f32)
    fmax              (f32, 0.0 means "use Nyquist")
    eps               (f32)
    input  : f32 × time              (one mono waveform)
    output : f32 × n_mels × n_frames (compute_mel_numpy(...).reshape(...))

`compute_mel_numpy` is the single numerical ground truth per
MIGRATION_PLAN.md §1 — we deliberately do NOT run torchaudio's
`MelSpectrogram` here. The Rust `MelExtractor` must match this.
"""

from __future__ import annotations

import math
import pathlib
import struct

import numpy as np

# The NumPy reference lives in the sibling Python package.
from noise_detect.config import MelConfig
from noise_detect.features.mel import compute_mel_numpy


HERE = pathlib.Path(__file__).parent


def _synth_waveform(n: int, seed: int) -> np.ndarray:
    """Deterministic random waveform in `[-1, 1)`. Rust reads the samples
    straight out of the fixture, so this doesn't need to match any
    specific Rust PRNG — only needs to be reproducible run-to-run."""

    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n).astype(np.float32)


def _write_fixture(
    path: pathlib.Path,
    *,
    sample_rate: int,
    time: int,
    seed: int,
    cfg: MelConfig,
):
    x = _synth_waveform(time, seed)
    mel = compute_mel_numpy(x, sample_rate, cfg)
    # compute_mel_numpy returns (1, n_mels, n_frames); squeeze the channel.
    assert mel.shape[0] == 1
    mel = mel[0].astype("float32", copy=False)
    n_mels, n_frames = mel.shape
    assert n_mels == cfg.n_mels

    # Hop from ms → samples, the way mel.py / MelConfig.hop_length do.
    hop = max(int(round(cfg.hop_length_ms / 1000.0 * sample_rate)), 1)

    with path.open("wb") as f:
        f.write(
            struct.pack(
                "<6I",
                sample_rate,
                cfg.n_fft,
                hop,
                cfg.n_mels,
                time,
                n_frames,
            )
        )
        f.write(
            struct.pack(
                "<4B",
                1 if cfg.center else 0,
                1 if cfg.log else 0,
                1 if cfg.normalize else 0,
                0,  # pad to multiple of 4 so the f32s that follow align
            )
        )
        fmax = float(cfg.fmax) if cfg.fmax is not None else 0.0
        f.write(struct.pack("<3f", float(cfg.fmin), fmax, float(cfg.eps)))
        f.write(x.astype("float32").tobytes())
        f.write(mel.tobytes())

    print(
        f"wrote {path.name}: sr={sample_rate} n_fft={cfg.n_fft} hop={hop} "
        f"n_mels={cfg.n_mels} time={time} n_frames={n_frames} "
        f"center={cfg.center} log={cfg.log} normalize={cfg.normalize}"
    )


def main():
    # Primary target: the training pipeline's actual setup
    # (nn/configs/features/mel.yaml + dataset/manifest.yaml). 1 s of 32 kHz
    # audio, n_fft=1024, hop=320, 64 HTK mel bins, log + normalize.
    cfg_full = MelConfig(
        n_mels=64,
        fmin=50.0,
        fmax=None,
        n_fft=1024,
        hop_length_ms=10.0,
        log=True,
        eps=1e-10,
        normalize=True,
        center=True,
    )
    _write_fixture(
        HERE / "mel_robust_session.bin",
        sample_rate=32_000,
        time=32_000,
        seed=42,
        cfg=cfg_full,
    )

    # Minimal variant: log off + normalize off isolates the STFT → power →
    # filterbank path from the post-processing, so a parity drift points
    # somewhere specific.
    cfg_raw = MelConfig(
        n_mels=32,
        fmin=20.0,
        fmax=8_000.0,
        n_fft=512,
        hop_length_ms=8.0,
        log=False,
        eps=1e-10,
        normalize=False,
        center=True,
    )
    _write_fixture(
        HERE / "mel_raw_power.bin",
        sample_rate=16_000,
        time=8_000,
        seed=777,
        cfg=cfg_raw,
    )


if __name__ == "__main__":
    main()
