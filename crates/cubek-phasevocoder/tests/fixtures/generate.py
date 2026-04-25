#!/usr/bin/env python3
"""
Generate torchaudio phase_vocoder fixtures for `cubek-phasevocoder` parity
tests. Called once; the produced .bin files are committed.

Run from the `nn/` venv:

    cd nn && uv run --no-dev python \
        ../crates/cubek-phasevocoder/tests/fixtures/generate.py

Layout of each fixture (little-endian, packed):

    header (5 × u32):
        batch
        n_freq
        n_in
        n_out
        hop
    rate : f32
    phase_advance : f32 × n_freq
    input_re      : f32 × batch × n_freq × n_in        (row-major C order)
    input_im      : f32 × batch × n_freq × n_in
    output_re     : f32 × batch × n_freq × n_out
    output_im     : f32 × batch × n_freq × n_out

Python side generates everything including the input waveform — the Rust
tests read the full fixture, so there's no second PRNG to keep in sync.
"""

import math
import pathlib
import struct

import torch
import torchaudio.functional as F


HERE = pathlib.Path(__file__).parent


def _phase_advance(n_freq: int, hop: int) -> torch.Tensor:
    if n_freq == 1:
        return torch.zeros(1, dtype=torch.float32)
    return torch.linspace(0, math.pi * hop, n_freq, dtype=torch.float32)


def _write_fixture(path: pathlib.Path, *, batch, n_freq, n_in, hop, rate, seed):
    gen = torch.Generator().manual_seed(seed)
    re = torch.empty(batch, n_freq, n_in, dtype=torch.float32).uniform_(-1, 1, generator=gen)
    im = torch.empty(batch, n_freq, n_in, dtype=torch.float32).uniform_(-1, 1, generator=gen)
    pa = _phase_advance(n_freq, hop)

    spec = torch.complex(re, im)
    out = F.phase_vocoder(spec, rate, pa[:, None])
    assert out.dtype == torch.complex64
    n_out = out.size(-1)
    out_re = out.real.contiguous().numpy().astype("float32")
    out_im = out.imag.contiguous().numpy().astype("float32")

    with path.open("wb") as f:
        f.write(struct.pack("<5I", batch, n_freq, n_in, n_out, hop))
        f.write(struct.pack("<f", rate))
        f.write(pa.numpy().astype("float32").tobytes())
        f.write(re.numpy().astype("float32").tobytes())
        f.write(im.numpy().astype("float32").tobytes())
        f.write(out_re.tobytes())
        f.write(out_im.tobytes())

    print(
        f"wrote {path.name}: batch={batch} n_freq={n_freq} n_in={n_in} "
        f"n_out={n_out} rate={rate}"
    )


def main():
    _write_fixture(
        HERE / "small_rate_1p3.bin",
        batch=1, n_freq=65, n_in=32, hop=16, rate=1.3, seed=12345,
    )
    _write_fixture(
        HERE / "mid_rate_0p8.bin",
        batch=1, n_freq=129, n_in=48, hop=32, rate=0.8, seed=67890,
    )
    _write_fixture(
        HERE / "batched_rate_1p1.bin",
        batch=2, n_freq=33, n_in=20, hop=8, rate=1.1, seed=42,
    )


if __name__ == "__main__":
    main()
