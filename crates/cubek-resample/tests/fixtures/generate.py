#!/usr/bin/env python3
"""
Generate `julius.resample_frac` fixtures for `cubek-resample` parity
tests. Called once; the produced .bin files are committed.

Run from the `nn/` venv:

    cd nn && uv run --no-dev python \
        ../crates/cubek-resample/tests/fixtures/generate.py

Layout of each fixture (little-endian, packed):

    header (5 × u32):
        batch
        time          (input length per row)
        out_len       (= applied_output_length chosen by julius)
        old_sr
        new_sr
    u32:
        zeros
    f32:
        rolloff
    input   : f32 × batch × time           (row-major C order)
    output  : f32 × batch × out_len

Python side generates the waveform, so the Rust test reads the whole
fixture and doesn't need to stay in sync with a second PRNG.
"""

import pathlib
import struct

import julius
import torch


HERE = pathlib.Path(__file__).parent


def _write_fixture(path: pathlib.Path, *, batch, time, old_sr, new_sr, zeros, rolloff, seed):
    gen = torch.Generator().manual_seed(seed)
    signal = torch.empty(batch, time, dtype=torch.float32).uniform_(-1, 1, generator=gen)

    # Julius default is `output_length=None, full=False` -> floor.
    out = julius.resample_frac(signal, old_sr, new_sr, zeros=zeros, rolloff=rolloff)
    out_len = out.shape[-1]

    with path.open("wb") as f:
        f.write(struct.pack("<5I", batch, time, out_len, old_sr, new_sr))
        f.write(struct.pack("<I", zeros))
        f.write(struct.pack("<f", rolloff))
        f.write(signal.numpy().astype("float32").tobytes())
        f.write(out.numpy().astype("float32").tobytes())

    print(
        f"wrote {path.name}: batch={batch} time={time} out_len={out_len} "
        f"old_sr={old_sr} new_sr={new_sr} zeros={zeros} rolloff={rolloff}"
    )


def main():
    _write_fixture(
        HERE / "upsample_4_to_5.bin",
        batch=1, time=256, old_sr=4, new_sr=5, zeros=24, rolloff=0.945, seed=1001,
    )
    _write_fixture(
        HERE / "downsample_5_to_4.bin",
        batch=1, time=256, old_sr=5, new_sr=4, zeros=24, rolloff=0.945, seed=1002,
    )
    _write_fixture(
        HERE / "batched_3_to_2.bin",
        batch=3, time=192, old_sr=3, new_sr=2, zeros=24, rolloff=0.945, seed=1003,
    )
    _write_fixture(
        HERE / "audio_16k_to_44k1.bin",
        # Reduced ratio: (160, 441). Keep `time` a multiple of old_sr
        # so the conv edge effects line up predictably.
        batch=1, time=1600, old_sr=16_000, new_sr=44_100, zeros=24, rolloff=0.945, seed=1004,
    )


if __name__ == "__main__":
    main()
