#!/usr/bin/env python3
"""
Generate `julius.lowpass_filter` fixtures for `cubek-sinc-filter` parity
tests. Called once; the produced .bin files are committed.

Run from the `nn/` venv:

    cd nn && uv run --no-dev python \
        ../crates/cubek-sinc-filter/tests/fixtures/generate.py

Layout of each fixture (little-endian, packed):

    header:
        u32 batch
        u32 time          (input length per row; output length == input length)
        u32 n_cutoffs
        u32 zeros
        u32 mode          (0 = low-pass, 1 = high-pass)
    f32 * n_cutoffs:
        cutoffs (normalized f_c / f_s)
    u32 * batch:
        indices (cutoff row assigned to each batch row)
    f32 * batch * time:
        input  (row-major C order)
    f32 * batch * time:
        output (row-major C order)

Python side generates the waveform so the Rust test reads the whole
fixture and doesn't need to stay in sync with a second PRNG.
"""

import pathlib
import struct

import julius
import torch


HERE = pathlib.Path(__file__).parent


def _run_lowpass(signal: torch.Tensor, cutoffs, indices, zeros: int, highpass: bool) -> torch.Tensor:
    # Match the bank semantics of `cubek-sinc-filter`: one filter length
    # derived from the smallest positive cutoff, shared across all filters
    # in the bank. Julius's `LowPassFilters([c1, c2, ...])` has exactly this
    # behaviour (see the class docstring in
    # `_reference/julius/julius/lowpass.py:50`).
    #
    # This deliberately differs from `torch_audiomentations.LowPassFilter`,
    # which calls `julius.lowpass_filter` per row with that row's own
    # cutoff — so each row ends up with a different filter length. The
    # migration plan (§2.1 step 4) trades that per-row length variability
    # for a 1–2 % cutoff-bucket quantization, which is what makes a single
    # batched GPU launch possible.
    #
    # `fft=False` so the reference matches our direct-conv kernel's
    # arithmetic within rounding. Julius's FFT path is algebraically
    # equivalent but picks up extra f32 rounding from the overlap-add.
    cutoffs_f = [float(c) for c in cutoffs]
    bank = julius.LowPassFilters(cutoffs_f, zeros=zeros, fft=False)
    # Forward: signal is (batch, time) -> output (n_cutoffs, batch, time).
    banked = bank(signal)
    out = torch.empty_like(signal)
    for i, idx in enumerate(indices):
        lp = banked[idx, i]
        if highpass:
            out[i] = signal[i] - lp
        else:
            out[i] = lp
    return out


def _write_fixture(
    path: pathlib.Path,
    *,
    batch: int,
    time: int,
    cutoffs,
    zeros: int,
    highpass: bool,
    indices,
    seed: int,
):
    assert len(indices) == batch
    gen = torch.Generator().manual_seed(seed)
    signal = torch.empty(batch, time, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)

    out = _run_lowpass(signal, cutoffs, indices, zeros, highpass)

    with path.open("wb") as f:
        f.write(struct.pack("<5I", batch, time, len(cutoffs), zeros, 1 if highpass else 0))
        f.write(struct.pack(f"<{len(cutoffs)}f", *[float(c) for c in cutoffs]))
        f.write(struct.pack(f"<{batch}I", *[int(i) for i in indices]))
        f.write(signal.numpy().astype("float32").tobytes())
        f.write(out.numpy().astype("float32").tobytes())

    tag = "hp" if highpass else "lp"
    print(
        f"wrote {path.name}: batch={batch} time={time} n_cutoffs={len(cutoffs)} "
        f"zeros={zeros} mode={tag}"
    )


def main():
    # Normalized cutoffs used across the fixtures. Cover the whole LPF /
    # HPF range used by `torch_audiomentations` at 32 kHz (150 Hz – 7500 Hz
    # low-pass, 20 Hz – 2400 Hz high-pass).
    lp_cutoffs = [150.0 / 32_000, 600.0 / 32_000, 2_400.0 / 32_000, 7_500.0 / 32_000]
    hp_cutoffs = [20.0 / 32_000, 200.0 / 32_000, 1_000.0 / 32_000, 2_400.0 / 32_000]

    _write_fixture(
        HERE / "lp_single_cutoff.bin",
        batch=1,
        time=512,
        cutoffs=lp_cutoffs,
        zeros=8,
        highpass=False,
        indices=[2],
        seed=2_001,
    )

    _write_fixture(
        HERE / "lp_batched_mixed_cutoffs.bin",
        batch=4,
        time=768,
        cutoffs=lp_cutoffs,
        zeros=8,
        highpass=False,
        indices=[0, 2, 3, 1],
        seed=2_002,
    )

    _write_fixture(
        HERE / "hp_single_cutoff.bin",
        batch=1,
        time=512,
        cutoffs=hp_cutoffs,
        zeros=8,
        highpass=True,
        indices=[1],
        seed=2_003,
    )

    _write_fixture(
        HERE / "hp_batched_mixed_cutoffs.bin",
        batch=3,
        time=640,
        cutoffs=hp_cutoffs,
        zeros=8,
        highpass=True,
        indices=[0, 3, 1],
        seed=2_004,
    )


if __name__ == "__main__":
    main()
