from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import soundfile as sf
import numpy as np


SUPPORTED_EXTS: Tuple[str, ...] = (".wav", ".wave")


@dataclass(frozen=True)
class FileStats:
    path: Path
    label: Optional[str]
    samplerate: int
    channels: int
    bits_per_sample: Optional[int]
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.samplerate if self.samplerate > 0 else 0.0

    @property
    def bitrate_kbps(self) -> Optional[float]:
        # For uncompressed PCM WAV: bitrate = sr * bits * ch
        if self.bits_per_sample is None:
            return None
        return (self.samplerate * self.bits_per_sample * self.channels) / 1000.0


def find_recordings_root(explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        return explicit_root
    # Default root per AGENTS.md/README
    primary = Path("../recordings").resolve()
    fallback = Path("../host-tools/recordings").resolve()
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    # If neither exists, default to primary path to produce a clear error later
    return primary


def load_manifest(manifest_path: Optional[Path]) -> Dict[Path, str]:
    mapping: Dict[Path, str] = {}
    if manifest_path is None:
        # Try common locations
        candidates: List[Path] = [
            Path("../recordings/manifest.jsonl"),
            Path("./manifest.jsonl"),
        ]
    else:
        candidates = [manifest_path]

    for cand in candidates:
        if cand.exists():
            with cand.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    apath = row.get("audio_path")
                    label = row.get("label")
                    if isinstance(apath, str) and isinstance(label, str):
                        mapping[Path(apath).resolve()] = label
            break
    return mapping


def infer_label_from_path(path: Path) -> Optional[str]:
    # Heuristics when no manifest: inspect parent dir and filename tokens
    tokens = [p.lower() for p in path.parts]
    name = path.name.lower()
    candidates_on = {"pump_on", "on", "pump-on", "pumpon"}
    candidates_off = {"pump_off", "off", "pump-off", "pumpoff"}

    def has_token(tokens_iter: Iterable[str], keys: set[str]) -> bool:
        for t in tokens_iter:
            for k in keys:
                # Match whole token or clear separators in name
                if t == k or f"_{k}_" in f"_{t}_" or f"-{k}-" in f"-{t}-":
                    return True
        return False

    if has_token(tokens, candidates_on) or has_token([name], candidates_on):
        return "pump_on"
    if has_token(tokens, candidates_off) or has_token([name], candidates_off):
        return "pump_off"
    return None


def scan_audio_files(root: Path, manifest: Dict[Path, str]) -> List[FileStats]:
    results: List[FileStats] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            info = sf.info(str(p))
        except RuntimeError:
            # Unreadable/unsupported file
            continue

        # Map subtype to bit depth when possible (e.g., PCM_16 → 16)
        bps: Optional[int]
        subtype = (info.subtype or "").upper()
        if subtype.startswith("PCM_"):
            try:
                bps = int(subtype.split("_")[-1])
            except ValueError:
                bps = None
        else:
            # e.g., FLOAT, DOUBLE, or other codecs
            # FLOAT corresponds to 32-bit float, but that's not a bit depth decision
            bps = 32 if subtype == "FLOAT" else None

        # Determine label from manifest or heuristics
        label: Optional[str] = manifest.get(p.resolve())
        if label is None:
            label = infer_label_from_path(p)

        results.append(
            FileStats(
                path=p.resolve(),
                label=label,
                samplerate=int(info.samplerate),
                channels=int(info.channels),
                bits_per_sample=bps,
                frames=int(info.frames),
            )
        )
    return results


def human_time(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts: List[str] = []
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1:
        parts.append(f"{int(minutes)}m")
    # Show seconds if small or no hours
    if hours < 1 and sec > 0:
        parts.append(f"{int(sec)}s")
    return " ".join(parts)


def summarize(stats: List[FileStats]) -> str:
    total_dur = sum(s.duration_s for s in stats)
    n_files = len(stats)
    by_label: Dict[str, float] = {}
    for s in stats:
        key = s.label or "unlabeled"
        by_label[key] = by_label.get(key, 0.0) + s.duration_s

    # Distributions
    sr_counts: Dict[int, int] = {}
    sr_durations: Dict[int, float] = {}
    bits_counts: Dict[str, int] = {}
    ch_counts: Dict[int, int] = {}
    for s in stats:
        sr_counts[s.samplerate] = sr_counts.get(s.samplerate, 0) + 1
        sr_durations[s.samplerate] = sr_durations.get(s.samplerate, 0.0) + s.duration_s
        bit_key = str(s.bits_per_sample) if s.bits_per_sample is not None else "unknown"
        bits_counts[bit_key] = bits_counts.get(bit_key, 0) + 1
        ch_counts[s.channels] = ch_counts.get(s.channels, 0) + 1

    lines: List[str] = []
    lines.append("Recordings Analysis Summary")
    lines.append("-" * 32)
    lines.append(f"Root files: {n_files}")
    lines.append(f"Total duration: {human_time(total_dur)} ({total_dur:.1f}s)")
    lines.append("")

    # Labels
    lines.append("By label (duration):")
    for k, dur in sorted(by_label.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {human_time(dur)} ({dur:.1f}s)")
    lines.append("")

    # Sample rates
    lines.append("Sample rates (files | duration):")
    for sr in sorted(sr_counts.keys()):
        cnt = sr_counts[sr]
        dur = sr_durations.get(sr, 0.0)
        lines.append(f"- {sr/1000:.1f} kHz: {cnt} files | {human_time(dur)}")
    lines.append("")

    # Bit depths
    lines.append("Bit depths (file counts):")
    for b, cnt in sorted(bits_counts.items(), key=lambda x: (x[0])):
        label = f"{b}-bit" if b.isdigit() else b
        lines.append(f"- {label}: {cnt}")
    lines.append("")

    # Channels
    lines.append("Channels (file counts):")
    for ch, cnt in sorted(ch_counts.items()):
        lines.append(f"- {ch} ch: {cnt}")
    lines.append("")

    # Bitrate examples (median over files with known bps)
    br_kbps = [s.bitrate_kbps for s in stats if s.bitrate_kbps is not None]
    if br_kbps:
        sorted_br = sorted(br_kbps)
        median = sorted_br[len(sorted_br) // 2]
        lines.append(f"Median uncompressed bitrate: {median:.1f} kbps")
    else:
        lines.append("Median uncompressed bitrate: unknown")

    return "\n".join(lines)


def write_csv(stats: List[FileStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "label",
                "duration_s",
                "samplerate",
                "channels",
                "bits_per_sample",
                "bitrate_kbps",
            ]
        )
        for s in stats:
            writer.writerow(
                [
                    str(s.path),
                    s.label or "",
                    f"{s.duration_s:.6f}",
                    s.samplerate,
                    s.channels,
                    s.bits_per_sample if s.bits_per_sample is not None else "",
                    f"{s.bitrate_kbps:.3f}" if s.bitrate_kbps is not None else "",
                ]
            )


@dataclass(frozen=True)
class SpectralMetrics:
    duration_s: float
    samplerate: int
    energy_frac_lt_4k: float
    energy_frac_lt_6k: float
    energy_frac_lt_8k: float
    energy_frac_lt_10k: float
    energy_frac_lt_12k: float
    rolloff_90_hz: float
    rolloff_95_hz: float
    rolloff_99_hz: float
    centroid_hz: float
    peak_freq_hz: float


def _stft_energy_streamed(
    data_iter: Iterable[np.ndarray],
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate magnitude-squared spectrum over time, streamed.

    Returns tuple (freqs, psd_sum) where psd_sum has shape [n_freqs].
    """
    window = np.hanning(n_fft).astype(np.float32)
    buf = np.zeros(0, dtype=np.float32)
    psd_sum: Optional[np.ndarray] = None

    for chunk in data_iter:
        if chunk.ndim == 2:  # mix to mono
            chunk = chunk.mean(axis=1)
        chunk = chunk.astype(np.float32, copy=False)
        if buf.size:
            x = np.concatenate([buf, chunk], axis=0)
        else:
            x = chunk
        i = 0
        while i + n_fft <= x.shape[0]:
            frame = x[i : i + n_fft]
            i += hop_length
            frame = frame * window
            spec = np.fft.rfft(frame, n=n_fft)
            power = (spec.real ** 2 + spec.imag ** 2)
            if psd_sum is None:
                psd_sum = power
            else:
                psd_sum += power
        # keep leftover
        buf = x[i:]

    if psd_sum is None:
        psd_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    return freqs, psd_sum


def compute_spectral_metrics(path: Path, block_size: int = 262144) -> Optional[SpectralMetrics]:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return None

    sr = int(info.samplerate)
    total_frames = int(info.frames)
    duration_s = total_frames / sr if sr > 0 else 0.0
    if total_frames <= 0 or sr <= 0:
        return None

    def chunk_iter() -> Iterable[np.ndarray]:
        with sf.SoundFile(str(path), mode="r") as f:
            while True:
                data = f.read(block_size, dtype="float32", always_2d=False)
                if data is None or (isinstance(data, np.ndarray) and data.size == 0):
                    break
                yield data

    freqs, psd_sum = _stft_energy_streamed(chunk_iter(), sr)
    total_energy = float(psd_sum.sum()) if psd_sum.size else 0.0
    if total_energy <= 0.0:
        # Silence or unreadable
        return SpectralMetrics(
            duration_s=duration_s,
            samplerate=sr,
            energy_frac_lt_4k=1.0,
            energy_frac_lt_6k=1.0,
            energy_frac_lt_8k=1.0,
            energy_frac_lt_10k=1.0,
            energy_frac_lt_12k=1.0,
            rolloff_90_hz=0.0,
            rolloff_95_hz=0.0,
            rolloff_99_hz=0.0,
            centroid_hz=0.0,
            peak_freq_hz=0.0,
        )

    def frac_below(freq_cut: float) -> float:
        if freq_cut >= sr / 2:
            return 1.0
        mask = freqs <= freq_cut
        return float(psd_sum[mask].sum() / total_energy)

    # Spectral centroid
    centroid = float((freqs * psd_sum).sum() / total_energy)
    peak_freq = float(freqs[int(psd_sum.argmax())]) if psd_sum.size else 0.0

    # Rolloff percentiles
    cumsum = np.cumsum(psd_sum)
    def rolloff(p: float) -> float:
        target = p * total_energy
        idx = int(np.searchsorted(cumsum, target, side="left"))
        idx = min(max(idx, 0), len(freqs) - 1)
        return float(freqs[idx])

    return SpectralMetrics(
        duration_s=duration_s,
        samplerate=sr,
        energy_frac_lt_4k=frac_below(4000.0),
        energy_frac_lt_6k=frac_below(6000.0),
        energy_frac_lt_8k=frac_below(8000.0),
        energy_frac_lt_10k=frac_below(10000.0),
        energy_frac_lt_12k=frac_below(12000.0),
        rolloff_90_hz=rolloff(0.90),
        rolloff_95_hz=rolloff(0.95),
        rolloff_99_hz=rolloff(0.99),
        centroid_hz=centroid,
        peak_freq_hz=peak_freq,
    )


def write_spectral_csv(rows: List[Tuple[Path, Optional[str], SpectralMetrics]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "label",
                "samplerate",
                "duration_s",
                "energy_frac_lt_4k",
                "energy_frac_lt_6k",
                "energy_frac_lt_8k",
                "energy_frac_lt_10k",
                "energy_frac_lt_12k",
                "rolloff_90_hz",
                "rolloff_95_hz",
                "rolloff_99_hz",
                "centroid_hz",
                "peak_freq_hz",
            ]
        )
        for path, label, m in rows:
            writer.writerow(
                [
                    str(path),
                    label or "",
                    m.samplerate,
                    f"{m.duration_s:.6f}",
                    f"{m.energy_frac_lt_4k:.6f}",
                    f"{m.energy_frac_lt_6k:.6f}",
                    f"{m.energy_frac_lt_8k:.6f}",
                    f"{m.energy_frac_lt_10k:.6f}",
                    f"{m.energy_frac_lt_12k:.6f}",
                    f"{m.rolloff_90_hz:.2f}",
                    f"{m.rolloff_95_hz:.2f}",
                    f"{m.rolloff_99_hz:.2f}",
                    f"{m.centroid_hz:.2f}",
                    f"{m.peak_freq_hz:.2f}",
                ]
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze WAV recordings: duration, SR, bit depth, labels")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root folder with recordings (defaults to ../recordings with fallback)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.jsonl mapping audio_path→label (falls back to ../recordings/manifest.jsonl)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("runs/recordings_analysis.csv"),
        help="Optional CSV output path (default: runs/recordings_analysis.csv)",
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Compute spectral energy metrics (fractions below 4/6/8/10/12 kHz, rolloff, centroid)",
    )
    parser.add_argument(
        "--spectral-csv",
        type=Path,
        default=Path("runs/recordings_spectral.csv"),
        help="Optional CSV output path for spectral metrics",
    )

    args = parser.parse_args(argv)
    root = find_recordings_root(args.root)
    if not root.exists():
        print(f"Recordings root not found: {root}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    stats = scan_audio_files(root, manifest)
    if not stats:
        print(f"No supported audio files found under {root}")
        return 0

    # Emit report
    report = summarize(stats)
    print(report)

    # Optional CSV of per-file details
    if args.csv:
        write_csv(stats, args.csv)
        print(f"\nWrote per-file details: {args.csv}")

    # Optional spectral analysis
    if args.spectral:
        spectral_rows: List[Tuple[Path, Optional[str], SpectralMetrics]] = []
        for s in stats:
            m = compute_spectral_metrics(Path(s.path))
            if m is not None:
                spectral_rows.append((Path(s.path), s.label, m))

        if not spectral_rows:
            print("\nNo spectral metrics computed (files unreadable or silent)")
        else:
            # Aggregate guidance
            fracs_8k = np.array([r[2].energy_frac_lt_8k for r in spectral_rows], dtype=np.float64)
            roll95 = np.array([r[2].rolloff_95_hz for r in spectral_rows], dtype=np.float64)
            med_frac8 = float(np.median(fracs_8k))
            p10_roll95 = float(np.percentile(roll95, 10))
            p90_roll95 = float(np.percentile(roll95, 90))

            print("\nSpectral energy summary:")
            print(f"- Median fraction of energy < 8 kHz: {med_frac8:.3f}")
            print(
                f"- 95% rolloff (10th–90th pct): {p10_roll95:.0f}–{p90_roll95:.0f} Hz"
            )
            if med_frac8 >= 0.98 and p90_roll95 < 7500:
                print(
                    "- Guidance: Strong evidence most energy lies below 8 kHz. 16 kHz SR is likely safe."
                )
            else:
                print(
                    "- Guidance: Consider mixed SR experiments (16 kHz vs 32 kHz); pump harmonics may extend above 8 kHz."
                )

            if args.spectral_csv:
                write_spectral_csv(spectral_rows, args.spectral_csv)
                print(f"Wrote spectral metrics: {args.spectral_csv}")

    # Quick guidance: bit depth and SR considerations
    # This is non-binding; actual decision should consider model performance & noise floor.
    known_bps = {s.bits_per_sample for s in stats if s.bits_per_sample is not None}
    known_sr = {s.samplerate for s in stats}
    if known_bps:
        if all(b in {16, 24} for b in known_bps):
            print(
                "\nGuidance: Data contains 16/24-bit PCM. For classification models, 16-bit is often sufficient,\n"
                "especially if recordings are not extremely quiet. Consider training with 16-bit exports\n"
                "and verify no accuracy drop before standardizing."
            )
    if known_sr:
        sr_sorted = sorted(known_sr)
        print(
            "Suggested SR check: try 32 kHz (baseline) and downsampled 16 kHz;\n"
            "evaluate accuracy/latency trade-offs. If pump energy is mostly <8 kHz, 16 kHz is safe."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
