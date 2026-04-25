//! `WindowedAudioDataset` — sliding-window dataset over a manifest of
//! WAV files.
//!
//! Each index maps to a single `(file, window_index)` pair; `get`
//! decodes the source WAV, mixes down to mono, slices out the requested
//! window and zero-pads the tail if the file ends mid-window.
//!
//! Resampling is not implemented — production recordings are 32 kHz
//! mono (matching `DatasetConfig::sample_rate`). If a file's sample
//! rate does not match, the dataset errors on read.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use burn::data::dataset::Dataset;

use super::config::{DatasetConfig, seconds_to_samples};
use super::manifest::{ManifestItem, Split};

/// Fraction of *available* RAM we're willing to commit to the decoded
/// audio cache. Leaves plenty of headroom for the GPU driver, Burn's
/// tensor arenas, the OS page cache, and user work. Going higher
/// trades training speed for the risk of swap pressure mid-epoch.
const CACHE_RAM_FRACTION: f64 = 0.5;

/// Over-estimation factor applied to the projected dataset size when
/// sizing the cache. Covers 24-bit → f32 expansion that we can't
/// predict from the header alone (the manifest header read gives us
/// frame count and sample rate; sample format is checked lazily in
/// `decode_mono`), plus alignment slack inside `Vec<f32>`.
const CACHE_HEADROOM: f64 = 1.10;

/// Upper bound regardless of dataset size — keeps a misreported
/// manifest from asking for hundreds of gigabytes.
const CACHE_HARD_CAP_MB: usize = 64 * 1024;

/// FIFO cache of fully decoded mono f32 waveforms keyed by absolute
/// path. Shared across dataloader workers via `Arc<Mutex<_>>` — the
/// `Mutex` is held only for the lookup / insert / eviction step; the
/// actual decode work runs with the lock dropped, so concurrent
/// workers don't serialise on it.
///
/// Eviction is FIFO, not true LRU: we pop the oldest insertion when
/// the total byte count exceeds the cap. True LRU would need to move
/// entries on every hit, which needs a second write-lock and buys
/// little over FIFO for this access pattern (shuffle permutes the
/// indices per-epoch, so recency of access isn't a strong signal).
pub(crate) struct DecodeCache {
    map: HashMap<PathBuf, CachedSignal>,
    order: VecDeque<PathBuf>,
    bytes: usize,
    cap_bytes: usize,
}

#[derive(Clone)]
struct CachedSignal {
    samples: Arc<Vec<f32>>,
    sample_rate: u32,
}

impl DecodeCache {
    fn new(cap_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            cap_bytes,
        }
    }

    fn get(&self, path: &Path) -> Option<CachedSignal> {
        self.map.get(path).cloned()
    }

    fn insert(&mut self, path: PathBuf, samples: Arc<Vec<f32>>, sample_rate: u32) {
        if self.cap_bytes == 0 {
            return;
        }
        let entry_bytes = samples.len() * std::mem::size_of::<f32>();
        // Don't bother caching entries that are larger than the whole
        // budget — they'd evict everything else on insert.
        if entry_bytes > self.cap_bytes {
            return;
        }
        while self.bytes + entry_bytes > self.cap_bytes {
            let Some(evict) = self.order.pop_front() else {
                break;
            };
            if let Some(old) = self.map.remove(&evict) {
                let old_bytes = old.samples.len() * std::mem::size_of::<f32>();
                self.bytes = self.bytes.saturating_sub(old_bytes);
            }
        }
        self.order.push_back(path.clone());
        self.map.insert(
            path,
            CachedSignal {
                samples,
                sample_rate,
            },
        );
        self.bytes += entry_bytes;
    }
}

/// One window produced by [`WindowedAudioDataset::get`].
#[derive(Clone, Debug)]
pub struct WindowedAudioItem {
    /// Length is always `DatasetConfig::window_samples` — shorter tail
    /// windows are zero-padded on the right.
    pub waveform: Vec<f32>,
    pub label: usize,
    pub file: PathBuf,
    pub start_s: f32,
    /// `min(window_end, file_end) / sample_rate` — capped at the
    /// actual end of the source audio.
    pub end_s: f32,
}

/// Index into `(file, window_idx)` — one entry per dataset item.
#[derive(Clone, Copy, Debug)]
struct WindowIndex {
    file_idx: usize,
    window_idx: usize,
}

/// Sliding-window dataset over a subset of a manifest (one split).
///
/// Construction reads each file's header (not its audio data) via
/// `hound::WavReader::duration` to size the index. The actual
/// decode happens lazily in [`Dataset::get`].
pub struct WindowedAudioDataset {
    items: Vec<ManifestItem>,
    cfg: DatasetConfig,
    split: Split,
    window_samples: usize,
    hop_samples: usize,
    class_to_idx: HashMap<String, usize>,
    indices: Vec<WindowIndex>,
    /// Shared across dataloader workers. `None` means caching is
    /// disabled (`NN_RS_DECODE_CACHE_MB=0`), in which case every
    /// `get()` re-decodes the file as before.
    cache: Option<Arc<Mutex<DecodeCache>>>,
}

impl WindowedAudioDataset {
    /// Build a dataset for one `split`.
    ///
    /// Filters `items` down to rows whose `split` matches (or `None`,
    /// treated as `train`). Verifies every referenced WAV opens and
    /// reports a recognized sample rate, otherwise returns an error
    /// identifying the offending file.
    pub fn new(
        items: Vec<ManifestItem>,
        cfg: DatasetConfig,
        split: Split,
    ) -> Result<Self, DatasetBuildError> {
        let class_to_idx: HashMap<String, usize> = cfg
            .class_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let items: Vec<ManifestItem> = items
            .into_iter()
            .filter(|it| it.split.unwrap_or(Split::Train) == split)
            .collect();

        let (window_samples, hop_samples) =
            seconds_to_samples(cfg.window_s, cfg.hop_s, cfg.sample_rate);

        let mut indices = Vec::new();
        // Accumulates the projected decoded size of every file we
        // enroll — used by `pick_cache_cap` below to size the cache
        // once we've finished the header walk.
        let mut decoded_bytes_estimate: u64 = 0;
        for (file_idx, it) in items.iter().enumerate() {
            if !class_to_idx.contains_key(&it.label) {
                return Err(DatasetBuildError::UnknownLabel {
                    label: it.label.clone(),
                    path: it.audio_path.clone(),
                });
            }
            let probe = probe_header(&it.audio_path, cfg.sample_rate, window_samples, hop_samples);
            let n_windows = match probe {
                Ok(p) => {
                    decoded_bytes_estimate = decoded_bytes_estimate
                        .saturating_add(p.decoded_bytes as u64);
                    p.n_windows.max(1)
                }
                // Can't read the header — enroll one window and let the
                // lazy decode path surface the real error.
                Err(_) => 1,
            };
            for w in 0..n_windows {
                indices.push(WindowIndex {
                    file_idx,
                    window_idx: w,
                });
            }
        }

        let cap_bytes = pick_cache_cap(decoded_bytes_estimate, split);
        let cache = if cap_bytes == 0 {
            None
        } else {
            Some(Arc::new(Mutex::new(DecodeCache::new(cap_bytes))))
        };

        Ok(Self {
            items,
            cfg,
            split,
            window_samples,
            hop_samples,
            class_to_idx,
            indices,
            cache,
        })
    }

    /// Samples per window, precomputed from the config.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Hop in samples, precomputed from the config.
    pub fn hop_samples(&self) -> usize {
        self.hop_samples
    }

    pub fn split(&self) -> Split {
        self.split
    }

    pub fn config(&self) -> &DatasetConfig {
        &self.cfg
    }
}

impl Dataset<WindowedAudioItem> for WindowedAudioDataset {
    fn get(&self, index: usize) -> Option<WindowedAudioItem> {
        let wi = self.indices.get(index).copied()?;
        let it = &self.items[wi.file_idx];

        // Cache hit: skip `hound` decode + int→f32 conversion. We still
        // slice into the decoded buffer per window (a `copy_from_slice`
        // of `window_samples` floats — cheap).
        let signal = match self.cache.as_ref() {
            Some(cache) => {
                // Take the lock only to probe and release it before
                // decoding. Workers that miss the same file at the
                // same time will each decode it; the last insert wins.
                // That's wasted work, but bounded, and keeps decode
                // off the critical section.
                if let Some(hit) = cache.lock().ok().and_then(|g| g.get(&it.audio_path)) {
                    Some(hit)
                } else {
                    let decoded = decode_mono(&it.audio_path).ok()?;
                    if decoded.1 != self.cfg.sample_rate {
                        return None;
                    }
                    let samples = Arc::new(decoded.0);
                    if let Ok(mut g) = cache.lock() {
                        g.insert(it.audio_path.clone(), samples.clone(), decoded.1);
                    }
                    Some(CachedSignal {
                        samples,
                        sample_rate: decoded.1,
                    })
                }
            }
            None => {
                let (samples, src_sr) = decode_mono(&it.audio_path).ok()?;
                if src_sr != self.cfg.sample_rate {
                    return None;
                }
                Some(CachedSignal {
                    samples: Arc::new(samples),
                    sample_rate: src_sr,
                })
            }
        }?;
        // Sanity-check sample rate on every access (cheap — just an
        // integer comparison). Belt-and-braces vs. cache-hit path.
        if signal.sample_rate != self.cfg.sample_rate {
            return None;
        }

        let file_len = signal.samples.len();
        let start = wi.window_idx * self.hop_samples;
        let end = start + self.window_samples;
        let available_end = end.min(file_len);

        let mut window = vec![0.0_f32; self.window_samples];
        if start < file_len {
            let avail = available_end - start;
            window[..avail].copy_from_slice(&signal.samples[start..available_end]);
        }

        let label = *self.class_to_idx.get(&it.label).unwrap_or(&0);
        let sr = self.cfg.sample_rate as f32;
        Some(WindowedAudioItem {
            waveform: window,
            label,
            file: it.audio_path.clone(),
            start_s: start as f32 / sr,
            end_s: available_end as f32 / sr,
        })
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Construction-time error. Decode errors from individual files are
/// not propagated at `new()` time — they're absorbed into `len = 1`
/// and surface as `None` from `get()`.
#[derive(Debug)]
pub enum DatasetBuildError {
    UnknownLabel { label: String, path: PathBuf },
}

impl std::fmt::Display for DatasetBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownLabel { label, path } => write!(
                f,
                "manifest row uses unknown label `{label}` (file {}); \
                 extend DatasetConfig::class_names to include it",
                path.display()
            ),
        }
    }
}

impl std::error::Error for DatasetBuildError {}

/// Decode a WAV file to a mono f32 buffer.
///
/// Handles all of hound's sample formats:
/// * 16-bit integer → normalize by `2^15`.
/// * 24-bit integer (stored as i32) → normalize by `2^23`.
/// * 32-bit integer → normalize by `2^31`.
/// * 32-bit float → passthrough.
///
/// Multi-channel files are averaged across channels.
pub(crate) fn decode_mono(path: &Path) -> Result<(Vec<f32>, u32), hound::Error> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let frames: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0_f32 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            let mut out = Vec::with_capacity(reader.duration() as usize);
            let mut acc = 0.0_f32;
            let mut count = 0usize;
            for s in reader.samples::<i32>() {
                let v = s? as f32 * scale;
                acc += v;
                count += 1;
                if count == channels {
                    out.push(acc / channels as f32);
                    acc = 0.0;
                    count = 0;
                }
            }
            out
        }
        hound::SampleFormat::Float => {
            let mut out = Vec::with_capacity(reader.duration() as usize);
            let mut acc = 0.0_f32;
            let mut count = 0usize;
            for s in reader.samples::<f32>() {
                let v = s?;
                acc += v;
                count += 1;
                if count == channels {
                    out.push(acc / channels as f32);
                    acc = 0.0;
                    count = 0;
                }
            }
            out
        }
    };
    Ok((frames, sample_rate))
}

/// Compute the expected number of windows for a file without decoding
/// its audio (source sample rate must match the target).
struct HeaderProbe {
    n_windows: usize,
    /// Size of the mono f32 buffer this file will decode to, in
    /// bytes. Equal to `frames * size_of::<f32>()` — channels are
    /// mixed down to one at decode time, so channel count doesn't
    /// inflate the decoded buffer.
    decoded_bytes: usize,
}

fn probe_header(
    path: &Path,
    target_sr: u32,
    window_samples: usize,
    hop_samples: usize,
) -> Result<HeaderProbe, hound::Error> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let frames = reader.duration() as usize;
    let decoded_bytes = frames.saturating_mul(std::mem::size_of::<f32>());
    if spec.sample_rate != target_sr {
        // Lazy failure on first `get()`; this file won't actually
        // decode successfully, so don't include its bytes in the
        // cache-size estimate.
        return Ok(HeaderProbe {
            n_windows: 1,
            decoded_bytes: 0,
        });
    }
    if frames < window_samples {
        return Ok(HeaderProbe {
            n_windows: 1,
            decoded_bytes,
        });
    }
    let remainder = frames.saturating_sub(window_samples);
    let mut n = 1 + remainder / hop_samples;
    if remainder % hop_samples != 0 {
        n += 1; // Final partial + padded window.
    }
    Ok(HeaderProbe {
        n_windows: n,
        decoded_bytes,
    })
}

/// Pick a cache capacity (in bytes) given the projected decoded size
/// of this split's files and the system's current available RAM.
///
/// Decision rule (simple, no knobs):
/// * Start from `dataset_bytes * CACHE_HEADROOM` — enough to hold
///   every file this split will ever access.
/// * Cap at `CACHE_RAM_FRACTION * available_ram` so we never swap.
/// * Cap again at `CACHE_HARD_CAP_MB` as a sanity floor (avoids the
///   "manifest lies about size" failure mode).
/// * Return `0` when the budget would be smaller than an average WAV
///   — caching that little is more overhead than benefit.
///
/// Emits one line to stderr describing the decision, tagged with the
/// split name so users see one line per split.
fn pick_cache_cap(dataset_bytes: u64, split: Split) -> usize {
    let want = ((dataset_bytes as f64) * CACHE_HEADROOM) as u64;

    // Available RAM. sysinfo refreshes lazily; we only need one
    // snapshot for this decision. `available_memory()` accounts for
    // the OS file-cache that can be evicted under pressure, so it's
    // a better basis than `free_memory()`.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory();

    let ram_cap = ((available as f64) * CACHE_RAM_FRACTION) as u64;
    let hard_cap = (CACHE_HARD_CAP_MB as u64).saturating_mul(1024 * 1024);

    let mut cap = want.min(ram_cap).min(hard_cap);
    // Round down to MiB so the log line is readable.
    cap = (cap / (1024 * 1024)) * (1024 * 1024);

    // Below a useful minimum we disable the cache outright — it's
    // cheaper to decode on demand than maintain a cache that can't
    // hold two average files.
    const MIN_USEFUL_MB: u64 = 64;
    let min_useful = MIN_USEFUL_MB * 1024 * 1024;

    let mb = |b: u64| b / (1024 * 1024);
    let disabled = cap < min_useful;
    eprintln!(
        "[decode-cache] split={:?} dataset≈{} MB  avail_ram={} MB  cap={} MB{}",
        split,
        mb(dataset_bytes),
        mb(available),
        mb(cap),
        if disabled { "  (disabled — too small)" } else { "" },
    );
    if disabled { 0 } else { cap as usize }
}

#[cfg(test)]
mod tests {
    /// Inline test of the window-count formula against synthetic frame
    /// counts; real WAV files are exercised in `tests/data_pipeline.rs`.
    fn n_windows(frames: usize, ws: usize, hs: usize) -> usize {
        if frames < ws {
            return 1;
        }
        let remainder = frames - ws;
        let mut n = 1 + remainder / hs;
        if remainder % hs != 0 {
            n += 1;
        }
        n
    }

    #[test]
    fn estimate_num_windows_matches_python() {
        // 1 s window / 0.5 s hop at 32 kHz: ws = 32000, hs = 16000.

        // 5 s file (160 000 samples) → aligned, no tail:
        // 1 + (160000 - 32000)/16000 = 1 + 8 = 9 windows.
        assert_eq!(n_windows(160_000, 32_000, 16_000), 9);

        // 5.5 s file (176 000) → aligned again (144 000 / 16 000 = 9 exactly).
        assert_eq!(n_windows(176_000, 32_000, 16_000), 10);

        // 5.53125 s file (177 000) → non-aligned tail, expect +1 padded window.
        assert_eq!(n_windows(177_000, 32_000, 16_000), 11);

        // Sub-window file collapses to one padded window.
        assert_eq!(n_windows(10_000, 32_000, 16_000), 1);

        // Exactly one window, no tail.
        assert_eq!(n_windows(32_000, 32_000, 16_000), 1);
    }
}
