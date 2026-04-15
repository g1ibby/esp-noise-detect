use anyhow::{Context, Result};
use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use wire_protocol::{Metadata, SegmentLabel, StopReason};

use super::{AudioSamples, Sink};

/// WAV file sink used by the server to persist segments.
///
/// Files are created per segment; repeated calls will create a series of
/// files that together represent one streaming session. Filenames encode
/// the device id, the session start timestamp, and the zero-based segment
/// number so that ordering is clear and resumable.

#[derive(Clone, Debug)]
pub struct WavSink {
    output_dir: PathBuf,
}

impl WavSink {
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }
}

impl Sink for WavSink {
    fn save_segment(
        &mut self,
        device_id: &str,
        chunk_number: u32,
        session_start: SystemTime,
        samples: AudioSamples,
        meta: Metadata,
        label: Option<SegmentLabel>,
        cycle_id: Option<u32>,
        reason: Option<StopReason>,
    ) -> Result<PathBuf> {
        let session_timestamp = session_start.duration_since(UNIX_EPOCH)?.as_secs();
        let filename = if let (Some(lbl), Some(cycle)) = (label, cycle_id) {
            let part = match lbl {
                SegmentLabel::On => "on",
                SegmentLabel::Off => "off",
                SegmentLabel::Undefined => "undefined",
            };
            let reason_suffix = match reason {
                Some(StopReason::Timeout) => "_timeout",
                Some(StopReason::SafetyCutoff) => "_safety",
                Some(StopReason::Canceled) => "_canceled",
                Some(StopReason::Normal) | None => "",
            };
            format!(
                "{}_{}_c{:03}_{}_chunk{:03}{}.wav",
                device_id, session_timestamp, cycle, part, chunk_number, reason_suffix
            )
        } else {
            // Continuous mode or legacy save: keep current pattern
            let part = match label.unwrap_or(SegmentLabel::Undefined) {
                SegmentLabel::On => "on",
                SegmentLabel::Off => "off",
                SegmentLabel::Undefined => "undefined",
            };
            let cycle = cycle_id.unwrap_or(0);
            format!(
                "{}_{}_c{:03}_{}_chunk{:03}.wav",
                device_id, session_timestamp, cycle, part, chunk_number
            )
        };
        let filepath = self.output_dir.join(filename);

        let spec = WavSpec {
            channels: meta.channels,
            sample_rate: meta.sample_rate,
            bits_per_sample: meta.bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(&filepath, spec).context("Failed to create WAV file")?;
        match samples {
            AudioSamples::Sample16(samples_16) => {
                for &s in samples_16 {
                    writer.write_sample(s)?;
                }
            }
            AudioSamples::Sample24(samples_24) => {
                for &s in samples_24 {
                    writer.write_sample(s)?;
                }
            }
        }
        writer.finalize().context("Failed to finalize WAV file")?;
        Ok(filepath)
    }
}
