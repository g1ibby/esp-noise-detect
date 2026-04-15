use anyhow::{anyhow, Result};
use std::time::SystemTime;

use crate::storage::{AudioSamples, Sink};
use crate::util::{format_bytes, format_duration};
use tracing::info;
use wire_protocol::{Metadata as MetadataMsg, RecordingModeKind, SegmentLabel, StopReason};

// Helper enum for handling both sample types
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

pub const RECORDING_DURATION_SECS: u32 = 300; // Auto-save every 5 minutes
pub const PROGRESS_INTERVAL_SECS: u64 = 10; // Emit progress roughly every 10s

#[derive(Debug, Clone)]
pub struct AudioSession {
    pub device_id: String,
    pub metadata: Option<MetadataMsg>,
    pub audio_samples_16: Vec<i16>,
    pub audio_samples_24: Vec<i32>,
    pub packets_received: u32,
    pub bytes_received: usize,
    pub session_start: SystemTime,
    pub current_segment_start: SystemTime,
    pub chunk_number: u32,
    pub total_files_saved: u32,
    pub last_progress_time: SystemTime,
    pub last_keepalive: SystemTime,
    pub recording_mode: Option<RecordingModeKind>,
    pub current_label: Option<SegmentLabel>,
    pub current_cycle: Option<u32>,
    pub last_stop_reason: Option<StopReason>,
    pub total_samples_saved: usize,
}

impl AudioSession {
    pub fn new(device_id: String) -> Self {
        let now = SystemTime::now();
        Self {
            device_id,
            metadata: None,
            audio_samples_16: Vec::new(),
            audio_samples_24: Vec::new(),
            packets_received: 0,
            bytes_received: 0,
            session_start: now,
            current_segment_start: now,
            chunk_number: 0,
            total_files_saved: 0,
            last_progress_time: now,
            last_keepalive: now,
            recording_mode: None,
            current_label: None,
            current_cycle: None,
            last_stop_reason: None,
            total_samples_saved: 0,
        }
    }

    pub fn add_audio_data(&mut self, data: &[u8]) -> Result<()> {
        if let Some(metadata) = &self.metadata {
            match metadata.bits_per_sample {
                16 => {
                    for chunk in data.chunks_exact(2) {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        self.audio_samples_16.push(sample);
                    }
                }
                24 => {
                    for chunk in data.chunks_exact(4) {
                        let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        self.audio_samples_24.push(sample);
                    }
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported bits_per_sample: {}",
                        metadata.bits_per_sample
                    ));
                }
            }
        } else {
            // Default to 16-bit if no metadata available yet
            for chunk in data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                self.audio_samples_16.push(sample);
            }
        }

        self.packets_received += 1;
        self.bytes_received += data.len();

        // Emit periodic progress if due
        self.log_progress_if_due();
        Ok(())
    }

    /// Get the current sample count (works with both 16-bit and 24-bit)
    pub fn sample_count(&self) -> usize {
        if !self.audio_samples_16.is_empty() {
            self.audio_samples_16.len()
        } else {
            self.audio_samples_24.len()
        }
    }

    /// Check if we have any audio samples
    pub fn has_audio_samples(&self) -> bool {
        !self.audio_samples_16.is_empty() || !self.audio_samples_24.is_empty()
    }

    /// Clear all audio samples
    pub fn clear_audio_samples(&mut self) {
        self.audio_samples_16.clear();
        self.audio_samples_24.clear();
    }

    /// Get the appropriate sample slice for saving (returns either 16-bit or 24-bit samples)
    pub fn get_samples_for_saving(&self) -> Either<&[i16], &[i32]> {
        if !self.audio_samples_16.is_empty() {
            Either::Left(&self.audio_samples_16)
        } else {
            Either::Right(&self.audio_samples_24)
        }
    }

    /// Emit a progress line if the configured interval has elapsed since the last one.
    pub fn log_progress_if_due(&mut self) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_progress_time)
            .unwrap_or_default()
            .as_secs()
            >= PROGRESS_INTERVAL_SECS
        {
            let segment_duration = self.current_segment_start.elapsed().unwrap_or_default();
            // Prefer audio-derived total duration if metadata is available
            if let Some(meta) = &self.metadata {
                let total_samples = self.total_samples_saved + self.sample_count();
                let total_secs =
                    total_samples as f32 / (meta.sample_rate as f32 * meta.channels as f32);
                info!(
                    "🎙️  [{}] Audio: {} | Current: {} | {} samples ({})",
                    self.device_id,
                    format_duration(total_secs),
                    format_duration(segment_duration.as_secs_f32()),
                    self.sample_count(),
                    format_bytes(self.bytes_received)
                );
            } else {
                // Fallback to wall clock
                let total_duration = self.session_start.elapsed().unwrap_or_default();
                info!(
                    "🎙️  [{}] Recording: {} | Current: {} | {} samples ({})",
                    self.device_id,
                    format_duration(total_duration.as_secs_f32()),
                    format_duration(segment_duration.as_secs_f32()),
                    self.sample_count(),
                    format_bytes(self.bytes_received)
                );
            }
            self.last_progress_time = now;
        }
    }

    pub fn should_auto_save(&self) -> bool {
        if !self.has_audio_samples() || self.metadata.is_none() {
            return false;
        }
        let elapsed = self.current_segment_start.elapsed().unwrap_or_default();
        elapsed.as_secs() >= RECORDING_DURATION_SECS as u64
    }

    pub fn save_current_segment<S: Sink>(
        &mut self,
        sink: &mut S,
        label: Option<SegmentLabel>,
        cycle_id: Option<u32>,
        reason: Option<StopReason>,
    ) -> Result<Option<std::path::PathBuf>> {
        if !self.has_audio_samples() {
            return Ok(None);
        }
        let metadata = self
            .metadata
            .as_ref()
            .ok_or_else(|| anyhow!("No metadata available"))?;

        let samples = match self.get_samples_for_saving() {
            Either::Left(samples_16) => AudioSamples::Sample16(samples_16),
            Either::Right(samples_24) => AudioSamples::Sample24(samples_24),
        };

        let filepath = sink.save_segment(
            &self.device_id,
            self.chunk_number,
            self.session_start,
            samples,
            *metadata,
            label,
            cycle_id,
            reason,
        )?;

        let sample_count = self.sample_count();
        let duration_secs =
            sample_count as f32 / (metadata.sample_rate as f32 * metadata.channels as f32);
        let file_size = std::fs::metadata(&filepath).map(|m| m.len()).unwrap_or(0) as usize;
        info!(
            "💾 [{}] Saved chunk {}: {} ({} samples, {}) -> {}",
            self.device_id,
            self.chunk_number + 1,
            format_duration(duration_secs),
            sample_count,
            format_bytes(file_size),
            filepath.display()
        );

        // Accumulate total saved sample count then clear current buffer
        let saved = self.sample_count();
        self.total_samples_saved = self.total_samples_saved.saturating_add(saved);
        self.clear_audio_samples();
        self.current_segment_start = SystemTime::now();
        self.last_progress_time = SystemTime::now();
        // Increment chunk number for time-based rolls (None or SafetyCutoff auto-save)
        match reason {
            None | Some(StopReason::SafetyCutoff) => {
                self.chunk_number = self.chunk_number.saturating_add(1);
            }
            _ => {}
        }
        self.total_files_saved += 1;

        Ok(Some(filepath))
    }
}
