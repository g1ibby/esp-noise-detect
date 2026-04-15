//! Storage sinks for received audio.
//!
//! Segmentation strategy:
//! - The server appends incoming samples to the current session buffer.
//! - Every `RECORDING_DURATION_SECS` (see `crate::session`) or on shutdown,
//!   the current segment is written to disk via the active `Sink`.
//! - `WavSink` produces a `*.wav` file per chunk (time-based save) in the
//!   configured output directory and uses a stable filename pattern including
//!   the device id, session start timestamp, cycle id, label, and chunk number.
//!
//! Implementors of `Sink` can route segments to other backends (e.g. pipes,
//! network, or different file formats) without changing session logic.

use anyhow::Result;
use std::path::PathBuf;
use std::time::SystemTime;
use wire_protocol::{Metadata, SegmentLabel, StopReason};

/// Audio samples that can be either 16-bit or 24-bit
pub enum AudioSamples<'a> {
    Sample16(&'a [i16]),
    Sample24(&'a [i32]),
}

pub trait Sink {
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
    ) -> Result<PathBuf>;
}

mod wav;
pub use wav::WavSink;
