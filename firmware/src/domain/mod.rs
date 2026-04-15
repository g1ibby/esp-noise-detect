// Domain types used by firmware; protocol types are imported where needed.
use alloc::vec::Vec;
use wire_protocol::{SegmentLabel, StopReason};

/// Audio sample format enumeration
#[derive(Clone, Debug, PartialEq)]
pub enum SampleFormat {
    Sample16,
    Sample24,
}

/// Generic audio data that can hold either 16-bit or 24-bit samples
#[derive(Clone, Debug)]
pub enum AudioData {
    Sample16(Vec<i16>),
    Sample24(Vec<i32>),
}

impl AudioData {
    pub fn new_16bit(data: Vec<i16>) -> Self {
        Self::Sample16(data)
    }

    pub fn new_24bit(data: Vec<i32>) -> Self {
        Self::Sample24(data)
    }

    pub fn format(&self) -> SampleFormat {
        match self {
            AudioData::Sample16(_) => SampleFormat::Sample16,
            AudioData::Sample24(_) => SampleFormat::Sample24,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            AudioData::Sample16(data) => data.len(),
            AudioData::Sample24(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize audio data to bytes for transmission
    pub fn to_bytes(&self) -> alloc::vec::Vec<u8> {
        match self {
            AudioData::Sample16(data) => {
                let mut bytes = alloc::vec::Vec::with_capacity(data.len() * 2);
                for &sample in data {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                bytes
            }
            AudioData::Sample24(data) => {
                let mut bytes = alloc::vec::Vec::with_capacity(data.len() * 4);
                for &sample in data {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                bytes
            }
        }
    }

    /// Get bytes per sample for this audio format
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            AudioData::Sample16(_) => 2,
            AudioData::Sample24(_) => 4,
        }
    }
}

/// Control messages emitted by the orchestrator and consumed by the transmitter.
#[derive(Clone, Debug)]
pub enum CtrlMsg {
    /// Segment boundary: starts a new segment and closes the previous with `prev_reason`.
    Segment {
        cycle_id: u32,
        label: SegmentLabel,
        prev_reason: StopReason,
        plan_ms: u32,
    },
}

/// Unified output type to preserve strict ordering between control and audio.
#[derive(Clone, Debug)]
pub enum NetOut {
    Audio(AudioData),
    Ctrl(CtrlMsg),
}
