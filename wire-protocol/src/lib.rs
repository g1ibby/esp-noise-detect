#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

//! Wire protocol: message types, headers, and (de)serialization.
//!
//! - no_std by default (uses `core`).
//! - Provides zero-allocation parsing of headers and borrowed views
//!   for simple payloads (HELLO, METADATA).

use core::fmt;

/// Fixed header length in bytes.
pub const HEADER_LEN: usize = 8;

/// Message types supported by the protocol.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MessageType {
    Hello = 0x01,
    Metadata = 0x02,
    Audio = 0x03,
    End = 0x04,
    Keepalive = 0x05,
    Segment = 0x08,
    PumpStatus = 0x09,
}

impl TryFrom<u8> for MessageType {
    type Error = Error;
    fn try_from(v: u8) -> Result<Self, Error> {
        Ok(match v {
            0x01 => MessageType::Hello,
            0x02 => MessageType::Metadata,
            0x03 => MessageType::Audio,
            0x04 => MessageType::End,
            0x05 => MessageType::Keepalive,
            0x08 => MessageType::Segment,
            0x09 => MessageType::PumpStatus,
            _ => return Err(Error::UnknownMessageType(v)),
        })
    }
}

/// Protocol header (8 bytes):
/// - msg_type: 1 byte
/// - length: 3 bytes (little-endian, lower 24 bits)
/// - sequence: 4 bytes (little-endian)
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Header {
    pub msg_type: MessageType,
    pub length: u32,   // lower 24 bits used
    pub sequence: u32, // full 32-bit sequence
}

impl Header {
    /// Encode header into the provided 8-byte buffer.
    pub fn encode_into(&self, out: &mut [u8; HEADER_LEN]) {
        out[0] = self.msg_type as u8;
        // write 24-bit length (little-endian in first 3 bytes)
        out[1] = (self.length & 0xFF) as u8;
        out[2] = ((self.length >> 8) & 0xFF) as u8;
        out[3] = ((self.length >> 16) & 0xFF) as u8;
        out[4..8].copy_from_slice(&self.sequence.to_le_bytes());
    }

    /// Parse a header from a byte slice (must be at least 8 bytes).
    pub fn parse(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < HEADER_LEN {
            return Err(Error::HeaderTooShort(buf.len()));
        }
        let msg_type = MessageType::try_from(buf[0])?;
        let length = (buf[1] as u32) | ((buf[2] as u32) << 8) | ((buf[3] as u32) << 16);
        let sequence = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        Ok(Header {
            msg_type,
            length,
            sequence,
        })
    }
}

/// Borrowed view of a HELLO payload: `"device_id:version"`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Hello<'a> {
    pub device_id: &'a str,
    pub version: &'a str,
}

impl<'a> Hello<'a> {
    /// Parse from `payload` like `b"dev:1.0"`.
    pub fn parse(payload: &'a [u8]) -> Result<Self, Error> {
        let s = core::str::from_utf8(payload).map_err(|_| Error::InvalidUtf8)?;
        let mut parts = s.splitn(2, ':');
        let device_id = parts.next().ok_or(Error::InvalidHello)?;
        let version = parts.next().ok_or(Error::InvalidHello)?;
        Ok(Hello { device_id, version })
    }

    /// Encode into provided buffer as `device_id:version`.
    /// Returns number of bytes written.
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize, Error> {
        let needed = self.device_id.len() + 1 + self.version.len();
        if out.len() < needed {
            return Err(Error::BufferTooSmall {
                needed,
                available: out.len(),
            });
        }
        let mut n = 0;
        out[n..n + self.device_id.len()].copy_from_slice(self.device_id.as_bytes());
        n += self.device_id.len();
        out[n] = b':';
        n += 1;
        out[n..n + self.version.len()].copy_from_slice(self.version.as_bytes());
        n += self.version.len();
        Ok(n)
    }
}

/// Borrowed/parsed view of METADATA payload: `"sample_rate:channels:bits"`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Metadata {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub recording_mode: RecordingModeKind,
}

impl Metadata {
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let s = core::str::from_utf8(payload).map_err(|_| Error::InvalidUtf8)?;
        // New (first) version: 4 fields required `{sr}:{ch}:{bits}:{mode}`
        let mut parts = s.splitn(4, ':');
        let sr = parts.next().ok_or(Error::InvalidMetadata)?;
        let ch = parts.next().ok_or(Error::InvalidMetadata)?;
        let bps = parts.next().ok_or(Error::InvalidMetadata)?;
        let mode_str = parts.next().ok_or(Error::InvalidMetadata)?;
        let sample_rate = parse_u32(sr).ok_or(Error::InvalidMetadata)?;
        let channels = parse_u16(ch).ok_or(Error::InvalidMetadata)?;
        let bits_per_sample = parse_u16(bps).ok_or(Error::InvalidMetadata)?;
        let recording_mode = parse_mode(mode_str).ok_or(Error::InvalidMetadata)?;
        Ok(Metadata {
            sample_rate,
            channels,
            bits_per_sample,
            recording_mode,
        })
    }

    /// Encode to `b"{sr}:{ch}:{bits}"`.
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize, Error> {
        // small, allocation-free integer formatting
        let mut n = 0;
        n += write_uint(self.sample_rate, &mut out[n..])?;
        n += write_byte(b':', &mut out[n..])?;
        n += write_uint(self.channels as u32, &mut out[n..])?;
        n += write_byte(b':', &mut out[n..])?;
        n += write_uint(self.bits_per_sample as u32, &mut out[n..])?;
        n += write_byte(b':', &mut out[n..])?;
        n += write_str(mode_str(self.recording_mode), &mut out[n..])?;
        Ok(n)
    }
}

/// Protocol errors.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    HeaderTooShort(usize),
    UnknownMessageType(u8),
    InvalidUtf8,
    InvalidHello,
    InvalidMetadata,
    InvalidSegment,
    InvalidPumpStatus,
    BufferTooSmall { needed: usize, available: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::HeaderTooShort(n) => write!(f, "header too short: {} bytes", n),
            Error::UnknownMessageType(b) => write!(f, "unknown message type: 0x{:02X}", b),
            Error::InvalidUtf8 => write!(f, "invalid utf-8"),
            Error::InvalidHello => write!(f, "invalid hello payload"),
            Error::InvalidMetadata => write!(f, "invalid metadata payload"),
            Error::InvalidSegment => write!(f, "invalid segment payload"),
            Error::InvalidPumpStatus => write!(f, "invalid pump status payload"),
            Error::BufferTooSmall { needed, available } => write!(
                f,
                "buffer too small: needed {}, available {}",
                needed, available
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Parse ASCII decimal u32 without allocation; returns None on failure.
pub fn parse_u32(s: &str) -> Option<u32> {
    let mut acc: u32 = 0;
    if s.is_empty() {
        return None;
    }
    for b in s.as_bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?;
        acc = acc.checked_add((b - b'0') as u32)?;
    }
    Some(acc)
}

/// Parse ASCII decimal u16 without allocation; returns None on failure.
pub fn parse_u16(s: &str) -> Option<u16> {
    parse_u32(s).and_then(|v| u16::try_from(v).ok())
}

/// Parse ASCII decimal u8 without allocation; returns None on failure.
pub fn parse_u8(s: &str) -> Option<u8> {
    parse_u32(s).and_then(|v| u8::try_from(v).ok())
}

fn write_byte(b: u8, out: &mut [u8]) -> Result<usize, Error> {
    if out.is_empty() {
        return Err(Error::BufferTooSmall {
            needed: 1,
            available: 0,
        });
    }
    out[0] = b;
    Ok(1)
}

/// Write an ASCII decimal representation of `value` into `out`.
/// Returns bytes written or BufferTooSmall.
fn write_uint(mut value: u32, out: &mut [u8]) -> Result<usize, Error> {
    // max 10 digits for u32
    let mut buf = [0u8; 10];
    let mut i = 10;
    if value == 0 {
        buf[9] = b'0';
        i = 9;
    }
    while value > 0 {
        i -= 1;
        buf[i] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    let digits = &buf[i..];
    if out.len() < digits.len() {
        return Err(Error::BufferTooSmall {
            needed: digits.len(),
            available: out.len(),
        });
    }
    out[..digits.len()].copy_from_slice(digits);
    Ok(digits.len())
}

// --- Pump-gated control protocol extensions ---

/// Recording modes supported by the unified protocol.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RecordingModeKind {
    Continuous,
    PumpGated,
}

fn parse_mode(s: &str) -> Option<RecordingModeKind> {
    match s {
        "continuous" => Some(RecordingModeKind::Continuous),
        "pump_gated" => Some(RecordingModeKind::PumpGated),
        _ => None,
    }
}

fn mode_str(m: RecordingModeKind) -> &'static str {
    match m {
        RecordingModeKind::Continuous => "continuous",
        RecordingModeKind::PumpGated => "pump_gated",
    }
}

/// Segment label: whether the segment corresponds to pump ON or OFF, or undefined (continuous).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SegmentLabel {
    On,
    Off,
    Undefined,
}

fn parse_label(s: &str) -> Option<SegmentLabel> {
    match s {
        "on" => Some(SegmentLabel::On),
        "off" => Some(SegmentLabel::Off),
        "undefined" => Some(SegmentLabel::Undefined),
        _ => None,
    }
}

fn write_str(src: &str, out: &mut [u8]) -> Result<usize, Error> {
    if out.len() < src.len() {
        return Err(Error::BufferTooSmall {
            needed: src.len(),
            available: out.len(),
        });
    }
    out[..src.len()].copy_from_slice(src.as_bytes());
    Ok(src.len())
}

/// Stop reasons for segment termination.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    Normal,
    Canceled,
    SafetyCutoff,
    Timeout,
}

fn parse_reason(s: &str) -> Option<StopReason> {
    match s {
        "normal" => Some(StopReason::Normal),
        "canceled" => Some(StopReason::Canceled),
        "safety" | "safetycutoff" => Some(StopReason::SafetyCutoff),
        "timeout" => Some(StopReason::Timeout),
        _ => None,
    }
}

fn reason_str(r: StopReason) -> &'static str {
    match r {
        StopReason::Normal => "normal",
        StopReason::Canceled => "canceled",
        StopReason::SafetyCutoff => "safety",
        StopReason::Timeout => "timeout",
    }
}

/// Segment payload encoded as "cycle:label:reason:plan_ms".
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SegmentMsg {
    pub cycle_id: u32,
    pub label: SegmentLabel,
    pub prev_reason: StopReason,
    pub plan_ms: u32,
}

impl SegmentMsg {
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let s = core::str::from_utf8(payload).map_err(|_| Error::InvalidUtf8)?;
        let mut parts = s.splitn(4, ':');
        let cycle = parts.next().ok_or(Error::InvalidSegment)?;
        let label_s = parts.next().ok_or(Error::InvalidSegment)?;
        let reason_s = parts.next().ok_or(Error::InvalidSegment)?;
        let plan_s = parts.next().ok_or(Error::InvalidSegment)?;
        let cycle_id = parse_u32(cycle).ok_or(Error::InvalidSegment)?;
        let label = parse_label(label_s).ok_or(Error::InvalidSegment)?;
        let prev_reason = parse_reason(reason_s).ok_or(Error::InvalidSegment)?;
        let plan_ms = parse_u32(plan_s).ok_or(Error::InvalidSegment)?;
        Ok(SegmentMsg {
            cycle_id,
            label,
            prev_reason,
            plan_ms,
        })
    }

    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize, Error> {
        let mut n = 0;
        n += write_uint(self.cycle_id, &mut out[n..])?;
        n += write_byte(b':', &mut out[n..])?;
        let label_s = match self.label {
            SegmentLabel::On => "on",
            SegmentLabel::Off => "off",
            SegmentLabel::Undefined => "undefined",
        };
        n += write_str(label_s, &mut out[n..])?;
        n += write_byte(b':', &mut out[n..])?;
        n += write_str(reason_str(self.prev_reason), &mut out[n..])?;
        n += write_byte(b':', &mut out[n..])?;
        n += write_uint(self.plan_ms, &mut out[n..])?;
        Ok(n)
    }
}

/// Pump state: On or Off.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PumpState {
    On,
    Off,
}

fn parse_pump_state(s: &str) -> Option<PumpState> {
    match s {
        "on" => Some(PumpState::On),
        "off" => Some(PumpState::Off),
        _ => None,
    }
}

/// Pump status message payload: "status".
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PumpStatusMsg {
    pub status: PumpState,
}

impl PumpStatusMsg {
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let s = core::str::from_utf8(payload).map_err(|_| Error::InvalidUtf8)?;
        let status = parse_pump_state(s).ok_or(Error::InvalidPumpStatus)?;
        Ok(PumpStatusMsg { status })
    }

    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize, Error> {
        let s = match self.status {
            PumpState::On => "on",
            PumpState::Off => "off",
        };
        write_str(s, out)
    }
}

// --- Tests (use std) ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = Header {
            msg_type: MessageType::Audio,
            length: 0x00FF_FF,
            sequence: 42,
        };
        let mut buf = [0u8; HEADER_LEN];
        h.encode_into(&mut buf);
        assert_eq!(buf[0], 0x03);
        let p = Header::parse(&buf).unwrap();
        assert_eq!(p, h);
    }

    #[test]
    fn hello_parse_and_encode() {
        let payload = b"dev123:1.0";
        let h = Hello::parse(payload).unwrap();
        assert_eq!(h.device_id, "dev123");
        assert_eq!(h.version, "1.0");
        let mut out = [0u8; 16];
        let n = h.encode_into(&mut out).unwrap();
        assert_eq!(&out[..n], payload);
    }

    #[test]
    fn metadata_parse_and_encode() {
        let payload = b"16000:2:16:pump_gated";
        let m = Metadata::parse(payload).unwrap();
        assert_eq!(m.sample_rate, 16000);
        assert_eq!(m.channels, 2);
        assert_eq!(m.bits_per_sample, 16);
        assert_eq!(m.recording_mode, RecordingModeKind::PumpGated);
        let mut out = [0u8; 32];
        let n = m.encode_into(&mut out).unwrap();
        assert_eq!(&out[..n], payload);
    }

    // No legacy support required; protocol requires 4 fields.

    #[test]
    fn segment_roundtrip() {
        let payload = b"5:off:normal:1234";
        let s = SegmentMsg::parse(payload).unwrap();
        assert_eq!(s.cycle_id, 5);
        assert_eq!(s.label, SegmentLabel::Off);
        assert_eq!(s.prev_reason, StopReason::Normal);
        assert_eq!(s.plan_ms, 1234);
        let mut out = [0u8; 32];
        let n = s.encode_into(&mut out).unwrap();
        assert_eq!(&out[..n], payload);
    }

    #[test]
    fn pump_status_roundtrip() {
        let payload = b"on";
        let p = PumpStatusMsg::parse(payload).unwrap();
        assert_eq!(p.status, PumpState::On);
        let mut out = [0u8; 16];
        let n = p.encode_into(&mut out).unwrap();
        assert_eq!(&out[..n], payload);

        let payload = b"off";
        let p = PumpStatusMsg::parse(payload).unwrap();
        assert_eq!(p.status, PumpState::Off);
        let mut out = [0u8; 16];
        let n = p.encode_into(&mut out).unwrap();
        assert_eq!(&out[..n], payload);
    }
}
