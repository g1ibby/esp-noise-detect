//! `.espdl` container framing.
//!
//! On disk an `.espdl` is a 16-byte header followed by a FlatBuffers
//! payload that the device loader parses as a `dl::Model`. Layout
//! mirrors `esp_ppq/parser/espdl/helper.py::save` (lines 682-694):
//!
//! ```text
//! offset 0   "EDL2" (4 bytes)
//! offset 4   encrypt flag      (u32 little-endian, 0 = unencrypted)
//! offset 8   payload size      (u32 little-endian, in bytes)
//! offset 12  4 bytes of zero padding
//! offset 16  FlatBuffers payload (Model root)
//! ```
//!
//! ESP32-S3 inference does not use the encrypted (`PDL2`) variant, so
//! this module deliberately only handles the unencrypted single-model
//! `EDL2` form.

const MAGIC: &[u8; 4] = b"EDL2";
const HEADER_LEN: usize = 16;

/// Failure modes when parsing the 16-byte `EDL2` container header.
#[derive(Debug, Clone)]
pub enum EspdlContainerError {
    /// Buffer is shorter than the 16-byte header.
    TooShort { got: usize },
    /// First four bytes are not the ASCII magic `"EDL2"`.
    BadMagic { got: [u8; 4] },
    /// `encrypt` flag is non-zero. The encrypted `PDL2` variant is
    /// deliberately not supported.
    Encrypted,
    /// Header-declared payload size disagrees with the bytes that follow.
    SizeMismatch { declared: u32, actual: usize },
}

impl core::fmt::Display for EspdlContainerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort { got } => write!(
                f,
                "espdl: file shorter than 16-byte header (got {got} bytes)"
            ),
            Self::BadMagic { got } => write!(f, "espdl: expected magic 'EDL2', got {got:?}"),
            Self::Encrypted => write!(
                f,
                "espdl: encrypted payload (encrypt flag != 0) not supported"
            ),
            Self::SizeMismatch { declared, actual } => {
                write!(
                    f,
                    "espdl: header declares payload {declared} bytes, file has {actual} bytes after header"
                )
            }
        }
    }
}

impl std::error::Error for EspdlContainerError {}

/// Parsed `.espdl` container with borrowed payload bytes.
#[derive(Debug, Clone, Copy)]
pub struct EspdlContainer<'a> {
    /// FlatBuffers payload (`Model` root) without the 16-byte `EDL2` header.
    pub payload: &'a [u8],
}

impl<'a> EspdlContainer<'a> {
    /// Parse the 16-byte `EDL2` header and return a view of the
    /// FlatBuffers payload. Refuses encrypted payloads.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, EspdlContainerError> {
        if bytes.len() < HEADER_LEN {
            return Err(EspdlContainerError::TooShort { got: bytes.len() });
        }
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if &magic != MAGIC {
            return Err(EspdlContainerError::BadMagic { got: magic });
        }
        let encrypt = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if encrypt != 0 {
            return Err(EspdlContainerError::Encrypted);
        }
        let declared = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload = &bytes[HEADER_LEN..];
        if payload.len() != declared as usize {
            return Err(EspdlContainerError::SizeMismatch {
                declared,
                actual: payload.len(),
            });
        }
        Ok(Self { payload })
    }

    /// Build an `EDL2` container around an existing FlatBuffers payload.
    /// Returns the full file bytes (header + payload).
    pub fn pack(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(0u32).to_le_bytes()); // encrypt flag
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 4]); // padding
        out.extend_from_slice(payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_then_parse_roundtrips() {
        let payload = b"flatbuffers-go-here";
        let bytes = EspdlContainer::pack(payload);
        assert_eq!(&bytes[0..4], MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            payload.len() as u32
        );
        let parsed = EspdlContainer::parse(&bytes).expect("parse");
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = EspdlContainer::pack(b"x");
        bytes[0] = b'X';
        assert!(matches!(
            EspdlContainer::parse(&bytes),
            Err(EspdlContainerError::BadMagic { .. })
        ));
    }

    #[test]
    fn rejects_encrypted() {
        let mut bytes = EspdlContainer::pack(b"x");
        bytes[4] = 1;
        assert!(matches!(
            EspdlContainer::parse(&bytes),
            Err(EspdlContainerError::Encrypted)
        ));
    }
}
