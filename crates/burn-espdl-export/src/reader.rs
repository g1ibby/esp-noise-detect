//! Tiny `.espdl` reader.
//!
//! The on-device loader (in the `edgedl` crate) is the *real* reader for
//! `.espdl` files. This module exists for tests on the host: it parses
//! the 16-byte `EDL2` container with [`EspdlContainer`] and verifies the
//! payload as a [`dl::Model`] FlatBuffers root, returning a single struct
//! that owns both views.
//!
//! Errors are reported as [`EspdlReadError`]; both container framing and
//! FlatBuffers verification problems land here.

use crate::container::{EspdlContainer, EspdlContainerError};
use crate::dl_generated::dl;

/// Failure modes when parsing `.espdl` bytes.
#[derive(Debug)]
pub enum EspdlReadError {
    /// Container header (`EDL2`, sizes, encryption flag) is malformed or
    /// describes an unsupported variant.
    Container(EspdlContainerError),
    /// FlatBuffers payload failed verification against the `Dl.fbs` schema.
    InvalidFlatbuffers(flatbuffers::InvalidFlatbuffer),
}

impl core::fmt::Display for EspdlReadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Container(e) => write!(f, "{e}"),
            Self::InvalidFlatbuffers(e) => write!(f, "espdl: invalid flatbuffers payload: {e}"),
        }
    }
}

impl std::error::Error for EspdlReadError {}

impl From<EspdlContainerError> for EspdlReadError {
    fn from(e: EspdlContainerError) -> Self {
        Self::Container(e)
    }
}

impl From<flatbuffers::InvalidFlatbuffer> for EspdlReadError {
    fn from(e: flatbuffers::InvalidFlatbuffer) -> Self {
        Self::InvalidFlatbuffers(e)
    }
}

/// Parsed `.espdl` file with borrowed payload bytes and a verified
/// [`dl::Model`] root.
#[derive(Debug, Clone, Copy)]
pub struct EspdlFile<'a> {
    container: EspdlContainer<'a>,
    model: dl::Model<'a>,
}

impl<'a> EspdlFile<'a> {
    /// Parse the 16-byte `EDL2` header and verify the FlatBuffers payload.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, EspdlReadError> {
        let container = EspdlContainer::parse(bytes)?;
        let model = flatbuffers::root::<dl::Model>(container.payload)?;
        Ok(Self { container, model })
    }

    /// Access the underlying `dl::Model` reader.
    pub fn model(&self) -> dl::Model<'a> {
        self.model
    }

    /// Borrow the FlatBuffers payload without the `EDL2` header.
    pub fn payload(&self) -> &'a [u8] {
        self.container.payload
    }
}
