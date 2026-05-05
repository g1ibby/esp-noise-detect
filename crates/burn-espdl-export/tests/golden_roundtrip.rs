//! Step-2 test: golden `.espdl` round-trip through the public API.
//!
//! Reads `/tmp/nn-rs-robust-cuda/export/model.espdl`, walks the graph
//! through [`EspdlFile`] (the reader half of Step 2) and re-emits it via
//! [`write_model`] (the writer half). The reserialized bytes must parse
//! back into a structurally identical [`dl::Model`].
//!
//! Byte-equality with the original file is **not** required — the
//! FlatBuffers spec does not promise stable field ordering, so the
//! same logical graph can have multiple valid binary encodings. The
//! Step-2 acceptance bar is structural equality across every field
//! the device-side loader consumes (see `tests/common/mod.rs`).

mod common;

use burn_espdl_export::{EspdlContainer, EspdlFile, write_model};

#[test]
fn golden_espdl_roundtrips_through_writer() {
    let Some(bytes) = common::read_golden() else {
        return;
    };

    let original = EspdlFile::parse(&bytes).expect("parse golden");
    let reserialized_bytes = write_model(&original.model());

    assert_eq!(
        &reserialized_bytes[0..4],
        b"EDL2",
        "writer emits EDL2 magic"
    );
    let reparsed = EspdlFile::parse(&reserialized_bytes).expect("re-parse writer output");

    common::assert_models_structurally_equal(&original.model(), &reparsed.model());

    // Container framing must round-trip independently of the FlatBuffers payload.
    let unpacked = EspdlContainer::parse(&reserialized_bytes).expect("container parse");
    assert_eq!(
        unpacked.payload,
        reparsed.payload(),
        "writer's container payload matches the reader's view"
    );
}
