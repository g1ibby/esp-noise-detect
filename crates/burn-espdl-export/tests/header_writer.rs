//! Step-2 test: empty graph header layout.
//!
//! Builds a `.espdl` file containing a single empty `Graph` and asserts:
//!
//! 1. The first 16 bytes match the `EDL2` header layout (magic, encrypt
//!    flag = 0, payload size, 4 bytes of zero padding).
//! 2. The declared payload size in the header equals the FlatBuffers
//!    payload that follows it.
//! 3. The payload parses cleanly back through [`EspdlFile`] and contains
//!    a graph with no nodes / initializers / IO.

use burn_espdl_export::{EspdlFile, write_empty};

#[test]
fn empty_graph_writes_valid_edl2_header() {
    let bytes = write_empty();

    assert!(
        bytes.len() > 16,
        "empty graph file must contain at least the 16-byte header plus a non-empty FlatBuffers payload, got {} bytes",
        bytes.len()
    );
    assert_eq!(&bytes[0..4], b"EDL2", "magic");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        0,
        "encrypt flag (must be 0; encrypted PDL2 is not supported)"
    );
    let declared = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    assert_eq!(
        declared,
        bytes.len() - 16,
        "payload size in header must match actual payload length"
    );
    assert_eq!(
        &bytes[12..16],
        &[0u8; 4],
        "trailing 4 bytes of header padding"
    );

    // Round-trip through the reader to confirm the empty graph parses.
    let parsed = EspdlFile::parse(&bytes).expect("parse");
    let model = parsed.model();
    let graph = model.graph().expect("model has a graph");

    let no_nodes = graph.node().map(|v| v.len()).unwrap_or(0);
    let no_inits = graph.initializer().map(|v| v.len()).unwrap_or(0);
    let no_input = graph.input().map(|v| v.len()).unwrap_or(0);
    let no_output = graph.output().map(|v| v.len()).unwrap_or(0);
    assert_eq!(no_nodes, 0, "empty graph has no nodes");
    assert_eq!(no_inits, 0, "empty graph has no initializers");
    assert_eq!(no_input, 0, "empty graph has no inputs");
    assert_eq!(no_output, 0, "empty graph has no outputs");
}
