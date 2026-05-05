//! BurnGraph IR — Step 3 deliverable.
//!
//! The IR is a flat `Vec<Layer>` produced by [`extract::from_tinyconv`]
//! and rewritten in place by [`fold_batchnorm`] and [`fuse_relu`].
//! Step 4 (calibration) consumes the post-fold/post-fuse form via
//! [`forward::forward`]; Step 5 (writer) consumes the same shape and
//! emits FlatBuffers nodes.
//!
//! Why these three rewrites and nothing else lives here:
//!
//! * BN folding is mathematically transparent at inference time and
//!   removes a node esp-ppq's golden export does not contain.
//! * Relu fusion turns a graph rewrite into an attribute on the
//!   upstream Conv, also matching the golden node count.
//! * NCHW→NHWC layout is *not* applied here — it lives in Step 5
//!   alongside the weight-pack transform, because per-tensor scales
//!   are layout-invariant and computing scales pre-rewrite keeps the
//!   IR easier to reason about.

mod bn_fold;
pub mod extract;
mod forward;
mod layer;
mod relu_fuse;

pub use bn_fold::fold_batchnorm;
pub use forward::{forward, forward_with_fake_quant_hook, forward_with_hook};
pub use layer::{Activation, BurnGraph, Layer, Tensor};
pub use relu_fuse::fuse_relu;
