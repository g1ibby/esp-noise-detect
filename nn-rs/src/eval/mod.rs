//! Evaluation pipeline.
//!
//! * [`metrics`] — full binary-classification metric bundle
//!   (macro / per-class F1, balanced accuracy, AUROC, AUPRC,
//!   confusion counts) plus coarse-to-fine threshold calibration.
//! * [`audit`] — opt-in per-session / per-day error audit, grouping
//!   window-level predictions by session timestamp.
//! * [`runner`] — the [`Evaluator`](runner::Evaluator) loop: loads
//!   a checkpoint, walks the requested split, aggregates to
//!   file level, writes artifacts.

pub mod audit;
pub mod metrics;
pub mod runner;

pub use metrics::{BinaryMetrics, binary_metrics, calibrate_threshold, compute_auprc, compute_auroc};
pub use runner::{AggregateMode, EvalOptions, EvalOutcome, Evaluator};
