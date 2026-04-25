//! Binary classification metrics used by the eval CLI.
//!
//! Full metric bundle: accuracy, balanced accuracy, macro / per-class
//! F1, AUROC, AUPRC, confusion counts, and the threshold used. The
//! training loop's metrics (`crate::train::metrics`) cover only
//! macro-F1 / accuracy / mean loss.
//!
//! Inputs are flat host buffers — one `y_true` byte per file (or
//! window) and one `y_prob` float (probability of the positive class).
//! Keeping them as plain slices lets the same code feed both
//! file-level and window-level metrics.

use serde::{Deserialize, Serialize};

/// Full binary-classification metric bundle. `fn_` is serialized as
/// `fn` (Rust keyword) via serde rename.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryMetrics {
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub macro_f1: f64,
    pub f1_pos: f64,
    pub f1_neg: f64,
    pub auroc: f64,
    pub auprc: f64,
    pub tp: f64,
    pub tn: f64,
    pub fp: f64,
    /// Serialized as `fn` (Rust keyword — renamed on the wire).
    #[serde(rename = "fn")]
    pub fn_: f64,
    pub threshold: f64,
}

/// Compute the full metric bundle at a fixed decision threshold.
///
/// `y_true` values are expected to be `0` (negative class) or `1`
/// (positive class). `y_prob` is the positive-class probability in
/// `[0, 1]`. Panics on length mismatch.
pub fn binary_metrics(y_true: &[u8], y_prob: &[f32], threshold: f32) -> BinaryMetrics {
    assert_eq!(
        y_true.len(),
        y_prob.len(),
        "binary_metrics: y_true and y_prob must have the same length",
    );
    let (mut tp, mut tn, mut fp, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for (&y, &p) in y_true.iter().zip(y_prob.iter()) {
        let pred = if p >= threshold { 1 } else { 0 };
        match (pred, y) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            _ => fn_ += 1,
        }
    }
    let total = (tp + tn + fp + fn_).max(1) as f64;
    let acc = (tp + tn) as f64 / total;
    let f1_pos = f1(tp, fp, fn_);
    let f1_neg = f1(tn, fn_, fp);
    let macro_f1 = 0.5 * (f1_pos + f1_neg);
    let tpr = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let tnr = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    let balanced_accuracy = 0.5 * (tpr + tnr);
    let auroc = compute_auroc(y_true, y_prob);
    let auprc = compute_auprc(y_true, y_prob);
    BinaryMetrics {
        accuracy: acc,
        balanced_accuracy,
        macro_f1,
        f1_pos,
        f1_neg,
        auroc,
        auprc,
        tp: tp as f64,
        tn: tn as f64,
        fp: fp as f64,
        fn_: fn_ as f64,
        threshold: threshold as f64,
    }
}

fn f1(tp: u64, fp: u64, fn_: u64) -> f64 {
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Area under the ROC curve via the rank-sum (Mann-Whitney U)
/// formulation. Ties contribute half a rank each.
///
/// Returns `NaN` when the split is degenerate (all-positive or
/// all-negative).
pub fn compute_auroc(y_true: &[u8], y_prob: &[f32]) -> f64 {
    let n_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;
    let n_neg = y_true.iter().filter(|&&y| y == 0).count() as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    let mut pairs: Vec<(f32, u8)> = y_prob
        .iter()
        .zip(y_true.iter())
        .map(|(p, l)| (*p, *l))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Samples [i, j) share the same score; average rank is ((i+1) + j) / 2.
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            if pairs[k].1 == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0;
    u / (n_pos * n_neg)
}

/// Average precision (AUPRC).
///
/// Sort by descending score; at each unique threshold the precision /
/// recall pair forms one PR step, and AP accumulates
/// `(recall_i - recall_{i-1}) * precision_i`. Returns `NaN` when
/// there are no positives.
pub fn compute_auprc(y_true: &[u8], y_prob: &[f32]) -> f64 {
    let n_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;
    if n_pos == 0.0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = (0..y_prob.len()).collect();
    idx.sort_by(|&a, &b| {
        y_prob[b]
            .partial_cmp(&y_prob[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut ap = 0.0_f64;
    let mut prev_recall = 0.0_f64;
    let mut i = 0;
    while i < idx.len() {
        let cur = y_prob[idx[i]];
        let mut j = i;
        while j < idx.len() && y_prob[idx[j]] == cur {
            if y_true[idx[j]] == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            j += 1;
        }
        let precision = tp / (tp + fp);
        let recall = tp / n_pos;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
        i = j;
    }
    ap
}

/// Coarse-to-fine threshold calibration, maximizing macro-F1.
///
/// Three passes at step sizes `[0.01, 0.002, 0.001]`; each pass scans a
/// `+-0.05` window centered on the best threshold found so far. Starts
/// centered at 0.5.
pub fn calibrate_threshold(y_true: &[u8], y_prob: &[f32]) -> f32 {
    let mut best_t: f32 = 0.5;
    let mut best_v: f64 = -1.0;
    for &step in &[0.01_f32, 0.002, 0.001] {
        let start = (best_t - 0.05).max(0.0);
        let end = (best_t + 0.05).min(1.0);
        let mut t = start;
        // Inclusive upper bound.
        while t <= end + 1e-9 {
            let m = binary_metrics(y_true, y_prob, t);
            if m.macro_f1.is_finite() && m.macro_f1 > best_v {
                best_v = m.macro_f1;
                best_t = t;
            }
            t += step;
        }
    }
    best_t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_separation_gives_unity_auroc_auprc() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.1, 0.2, 0.8, 0.9];
        assert!((compute_auroc(&y_true, &y_prob) - 1.0).abs() < 1e-9);
        assert!((compute_auprc(&y_true, &y_prob) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fully_wrong_gives_zero_auroc() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.9, 0.8, 0.2, 0.1];
        assert!(compute_auroc(&y_true, &y_prob).abs() < 1e-9);
    }

    #[test]
    fn all_same_score_gives_half_auroc() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.5, 0.5, 0.5, 0.5];
        assert!((compute_auroc(&y_true, &y_prob) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn auroc_ties_use_half_rank() {
        // One tie straddling a positive and a negative.
        // ranks sorted ascending: prob=[0.3(neg), 0.5(neg), 0.5(pos), 0.9(pos)].
        // tied pair at rank avg = (2 + 3)/2 = 2.5 for both.
        // rank_sum_pos = 2.5 + 4 = 6.5; U = 6.5 - 3 = 3.5; AUROC = 3.5/4 = 0.875.
        let y_true = [0u8, 0, 1, 1];
        let y_prob = [0.3_f32, 0.5, 0.5, 0.9];
        assert!((compute_auroc(&y_true, &y_prob) - 0.875).abs() < 1e-9);
    }

    #[test]
    fn perfect_split_metrics() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.1, 0.2, 0.8, 0.9];
        let m = binary_metrics(&y_true, &y_prob, 0.5);
        assert!((m.accuracy - 1.0).abs() < 1e-9);
        assert!((m.macro_f1 - 1.0).abs() < 1e-9);
        assert!((m.f1_pos - 1.0).abs() < 1e-9);
        assert!((m.f1_neg - 1.0).abs() < 1e-9);
        assert!((m.balanced_accuracy - 1.0).abs() < 1e-9);
        assert_eq!(m.tp, 2.0);
        assert_eq!(m.tn, 2.0);
        assert_eq!(m.fp, 0.0);
        assert_eq!(m.fn_, 0.0);
    }

    #[test]
    fn calibration_finds_perfect_threshold() {
        // With clean separation at prob=0.5, any threshold in
        // (0.2, 0.8) gives macro_f1=1.0; calibration must find one.
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.1_f32, 0.2, 0.8, 0.9];
        let t = calibrate_threshold(&y_true, &y_prob);
        let m = binary_metrics(&y_true, &y_prob, t);
        assert!((m.macro_f1 - 1.0).abs() < 1e-9, "t={t} macro_f1={}", m.macro_f1);
    }

    #[test]
    fn serde_renames_fn_field() {
        let m = BinaryMetrics {
            accuracy: 1.0,
            balanced_accuracy: 1.0,
            macro_f1: 1.0,
            f1_pos: 1.0,
            f1_neg: 1.0,
            auroc: 1.0,
            auprc: 1.0,
            tp: 1.0,
            tn: 1.0,
            fp: 0.0,
            fn_: 0.0,
            threshold: 0.5,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"fn\":0"), "fn_ field must serialize as `fn`: {json}");
    }
}
