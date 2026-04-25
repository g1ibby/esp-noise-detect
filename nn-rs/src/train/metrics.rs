//! Host-side classification metrics used by the training loop.
//!
//! The training loop needs macro-F1 to drive early stopping and
//! checkpoint selection, plus accuracy / mean loss for the progress
//! print. The full metric bundle lives in `crate::eval::metrics`.
//!
//! All accumulation happens on the host. Per-batch predictions arrive
//! via `update(logits, targets)` after pulling data off the GPU.

/// Running tally over one validation (or training) epoch.
///
/// `update` takes `logits` with shape `(batch, n_classes)` flattened
/// row-major, and `targets` with shape `(batch,)` as class indices.
#[derive(Clone, Debug, Default)]
pub struct BinaryClassificationStats {
    /// `[TP_class_0, TP_class_1, ...]` etc. For binary tasks the
    /// confusion matrix is 2×2.
    tp: [u64; 2],
    fp: [u64; 2],
    fn_: [u64; 2],
    n_total: u64,
    loss_sum: f64,
    loss_count: u64,
}

impl BinaryClassificationStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with one batch's worth of logits and targets.
    ///
    /// `n_classes` is baked in at 2 — pump-off / pump-on. Asserts the
    /// batch size matches between the two slices.
    pub fn update(&mut self, logits: &[f32], targets: &[i64]) {
        let n_classes = 2;
        assert_eq!(
            logits.len(),
            targets.len() * n_classes,
            "logits shape mismatch: {} elems for {} targets at n_classes={n_classes}",
            logits.len(),
            targets.len()
        );
        for (batch_i, &target) in targets.iter().enumerate() {
            let row = &logits[batch_i * n_classes..(batch_i + 1) * n_classes];
            let pred = row
                .iter()
                .enumerate()
                .fold((0usize, row[0]), |(best_i, best_v), (i, &v)| {
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_i, best_v)
                    }
                })
                .0;
            let target = target as usize;
            assert!(target < n_classes, "target {target} out of range");
            if pred == target {
                self.tp[pred] += 1;
            } else {
                self.fp[pred] += 1;
                self.fn_[target] += 1;
            }
            self.n_total += 1;
        }
    }

    /// Record the mean loss for this batch (for progress reporting only).
    pub fn add_loss(&mut self, loss: f32, batch_size: usize) {
        self.loss_sum += loss as f64 * batch_size as f64;
        self.loss_count += batch_size as u64;
    }

    /// Mean cross-entropy loss since the last `reset`.
    pub fn mean_loss(&self) -> f64 {
        if self.loss_count == 0 {
            0.0
        } else {
            self.loss_sum / self.loss_count as f64
        }
    }

    /// Macro-F1 over the two classes.
    pub fn macro_f1(&self) -> f64 {
        let mut sum = 0.0;
        for c in 0..2 {
            let tp = self.tp[c] as f64;
            let fp = self.fp[c] as f64;
            let fn_ = self.fn_[c] as f64;
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            sum += f1;
        }
        sum / 2.0
    }

    /// Overall accuracy (micro, i.e. total-correct / total).
    pub fn accuracy(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            (self.tp[0] + self.tp[1]) as f64 / self.n_total as f64
        }
    }

    pub fn total(&self) -> u64 {
        self.n_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions_give_f1_one() {
        let mut stats = BinaryClassificationStats::new();
        // Row i: logits point to target class i.
        let logits = [2.0, -2.0, -2.0, 2.0, 2.0, -2.0];
        let targets = [0i64, 1, 0];
        stats.update(&logits, &targets);
        assert!((stats.macro_f1() - 1.0).abs() < 1e-9);
        assert!((stats.accuracy() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn all_wrong_gives_f1_zero() {
        let mut stats = BinaryClassificationStats::new();
        let logits = [2.0, -2.0, -2.0, 2.0, 2.0, -2.0];
        let targets = [1i64, 0, 1];
        stats.update(&logits, &targets);
        assert!(stats.macro_f1() < 1e-9);
        assert!(stats.accuracy() < 1e-9);
    }

    #[test]
    fn macro_f1_averages_per_class() {
        // 4 samples: 3 correct class-0, 1 wrong class-1 predicted as class-0.
        // tp0=3, fp0=1 (one wrong class-1 was predicted 0), fn0=0.
        // tp1=0, fp1=0, fn1=1.
        // F1_0 = 2*3/(2*3 + 1 + 0) = 6/7; F1_1 = 0; macro = 3/7.
        let mut stats = BinaryClassificationStats::new();
        let logits = [
            1.0, -1.0, // pred 0, true 0
            1.0, -1.0, // pred 0, true 0
            1.0, -1.0, // pred 0, true 0
            1.0, -1.0, // pred 0, true 1
        ];
        let targets = [0i64, 0, 0, 1];
        stats.update(&logits, &targets);
        let expected = 0.5 * ((2.0 * 3.0 / (2.0 * 3.0 + 1.0 + 0.0)) + 0.0);
        assert!(
            (stats.macro_f1() - expected).abs() < 1e-9,
            "got {}",
            stats.macro_f1()
        );
    }

    #[test]
    fn mean_loss_weighted_by_batch_size() {
        let mut stats = BinaryClassificationStats::new();
        stats.add_loss(2.0, 10); // sum=20, count=10
        stats.add_loss(1.0, 30); // sum=50, count=40
        assert!((stats.mean_loss() - 1.25).abs() < 1e-9);
    }
}
