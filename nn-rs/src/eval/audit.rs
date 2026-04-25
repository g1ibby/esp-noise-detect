//! Opt-in per-session / per-day error audit.
//!
//! Window-level predictions are grouped by session-timestamp and UTC
//! day parsed from the file name, and the worst-accuracy groups are
//! printed first. Useful when a regression clusters on one noisy
//! session instead of being spread evenly across the split.
//!
//! The filename pattern we match is the one produced by the ESP32
//! firmware's dataset collector:
//! `xiao_esp32s3_<ts>_c<nnn>_<label>_chunk<nnn>.wav` where `<ts>` is a
//! 10-digit Unix timestamp. Windows whose filename doesn't contain a
//! 10-digit `_<digits>_` group are skipped and counted separately so
//! surprise drift doesn't fail the audit silently.

use std::collections::HashMap;
use std::path::Path;

/// Parse a `(session_ts, day)` tuple from a file path. Both entries are
/// `None` if the filename doesn't match the `_<10 digits>_` shape.
/// Day is formatted as `YYYY-MM-DD` in UTC.
pub fn audit_key_of(path: &Path) -> (Option<u64>, Option<String>) {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return (None, None);
    };
    let bytes = name.as_bytes();
    // Need at least `_xxxxxxxxxx_` — 12 bytes.
    if bytes.len() < 12 {
        return (None, None);
    }
    // Scan for `_<10 digits>_`: first occurrence, leftmost.
    for i in 0..=bytes.len() - 12 {
        if bytes[i] == b'_'
            && bytes[i + 11] == b'_'
            && bytes[i + 1..i + 11].iter().all(|b| b.is_ascii_digit())
        {
            let digits = std::str::from_utf8(&bytes[i + 1..i + 11]).unwrap();
            let ts: u64 = match digits.parse() {
                Ok(v) => v,
                Err(_) => return (None, None),
            };
            return (Some(ts), Some(format_utc_day(ts)));
        }
    }
    (None, None)
}

/// Format a Unix timestamp as `YYYY-MM-DD` in UTC.
///
/// Uses Howard Hinnant's civil-from-days formula. No external crate —
/// we only need one call path in this whole codebase so a fresh
/// dependency is unjustified.
fn format_utc_day(ts: u64) -> String {
    let days = (ts / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y_base = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y_base + 1 } else { y_base };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Print the per-day / per-session / per-file audit table to stdout.
///
/// `files`, `probs`, `labels` are parallel slices — one entry per
/// window. `top_n` caps the worst-file list.
pub fn print_audit(files: &[&Path], probs: &[f32], labels: &[u8], threshold: f32, top_n: usize) {
    if files.is_empty() {
        println!("[audit] no window data available (set window_metrics: true)");
        return;
    }
    assert_eq!(files.len(), probs.len(), "audit: files/probs length mismatch");
    assert_eq!(files.len(), labels.len(), "audit: files/labels length mismatch");

    let mut per_session = HashMap::<u64, Bucket>::new();
    let mut per_day = HashMap::<String, Bucket>::new();
    let mut per_file_fp = HashMap::<String, u64>::new();
    let mut per_file_fn = HashMap::<String, u64>::new();
    let mut skipped = 0u64;

    for (&f, (&p, &y)) in files.iter().zip(probs.iter().zip(labels.iter())) {
        let pred: u8 = if p >= threshold { 1 } else { 0 };
        let (ts, day) = audit_key_of(f);
        let (Some(ts), Some(day)) = (ts, day) else {
            skipped += 1;
            continue;
        };
        let sbucket = per_session.entry(ts).or_default();
        let dbucket = per_day.entry(day).or_default();
        match (pred, y) {
            (1, 1) => {
                sbucket.tp += 1;
                dbucket.tp += 1;
            }
            (0, 0) => {
                sbucket.tn += 1;
                dbucket.tn += 1;
            }
            (1, 0) => {
                sbucket.fp += 1;
                dbucket.fp += 1;
                let key = f.to_string_lossy().into_owned();
                *per_file_fp.entry(key).or_insert(0) += 1;
            }
            _ => {
                sbucket.fn_ += 1;
                dbucket.fn_ += 1;
                let key = f.to_string_lossy().into_owned();
                *per_file_fn.entry(key).or_insert(0) += 1;
            }
        }
    }

    println!();
    println!("{}", "=".repeat(72));
    println!("Error audit (threshold={threshold:.3})");
    if skipped > 0 {
        println!("  skipped {skipped} windows whose filename has no session timestamp");
    }
    println!("{}", "=".repeat(72));

    println!("\nBy day (worst-first):");
    println!("  {:<12} {:>5} {:>7}  TP TN FP FN", "day", "n", "acc");
    let mut day_rows: Vec<_> = per_day.iter().map(|(k, b)| (k.clone(), b)).collect();
    day_rows.sort_by(|a, b| {
        let acc_a = a.1.accuracy();
        let acc_b = b.1.accuracy();
        acc_a
            .partial_cmp(&acc_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.total().cmp(&a.1.total()))
    });
    for (k, b) in &day_rows {
        println!(
            "  {:<12} {:>5} {:>7.4}  {} {} {} {}",
            k,
            b.total(),
            b.accuracy(),
            b.tp,
            b.tn,
            b.fp,
            b.fn_,
        );
    }

    println!("\nBy session (worst-first, up to 15):");
    println!("  {:>12} {:>5} {:>7}  TP TN FP FN", "session_ts", "n", "acc");
    let mut session_rows: Vec<_> = per_session.iter().collect();
    session_rows.sort_by(|a, b| {
        let acc_a = a.1.accuracy();
        let acc_b = b.1.accuracy();
        acc_a
            .partial_cmp(&acc_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.total().cmp(&a.1.total()))
    });
    for (k, b) in session_rows.iter().take(15) {
        println!(
            "  {:>12} {:>5} {:>7.4}  {} {} {} {}",
            k,
            b.total(),
            b.accuracy(),
            b.tp,
            b.tn,
            b.fp,
            b.fn_,
        );
    }

    if !per_file_fp.is_empty() {
        println!("\nTop {top_n} files by FP count:");
        let mut rows: Vec<_> = per_file_fp.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1));
        for (f, c) in rows.into_iter().take(top_n) {
            println!("  {c:>4}  {}", Path::new(f).file_name().and_then(|s| s.to_str()).unwrap_or(f));
        }
    }
    if !per_file_fn.is_empty() {
        println!("\nTop {top_n} files by FN count:");
        let mut rows: Vec<_> = per_file_fn.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1));
        for (f, c) in rows.into_iter().take(top_n) {
            println!("  {c:>4}  {}", Path::new(f).file_name().and_then(|s| s.to_str()).unwrap_or(f));
        }
    }
    println!("{}", "=".repeat(72));
}

#[derive(Default)]
struct Bucket {
    tp: u64,
    tn: u64,
    fp: u64,
    fn_: u64,
}

impl Bucket {
    fn total(&self) -> u64 {
        self.tp + self.tn + self.fp + self.fn_
    }
    fn accuracy(&self) -> f64 {
        let n = self.total().max(1) as f64;
        (self.tp + self.tn) as f64 / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_session_ts_from_firmware_filename() {
        let p = PathBuf::from("xiao_esp32s3_1759581195_c000_off_chunk000.wav");
        let (ts, day) = audit_key_of(&p);
        assert_eq!(ts, Some(1_759_581_195));
        assert_eq!(day.as_deref(), Some("2025-10-04"));
    }

    #[test]
    fn unparseable_filename_returns_none() {
        let p = PathBuf::from("random_clip_no_timestamp.wav");
        let (ts, day) = audit_key_of(&p);
        assert_eq!(ts, None);
        assert_eq!(day, None);
    }

    #[test]
    fn short_filename_returns_none() {
        let p = PathBuf::from("tiny.wav");
        let (ts, _) = audit_key_of(&p);
        assert_eq!(ts, None);
    }

    #[test]
    fn utc_day_crosses_day_boundary() {
        // 1970-01-01 00:00:00 UTC
        assert_eq!(format_utc_day(0), "1970-01-01");
        // 1970-01-01 23:59:59 UTC
        assert_eq!(format_utc_day(86_399), "1970-01-01");
        // 1970-01-02 00:00:00 UTC
        assert_eq!(format_utc_day(86_400), "1970-01-02");
        // 2024-02-29 12:00:00 UTC — leap year sanity check
        assert_eq!(format_utc_day(1_709_208_000), "2024-02-29");
    }
}
