//! `DatasetConfig` + `seconds_to_samples`.

use std::path::PathBuf;

/// Audio dataset configuration. The pump-noise class set is
/// `["pump_off", "pump_on"]` (label index 0 and 1 respectively).
#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub sample_rate: u32,
    pub window_s: f32,
    pub hop_s: f32,
    pub class_names: Vec<String>,
    pub manifest_path: Option<PathBuf>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            sample_rate: 32_000,
            window_s: 1.0,
            hop_s: 0.5,
            class_names: vec!["pump_off".to_string(), "pump_on".to_string()],
            manifest_path: None,
        }
    }
}

/// Convert `(window_s, hop_s)` to `(window_samples, hop_samples)`, each
/// floored to 1.
pub fn seconds_to_samples(window_s: f32, hop_s: f32, sample_rate: u32) -> (usize, usize) {
    let ws = (window_s * sample_rate as f32).round() as i64;
    let hs = (hop_s * sample_rate as f32).round() as i64;
    (ws.max(1) as usize, hs.max(1) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let cfg = DatasetConfig::default();
        assert_eq!(cfg.sample_rate, 32_000);
        assert!((cfg.window_s - 1.0).abs() < 1e-6);
        assert!((cfg.hop_s - 0.5).abs() < 1e-6);
        assert_eq!(cfg.class_names, vec!["pump_off", "pump_on"]);
        assert!(cfg.manifest_path.is_none());
    }

    #[test]
    fn seconds_to_samples_floors_to_one() {
        // 1 s window, 0.5 s hop at 32 kHz — matches robust_session.
        assert_eq!(seconds_to_samples(1.0, 0.5, 32_000), (32_000, 16_000));
        // Zero-length inputs clamp to 1.
        assert_eq!(seconds_to_samples(0.0, 0.0, 32_000), (1, 1));
    }
}
