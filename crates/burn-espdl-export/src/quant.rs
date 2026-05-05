//! Integer quantization helpers for Step 5.
//!
//! Mirrors the S3 path of esp-ppq's
//! `PPQLinearQuant_toInt`: divide by the power-of-two scale,
//! `ROUND_HALF_UP` via `floor(x + 0.5)`, then clamp to the target
//! signed range.

use crate::QuantConfig;

/// Quantize `values` into signed integers using `cfg`.
pub fn quantize_i64(values: &[f32], cfg: QuantConfig) -> Vec<i64> {
    let (qmin, qmax) = signed_range(cfg.num_bits);
    values
        .iter()
        .map(|&v| {
            let q = (v / cfg.scale + cfg.zero_point as f32 + 0.5).floor() as i64;
            q.clamp(qmin, qmax)
        })
        .collect()
}

/// Quantize to int8 storage.
pub fn quantize_i8(values: &[f32], cfg: QuantConfig) -> Vec<i8> {
    quantize_i64(values, cfg)
        .into_iter()
        .map(|v| v.clamp(i8::MIN as i64, i8::MAX as i64) as i8)
        .collect()
}

/// Quantize to int16 storage.
pub fn quantize_i16(values: &[f32], cfg: QuantConfig) -> Vec<i16> {
    quantize_i64(values, cfg)
        .into_iter()
        .map(|v| v.clamp(i16::MIN as i64, i16::MAX as i64) as i16)
        .collect()
}

/// Quantize to int32 storage. ESP-DL stores bias payloads as INT32 even
/// though the S3 INT16 config carries a wider logical bias bit-width.
pub fn quantize_i32(values: &[f32], cfg: QuantConfig) -> Vec<i32> {
    quantize_i64(values, cfg)
        .into_iter()
        .map(|v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect()
}

fn signed_range(bits: u8) -> (i64, i64) {
    match bits {
        0 => (0, 0),
        1..=62 => {
            let half = 1_i64 << (bits - 1);
            (-half, half - 1)
        }
        _ => (i64::MIN, i64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(scale: f32, num_bits: u8) -> QuantConfig {
        QuantConfig {
            scale,
            zero_point: 0,
            exponent: scale.log2() as i32,
            num_bits,
        }
    }

    #[test]
    fn round_half_up_matches_ppq_formula() {
        let out = quantize_i64(&[-1.6, -1.5, -1.4, -0.5, 0.5, 1.4, 1.5], cfg(1.0, 8));
        assert_eq!(out, vec![-2, -1, -1, 0, 1, 1, 2]);
    }

    #[test]
    fn clamps_to_signed_range() {
        let out = quantize_i8(&[-200.0, 0.0, 200.0], cfg(1.0, 8));
        assert_eq!(out, vec![-128, 0, 127]);
    }
}
