use embassy_sync::watch;
use embassy_time::{Duration, Instant, Timer};
use esp_hal::{
    analog::adc::{Adc, AdcPin},
    peripherals::{ADC1, GPIO3},
};
use libm::{cosf, sinf, sqrtf};
use log::{info, warn};

use crate::app::config::PumpConfig;

const ADC_MAX_CODE: u16 = 4095;
const ADC_MIN_CODE: u16 = 0;

fn tone_rms(samples: &[u16], mean: f32, fs_hz: f32, target_hz: f32) -> u16 {
    let n = samples.len() as f32;
    let w = 2.0 * core::f32::consts::PI * target_hz / fs_hz;
    let mut proj_cos = 0.0f32;
    let mut proj_sin = 0.0f32;

    for (i, &raw) in samples.iter().enumerate() {
        let x = raw as f32 - mean;
        let angle = w * i as f32;
        proj_cos += x * cosf(angle);
        proj_sin += x * sinf(angle);
    }

    let amp_peak = (2.0 / n) * sqrtf(proj_cos * proj_cos + proj_sin * proj_sin);
    (amp_peak / core::f32::consts::SQRT_2) as u16
}

fn count_zero_crossings(samples: &[u16], dc: u16) -> u16 {
    if samples.len() < 2 {
        return 0;
    }

    let mut crossings: u16 = 0;
    let mut prev = samples[0] as i32 - dc as i32;

    for &sample in &samples[1..] {
        let cur = sample as i32 - dc as i32;
        if (prev < 0 && cur >= 0) || (prev > 0 && cur <= 0) {
            crossings = crossings.saturating_add(1);
        }
        prev = cur;
    }

    crossings
}

/// Board-specific pump monitor using ADC1 on GPIO3 / A2 (XIAO ESP32S3).
///
/// Samples a window of raw ADC counts, computes broadband RMS plus a
/// normalized 50Hz tone RMS, and applies hysteresis gated by zero crossings
/// to produce a stable ON/OFF signal.
#[embassy_executor::task]
pub async fn pump_monitor_xiao_adc(
    mut adc1: Adc<'static, ADC1<'static>, esp_hal::Blocking>,
    mut adc_pin: AdcPin<GPIO3<'static>, ADC1<'static>>,
    pump_state_tx: watch::DynSender<'static, bool>,
    cfg: PumpConfig,
) {
    let fs_hz = 1_000_000.0 / cfg.sample_interval_us as f32;
    info!(
        "PumpMonitor A2/GPIO3: rms_on>{} rms_off<{} f50_on>{} f50_off<{} zc={}..{} window={} sample_us={} poll_ms={}",
        cfg.broadband_rms_on_threshold,
        cfg.broadband_rms_off_threshold,
        cfg.tone50_rms_on_threshold,
        cfg.tone50_rms_off_threshold,
        cfg.zc_mains_min,
        cfg.zc_mains_max,
        cfg.window_samples,
        cfg.sample_interval_us,
        cfg.poll_interval_ms
    );

    pump_state_tx.send(false);
    let mut pump_on = false;
    let mut window_count: u32 = 0;
    const DIAG_EVERY: u32 = 15;

    let mut samples = [0u16; 200];
    let n = cfg.window_samples.min(samples.len());

    loop {
        // Discard first ADC read after pause — ESP32 S&H has stale charge
        let _: u16 = adc1.read_blocking(&mut adc_pin);
        Timer::after(Duration::from_micros(cfg.sample_interval_us as u64)).await;

        // Collect samples
        let mut sum: u32 = 0;
        let mut min_raw: u16 = u16::MAX;
        let mut max_raw: u16 = 0;
        let mut clip_hi: u16 = 0;
        let mut clip_lo: u16 = 0;

        for i in 0..n {
            let raw: u16 = adc1.read_blocking(&mut adc_pin);
            samples[i] = raw;
            sum += raw as u32;
            if raw >= ADC_MAX_CODE {
                clip_hi = clip_hi.saturating_add(1);
            }
            if raw <= ADC_MIN_CODE {
                clip_lo = clip_lo.saturating_add(1);
            }
            if raw < min_raw {
                min_raw = raw;
            }
            if raw > max_raw {
                max_raw = raw;
            }
            Timer::after(Duration::from_micros(cfg.sample_interval_us as u64)).await;
        }

        // Compute DC offset (mean) and AC RMS
        let dc = (sum / n as u32) as u16;

        let mut sq_sum: u64 = 0;
        for i in 0..n {
            let dev = samples[i] as i32 - dc as i32;
            sq_sum += (dev * dev) as u64;
        }
        let rms = sqrtf((sq_sum / n as u64) as f32) as u16;
        let pp = max_raw.saturating_sub(min_raw);
        let mean = sum as f32 / n as f32;
        let tone50_rms = tone_rms(&samples[..n], mean, fs_hz, 50.0);
        let tone400_rms = tone_rms(&samples[..n], mean, fs_hz, 400.0);
        let zero_crossings = count_zero_crossings(&samples[..n], dc);
        let mains_like = (cfg.zc_mains_min..=cfg.zc_mains_max).contains(&zero_crossings);
        let strong_broadband = rms >= cfg.broadband_rms_on_threshold;
        let weak_broadband = rms <= cfg.broadband_rms_off_threshold;
        let strong_tone = mains_like && tone50_rms >= cfg.tone50_rms_on_threshold;
        let weak_tone = tone50_rms <= cfg.tone50_rms_off_threshold;

        // Hysteresis detection
        let prev = pump_on;
        if !pump_on && (strong_broadband || strong_tone) {
            pump_on = true;
        } else if pump_on && weak_broadband && weak_tone {
            pump_on = false;
        }

        if pump_on != prev {
            let t = Instant::now().as_millis();
            if pump_on {
                info!(
                    "[{:08} ms] PumpMonitor: ON  (f50_rms={} zc={} pp={})",
                    t, tone50_rms, zero_crossings, pp
                );
            } else {
                info!(
                    "[{:08} ms] PumpMonitor: OFF (f50_rms={} pp={})",
                    t, tone50_rms, pp
                );
            }
            pump_state_tx.send(pump_on);
        }

        if clip_hi > 0 || clip_lo > 0 {
            warn!(
                "PumpMonitor: ADC clipping hi={} lo={} | reduce burden or shift bias lower",
                clip_hi,
                clip_lo
            );
        }

        window_count = window_count.wrapping_add(1);
        if window_count % DIAG_EVERY == 0 {
            log::info!(
                "PumpMonitor: rms={} f50_rms={} f400_rms={} dc={} pp={} zc={} min={} max={} clip_hi={} clip_lo={} pump={}",
                rms,
                tone50_rms,
                tone400_rms,
                dc,
                pp,
                zero_crossings,
                min_raw,
                max_raw,
                clip_hi,
                clip_lo,
                pump_on
            );
        }

        Timer::after(Duration::from_millis(cfg.poll_interval_ms)).await;
    }
}
