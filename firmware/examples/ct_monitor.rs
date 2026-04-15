// CT Clamp Current Monitor
//
// Detects whether an AC load (e.g. a pump) is ON or OFF by measuring current
// with a CT (current transformer) clamp connected to the ESP32's ADC.
//
// ## How it works
//
// 1. The CT clamp wraps around a single AC wire. When current flows, the clamp outputs a small AC
//    signal proportional to the current.
//
// 2. A resistor divider biases the ADC pin near mid-supply, so the AC signal from the CT rides on
//    top of this steady DC level. The exact raw ADC code for that bias depends on the pin,
//    attenuation, and chip calibration, so use the observed `dc=` logs rather than assuming 2048.
//
// 3. Every 250ms we take a "window" of 200 ADC samples (~80ms, covering ~4 full cycles of 50Hz
//    mains). From these samples we compute:
//
//      dc  = average of all samples (the steady bias level)
//      rms = "root mean square" of the AC part — how much the samples wobble
//            around the dc level. This is the main signal strength indicator.
//      min = lowest sample in the window
//      max = highest sample in the window
//      pp  = peak-to-peak = max - min (total swing)
//
// 4. When load is OFF: broadband rms is tiny and zero-crossings are very high because the ADC is
//    mostly seeing noise. When load is ON: a strong 50Hz tone appears and zero-crossings collapse
//    to roughly 8 crossings per 80ms window. This example therefore uses normalized 50Hz tone RMS
//    as the primary detector and keeps broadband RMS as a secondary diagnostic.
//
//
// 3V3 Rail (from ESP32)
//                       |
//                     [FB1] Ferrite Bead
//                       |
//                  3V3 Analog
//                       |
//                +------+------+
//                |      |      |
//              [C6]   [C7]    |
//              10µF   100nF   |
//                |      |      |
//                +------+------+
//                       |
//                       GND

//                  3V3 Analog
//                       |
//                     [R2] 10kΩ
//                       |
//                       +---------- VBIAS --------+
//                       |                          |
//                     [R3] 10kΩ                 [C2] 1µF ceramic
//                       |                          |
//                      GND                        GND

//                    CT Clamp (80A, 3000:1)
//                       |           |
//                      CT+         CT-
//                       |           |
//                       +--[R1]----+
//                           150Ω
//                       |           |
//                       |           +------------ VBIAS
//                       |
//                    ADC_RAW
//                       |
//                     [R4] 1kΩ
//                       |
//                       +------ NODE_A
//                       |
//                     [C4] 100nF ceramic
//                       |
//                      GND

//                    NODE_A
//                       |
//                     [R5] 1kΩ
//                       |
//                       +------ GPIO3 / A2 (ESP32-S3 ADC)
//                       |
//                     [C5] 100nF ceramic
//                       |
//                      GND
// The 10k/10k divider creates a 1.65V reference. The 47uF cap blocks DC
// (the CT secondary winding has low DC resistance that would collapse the
// bias) while passing 50Hz easily (impedance ~68 ohm at 50Hz). The 150 ohm
// burden resistor converts the CT's secondary current into a voltage.
// The 100nF cap to GND filters RF/WiFi noise.
//
// ## CT clamp sizing
//
// Signal strength depends on the CT ratio and load current. Example with
// 80A/3000:1 CT and 150 ohm burden at 220V mains:
//
//   300W (1.36A) -> ~60 counts RMS  (easily detected)
//   100W (0.45A) -> ~20 counts RMS  (marginal)
//     8W (36mA)  -> ~1 count RMS    (undetectable)

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    analog::adc::{Adc, AdcConfig, Attenuation},
    delay::Delay,
    gpio::Io,
    main,
};
use libm::{cosf, sinf, sqrtf};
use log::{info, warn};
use firmware::XiaoLed;

esp_bootloader_esp_idf::esp_app_desc!();

// Sampling: 200 samples at 400us intervals = ~80ms window (~4 cycles at 50Hz)
const SAMPLES_PER_WINDOW: usize = 200;
const SAMPLE_INTERVAL_US: u32 = 400;
const WINDOW_PAUSE_MS: u32 = 250;
const DIAG_EVERY_WINDOWS: u32 = 10;
const RAW_PREVIEW_SAMPLES: usize = 24;

// Detection thresholds for the working A2 / GPIO3 path on XIAO ESP32-S3.
const TONE50_RMS_ON_THRESHOLD: u16 = 80;
const TONE50_RMS_OFF_THRESHOLD: u16 = 30;
const ZC_MAINS_MIN: u16 = 6;
const ZC_MAINS_MAX: u16 = 12;

const FS_HZ: f32 = 1_000_000.0 / SAMPLE_INTERVAL_US as f32;
const TARGET_F50_HZ: f32 = 50.0;
const TARGET_F400_HZ: f32 = 400.0;
const ADC_MAX_CODE: u16 = 4095;
const ADC_MIN_CODE: u16 = 0;

fn tone_rms(samples: &[u16], mean: f32, target_hz: f32) -> u16 {
    let n = samples.len() as f32;
    let w = 2.0 * core::f32::consts::PI * target_hz / FS_HZ;
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

#[main]
fn main() -> ! {
    firmware::logging::init();

    info!("CT Clamp Monitor");
    info!(
        "Samples: {}, interval: {}us, window: ~{}ms",
        SAMPLES_PER_WINDOW,
        SAMPLE_INTERVAL_US,
        (SAMPLES_PER_WINDOW as u32 * SAMPLE_INTERVAL_US) / 1000
    );
    info!(
        "50Hz thresholds: ON>={} OFF<={} counts RMS | mains zc={}..{}",
        TONE50_RMS_ON_THRESHOLD,
        TONE50_RMS_OFF_THRESHOLD,
        ZC_MAINS_MIN,
        ZC_MAINS_MAX
    );
    info!("ADC input: A2 / GPIO3");

    let peripherals = esp_hal::init(esp_hal::Config::default());
    let _io = Io::new(peripherals.IO_MUX);

    let mut led = XiaoLed::new(peripherals.GPIO21);

    let mut adc1_cfg = AdcConfig::new();
    let mut adc_pin = adc1_cfg.enable_pin(peripherals.GPIO3, Attenuation::_11dB);
    let mut adc1 = Adc::new(peripherals.ADC1, adc1_cfg);

    let delay = Delay::new();
    let mut load_on = false;
    let mut window_num: u32 = 0;
    let mut samples = [0u16; SAMPLES_PER_WINDOW];

    loop {
        // Discard first read — ESP32 sample-and-hold has stale charge after pause
        let _: u16 = adc1.read_blocking(&mut adc_pin);
        delay.delay_micros(SAMPLE_INTERVAL_US);

        // Collect samples
        let mut sum: u32 = 0;
        let mut min_raw: u16 = u16::MAX;
        let mut max_raw: u16 = 0;
        let mut clip_hi: u16 = 0;
        let mut clip_lo: u16 = 0;

        for i in 0..SAMPLES_PER_WINDOW {
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
            delay.delay_micros(SAMPLE_INTERVAL_US);
        }

        // Compute DC offset (average) and AC RMS (wobble strength)
        let dc = (sum / SAMPLES_PER_WINDOW as u32) as u16;

        let mut sq_sum: u64 = 0;
        for i in 0..SAMPLES_PER_WINDOW {
            let dev = samples[i] as i32 - dc as i32;
            sq_sum += (dev * dev) as u64;
        }
        let rms = sqrtf((sq_sum / SAMPLES_PER_WINDOW as u64) as f32) as u16;
        let pp = max_raw.saturating_sub(min_raw);
        let mean = sum as f32 / SAMPLES_PER_WINDOW as f32;
        let tone50_rms = tone_rms(&samples, mean, TARGET_F50_HZ);
        let tone400_rms = tone_rms(&samples, mean, TARGET_F400_HZ);
        let zero_crossings = count_zero_crossings(&samples, dc);
        let mains_like = (ZC_MAINS_MIN..=ZC_MAINS_MAX).contains(&zero_crossings);

        // Hysteresis detection
        let prev = load_on;
        if !load_on && mains_like && tone50_rms > TONE50_RMS_ON_THRESHOLD {
            load_on = true;
        } else if load_on && tone50_rms < TONE50_RMS_OFF_THRESHOLD {
            load_on = false;
        }

        if load_on {
            led.on();
        } else {
            led.off();
        }

        if load_on != prev {
            if load_on {
                info!(
                    ">>> LOAD ON  (f50_rms={} zc={} in-band)",
                    tone50_rms,
                    zero_crossings
                );
            } else {
                info!(
                    ">>> LOAD OFF (f50_rms={} below {})",
                    tone50_rms,
                    TONE50_RMS_OFF_THRESHOLD
                );
            }
        }

        info!(
            "win={} dc={} rms={} f50_rms={} f400_rms={} min={} max={} pp={} zc={} clip_hi={} clip_lo={} load={}",
            window_num,
            dc,
            rms,
            tone50_rms,
            tone400_rms,
            min_raw,
            max_raw,
            pp,
            zero_crossings,
            clip_hi,
            clip_lo,
            if load_on { "ON" } else { "OFF" }
        );

        if clip_hi > 0 || clip_lo > 0 {
            warn!(
                "ADC clipping detected: hi={} lo={} | reduce burden or shift bias lower if only the high side clips",
                clip_hi,
                clip_lo
            );
        }

        if window_num % DIAG_EVERY_WINDOWS == 0 {
            let preview_len = RAW_PREVIEW_SAMPLES.min(SAMPLES_PER_WINDOW);
            info!(
                "diag raw[0..{}]={:?}",
                preview_len,
                &samples[..preview_len]
            );
        }

        window_num = window_num.wrapping_add(1);
        delay.delay_millis(WINDOW_PAUSE_MS);
    }
}
