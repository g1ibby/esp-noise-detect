use embassy_sync::{blocking_mutex::raw::NoopRawMutex, channel::Channel};

use crate::domain::{AudioData, NetOut, SampleFormat};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OperationMode {
    Continuous,
    PumpGated,
}

#[cfg(mode_continuous)]
pub const MODE: OperationMode = OperationMode::Continuous;
#[cfg(mode_pump_gated)]
pub const MODE: OperationMode = OperationMode::PumpGated;
#[cfg(not(any(mode_continuous, mode_pump_gated)))]
compile_error!("MODE cfg not set. Ensure build.rs emits mode_* flag.");

pub const SAMPLE_RATE: u32 = 32_000;
// Set to 1 for mono (beamformed) output, 2 for stereo passthrough
pub const CHANNELS: u16 = 1;
pub const BITS_PER_SAMPLE: u16 = 24; // Set to 24 for 24-bit mode

// Depth of the inter-task audio channel (number of queued chunks).
// Keep small to avoid building minutes of backlog under network stalls.
pub const AUDIO_CHANNEL_DEPTH: usize = 16;
pub const NETOUT_CHANNEL_DEPTH: usize = 16;

// Streaming chunk size (samples per message). Keep independent from channel depth.
pub const STREAMING_CHUNK_SAMPLES: usize = 1024;

// Audio channel type alias (bounded queue between recorder and network sender)
pub type AudioDataChannel = Channel<NoopRawMutex, AudioData, AUDIO_CHANNEL_DEPTH>;
pub type NetOutChannel = Channel<NoopRawMutex, NetOut, NETOUT_CHANNEL_DEPTH>;

/// Determine sample format at compile time based on BITS_PER_SAMPLE constant
pub const fn get_sample_format() -> SampleFormat {
    match BITS_PER_SAMPLE {
        16 => SampleFormat::Sample16,
        24 => SampleFormat::Sample24,
        _ => panic!("Unsupported BITS_PER_SAMPLE value. Use 16 or 24."),
    }
}

/// Check if we're using 16-bit samples at compile time
pub const fn is_16bit_mode() -> bool {
    BITS_PER_SAMPLE == 16
}

/// Check if we're using 24-bit samples at compile time
pub const fn is_24bit_mode() -> bool {
    BITS_PER_SAMPLE == 24
}

// Beamformer tuning (used when CHANNELS == 1 and BITS_PER_SAMPLE == 24)
// - With mic spacing ≈3–5 cm at fs=32 kHz, max delay ≈ 3 samples
// - Start with 0 and test -3…+3 to steer
pub const BEAMFORM_RING_SAMPLES: usize = 32;
pub const BEAMFORM_DELAY_SAMPLES: isize = 0; // R relative to L; +d delays R, -d advances R
pub const BEAMFORM_HPF_FC_HZ: f32 = 80.0; // rumble/DC removal
pub const BEAMFORM_LPF_ENABLED: bool = true;
pub const BEAMFORM_LPF_FC_HZ: f32 = 12_000.0; // band-limit for SNR
pub const BEAMFORM_GAIN: f32 = 8.0; // fixed digital gain (x8)

// Pump-gated configuration (compile-time knobs)
// Thresholds are in raw ADC counts RMS (12-bit, 0-4095) for the normalized
// 50Hz tone detector used on the A2 / GPIO3 CT input.
#[derive(Copy, Clone, Debug)]
pub struct PumpConfig {
    pub broadband_rms_on_threshold: u16,
    pub broadband_rms_off_threshold: u16,
    pub tone50_rms_on_threshold: u16,
    pub tone50_rms_off_threshold: u16,
    pub zc_mains_min: u16,
    pub zc_mains_max: u16,
    pub window_samples: usize,
    pub sample_interval_us: u32,
    pub poll_interval_ms: u64,
}

pub const PUMP_BROADBAND_RMS_ON_THRESHOLD: u16 = 80; // counts RMS on A2 / GPIO3
pub const PUMP_BROADBAND_RMS_OFF_THRESHOLD: u16 = 20; // counts RMS on A2 / GPIO3
pub const PUMP_TONE50_RMS_ON_THRESHOLD: u16 = 80; // counts RMS on A2 / GPIO3
pub const PUMP_TONE50_RMS_OFF_THRESHOLD: u16 = 30; // counts RMS on A2 / GPIO3
pub const PUMP_ZC_MAINS_MIN: u16 = 6; // ~50Hz sine over an 80ms window
pub const PUMP_ZC_MAINS_MAX: u16 = 12; // allow mild distortion / noise
pub const PUMP_WINDOW_SAMPLES: usize = 200; // ~80ms window (~4 cycles at 50Hz)
pub const PUMP_SAMPLE_INTERVAL_US: u32 = 400; // 400us between samples
pub const PUMP_POLL_INTERVAL_MS: u64 = 250; // 250ms between windows

pub const fn pump_config() -> PumpConfig {
    PumpConfig {
        broadband_rms_on_threshold: PUMP_BROADBAND_RMS_ON_THRESHOLD,
        broadband_rms_off_threshold: PUMP_BROADBAND_RMS_OFF_THRESHOLD,
        tone50_rms_on_threshold: PUMP_TONE50_RMS_ON_THRESHOLD,
        tone50_rms_off_threshold: PUMP_TONE50_RMS_OFF_THRESHOLD,
        zc_mains_min: PUMP_ZC_MAINS_MIN,
        zc_mains_max: PUMP_ZC_MAINS_MAX,
        window_samples: PUMP_WINDOW_SAMPLES,
        sample_interval_us: PUMP_SAMPLE_INTERVAL_US,
        poll_interval_ms: PUMP_POLL_INTERVAL_MS,
    }
}

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub ssid: &'static str,
    pub password: &'static str,
    pub server_ip: [u8; 4],
    pub server_port: u16,
    pub device_id: &'static str,
    pub device_version: &'static str,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub chunk_samples: usize,
}

impl AppConfig {
    pub fn from_env() -> Self {
        // Mandatory (compile-time) env vars
        let ssid: &'static str = env!("SSID");
        let password: &'static str = env!("PASSWORD");
        let server_ip_str: &'static str = env!("SERVER_IP");
        let server_port_str: &'static str = env!("SERVER_PORT");

        // Optional env vars with defaults
        let device_id: &'static str = option_env!("DEVICE_ID").unwrap_or("xiao_esp32s3");
        let device_version: &'static str = option_env!("DEVICE_VERSION").unwrap_or("1.0");

        let server_ip = parse_ip(server_ip_str).unwrap_or_else(|| {
            panic!("Invalid SERVER_IP format: '{server_ip_str}'. Expected 'a.b.c.d'")
        });
        let server_port = wire_protocol::parse_u16(server_port_str).unwrap_or_else(|| {
            panic!("Invalid SERVER_PORT: '{server_port_str}'. Expected 1..65535")
        });

        Self {
            ssid,
            password,
            server_ip,
            server_port,
            device_id,
            device_version,
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
            bits_per_sample: BITS_PER_SAMPLE,
            // Smaller chunks lower per-write latency and smooth TCP/WiFi backpressure.
            // Use an explicit chunk size separate from channel depth.
            chunk_samples: STREAMING_CHUNK_SAMPLES,
        }
    }
}

fn parse_ip(s: &str) -> Option<[u8; 4]> {
    let mut out = [0u8; 4];
    let mut idx = 0;
    for part in s.split('.') {
        if idx >= 4 {
            return None;
        }
        out[idx] = wire_protocol::parse_u8(part)?;
        idx += 1;
    }
    if idx == 4 { Some(out) } else { None }
}
