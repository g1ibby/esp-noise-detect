use embassy_sync::{blocking_mutex::raw::NoopRawMutex, channel::Channel};

use crate::domain::{AudioData, NetOut, SampleFormat};

pub const SAMPLE_RATE: u32 = 32_000;
pub const CHANNELS: u16 = 1;
pub const BITS_PER_SAMPLE: u16 = 24; // Set to 24 for 24-bit mode

// Depth of the inter-task audio channel (number of queued chunks).
// Keep small to avoid building minutes of backlog under network stalls.
pub const AUDIO_CHANNEL_DEPTH: usize = 64;
pub const NETOUT_CHANNEL_DEPTH: usize = 64; // may be tuned in pump-gated mode

// Streaming chunk size (samples per message). Keep independent from channel depth.
pub const STREAMING_CHUNK_SAMPLES: usize = 1024;

// Beamformer tuning (used when CHANNELS == 1 and BITS_PER_SAMPLE == 24)
// - With mic spacing ~3-5 cm at fs=32 kHz, max delay ~ 3 samples
// - Start with 0 and test -3...+3 to steer
pub const BEAMFORM_RING_SAMPLES: usize = 32;
pub const BEAMFORM_DELAY_SAMPLES: isize = 0; // R relative to L; +d delays R, -d advances R
pub const BEAMFORM_HPF_FC_HZ: f32 = 80.0; // rumble/DC removal
pub const BEAMFORM_LPF_ENABLED: bool = true;
pub const BEAMFORM_LPF_FC_HZ: f32 = 12_000.0; // band-limit for SNR
pub const BEAMFORM_GAIN: f32 = 8.0; // fixed digital gain (x8)

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
    // MQTT configuration (used when mqtt feature is enabled)
    #[cfg(feature = "mqtt")]
    pub mqtt_broker_ip: [u8; 4],
    #[cfg(feature = "mqtt")]
    pub mqtt_broker_port: u16,
    #[cfg(feature = "mqtt")]
    pub mqtt_topic: &'static str,
    #[cfg(feature = "mqtt")]
    pub mqtt_client_id: &'static str,
    #[cfg(feature = "mqtt")]
    pub mqtt_username: Option<&'static str>,
    #[cfg(feature = "mqtt")]
    pub mqtt_password: Option<&'static str>,
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

        // MQTT configuration (only parsed when mqtt feature is enabled)
        #[cfg(feature = "mqtt")]
        let mqtt_broker_ip = {
            let ip_str = option_env!("MQTT_BROKER_IP").unwrap_or(server_ip_str);
            parse_ip(ip_str).unwrap_or_else(|| {
                panic!("Invalid MQTT_BROKER_IP format: '{ip_str}'. Expected 'a.b.c.d'")
            })
        };
        #[cfg(feature = "mqtt")]
        let mqtt_broker_port = {
            let port_str = option_env!("MQTT_BROKER_PORT").unwrap_or("1883");
            wire_protocol::parse_u16(port_str).unwrap_or_else(|| {
                panic!("Invalid MQTT_BROKER_PORT: '{port_str}'. Expected 1..65535")
            })
        };
        #[cfg(feature = "mqtt")]
        let mqtt_topic: &'static str = option_env!("MQTT_TOPIC").unwrap_or("pump/status");
        #[cfg(feature = "mqtt")]
        let mqtt_client_id: &'static str = option_env!("MQTT_CLIENT_ID").unwrap_or(device_id);
        #[cfg(feature = "mqtt")]
        let mqtt_username: Option<&'static str> = option_env!("MQTT_USERNAME");
        #[cfg(feature = "mqtt")]
        let mqtt_password: Option<&'static str> = option_env!("MQTT_PASSWORD");

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
            chunk_samples: STREAMING_CHUNK_SAMPLES,
            #[cfg(feature = "mqtt")]
            mqtt_broker_ip,
            #[cfg(feature = "mqtt")]
            mqtt_broker_port,
            #[cfg(feature = "mqtt")]
            mqtt_topic,
            #[cfg(feature = "mqtt")]
            mqtt_client_id,
            #[cfg(feature = "mqtt")]
            mqtt_username,
            #[cfg(feature = "mqtt")]
            mqtt_password,
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
