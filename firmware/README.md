# Firmware

ESP32-S3 firmware for audio capture and pump noise classification.

## Features

The firmware supports two main operating modes, selectable via Cargo features:

| Feature | Description |
|---------|-------------|
| `streaming` | Audio streaming to server (continuous or pump-gated via ADC sensor) |
| `inference` | NN-based pump detection (dual-core, outputs pump ON/OFF state) |
| `mqtt` | MQTT output for inference feature (Home Assistant compatible) |

## Prerequisites

Install the ESP Rust toolchain:

```bash
rustup toolchain install esp
```

## Configuration

Create a `.env` file in this directory:

```bash
cp .env.example .env  # if example exists, or create manually
```

Required variables:

```env
SSID=your_wifi_network
PASSWORD=your_wifi_password
SERVER_IP=192.168.1.100
SERVER_PORT=3000
```

Optional variables:

```env
MODE=PumpGated            # PumpGated (default) or Continuous
DEVICE_ID=pump-sensor-1   # Device identifier
DEVICE_VERSION=1.0.0      # Firmware version
LOG_LEVEL=info            # error|warn|info|debug|trace
```

MQTT configuration (when using `mqtt` feature):

```env
MQTT_BROKER_IP=192.168.1.100  # Defaults to SERVER_IP if not set
MQTT_BROKER_PORT=1883         # Defaults to 1883
MQTT_TOPIC=pump/status        # MQTT topic for pump state
MQTT_CLIENT_ID=pump-sensor-1  # Defaults to DEVICE_ID
MQTT_USERNAME=user            # Optional
MQTT_PASSWORD=pass            # Optional
```

## Build and Flash

Source the ESP environment and run with the appropriate feature.

### Binaries

Production firmware entry points are in `src/bin/`:

#### dataset_collector (requires `streaming` feature)

Streams audio over WiFi to host server for dataset collection:

```bash
source ~/export-esp.sh && cargo run --release --features streaming --bin dataset_collector
```

#### pump_monitor (requires `inference` feature)

Runs pump classification and sends state events:

```bash
# TCP mode
source ~/export-esp.sh && cargo run --release --features inference --bin pump_monitor

# MQTT mode (with Home Assistant auto-discovery)
source ~/export-esp.sh && cargo run --release --features inference,mqtt --bin pump_monitor
```

### Examples

Diagnostic and experimental tools are in `examples/`:

#### ct_monitor

Current transformer monitoring (standalone diagnostic tool, no features required):

```bash
source ~/export-esp.sh && cargo run --release --example ct_monitor
```

## Architecture

### Streaming Mode (dataset_collector)

Single-core operation:
- I2S audio capture -> TCP streaming to server
- Supports continuous or pump-gated recording (via ADC pump sensor)

### Inference Mode (pump_monitor)

Dual-core operation for optimal performance:
- **Core 0**: WiFi networking, pump state transmission (TCP or MQTT)
- **Core 1**: I2S audio capture, mel spectrogram computation, NN inference

## Notes

- Target board: XIAO ESP32-S3 with octal PSRAM
- Audio: 32kHz, 24-bit I2S input, mono
- The `espflash` tool handles flashing and serial monitoring automatically
