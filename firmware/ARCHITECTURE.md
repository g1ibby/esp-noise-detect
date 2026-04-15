# Firmware Architecture

ESP32-S3 firmware with two operating modes selectable via Cargo features.

## Features

| Feature | Description |
|---------|-------------|
| `streaming` | Audio streaming to server (continuous or pump-gated via ADC) |
| `inference` | NN-based pump detection (dual-core, outputs pump ON/OFF) |
| `mqtt` | MQTT output for inference (requires `inference`) |

Build with exactly one of `streaming` or `inference` enabled.

## Directory Structure

```
src/
├── lib.rs                 # Crate root, feature-gated module exports
├── logging.rs             # Log initialization (shared)
├── util.rs                # Allocation helpers (shared)
│
├── domain/                # Data types (shared)
│   └── mod.rs             # AudioData, NetOut, CtrlMsg, SampleFormat
│
├── drivers/               # Hardware drivers (shared)
│   └── gpio.rs            # XiaoLed
│
├── infra/                 # Infrastructure layer
│   ├── i2s_capture/       # I2S DMA audio capture (shared)
│   ├── net_stack.rs       # WiFi connection manager (shared)
│   ├── instrumentation.rs # Metrics counters (shared)
│   │
│   │ # streaming feature only:
│   ├── tcp_client.rs      # Audio streaming TCP client
│   ├── recording_orchestrator.rs
│   ├── pump_monitor.rs    # ADC-based pump detection
│   ├── led_xiao.rs        # LED status task
│   │
│   │ # inference feature only:
│   ├── pump_classifier.rs # Mel spectrogram + NN inference
│   ├── tcp_client_pump.rs # Pump state TCP client (default)
│   └── mqtt_client_pump.rs # Pump state MQTT client (with mqtt feature)
│
├── app/                   # streaming feature application
│   ├── mod.rs             # Entry point, single-core setup
│   └── config.rs          # Streaming config, beamformer constants
│
├── pump_app/              # inference feature application
│   ├── mod.rs             # Entry point, dual-core setup
│   └── config.rs          # Inference config, MQTT fields
│
└── features/              # streaming feature extras
    └── ...
```

## Architecture Comparison

### Streaming Mode (`--features streaming`)

```
┌─────────────────────────────────────────────────────────────┐
│                        Core 0 (single-core)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────┐  │
│  │ I2S Capture  │───>│ NetOut Channel  │───>│TCP Client │  │
│  │ (DMA audio)  │    │ (audio chunks)  │    │(streaming)│  │
│  └──────────────┘    └─────────────────┘    └───────────┘  │
│         │                                         │         │
│         v                                         v         │
│  ┌──────────────┐                         ┌───────────┐    │
│  │  Beamformer  │                         │   WiFi    │    │
│  │  (optional)  │                         │  Stack    │    │
│  └──────────────┘                         └───────────┘    │
│                                                             │
│  ┌──────────────────┐    ┌──────────────┐                  │
│  │ Recording Orch.  │<───│ Pump Monitor │ (ADC, optional)  │
│  │ (state machine)  │    │ (CT clamp)   │                  │
│  └──────────────────┘    └──────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Output: Raw audio stream over TCP (wire-protocol)
Modes: Continuous or PumpGated (via MODE env)
```

### Inference Mode (`--features inference`)

```
┌─────────────────────────────────┐ ┌─────────────────────────┐
│            Core 1               │ │         Core 0          │
├─────────────────────────────────┤ ├─────────────────────────┤
│                                 │ │                         │
│  ┌──────────────┐               │ │  ┌───────────────────┐  │
│  │ I2S Capture  │               │ │  │  TCP/MQTT Client  │  │
│  │ (DMA audio)  │               │ │  │  (pump state)     │  │
│  └──────┬───────┘               │ │  └─────────┬─────────┘  │
│         │                       │ │            │            │
│         v                       │ │            v            │
│  ┌──────────────┐               │ │  ┌───────────────────┐  │
│  │ NetOut Chan  │               │ │  │    WiFi Stack     │  │
│  │ (audio)      │               │ │  │                   │  │
│  └──────┬───────┘               │ │  └───────────────────┘  │
│         │                       │ │            ^            │
│         v                       │ │            │            │
│  ┌──────────────┐  PumpStatus   │ │            │            │
│  │   Pump       │───Channel─────┼─┼────────────┘            │
│  │  Classifier  │  (cross-core) │ │                         │
│  │  (NN model)  │               │ │                         │
│  └──────────────┘               │ │                         │
│         │                       │ │                         │
│    ┌────┴────┐                  │ │                         │
│    │Mel Spec │                  │ │                         │
│    │Quantize │                  │ │                         │
│    │Predict  │                  │ │                         │
│    └─────────┘                  │ │                         │
│                                 │ │                         │
└─────────────────────────────────┘ └─────────────────────────┘

Output: Pump state (ON/OFF) via TCP or MQTT
```

## Shared Components

| Component | Purpose |
|-----------|---------|
| `i2s_capture` | DMA-based I2S audio capture with beamforming |
| `net_stack` | WiFi connection manager with reconnection |
| `domain` | `AudioData` (Sample16/Sample24), `NetOut`, `SampleFormat` |
| `drivers` | `XiaoLed` GPIO wrapper |
| `util` | `mk_static!`, PSRAM/internal heap allocators |
| `logging` | ESP log initialization |

## Feature-Specific Components

### Streaming Only

| Component | Purpose |
|-----------|---------|
| `app` | Application entry, single-core orchestration |
| `tcp_client` | Full audio streaming with wire-protocol |
| `recording_orchestrator` | Recording state machine (start/stop/flush) |
| `pump_monitor` | ADC-based pump detection for gated recording |
| `led_xiao` | LED status indication task |

### Inference Only

| Component | Purpose |
|-----------|---------|
| `pump_app` | Application entry, dual-core setup |
| `pump_classifier` | Mel spectrogram + NN inference (~98ms/window) |
| `tcp_client_pump` | Simple pump state TCP transmission |
| `mqtt_client_pump` | MQTT with Home Assistant auto-discovery |

## Configuration

### Environment Variables (`.env`)

**Common:**
```env
SSID=wifi_network
PASSWORD=wifi_password
SERVER_IP=192.168.1.100
SERVER_PORT=3000
DEVICE_ID=pump-sensor-1
```

**Streaming only:**
```env
MODE=PumpGated  # or Continuous
```

**Inference + MQTT:**
```env
MQTT_BROKER_IP=192.168.1.100
MQTT_BROKER_PORT=1883
MQTT_TOPIC=pump/status
MQTT_CLIENT_ID=pump-sensor
MQTT_USERNAME=user
MQTT_PASSWORD=pass
```

## Key Design Decisions

1. **Dual-core for inference**: NN inference (~35ms) would block I2S DMA pops on single-core, causing audio loss. Core 1 handles I2S + inference, Core 0 handles WiFi.

2. **Single-core for streaming**: No heavy computation, simpler architecture suffices.

3. **CriticalSectionRawMutex for cross-core**: `PumpStatusChannel` uses `CriticalSectionRawMutex` because it crosses cores. `NetOutChannel` uses `NoopRawMutex` (stays on one core).

4. **Feature gates for memory**: ESP32-S3 has limited SRAM. Unused code is excluded at compile time.

5. **PSRAM for audio buffers**: Audio data allocated in PSRAM to preserve internal SRAM for WiFi and stacks.
