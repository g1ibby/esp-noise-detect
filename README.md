# esp-noise-detect

On-device audio classifier for ESP32-S3 that detects whether an electric pump is running. The firmware captures audio over I2S, runs a small quantised neural network on-device, and reports pump state (ON/OFF) over TCP or MQTT.

Built and tested on the **XIAO ESP32-S3** (with octal PSRAM) and a stereo I2S microphone pair. Inference runs in ~31 ms per 1-second window.

---

## Repository layout

| Directory        | What it is                                                                 |
|------------------|----------------------------------------------------------------------------|
| `firmware/`      | Rust firmware for the ESP32-S3 (`no_std`, embassy async, dual-core).        |
| `host-tools/`    | Small Rust servers that run on a desktop / home server and receive data from the firmware. |
| `nn/`            | Python training pipeline (PyTorch + Lightning + Hydra) that produces the `.espdl` model the firmware embeds. |
| `wire-protocol/` | `no_std`-compatible Rust crate with the shared TCP message types used by firmware and host-tools. |
| `hardware/`      | PCB designs ([tscircuit](https://tscircuit.com)). Currently: `pump-ct-sensor` — CT-clamp signal-conditioning board for the optional pump-gated dataset collection. |

---

## What each component does

### `firmware/` — two modes of operation

The firmware has **two separate binaries**, selected at build time by a Cargo feature. You flash one or the other depending on what you want to do.

1. **`dataset_collector`** &nbsp;— &nbsp;`--features streaming`
   Streams raw audio over WiFi/TCP to the `wifi_audio_server` host tool so you can record a labelled dataset.
   - *Continuous* mode: always streams.
   - *Pump-gated* mode: streams only while a current-transformer sensor on GPIO3 sees pump current, and tags each segment on/off. Useful for auto-labelling.

2. **`pump_monitor`** &nbsp;— &nbsp;`--features inference` (optionally `+mqtt`)
   Runs the neural network on-device and reports pump state.
   - Core 1: I2S capture → mel-spectrogram → NN inference.
   - Core 0: WiFi + TCP (or MQTT, with Home Assistant auto-discovery).

See `firmware/README.md` and `firmware/ARCHITECTURE.md` for wiring, env vars, and internals.

### `host-tools/` — desktop-side receivers

Two small Tokio TCP servers that listen for connections from the firmware:

- **`wifi_audio_server`** — accepts an audio stream from `dataset_collector` and writes WAV files (rolled every 5 minutes) into an output directory. Used to build a training dataset.
- **`pump_monitor`** — accepts `PumpStatus` / `Keepalive` messages from the firmware's `pump_monitor` binary and logs pump state transitions. Does not persist anything; monitoring only.

See `host-tools/README.md`.

### `nn/` — training pipeline (not required at runtime)

Python project that trains the classifier and exports it to `.espdl` for on-device inference. You only need this if you want to retrain the model; the firmware ships with a pre-trained model in `firmware/models/`.

Pipeline: WAV files → filename-based labels → mel-spectrogram → TinyConv CNN → ONNX → `.espdl` (INT8, via [esp-ppq](https://github.com/espressif/esp-ppq) in Docker). See `nn/README.md` for the full training / evaluation / export walkthrough.

### `wire-protocol/` — shared message types

Zero-allocation, `no_std`-compatible frame format shared by firmware and host-tools: an 8-byte header (type, length, sequence) plus a small set of payloads (`Hello`, `Metadata`, `Audio`, `Segment`, `PumpStatus`, `Keepalive`).

---

## Quick start

### 1. Flash the firmware for inference

```bash
rustup toolchain install esp        # one-time

cd firmware
cp .env.example .env
# edit .env — set SSID, PASSWORD, SERVER_IP, SERVER_PORT
cargo run --release --bin pump_monitor --features inference
```

### 2. Run the host-side monitor

```bash
cd host-tools
cargo run --release --bin pump_monitor
```

You should see `ON` / `OFF` transitions logged whenever the firmware detects a state change.

### 3. (Optional) Collect your own dataset and retrain

```bash
# On the device:
cd firmware
cargo run --release --bin dataset_collector --features streaming

# On the host, in parallel:
cd host-tools
cargo run --release --bin wifi_audio_server -- -o ./recordings

# Then train and export a new model:
cd nn
uv venv .venv && source .venv/bin/activate
uv pip install -e .[dev,export]
uv run python scripts/build_manifest.py --data-root ./recordings
uv run -m noise_detect.train   dataset.manifest_path=./recordings/manifest.jsonl
uv run -m noise_detect.export  dataset.manifest_path=./recordings/manifest.jsonl \
                               checkpoint=runs/<timestamp>/checkpoints/best.ckpt
cp runs/<timestamp>/export/model.espdl ../firmware/models/
```

---

## Hardware / signal chain

- Board: XIAO ESP32-S3 with octal PSRAM.
- Mics: stereo I2S pair (BCLK/WS/DIN on GPIO7/6/5), optionally beamformed to mono on-device.
- Optional CT clamp on GPIO3 for pump-gated dataset collection.
- Audio: 32 kHz, 24-bit, mono.
- Model input: `[1, 64, 101, 1]` INT8 mel-spectrogram (64 mel bands, 101 frames ≈ 1 s).
- Model output: `[1, 2]` INT8 logits — `pump_off` / `pump_on`.

This is a small personal project: it has been validated on a single hardware setup and a single pump. Expect to retrain on your own recordings before it classifies your pump reliably.

---

## License

Licensed under the MIT License — see [LICENSE](LICENSE).
