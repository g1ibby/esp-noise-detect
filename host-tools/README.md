# Host Tools

Desktop utilities for receiving data from ESP32-S3 audio classifier.

## Build

```bash
cargo build --release
```

## Binaries

### pump_monitor

TCP server that receives pump status events from ESP32-S3.

```bash
cargo run --release --bin pump_monitor
```

Options:
- `-p, --port <PORT>` - Port to listen on (default: 3000)
- `-v, --verbose` - Enable verbose logging

### wifi_audio_server

TCP server that receives audio streams and saves them as WAV files.

```bash
cargo run --release --bin wifi_audio_server
```

Options:
- `-p, --port <PORT>` - Port to listen on (default: 3000)
- `-o, --output-dir <DIR>` - Output directory for WAV files (default: `./recordings`)
- `-v, --verbose` - Enable verbose logging

## Examples

```bash
# Run pump_monitor on default port
cargo run --release --bin pump_monitor

# Run wifi_audio_server on port 4000, save to custom directory
cargo run --release --bin wifi_audio_server -- -p 4000 -o ~/audio

# Get help
cargo run --release --bin pump_monitor -- --help
cargo run --release --bin wifi_audio_server -- --help
```

Both servers listen on all interfaces (`[::]`), allowing ESP32 devices to connect from the network.
