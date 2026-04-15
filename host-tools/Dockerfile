FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libudev-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY host-tools/ host-tools/
COPY wire-protocol/ wire-protocol/

RUN cargo build --manifest-path host-tools/Cargo.toml --bin wifi_audio_server --release

FROM debian:bookworm-slim

COPY --from=builder /src/host-tools/target/release/wifi_audio_server /usr/local/bin/wifi_audio_server

VOLUME /recordings
EXPOSE 3000

ENTRYPOINT ["wifi_audio_server"]
CMD ["--port", "3000", "--output-dir", "/recordings"]
