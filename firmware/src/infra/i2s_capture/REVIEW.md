Quick note: I didn’t find an i2c_capture module; assuming you meant firmware/src/infra/i2s_capture (audio). I reviewed that module and its usage in app/mod.rs and tcp_client.rs.

**Most Important**
- Avoid blocking the capture loop on channel send
  - Where: firmware/src/infra/i2s_capture/mod.rs in flush_full_chunks and send_final_partial_chunk.
  - Issue: Using `audio_data_channel.send(...).await` can block if the network task falls behind, causing DMA Late errors and recoveries that drop audio anyway.
  - Fix: Prefer `try_send` with a drop policy + rate-limited warning/metrics. Example: `match audio_data_channel.try_send(audio_data) { Ok(()) => {}, Err(_) => { dropped += 1; /* ratelimit log */ } }`.
  - Impact: Keeps DMA draining smooth and reduces Late churn/CPU overhead, trading rare chunk drops for stable capture.

- Fix metrics rate check (hard-coded 32000)
  - Where: firmware/src/infra/i2s_capture/metrics.rs, `maybe_emit_metrics`.
  - Issue: Status compares to a hard-coded `32000.0` rather than `sample_rate * channels`.
  - Fix: Compute `expected_rate = sample_rate as f32 * channels as f32; let status = if (actual_rate - expected_rate).abs() <= expected_rate * 0.03 { "OK" } else { "RATE_DRIFT" };`.
  - Impact: Accurate drift signal across any configured sample rate/channels.

- Remove extra copying when flushing chunks
  - Where: firmware/src/infra/i2s_capture/mod.rs, `flush_full_chunks`.
  - Issue: `drain(..).collect()` copies samples to a new Vec every time.
  - Fix: Use `split_off + core::mem::replace` to move the head chunk without copying. Pattern: `let tail = audio_chunk.split_off(cfg.chunk_samples); let head = core::mem::replace(audio_chunk, tail); audio_data_channel.try_send(AudioData::new(head)) ...`.
  - Impact: Less CPU/memory bandwidth, fewer allocations, smoother timing.

- Make Late error recovery consistent
  - Where: firmware/src/infra/i2s_capture/mod.rs (both available() and pop() Late branches) + drain.rs macro.
  - Issue: On Late during `pop()`, recovered bytes may be processed; during `available()` Late they’re always dropped. Mixed policies complicate timing and cause inconsistent skips.
  - Fix: Pick one policy and apply in both places. Recommended: always drop recovered bytes to quickly re-sync with DMA and keep timing consistent (or, if you want to salvage, process recovered bytes in both branches).
  - Impact: Predictable behavior and simpler reasoning about audio continuity.

**High Value**
- Assert 32-bit word alignment and optionally handle remainders
  - Where: firmware/src/infra/i2s_capture/mod.rs after `pop`.
  - Fix: `debug_assert_eq!(bytes_read % 4, 0, "unaligned I2S word count");` In release, either ignore or buffer the tiny remainder for the next iteration.
  - Impact: Catches surprising hardware/driver changes early; avoids silent sample loss.

- Improve idle-drain byte logging
  - Where: firmware/src/infra/i2s_capture/drain.rs.
  - Issue: `total_drained % interval_bytes == 0` rarely hits exactly.
  - Fix: Track `last_log_total` and log when `(total_drained - last_log_total) >= interval_bytes`, then set `last_log_total = total_drained`. Alternatively, use `RateLimiter` by bytes with a running delta counter.
  - Impact: More reliable visibility into drain progress.

- Make RecorderTuning externally configurable
  - Where: firmware/src/infra/i2s_capture/mod.rs `RecorderTuning`.
  - Fix: Thread a `RecorderTuning` instance from `app/mod.rs` (or cfg/env) so you can adjust drain/late thresholds per build or at runtime.
  - Impact: Faster iteration on field tuning without code edits.

- Clarify processor naming and extend options
  - Where: firmware/src/infra/i2s_capture/processors.rs.
  - Issue: `StereoHigh16Processor` name can be misleading; it pushes one 16-bit sample per 32-bit slot (L,R interleaved), not two at a time.
  - Fix: Rename to `InterleavedHigh16` or add processors explicitly: `MonoHigh16` (mix L/R), `InterleavedHigh16`, and (later) `InterleavedHigh24To16` with proper scaling/dither.
  - Impact: Clearer intent; easy path to full 24-bit fidelity later.

**Nice To Have**
- Parameterize I2S work buffer size
  - Where: firmware/src/infra/i2s_capture/mod.rs `I2S_DATA_BUFFER_LEN`.
  - Fix: Derive from DMA descriptor size or make it a tuning parameter; document the relation to `pop()` return sizes.
  - Impact: Easier to match hardware behavior and reduce memory footprint.

- Consider heapless buffer for audio_chunk
  - Where: firmware/src/infra/i2s_capture/mod.rs `audio_chunk`.
  - Fix: Optional compile-time variant using `heapless::Vec<i16, N>` sized to `streaming_chunk_samples * 2`, if you want to avoid PSRAM allocations and fragmentation. Keep current PSRAM-based Vec as default.
  - Impact: More deterministic memory at the cost of flexibility.

- Centralize stream-late counters in one helper
  - Where: firmware/src/infra/i2s_capture/metrics.rs + usages.
  - Fix: Consolidate Late counting/logging into a single small helper that both available() and pop() branches call. You already have `StreamLateLog`; extend usage to cover every Late path.
  - Impact: Less duplication and fewer logging edge cases.

- Tighten docs on chunk units and channel semantics
  - Where: firmware/src/infra/i2s_capture/mod.rs (module docs) + app config comments.
  - Fix: Explicitly say `streaming_chunk_samples` is count of i16 samples (interleaved across channels), not per-channel frames.
  - Impact: Avoids misconfigurations that harm throughput.

- Optional: retry DMA on startup failure
  - Where: firmware/src/infra/i2s_capture/mod.rs when `read_dma_circular_async` fails.
  - Fix: Retry with backoff and a limited attempt count; log clearly before giving up.
  - Impact: Slightly more robust boot on transient failures.

**Callouts By File**
- firmware/src/infra/i2s_capture/mod.rs
  - Replace blocking channel sends with `try_send` and add drop metrics.
  - Switch flush logic to move chunk head without copy.
  - Align Late handling policies across available()/pop().
  - Add `debug_assert_eq!(bytes_read % 4, 0)`.
  - Consider threading in `RecorderTuning`.

- firmware/src/infra/i2s_capture/metrics.rs
  - Fix rate status calculation to use `sample_rate * channels`.
  - Keep `StreamLateLog` as the single place for Late rate-limited logs.

- firmware/src/infra/i2s_capture/drain.rs
  - Improve drained-bytes logging thresholding.
  - Ensure Late recovery behavior matches the streaming path’s policy (drop vs salvage).

- firmware/src/infra/i2s_capture/processors.rs
  - Rename `StereoHigh16Processor` to something like `InterleavedHigh16`.
  - Add variants for mono mixdown and future 24-bit utilization.

If you want, I can implement the top 2–3 changes (non-blocking channel send, metrics status fix, and zero-copy flush) in a focused PR-sized patch.

