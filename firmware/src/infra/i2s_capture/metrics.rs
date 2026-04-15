use embassy_time::{Duration, Instant};
use log::{debug, warn};

pub const STREAM_LATE_INTERVAL_MS: u64 = 1200;
pub const METRICS_INTERVAL_MS: u64 = 2000;

#[inline]
pub fn maybe_log_stream_late(
    stream_late_count: &mut u32,
    stream_late_last: &mut Instant,
    stream_late_drained_bytes: &mut usize,
    total_samples_streamed: usize,
    streaming_start: Instant,
    context: &str,
) {
    if stream_late_last.elapsed() >= Duration::from_millis(STREAM_LATE_INTERVAL_MS) {
        warn!(
            "Late errors during streaming{}: {}x in last {}ms; streamed={} elapsed_ms={}",
            context,
            *stream_late_count,
            STREAM_LATE_INTERVAL_MS,
            total_samples_streamed,
            streaming_start.elapsed().as_millis()
        );
        if *stream_late_drained_bytes > 0 {
            debug!(
                "Late{} recovery drained {} bytes in last {}ms",
                context, *stream_late_drained_bytes, STREAM_LATE_INTERVAL_MS
            );
        }
        *stream_late_count = 0;
        *stream_late_drained_bytes = 0;
        *stream_late_last = Instant::now();
    }
}

#[inline]
pub fn maybe_emit_metrics(
    last_metrics: &mut Instant,
    total_samples_streamed: usize,
    streaming_start: Instant,
    sample_rate: u32,
    channels: u8,
    chunks_sent: usize,
) {
    if last_metrics.elapsed() >= Duration::from_millis(METRICS_INTERVAL_MS) {
        let streaming_duration = streaming_start.elapsed();
        let expected_duration =
            total_samples_streamed as f32 / (sample_rate as f32 * channels as f32);
        let actual_rate =
            total_samples_streamed as f32 / (streaming_duration.as_millis() as f32 / 1000.0);
        let status = if (actual_rate - 32000.0).abs() < 1000.0 {
            "OK"
        } else {
            "RATE_DRIFT"
        };
        debug!(
            "Recording: samples={} chunks={} elapsed_s={:.1} expected_s={:.1} rate_sps={:.0} status={}",
            total_samples_streamed,
            chunks_sent,
            streaming_duration.as_millis() as f32 / 1000.0,
            expected_duration,
            actual_rate,
            status
        );
        *last_metrics = Instant::now();
    }
}

/// Simple rate limiter for periodic logs.
#[allow(dead_code)]
pub struct RateLimiter {
    pub count: u32,
    pub last: Instant,
    pub interval: Duration,
}

/// Streaming-late counters + ratelimit wrapper.
pub struct StreamLateLog {
    pub count: u32,
    pub last: Instant,
    pub drained_bytes: usize,
    pub _interval: Duration,
}

impl StreamLateLog {
    pub fn new(interval: Duration) -> Self {
        Self {
            count: 0,
            last: Instant::now(),
            drained_bytes: 0,
            _interval: interval,
        }
    }

    #[inline]
    pub fn inc(&mut self) {
        self.count = self.count.saturating_add(1);
    }

    #[inline]
    pub fn add_drained(&mut self, n: usize) {
        self.drained_bytes = self.drained_bytes.saturating_add(n);
    }

    #[inline]
    pub fn maybe_log(
        &mut self,
        total_samples_streamed: usize,
        streaming_start: Instant,
        context: &str,
    ) {
        // Reuse the existing helper; it rate-limits and resets fields when logging.
        maybe_log_stream_late(
            &mut self.count,
            &mut self.last,
            &mut self.drained_bytes,
            total_samples_streamed,
            streaming_start,
            context,
        );
    }
}

#[allow(dead_code)]
impl RateLimiter {
    pub fn new(interval: Duration) -> Self {
        Self {
            count: 0,
            last: Instant::now(),
            interval,
        }
    }

    pub fn inc(&mut self) {
        self.count = self.count.saturating_add(1);
    }

    pub fn take_and_maybe<F: FnOnce(u32)>(&mut self, f: F) {
        if self.last.elapsed() >= self.interval {
            let c = self.count;
            self.count = 0;
            self.last = Instant::now();
            if c > 0 {
                f(c);
            }
        }
    }
}
