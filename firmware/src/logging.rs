//! Lightweight logging initializer for firmware examples.
//!
//! Uses `esp_println` as the `log` backend and selects the level at
//! compile time based on `LOG_LEVEL` (one of: error, warn, info,
//! debug, trace). Defaults to `info` if unset or unknown.

#[inline]
pub fn init() {
    let level = match option_env!("LOG_LEVEL") {
        Some("error") => log::LevelFilter::Error,
        Some("warn") => log::LevelFilter::Warn,
        Some("info") | None => log::LevelFilter::Info,
        Some("debug") => log::LevelFilter::Debug,
        Some("trace") => log::LevelFilter::Trace,
        Some(_) => log::LevelFilter::Info,
    };
    esp_println::logger::init_logger(level);
}
