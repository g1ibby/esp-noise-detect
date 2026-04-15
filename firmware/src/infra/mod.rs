// Placeholder for ESP32-specific adapters. To be implemented incrementally.

// Shared modules
pub mod instrumentation;
#[cfg(any(feature = "streaming", feature = "inference"))]
pub mod net_stack;
#[cfg(any(feature = "streaming", feature = "inference"))]
pub mod i2s_capture;

// Streaming feature modules
#[cfg(feature = "streaming")]
pub mod led_xiao;
#[cfg(feature = "streaming")]
pub mod pump_monitor;
#[cfg(feature = "streaming")]
pub mod recording_orchestrator;
#[cfg(feature = "streaming")]
pub mod tcp_client;

// Inference feature modules
#[cfg(feature = "inference")]
pub mod pump_classifier;
#[cfg(all(feature = "inference", not(feature = "mqtt")))]
pub mod tcp_client_pump;
#[cfg(all(feature = "inference", feature = "mqtt"))]
pub mod mqtt_client_pump;
