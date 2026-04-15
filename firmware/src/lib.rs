#![no_std]
#![feature(asm_experimental_arch)]

extern crate alloc;

// Feature-gated application modules
#[cfg(feature = "streaming")]
pub mod app;
#[cfg(feature = "inference")]
pub mod pump_app;

// Shared modules
pub mod domain;
pub mod drivers;
#[cfg(feature = "streaming")]
pub mod features;
pub mod infra;
pub mod logging;
pub mod util;

// Re-export the main components for easy access
pub use drivers::gpio::XiaoLed;
