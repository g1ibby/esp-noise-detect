#![no_std]
#![no_main]

extern crate alloc;

#[cfg(feature = "defmt-rtt")]
use defmt_rtt as _;
use embassy_executor::Spawner;
use esp_backtrace as _;
use log::info;
use firmware::pump_app::AppConfig;

// ESP-IDF app descriptor for espflash compatibility
esp_bootloader_esp_idf::esp_app_desc!();

#[esp_rtos::main]
async fn main(spawner: Spawner) {
    firmware::logging::init();
    info!("XIAO ESP32-S3 Pump Monitor");
    info!("Credentials and server are provided via compile-time env");
    let cfg = AppConfig::from_env();
    firmware::pump_app::start(spawner, &cfg).await;
}
