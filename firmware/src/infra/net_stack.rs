use core::sync::atomic::{AtomicBool, Ordering};
use embassy_net::Runner;
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, signal::Signal};
use embassy_time::{Duration, Timer};
use esp_radio::wifi::{
    Config, Interface, PowerSaveMode, WifiController,
    sta::StationConfig,
};
use log::{debug, error, info, warn};

const RECONNECT_DELAY_MS: u64 = 2000;
const DISCONNECTED_POLL_MS: u64 = 500;

/// WiFi connection manager task - only manages WiFi, not server connections
#[embassy_executor::task]
pub async fn connection_manager(
    mut controller: WifiController<'static>,
    connected_event: &'static Signal<NoopRawMutex, ()>,
    connected_state: &'static AtomicBool,
    ssid: &'static str,
    password: &'static str,
) {
    info!("Starting WiFi connection manager");

    // Initial WiFi setup - configure station mode with credentials
    let station_config = Config::Station(
        StationConfig::default()
            .with_ssid(ssid)
            .with_password(alloc::string::String::from(password)),
    );
    controller.set_config(&station_config).ok();
    info!("WiFi configured");
    // Maximize radio stability for continuous streaming
    if let Err(e) = controller.set_power_saving(PowerSaveMode::None) {
        warn!("Failed to set power saving mode: {:?}", e);
    } else {
        info!("WiFi power saving disabled (Performance mode)");
    }

    // Initial connection attempt
    info!("Connecting to WiFi network: {}", ssid);
    match controller.connect_async().await {
        Ok(_) => {
            info!("WiFi connected!");
            connected_state.store(true, Ordering::Relaxed);
            connected_event.signal(());
        }
        Err(e) => {
            error!("Initial WiFi connection failed: {:?}", e);
        }
    }

    // Monitor WiFi state and only reconnect on actual WiFi disconnection
    loop {
        if controller.is_connected() {
            let _ = controller.wait_for_disconnect_async().await;
            warn!("WiFi connection lost - attempting reconnection");
            connected_state.store(false, Ordering::Relaxed);
            Timer::after(Duration::from_millis(RECONNECT_DELAY_MS)).await;

            if controller.is_connected() {
                continue;
            }
            info!("Reconnecting to WiFi network: {}", ssid);
            if controller.connect_async().await.is_ok() {
                info!("WiFi reconnected!\n");
                connected_state.store(true, Ordering::Relaxed);
                connected_event.signal(());
            } else {
                error!("WiFi reconnect failed");
                connected_state.store(false, Ordering::Relaxed);
            }
        } else {
            Timer::after(Duration::from_millis(DISCONNECTED_POLL_MS)).await;
        }
    }
}

/// Network stack runner task
#[embassy_executor::task]
pub async fn net_task(mut runner: Runner<'static, Interface<'static>>) {
    runner.run().await;
}

/// Periodically logs WiFi RSSI and basic health for diagnostics.
#[embassy_executor::task]
pub async fn wifi_health_monitor(controller: WifiController<'static>) {
    loop {
        if controller.is_connected() {
            match controller.rssi() {
                Ok(rssi) => debug!("WiFi RSSI: {} dBm", rssi),
                Err(e) => debug!("WiFi RSSI unavailable: {:?}", e),
            }
        }
        Timer::after(Duration::from_secs(10)).await;
    }
}
