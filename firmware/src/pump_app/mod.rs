pub mod config;

use alloc::vec::Vec;
use core::sync::atomic::AtomicBool;

pub use config::AppConfig;
use embassy_executor::Spawner;
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, channel::Channel, signal::Signal};
use embassy_time::{Duration, Timer};
use esp_hal::{
    clock::{Clocks, CpuClock},
    dma_circular_buffers,
    i2s::master::{Channels, Config as I2sConfig, DataFormat, I2s},
    interrupt::software::SoftwareInterruptControl,
    ram,
    system::Stack,
    time::Rate,
    timer::timg::TimerGroup,
};
use esp_radio::wifi;
use esp_rtos::embassy::Executor;
use log::{error, info};
use static_cell::StaticCell;

fn log_internal_heap(label: &str) {
    let free = esp_alloc::HEAP.free_caps(esp_alloc::MemoryCapability::Internal.into());
    info!("HEAP [{}]: {} bytes free ({} KB)", label, free, free / 1024);
}

use crate::{infra::net_stack, infra::pump_classifier, mk_static};

#[cfg(not(feature = "mqtt"))]
use crate::infra::tcp_client_pump;
#[cfg(feature = "mqtt")]
use crate::infra::mqtt_client_pump;

pub async fn start(spawner: Spawner, config: &AppConfig) {
    info!(
        "Starting Pump Monitor | WiFi SSID: {} | server: {}.{}.{}.{}:{}",
        config.ssid,
        config.server_ip[0],
        config.server_ip[1],
        config.server_ip[2],
        config.server_ip[3],
        config.server_port
    );

    // Initialize esp-hal peripherals and clocks.
    //
    // esp-hal defaults CPU clock to 80MHz; NN inference needs full speed for reasonable latency.
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    let clocks = Clocks::get();
    info!("CPU clock: {} Hz", clocks.cpu_clock.as_hz());

    // Small internal heap + PSRAM region for audio buffering
    esp_alloc::heap_allocator!(#[ram(reclaimed)] size: 70 * 1024);
    log_internal_heap("after heap init");

    let psram = esp_hal::psram::Psram::new(peripherals.PSRAM, Default::default());
    let (psram_start, psram_size) = psram.raw_parts();
    if psram_size == 0 {
        panic!("No PSRAM detected – required for audio");
    }
    init_psram_heap(psram_start, psram_size);
    log_internal_heap("after psram init");

    // Allocate Core 1 stack from internal heap (10KB) - mel scratch buffers moved to static BSS
    // reduces stack requirement from ~17KB to ~8KB. Internal heap budget: 70KB - 10KB - 48KB WiFi =
    // 12KB margin.
    let app_core_stack: &'static mut Stack<10240> =
        crate::util::alloc_internal_zeroed_aligned::<Stack<10240>>(16);
    log_internal_heap("after stack alloc");

    // Start scheduler + embassy support (required by esp-radio).
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_int = SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_int.software_interrupt0);

    // Extract peripherals we need before moving out of the struct
    let wifi = peripherals.WIFI;
    let cpu_ctrl = peripherals.CPU_CTRL;
    // I2S peripherals - will be moved to Core 1
    let i2s0 = peripherals.I2S0;
    let dma_ch0 = peripherals.DMA_CH0;
    let gpio7 = peripherals.GPIO7; // BCLK
    let gpio6 = peripherals.GPIO6; // WS
    let gpio5 = peripherals.GPIO5; // DIN

    let (wifi_controller, interfaces) = match wifi::new(wifi, Default::default()) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to init WiFi: {:?}", e);
            return;
        }
    };
    log_internal_heap("after wifi::new");
    let wifi_interface = interfaces.station;

    // Network stack (DHCPv4)
    let net_cfg = embassy_net::Config::dhcpv4(Default::default());
    let rng = esp_hal::rng::Rng::new();
    let seed = (rng.random() as u64) << 32 | rng.random() as u64;

    let (stack, runner) = embassy_net::new(
        wifi_interface,
        net_cfg,
        mk_static!(
            embassy_net::StackResources<3>,
            embassy_net::StackResources::<3>::new()
        ),
        seed,
    );
    let stack = mk_static!(embassy_net::Stack<'static>, stack);

    // IPC primitives
    // PumpStatusChannel crosses cores (Core 1 -> Core 0), uses CriticalSectionRawMutex
    static PUMP_STATUS_CHANNEL: StaticCell<pump_classifier::PumpStatusChannel> = StaticCell::new();
    let pump_status_channel: &'static _ = &*PUMP_STATUS_CHANNEL.init(Channel::new());

    // NetOutChannel will be created inside Core 1 (NoopRawMutex is not Send)

    static CONNECTED_EVENT: StaticCell<Signal<NoopRawMutex, ()>> = StaticCell::new();
    static CONNECTED_STATE: StaticCell<AtomicBool> = StaticCell::new();
    static SERVER_CONNECTED_EVENT: StaticCell<Signal<NoopRawMutex, ()>> = StaticCell::new();

    let connected_event = CONNECTED_EVENT.init(Signal::new());
    let connected_state = CONNECTED_STATE.init(AtomicBool::new(false));
    let server_connected_event = SERVER_CONNECTED_EVENT.init(Signal::new());

    // Spawn network runner and WiFi connection manager
    spawner.spawn(net_stack::net_task(runner).unwrap());
    spawner
        .spawn(net_stack::connection_manager(
            wifi_controller,
            connected_event,
            connected_state,
            config.ssid,
            config.password,
        ).unwrap());

    // Spawn pump network transmitter (TCP or MQTT based on feature)
    #[cfg(not(feature = "mqtt"))]
    spawner
        .spawn(tcp_client_pump::pump_network_transmitter(
            *stack,
            pump_status_channel,
            connected_event,
            server_connected_event,
            config.server_ip,
            config.server_port,
            config.device_id,
            config.device_version,
        ).unwrap());

    #[cfg(feature = "mqtt")]
    spawner
        .spawn(mqtt_client_pump::mqtt_pump_transmitter(
            *stack,
            pump_status_channel,
            connected_event,
            server_connected_event,
            config.mqtt_broker_ip,
            config.mqtt_broker_port,
            config.mqtt_client_id,
            config.mqtt_topic,
            config.device_id,
            config.mqtt_username,
            config.mqtt_password,
        ).unwrap());

    info!("Waiting for server connection...");
    server_connected_event.wait().await;
    server_connected_event.reset();
    log_internal_heap("after server connect");
    info!("Server connected; starting Core 1 with I2S + classifier");

    // Copy config values for Core 1 closure
    let sample_rate = config.sample_rate;
    let channels = config.channels as u8;
    let chunk_samples = config.chunk_samples;

    // Start Core 1 with I2S audio recorder + pump classifier
    log_internal_heap("before core1 start");

    esp_rtos::start_second_core(
        cpu_ctrl,
        sw_int.software_interrupt1,
        app_core_stack,
        move || {
            // Initialize I2S on Core 1 where it will run
            const I2S_BUFFER_SIZE: usize = 4092 * 8;
            let (rx_buffer, rx_descriptors, _, _) = dma_circular_buffers!(I2S_BUFFER_SIZE, 0);

            let i2s_cfg = I2sConfig::new_tdm_philips()
                .with_sample_rate(Rate::from_hz(sample_rate))
                .with_data_format(DataFormat::Data32Channel32)
                .with_channels(Channels::STEREO);
            let i2s = I2s::new(i2s0, dma_ch0, i2s_cfg)
                .expect("Failed to initialize I2S")
                .into_async();

            let i2s_rx = i2s
                .i2s_rx
                .with_bclk(gpio7)
                .with_ws(gpio6)
                .with_din(gpio5)
                .build(rx_descriptors);

            // NetOutChannel - created on Core 1 where both producer and consumer run
            static NET_OUT_CHANNEL: StaticCell<crate::pump_app::config::NetOutChannel> =
                StaticCell::new();
            let net_out_channel: &'static _ = &*NET_OUT_CHANNEL.init(Channel::new());

            // Recording state watches (dummy - always recording)
            static RECORDING_STATE: StaticCell<embassy_sync::watch::Watch<NoopRawMutex, bool, 1>> =
                StaticCell::new();
            let recording_state_watch = RECORDING_STATE.init(embassy_sync::watch::Watch::new());
            recording_state_watch.sender().send(true);
            let recording_state_rx = recording_state_watch.receiver().unwrap();

            static FLUSH_REQ: StaticCell<embassy_sync::watch::Watch<NoopRawMutex, u32, 1>> =
                StaticCell::new();
            let flush_watch = FLUSH_REQ.init(embassy_sync::watch::Watch::new());
            let flush_rx = flush_watch.receiver().unwrap();

            static FLUSH_ACK: StaticCell<Signal<NoopRawMutex, ()>> = StaticCell::new();
            let flush_ack = FLUSH_ACK.init(Signal::new());

            static EXECUTOR: StaticCell<Executor> = StaticCell::new();
            let executor = EXECUTOR.init(Executor::new());
            executor.run(|spawner| {
                spawner
                    .spawn(crate::infra::i2s_capture::audio_recorder(
                        i2s_rx,
                        rx_buffer,
                        net_out_channel,
                        recording_state_rx.into(),
                        flush_rx.into(),
                        sample_rate,
                        channels,
                        chunk_samples,
                        flush_ack,
                    ).unwrap());
                spawner
                    .spawn(pump_classifier::pump_classifier(
                        net_out_channel,
                        pump_status_channel,
                    ).unwrap());
            });
        },
    );

    info!("Core 1 started; pump monitor running on dual cores (I2S + classifier)");

    // Keep the app alive
    loop {
        Timer::after(Duration::from_secs(1)).await;
    }
}

fn init_psram_heap(start: *mut u8, size: usize) {
    unsafe {
        esp_alloc::HEAP.add_region(esp_alloc::HeapRegion::new(
            start,
            size,
            esp_alloc::MemoryCapability::External.into(),
        ));
    }
}
