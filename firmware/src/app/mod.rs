pub mod config;

use alloc::vec::Vec;
use core::sync::atomic::AtomicBool;

pub use config::{AppConfig, NetOutChannel, OperationMode};
use embassy_executor::Spawner;
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, channel::Channel, signal::Signal, watch};
use embassy_time::{Duration, Timer};
use esp_hal::{
    analog::adc::{Adc, AdcConfig, Attenuation},
    dma_buffers,
    i2s::master::{Channels, Config as I2sConfig, DataFormat, I2s},
    interrupt::software::SoftwareInterruptControl,
    time::Rate,
    timer::timg::TimerGroup,
};
use esp_radio::wifi;
use log::{error, info};
use static_cell::StaticCell;
use wire_protocol::RecordingModeKind;

use crate::{
    infra::{i2s_capture, led_xiao, net_stack, tcp_client},
    mk_static,
};

pub async fn start(spawner: Spawner, config: &AppConfig) {
    info!(
        "Starting dataset collector | WiFi SSID: {} | server: {}.{}.{}.{}:{}",
        config.ssid,
        config.server_ip[0],
        config.server_ip[1],
        config.server_ip[2],
        config.server_ip[3],
        config.server_port
    );

    // Initialize esp-hal peripherals and clocks
    let peripherals = esp_hal::init(esp_hal::Config::default());

    // Small internal heap + PSRAM region for audio buffering
    esp_alloc::heap_allocator!(size: 72 * 1024);
    let psram = esp_hal::psram::Psram::new(peripherals.PSRAM, Default::default());
    let (psram_start, psram_size) = psram.raw_parts();
    if psram_size == 0 {
        // PSRAM is required for sustained audio buffering (DMA + streaming).
        // Without external memory, heap pressure would cause instability.
        panic!("No PSRAM detected – required for audio");
    }
    // Expose the entire PSRAM region to the global allocator. Typical XIAO
    // ESP32S3 boards provide enough PSRAM to cover multi-second audio buffers
    // while keeping internal RAM free for stacks and peripherals.
    init_psram_heap(psram_start, psram_size);
    let _test_vec = Vec::<u32>::with_capacity(1024);

    // Start scheduler + embassy support (required by esp-radio).
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_int = SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_int.software_interrupt0);

    // Bring up WiFi
    // Extract peripherals we need before moving out of the struct
    let wifi = peripherals.WIFI;
    let i2s0 = peripherals.I2S0;
    let dma_ch0 = peripherals.DMA_CH0;
    let gpio7 = peripherals.GPIO7; // BCLK
    let gpio6 = peripherals.GPIO6; // WS
    let gpio5 = peripherals.GPIO5; // DIN
    let gpio21 = peripherals.GPIO21; // LED
    let io_mux = peripherals.IO_MUX;

    let wifi_cfg = wifi::ControllerConfig::default()
        .with_tx_queue_size(5)
        // Reserve static TX buffers (~1.6KB each) from internal SRAM at init time,
        // before audio Vecs can fill the internal heap via LLFF.
        .with_static_tx_buf_num(4);
    let (wifi_controller, interfaces) = match wifi::new(wifi, wifi_cfg) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to init WiFi: {:?}", e);
            return;
        }
    };
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
    const RECORDING_WATCH_CAPACITY: usize = 4;

    static NET_OUT_CHANNEL: StaticCell<NetOutChannel> = StaticCell::new();
    static RECORDING_STATE: StaticCell<watch::Watch<NoopRawMutex, bool, RECORDING_WATCH_CAPACITY>> =
        StaticCell::new();
    static CONNECTED_EVENT: StaticCell<Signal<NoopRawMutex, ()>> = StaticCell::new();
    static CONNECTED_STATE: StaticCell<AtomicBool> = StaticCell::new();
    static FLUSH_ACK: StaticCell<Signal<NoopRawMutex, ()>> = StaticCell::new();
    const FLUSH_REQ_CAPACITY: usize = 2;
    static FLUSH_REQ: StaticCell<watch::Watch<NoopRawMutex, u32, FLUSH_REQ_CAPACITY>> =
        StaticCell::new();
    const PUMP_WATCH_CAPACITY: usize = 4;
    static PUMP_STATE: StaticCell<watch::Watch<NoopRawMutex, bool, PUMP_WATCH_CAPACITY>> =
        StaticCell::new();

    let net_out_channel = NET_OUT_CHANNEL.init(Channel::new());
    let recording_state_watch = RECORDING_STATE.init(watch::Watch::new());
    let recording_state_sender_tx: watch::DynSender<'static, bool> =
        recording_state_watch.sender().into();
    let recording_state_sender_orch: watch::DynSender<'static, bool> =
        recording_state_watch.sender().into();
    recording_state_sender_tx.send(false);
    let recording_state_for_audio = recording_state_watch
        .receiver()
        .expect("audio recorder watch receiver")
        .into();
    let recording_state_for_led = recording_state_watch
        .receiver()
        .expect("led status watch receiver")
        .into();
    let connected_event = CONNECTED_EVENT.init(Signal::new());
    let connected_state = CONNECTED_STATE.init(AtomicBool::new(false));
    let flush_ack = FLUSH_ACK.init(Signal::new());
    let flush_watch = FLUSH_REQ.init(watch::Watch::new());
    let flush_req_sender: watch::DynSender<'static, u32> = flush_watch.sender().into();
    // Initialize baseline value for flush sequence
    flush_req_sender.send(0);
    let flush_rx: watch::DynReceiver<'static, u32> =
        flush_watch.receiver().expect("flush watch receiver").into();

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

    // Determine recording mode at compile time via MODE env config
    let mode = match config::MODE {
        OperationMode::PumpGated => RecordingModeKind::PumpGated,
        OperationMode::Continuous => RecordingModeKind::Continuous,
    };

    // Spawn network transmitter (uses wire-protocol)
    // Network readiness watch
    const NET_READY_CAPACITY: usize = 2;
    static NET_READY: StaticCell<watch::Watch<NoopRawMutex, bool, NET_READY_CAPACITY>> =
        StaticCell::new();
    let net_ready_watch = NET_READY.init(watch::Watch::new());
    let net_ready_sender: watch::DynSender<'static, bool> = net_ready_watch.sender().into();
    net_ready_sender.send(false);

    spawner
        .spawn(tcp_client::network_transmitter(
            stack,
            net_out_channel,
            net_ready_sender,
            recording_state_sender_tx,
            connected_event,
            config.server_ip,
            config.server_port,
            config.device_id,
            config.device_version,
            config.sample_rate,
            config.channels as u8,
            config.bits_per_sample as u8,
            config.chunk_samples,
            mode,
        ).unwrap());

    // Initialize I2S and spawn audio_recorder
    {
        const I2S_BUFFER_SIZE: usize = 4092 * 8;
        let (rx_buffer, rx_descriptors, _, _) = dma_buffers!(I2S_BUFFER_SIZE, 0);

        // Philips standard with 32-bit slots per channel. Many microphones place
        // the meaningful 16/24-bit sample in the upper bits; we later extract a
        // mono 16-bit sample from the high 16 bits of each 32-bit word.
        let i2s_cfg = I2sConfig::new_tdm_philips()
            .with_sample_rate(Rate::from_hz(config.sample_rate))
            .with_data_format(DataFormat::Data32Channel32)
            .with_channels(Channels::STEREO);
        let i2s = I2s::new(i2s0, dma_ch0, i2s_cfg)
            .expect("Failed to initialize I2S")
            .into_async();

        // Pin mapping for XIAO ESP32S3: BCLK=GPIO7, WS/LRCK=GPIO6, DIN=GPIO5
        let i2s_rx = i2s
            .i2s_rx
            .with_bclk(gpio7)
            .with_ws(gpio6)
            .with_din(gpio5)
            .build(rx_descriptors);

        spawner
            .spawn(i2s_capture::audio_recorder(
                i2s_rx,
                rx_buffer,
                net_out_channel,
                recording_state_for_audio,
                flush_rx,
                config.sample_rate,
                config.channels as u8,
                config.chunk_samples,
                flush_ack,
            ).unwrap());
    }

    // Orchestrator (runs in both modes)
    {
        // Wire pump monitor in PumpGated mode
        let pump_rx_opt: Option<watch::DynReceiver<'static, bool>> =
            if matches!(config::MODE, OperationMode::PumpGated) {
                // Wire ADC1 on GPIO3 / A2 with calibration (_11dB) and spawn pump monitor
                let mut adc1_cfg = AdcConfig::new();
                let adc_pin = adc1_cfg.enable_pin(peripherals.GPIO3, Attenuation::_11dB);
                let adc1 = Adc::new(peripherals.ADC1, adc1_cfg);
                let pump_cfg = crate::app::config::pump_config();

                let pump_watch = PUMP_STATE.init(watch::Watch::new());
                let pump_tx: watch::DynSender<'static, bool> = pump_watch.sender().into();
                // initialize to OFF
                pump_tx.send(false);
                let pump_rx = pump_watch
                    .receiver()
                    .expect("pump state watch receiver")
                    .into();
                // spawn pump monitor tied to ADC
                spawner
                    .spawn(crate::infra::pump_monitor::pump_monitor_xiao_adc(
                        adc1, adc_pin, pump_tx, pump_cfg,
                    ).unwrap());
                Some(pump_rx)
            } else {
                None
            };

        // Orchestrator gets net readiness receiver
        let net_ready_rx: watch::DynReceiver<'static, bool> = net_ready_watch
            .receiver()
            .expect("net ready watch receiver")
            .into();

        spawner
            .spawn(
                crate::infra::recording_orchestrator::recording_orchestrator(
                    net_out_channel,
                    recording_state_sender_orch,
                    net_ready_rx,
                    mode,
                    pump_rx_opt,
                    flush_req_sender,
                    flush_ack,
                ).unwrap(),
            );
    }

    // LED status (GPIO21 on XIAO ESP32S3, inverted logic)
    let _io = esp_hal::gpio::Io::new(io_mux);
    let status_led = crate::drivers::gpio::XiaoLed::new(gpio21);
    spawner
        .spawn(led_xiao::led_status(
            status_led,
            connected_state,
            recording_state_for_led,
        ).unwrap());

    // Keep the app alive and let spawned tasks run
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
