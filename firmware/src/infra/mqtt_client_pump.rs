use core::fmt::Write as FmtWrite;

use embassy_net::{tcp::TcpSocket, Stack};
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, signal::Signal};
use embassy_time::{Duration, Instant, Timer};
use heapless::String;
use log::{error, info, warn};
use rust_mqtt::{
    Bytes,
    buffer::BumpBuffer,
    client::{
        Client, MqttError,
        options::{ConnectOptions, PublicationOptions, TopicReference},
    },
    config::KeepAlive,
    types::{MqttBinary, MqttString, QoS, TopicName},
};
use static_cell::StaticCell;
use wire_protocol::PumpState;

use crate::infra::pump_classifier::PumpStatusChannel;

/// Home Assistant MQTT Discovery topic prefix
const HA_DISCOVERY_PREFIX: &str = "homeassistant";

// Static buffers - reduced sizes to minimize stack usage during init
static RX_BUFFER: StaticCell<[u8; 512]> = StaticCell::new();
static TX_BUFFER: StaticCell<[u8; 512]> = StaticCell::new();

/// MQTT pump status transmitter task - publishes pump state to MQTT broker
#[embassy_executor::task]
pub async fn mqtt_pump_transmitter(
    stack: Stack<'static>,
    status_channel: &'static PumpStatusChannel,
    connected_event: &'static Signal<NoopRawMutex, ()>,
    server_connected_event: &'static Signal<NoopRawMutex, ()>,
    broker_ip: [u8; 4],
    broker_port: u16,
    client_id: &'static str,
    topic: &'static str,
    device_id: &'static str,
    username: Option<&'static str>,
    password: Option<&'static str>,
) {
    // Initialize static buffers (only happens once, panics if called twice)
    let rx_buffer = RX_BUFFER.init([0; 512]);
    let tx_buffer = TX_BUFFER.init([0; 512]);

    // Wait for initial WiFi connection
    connected_event.wait().await;
    connected_event.reset();

    // Wait for link up
    loop {
        if stack.is_link_up() {
            break;
        }
        Timer::after(Duration::from_millis(500)).await;
    }

    // Small delay to ensure network is fully ready (DHCP route installed, etc.)
    Timer::after(Duration::from_secs(2)).await;

    // State tracking for deduplication: only publish on change or heartbeat
    const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes
    let mut last_published_state: Option<PumpState> = None;
    let mut last_publish_time = Instant::now();

    loop {
        if !stack.is_link_up() {
            connected_event.wait().await;
            connected_event.reset();
            continue;
        }

        info!(
            "MQTT: Connecting to {}.{}.{}.{}:{}...",
            broker_ip[0], broker_ip[1], broker_ip[2], broker_ip[3], broker_port
        );

        let mut socket = TcpSocket::new(stack, rx_buffer, tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(10)));

        let remote_endpoint = (
            embassy_net::IpAddress::v4(broker_ip[0], broker_ip[1], broker_ip[2], broker_ip[3]),
            broker_port,
        );

        if let Err(e) = socket.connect(remote_endpoint).await {
            error!("MQTT: TCP connect failed: {:?}", e);
            Timer::after(Duration::from_secs(5)).await;
            continue;
        }

        info!("MQTT: TCP connected, establishing MQTT session...");

        // Buffer for rust-mqtt's internal packet handling
        let mut bump_buf = [0u8; 384];
        let mut bump = BumpBuffer::new(&mut bump_buf);

        // Create MQTT client: <_, _, _, MAX_SUBSCRIBES, RECEIVE_MAXIMUM, SEND_MAXIMUM, MAX_SUBSCRIPTION_IDENTIFIERS>
        let mut client = Client::<_, _, 0, 1, 1, 0>::new(&mut bump);

        // Build connect options
        let mut connect_opts = ConnectOptions::new()
            .clean_start()
            .keep_alive(KeepAlive::Seconds(core::num::NonZero::new(30).unwrap()));

        if let Some(u) = username {
            connect_opts = connect_opts.user_name(MqttString::from_str_unchecked(u));
        }
        if let Some(p) = password {
            connect_opts = connect_opts.password(MqttBinary::from_slice_unchecked(p.as_bytes()));
        }

        let client_id_str = MqttString::from_str_unchecked(client_id);

        // Connect to MQTT broker
        match client
            .connect(socket, &connect_opts, Some(client_id_str))
            .await
        {
            Ok(_info) => info!("MQTT: Connected to broker as '{}'", client_id),
            Err(e) => {
                error!("MQTT: Broker connect failed: {:?}", e);
                Timer::after(Duration::from_secs(5)).await;
                continue;
            }
        }

        // Safety: ConnectInfo from connect() is dropped, so no references into bump buffer remain
        unsafe { client.buffer_mut().reset() };

        // Publish Home Assistant MQTT Discovery config
        if let Err(e) = publish_ha_discovery(&mut client, device_id, topic).await {
            warn!("MQTT: Failed to publish HA discovery: {:?}", e);
            // Continue anyway - manual config still works
        } else {
            info!("MQTT: Home Assistant discovery published");
        }

        server_connected_event.signal(());
        info!("MQTT: Ready to publish on topic '{}'", topic);

        // Reset tracking on reconnect to force re-publish current state
        last_published_state = None;
        last_publish_time = Instant::now();

        // Main loop: send pump status as MQTT messages
        loop {
            let status_future = status_channel.receive();
            let keepalive_timeout = Timer::after(Duration::from_secs(30));

            match embassy_futures::select::select(status_future, keepalive_timeout).await {
                embassy_futures::select::Either::First(status) => {
                    let state_changed = last_published_state != Some(status);
                    let heartbeat_due = last_publish_time.elapsed() >= HEARTBEAT_INTERVAL;

                    // Only publish on state change or heartbeat interval
                    if state_changed || heartbeat_due {
                        let payload = match status {
                            PumpState::On => b"ON".as_slice(),
                            PumpState::Off => b"OFF".as_slice(),
                        };

                        let topic_name = TopicName::new_unchecked(
                            MqttString::from_str_unchecked(topic),
                        );
                        let pub_opts = PublicationOptions::new(TopicReference::Name(topic_name))
                            .qos(QoS::AtLeastOnce)
                            .retain();

                        match client
                            .publish(&pub_opts, Bytes::from(payload))
                            .await
                        {
                            Ok(_) => {
                                let reason = if state_changed { "changed" } else { "heartbeat" };
                                info!("MQTT: Published {:?} to {} ({})", status, topic, reason);
                                last_published_state = Some(status);
                                last_publish_time = Instant::now();
                            }
                            Err(ref e) if !e.is_recoverable() => {
                                warn!("MQTT: Unrecoverable error during publish: {:?}, reconnecting...", e);
                                break;
                            }
                            Err(e) => {
                                error!("MQTT: Publish failed: {:?}", e);
                                break;
                            }
                        }

                        // Poll for PUBACK since we use QoS 1
                        match client.poll().await {
                            Ok(_event) => {}
                            Err(ref e) if !e.is_recoverable() => {
                                warn!("MQTT: Unrecoverable error during poll: {:?}, reconnecting...", e);
                                break;
                            }
                            Err(e) => {
                                error!("MQTT: Poll failed: {:?}", e);
                                break;
                            }
                        }
                    }
                }
                embassy_futures::select::Either::Second(_) => {
                    // Send MQTT ping to keep connection alive
                    match client.ping().await {
                        Ok(()) => {}
                        Err(ref e) if !e.is_recoverable() => {
                            warn!("MQTT: Unrecoverable error during ping: {:?}, reconnecting...", e);
                            break;
                        }
                        Err(e) => {
                            error!("MQTT: Ping failed: {:?}", e);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Publish Home Assistant MQTT Discovery configuration
async fn publish_ha_discovery<'c, N: embedded_io_async::Read + embedded_io_async::Write>(
    client: &mut Client<'c, N, BumpBuffer<'c>, 0, 1, 1, 0>,
    device_id: &str,
    state_topic: &str,
) -> Result<(), MqttError<'c>> {
    // Build discovery topic: homeassistant/binary_sensor/<device_id>/pump/config
    let mut discovery_topic: String<96> = String::new();
    let _ = write!(
        discovery_topic,
        "{}/binary_sensor/{}/pump/config",
        HA_DISCOVERY_PREFIX, device_id
    );

    // Build discovery payload JSON
    let mut payload: String<320> = String::new();
    let _ = write!(
        payload,
        r#"{{"name":"Pump","state_topic":"{}","payload_on":"ON","payload_off":"OFF","device_class":"running","unique_id":"{}_pump","device":{{"identifiers":["{}"],"name":"Pump Sensor","model":"ESP32-S3"}}}}"#,
        state_topic, device_id, device_id
    );

    let topic_name = TopicName::new_unchecked(
        MqttString::from_str_unchecked(discovery_topic.as_str()),
    );

    // Publish with retain so HA discovers device even after restart
    let pub_opts = PublicationOptions::new(TopicReference::Name(topic_name))
        .qos(QoS::AtLeastOnce)
        .retain();

    client
        .publish(&pub_opts, Bytes::from(payload.as_bytes()))
        .await?;

    // Poll for PUBACK
    client.poll().await?;

    Ok(())
}
