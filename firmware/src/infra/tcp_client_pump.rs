use embassy_net::Stack;
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, signal::Signal};
use embassy_time::{Duration, Timer};
use embedded_io_async::Write;
use log::{error, info};
use wire_protocol::{Header, Hello, MessageType, PumpState, PumpStatusMsg};

use crate::infra::pump_classifier::PumpStatusChannel;

#[embassy_executor::task]
pub async fn pump_network_transmitter(
    stack: Stack<'static>,
    status_channel: &'static PumpStatusChannel,
    connected_event: &'static Signal<NoopRawMutex, ()>,
    server_connected_event: &'static Signal<NoopRawMutex, ()>,
    server_ip: [u8; 4],
    server_port: u16,
    device_id: &'static str,
    device_version: &'static str,
) {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 1024];

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

    loop {
        if !stack.is_link_up() {
            connected_event.wait().await;
            connected_event.reset();
            continue;
        }

        info!("Connecting to {}.{}.{}.{}:{}...", 
            server_ip[0], server_ip[1], server_ip[2], server_ip[3], server_port);

        let mut socket = embassy_net::tcp::TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(10)));

        let remote_endpoint = (
            embassy_net::IpAddress::v4(server_ip[0], server_ip[1], server_ip[2], server_ip[3]),
            server_port,
        );

        if let Err(e) = socket.connect(remote_endpoint).await {
            error!("Connect failed: {:?}", e);
            Timer::after(Duration::from_secs(5)).await;
            continue;
        }

        info!("Connected!");

        // Send HELLO
        if let Err(e) = send_hello(&mut socket, device_id, device_version).await {
            error!("Failed to send HELLO: {:?}", e);
            continue;
        }

        server_connected_event.signal(());

        // Main loop: wait for status updates or send keepalives
        let mut seq = 0;

        loop {
            // Wait for status or timeout for keepalive
            // We use select with a timeout or just check elapsed
            
            let status_future = status_channel.receive();
            let keepalive_timeout = Timer::after(Duration::from_secs(5));

            match embassy_futures::select::select(status_future, keepalive_timeout).await {
                embassy_futures::select::Either::First(status) => {
                    // Send PumpStatus
                    if let Err(e) = send_pump_status(&mut socket, status, seq).await {
                        error!("Failed to send status: {:?}", e);
                        break; // Reconnect
                    }
                    seq += 1;
                }
                embassy_futures::select::Either::Second(_) => {
                    // Send KeepAlive
                    if let Err(e) = send_keepalive(&mut socket, seq).await {
                        error!("Failed to send keepalive: {:?}", e);
                        break; // Reconnect
                    }
                    seq += 1;
                }
            }
        }
    }
}

async fn send_hello(
    socket: &mut embassy_net::tcp::TcpSocket<'_>,
    device_id: &str,
    version: &str,
) -> Result<(), ()> {
    let hello = Hello { device_id, version };
    let mut payload = [0u8; 128];
    let len = hello.encode_into(&mut payload).map_err(|_| ())?;
    
    let header = Header {
        msg_type: MessageType::Hello,
        length: len as u32,
        sequence: 0,
    };
    
    let mut head_buf = [0u8; 8];
    header.encode_into(&mut head_buf);

    socket.write_all(&head_buf).await.map_err(|_| ())?;
    socket.write_all(&payload[..len]).await.map_err(|_| ())?;
    
    Ok(())
}

async fn send_pump_status(
    socket: &mut embassy_net::tcp::TcpSocket<'_>,
    status: PumpState,
    seq: u32,
) -> Result<(), ()> {
    let msg = PumpStatusMsg { status };
    let mut payload = [0u8; 16];
    let len = msg.encode_into(&mut payload).map_err(|_| ())?;

    let header = Header {
        msg_type: MessageType::PumpStatus,
        length: len as u32,
        sequence: seq,
    };

    let mut head_buf = [0u8; 8];
    header.encode_into(&mut head_buf);

    socket.write_all(&head_buf).await.map_err(|_| ())?;
    socket.write_all(&payload[..len]).await.map_err(|_| ())?;

    Ok(())
}

async fn send_keepalive(
    socket: &mut embassy_net::tcp::TcpSocket<'_>,
    seq: u32,
) -> Result<(), ()> {
    let header = Header {
        msg_type: MessageType::Keepalive,
        length: 0,
        sequence: seq,
    };

    let mut head_buf = [0u8; 8];
    header.encode_into(&mut head_buf);

    socket.write_all(&head_buf).await.map_err(|_| ())?;
    Ok(())
}
