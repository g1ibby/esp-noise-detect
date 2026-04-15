use crate::drivers::gpio::XiaoLed;
use core::sync::atomic::{AtomicBool, Ordering};
use embassy_sync::watch;
use embassy_time::{Duration, Timer};

const NOT_CONNECTED_ON_MS: u64 = 100;
const NOT_CONNECTED_OFF_MS: u64 = 300;
const RECORDING_ON_MS: u64 = 500;
const IDLE_BLINK_MS: u64 = 150;
const IDLE_PAUSE_MS: u64 = 500;

async fn wait_with_recording_update(
    duration_ms: u64,
    recording_state: &mut watch::DynReceiver<'static, bool>,
    is_recording: &mut bool,
) {
    Timer::after(Duration::from_millis(duration_ms)).await;
    if let Some(next) = recording_state.try_changed() {
        *is_recording = next;
    }
}

#[embassy_executor::task]
pub async fn led_status(
    mut status_led: XiaoLed,
    connected_state: &'static AtomicBool,
    mut recording_state: watch::DynReceiver<'static, bool>,
) {
    let mut is_recording = recording_state.get().await;

    loop {
        if let Some(next) = recording_state.try_changed() {
            is_recording = next;
        }

        if !connected_state.load(Ordering::Relaxed) {
            status_led.on();
            wait_with_recording_update(
                NOT_CONNECTED_ON_MS,
                &mut recording_state,
                &mut is_recording,
            )
            .await;
            status_led.off();
            wait_with_recording_update(
                NOT_CONNECTED_OFF_MS,
                &mut recording_state,
                &mut is_recording,
            )
            .await;
            continue;
        }

        if is_recording {
            status_led.on();
            wait_with_recording_update(RECORDING_ON_MS, &mut recording_state, &mut is_recording)
                .await;
            continue;
        }

        status_led.on();
        wait_with_recording_update(IDLE_BLINK_MS, &mut recording_state, &mut is_recording).await;
        status_led.off();
        wait_with_recording_update(IDLE_BLINK_MS, &mut recording_state, &mut is_recording).await;
        if !connected_state.load(Ordering::Relaxed) {
            continue;
        }
        status_led.on();
        wait_with_recording_update(IDLE_BLINK_MS, &mut recording_state, &mut is_recording).await;
        status_led.off();
        wait_with_recording_update(IDLE_PAUSE_MS, &mut recording_state, &mut is_recording).await;
    }
}
