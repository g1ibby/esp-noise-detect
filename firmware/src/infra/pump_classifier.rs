#![allow(unused_imports)]
use alloc::vec::Vec;

use edgedl::features::mel::{
    MelScratch, compute_log_mel_simd_with_scratch, quantize_by_engine_exp,
};
use embassy_sync::{
    blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel, signal::Signal, watch,
};
use embassy_time::{Duration, Instant, Timer};
use log::{debug, error, info};
use static_cell::{ConstStaticCell, StaticCell};
use wire_protocol::PumpState;

use crate::{
    domain::{AudioData, NetOut},
    pump_app::config::NetOutChannel,
};

// Import edgedl macros
extern crate edgedl_macros;

// Bind the model
#[edgedl_macros::espdl_model(path = "./models/noise_model_3.espdl")]
struct __ModelBind;

// Constants from the noise example/model
const SAMPLE_RATE: usize = 32000;
const N_MELS: usize = 64;
const N_FRAMES: usize = 101;
const N_FFT: usize = 1024;
const HOP_LENGTH: usize = 320;
const FMIN_HZ: f32 = 50.0;
const FMAX_HZ: f32 = 16000.0;
const LOG_EPS: f32 = 1e-10;
const CENTER: bool = true;

// Buffer sizes
// We need 1 second of audio at 32kHz.
const INFER_WINDOW_SAMPLES: usize = 32000;

static INPUT: ConstStaticCell<edgedl::Aligned16<[i8; N_MELS * N_FRAMES]>> =
    ConstStaticCell::new(edgedl::Aligned16([0; N_MELS * N_FRAMES]));
// Arena in static BSS (dram_seg) for fast access. Core 1 stack is heap-allocated
// from dram2_seg to avoid competing for dram_seg space.
static ARENA: ConstStaticCell<edgedl::Aligned16<[i8; __ModelBind::ARENA_SIZE]>> =
    ConstStaticCell::new(edgedl::Aligned16([0; __ModelBind::ARENA_SIZE]));
// Mel scratch buffers in static BSS (dram_seg) - reduces stack usage from ~13KB to ~1KB
static MEL_SCRATCH: ConstStaticCell<MelScratch> = ConstStaticCell::new(MelScratch::new());

pub type PumpStatusChannel = Channel<CriticalSectionRawMutex, PumpState, 4>;

#[embassy_executor::task]
pub async fn pump_classifier(
    net_out_channel: &'static NetOutChannel,
    status_channel: &'static PumpStatusChannel,
) {
    info!("Pump classifier task started");

    // Keep large scratch buffers out of internal DRAM to preserve stack headroom for WiFi.
    // `compute_log_mel_simd` overwrites the entire buffer each inference window, so zero-init is
    // not strictly required, but we keep it for safety.
    let mel: &mut [[f32; N_FRAMES]; N_MELS] =
        crate::util::alloc_external_zeroed_aligned::<[[f32; N_FRAMES]; N_MELS]>(16);
    let input = INPUT.take();
    let arena = ARENA.take();
    let mel_scratch = MEL_SCRATCH.take();

    // Model metadata
    let in_id = __ModelBind::SPEC.inputs[0];
    let in_meta = __ModelBind::SPEC.values[in_id as usize];

    // Audio accumulation buffer (mono, 16-bit)
    let mut audio_buffer: Vec<i16> = Vec::with_capacity(INFER_WINDOW_SAMPLES + 4096);

    loop {
        // Receive audio chunks from NetOutChannel
        let msg = net_out_channel.receive().await;

        if let NetOut::Audio(audio_data) = msg {
            match audio_data {
                AudioData::Sample16(samples) => {
                    audio_buffer.extend_from_slice(&samples);
                }
                AudioData::Sample24(samples) => {
                    // Downsample 24-bit to 16-bit if necessary, or just take MSB
                    // Assuming the model was trained on 16-bit audio.
                    // If i2s_capture is configured for 24-bit, we might need to scale.
                    // For now, let's assume 16-bit mode is used or we just truncate.
                    for s in samples {
                        audio_buffer.push((s >> 8) as i16);
                    }
                }
            }

            // Check if we have enough samples for inference
            if audio_buffer.len() >= INFER_WINDOW_SAMPLES {
                let samples_ready = audio_buffer.len();
                info!("Running inference on {} samples", samples_ready);
                let t_total = Instant::now();

                // 1. Compute Mel Spectrogram (using scratch buffers to reduce stack usage)
                // Use the first INFER_WINDOW_SAMPLES
                let t_mel = Instant::now();
                compute_log_mel_simd_with_scratch::<N_MELS, N_FRAMES>(
                    &audio_buffer[..INFER_WINDOW_SAMPLES],
                    SAMPLE_RATE,
                    N_FFT,
                    HOP_LENGTH,
                    FMIN_HZ,
                    FMAX_HZ,
                    LOG_EPS,
                    CENTER,
                    mel,
                    mel_scratch,
                );
                let mel_ms = t_mel.elapsed().as_millis();
                embassy_futures::yield_now().await;

                // 2. Quantize
                let t_quant = Instant::now();
                quantize_by_engine_exp::<N_MELS, N_FRAMES>(mel, in_meta.exp, &mut input.0);
                let quant_ms = t_quant.elapsed().as_millis();
                embassy_futures::yield_now().await;

                // 3. Predict
                let mut probs: [f32; 2] = [0.0; 2];
                let t_rt_new = Instant::now();
                let mut rt = __ModelBind::new(&mut arena.0[..]).expect("runtime new");
                let rt_new_ms = t_rt_new.elapsed().as_millis();
                embassy_futures::yield_now().await;
                let t_pred = Instant::now();
                if let Err(e) = rt.predict_simd(&input.0, &mut probs) {
                    error!("Prediction failed: {:?}", e);
                } else {
                    let pred_ms = t_pred.elapsed().as_millis();
                    let off_prob = probs[0];
                    let on_prob = probs[1];
                    let label = if on_prob > off_prob {
                        PumpState::On
                    } else {
                        PumpState::Off
                    };

                    let total_ms = t_total.elapsed().as_millis();
                    info!(
                        "Inference timings: mel={}ms quant={}ms rt_new={}ms pred={}ms total={}ms | probs: OFF={:.3} ON={:.3} -> {:?}",
                        mel_ms, quant_ms, rt_new_ms, pred_ms, total_ms, off_prob, on_prob, label
                    );

                    // Send status
                    status_channel.try_send(label).ok();
                }

                // Clear buffer for next window
                // We discard the processed samples.
                // If we want overlap, we would keep some.
                // For now, simple non-overlapping windows.
                audio_buffer.clear();
            }
        }
    }
}
