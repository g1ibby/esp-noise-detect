//! Burn `Batcher` that stacks [`WindowedAudioItem`]s into a
//! `(batch, time)` waveform tensor and a `(batch,)` label tensor.
//!
//! The output struct carries waveform + label as tensors, plus per-item
//! metadata (file path, start/end seconds) passed through unchanged.

use std::marker::PhantomData;
use std::path::PathBuf;

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

use super::dataset::WindowedAudioItem;

/// One batched set of windows. Shapes:
///
/// * `waveforms`: `(batch, window_samples)` f32.
/// * `labels`: `(batch,)` int, values in `0..class_names.len()`.
#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub waveforms: Tensor<B, 2>,
    pub labels: Tensor<B, 1, Int>,
    pub files: Vec<PathBuf>,
    pub starts: Vec<f32>,
    pub ends: Vec<f32>,
}

/// Batcher that moves CPU-side [`WindowedAudioItem`]s onto the target
/// Burn device.
///
/// Stateless; one flat `Vec<f32>` is built for the batch and handed to
/// `TensorData::new`, which uploads once rather than stacking N
/// per-item tensors.
#[derive(Clone, Debug)]
pub struct AudioBatcher<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> Default for AudioBatcher<B> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> AudioBatcher<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Batcher<B, WindowedAudioItem, AudioBatch<B>> for AudioBatcher<B> {
    fn batch(&self, items: Vec<WindowedAudioItem>, device: &B::Device) -> AudioBatch<B> {
        assert!(!items.is_empty(), "AudioBatcher received empty batch");
        let batch_size = items.len();
        let window_samples = items[0].waveform.len();

        let mut wave_buf = Vec::with_capacity(batch_size * window_samples);
        let mut label_buf = Vec::with_capacity(batch_size);
        let mut files = Vec::with_capacity(batch_size);
        let mut starts = Vec::with_capacity(batch_size);
        let mut ends = Vec::with_capacity(batch_size);
        for it in items {
            assert_eq!(
                it.waveform.len(),
                window_samples,
                "batch window length mismatch: expected {window_samples}, got {}",
                it.waveform.len(),
            );
            wave_buf.extend_from_slice(&it.waveform);
            label_buf.push(it.label as i64);
            files.push(it.file);
            starts.push(it.start_s);
            ends.push(it.end_s);
        }

        let waveforms = Tensor::<B, 2>::from_data(
            TensorData::new(wave_buf, [batch_size, window_samples]),
            device,
        );
        let labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(label_buf, [batch_size]),
            device,
        );

        AudioBatch {
            waveforms,
            labels,
            files,
            starts,
            ends,
        }
    }
}
