//! Custom training loop for TinyConv.
//!
//! We use a custom loop instead of Burn's `LearnerBuilder` because the
//! mel front-end and augmentation transforms operate on CubeCL
//! `TensorHandle`s and are not `Module`s.
//!
//! Per batch: pull the inner-backend waveform, apply augmentation (no
//! autodiff), compute mel spectrogram (no autodiff), re-wrap as an
//! autodiff tensor, forward through TinyConv, CE-loss, backward, step.
//!
//! Per epoch: validate with `model.valid()` on the inner backend,
//! compute macro-F1 / accuracy / mean loss on the host, track
//! best-by-macro-F1 for early stopping.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn_audiomentations::{Compose, Transform, TransformRng};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use cubecl::client::ComputeClient;
use cubecl::future::block_on;
use cubecl::std::tensor::TensorHandle;

use crate::data::{
    AudioBatch, AudioBatcher, DatasetConfig, Split, WindowedAudioDataset, WindowedAudioItem,
    load_manifest,
};
use crate::mel::{MelConfig, MelExtractor};
use crate::model::{TinyConv, TinyConvConfig};

use super::augment::build_pipeline;
use super::config::{MelCfg, ModelCfg, OptimCfg, SchedCfg, TrainAppConfig};
use super::metrics::BinaryClassificationStats;

// ---------------------------------------------------------------------------
// Backend-feature guards — fire BEFORE Cargo tries to build cubecl-cuda /
// cubecl-wgpu so the user sees a clear, actionable message instead of an
// opaque build-script panic from a transitive dep.
//
// `test-*` features are accepted alongside production features because the
// lib still has to compile when `cargo test --features test-metal` is the
// only invocation — tests use `cubecl::TestRuntime` and don't need the
// production `Selected*` aliases below.
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "test-cuda",
    feature = "test-metal",
    feature = "test-vulkan",
    feature = "test-cpu",
)))]
compile_error!(
    "nn-rs requires exactly one GPU backend feature: \
     --features cuda | metal | vulkan | webgpu \
     (or, for tests only: test-cuda | test-metal | test-vulkan | test-cpu)"
);

// CUDA on macOS: nvcc / libnvrtc is Linux/Windows only. Catch it here
// before cubecl-cuda's build.rs panics with an opaque "cuda not found"
// message.
#[cfg(all(any(feature = "cuda", feature = "test-cuda"), target_os = "macos"))]
compile_error!(
    "the `cuda` / `test-cuda` features are not supported on macOS. \
     Use `--no-default-features --features \"std,metal\"` (build) or \
     `--features \"std,test-metal\"` (tests) instead. \
     CUDA requires the NVIDIA toolkit (nvcc + libnvrtc), which is \
     Linux/Windows only."
);

// Metal on non-macOS: the MSL shader compiler is not available there.
#[cfg(all(any(feature = "metal", feature = "test-metal"), not(target_os = "macos")))]
compile_error!(
    "the `metal` / `test-metal` features require macOS. \
     On Linux/NVIDIA use `--no-default-features --features \"std,cuda\"` \
     (or `\"std,test-cuda\"` for tests); on Linux/AMD or as a fallback \
     use `--features \"std,vulkan\"` (or `\"std,test-vulkan\"`)."
);

// ---------------------------------------------------------------------------
// Compile-time backend selector. Exactly one of {cuda, metal, vulkan, webgpu}
// must be enabled via cargo features; the binaries' `required-features =
// ["gpu"]` enforces that at build time, and the `compile_error!` block at
// the bottom of this file gives an actionable message if zero backends — or
// the wrong-OS combination — is selected.
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
mod backend_sel {
    use burn_cubecl::CubeBackend;
    pub use cubecl::cuda::{CudaDevice as SelectedDevice, CudaRuntime as SelectedRuntime};
    pub type SelectedCube = CubeBackend<SelectedRuntime, f32, i32, u8>;
    pub type SelectedAutodiff = burn::backend::Autodiff<SelectedCube>;
}

#[cfg(all(
    not(feature = "cuda"),
    any(feature = "metal", feature = "vulkan", feature = "webgpu"),
))]
mod backend_sel {
    use burn_cubecl::CubeBackend;
    pub use cubecl::wgpu::{WgpuDevice as SelectedDevice, WgpuRuntime as SelectedRuntime};
    pub type SelectedCube = CubeBackend<SelectedRuntime, f32, i32, u8>;
    pub type SelectedAutodiff = burn::backend::Autodiff<SelectedCube>;
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
))]
mod selected_aliases {
    use super::Trainer;
    pub use super::backend_sel::{
        SelectedAutodiff, SelectedCube, SelectedDevice, SelectedRuntime,
    };

    // Back-compat aliases so existing tests / `export_weights.rs` don't
    // need a mass rename in the same commit. Prefer `SelectedCube` /
    // `SelectedAutodiff` / `TrainerConcrete` in new code; these `Wgpu*`
    // names are scheduled for removal once downstream code migrates.
    pub type WgpuCube = SelectedCube;
    pub type WgpuAutodiff = SelectedAutodiff;
    /// Kept as `Backend` for backwards-compat with the module's pub
    /// surface — type alias, not trait. Prefer [`SelectedCube`] in new
    /// code.
    pub type Backend = SelectedCube;
    pub type AutodiffBackendConcrete = SelectedAutodiff;
    /// The common Trainer shape the CLI and smoke tests use.
    pub type TrainerConcrete = Trainer<SelectedAutodiff, SelectedRuntime, f32, i32, u8>;
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
))]
pub use selected_aliases::{
    AutodiffBackendConcrete, Backend, SelectedAutodiff, SelectedCube, SelectedDevice,
    SelectedRuntime, TrainerConcrete, WgpuAutodiff, WgpuCube,
};

/// Per-step stage timings populated by [`Trainer::forward_train`]. All
/// fields are wall-clock milliseconds measured with `client.sync()`
/// between stages — only meaningful as diagnostic data, not as a
/// steady-state profile (the sync forces the GPU queue to drain and
/// pessimises throughput by ~10%). Used by `scripts/bench_augment.sh`
/// to attribute per-transform GPU cost.
///
/// Populated only when `Trainer::profile_stages` is true; otherwise all
/// fields stay at zero and the training loop keeps the GPU queue full.
///
/// `block_*_fwd_ms` break down `fwd_ms` into per-`TinyConvBlock` forward
/// cost (conv_down → bn_down → relu → conv_refine → bn_refine → relu),
/// plus the head. Three blocks today (`channels=[16,32,64]`) — if the
/// config adds a block the array grows; unused slots stay at zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimings {
    pub aug_ms: f32,
    pub mel_ms: f32,
    pub fwd_ms: f32,
    /// `backward()` kernel submission + GPU execution time. Excludes
    /// `from_grads` (graph walk) — that's `from_grads_ms` below.
    pub bwd_ms: f32,
    /// `GradientsParams::from_grads(grads, &model)` — host-side
    /// iteration over module parameters to bundle grads for the
    /// optimizer. Pure CPU work; if it shows up here, the fix is
    /// structural, not kernel-side.
    pub from_grads_ms: f32,
    /// `optim.step(lr, model, grads)`.
    pub opt_ms: f32,
    /// Per-block forward time. Up to 4 blocks tracked; zeroed beyond
    /// `len(self.channels)`.
    pub block_fwd_ms: [f32; 4],
    pub head_fwd_ms: f32,
}

/// Outcome of a completed training run.
#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub best_epoch: usize,
    pub best_val_macro_f1: f64,
    pub best_val_loss: f64,
    pub best_val_acc: f64,
    pub total_epochs_run: usize,
    pub best_checkpoint: PathBuf,
}

/// Everything the training loop needs at construction time.
///
/// Model weights live on `AB` (autodiff), mel / augment on the inner
/// `CubeBackend` — no autodiff, no wasted graph nodes. `AB::InnerBackend`
/// must equal `CubeBackend<R, F, I, BT>` exactly; that's a genuine
/// restriction (we're only supporting wgpu / CubeCL) but it buys us the
/// zero-copy bridge in `apply_augment`.
pub struct Trainer<AB, R, F, I, BT>
where
    AB: AutodiffBackend<InnerBackend = CubeBackend<R, F, I, BT>>,
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    pub cfg: TrainAppConfig,
    pub device: AB::Device,
    pub inner_device: <AB::InnerBackend as burn::tensor::backend::Backend>::Device,
    pub client: ComputeClient<R>,
    pub mel: MelExtractor<R>,
    pub augment_train: Option<Compose<R>>,
    /// Enable per-stage `client.sync()` + wall-clock timing in
    /// `forward_train`. Off by default because the syncs drain the GPU
    /// queue between stages and cost ~10% throughput. Turn on with the
    /// `--profile-stages` CLI flag when profiling.
    pub profile_stages: bool,
}

impl<AB, R, F, I, BT> Trainer<AB, R, F, I, BT>
where
    AB: AutodiffBackend<InnerBackend = CubeBackend<R, F, I, BT>> + 'static,
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    pub fn new(
        cfg: TrainAppConfig,
        client: ComputeClient<R>,
        device: AB::Device,
        inner_device: <AB::InnerBackend as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        let mel_cfg = to_mel_config(&cfg.features);
        let mel = MelExtractor::<R>::new(
            client.clone(),
            inner_device.clone(),
            mel_cfg,
            cfg.dataset.sample_rate,
        );
        let augment_train =
            build_pipeline::<R>(&client, cfg.dataset.sample_rate, &cfg.augment.train);
        Self {
            cfg,
            device,
            inner_device,
            client,
            mel,
            augment_train,
            profile_stages: false,
        }
    }

    /// Toggle per-stage sync + wall-clock timing in `forward_train`.
    /// Leaves the rest of the trainer untouched; fluent so the CLI can
    /// chain it after `Trainer::new`.
    pub fn with_profile_stages(mut self, on: bool) -> Self {
        self.profile_stages = on;
        self
    }

    /// End-to-end training with macro-F1 early stopping.
    pub fn fit(&self) -> std::io::Result<TrainOutcome> {
        let manifest_path = self
            .cfg
            .resolved_manifest()
            .expect("dataset.manifest_path is required");
        let items = load_manifest(&manifest_path)?;

        let ds_cfg = to_dataset_config(&self.cfg);
        let train_ds = WindowedAudioDataset::new(items.clone(), ds_cfg.clone(), Split::Train)
            .expect("build train dataset");
        let val_ds = WindowedAudioDataset::new(items, ds_cfg, Split::Val)
            .expect("build val dataset");

        let artifact_dir = PathBuf::from(&self.cfg.trainer.artifact_dir);
        std::fs::create_dir_all(&artifact_dir)?;

        let num_workers = self.cfg.dm.resolved_num_workers();
        let num_workers_auto = self.cfg.dm.num_workers.is_none();
        let train_loader: Arc<dyn DataLoader<AB, AudioBatch<AB>>> = build_loader::<AB>(
            train_ds,
            self.cfg.dm.batch_size,
            num_workers,
            true,
            self.cfg.trainer.seed,
        );
        let val_workers = num_workers.saturating_sub(1).max(1);
        let val_loader: Arc<dyn DataLoader<AB::InnerBackend, AudioBatch<AB::InnerBackend>>> =
            build_loader::<AB::InnerBackend>(
                val_ds,
                self.cfg.dm.batch_size,
                val_workers,
                false,
                self.cfg.trainer.seed,
            );

        let model_cfg = to_tinyconv_config(&self.cfg.model);
        let mut model = model_cfg.init::<AB>(&self.device);
        let mut optim = build_optimizer::<AB>(&self.cfg.optim);
        let mut rng = TransformRng::new(self.cfg.trainer.seed);

        let mut best = TrainOutcome {
            best_epoch: 0,
            best_val_macro_f1: -1.0,
            best_val_loss: f64::INFINITY,
            best_val_acc: 0.0,
            total_epochs_run: 0,
            best_checkpoint: artifact_dir.join("checkpoints/best"),
        };
        let mut epochs_since_best: usize = 0;

        println!(
            "training config: batch_size={} num_workers={}{} epochs={} lr={:.2e} \
             augment={} sched={} artifact_dir={}",
            self.cfg.dm.batch_size,
            num_workers,
            if num_workers_auto { " (auto)" } else { "" },
            self.cfg.trainer.max_epochs,
            self.cfg.optim.lr,
            self.augment_train.is_some(),
            self.cfg.sched.name,
            self.cfg.trainer.artifact_dir,
        );

        for epoch in 1..=self.cfg.trainer.max_epochs {
            let lr = epoch_lr(&self.cfg.sched, &self.cfg.optim, epoch);
            let epoch_start = std::time::Instant::now();

            // --- train -----------------------------------------------------
            //
            // Per-stage syncs in forward_train / bwd / opt are gated by
            // `self.profile_stages` — the hot path enqueues the whole
            // step (aug → mel → fwd → bwd → opt) into the GPU queue
            // without intermediate drains. The `loss.clone().into_scalar()`
            // host readback stays on every step: it's one sync per
            // step rather than five, and empirically it acts as a
            // useful rate limiter — removing it doesn't improve
            // throughput on this workload (GPU is already near
            // saturation) and tends to starve the data loader.
            let mut train_stats = BinaryClassificationStats::new();
            let profile = self.profile_stages;
            let mut step: usize = 0;
            for batch in train_loader.iter() {
                let batch_size = batch.labels.dims()[0];
                let (_logits, loss, mut stages) = self.forward_train(&model, batch, &mut rng);
                let loss_scalar = loss.clone().into_scalar().elem::<f32>();

                // backward() and from_grads() are separate CPU/GPU
                // costs. Measuring them together hid the split in the
                // earlier profile. `backward()` submits grad kernels +
                // (optionally) syncs; `from_grads` then iterates the
                // module tree on the host to package the grads for the
                // optimizer.
                let t_bwd = profile.then(std::time::Instant::now);
                let grads = loss.backward();
                if profile {
                    block_on(self.client.sync()).ok();
                    stages.bwd_ms = t_bwd.unwrap().elapsed().as_secs_f32() * 1000.0;
                }

                let t_fg = profile.then(std::time::Instant::now);
                let grads = GradientsParams::from_grads(grads, &model);
                if profile {
                    // No GPU work here — pure host walk over params —
                    // but sync anyway so any async param updates
                    // scheduled during the walk are accounted for.
                    block_on(self.client.sync()).ok();
                    stages.from_grads_ms = t_fg.unwrap().elapsed().as_secs_f32() * 1000.0;
                }

                let t_opt = profile.then(std::time::Instant::now);
                model = optim.step(lr, model, grads);
                if profile {
                    block_on(self.client.sync()).ok();
                    stages.opt_ms = t_opt.unwrap().elapsed().as_secs_f32() * 1000.0;
                }

                train_stats.add_loss(loss_scalar, batch_size);
                if step % 25 == 0 {
                    if profile {
                        let bf = stages.block_fwd_ms;
                        println!(
                            "[train] epoch {epoch} step {step:>5}  loss={loss_scalar:.4}  \
                             lr={lr:.2e}  elapsed={:.1}s  \
                             aug={:.0}ms mel={:.0}ms fwd={:.0}ms \
                             [blk0={:.0} blk1={:.0} blk2={:.0} blk3={:.0} head={:.0}]  \
                             bwd={:.0}ms from_grads={:.0}ms opt={:.0}ms",
                            epoch_start.elapsed().as_secs_f32(),
                            stages.aug_ms, stages.mel_ms, stages.fwd_ms,
                            bf[0], bf[1], bf[2], bf[3], stages.head_fwd_ms,
                            stages.bwd_ms, stages.from_grads_ms, stages.opt_ms,
                        );
                    } else {
                        println!(
                            "[train] epoch {epoch} step {step:>5}  loss={loss_scalar:.4}  \
                             lr={lr:.2e}  elapsed={:.1}s",
                            epoch_start.elapsed().as_secs_f32(),
                        );
                    }
                }
                step += 1;
            }

            // --- val -------------------------------------------------------
            let model_valid = model.valid();
            let mut val_stats = BinaryClassificationStats::new();
            for batch in val_loader.iter() {
                let batch_size = batch.labels.dims()[0];
                let labels = batch.labels.clone();
                let (logits, loss) = self.forward_eval(&model_valid, batch);
                let loss_scalar = loss.into_scalar().elem::<f32>();
                val_stats.add_loss(loss_scalar, batch_size);
                let logits_vec =
                    logits.into_data().convert::<f32>().to_vec::<f32>().unwrap();
                let targets_vec =
                    labels.into_data().convert::<i64>().to_vec::<i64>().unwrap();
                val_stats.update(&logits_vec, &targets_vec);
            }

            let val_loss = val_stats.mean_loss();
            let val_acc = val_stats.accuracy();
            let val_f1 = val_stats.macro_f1();
            let train_loss = train_stats.mean_loss();
            println!(
                "epoch {epoch:>3}/{total}  lr={lr:.2e}  train_loss={train_loss:.4}  \
                 val_loss={val_loss:.4}  val_acc={val_acc:.4}  val_f1_macro={val_f1:.4}",
                total = self.cfg.trainer.max_epochs
            );

            best.total_epochs_run = epoch;
            let improved = val_f1 > best.best_val_macro_f1 + 1e-9;
            if improved {
                best.best_epoch = epoch;
                best.best_val_macro_f1 = val_f1;
                best.best_val_loss = val_loss;
                best.best_val_acc = val_acc;
                epochs_since_best = 0;
                save_checkpoint(&model, &best.best_checkpoint)?;
            } else {
                epochs_since_best += 1;
            }

            if epochs_since_best >= self.cfg.trainer.early_stopping_patience {
                println!(
                    "early stopping at epoch {epoch} — no val_f1_macro improvement in {} epochs \
                     (best {:.4} at epoch {})",
                    self.cfg.trainer.early_stopping_patience,
                    best.best_val_macro_f1,
                    best.best_epoch,
                );
                break;
            }
        }

        Ok(best)
    }

    /// Forward pass that returns `(logits, loss, stages)` on the autodiff
    /// backend with augmentation + mel computed off-graph.
    ///
    /// When `self.profile_stages` is false (the default) the GPU queue
    /// is left pipelined: all three stages are enqueued back-to-back
    /// and the returned `StageTimings` is zero-filled. When the flag is
    /// on, `client.sync()` drains the queue between stages and the
    /// returned `StageTimings` carries per-phase wall-clock ms — at
    /// ~10% steady-state throughput cost.
    pub fn forward_train(
        &self,
        model: &TinyConv<AB>,
        batch: AudioBatch<AB>,
        rng: &mut TransformRng,
    ) -> (Tensor<AB, 2>, Tensor<AB, 1>, StageTimings) {
        let targets = batch.labels;
        let waveforms_inner = batch.waveforms.inner();

        let profile = self.profile_stages;
        let t_aug = if profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let waveforms_aug = match self.augment_train.as_ref() {
            Some(compose) => apply_augment::<R, F, I, BT>(
                waveforms_inner,
                compose,
                rng,
                &self.inner_device,
            ),
            None => waveforms_inner,
        };
        let aug_ms = if profile {
            block_on(self.client.sync()).ok();
            t_aug.unwrap().elapsed().as_secs_f32() * 1000.0
        } else {
            0.0
        };

        let t_mel = if profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mel = self.mel.forward(waveforms_aug);
        let mel_ms = if profile {
            block_on(self.client.sync()).ok();
            t_mel.unwrap().elapsed().as_secs_f32() * 1000.0
        } else {
            0.0
        };

        // Forward path splits: hot path runs `model.forward` with no
        // extra clones or syncs; profile path uses `forward_with_probes`
        // and syncs between blocks so per-layer cost can be read off.
        let mel_ad: Tensor<AB, 4> = Tensor::from_inner(mel);
        let mut block_fwd_ms = [0.0_f32; 4];
        let mut head_fwd_ms = 0.0_f32;
        let (logits, fwd_ms) = if profile {
            let t_fwd = std::time::Instant::now();
            let client = self.client.clone();
            let t_stage = std::time::Instant::now();
            // State threaded through the probe closure — we reuse
            // `last_ms` and `block_fwd_ms` across iterations. RefCell
            // is overkill; a local struct captured by `&mut` does it.
            struct ProbeState {
                last_ms: f32,
                block_fwd_ms: [f32; 4],
            }
            let mut state = ProbeState {
                last_ms: 0.0,
                block_fwd_ms: [0.0; 4],
            };
            let logits = model.forward_with_probes(mel_ad, |i| {
                block_on(client.sync()).ok();
                let now_ms = t_stage.elapsed().as_secs_f32() * 1000.0;
                if i < state.block_fwd_ms.len() {
                    state.block_fwd_ms[i] = now_ms - state.last_ms;
                }
                state.last_ms = now_ms;
            });
            block_on(self.client.sync()).ok();
            let total_ms = t_stage.elapsed().as_secs_f32() * 1000.0;
            head_fwd_ms = total_ms - state.last_ms;
            block_fwd_ms = state.block_fwd_ms;
            let fwd_ms = t_fwd.elapsed().as_secs_f32() * 1000.0;
            (logits, fwd_ms)
        } else {
            let logits = model.forward(mel_ad);
            (logits, 0.0)
        };
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets);

        (
            logits,
            loss,
            StageTimings {
                aug_ms,
                mel_ms,
                fwd_ms,
                bwd_ms: 0.0,
                from_grads_ms: 0.0,
                opt_ms: 0.0,
                block_fwd_ms,
                head_fwd_ms,
            },
        )
    }

    /// Validation / eval forward. Runs end-to-end on the inner backend.
    pub fn forward_eval(
        &self,
        model: &<TinyConv<AB> as AutodiffModule<AB>>::InnerModule,
        batch: AudioBatch<AB::InnerBackend>,
    ) -> (
        Tensor<AB::InnerBackend, 2>,
        Tensor<AB::InnerBackend, 1>,
    ) {
        let targets = batch.labels;
        let mel = self.mel.forward(batch.waveforms);
        let logits = model.forward(mel);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets);
        (logits, loss)
    }

    /// Overfit a small set of batches for `steps` iterations — the Step
    /// 10 acceptance test. Returns the sequence of per-step losses.
    pub fn overfit_batches(
        &self,
        batches: Vec<AudioBatch<AB>>,
        steps: usize,
    ) -> Vec<f32> {
        let model_cfg = to_tinyconv_config(&self.cfg.model);
        let mut model = model_cfg.init::<AB>(&self.device);
        let mut optim = build_optimizer::<AB>(&self.cfg.optim);
        let lr = self.cfg.optim.lr;
        let mut rng = TransformRng::new(self.cfg.trainer.seed);

        let mut history = Vec::with_capacity(steps * batches.len());
        for _step in 0..steps {
            for batch in &batches {
                let (_, loss, _) =
                    self.forward_train(&model, clone_batch::<AB>(batch), &mut rng);
                let loss_scalar = loss.clone().into_scalar().elem::<f32>();
                history.push(loss_scalar);
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads);
            }
        }
        history
    }
}

/// Apply a `Compose<R>` to a `(batch, time)` Burn tensor by pulling the
/// underlying `CubeTensor` handle, running the augmentation pipeline,
/// then rewrapping the resulting handle as a Burn tensor of the same
/// shape. Zero-copy — same storage buffer is reused.
fn apply_augment<R, F, I, BT>(
    waveforms: Tensor<CubeBackend<R, F, I, BT>, 2>,
    compose: &Compose<R>,
    rng: &mut TransformRng,
    device: &<CubeBackend<R, F, I, BT> as burn::tensor::backend::Backend>::Device,
) -> Tensor<CubeBackend<R, F, I, BT>, 2>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let dims = waveforms.dims();
    let primitive: CubeTensor<R> = waveforms.into_primitive().tensor();
    let dtype_burn = primitive.dtype;
    let client = primitive.client.clone();
    let handle: TensorHandle<R> = primitive.into();
    let out_handle = compose.apply(handle, rng);
    let out_shape = out_handle.shape().to_vec();
    debug_assert_eq!(out_shape, vec![dims[0], dims[1]], "augment changed shape");
    let cube = CubeTensor::<R>::new_contiguous(
        client,
        device.clone(),
        burn::tensor::Shape::new([out_shape[0], out_shape[1]]),
        out_handle.handle,
        dtype_burn,
    );
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(cube))
}

fn clone_batch<B: burn::tensor::backend::Backend>(b: &AudioBatch<B>) -> AudioBatch<B> {
    AudioBatch {
        waveforms: b.waveforms.clone(),
        labels: b.labels.clone(),
        files: b.files.clone(),
        starts: b.starts.clone(),
        ends: b.ends.clone(),
    }
}

fn to_mel_config(cfg: &MelCfg) -> MelConfig {
    MelConfig {
        n_mels: cfg.n_mels,
        fmin: cfg.fmin,
        fmax: cfg.fmax,
        n_fft: cfg.n_fft,
        hop_length_ms: cfg.hop_length_ms,
        log: cfg.log,
        eps: cfg.eps,
        normalize: cfg.normalize,
        center: cfg.center,
    }
}

fn to_dataset_config(cfg: &TrainAppConfig) -> DatasetConfig {
    DatasetConfig {
        sample_rate: cfg.dataset.sample_rate,
        window_s: cfg.dataset.window_s,
        hop_s: cfg.dataset.hop_s,
        class_names: cfg.dataset.class_names.clone(),
        manifest_path: cfg.dataset.manifest_path.as_ref().map(PathBuf::from),
    }
}

fn to_tinyconv_config(cfg: &ModelCfg) -> TinyConvConfig {
    TinyConvConfig {
        channels: cfg.channels.clone(),
        dropout: cfg.dropout,
        n_classes: 2,
    }
}

fn build_optimizer<AB: AutodiffBackend>(
    cfg: &OptimCfg,
) -> OptimizerAdaptor<AdamW, TinyConv<AB>, AB> {
    assert_eq!(cfg.name, "adamw", "only adamw is supported");
    AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .init::<AB, TinyConv<AB>>()
}

fn epoch_lr(sched: &SchedCfg, optim: &OptimCfg, epoch: usize) -> f64 {
    if sched.name == "cosine" && sched.t_max > 0 {
        let base = optim.lr;
        let min = sched.min_lr;
        let t = epoch.saturating_sub(1).min(sched.t_max) as f64;
        let frac = t / sched.t_max as f64;
        let cos = (frac * std::f64::consts::PI).cos();
        min + 0.5 * (base - min) * (1.0 + cos)
    } else {
        optim.lr
    }
}

/// Build a DataLoader on `B`. The batcher on `B` handles the device
/// upload; workers deserialize WAV samples on CPU and the main thread
/// uploads to the GPU inside the batcher.
fn build_loader<B: burn::tensor::backend::Backend>(
    dataset: WindowedAudioDataset,
    batch_size: usize,
    num_workers: usize,
    shuffle: bool,
    seed: u64,
) -> Arc<dyn DataLoader<B, AudioBatch<B>>> {
    let batcher = AudioBatcher::<B>::new();
    let mut builder = DataLoaderBuilder::<
        B,
        WindowedAudioItem,
        AudioBatch<B>,
    >::new(batcher)
    .batch_size(batch_size)
    .num_workers(num_workers);
    if shuffle {
        builder = builder.shuffle(seed);
    }
    builder.build(dataset)
}

fn save_checkpoint<AB: AutodiffBackend>(
    model: &TinyConv<AB>,
    path: &Path,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let recorder = CompactRecorder::new();
    recorder
        .record(model.clone().into_record(), path.to_path_buf())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("save: {e}")))
}
