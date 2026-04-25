//! Regression test for the 1-D `CubeCount` overflow in the framing
//! kernel.
//!
//! `stft()`'s `frame_and_window_kernel` launches
//! `num_elems = batch * n_frames * n_fft` threads. Before the fix the
//! launcher used `CubeCount::new_1d(num_elems / 256)`, which wgpu rejects
//! at `num_elems / 256 > 65535`.
//!
//! Sizing: `batch = 1`, `n_fft = 512`, `hop = 512` (non-overlapping
//! rectangular frames), `n_frames = 32768`. That puts the framing kernel
//! at `num_elems = 16_777_216` → 65536 cubes, one past the 1-D limit.
//!
//! We deliberately pick `n_fft = 512` (not 256) because `cubek-fft::rfft`
//! launches its own `CubeCount::new_1d(B * n_frames)` and upstream has the
//! same latent overflow (per the task scope, `cubek-fft` is out-of-scope).
//! Staying at `B * n_frames = 32768 <= 65535` keeps the rfft path safe so
//! this test isolates the framing-kernel fix.
//!
//! Correctness check: rectangular window + all-ones signal → every frame's
//! DFT is `[n_fft, 0, 0, ..., 0]` because `sum(1) = n_fft`.

mod common;

use common::{client, dtype_f32, upload_2d, upload_1d, read_tensor, Runtime};
use cubecl::std::tensor::TensorHandle;
use cubek_stft::stft;

const BATCH: usize = 1;
const N_FFT: usize = 512;
const HOP: usize = 512;
const N_FRAMES: usize = 32_768;
const TIME: usize = (N_FRAMES - 1) * HOP + N_FFT; // = 16_777_216

#[test]
fn stft_over_1d_dispatch_cap() {
    let client = client();

    // All-ones signal → DFT of each framed window (all ones * rectangular
    // window) is [n_fft, 0, 0, ..., 0]. That gives us a cheap per-frame
    // invariant to check.
    let sig = vec![1.0f32; BATCH * TIME];
    let window = vec![1.0f32; N_FFT];

    let sig_t: TensorHandle<Runtime> = upload_2d(&client, &sig, BATCH, TIME);
    let win_t: TensorHandle<Runtime> = upload_1d(&client, &window);

    let (re, im) = stft(sig_t, win_t, N_FFT, HOP, dtype_f32());
    let re_host = read_tensor(&client, re);
    let im_host = read_tensor(&client, im);

    let n_freq = N_FFT / 2 + 1;
    assert_eq!(re_host.len(), BATCH * N_FRAMES * n_freq);

    // Spot-check the first, last, and a frame past the 65535-cube
    // boundary. With n_fft=512 the 65536th cube begins at flat index
    // 65535*256 = 16_776_960, which lands in frame 32767 (= 16_776_960 /
    // 512). Picking a frame at that boundary plus one and the final frame
    // confirms the kernel wrote both halves of the tiled dispatch grid.
    for &frame in &[0usize, 32_767, N_FRAMES - 1] {
        let base = frame * n_freq;
        let dc_re = re_host[base];
        let dc_im = im_host[base];
        assert!(
            (dc_re - N_FFT as f32).abs() < 1e-2,
            "frame {frame}: DC real = {dc_re}, want {}",
            N_FFT,
        );
        assert!(dc_im.abs() < 1e-2, "frame {frame}: DC imag = {dc_im}, want 0");
        // Non-DC bins should be ~0 for the all-ones signal.
        let mid_re = re_host[base + n_freq / 2];
        assert!(
            mid_re.abs() < 1e-2,
            "frame {frame}: bin {} real = {mid_re}, want 0",
            n_freq / 2,
        );
    }
}
