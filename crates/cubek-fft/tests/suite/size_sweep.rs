//! Large-`n_fft` regression tests.
//!
//! The intra-cube-parallel `rfft_kernel` silently produces zeros for
//! `n_fft >= 8192` on wgpu-Metal (Apple Silicon) when its two
//! `SharedMemory::<f32>::new(n_fft)` allocations per cube exceed the
//! 32 KB threadgroup-memory cap.
//!
//! Both tests sweep `n_fft` across the shared-memory boundary so a future
//! regression at 8 192 or 16 384 fails the test suite loudly.

use cubecl::{
    CubeElement, Runtime, TestRuntime, frontend::CubePrimitive, std::tensor::TensorHandle,
};
use cubek_fft::{irfft, rfft};
use cubek_test_utils::{HostData, HostDataType, HostDataVec};

const SIZES: &[usize] = &[1024, 2048, 4096, 8192, 16384];
const BATCH: usize = 1;

fn to_f32(host: HostData) -> Vec<f32> {
    match host.data {
        HostDataVec::F32(v) => v,
        _ => panic!("expected f32 host data"),
    }
}

/// All-ones input to RFFT must yield a spectrum whose DC bin equals
/// `n_fft` and whose other bins are ~0 (the DFT of a constant signal).
#[test]
fn rfft_dc_probe_sweep() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    for &n_fft in SIZES {
        let n_freq = n_fft / 2 + 1;
        let sig_data = vec![1.0f32; BATCH * n_fft];
        let sig_handle = client.create_from_slice(f32::as_bytes(&sig_data));
        let signal = TensorHandle::<TestRuntime>::new_contiguous(
            vec![BATCH, n_fft],
            sig_handle,
            dtype,
        );

        let (re_t, im_t) = rfft::<TestRuntime>(signal, 1, dtype);
        let re = to_f32(HostData::from_tensor_handle(&client, re_t, HostDataType::F32));
        let im = to_f32(HostData::from_tensor_handle(&client, im_t, HostDataType::F32));

        assert!(
            (re[0] - n_fft as f32).abs() < 1e-2,
            "n_fft={n_fft}: DC real = {}, want {}",
            re[0],
            n_fft,
        );
        assert!(im[0].abs() < 1e-2, "n_fft={n_fft}: DC imag = {}", im[0]);

        // Spot-check non-DC bins including indices beyond the shared-memory
        // boundary. All should be ~0 for a constant signal.
        for &k in &[1usize, n_freq / 2, n_freq - 1] {
            assert!(
                re[k].abs() < 1.0,
                "n_fft={n_fft}: bin {k} real = {} (want ~0)",
                re[k],
            );
            assert!(
                im[k].abs() < 1.0,
                "n_fft={n_fft}: bin {k} imag = {} (want ~0)",
                im[k],
            );
        }
    }
}

/// `irfft(rfft(x)) ≈ x` on a sinusoidal signal, swept across sizes.
#[test]
fn rfft_irfft_roundtrip_sweep() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    for &n_fft in SIZES {
        // Mix two sinusoids so no single bin dominates — the reconstruction
        // has to get several non-trivial FFT coefficients right.
        let sig_data: Vec<f32> = (0..n_fft)
            .map(|i| {
                let t = i as f32 / n_fft as f32;
                (2.0 * std::f32::consts::PI * 5.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 19.0 * t).cos()
            })
            .collect();
        let sig_handle = client.create_from_slice(f32::as_bytes(&sig_data));
        let signal = TensorHandle::<TestRuntime>::new_contiguous(
            vec![BATCH, n_fft],
            sig_handle,
            dtype,
        );

        let (re_t, im_t) = rfft::<TestRuntime>(signal, 1, dtype);
        let recovered = irfft::<TestRuntime>(re_t, im_t, 1, dtype);
        let recovered_host = to_f32(HostData::from_tensor_handle(
            &client,
            recovered,
            HostDataType::F32,
        ));

        let max_abs_err = sig_data
            .iter()
            .zip(recovered_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms: f32 =
            (sig_data.iter().map(|v| v * v).sum::<f32>() / n_fft as f32).sqrt();

        // f32 round-trip error scales with log2(n_fft). 1e-4 relative is
        // comfortable at n_fft=16384 and well under the 0.03 absolute bound
        // used by `large_fft_roundtrip`.
        let rel_err = max_abs_err / rms;
        assert!(
            rel_err < 1e-4,
            "n_fft={n_fft}: round-trip max_abs_err={max_abs_err}, rms={rms}, rel={rel_err}",
        );
    }
}
