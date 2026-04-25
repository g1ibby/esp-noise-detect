//! Unit tests for `burn_audiomentations::{LowPassFilter, HighPassFilter}`.
//!
//! We rely on `cubek-sinc-filter` being correct (it has its own test suite)
//! and only verify the *augmentation glue*:
//!
//! * Mel-space parameter sampling and bucket quantization work.
//! * `probability = 0.0` routes every row through the synthetic identity
//!   row at bank index 0 — exact for both HPF (cutoff 0) and LPF (cutoff
//!   at Nyquist — the windowed sinc collapses to a Kronecker delta).
//! * `probability = 1.0` perturbs the signal (stop-band energy is
//!   attenuated).

mod common;

use burn_audiomentations::{HighPassFilter, LowPassFilter, Transform, TransformRng};
use common::{client, dtype_f32, read_tensor, rms, sine, upload_2d, Runtime};
use cubecl::std::tensor::TensorHandle;

const SR: u32 = 32_000;

fn upload_single(
    c: &cubecl::client::ComputeClient<Runtime>,
    data: &[f32],
) -> TensorHandle<Runtime> {
    upload_2d(c, data, 1, data.len())
}

#[test]
fn lpf_probability_zero_is_exact_identity() {
    // LPF bucket 0 is a cutoff-at-Nyquist row whose windowed sinc
    // collapses to a Kronecker delta — exact identity after DC normalize.
    let client = client();
    let time = 1024;
    let signal = sine(3_000.0, time, SR);

    let lpf = LowPassFilter::new(client.clone(), 8_000.0, 14_000.0, SR, 0.0, 16, dtype_f32());
    let mut rng = TransformRng::new(42);
    let out = <LowPassFilter<Runtime> as Transform<Runtime>>::apply(
        &lpf,
        upload_single(&client, &signal),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let err = signal
        .iter()
        .zip(out_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(err < 1e-5, "lpf p=0 should be identity, got {err}");
}

#[test]
fn hpf_probability_zero_is_exact_identity() {
    // HPF uses bucket 0 = cutoff 0 as its identity row. Unlike LPF this is
    // exact per `cubek-sinc-filter`'s convention.
    let client = client();
    let time = 1024;
    let signal = sine(2_000.0, time, SR);

    let hpf = HighPassFilter::new(client.clone(), 40.0, 200.0, SR, 0.0, 16, dtype_f32());
    let mut rng = TransformRng::new(42);
    let out = <HighPassFilter<Runtime> as Transform<Runtime>>::apply(
        &hpf,
        upload_single(&client, &signal),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let err = signal
        .iter()
        .zip(out_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(err < 1e-5, "hpf p=0 should be identity, got {err}");
}

#[test]
fn hpf_probability_one_attenuates_sub_cutoff_tones() {
    // High-pass with cutoffs in [2000, 4000] Hz should strongly attenuate
    // a 100 Hz tone (well below any sampled cutoff).
    let client = client();
    let time = 4096;
    let signal = sine(100.0, time, SR);

    let hpf = HighPassFilter::new(client.clone(), 2_000.0, 4_000.0, SR, 1.0, 16, dtype_f32());
    let mut rng = TransformRng::new(5);
    let out = <HighPassFilter<Runtime> as Transform<Runtime>>::apply(
        &hpf,
        upload_single(&client, &signal),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let r_in = rms(&signal);
    let r_out = rms(&out_host);
    assert!(
        r_out < r_in * 0.2,
        "hpf should kill 100 Hz tone (in rms={r_in}, out rms={r_out})",
    );
}

#[test]
fn lpf_probability_one_attenuates_above_cutoff_tones() {
    let client = client();
    let time = 4096;
    // A 10 kHz tone, filter range 1-3 kHz. Should be cleanly attenuated.
    let signal = sine(10_000.0, time, SR);

    let lpf = LowPassFilter::new(client.clone(), 1_000.0, 3_000.0, SR, 1.0, 16, dtype_f32());
    let mut rng = TransformRng::new(11);
    let out = <LowPassFilter<Runtime> as Transform<Runtime>>::apply(
        &lpf,
        upload_single(&client, &signal),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let r_in = rms(&signal);
    let r_out = rms(&out_host);
    assert!(
        r_out < r_in * 0.2,
        "lpf should kill 10 kHz tone (in rms={r_in}, out rms={r_out})",
    );
}

#[test]
fn lpf_probability_one_preserves_below_cutoff_tones() {
    let client = client();
    let time = 4096;
    let signal = sine(500.0, time, SR);

    // Cutoffs comfortably above 500 Hz; the sample tone should survive.
    let lpf = LowPassFilter::new(client.clone(), 4_000.0, 8_000.0, SR, 1.0, 16, dtype_f32());
    let mut rng = TransformRng::new(2);
    let out = <LowPassFilter<Runtime> as Transform<Runtime>>::apply(
        &lpf,
        upload_single(&client, &signal),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);
    let r_in = rms(&signal);
    let r_out = rms(&out_host);
    assert!(
        r_out > r_in * 0.9,
        "lpf in-band signal should survive (in rms={r_in}, out rms={r_out})",
    );
}

#[test]
fn per_row_rows_get_independent_cutoffs() {
    // Feed the same tone into multiple rows with p=1.0 and confirm outputs
    // differ from each other — per-row cutoff sampling must actually take
    // effect.
    let client = client();
    let batch = 4;
    let time = 4096;
    let one = sine(6_000.0, time, SR);
    let mut sig = vec![0.0f32; batch * time];
    for b in 0..batch {
        sig[b * time..(b + 1) * time].copy_from_slice(&one);
    }

    let lpf = LowPassFilter::new(client.clone(), 1_000.0, 8_000.0, SR, 1.0, 32, dtype_f32());
    let mut rng = TransformRng::new(99);
    let out = <LowPassFilter<Runtime> as Transform<Runtime>>::apply(
        &lpf,
        upload_2d(&client, &sig, batch, time),
        &mut rng,
    );
    let out_host = read_tensor(&client, out);

    let rms_per_row: Vec<f32> = (0..batch)
        .map(|b| rms(&out_host[b * time..(b + 1) * time]))
        .collect();
    let min = rms_per_row.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = rms_per_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max - min > 0.05,
        "per-row RMS should differ (got {min:.3}..{max:.3})",
    );
}
