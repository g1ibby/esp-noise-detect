//! Batched mixed-ratio resampling.
//!
//! Builds several [`Resampler`] instances, splits the batch into equal
//! groups, runs each group through its resampler, then compares each
//! group's output row-by-row against the CPU reference.

mod common;

use common::{
    client, dtype_f32, max_abs_diff, peak_abs, read_tensor, resample_cpu, synth_reals, upload_2d,
    Runtime,
};
use cubek_resample::Resampler;

#[test]
fn mix_of_three_ratios_matches_per_row_reference() {
    let client = client();
    let dtype = dtype_f32();

    let time = 256usize;
    // Three resamplers emulating a pitch-shift bucket set.
    let configs: [(u32, u32); 3] = [(4, 5), (5, 4), (8, 5)];
    let rows_per_config = 2usize;
    let batch = rows_per_config * configs.len();

    // Synthesize a distinct seeded waveform per batch row.
    let mut signals = vec![0.0f32; batch * time];
    for b in 0..batch {
        let row = synth_reals(time, 1_000 + b as u64);
        signals[b * time..(b + 1) * time].copy_from_slice(&row);
    }

    // Dispatch per bucket: slice the batch, upload, run, read back.
    for (idx, &(old_sr, new_sr)) in configs.iter().enumerate() {
        let b_start = idx * rows_per_config;
        let b_end = b_start + rows_per_config;
        let group_signal = &signals[b_start * time..b_end * time];

        let resampler = Resampler::<Runtime>::new(
            client.clone(),
            old_sr,
            new_sr,
            24,
            0.945,
            dtype,
        );
        let signal_t = upload_2d(&client, group_signal, rows_per_config, time);
        let out_t = resampler.apply(signal_t, None);
        let actual = read_tensor(&client, out_t);

        let expected = resample_cpu(
            group_signal,
            rows_per_config,
            time,
            old_sr,
            new_sr,
            24,
            0.945,
            None,
        );

        assert_eq!(actual.len(), expected.len());
        let err = max_abs_diff(&actual, &expected);
        let peak = peak_abs(&expected);
        eprintln!(
            "[bucket {old_sr}/{new_sr}] rows={rows_per_config} err={err:.3e} peak={peak:.3e}",
        );
        // Same computation on both sides; only f32 ordering can drift.
        assert!(err < 1e-5, "bucket ({old_sr}/{new_sr}) diverged: {err:.3e}");
    }
}

#[test]
fn batched_single_ratio_matches_row_by_row_reference() {
    // Sanity check that the `(batch, time) -> (batch, out_len)` launcher
    // computes each row independently. If rows leaked into one another
    // the single-ratio baseline would fail.
    let client = client();
    let dtype = dtype_f32();

    let batch = 4usize;
    let time = 128usize;
    let mut signals = vec![0.0f32; batch * time];
    for b in 0..batch {
        let row = synth_reals(time, 7 + b as u64);
        signals[b * time..(b + 1) * time].copy_from_slice(&row);
    }
    let signal_t = upload_2d(&client, &signals, batch, time);

    let resampler = Resampler::<Runtime>::new(
        client.clone(),
        3,
        4,
        24,
        0.945,
        dtype,
    );
    let out_t = resampler.apply(signal_t, None);
    let actual = read_tensor(&client, out_t);

    let expected = resample_cpu(&signals, batch, time, 3, 4, 24, 0.945, None);
    let err = max_abs_diff(&actual, &expected);
    eprintln!("[batched 3->4] batch={batch} err={err:.3e}");
    assert!(err < 1e-5, "batched single-ratio diverged: {err:.3e}");
}

#[test]
fn identity_passes_through_unchanged() {
    let client = client();
    let dtype = dtype_f32();

    let time = 64usize;
    let signal = synth_reals(time, 123);
    let signal_t = upload_2d(&client, &signal, 1, time);

    let resampler = Resampler::<Runtime>::new(
        client.clone(),
        44_100,
        44_100,
        24,
        0.945,
        dtype,
    );
    let out_t = resampler.apply(signal_t, None);
    let actual = read_tensor(&client, out_t);

    // Identity resampler returns the input handle unchanged; reading it
    // back must give the original bytes.
    assert_eq!(actual, signal);
}
