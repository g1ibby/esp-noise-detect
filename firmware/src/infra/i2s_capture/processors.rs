use alloc::vec;
use alloc::vec::Vec;

#[inline]
fn extract_sample_from_word(word: u32) -> i16 {
    // Extract the upper 16 bits from the 32-bit word. On typical PDM->I2S paths
    // configured as Philips+Data32Channel32, the valid 16 bits reside in the MSBs.
    // This yields a mono stream. Adjust here if channel selection changes.
    ((word as i32) >> 16) as i16
}

#[inline]
fn extract_24bit_sample_from_word(word: u32) -> i32 {
    // Extract 24-bit signed sample from 32-bit I2S word
    // The microphone delivers 24 bits of usable data (MSB-aligned in 32-bit slot)
    // Right shift by 8 to move the 24-bit value into the low bits, preserving sign.
    (word as i32) >> 8
}

/// Audio frame processor abstraction over sample type `S`.
pub trait Processor<S> {
    fn process_word(&mut self, word: u32, out: &mut Vec<S>);
    fn process_block(&mut self, block: &[u8], out: &mut Vec<S>) -> usize {
        let before = out.len();
        for w in block.chunks_exact(4) {
            let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            self.process_word(word, out);
        }
        out.len() - before
    }
    #[allow(dead_code)]
    fn reset(&mut self) {}
}

/// Default stereo processor: push one 16-bit sample per 32-bit word.
///
/// With ICS43434 x2, words alternate L,R,L,R… so this preserves stereo interleaving.
pub struct StereoHigh16Processor;

impl Processor<i16> for StereoHigh16Processor {
    #[inline]
    fn process_word(&mut self, word: u32, out: &mut Vec<i16>) {
        out.push(extract_sample_from_word(word));
    }
}

/// Stereo processor for 24-bit samples: push one 24-bit sample (in i32) per 32-bit word.
///
/// Extracts the full 24-bit precision from the microphone data.
/// With ICS43434 x2, words alternate L,R,L,R… so this preserves stereo interleaving.
pub struct StereoHigh24Processor;

impl Processor<i32> for StereoHigh24Processor {
    #[inline]
    fn process_word(&mut self, word: u32, out: &mut Vec<i32>) {
        out.push(extract_24bit_sample_from_word(word));
    }
}

/// Mono beamformer (integer delay-and-sum) with one-pole HPF/LPF and fixed gain.
///
/// Why these stages exist:
/// - Delay-and-sum aligns the two mics for a target look direction and sums them,
///   improving SNR by ~3–6 dB at mid/high frequencies (useful at several meters).
/// - HPF (~60–80 Hz) removes DC/rumble and wind/handling noise.
/// - Optional LPF (~10–12 kHz) band-limits and removes ultrasonic junk.
/// - Fixed digital gain lifts toward full-scale for better quantization without AGC.
/// - Keep 24-bit precision in an i32 container to preserve valid LSBs.
pub struct MonoBeamformerProcessor {
    // Ring buffers for recent samples per channel (24-bit in i32)
    ring_l: Vec<i32>,
    ring_r: Vec<i32>,
    ring_size: usize,
    write_idx: usize,

    // Track word parity to pair L and R (0 = expect L, 1 = expect R)
    lr_parity: u8,

    // Integer delay in samples for delay-and-sum: R relative to L.
    //  d > 0  => delay R by d (use older R)
    //  d < 0  => advance R by |d| (equivalently delay L by |d|)
    delay_samples: isize,

    // One-pole HPF state: y[n] = a*(y[n-1] + x[n] - x[n-1])
    hpf_a: f32,
    hpf_x_prev: f32,
    hpf_y_prev: f32,

    // Optional one-pole LPF: y[n] = b*y[n-1] + (1-b)*x[n]
    lpf_enabled: bool,
    lpf_b: f32,
    lpf_y_prev: f32,

    // Fixed digital gain
    gain: f32,
}

impl MonoBeamformerProcessor {
    pub fn new(
        ring_capacity_samples: usize,
        fs_hz: f32,
        hpf_fc_hz: f32,
        lpf_enabled: bool,
        lpf_fc_hz: f32,
        delay_samples: isize,
        gain: f32,
    ) -> Self {
        let ring_size = ring_capacity_samples.max(16);

        // Discrete-time constants
        let ts = 1.0 / fs_hz;

        // HPF coefficient for y = a*(y + x - x_prev)
        let tau_hpf = 1.0 / (2.0 * core::f32::consts::PI * hpf_fc_hz);
        let hpf_a = tau_hpf / (tau_hpf + ts);

        // LPF coefficient for y = b*y + (1-b)*x
        let (lpf_b, lpf_enabled) = if lpf_enabled {
            let tau_lpf = 1.0 / (2.0 * core::f32::consts::PI * lpf_fc_hz);
            (tau_lpf / (tau_lpf + ts), true)
        } else {
            (0.0, false)
        };

        Self {
            ring_l: vec![0; ring_size],
            ring_r: vec![0; ring_size],
            ring_size,
            write_idx: 0,
            lr_parity: 0,
            delay_samples,
            hpf_a,
            hpf_x_prev: 0.0,
            hpf_y_prev: 0.0,
            lpf_enabled,
            lpf_b,
            lpf_y_prev: 0.0,
            gain,
        }
    }

    #[inline]
    fn wrap_idx(&self, idx: usize, offset: isize) -> usize {
        // Safe wrapping index for negative offsets
        let n = self.ring_size as isize;
        (((idx as isize + offset) % n) + n) as usize % self.ring_size
    }
}

impl Processor<i32> for MonoBeamformerProcessor {
    #[inline]
    fn process_word(&mut self, word: u32, out: &mut Vec<i32>) {
        let s24 = extract_24bit_sample_from_word(word);

        let idx = self.write_idx;
        if self.lr_parity == 0 {
            // Left sample
            self.ring_l[idx] = s24;
            self.lr_parity = 1;
            return;
        }

        // Right sample completes the (L,R) pair for time index `idx`.
        self.ring_r[idx] = s24;
        self.lr_parity = 0;

        // Compute delayed indices according to delay_samples meaning (R relative to L)
        let d = self.delay_samples;
        let (idx_l, idx_r) = if d >= 0 {
            (idx, self.wrap_idx(idx, -d)) // delay R by d => use older R
        } else {
            (self.wrap_idx(idx, d), idx) // advance R by |d| => delay L by |d|
        };

        let s_l = self.ring_l[idx_l] as i64;
        let s_r = self.ring_r[idx_r] as i64;

        // Integer sum then average to keep level similar to single channel
        let mut mono = ((s_l + s_r) / 2) as f32;

        // High-pass filter to remove DC/rumble
        let x = mono;
        let y_hpf = self.hpf_a * (self.hpf_y_prev + x - self.hpf_x_prev);
        self.hpf_x_prev = x;
        self.hpf_y_prev = y_hpf;
        mono = y_hpf;

        // Optional low-pass filter for band-limiting
        if self.lpf_enabled {
            let y_lpf = self.lpf_b * self.lpf_y_prev + (1.0 - self.lpf_b) * mono;
            self.lpf_y_prev = y_lpf;
            mono = y_lpf;
        }

        // Fixed gain and 24-bit clipping
        mono *= self.gain;
        const MAX_24: f32 = 8_388_607.0;
        const MIN_24: f32 = -8_388_608.0;
        if mono > MAX_24 {
            mono = MAX_24;
        } else if mono < MIN_24 {
            mono = MIN_24;
        }

        out.push(mono as i32);

        // Advance ring index after forming the mono sample for this pair
        self.write_idx = (self.write_idx + 1) % self.ring_size;
    }

    fn reset(&mut self) {
        self.write_idx = 0;
        self.lr_parity = 0;
        self.hpf_x_prev = 0.0;
        self.hpf_y_prev = 0.0;
        self.lpf_y_prev = 0.0;
        self.ring_l.fill(0);
        self.ring_r.fill(0);
    }
}
