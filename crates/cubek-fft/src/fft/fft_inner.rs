//! Shared FFT mode tag. The actual butterfly implementation lives in
//! [`super::fft_parallel`]; this module just hosts the enum it takes.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Whether doing RFFT or IRFFT.
pub enum FftMode {
    Forward,
    Inverse,
}

impl FftMode {
    pub fn sign(&self) -> f32 {
        match self {
            FftMode::Forward => -1.,
            FftMode::Inverse => 1.,
        }
    }
}
