//! ESP-DL layout annotations and filter packing for Step 5.

use crate::ir::Tensor;

/// ESP-DL tensor layout annotation strings.
pub mod annotation {
    pub const NCHW: &str = "NCHW";
    pub const NHWC: &str = "NHWC";
    pub const N16HWC16: &str = "(N/16)HWC16";
    pub const N8HWC8: &str = "(N/8)HWC8";
    pub const N16HWC16_UNALIGNED: &str = "(N/16)HWC16_UNALIGNED";
    pub const N8HWC8_UNALIGNED: &str = "(N/8)HWC8_UNALIGNED";
    pub const UNKNOWN: &str = "UNK";
}

/// Integer payload with the logical shape and ESP-DL doc string that
/// must be written into the FlatBuffers tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedTensor<T> {
    pub values: Vec<T>,
    pub shape: Vec<usize>,
    pub annotation: &'static str,
}

/// Return the NHWC shape corresponding to an NCHW activation shape.
pub fn nchw_to_nhwc(shape: [usize; 4]) -> [usize; 4] {
    [shape[0], shape[2], shape[3], shape[1]]
}

/// Rewrite NCHW reduce axes into the current NHWC axis space.
pub fn reduce_axes_nchw_to_nhwc(axes: &[i64], rank: usize) -> Vec<i64> {
    let perm = [0_i64, 2, 3, 1];
    let mut out: Vec<i64> = axes
        .iter()
        .map(|&axis| {
            let positive = if axis < 0 { axis + rank as i64 } else { axis };
            perm.iter()
                .position(|&p| p == positive)
                .expect("axis present in NHWC perm") as i64
        })
        .collect();
    out.sort_unstable();
    out
}

/// Pack a Conv2d filter from Burn/PyTorch `[N,C,H,W]` into ESP-DL's
/// blocked HWCN byte order.
pub fn pack_conv_filter<T: Copy>(values: &[T], shape: &[usize], num_bits: u8) -> PackedTensor<T> {
    assert_eq!(shape.len(), 4, "Conv filter must be rank-4 [N,C,H,W]");
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    assert_eq!(values.len(), n * c * h * w);

    let align = align_for_bits(num_bits);
    let aligned_len = n / align * align;
    let mut packed_nhwc = Vec::with_capacity(values.len());

    for block in 0..(aligned_len / align) {
        for hh in 0..h {
            for ww in 0..w {
                for cc in 0..c {
                    for lane in 0..align {
                        let nn = block * align + lane;
                        packed_nhwc.push(values[nchw_index(nn, cc, hh, ww, c, h, w)]);
                    }
                }
            }
        }
    }

    for nn in aligned_len..n {
        for hh in 0..h {
            for ww in 0..w {
                for cc in 0..c {
                    packed_nhwc.push(values[nchw_index(nn, cc, hh, ww, c, h, w)]);
                }
            }
        }
    }

    let annotation = match (num_bits, n % align == 0) {
        (8, true) => annotation::N16HWC16,
        (8, false) => annotation::N16HWC16_UNALIGNED,
        (16, true) => annotation::N8HWC8,
        (16, false) => annotation::N8HWC8_UNALIGNED,
        _ => panic!("unsupported ESP-DL weight bit width {num_bits}"),
    };

    PackedTensor {
        values: packed_nhwc,
        shape: vec![h, w, c, n],
        annotation,
    }
}

/// Pack a Burn Linear filter from `[in,out]` through esp-ppq's Gemm
/// path: transpose to `[out,in]`, unsqueeze to `[out,in,1,1]`, then
/// use the Conv2d filter packer.
pub fn pack_linear_filter<T: Copy>(values: &[T], shape: &[usize], num_bits: u8) -> PackedTensor<T> {
    assert_eq!(shape.len(), 2, "Linear filter must be rank-2 [in,out]");
    let din = shape[0];
    let dout = shape[1];
    assert_eq!(values.len(), din * dout);

    let mut transposed = Vec::with_capacity(values.len());
    for out in 0..dout {
        for input in 0..din {
            transposed.push(values[input * dout + out]);
        }
    }
    pack_conv_filter(&transposed, &[dout, din, 1, 1], num_bits)
}

/// Pack an already-quantized IR tensor as Conv2d filter data.
pub fn pack_quantized_conv<T: Copy>(
    quantized: Vec<T>,
    tensor: &Tensor,
    num_bits: u8,
) -> PackedTensor<T> {
    pack_conv_filter(&quantized, &tensor.shape, num_bits)
}

/// Pack an already-quantized IR tensor as Linear/Gemm filter data.
pub fn pack_quantized_linear<T: Copy>(
    quantized: Vec<T>,
    tensor: &Tensor,
    num_bits: u8,
) -> PackedTensor<T> {
    pack_linear_filter(&quantized, &tensor.shape, num_bits)
}

fn align_for_bits(num_bits: u8) -> usize {
    match num_bits {
        8 => 16,
        16 => 8,
        _ => panic!("unsupported ESP-DL weight bit width {num_bits}"),
    }
}

fn nchw_index(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    c_len: usize,
    h_len: usize,
    w_len: usize,
) -> usize {
    (((n * c_len + c) * h_len + h) * w_len) + w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_axes_rewrite_matches_nhwc_permutation() {
        assert_eq!(reduce_axes_nchw_to_nhwc(&[2, 3], 4), vec![1, 2]);
        assert_eq!(reduce_axes_nchw_to_nhwc(&[-2, -1], 4), vec![1, 2]);
    }

    #[test]
    fn conv_filter_pack_blocks_output_channels() {
        let values: Vec<i8> = (0..32).map(|v| v as i8).collect();
        let packed = pack_conv_filter(&values, &[16, 2, 1, 1], 8);
        let expected: Vec<i8> = (0..2)
            .flat_map(|c| (0..16).map(move |n| (n * 2 + c) as i8))
            .collect();
        assert_eq!(packed.values, expected);
        assert_eq!(packed.shape, vec![1, 1, 2, 16]);
        assert_eq!(packed.annotation, annotation::N16HWC16);
    }

    #[test]
    fn linear_pack_transposes_before_blocking() {
        let values = vec![1_i8, 2, 3, 4, 5, 6]; // [in=2,out=3]
        let packed = pack_linear_filter(&values, &[2, 3], 8);
        assert_eq!(packed.values, vec![1, 4, 2, 5, 3, 6]);
        assert_eq!(packed.shape, vec![1, 1, 2, 3]);
        assert_eq!(packed.annotation, annotation::N16HWC16_UNALIGNED);
    }
}
