//! Step 5 — ESP-DL weight layout tests.

use burn_espdl_export::layout::{self, annotation};

#[test]
fn conv_int8_aligned_filter_uses_n16hwc16_order() {
    let values: Vec<i8> = (0..32).map(|v| v as i8).collect();
    let packed = layout::pack_conv_filter(&values, &[16, 2, 1, 1], 8);

    let expected: Vec<i8> = (0..2)
        .flat_map(|c| (0..16).map(move |n| (n * 2 + c) as i8))
        .collect();
    assert_eq!(packed.values, expected);
    assert_eq!(packed.shape, vec![1, 1, 2, 16]);
    assert_eq!(packed.annotation, annotation::N16HWC16);
}

#[test]
fn conv_int8_unaligned_suffix_keeps_nhwc_tail() {
    let values: Vec<i8> = (0..36).map(|v| v as i8).collect();
    let packed = layout::pack_conv_filter(&values, &[18, 2, 1, 1], 8);

    let expected: Vec<i8> = (0..2)
        .flat_map(|c| (0..16).map(move |n| (n * 2 + c) as i8))
        .chain((16..18).flat_map(|n| (0..2).map(move |c| (n * 2 + c) as i8)))
        .collect();
    assert_eq!(packed.values, expected);
    assert_eq!(packed.shape, vec![1, 1, 2, 18]);
    assert_eq!(packed.annotation, annotation::N16HWC16_UNALIGNED);
}

#[test]
fn linear_pack_reuses_conv_filter_layout() {
    let values = vec![1_i8, 2, 3, 4, 5, 6]; // [in=2,out=3]
    let packed = layout::pack_linear_filter(&values, &[2, 3], 8);

    assert_eq!(packed.values, vec![1, 4, 2, 5, 3, 6]);
    assert_eq!(packed.shape, vec![1, 1, 2, 3]);
    assert_eq!(packed.annotation, annotation::N16HWC16_UNALIGNED);
}

#[test]
fn int16_uses_eight_channel_blocks() {
    let values: Vec<i16> = (0..16).map(|v| v as i16).collect();
    let packed = layout::pack_conv_filter(&values, &[8, 2, 1, 1], 16);

    let expected: Vec<i16> = (0..2)
        .flat_map(|c| (0..8).map(move |n| (n * 2 + c) as i16))
        .collect();
    assert_eq!(packed.values, expected);
    assert_eq!(packed.shape, vec![1, 1, 2, 8]);
    assert_eq!(packed.annotation, annotation::N8HWC8);
}
