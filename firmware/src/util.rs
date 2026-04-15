//! Utilities shared across firmware modules.
//!
//! Contains small helpers that improve readability without changing logic.

use core::alloc::Layout;

/// Initialize and return a reference to a value in a private StaticCell.
///
/// This macro is useful to create `'static` references to runtime values
/// without sprinkling explicit `StaticCell` declarations and names.
///
/// Example:
/// let (stack, runner) = embassy_net::new(...);
/// let stack = mk_static!(embassy_net::Stack<'static>, stack);
#[macro_export]
macro_rules! mk_static {
    ($t:ty, $val:expr) => {{
        static CELL: static_cell::StaticCell<$t> = static_cell::StaticCell::new();
        CELL.init($val)
    }};
}

pub fn alloc_external_zeroed_aligned<T>(alignment: usize) -> &'static mut T {
    let size = core::mem::size_of::<T>();
    let align = core::cmp::max(alignment, core::mem::align_of::<T>());
    let layout = Layout::from_size_align(size, align).expect("alloc layout");

    unsafe {
        let ptr = esp_alloc::HEAP.alloc_caps(esp_alloc::MemoryCapability::External.into(), layout);
        if ptr.is_null() {
            panic!("alloc_external_zeroed_aligned: OOM");
        }
        core::ptr::write_bytes(ptr, 0, size);
        &mut *(ptr as *mut T)
    }
}

/// Collect an exact-size iterator into a `Vec` backed by PSRAM.
///
/// The returned Vec is fully compatible with the global allocator (drop works
/// correctly) because `esp_alloc::HEAP` manages both internal and PSRAM regions.
/// This avoids audio data competing with WiFi for internal SRAM.
pub fn collect_to_external_vec<T>(iter: impl ExactSizeIterator<Item = T>) -> alloc::vec::Vec<T> {
    let len = iter.len();
    let byte_len = len * core::mem::size_of::<T>();
    if byte_len == 0 {
        return alloc::vec::Vec::new();
    }
    let layout =
        Layout::from_size_align(byte_len, core::mem::align_of::<T>()).expect("vec layout");

    unsafe {
        let ptr = esp_alloc::HEAP.alloc_caps(esp_alloc::MemoryCapability::External.into(), layout);
        if ptr.is_null() {
            panic!("PSRAM alloc failed for audio Vec ({} bytes)", byte_len);
        }
        let dst = ptr as *mut T;
        for (i, item) in iter.enumerate() {
            dst.add(i).write(item);
        }
        alloc::vec::Vec::from_raw_parts(dst, len, len)
    }
}

pub fn alloc_external_slice_zeroed_aligned(len: usize, alignment: usize) -> &'static mut [u8] {
    let layout = Layout::from_size_align(len, alignment).expect("alloc layout");

    unsafe {
        let ptr = esp_alloc::HEAP.alloc_caps(esp_alloc::MemoryCapability::External.into(), layout);
        if ptr.is_null() {
            panic!("alloc_external_slice_zeroed_aligned: OOM");
        }
        core::ptr::write_bytes(ptr, 0, len);
        core::slice::from_raw_parts_mut(ptr, len)
    }
}

/// Allocate from internal heap (dram2_seg) with alignment, zeroed.
/// This uses the bootloader-reclaimed memory region, separate from dram_seg statics/BSS.
pub fn alloc_internal_zeroed_aligned<T>(alignment: usize) -> &'static mut T {
    let size = core::mem::size_of::<T>();
    let align = core::cmp::max(alignment, core::mem::align_of::<T>());
    let layout = Layout::from_size_align(size, align).expect("alloc layout");

    unsafe {
        let ptr = esp_alloc::HEAP.alloc_caps(esp_alloc::MemoryCapability::Internal.into(), layout);
        if ptr.is_null() {
            panic!("alloc_internal_zeroed_aligned: OOM ({} bytes)", size);
        }
        core::ptr::write_bytes(ptr, 0, size);
        &mut *(ptr as *mut T)
    }
}
