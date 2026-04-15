use core::sync::atomic::{AtomicUsize, Ordering};

use embassy_time::Instant;

static ENQUEUED: AtomicUsize = AtomicUsize::new(0);
static DEQUEUED: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn inc_enq(delta: usize) {
    ENQUEUED.fetch_add(delta, Ordering::Relaxed);
}

#[inline]
pub fn inc_deq(delta: usize) {
    DEQUEUED.fetch_add(delta, Ordering::Relaxed);
}

#[inline]
pub fn backlog() -> usize {
    let enq = ENQUEUED.load(Ordering::Relaxed);
    let deq = DEQUEUED.load(Ordering::Relaxed);
    enq.saturating_sub(deq)
}

#[inline]
pub fn now_ms() -> u64 {
    Instant::now().as_millis() as u64
}
