#[macro_export]
macro_rules! recover_late_pop {
    // Drains up to `attempts` times and returns (total_drained, recovered_bytes).
    ($transaction:ident, $buf:ident, $attempts:expr) => {{
        let mut drained_local = 0usize;
        let mut recovered_local = 0usize;
        for _ in 0..$attempts {
            match $transaction.pop($buf).await {
                Ok(bytes) => {
                    drained_local += bytes;
                    if bytes > 0 {
                        recovered_local = bytes;
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        (drained_local, recovered_local)
    }};
}
