//! `get_fast_shifts` — host-side enumeration of integer-ratio pitch shifts.
//!
//! Enumerates the finite set of ratios that can be resampled with a small
//! `(old_sr, new_sr)` pair derived from a sample rate's prime
//! factorization. Returns `(num, den)` pairs of `u32`, reduced by GCD.
//! Duplicates are suppressed via `BTreeSet`, giving deterministic order.
//!
//! ## Default `condition`
//!
//! `|ratio| 0.5 <= ratio <= 2.0 && ratio != 1`, i.e. the interval
//! `[-12, +12]` semitones excluding unison. Callers pass a different
//! closure to narrow the range.

use core::cmp::Ordering;

/// Default shift condition: one octave down to one octave up, excluding
/// unison. `num / den` is the resampling ratio (old_sr / new_sr).
pub fn default_condition(num: u32, den: u32) -> bool {
    // 0.5 <= num/den <= 2.0 && num != den
    // Expressed in integer arithmetic to avoid the f32 0.5 comparison
    // being off-by-one at the boundary.
    num != den && num * 2 >= den && num <= den * 2
}

/// Enumerate fast pitch-shift ratios for `sample_rate`.
///
/// Returns a sorted, de-duplicated list of `(num, den)` pairs (GCD-
/// reduced) where `ratio = num/den` satisfies `condition`. The list is
/// sorted by `num*den` ascending then lexicographically — small
/// denominators first, giving the cheapest resamplers near the head.
///
/// Enumerates all products of subsets of the prime factorization of
/// `sample_rate`, then cross-pairs those products to form candidate
/// ratios.
pub fn get_fast_shifts(
    sample_rate: u32,
    condition: impl Fn(u32, u32) -> bool,
) -> Vec<(u32, u32)> {
    assert!(sample_rate > 0, "sample_rate must be > 0");

    let factors = prime_factors(sample_rate);
    let products = distinct_subset_products(&factors);

    let mut seen = std::collections::BTreeSet::new();
    for &num in &products {
        for &den in &products {
            let g = gcd(num, den);
            let n = num / g;
            let d = den / g;
            if condition(n, d) {
                seen.insert((n, d));
            }
        }
    }

    let mut out: Vec<(u32, u32)> = seen.into_iter().collect();
    // Cheap-resampler-first ordering: smallest (num * den) wins. Ties
    // broken by num ascending so the output is deterministic.
    out.sort_by(|a, b| {
        let ka = a.0 as u64 * a.1 as u64;
        let kb = b.0 as u64 * b.1 as u64;
        match ka.cmp(&kb) {
            Ordering::Equal => a.cmp(b),
            o => o,
        }
    });
    out
}

/// Return the prime factorization of `n` as a Vec of primes with
/// repetition. For `n = 32_000 = 2^8 * 5^3`, returns
/// `[2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5]`.
fn prime_factors(mut n: u32) -> Vec<u32> {
    assert!(n > 0);
    let mut out = Vec::new();
    let mut p: u32 = 2;
    while p as u64 * p as u64 <= n as u64 {
        while n % p == 0 {
            out.push(p);
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Distinct products of every non-empty subset of `factors`
/// (multiset-aware). Enumerates subset sizes `1..=len`, collects
/// multiset combinations without repetition, takes their product, and
/// deduplicates.
fn distinct_subset_products(factors: &[u32]) -> Vec<u32> {
    let mut seen = std::collections::BTreeSet::new();
    seen.insert(1u32); // The empty subset's product — included so
                      // `Fraction(1, prod)` candidates get formed.

    // The factors can repeat. Multiset combinations of size `r` are
    // tricky to enumerate directly, but because we just need the set of
    // products, a simple DP over factor groups suffices: treat each
    // distinct prime power count as a dimension.
    //
    // Group factors by their distinct value:
    let mut grouped: std::collections::BTreeMap<u32, u32> =
        std::collections::BTreeMap::new();
    for &f in factors {
        *grouped.entry(f).or_insert(0) += 1;
    }

    // DP: start with {1}, then for each (prime, count), multiply by
    // prime^0, prime^1, ..., prime^count.
    let mut products = vec![1u32];
    for (prime, count) in grouped {
        let mut next = Vec::with_capacity(products.len() * (count as usize + 1));
        for &p in &products {
            let mut mult = 1u32;
            for _ in 0..=count {
                next.push(p * mult);
                if mult > u32::MAX / prime {
                    break;
                }
                mult *= prime;
            }
        }
        products = next;
    }

    for p in products {
        seen.insert(p);
    }
    // Including 1 keeps valid ratios like 1/2 and 2/1. The default
    // condition still filters out 1/1 (unison).
    seen.into_iter().collect()
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Find the `(num, den)` pair in `fast_shifts` closest to
/// `target_ratio`.
pub fn nearest_shift(fast_shifts: &[(u32, u32)], target_ratio: f32) -> (u32, u32) {
    assert!(!fast_shifts.is_empty(), "fast_shifts must be non-empty");
    fast_shifts
        .iter()
        .min_by(|a, b| {
            let da = ((a.0 as f32 / a.1 as f32) - target_ratio).abs();
            let db = ((b.0 as f32 / b.1 as f32) - target_ratio).abs();
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        })
        .copied()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_factors_small() {
        assert_eq!(prime_factors(1), Vec::<u32>::new());
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(12), vec![2, 2, 3]);
        assert_eq!(prime_factors(32_000), {
            let mut v = vec![2; 8];
            v.extend_from_slice(&[5, 5, 5]);
            v
        });
    }

    #[test]
    fn distinct_products_of_12() {
        // 12 = 2*2*3. Subsets (with multiset): {}, {2}, {2,2}, {3},
        // {2,3}, {2,2,3} → products 1, 2, 4, 3, 6, 12.
        let products = distinct_subset_products(&prime_factors(12));
        let mut sorted: Vec<u32> = products;
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 6, 12]);
    }

    #[test]
    fn default_condition_excludes_unison_and_out_of_range() {
        assert!(default_condition(2, 3)); // 0.667
        assert!(default_condition(1, 2)); // 0.5 exactly
        assert!(default_condition(2, 1)); // 2.0 exactly
        assert!(!default_condition(1, 1)); // unison excluded
        assert!(!default_condition(1, 3)); // 0.333 < 0.5
        assert!(!default_condition(3, 1)); // 3 > 2
    }

    #[test]
    fn fast_shifts_32k_contains_common_ratios() {
        let shifts = get_fast_shifts(32_000, default_condition);
        // 32000 = 2^8 * 5^3 → plenty of small ratios should appear.
        assert!(shifts.contains(&(1, 2)));
        assert!(shifts.contains(&(2, 1)));
        assert!(shifts.contains(&(4, 5)));
        assert!(shifts.contains(&(5, 4)));
        assert!(shifts.contains(&(5, 8)));
        assert!(shifts.contains(&(8, 5)));
        assert!(!shifts.contains(&(1, 1)));
        // Deterministic ordering.
        let first = shifts[0];
        assert!(first.0 * first.1 <= shifts.last().unwrap().0 * shifts.last().unwrap().1);
    }

    #[test]
    fn nearest_shift_picks_closest() {
        let shifts = vec![(4u32, 5), (5, 4), (1, 2), (2, 1)];
        // 2^(2/12) ≈ 1.1225 → closest is 5/4 = 1.25
        let r = 2f32.powf(2.0 / 12.0);
        assert_eq!(nearest_shift(&shifts, r), (5, 4));
        // 2^(-2/12) ≈ 0.8908 → closest is 4/5 = 0.8
        assert_eq!(nearest_shift(&shifts, 2f32.powf(-2.0 / 12.0)), (4, 5));
    }
}
