//====== periodica/rust/periodica-runtime/src/sampling.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # sampling
//!
//! Deterministic low-discrepancy point sets for quasi-Monte-Carlo integration.
//!
//! [`HaltonSampler`] yields the 3-D Halton sequence (radical inverses in the
//! co-prime bases 2, 3 and 5), shifted by a Cranley-Patterson rotation drawn
//! from a seed with [`splitmix64`].
//!
//! ## Why not a pseudo-random generator
//!
//! A pseudo-random sampler's integration error falls as `N^-1/2`. A Halton
//! sequence fills the unit cube far more evenly, so its error falls faster:
//! close to `N^-1` for smooth integrands. The inside-test of an SDF solid is
//! discontinuous at the surface, which caps the gain; measured on a sphere it
//! is about `N^-0.8` (see the `mass` tests), still several times fewer SDF
//! and density evaluations for the same accuracy at practical sample counts.
//!
//! The unshifted sequence is fully deterministic, which would make every seed
//! give the same answer. The Cranley-Patterson rotation adds a seeded offset
//! to every point modulo 1. It keeps the low discrepancy (a torus shift of a
//! uniform point set is still uniform) while making each seed an independent
//! estimate, so callers can still average several seeds to estimate error.

/// One step of the splitmix64 generator (Steele, Lea and Flood, 2014).
///
/// Advances `state` by the 64-bit golden-ratio increment and returns a
/// well-mixed output. The first output is a bijection of the seed, so two
/// different seeds can never produce the same first output.
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map a 64-bit word to `[0, 1)` using its top 53 bits.
///
/// Every output is an exact multiple of `2^-53`, so the result is exactly
/// representable and never rounds up to `1.0`.
#[inline]
pub fn unit_from_bits(bits: u64) -> f64 {
    const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
    (bits >> 11) as f64 * SCALE
}

/// Largest `f64` strictly below 1.
const ONE_BELOW: f64 = 1.0 - f64::EPSILON / 2.0;

/// The van der Corput radical inverse of `index` in `base`.
///
/// Mirrors the base-`base` digits of `index` about the radix point:
/// `index = d0 + d1*b + d2*b^2 + ...` maps to `d0/b + d1/b^2 + d2/b^3 + ...`.
/// The first `b^k` indices therefore land exactly one per stratum of width
/// `b^-k`, which is the source of the sequence's even coverage.
///
/// # Panics
///
/// Debug builds assert `base >= 2`.
pub fn radical_inverse(base: u32, index: u64) -> f64 {
    debug_assert!(base >= 2, "radical inverse needs base >= 2, got {base}");
    let b = u64::from(base);
    let inv_base = 1.0 / f64::from(base);
    let mut remaining = index;
    let mut scale = inv_base;
    let mut acc = 0.0_f64;
    // Bounded: one iteration per base-`b` digit, so at most 64.
    while remaining > 0 {
        acc += (remaining % b) as f64 * scale;
        remaining /= b;
        scale *= inv_base;
    }
    // For astronomically large indices the float sum could round to 1.0;
    // keep the documented half-open range regardless.
    acc.min(ONE_BELOW)
}

/// The 3-D Halton sequence with a seeded Cranley-Patterson rotation.
///
/// Point `i` is `frac(radical_inverse(BASES[d], i) + shift[d])` for each
/// dimension `d`, always inside `[0, 1)^3`. Index 0 is the first point (the
/// rotation keeps it away from the origin corner).
///
/// Same seed, same points, bit for bit. Iterating yields points 0, 1, 2, ...
/// without end; use [`Iterator::take`] or [`HaltonSampler::point`].
#[derive(Debug, Clone, PartialEq)]
pub struct HaltonSampler {
    shift: [f64; 3],
    next_index: u64,
}

impl HaltonSampler {
    /// The co-prime bases of the three dimensions.
    pub const BASES: [u32; 3] = [2, 3, 5];

    /// A sampler whose rotation is drawn from `seed` via [`splitmix64`].
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let shift = [
            unit_from_bits(splitmix64(&mut state)),
            unit_from_bits(splitmix64(&mut state)),
            unit_from_bits(splitmix64(&mut state)),
        ];
        Self::with_shift(shift)
    }

    /// A sampler with an explicit rotation. Each component is reduced into
    /// `[0, 1)`; a non-finite component is treated as 0. `[0.0; 3]` gives the
    /// plain Halton sequence.
    pub fn with_shift(shift: [f64; 3]) -> Self {
        let wrap = |s: f64| {
            if s.is_finite() {
                let f = s.rem_euclid(1.0);
                // `rem_euclid` can return exactly 1.0 for tiny negative input.
                if f < 1.0 {
                    f
                } else {
                    0.0
                }
            } else {
                0.0
            }
        };
        Self {
            shift: shift.map(wrap),
            next_index: 0,
        }
    }

    /// The Cranley-Patterson rotation applied to every point.
    #[inline]
    pub fn shift(&self) -> [f64; 3] {
        self.shift
    }

    /// Point `index` of the rotated sequence, in `[0, 1)^3`.
    ///
    /// Random access: does not touch the iterator position.
    #[inline]
    pub fn point(&self, index: u64) -> [f64; 3] {
        [0, 1, 2].map(|d| rotate(radical_inverse(Self::BASES[d], index), self.shift[d]))
    }
}

impl Iterator for HaltonSampler {
    type Item = [f64; 3];

    #[inline]
    fn next(&mut self) -> Option<[f64; 3]> {
        let p = self.point(self.next_index);
        self.next_index = self.next_index.wrapping_add(1);
        Some(p)
    }
}

/// `frac(u + s)` for `u, s` in `[0, 1)`, without a floor call.
#[inline]
fn rotate(u: f64, s: f64) -> f64 {
    let v = u + s;
    if v >= 1.0 {
        v - 1.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_matches_hand_values() {
        // Base 2: 1 -> .1b, 2 -> .01b, 3 -> .11b, 6 -> .011b.
        assert_eq!(radical_inverse(2, 0), 0.0);
        assert_eq!(radical_inverse(2, 1), 0.5);
        assert_eq!(radical_inverse(2, 2), 0.25);
        assert_eq!(radical_inverse(2, 3), 0.75);
        assert_eq!(radical_inverse(2, 6), 0.375);
        // Base 3: 1 -> 1/3, 2 -> 2/3, 3 = 10 (base 3) -> 1/9.
        assert!((radical_inverse(3, 1) - 1.0 / 3.0).abs() < 1e-15);
        assert!((radical_inverse(3, 2) - 2.0 / 3.0).abs() < 1e-15);
        assert!((radical_inverse(3, 3) - 1.0 / 9.0).abs() < 1e-15);
        // Huge indices stay inside the half-open interval.
        assert!(radical_inverse(5, u64::MAX) < 1.0);
    }

    #[test]
    fn first_power_of_base_indices_stratify_exactly() {
        // The defining property: 2^k points hit each of 2^k strata once.
        let k = 8;
        let n = 1u64 << k;
        let mut hit = vec![false; n as usize];
        for i in 0..n {
            let stratum = (radical_inverse(2, i) * n as f64) as usize;
            assert!(!hit[stratum], "stratum {stratum} hit twice");
            hit[stratum] = true;
        }
        assert!(hit.iter().all(|&h| h));
    }

    #[test]
    fn splitmix64_matches_reference_vector() {
        // Reference outputs for seed 1234567 from the original C code.
        let mut s = 1_234_567_u64;
        assert_eq!(splitmix64(&mut s), 6_457_827_717_110_365_317);
        assert_eq!(splitmix64(&mut s), 3_203_168_211_198_807_973);
    }

    #[test]
    fn same_seed_is_bit_identical_and_different_seed_shifts() {
        let a: Vec<[f64; 3]> = HaltonSampler::new(42).take(64).collect();
        let b: Vec<[f64; 3]> = HaltonSampler::new(42).take(64).collect();
        assert_eq!(a, b);

        let s42 = HaltonSampler::new(42).shift();
        let s43 = HaltonSampler::new(43).shift();
        assert_ne!(s42, s43);
        assert_ne!(HaltonSampler::new(0).shift(), [0.0; 3]);
    }

    #[test]
    fn points_stay_in_the_unit_cube() {
        for seed in [0, 1, 42, u64::MAX] {
            for p in HaltonSampler::new(seed).take(2048) {
                assert!(p.iter().all(|&c| (0.0..1.0).contains(&c)), "{p:?}");
            }
        }
    }

    #[test]
    fn random_access_matches_iteration() {
        let s = HaltonSampler::new(9);
        for (i, p) in s.clone().take(100).enumerate() {
            assert_eq!(p, s.point(i as u64));
        }
    }

    #[test]
    fn zero_shift_is_the_plain_sequence() {
        let s = HaltonSampler::with_shift([0.0; 3]);
        assert_eq!(s.point(1), [0.5, 1.0 / 3.0, 0.2]);
        // Shift components are reduced into [0, 1).
        let w = HaltonSampler::with_shift([1.25, -0.25, f64::NAN]);
        assert_eq!(w.shift(), [0.25, 0.75, 0.0]);
    }
}
