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
/// ## Exact results
///
/// The value is defined by one fixed floating-point recipe: digits are taken
/// least significant first, `acc += digit * scale` with `scale` starting at
/// `1 / b` and multiplied by `1 / b` after each digit. Every base returns
/// that recipe's result bit for bit, but the hot bases take faster routes to
/// it:
///
/// - **base 2**, `index < 2^53`: one `reverse_bits`. The digits are bits and
///   every partial sum spans at most 53 bits, so the recipe is exact and
///   equals the reversed word scaled by `2^-64`.
/// - **bases 3 and 5**: the recipe with a compile-time divisor, so each digit
///   costs a multiply and a shift instead of a 64-bit hardware division.
/// - anything else (other bases, base-2 indices `>= 2^53` where the recipe
///   starts to round): the recipe with a runtime divisor.
///
/// # Panics
///
/// Debug builds assert `base >= 2`.
#[inline]
pub fn radical_inverse(base: u32, index: u64) -> f64 {
    debug_assert!(base >= 2, "radical inverse needs base >= 2, got {base}");
    match base {
        2 => radical_inverse_base2(index),
        3 => radical_inverse_const::<3>(index),
        5 => radical_inverse_const::<5>(index),
        _ => radical_inverse_generic(u64::from(base), index),
    }
}

/// Base-2 indices below this have at most 53 significant bits, so their
/// radical inverse is exactly representable (see [`radical_inverse`]).
const BASE2_EXACT_LIMIT: u64 = 1 << 53;

/// Base-2 radical inverse: O(1) below `2^53`, the reference recipe above.
#[inline]
fn radical_inverse_base2(index: u64) -> f64 {
    if index < BASE2_EXACT_LIMIT {
        radical_inverse_base2_exact(index)
    } else {
        radical_inverse_generic(2, index)
    }
}

/// Base-2 radical inverse of `index < 2^53` in O(1).
///
/// Bit `k` of `index` becomes bit `63 - k` of the reversed word, i.e. the
/// term `2^-(k+1)` once scaled by `2^-64`. The reversed word has at most 53
/// significant bits, so the `u64 -> f64` conversion and the power-of-two
/// scaling are both exact, and the result equals the digit-by-digit sum.
#[inline]
fn radical_inverse_base2_exact(index: u64) -> f64 {
    debug_assert!(index < BASE2_EXACT_LIMIT);
    /// `2^-64`, exactly.
    const TWO_POW_MINUS_64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
    index.reverse_bits() as f64 * TWO_POW_MINUS_64
}

/// The reference recipe with the base fixed at compile time. Integer division
/// by a constant compiles to a multiply and a shift; the floating-point
/// operations are the same, in the same order, as [`radical_inverse_generic`].
///
/// (Peeling two digits per division by `B^2` was measured too: it shortens
/// the integer chain but the serial `acc` additions dominate, and it came out
/// about 10% slower.)
#[inline]
fn radical_inverse_const<const B: u64>(index: u64) -> f64 {
    let inv_base = 1.0 / B as f64;
    let mut remaining = index;
    let mut scale = inv_base;
    let mut acc = 0.0_f64;
    // Bounded: one iteration per base-`B` digit, so at most 64.
    while remaining > 0 {
        acc += (remaining % B) as f64 * scale;
        remaining /= B;
        scale *= inv_base;
    }
    acc.min(ONE_BELOW)
}

/// The reference recipe for any base (see [`radical_inverse`]).
fn radical_inverse_generic(base: u64, index: u64) -> f64 {
    // `base` fits in a u32 (see the public signature), so this conversion is
    // exact and `inv_base` matches the specialised paths bit for bit.
    let inv_base = 1.0 / base as f64;
    let mut remaining = index;
    let mut scale = inv_base;
    let mut acc = 0.0_f64;
    // Bounded: one iteration per base-`base` digit, so at most 64.
    while remaining > 0 {
        acc += (remaining % base) as f64 * scale;
        remaining /= base;
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
        // The specialised digit routines for `BASES`, called directly so no
        // base dispatch survives into the per-sample loop.
        [
            rotate(radical_inverse_base2(index), self.shift[0]),
            rotate(radical_inverse_const::<3>(index), self.shift[1]),
            rotate(radical_inverse_const::<5>(index), self.shift[2]),
        ]
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

    // --- Bit-identity of the fast paths ------------------------------------

    /// Indices spanning every regime: small, typical sample counts, the
    /// base-2 exact limit on both sides, and the top of the u64 range.
    const PINNED_INDICES: [u64; 12] = [
        1,
        2,
        7,
        12_345,
        (1 << 20) - 1,
        (1 << 20) + 12_345,
        0x0123_4567_89AB,
        (1 << 53) - 1,
        1 << 53,
        0x0F0F_F0F0_1234_5678,
        u64::MAX - 1,
        u64::MAX,
    ];

    /// `radical_inverse(base, PINNED_INDICES[i]).to_bits()`, recorded from
    /// the original digit-by-digit implementation (runtime `%` and `/`,
    /// before the fast paths existed). Any change here changes every seeded
    /// mass integral.
    const PINNED_BITS: [(u32, [u64; 12]); 3] = [
        (
            2,
            [
                0x3FE0000000000000,
                0x3FD0000000000000,
                0x3FEC000000000000,
                0x3FE3818000000000,
                0x3FEFFFFE00000000,
                0x3FE3818100000000,
                0x3FEAB23CD4589000,
                0x3FEFFFFFFFFFFFFF,
                0x3C90000000000000,
                0x3FBE6A2C480F0FF0,
                0x3FE0000000000000,
                0x3FEFFFFFFFFFFFFF,
            ],
        ),
        (
            3,
            [
                0x3FD5555555555555,
                0x3FE5555555555555,
                0x3FE1C71C71C71C72,
                0x3FCF888D31DA4193,
                0x3FC014FA116D37C0,
                0x3FD9D70294C6C9DC,
                0x3FB7055A0A64FADE,
                0x3FDFC2DDF23C4036,
                0x3FEA8C19A3C8CACA,
                0x3FD2BAA5D7C4819B,
                0x3FEBE1DADC20473A,
                0x3FD4357CD4B2558F,
            ],
        ),
        (
            5,
            [
                0x3FC999999999999A,
                0x3FD999999999999A,
                0x3FDC28F5C28F5C2A,
                0x3FC85AD538AC18FA,
                0x3F9E2E830F026B31,
                0x3FD7D0FB9F61F456,
                0x3FD6B8318552C766,
                0x3FD6BE589102E679,
                0x3FE1C592AEE7D9A3,
                0x3FD0B202A7D51C28,
                0x3FED4F3D8A29C27F,
                0x3FC3F548142C28B2,
            ],
        ),
    ];

    #[test]
    fn radical_inverse_matches_the_recorded_original_bits() {
        for (base, bits) in PINNED_BITS {
            for (&index, &want) in PINNED_INDICES.iter().zip(&bits) {
                let got = radical_inverse(base, index).to_bits();
                assert_eq!(got, want, "base {base} index {index:#x}: {got:#018x}");
            }
        }
    }

    #[test]
    fn fast_paths_equal_the_reference_recipe_bit_for_bit() {
        let check = |index: u64| {
            for base in HaltonSampler::BASES {
                let fast = radical_inverse(base, index);
                let reference = radical_inverse_generic(u64::from(base), index);
                assert_eq!(
                    fast.to_bits(),
                    reference.to_bits(),
                    "base {base} index {index:#x}"
                );
            }
        };
        // Every index a 2^16-sample integration touches ...
        (0..1u64 << 16).for_each(check);
        // ... and pseudo-random indices of every bit length.
        let mut state = 0xC0FFEE;
        for bits in 1..=64u32 {
            for _ in 0..64 {
                check(splitmix64(&mut state) >> (64 - bits));
            }
        }
        // Either side of the base-2 exact limit.
        for d in 0..64 {
            check(BASE2_EXACT_LIMIT - 1 - d);
            check(BASE2_EXACT_LIMIT + d);
        }
    }

    #[test]
    fn seeded_points_match_the_recorded_original_bits() {
        // `HaltonSampler::new(42).point(i)`, recorded before the fast paths.
        let pinned: [(u64, [u64; 3]); 4] = [
            (
                0,
                [0x3FE7BAE644C5FD6D, 0x3FC477F199D93378, 0x3FD1D499D5C4C3E6],
            ),
            (
                1,
                [0x3FCEEB991317F5B0, 0x3FDF914E2241EF11, 0x3FDEA166A29190B3],
            ),
            (
                1000,
                [0x3FEAB2E644C5FD6D, 0x3FE03CC58054D28C, 0x3FD2287CABE8518A],
            ),
            (
                1 << 20,
                [0x3FE7BAE744C5FD6D, 0x3FE3CDE5957C4578, 0x3FE0422769C0DBB3],
            ),
        ];
        let s = HaltonSampler::new(42);
        for (index, want) in pinned {
            assert_eq!(s.point(index).map(f64::to_bits), want, "point {index}");
        }
        // `point` uses the specialised routines directly; it must agree with
        // the public dispatch over `BASES`.
        for index in [0, 5, 77, 1 << 33, u64::MAX] {
            let via_dispatch = [0, 1, 2].map(|d| {
                rotate(
                    radical_inverse(HaltonSampler::BASES[d], index),
                    s.shift()[d],
                )
            });
            assert_eq!(s.point(index), via_dispatch);
        }
    }
}
