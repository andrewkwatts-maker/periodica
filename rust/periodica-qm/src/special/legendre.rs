//====== periodica/rust/periodica-qm/src/special/legendre.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Normalised associated Legendre functions.
//!
//! ## Convention
//!
//! [`legendre_normalised`] returns
//!
//! ```text
//! Pbar_l^m(x) = sqrt((2l + 1)/2 * (l - m)!/(l + m)!) * P_l^m(x),   0 <= m <= l,
//! P_l^m(x)    = (-1)^m (1 - x^2)^{m/2} d^m/dx^m P_l(x)
//! ```
//!
//! i.e. **with** the Condon-Shortley phase `(-1)^m` (as in DLMF 14.7.10's
//! Ferrers functions and `scipy.special.lpmv`), normalised so that
//! `integral_{-1}^{1} Pbar_l^m(x)^2 dx = 1`. The complex spherical harmonic is
//! then simply `Y_lm = Pbar_l^m(cos theta) e^{i m phi} / sqrt(2 pi)`.
//!
//! ## Why the normalised recurrence
//!
//! The unnormalised `P_l^m` overflow near `m ~ 150` and the `(l-m)!/(l+m)!`
//! normaliser underflows; multiplying the two loses everything. Recurring
//! directly on `Pbar` keeps every intermediate `O(1)`:
//!
//! ```text
//! Pbar_0^0     = 1/sqrt(2)
//! Pbar_m^m     = -sqrt((2m+1)/(2m)) sqrt(1-x^2) Pbar_{m-1}^{m-1}
//! Pbar_{m+1}^m = sqrt(2m+3) x Pbar_m^m
//! Pbar_l^m     = a_lm (x Pbar_{l-1}^m - b_lm Pbar_{l-2}^m),
//!     a_lm = sqrt((4l^2 - 1)/(l^2 - m^2)),
//!     b_lm = sqrt(((l-1)^2 - m^2)/(4(l-1)^2 - 1))
//! ```
//!
//! (Limpanuparb and Milthorpe, arXiv:1410.1748, eqs. 11-13, rescaled to the
//! `[-1, 1]` normalisation.) `sqrt(1 - x^2)` is formed as
//! `sqrt((1 - x)(1 + x))`, which is accurate near the poles.

/// `Pbar_l^m(x)` as defined in the module docs (Condon-Shortley phase
/// included, unit `L^2` norm on `[-1, 1]`).
///
/// Returns `0` when `m > l`, and `NaN` when `|x| > 1` or `x` is `NaN`.
pub fn legendre_normalised(l: u32, m: u32, x: f64) -> f64 {
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    legendre_normalised_cs(l, m, x, libm::sqrt((1.0 - x) * (1.0 + x)))
}

/// [`legendre_normalised`] given both `x = cos(theta)` and `s = sin(theta) >= 0`.
///
/// Callers that know `theta` (or a Cartesian direction) pass `s` directly:
/// recovering it as `sqrt(1 - x^2)` loses half the significant digits near
/// the poles (at `theta = 1e-3`, `1 - x` keeps only ~10 of them).
pub(crate) fn legendre_normalised_cs(l: u32, m: u32, x: f64, s: f64) -> f64 {
    if m > l {
        return 0.0;
    }
    // Pbar_m^m.
    let mut pmm = core::f64::consts::FRAC_1_SQRT_2;
    for i in 1..=m {
        let fi = f64::from(i);
        pmm *= -libm::sqrt((2.0 * fi + 1.0) / (2.0 * fi)) * s;
    }
    if l == m {
        return pmm;
    }
    let mf = f64::from(m);
    // Pbar_{m+1}^m.
    let mut p_prev = pmm;
    let mut p_cur = libm::sqrt(2.0 * mf + 3.0) * x * pmm;
    for ll in (m + 2)..=l {
        let lf = f64::from(ll);
        let a = libm::sqrt((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf));
        let l1 = lf - 1.0;
        let b = libm::sqrt((l1 * l1 - mf * mf) / (4.0 * l1 * l1 - 1.0));
        let p_next = a * (x * p_cur - b * p_prev);
        p_prev = p_cur;
        p_cur = p_next;
    }
    p_cur
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-14 * b.abs().max(1.0)
    }

    #[test]
    fn low_orders_match_closed_forms() {
        for &x in &[-1.0_f64, -0.6, -0.1, 0.0, 0.35, 0.9, 1.0] {
            let s = libm::sqrt((1.0 - x) * (1.0 + x));
            let n = |l: f64, ratio: f64| ((2.0 * l + 1.0) / 2.0 * ratio).sqrt();
            // P_0 = 1; P_1 = x; P_1^1 = -s; P_2 = (3x^2 - 1)/2;
            // P_2^1 = -3 x s; P_2^2 = 3 s^2; P_3^3 = -15 s^3.
            assert!(close(legendre_normalised(0, 0, x), n(0.0, 1.0)));
            assert!(close(legendre_normalised(1, 0, x), n(1.0, 1.0) * x));
            assert!(close(legendre_normalised(1, 1, x), -n(1.0, 0.5) * s));
            assert!(close(
                legendre_normalised(2, 0, x),
                n(2.0, 1.0) * 0.5 * (3.0 * x * x - 1.0)
            ));
            assert!(close(
                legendre_normalised(2, 1, x),
                -n(2.0, 1.0 / 6.0) * 3.0 * x * s
            ));
            assert!(close(
                legendre_normalised(2, 2, x),
                n(2.0, 1.0 / 24.0) * 3.0 * s * s
            ));
            assert!(close(
                legendre_normalised(3, 3, x),
                -n(3.0, 1.0 / 720.0) * 15.0 * s * s * s
            ));
        }
    }

    #[test]
    fn domain_handling() {
        assert_eq!(legendre_normalised(2, 3, 0.5), 0.0);
        assert!(legendre_normalised(2, 1, 1.5).is_nan());
        assert!(legendre_normalised(2, 1, f64::NAN).is_nan());
        // Large degree stays finite and bounded (|Pbar| <= sqrt(l + 1/2)).
        for l in [50, 150, 400] {
            for m in [0, l / 2, l] {
                let v = legendre_normalised(l, m, 0.3);
                assert!(v.is_finite() && v.abs() <= (f64::from(l) + 0.5).sqrt());
            }
        }
    }
}
