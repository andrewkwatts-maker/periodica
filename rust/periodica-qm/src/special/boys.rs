//====== periodica/rust/periodica-qm/src/special/boys.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! The Boys function `F_m(T) = integral_0^1 t^{2m} e^{-T t^2} dt`, the kernel
//! of every Coulomb integral over Gaussian basis functions.
//!
//! ## Method
//!
//! * **Small and moderate `T`** (`T < 30`, or `T` below the highest order
//!   wanted): the series (Helgaker, Jorgensen and Olsen, eq. 9.8.12)
//!
//!   ```text
//!   F_M(T) = e^{-T} sum_{k>=0} (2T)^k / ((2M+1)(2M+3)...(2M+2k+1))
//!   ```
//!
//!   whose terms are all positive, evaluated once at the highest order `M`,
//!   then the **downward** recursion `F_m = (2T F_{m+1} + e^{-T})/(2m+1)`,
//!   which is unconditionally stable (it adds positive quantities).
//! * **Large `T`** (`T >= 30`): `F_0(T) = sqrt(pi/T) erf(sqrt T) / 2` -- the
//!   asymptotic form `sqrt(pi/T)/2` with the exact `erf` factor kept, which
//!   costs nothing -- then the **upward** recursion
//!   `F_{m+1} = ((2m+1) F_m - e^{-T}) / (2T)`, stable while `m < T` because
//!   each step multiplies the inherited error by `(2m+1)/(2T) < 1`.
//!
//! Accuracy against 128-point Gauss-Legendre quadrature over `T in [0, 100]`,
//! `m = 0..16`, is gated at `1e-14` absolute in `tests/special_accuracy.rs`.

/// Below this `T` the series is used.
const SERIES_T_MAX: f64 = 30.0;
/// Series terms are positive with ratio `2T/(2M+2k+3)`, so convergence takes
/// about `T + 40` terms; this cap is never reached for `T < 30` or `M >= T`.
const MAX_SERIES_TERMS: usize = 10_000;

/// `sqrt(pi) / 2`.
const HALF_SQRT_PI: f64 = 0.886_226_925_452_758;

/// `F_m(T)` for a single order. `NaN` for `T < 0` or non-finite `T`.
pub fn boys(m: u32, t: f64) -> f64 {
    if !t.is_finite() || t < 0.0 {
        return f64::NAN;
    }
    if use_series(m, t) {
        series(m, t)
    } else {
        let mut f = f0_large(t);
        let et = libm::exp(-t);
        for k in 0..m {
            f = (f64::from(2 * k + 1) * f - et) / (2.0 * t);
        }
        f
    }
}

/// Fill `out[m] = F_m(T)` for `m = 0..out.len()`. Cheaper than repeated
/// [`boys`] calls: one series or one `erf`, then a recursion.
/// Fills with `NaN` for `T < 0` or non-finite `T`.
pub fn boys_array(t: f64, out: &mut [f64]) {
    let Some(m_max) = out.len().checked_sub(1) else {
        return;
    };
    if !t.is_finite() || t < 0.0 {
        out.fill(f64::NAN);
        return;
    }
    let et = libm::exp(-t);
    let m_max_u = u32::try_from(m_max).unwrap_or(u32::MAX);
    if use_series(m_max_u, t) {
        out[m_max] = series(m_max_u, t);
        for m in (0..m_max).rev() {
            out[m] = (2.0 * t * out[m + 1] + et) / (2 * m + 1) as f64;
        }
    } else {
        out[0] = f0_large(t);
        for m in 0..m_max {
            out[m + 1] = ((2 * m + 1) as f64 * out[m] - et) / (2.0 * t);
        }
    }
}

/// The series is used for small `T`, and also whenever the highest order is
/// at least `T`, where upward recursion would amplify rounding.
#[inline]
fn use_series(m_max: u32, t: f64) -> bool {
    t < SERIES_T_MAX || f64::from(m_max) + 0.5 >= t
}

/// `F_M(T)` by its positive-term series.
fn series(m: u32, t: f64) -> f64 {
    let mut denom = f64::from(2 * m + 1);
    let mut term = 1.0 / denom;
    let mut sum = term;
    let two_t = 2.0 * t;
    for _ in 0..MAX_SERIES_TERMS {
        denom += 2.0;
        term *= two_t / denom;
        sum += term;
        if term <= sum * (0.5 * f64::EPSILON) {
            break;
        }
    }
    libm::exp(-t) * sum
}

/// `F_0(T) = sqrt(pi/T) erf(sqrt T) / 2` for `T > 0`.
#[inline]
fn f0_large(t: f64) -> f64 {
    let st = libm::sqrt(t);
    HALF_SQRT_PI * libm::erf(st) / st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // F_m(0) = 1/(2m+1).
        for m in 0..20 {
            assert_eq!(boys(m, 0.0), 1.0 / f64::from(2 * m + 1));
        }
        // F_0(T) = sqrt(pi/T) erf(sqrt T)/2 at T = 1: 0.746824132812427.
        assert!((boys(0, 1.0) - 0.746_824_132_812_427).abs() < 1e-15);
        // Both branches agree across the switch.
        for m in 0..8 {
            let a = series(m, 30.0);
            let b = boys(m, 30.0);
            assert!((a - b).abs() <= 1e-15 * a, "m={m}: {a} vs {b}");
        }
        assert!(boys(0, -1.0).is_nan() && boys(0, f64::NAN).is_nan());
        assert!(boys(0, f64::INFINITY).is_nan());
    }

    #[test]
    fn array_matches_single_orders() {
        for &t in &[0.0, 1e-9, 0.3, 5.0, 29.9, 30.0, 45.0, 100.0, 1e4] {
            let mut out = [0.0; 25];
            boys_array(t, &mut out);
            for (m, &v) in out.iter().enumerate() {
                let single = boys(m as u32, t);
                assert!(
                    (v - single).abs() <= 1e-14 * single.max(1e-300),
                    "T={t} m={m}: {v} vs {single}"
                );
            }
        }
        let mut empty: [f64; 0] = [];
        boys_array(1.0, &mut empty);
        let mut bad = [0.0; 3];
        boys_array(-2.0, &mut bad);
        assert!(bad.iter().all(|v| v.is_nan()));
    }
}
