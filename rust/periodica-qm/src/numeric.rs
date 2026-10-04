//====== periodica/rust/periodica-qm/src/numeric.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Crate-private numerical kernels: a safeguarded Newton root finder and
//! integer powers by repeated multiplication.
//!
//! Everything here uses only IEEE-754 `+ - * /` and `sqrt`, which are
//! correctly rounded on every platform, so results are bit-identical across
//! operating systems.

/// Hard cap on root-finder iterations. Bisection alone needs at most ~1100
/// halvings to exhaust an `f64` interval; Newton converges in a handful.
const MAX_ROOT_ITERATIONS: usize = 1200;

/// Find the root of `f` inside `[lo, hi]`, where `f(lo)` and `f(hi)` differ
/// in sign (or one is zero). `f` returns `(value, derivative)`.
///
/// Newton steps are taken while they stay inside the shrinking bracket and
/// reduce the step size fast enough; otherwise the bracket is bisected
/// (Numerical Recipes' `rtsafe`). Converges to the last representable bits
/// of the root: it stops when an update no longer moves `x` or the step falls
/// below `2 eps |x|`. A derivative of zero (a turning point of a CDF at a
/// node, say) simply forces bisection.
///
/// If the endpoints do not bracket a root the result is the endpoint with the
/// smaller `|f|`; callers construct brackets that cannot fail this way.
pub(crate) fn solve_bracketed<F>(mut f: F, lo: f64, hi: f64, x0: f64) -> f64
where
    F: FnMut(f64) -> (f64, f64),
{
    let (flo, _) = f(lo);
    if flo == 0.0 {
        return lo;
    }
    let (fhi, _) = f(hi);
    if fhi == 0.0 {
        return hi;
    }
    if (flo > 0.0) == (fhi > 0.0) {
        return if flo.abs() <= fhi.abs() { lo } else { hi };
    }
    // Orient so that f(xl) < 0 < f(xh).
    let (mut xl, mut xh) = if flo < 0.0 { (lo, hi) } else { (hi, lo) };
    let mut x = if x0 > lo.min(hi) && x0 < lo.max(hi) {
        x0
    } else {
        0.5 * (lo + hi)
    };
    let mut dx_old = (hi - lo).abs();
    let mut dx = dx_old;
    let (mut fx, mut dfx) = f(x);
    for _ in 0..MAX_ROOT_ITERATIONS {
        if fx == 0.0 {
            return x;
        }
        let newton_leaves_bracket = ((x - xh) * dfx - fx) * ((x - xl) * dfx - fx) > 0.0;
        let newton_too_slow = (2.0 * fx).abs() > (dx_old * dfx).abs();
        if newton_leaves_bracket || newton_too_slow || !dfx.is_finite() || dfx == 0.0 {
            dx_old = dx;
            dx = 0.5 * (xh - xl);
            let next = xl + dx;
            if next == xl {
                return next;
            }
            x = next;
        } else {
            dx_old = dx;
            dx = fx / dfx;
            let next = x - dx;
            if next == x {
                return x;
            }
            x = next;
        }
        if dx.abs() <= 2.0 * f64::EPSILON * x.abs() {
            return x;
        }
        (fx, dfx) = f(x);
        if fx < 0.0 {
            xl = x;
        } else {
            xh = x;
        }
    }
    x
}

/// `x^n` by binary exponentiation: a fixed sequence of IEEE multiplies, so
/// the result is identical on every platform (unlike `powi`, whose lowering
/// is target-dependent).
#[inline]
pub(crate) fn pow_u(mut x: f64, mut n: u32) -> f64 {
    let mut acc = 1.0;
    while n > 0 {
        if n & 1 == 1 {
            acc *= x;
        }
        x *= x;
        n >>= 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_finds_sqrt2_to_the_last_bit() {
        let r = solve_bracketed(|x| (x * x - 2.0, 2.0 * x), 0.0, 2.0, 1.0);
        assert!((r - core::f64::consts::SQRT_2).abs() <= f64::EPSILON * 2.0);
    }

    #[test]
    fn newton_survives_a_zero_derivative() {
        // x^3 has a zero derivative at its root; the solver must bisect.
        let r = solve_bracketed(|x| (x * x * x, 3.0 * x * x), -1.0, 2.0, 0.0);
        assert!(r.abs() < 1e-100, "{r}");
        // Decreasing function, start outside the bracket.
        let r = solve_bracketed(|x| (1.0 - x, -1.0), 0.0, 4.0, 10.0);
        assert_eq!(r, 1.0);
        // Root at an endpoint, and a non-bracket.
        assert_eq!(solve_bracketed(|x| (x, 1.0), 0.0, 1.0, 0.5), 0.0);
        assert_eq!(solve_bracketed(|x| (x + 1.0, 1.0), 0.0, 1.0, 0.5), 0.0);
    }

    #[test]
    fn pow_u_is_exact_on_small_integers() {
        assert_eq!(pow_u(3.0, 0), 1.0);
        assert_eq!(pow_u(3.0, 5), 243.0);
        assert_eq!(pow_u(-2.0, 7), -128.0);
        assert_eq!(pow_u(0.5, 10), 1.0 / 1024.0);
    }
}
