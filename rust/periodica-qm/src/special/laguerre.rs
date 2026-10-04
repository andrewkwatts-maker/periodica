//====== periodica/rust/periodica-qm/src/special/laguerre.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Generalised Laguerre polynomials `L_k^alpha(x)` (modern convention,
//! `L_k^alpha(0) = C(k + alpha, k)`), and their zeros.

use crate::numeric::solve_bracketed;

/// `L_k^alpha(x)` by the three-term recurrence
/// `(j+1) L_{j+1} = (2j + 1 + alpha - x) L_j - (j + alpha) L_{j-1}`,
/// starting from `L_0 = 1`, `L_1 = 1 + alpha - x`.
///
/// The forward recurrence is stable for Laguerre polynomials (the wanted
/// solution is the dominant one), and costs `O(k)` with no factorials, so it
/// stays accurate where the explicit sum of alternating binomial terms loses
/// every digit. This is the convention of DLMF 18.5.12 and of the
/// hydrogen radial function `R_nl ~ L_{n-l-1}^{2l+1}`.
pub fn laguerre(k: u32, alpha: f64, x: f64) -> f64 {
    laguerre_pair(k, alpha, x).0
}

/// `(L_k^alpha(x), L_{k-1}^alpha(x))`, with `L_{-1} = 0`.
pub(crate) fn laguerre_pair(k: u32, alpha: f64, x: f64) -> (f64, f64) {
    if k == 0 {
        return (1.0, 0.0);
    }
    let mut prev = 1.0;
    let mut cur = 1.0 + alpha - x;
    for j in 1..k {
        let jf = f64::from(j);
        let next = ((2.0 * jf + 1.0 + alpha - x) * cur - (jf + alpha) * prev) / (jf + 1.0);
        prev = cur;
        cur = next;
    }
    (cur, prev)
}

/// `d/dx L_k^alpha(x) = -L_{k-1}^{alpha+1}(x)` (DLMF 18.9.23).
pub fn laguerre_derivative(k: u32, alpha: f64, x: f64) -> f64 {
    if k == 0 {
        0.0
    } else {
        -laguerre(k - 1, alpha + 1.0, x)
    }
}

/// The `k` zeros of `L_k^alpha` (`alpha > -1`), ascending.
///
/// All zeros are real, simple and positive, and the zeros of `L_{j}^alpha`
/// interlace those of `L_{j+1}^alpha`. Each degree's zeros are therefore
/// bracketed by the previous degree's zeros together with `0` and the bound
/// `4j + 2 alpha + 3` (above Szego's bound on the largest zero, Thm 6.31.2),
/// and each bracket is solved with safeguarded Newton. Building up from
/// degree 1 costs `O(k^3)` -- trivial for the `k <= 30` of atomic physics --
/// and, unlike an eigenvalue solve, uses only IEEE `+ - * /`, so the zeros
/// are bit-identical on every platform.
///
/// These are the radial nodes of hydrogen-like orbitals (in `rho`) and the
/// Gauss-Laguerre abscissae for weight `x^alpha e^{-x}`.
///
/// Returns an empty vector for `k = 0` or `alpha <= -1`.
pub fn laguerre_roots(k: u32, alpha: f64) -> Vec<f64> {
    if k == 0 || alpha <= -1.0 || alpha.is_nan() {
        return Vec::new();
    }
    let mut roots = vec![1.0 + alpha];
    for j in 2..=k {
        let upper = 4.0 * f64::from(j) + 2.0 * alpha + 3.0;
        let mut next = Vec::with_capacity(j as usize);
        let mut lo = 0.0;
        for i in 0..j as usize {
            let hi = if i < roots.len() { roots[i] } else { upper };
            let f = |x: f64| {
                (
                    laguerre(j, alpha, x),
                    laguerre_derivative(j, alpha, x),
                )
            };
            next.push(solve_bracketed(f, lo, hi, 0.5 * (lo + hi)));
            lo = hi;
        }
        roots = next;
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Explicit sum `sum_i (-1)^i C(k+alpha, k-i) x^i / i!` for integer
    /// alpha, with the sum of the terms' magnitudes (its condition scale: the
    /// alternating sum cancels, so it is only good to `eps * that`).
    fn explicit(k: u32, alpha: u32, x: f64) -> (f64, f64) {
        let binom = |n: u32, r: u32| -> f64 {
            (0..r).fold(1.0, |acc, t| acc * f64::from(n - t) / f64::from(t + 1))
        };
        let mut sum = 0.0;
        let mut magnitude = 0.0;
        let mut xi_over_ifact = 1.0;
        for i in 0..=k {
            if i > 0 {
                xi_over_ifact *= x / f64::from(i);
            }
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let term = binom(k + alpha, k - i) * xi_over_ifact;
            sum += sign * term;
            magnitude += term;
        }
        (sum, magnitude)
    }

    #[test]
    fn recurrence_matches_explicit_sum() {
        for k in 0..=8 {
            for alpha in 0..=7 {
                for &x in &[0.0, 0.3, 1.0, 2.7, 6.0, 11.0] {
                    let got = laguerre(k, f64::from(alpha), x);
                    let (want, scale) = explicit(k, alpha, x);
                    assert!(
                        (got - want).abs() <= 1e-14 * scale,
                        "L_{k}^{alpha}({x}) = {got} vs {want}"
                    );
                }
            }
        }
        // Low orders by hand.
        assert_eq!(laguerre(1, 1.0, 2.0), 0.0);
        assert!((laguerre(2, 1.0, 3.0 - 3f64.sqrt())).abs() < 1e-14);
    }

    #[test]
    fn derivative_matches_finite_difference() {
        for k in 1..=6 {
            let x = 1.7;
            let h = 1e-6;
            let fd = (laguerre(k, 2.0, x + h) - laguerre(k, 2.0, x - h)) / (2.0 * h);
            assert!((laguerre_derivative(k, 2.0, x) - fd).abs() < 1e-7);
        }
        assert_eq!(laguerre_derivative(0, 1.0, 3.0), 0.0);
    }

    #[test]
    fn roots_are_zeros_and_interlace() {
        for alpha in [0.0, 0.5, 1.0, 5.0, 19.0] {
            let mut prev: Vec<f64> = Vec::new();
            for k in 1..=24 {
                let r = laguerre_roots(k, alpha);
                assert_eq!(r.len(), k as usize);
                for &x in &r {
                    // The sign flips within a relative 1e-13 of each zero.
                    let below = laguerre(k, alpha, x * (1.0 - 1e-13));
                    let above = laguerre(k, alpha, x * (1.0 + 1e-13));
                    assert!(
                        below * above <= 0.0,
                        "k={k} alpha={alpha} x={x}: {below} {above}"
                    );
                }
                assert!(r.windows(2).all(|w| w[0] < w[1]));
                for (i, &p) in prev.iter().enumerate() {
                    assert!(r[i] < p && p < r[i + 1], "interlacing k={k}");
                }
                prev = r;
            }
        }
        // Sum of zeros = k (k + alpha) (trace of the Jacobi matrix).
        let r = laguerre_roots(10, 3.0);
        let s: f64 = r.iter().sum();
        assert!((s - 10.0 * 13.0).abs() < 1e-11, "{s}");
        assert!(laguerre_roots(0, 1.0).is_empty());
        assert!(laguerre_roots(3, -1.0).is_empty());
    }
}
