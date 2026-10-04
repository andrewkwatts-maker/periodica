//====== periodica/rust/periodica-qm/src/quadrature.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # quadrature
//!
//! Gauss-Legendre rules, used by the samplers to evaluate the polynomial CDF
//! of `P_l^m(cos theta)^2` exactly (an `n`-point rule integrates every
//! polynomial of degree `<= 2n - 1` without error).

use core::f64::consts::PI;

/// Newton iteration cap for a Legendre node; convergence from the
/// Tricomi-style initial guess takes 3-6 steps.
const MAX_NODE_ITERATIONS: usize = 100;

/// `(P_n(x), P_n'(x))` by the three-term recurrence.
fn legendre_with_derivative(n: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = x;
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    // P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1); nodes are interior so x^2 < 1.
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// The `n`-point Gauss-Legendre rule on `[-1, 1]`: `(nodes, weights)`, nodes
/// ascending. Exact for polynomials of degree `<= 2n - 1`.
///
/// Nodes are found by Newton's method from `cos(pi (i - 1/4) / (n + 1/2))`,
/// and the rule is symmetrised so `x_i = -x_{n-1-i}` exactly. `n = 0` gives
/// empty vectors.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let half = n.div_ceil(2);
    for i in 0..half {
        // Largest node first.
        let mut x = libm::cos(PI * (i as f64 + 0.75) / (n as f64 + 0.5));
        let mut dp = 0.0;
        for _ in 0..MAX_NODE_ITERATIONS {
            let (p, d) = legendre_with_derivative(n, x);
            dp = d;
            let dx = p / d;
            x -= dx;
            if dx.abs() <= 1e-16 {
                break;
            }
        }
        let (_, d) = legendre_with_derivative(n, x);
        if d.is_finite() {
            dp = d;
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[n - 1 - i] = x;
        nodes[i] = -x;
        weights[n - 1 - i] = w;
        weights[i] = w;
    }
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
    (nodes, weights)
}

/// Integrate `f` over `[a, b]` with a prepared Gauss-Legendre rule.
#[inline]
pub fn integrate_with<F: FnMut(f64) -> f64>(
    nodes: &[f64],
    weights: &[f64],
    a: f64,
    b: f64,
    mut f: F,
) -> f64 {
    let half = 0.5 * (b - a);
    let mid = 0.5 * (b + a);
    let mut acc = 0.0;
    for (&x, &w) in nodes.iter().zip(weights) {
        acc += w * f(mid + half * x);
    }
    acc * half
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_is_exact_to_degree_2n_minus_1() {
        for n in 1..=24 {
            let (x, w) = gauss_legendre(n);
            assert_eq!(x.len(), n);
            assert!(x.windows(2).all(|p| p[0] < p[1]), "nodes ascending");
            let wsum: f64 = w.iter().sum();
            assert!((wsum - 2.0).abs() < 1e-14, "n={n} weights sum {wsum}");
            for deg in 0..(2 * n) as i32 {
                let got = integrate_with(&x, &w, -1.0, 1.0, |t| t.powi(deg));
                let want = if deg % 2 == 1 {
                    0.0
                } else {
                    2.0 / (deg as f64 + 1.0)
                };
                assert!((got - want).abs() < 1e-14, "n={n} deg={deg}: {got}");
            }
        }
    }

    #[test]
    fn mapped_interval_and_empty_rule() {
        let (x, w) = gauss_legendre(5);
        let got = integrate_with(&x, &w, 1.0, 3.0, |t| t * t * t);
        assert!((got - 20.0).abs() < 1e-13);
        let (x0, w0) = gauss_legendre(0);
        assert!(x0.is_empty() && w0.is_empty());
    }
}
