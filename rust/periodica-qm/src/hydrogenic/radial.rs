//====== periodica/rust/periodica-qm/src/hydrogenic/radial.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! The closed-form radial distribution of a hydrogen-like eigenstate.
//!
//! In `rho = scale * r` the radial density `R^2 r^2 dr` is
//! `f(rho) d rho = q(rho) e^{-rho} d rho` with the degree-`2n` polynomial
//!
//! ```text
//! q(rho) = C rho^{2l+2} [L_{n-l-1}^{2l+1}(rho)]^2 >= 0,   C = (n-l-1)! / (2n (n+l)!)
//! ```
//!
//! so the survival function is elementary:
//!
//! ```text
//! S(rho) = 1 - CDF(rho) = integral_rho^inf q(s) e^{-s} ds
//!        = e^{-rho} P(rho),   P(rho) = integral_0^inf q(rho + u) e^{-u} du,
//! ```
//!
//! a polynomial of degree `2n` times `e^{-rho}` (for 1s,
//! `S = e^{-rho}(1 + rho + rho^2/2)`).
//!
//! ## Evaluating the closed form stably
//!
//! Expanded in monomials, `P` has alternating coefficients that cancel badly:
//! for `n = 10` terms of size ~1e7 sum to `P(0) = 1`, and by `n = 20` even
//! double-double arithmetic runs out of digits. Instead `P(rho)` is evaluated
//! with the `(n+1)`-point Gauss-Laguerre rule, which integrates the
//! degree-`2n` polynomial `u -> q(rho + u)` **exactly**:
//!
//! ```text
//! S(rho) = e^{-rho} sum_i w_i q(rho + u_i),   q(s) = (sqrt(C) s^{l+1} L(s))^2
//! ```
//!
//! Every term is non-negative, so there is no cancellation at all: `S` keeps
//! a relative precision of about `10 n eps` (measured: 4.9e-14 at `n = 30`)
//! for every `n` and every `rho` up to 600.
//! Beyond `rho = 600` (`S < 1e-200`, before `e^{-rho}` goes subnormal) the
//! product is formed as `exp(ln(sum) - rho)`, good to `rho eps`. The nodes are the Laguerre
//! zeros from [`crate::special::laguerre_roots`] and `L` uses the stable
//! recurrence, so the result is bit-reproducible.

use super::RadialFunction;
use crate::numeric::{pow_u, solve_bracketed};
use crate::special::{laguerre, laguerre_roots, ln_gamma};

/// Doubling steps when searching for an upper bracket of a quantile; `S`
/// falls below 2^-1074 well within 64 doublings of the mean for any `n`.
const MAX_BRACKET_DOUBLINGS: usize = 64;

/// Beyond this `rho`, `S = exp(ln(sum) - rho)` instead of `sum * e^{-rho}`.
const LOG_SPACE_RHO: f64 = 600.0;

/// The distribution of the electron's distance from the nucleus in one
/// hydrogen-like eigenstate: PDF `R_nl(r)^2 r^2`, closed-form CDF, and an
/// exact quantile function (used by the Born-rule sampler).
#[derive(Debug, Clone, PartialEq)]
pub struct RadialDistribution {
    radial: RadialFunction,
    /// `sqrt(C) = sqrt((n-l-1)! / (2n (n+l)!))`.
    sqrt_c: f64,
    /// Gauss-Laguerre nodes `u_i` (in `rho`).
    nodes: Vec<f64>,
    /// Gauss-Laguerre weights `w_i`.
    weights: Vec<f64>,
}

impl RadialDistribution {
    pub(crate) fn new(radial: RadialFunction) -> Self {
        let (n, l) = (radial.n, radial.l);
        let sqrt_c = libm::exp(
            0.5 * (ln_gamma(f64::from(n - l))
                - libm::log(2.0 * f64::from(n))
                - ln_gamma(f64::from(n + l + 1))),
        );
        let points = n + 1;
        let nodes = laguerre_roots(points, 0.0);
        // w_i = u_i / ((N+1)^2 L_{N+1}(u_i)^2)  (Abramowitz and Stegun 25.4.45).
        let np1 = f64::from(points + 1);
        let weights = nodes
            .iter()
            .map(|&u| {
                let lag = laguerre(points + 1, 0.0, u);
                u / (np1 * np1 * lag * lag)
            })
            .collect();
        Self {
            radial,
            sqrt_c,
            nodes,
            weights,
        }
    }

    /// `q(s) = C s^{2l+2} L_{n-l-1}^{2l+1}(s)^2`.
    #[inline]
    fn q(&self, s: f64) -> f64 {
        let r = &self.radial;
        let v = self.sqrt_c * pow_u(s, r.l + 1) * laguerre(r.laguerre_order(), r.alpha(), s);
        v * v
    }

    /// `P(r) = R_nl(r)^2 r^2`, per bohr. Zero for `r < 0`.
    pub fn pdf(&self, r: f64) -> f64 {
        if r < 0.0 {
            return 0.0;
        }
        let v = self.radial.eval(r) * r;
        v * v
    }

    /// `P(R > r)`, in closed form. 1 for `r <= 0`.
    pub fn survival(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 1.0;
        }
        self.survival_rho(self.radial.scale * r)
    }

    /// `P(R <= r) = 1 - survival(r)`. 0 for `r <= 0`.
    pub fn cdf(&self, r: f64) -> f64 {
        1.0 - self.survival(r)
    }

    /// The `r` with `P(R > r) = v`, for `v` in `(0, 1]`; `0` for `v >= 1`,
    /// `+inf` for `v <= 0`, `NaN` for `NaN`.
    ///
    /// Inverting the survival function rather than the CDF keeps relative
    /// precision in the far tail, where the CDF is `1 - tiny`.
    pub fn inverse_survival(&self, v: f64) -> f64 {
        if v.is_nan() {
            return f64::NAN;
        }
        if v >= 1.0 {
            return 0.0;
        }
        if v <= 0.0 {
            return f64::INFINITY;
        }
        let scale = self.radial.scale;
        // Mean of rho = (3n^2 - l(l+1)) / n; double from there to bracket.
        let (nf, lf) = (f64::from(self.radial.n), f64::from(self.radial.l));
        let mean_rho = (3.0 * nf * nf - lf * (lf + 1.0)) / nf;
        let mut hi = mean_rho;
        for _ in 0..MAX_BRACKET_DOUBLINGS {
            if self.survival_rho(hi) < v {
                break;
            }
            hi *= 2.0;
        }
        let g = |rho: f64| (self.survival_rho(rho) - v, -self.density_rho(rho));
        let guess = mean_rho.min(0.5 * hi);
        solve_bracketed(g, 0.0, hi, guess) / scale
    }

    /// The quantile function: the `r` with `P(R <= r) = u`, `u` in `[0, 1)`.
    pub fn quantile(&self, u: f64) -> f64 {
        self.inverse_survival(1.0 - u)
    }

    /// `f(rho) = q(rho) e^{-rho}`, the density in `rho`.
    #[inline]
    fn density_rho(&self, rho: f64) -> f64 {
        let scale = self.radial.scale;
        self.pdf(rho / scale) / scale
    }

    /// `S(rho)` by the exact, cancellation-free Gauss-Laguerre sum.
    fn survival_rho(&self, rho: f64) -> f64 {
        let sum: f64 = self
            .nodes
            .iter()
            .zip(&self.weights)
            .map(|(&u, &w)| w * self.q(rho + u))
            .sum();
        if rho <= LOG_SPACE_RHO {
            sum * libm::exp(-rho)
        } else if sum.is_finite() && sum > 0.0 {
            libm::exp(libm::log(sum) - rho)
        } else {
            // The polynomial overflowed: only possible where e^{-rho} has
            // long since underflowed.
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::hydrogenic::{HydrogenLike, Orbital};

    fn dist(h: &HydrogenLike, n: u32, l: u32) -> super::RadialDistribution {
        h.orbital(Orbital::complex(n, l, 0).unwrap())
            .radial_distribution()
    }

    #[test]
    fn low_states_match_the_textbook_polynomials() {
        let h = HydrogenLike::infinite_mass(1.0).unwrap();
        let s1 = dist(&h, 1, 0);
        let s2 = dist(&h, 2, 0);
        for &r in &[0.01_f64, 0.1, 1.0, 4.0, 30.0, 300.0] {
            // 1s: rho = 2r, S = e^{-rho}(1 + rho + rho^2/2).
            let rho = 2.0 * r;
            let want = libm::exp(-rho) * (1.0 + rho + 0.5 * rho * rho);
            assert!((s1.survival(r) / want - 1.0).abs() < 4e-15, "1s r={r}");
            // 2s: rho = r, S = e^{-rho}(1 + rho + rho^2/2 + rho^4/8).
            let rho = r;
            let want = libm::exp(-rho) * (1.0 + rho + 0.5 * rho * rho + rho.powi(4) / 8.0);
            assert!((s2.survival(r) / want - 1.0).abs() < 4e-15, "2s r={r}");
        }
        assert_eq!(s1.survival(0.0), 1.0);
        assert_eq!(s1.cdf(-1.0), 0.0);
        assert_eq!(s1.pdf(-1.0), 0.0);
    }

    #[test]
    fn survival_starts_at_one_for_every_state_to_n_30() {
        let h = HydrogenLike::infinite_mass(3.0).unwrap();
        let mut worst = 0.0_f64;
        for n in 1..=30 {
            for l in 0..n {
                // S(0+) = 1 - O(r^{2l+3}); probe at a radius where that is < ulp.
                let s = dist(&h, n, l).survival(1e-9);
                worst = worst.max((s - 1.0).abs());
                assert!((s - 1.0).abs() <= 1e-13, "n={n} l={l}: {s}");
            }
        }
        eprintln!("max |S(0+) - 1| over n <= 30: {worst:e}");
    }

    #[test]
    fn quantile_inverts_the_cdf() {
        let h = HydrogenLike::hydrogen();
        for (n, l) in [(1, 0), (2, 0), (3, 1), (4, 3), (7, 2), (12, 5)] {
            let rd = dist(&h, n, l);
            for &u in &[1e-12, 0.01, 0.3, 0.5, 0.77, 0.999, 1.0 - 1e-9] {
                let r = rd.quantile(u);
                assert!((rd.cdf(r) - u).abs() < 1e-14, "n={n} l={l} u={u}");
            }
            for &v in &[1e-300, 1e-30, 1e-5] {
                let r = rd.inverse_survival(v);
                assert!((rd.survival(r) / v - 1.0).abs() < 1e-12, "n={n} v={v}");
            }
            assert_eq!(rd.inverse_survival(1.0), 0.0);
            assert_eq!(rd.inverse_survival(0.0), f64::INFINITY);
            assert!(rd.inverse_survival(f64::NAN).is_nan());
        }
    }
}
