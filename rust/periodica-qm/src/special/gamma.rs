//====== periodica/rust/periodica-qm/src/special/gamma.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! The log-gamma function and the regularised incomplete gamma functions of
//! integer order.

/// Iteration cap for the lower-tail series of [`gamma_p_int`]. Used only for
/// `x < n`, where the term ratio is `x / (n + j) < 1`; the cap is reached only
/// for `n` in the hundreds of thousands.
const MAX_SERIES_TERMS: usize = 100_000;

/// `ln Gamma(x)` for `x > 0`.
///
/// Delegates to `libm::lgamma_r` (musl's implementation: < 1 ulp-class
/// accuracy, pure Rust, bit-identical across platforms). Returns `NaN` for
/// `x <= 0` or `NaN`, and `+inf` for `x = +inf`.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    if x > 0.0 {
        libm::lgamma_r(x).0
    } else {
        f64::NAN
    }
}

/// `ln(n!) = ln Gamma(n + 1)`.
#[inline]
pub fn ln_factorial(n: u32) -> f64 {
    ln_gamma(f64::from(n) + 1.0)
}

/// Largest order evaluated as a direct product in [`poisson_term`].
const DIRECT_POISSON_K_MAX: u32 = 200;
/// Largest `x` for which `e^{-x}` is a normal `f64` with room to spare.
const DIRECT_POISSON_X_MAX: f64 = 700.0;

/// The prefactor `e^{-x} x^k / k!`, i.e. the Poisson probability of `k` at
/// mean `x`.
///
/// For moderate `k` and `x` it is the direct product `e^{-x} prod_j (x / j)`:
/// every partial product is itself a Poisson probability (`<= 1`), so nothing
/// overflows, and the error is `k + 1` roundings. Otherwise it is evaluated
/// in log space, whose error grows like `x * eps` (the rounding of the
/// exponent's argument).
#[inline]
fn poisson_term(k: u32, x: f64) -> f64 {
    if k == 0 {
        return libm::exp(-x);
    }
    if k <= DIRECT_POISSON_K_MAX && x <= DIRECT_POISSON_X_MAX {
        let mut p = libm::exp(-x);
        for j in 1..=k {
            p *= x / f64::from(j);
        }
        return p;
    }
    libm::exp(f64::from(k) * libm::log(x) - x - ln_factorial(k))
}

/// The regularised lower incomplete gamma function of integer order,
/// `P(n, x) = gamma(n, x) / Gamma(n) = 1 - e^{-x} sum_{k<n} x^k / k!`.
///
/// This is the CDF at `x` of a Gamma(`n`, 1) variable (a sum of `n` unit
/// exponentials). Evaluated as the series
/// `e^{-x} x^n / n! * sum_j x^j / ((n+1)...(n+j))` when `x < n`, so small
/// probabilities keep full relative precision, and as `1 - Q(n, x)` from the
/// closed-form finite sum otherwise.
///
/// Conventions: `P(0, x) = 1`; `P(n, x) = 0` for `x <= 0`; `NaN` in, `NaN` out.
pub fn gamma_p_int(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x < f64::from(n) {
        lower_series(n, x)
    } else {
        1.0 - upper_sum(n, x)
    }
}

/// The regularised upper incomplete gamma function of integer order,
/// `Q(n, x) = 1 - P(n, x) = e^{-x} sum_{k=0}^{n-1} x^k / k!`.
///
/// Uses the closed-form finite sum when `x >= n` (where `Q` is the small
/// side) and `1 - P` otherwise. Conventions mirror [`gamma_p_int`].
pub fn gamma_q_int(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if n == 0 {
        return 0.0;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < f64::from(n) {
        1.0 - lower_series(n, x)
    } else {
        upper_sum(n, x)
    }
}

/// `P(n, x)` for `0 < x < n` by the convergent series.
fn lower_series(n: u32, x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut denom = f64::from(n);
    for _ in 0..MAX_SERIES_TERMS {
        denom += 1.0;
        term *= x / denom;
        sum += term;
        if term <= sum * f64::EPSILON * 0.5 {
            break;
        }
    }
    poisson_term(n, x) * sum
}

/// `Q(n, x) = sum_{k<n} e^{-x} x^k / k!` for `x >= n`, summed from the
/// largest term (`k = n - 1`) downward with `t_{k-1} = t_k k / x`.
fn upper_sum(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let mut term = poisson_term(n - 1, x);
    let mut sum = term;
    let mut k = n - 1;
    while k > 0 {
        term *= f64::from(k) / x;
        sum += term;
        if term <= sum * f64::EPSILON * 0.5 {
            break;
        }
        k -= 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_hits_factorials_and_half_integers() {
        let mut fact = 1.0_f64;
        for n in 1..=25_u32 {
            fact *= f64::from(n);
            let got = ln_factorial(n);
            assert!(
                (got - fact.ln()).abs() <= 4.0 * f64::EPSILON * got.abs().max(1.0),
                "ln({n}!) = {got} vs {}",
                fact.ln()
            );
        }
        assert_eq!(ln_factorial(0), 0.0);
        assert_eq!(ln_factorial(1), 0.0);
        // Gamma(1/2) = sqrt(pi).
        let half = ln_gamma(0.5);
        assert!((half - 0.5 * core::f64::consts::PI.ln()).abs() < 1e-15);
        assert!(ln_gamma(0.0).is_nan() && ln_gamma(-1.5).is_nan());
        assert_eq!(ln_gamma(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn closed_forms_for_small_order() {
        for &x in &[1e-8_f64, 0.01, 0.5, 1.0, 2.5, 7.0, 30.0, 200.0] {
            let e = libm::exp(-x);
            // P(1, x) = 1 - e^-x ; P(2, x) = 1 - e^-x (1 + x);
            // P(3, x) = 1 - e^-x (1 + x + x^2/2).
            let q = [e, e * (1.0 + x), e * (1.0 + x + 0.5 * x * x)];
            for (i, &qq) in q.iter().enumerate() {
                let n = i as u32 + 1;
                let got_q = gamma_q_int(n, x);
                assert!(
                    (got_q - qq).abs() <= 4e-15 * qq,
                    "Q({n},{x}) = {got_q} vs {qq}"
                );
                let got_p = gamma_p_int(n, x);
                assert!((got_p + got_q - 1.0).abs() < 2e-16, "P+Q at n={n} x={x}");
            }
        }
        // Small x keeps relative precision: P(3, 1e-6) ~ x^3/6.
        let p = gamma_p_int(3, 1e-6);
        assert!((p / (1e-18 / 6.0) - 1.0).abs() < 1e-5, "{p}");
    }

    #[test]
    fn edge_cases() {
        assert_eq!(gamma_p_int(0, 3.0), 1.0);
        assert_eq!(gamma_q_int(0, 3.0), 0.0);
        assert_eq!(gamma_p_int(4, 0.0), 0.0);
        assert_eq!(gamma_q_int(4, -1.0), 1.0);
        assert_eq!(gamma_p_int(4, f64::INFINITY), 1.0);
        assert_eq!(gamma_q_int(4, f64::INFINITY), 0.0);
        assert!(gamma_p_int(4, f64::NAN).is_nan());
        assert!(gamma_q_int(4, f64::NAN).is_nan());
        // Large order: median of Gamma(n) is ~ n - 1/3.
        let p = gamma_p_int(400, 400.0 - 1.0 / 3.0);
        assert!((p - 0.5).abs() < 2e-3, "{p}");
    }
}
