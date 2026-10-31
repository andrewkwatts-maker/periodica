//! Cauchy dispersion: n = A + B / l^2 + C / l^4, l in micrometres.

use arithma_core::ArithmaExpression as E;

use super::{eval, num, square, var};

/// Cauchy coefficients; `b` in um^2, `c` in um^4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cauchy {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Cauchy {
    /// `n` as an expression in `um`.
    pub fn n_expr(&self) -> E {
        let l2 = square(var("um"));
        let second = E::div(num(self.b), l2.clone());
        let fourth = E::div(num(self.c), square(l2));
        E::add(E::add(num(self.a), second), fourth)
    }

    /// Refractive index at `wavelength_m` metres.
    pub fn refractive_index(&self, wavelength_m: f64) -> f64 {
        eval(&self.n_expr(), &[("um", wavelength_m * 1e6)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cauchy_matches_closed_form() {
        let m = Cauchy { a: 1.5, b: 0.004, c: 0.0001 };
        let l: f64 = 0.55;
        let want = 1.5 + 0.004 / (l * l) + 0.0001 / l.powi(4);
        assert!((m.refractive_index(550e-9) - want).abs() < 1e-12);
    }
}
