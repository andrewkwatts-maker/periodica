//! Sellmeier dispersion: n^2 = 1 + sum B_j l^2 / (l^2 - C_j), l in micrometres.

use arithma_core::ArithmaExpression as E;

use super::{eval, num, square, sum, var};

/// Sellmeier coefficients; `c` is in square micrometres.
#[derive(Debug, Clone, PartialEq)]
pub struct Sellmeier {
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

impl Sellmeier {
    /// Schott N-BK7.
    pub fn bk7() -> Self {
        Self {
            b: vec![1.03961212, 0.231792344, 1.01046945],
            c: vec![0.00600069867, 0.0200179144, 103.560653],
        }
    }

    /// `n^2` as an expression in the variable `um` (wavelength in micrometres).
    pub fn n_squared_expr(&self) -> E {
        let l2 = square(var("um"));
        let terms = self
            .b
            .iter()
            .zip(&self.c)
            .map(|(b, c)| E::div(E::mul(num(*b), l2.clone()), E::sub(l2.clone(), num(*c))))
            .collect();
        E::add(num(1.0), sum(terms))
    }

    /// `n` as an expression in `um`.
    pub fn n_expr(&self) -> E {
        E::sqrt(self.n_squared_expr())
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
    fn bk7_at_sodium_d() {
        let n = Sellmeier::bk7().refractive_index(587.6e-9);
        assert!((n - 1.5168).abs() < 1e-4, "{n}");
    }
}
