//! Sellmeier dispersion: n^2 = 1 + sum B_j l^2 / (l^2 - C_j), l in micrometres.

use arithma_core::ArithmaExpression as E;

use super::{check_range, eval, finite, num, square, sum, var, OpticsError};

/// Sellmeier coefficients; `c` is in square micrometres.
/// `range_m` is the wavelength range (metres) the coefficients were fitted over.
#[derive(Debug, Clone, PartialEq)]
pub struct Sellmeier {
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub range_m: (f64, f64),
}

impl Sellmeier {
    /// Schott N-BK7; the Schott datasheet states validity over 0.3 to 2.5 um.
    pub fn bk7() -> Self {
        Self {
            b: vec![1.03961212, 0.231792344, 1.01046945],
            c: vec![0.00600069867, 0.0200179144, 103.560653],
            range_m: (0.3e-6, 2.5e-6),
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

    /// Refractive index at `wavelength_m` metres; an error outside the fitted range.
    pub fn refractive_index(&self, wavelength_m: f64) -> Result<f64, OpticsError> {
        check_range(wavelength_m, self.range_m)?;
        finite(
            eval(&self.n_expr(), &[("um", wavelength_m * 1e6)]),
            wavelength_m,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bk7_at_sodium_d() {
        let n = Sellmeier::bk7().refractive_index(587.6e-9).unwrap();
        assert!((n - 1.5168).abs() < 1e-4, "{n}");
    }
}
