//! Emission lines: Gaussian-broadened lines added to a spectrum.

use arithma_core::ArithmaExpression as E;

use super::{eval, num, square, sum, var};
use super::spectrum::Spectrum;

/// One emission line: centre and width (standard deviation) in metres, peak radiance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub centre_m: f64,
    pub sigma_m: f64,
    pub peak: f64,
}

/// The summed line profile as an expression in `l` (metres).
pub fn profile_expr(lines: &[Line]) -> E {
    let terms = lines
        .iter()
        .map(|ln| {
            let z = E::div(E::sub(var("l"), num(ln.centre_m)), num(ln.sigma_m));
            let arg = E::neg(E::mul(num(0.5), square(z)));
            E::mul(num(ln.peak), E::exp(arg))
        })
        .collect();
    sum(terms)
}

/// Line radiance at `lambda` metres.
pub fn radiance(lines: &[Line], lambda: f64) -> f64 {
    eval(&profile_expr(lines), &[("l", lambda)])
}

/// The line spectrum over `[lo, hi]` metres.
pub fn spectrum(lines: &[Line], lo: f64, hi: f64, bins: usize) -> Spectrum {
    let expr = profile_expr(lines);
    Spectrum::sample(lo, hi, bins, |l| eval(&expr, &[("l", l)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_at_centre() {
        let l = [Line { centre_m: 589e-9, sigma_m: 1e-9, peak: 3.0 }];
        assert!((radiance(&l, 589e-9) - 3.0).abs() < 1e-9);
        assert!(radiance(&l, 600e-9) < 1e-6);
    }
}
