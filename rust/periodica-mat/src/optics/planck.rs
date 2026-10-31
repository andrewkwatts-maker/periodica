//! Black-body spectral radiance as an Arithma expression.

use std::sync::OnceLock;

use arithma_core::ArithmaExpression as E;

use super::{eval, num, var};
use super::spectrum::{luminous_band_m, spectrum_to_rgb, Spectrum};

pub const PLANCK_H: f64 = 6.626_070_15e-34;
pub const LIGHT_C: f64 = 299_792_458.0;
pub const BOLTZMANN_K: f64 = 1.380_649e-23;

/// `B(l, T) = 2 h c^2 / (l^5 (exp(h c / (l k T)) - 1))`, variables `l` (m) and `t` (K).
pub fn radiance_expr() -> E {
    let hc = PLANCK_H * LIGHT_C;
    let x = E::div(num(hc), E::mul(E::mul(var("l"), num(BOLTZMANN_K)), var("t")));
    let denom = E::mul(
        E::pow(var("l"), num(5.0)),
        E::sub(E::exp(x), num(1.0)),
    );
    E::div(num(2.0 * hc * LIGHT_C), denom)
}

/// Spectral radiance in W sr^-1 m^-3 at temperature `t` (K) and wavelength `lambda` (m).
pub fn planck(t: f64, lambda: f64) -> f64 {
    static EXPR: OnceLock<E> = OnceLock::new();
    eval(EXPR.get_or_init(radiance_expr), &[("l", lambda), ("t", t)])
}

/// The black-body spectrum over `[lo, hi]` metres in `bins` samples.
pub fn spectrum(t: f64, lo: f64, hi: f64, bins: usize) -> Spectrum {
    Spectrum::sample(lo, hi, bins, |l| planck(t, l))
}

/// Visible black-body radiance per linear sRGB channel, in W sr^-1 m^-2, at `t` kelvin.
pub fn visible_rgb(t: f64) -> [f64; 3] {
    let s = spectrum(t, 360e-9, 830e-9, 471);
    let band = luminous_band_m();
    spectrum_to_rgb(&s).map(|c| c * band)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wien_peak() {
        for t in [3000.0, 5800.0, 9000.0] {
            let (mut best_l, mut best_v, mut l) = (0.0, 0.0, 100e-9);
            while l < 5e-6 {
                let v = planck(t, l);
                if v > best_v {
                    (best_l, best_v) = (l, v);
                }
                l += 1e-9;
            }
            let product = best_l * t;
            assert!((product - 2.8978e-3).abs() / 2.8978e-3 < 0.005, "T={t} {product}");
        }
    }

    #[test]
    fn red_hot_ordering() {
        let [r, g, b] = visible_rgb(1500.0);
        assert!(r > g && g > b && b > 0.0, "{r} {g} {b}");
    }

    #[test]
    fn stefan_boltzmann() {
        let t = 1000.0;
        let s = spectrum(t, 100e-9, 100e-6, 20_000);
        let total = s.integral();
        let sigma = 5.670_374_419e-8;
        let want = sigma * t.powi(4) / std::f64::consts::PI;
        assert!((total - want).abs() / want < 0.01, "{total} vs {want}");
    }
}
