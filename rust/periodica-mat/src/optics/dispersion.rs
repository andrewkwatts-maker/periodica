//! A material's dispersion model, chosen from the coefficients a record carries.

use arithma_core::ArithmaExpression as E;

use super::cauchy::Cauchy;
use super::drude_lorentz::DrudeLorentz;
use super::sellmeier::Sellmeier;
use super::spectrum::{spectrum_to_rgb, Spectrum};
use super::{eval, num, square, var};
use crate::property::PropertyId as P;
use crate::table::PropertyTable;

const HBAR_J_S: f64 = 1.054_571_817e-34;
const EV_J: f64 = 1.602_176_634e-19;

/// Refractive index `n` and extinction coefficient `k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nk {
    pub n: f64,
    pub k: f64,
}

/// A dispersion model.
#[derive(Debug, Clone, PartialEq)]
pub enum Dispersion {
    Sellmeier(Sellmeier),
    Cauchy(Cauchy),
    DrudeLorentz(DrudeLorentz),
}

/// Normal-incidence Fresnel reflectance in air, variables `n` and `k`.
pub fn fresnel_expr() -> E {
    let k2 = square(var("k"));
    let up = E::add(square(E::sub(var("n"), num(1.0))), k2.clone());
    let down = E::add(square(E::add(var("n"), num(1.0))), k2);
    E::div(up, down)
}

fn rad_s_to_ev(w: f64) -> f64 {
    HBAR_J_S * w / EV_J
}

fn sellmeier_of(t: &PropertyTable) -> Option<Sellmeier> {
    let pairs = [(P::SellmeierB1, P::SellmeierC1), (P::SellmeierB2, P::SellmeierC2), (P::SellmeierB3, P::SellmeierC3)];
    let (mut b, mut c) = (Vec::new(), Vec::new());
    for (pb, pc) in pairs {
        if let (Some(vb), Some(vc)) = (t.get(pb), t.get(pc)) {
            b.push(vb);
            c.push(vc * 1e12);
        }
    }
    (!b.is_empty()).then_some(Sellmeier { b, c })
}

fn drude_of(t: &PropertyTable) -> Option<DrudeLorentz> {
    let wp = t.get(P::DrudePlasmaFrequency).filter(|v| *v > 0.0)?;
    let gamma = t.get(P::DrudeDamping).filter(|v| *v > 0.0)?;
    Some(DrudeLorentz {
        epsilon_inf: t.get_or(P::EpsilonInfinity, 1.0),
        plasma_ev: rad_s_to_ev(wp),
        damping_ev: rad_s_to_ev(gamma),
        oscillators: Vec::new(),
    })
}

impl Dispersion {
    /// The model a record's coefficients describe, if any: Sellmeier, else Drude.
    pub fn from_table(t: &PropertyTable) -> Option<Self> {
        sellmeier_of(t)
            .map(Self::Sellmeier)
            .or_else(|| drude_of(t).map(Self::DrudeLorentz))
    }

    /// `n` and `k` at `wavelength_m` metres.
    pub fn nk(&self, wavelength_m: f64) -> Nk {
        match self {
            Self::Sellmeier(s) => Nk { n: s.refractive_index(wavelength_m), k: 0.0 },
            Self::Cauchy(c) => Nk { n: c.refractive_index(wavelength_m), k: 0.0 },
            Self::DrudeLorentz(d) => {
                let (n, k) = d.nk(wavelength_m);
                Nk { n, k }
            }
        }
    }

    /// Normal-incidence reflectance at `wavelength_m` metres.
    pub fn reflectance(&self, wavelength_m: f64) -> f64 {
        let nk = self.nk(wavelength_m);
        eval(&fresnel_expr(), &[("n", nk.n), ("k", nk.k)])
    }

    /// Reflectance over `[lo, hi]` metres.
    pub fn reflectance_spectrum(&self, lo: f64, hi: f64, bins: usize) -> Spectrum {
        let expr = fresnel_expr();
        Spectrum::sample(lo, hi, bins, |l| {
            let nk = self.nk(l);
            eval(&expr, &[("n", nk.n), ("k", nk.k)])
        })
    }

    /// Linear sRGB base colour: the visible reflectance spectrum, converted.
    pub fn base_color(&self) -> [f64; 3] {
        spectrum_to_rgb(&self.reflectance_spectrum(360e-9, 830e-9, 95))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Source;

    #[test]
    fn gold_reflects_red_more_than_blue() {
        let g = Dispersion::DrudeLorentz(DrudeLorentz::gold());
        let [r, _, b] = g.base_color();
        assert!(r > b, "{r} {b}");
        assert!(g.reflectance(600e-9) > 0.9);
    }

    #[test]
    fn record_coefficients_select_sellmeier() {
        let mut t = PropertyTable::new();
        t.set(P::SellmeierB1, 1.03961212, Source::Datasheet);
        t.set(P::SellmeierC1, 0.00600069867e-12, Source::Datasheet);
        let d = Dispersion::from_table(&t).expect("model");
        assert!(d.nk(587.6e-9).n > 1.4);
    }
}
