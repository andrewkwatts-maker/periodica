//! A material's dispersion model, chosen from the coefficients a record carries.

use arithma_core::ArithmaExpression as E;

use super::cauchy::Cauchy;
use super::drude_lorentz::DrudeLorentz;
use super::sellmeier::Sellmeier;
use super::spectrum::{spectrum_to_rgb, Spectrum};
use super::{eval, num, square, var, OpticsError};
use crate::property::PropertyId as P;
use crate::table::PropertyTable;

/// Wavelength range (m) given to models built from a record, which does not yet store
/// the range its coefficients were fitted over: the visible and near infrared.
const TABLE_RANGE_M: (f64, f64) = (0.3e-6, 2.5e-6);

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
    let pairs = [
        (P::SellmeierB1, P::SellmeierC1),
        (P::SellmeierB2, P::SellmeierC2),
        (P::SellmeierB3, P::SellmeierC3),
    ];
    let (mut b, mut c) = (Vec::new(), Vec::new());
    for (pb, pc) in pairs {
        if let (Some(vb), Some(vc)) = (t.get(pb), t.get(pc)) {
            b.push(vb);
            c.push(vc * 1e12);
        }
    }
    (!b.is_empty()).then_some(Sellmeier {
        b,
        c,
        range_m: TABLE_RANGE_M,
    })
}

fn drude_of(t: &PropertyTable) -> Result<Option<DrudeLorentz>, OpticsError> {
    let (Some(wp), Some(gamma)) = (
        t.get(P::DrudePlasmaFrequency).filter(|v| *v > 0.0),
        t.get(P::DrudeDamping).filter(|v| *v > 0.0),
    ) else {
        return Ok(None);
    };
    // No silent default: epsilon_inf = 1 turns every metal into a Drude-only silver.
    let epsilon_inf = t
        .get(P::EpsilonInfinity)
        .ok_or(OpticsError::MissingParameter("EpsilonInfinity"))?;
    Ok(Some(DrudeLorentz {
        epsilon_inf,
        plasma_ev: rad_s_to_ev(wp),
        damping_ev: rad_s_to_ev(gamma),
        oscillators: Vec::new(),
        critical_points: Vec::new(),
        range_m: TABLE_RANGE_M,
    }))
}

impl Dispersion {
    /// The model a record's coefficients describe, if any: Sellmeier, else Drude.
    /// Drude coefficients without `EpsilonInfinity` are an error.
    pub fn from_table(t: &PropertyTable) -> Result<Option<Self>, OpticsError> {
        if let Some(s) = sellmeier_of(t) {
            return Ok(Some(Self::Sellmeier(s)));
        }
        Ok(drude_of(t)?.map(Self::DrudeLorentz))
    }

    /// `n` and `k` at `wavelength_m` metres; an error outside the model's fitted range.
    pub fn nk(&self, wavelength_m: f64) -> Result<Nk, OpticsError> {
        Ok(match self {
            Self::Sellmeier(s) => Nk {
                n: s.refractive_index(wavelength_m)?,
                k: 0.0,
            },
            Self::Cauchy(c) => Nk {
                n: c.refractive_index(wavelength_m)?,
                k: 0.0,
            },
            Self::DrudeLorentz(d) => {
                let (n, k) = d.nk(wavelength_m)?;
                Nk { n, k }
            }
        })
    }

    /// Normal-incidence reflectance at `wavelength_m` metres.
    pub fn reflectance(&self, wavelength_m: f64) -> Result<f64, OpticsError> {
        let nk = self.nk(wavelength_m)?;
        Ok(eval(&fresnel_expr(), &[("n", nk.n), ("k", nk.k)]))
    }

    /// Reflectance over `[lo, hi]` metres; an error if any sample is out of range.
    pub fn reflectance_spectrum(
        &self,
        lo: f64,
        hi: f64,
        bins: usize,
    ) -> Result<Spectrum, OpticsError> {
        let first_error = std::cell::RefCell::new(None);
        let spectrum = Spectrum::sample(lo, hi, bins, |l| {
            self.reflectance(l).unwrap_or_else(|e| {
                first_error.borrow_mut().get_or_insert(e);
                0.0
            })
        });
        match first_error.into_inner() {
            Some(e) => Err(e),
            None => Ok(spectrum),
        }
    }

    /// Linear sRGB base colour: the visible reflectance spectrum, converted.
    pub fn base_color(&self) -> Result<[f64; 3], OpticsError> {
        Ok(spectrum_to_rgb(
            &self.reflectance_spectrum(360e-9, 830e-9, 95)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Source;

    #[test]
    fn gold_reflects_red_more_than_blue() {
        let g = Dispersion::DrudeLorentz(DrudeLorentz::gold());
        let [r, _, b] = g.base_color().unwrap();
        assert!(r > b, "{r} {b}");
        assert!(g.reflectance(600e-9).unwrap() > 0.9);
    }

    #[test]
    fn record_coefficients_select_sellmeier() {
        let mut t = PropertyTable::new();
        t.set(P::SellmeierB1, 1.03961212, Source::Datasheet);
        t.set(P::SellmeierC1, 0.00600069867e-12, Source::Datasheet);
        let d = Dispersion::from_table(&t).unwrap().expect("model");
        assert!(d.nk(587.6e-9).unwrap().n > 1.4);
    }
}
