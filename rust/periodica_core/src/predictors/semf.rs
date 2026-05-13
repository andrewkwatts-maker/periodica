//====== periodica/rust/periodica_core/src/predictors/semf.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # semf
//!
//! Semi-Empirical Mass Formula (Bethe-WeizsÃ¤cker) predictor.
//!
//! Returns binding energy `B(Z, N)` in MeV and the nuclear radius `R(A)` in
//! femtometres. Coefficients follow Krane's *Introductory Nuclear Physics*
//! (1988); the Python reference lives at
//! `src/periodica/utils/predictors/semf.py`.
//!
//! ```text
//!     B(Z,N) = a_vÂ·A
//!            âˆ’ a_sÂ·A^{2/3}
//!            âˆ’ a_cÂ·Z(Zâˆ’1)/A^{1/3}
//!            âˆ’ a_aÂ·(Nâˆ’Z)^2/A
//!            + Î´(Z,N)
//! ```
//!
//! Coefficients live in [`CoefficientSet`]; default values come from
//! `default_constants.json` once the data_loader is wired up.

use anyhow::{anyhow, Result};

use super::Predictor;

/// SEMF coefficient bundle (volume, surface, Coulomb, asymmetry, pairing).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoefficientSet {
    pub volume: f64,
    pub surface: f64,
    pub coulomb: f64,
    pub asymmetry: f64,
    pub pairing: f64,
    /// Empirical nuclear-radius prefactor `r_0` in femtometres.
    pub radius_prefactor_fm: f64,
}

impl Default for CoefficientSet {
    fn default() -> Self {
        // Krane (1988) values, MeV.
        Self {
            volume: 15.5,
            surface: 16.8,
            coulomb: 0.72,
            asymmetry: 23.0,
            pairing: 34.0,
            radius_prefactor_fm: 1.2,
        }
    }
}

/// Predictor handle (zero-sized today; carries the coefficient set tomorrow).
#[derive(Debug, Default)]
pub struct SemfPredictor {
    pub coefficients: CoefficientSet,
}

impl Predictor for SemfPredictor {
    fn name(&self) -> &'static str {
        "semf"
    }
}

/// Public binding-energy entry-point.
///
/// Returns `Err` for `Z + N == 0` (undefined nucleon system) so callers cannot
/// silently consume `NaN`.
pub fn binding_energy(protons: u32, neutrons: u32) -> Result<f64> {
    binding_energy_with(protons, neutrons, &CoefficientSet::default())
}

/// Variant that accepts an explicit coefficient set (engine plugin can swap
/// the JSON-driven values in without touching this module).
pub fn binding_energy_with(_protons: u32, _neutrons: u32, _coeffs: &CoefficientSet) -> Result<f64> {
    Err(anyhow!(
        "predictors::semf::binding_energy is not yet implemented; \
         pending PTExpression integration via Arithmos bridge"
    ))
}

/// Empirical nuclear radius `R(A) = r_0 Â· A^{1/3}` in femtometres.
pub fn nuclear_radius(mass_number: u32) -> Result<f64> {
    nuclear_radius_with(mass_number, &CoefficientSet::default())
}

/// Variant accepting an explicit coefficient set.
pub fn nuclear_radius_with(mass_number: u32, coeffs: &CoefficientSet) -> Result<f64> {
    if mass_number == 0 {
        return Err(anyhow!(
            "predictors::semf::nuclear_radius: mass number must be > 0"
        ));
    }
    Ok(coeffs.radius_prefactor_fm * (mass_number as f64).cbrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_coefficients_are_in_krane_range() {
        let c = CoefficientSet::default();
        assert!((c.volume - 15.5).abs() < 1e-9);
        assert!((c.surface - 16.8).abs() < 1e-9);
        assert!((c.coulomb - 0.72).abs() < 1e-9);
    }

    #[test]
    fn binding_energy_unimplemented_for_now() {
        assert!(binding_energy(26, 30).is_err());
    }

    #[test]
    fn nuclear_radius_iron56() {
        // 1.2 Â· 56^(1/3) â‰ˆ 4.59 fm. Tolerance is generous because we only want
        // to confirm we're in the right ballpark before JSON coefficients land.
        let r = nuclear_radius(56).unwrap();
        assert!(
            (r - 4.59).abs() < 0.05,
            "iron-56 radius out of range: {r}"
        );
    }

    #[test]
    fn nuclear_radius_rejects_zero_mass_number() {
        assert!(nuclear_radius(0).is_err());
    }
}
