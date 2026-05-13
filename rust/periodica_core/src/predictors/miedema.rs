//====== periodica/rust/periodica_core/src/predictors/miedema.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # miedema
//!
//! Miedema model for binary-alloy heat of formation.
//!
//! `ΔH = f(c_A, c_B, φ*, n_ws, V)` per Miedema, Boom & de Boer (1980). The
//! Python reference is `src/periodica/utils/predictors/miedema.py`.

use anyhow::{anyhow, Result};

use super::Predictor;

/// Element-level Miedema parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiedemaParameters {
    /// Electronegativity (Miedema's φ*).
    pub electronegativity: f64,
    /// Wigner-Seitz electron density `n_ws^{1/3}`.
    pub n_ws_cube_root: f64,
    /// Molar volume in cm³/mol.
    pub molar_volume_cm3_per_mol: f64,
    /// Solute-class flag: 0 = transition metal, 1 = noble metal, 2 = main group.
    pub solute_class: u32,
}

/// Predictor handle.
#[derive(Debug, Default)]
pub struct MiedemaPredictor;

impl Predictor for MiedemaPredictor {
    fn name(&self) -> &'static str {
        "miedema"
    }
}

/// Heat of formation `ΔH_f` in kJ/mol for a binary alloy at composition
/// `(c_a, c_b)` (mole fractions).
pub fn heat_of_formation(
    _a: &MiedemaParameters,
    _b: &MiedemaParameters,
    c_a: f64,
    c_b: f64,
) -> Result<f64> {
    if !(0.0..=1.0).contains(&c_a) || !(0.0..=1.0).contains(&c_b) {
        return Err(anyhow!(
            "predictors::miedema: mole fractions must lie in [0,1]"
        ));
    }
    if (c_a + c_b - 1.0).abs() > 1e-6 {
        return Err(anyhow!(
            "predictors::miedema: mole fractions must sum to 1 (got {})",
            c_a + c_b
        ));
    }
    Err(anyhow!(
        "predictors::miedema::heat_of_formation: P/Q/R coefficient table import pending"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy() -> MiedemaParameters {
        MiedemaParameters {
            electronegativity: 5.2,
            n_ws_cube_root: 1.5,
            molar_volume_cm3_per_mol: 7.1,
            solute_class: 0,
        }
    }

    #[test]
    fn predictor_reports_stable_name() {
        assert_eq!(MiedemaPredictor.name(), "miedema");
    }

    #[test]
    fn heat_of_formation_validates_mole_fraction_range() {
        let a = dummy();
        let b = dummy();
        assert!(heat_of_formation(&a, &b, -0.1, 1.1).is_err());
    }

    #[test]
    fn heat_of_formation_validates_mole_fraction_sum() {
        let a = dummy();
        let b = dummy();
        assert!(heat_of_formation(&a, &b, 0.4, 0.4).is_err());
    }

    #[test]
    fn heat_of_formation_returns_unimplemented_for_valid_input() {
        let a = dummy();
        let b = dummy();
        assert!(heat_of_formation(&a, &b, 0.5, 0.5).is_err());
    }
}
