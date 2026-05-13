//====== periodica/rust/periodica_core/src/predictors/constituent.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # constituent
//!
//! Constituent quark-model mass predictor (De Rujula-Georgi-Glashow).
//!
//! Sums constituent quark masses plus colour-spin hyperfine corrections to
//! reproduce baryon and meson masses. The Python reference is
//! `src/periodica/utils/predictors/constituent_quark.py` (and the related
//! De-Rujula module).

use anyhow::{anyhow, Result};

use super::Predictor;

/// One of the six Standard-Model quark flavours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuarkFlavour {
    Up,
    Down,
    Strange,
    Charm,
    Bottom,
    Top,
}

/// Specification of a hadron's quark content.
///
/// Baryons supply three quarks; mesons supply one quark + one antiquark
/// (sign of the vector entry encodes antiquark: positive = quark, negative
/// = antiquark).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadronContent {
    pub composition: Vec<QuarkFlavour>,
}

/// Predictor handle.
#[derive(Debug, Default)]
pub struct ConstituentPredictor;

impl Predictor for ConstituentPredictor {
    fn name(&self) -> &'static str {
        "constituent_quark"
    }
}

/// Predicted hadron mass in MeV/c². Bridges to `physica_core` when the
/// `with-physica` feature is enabled (see `physica_bridge`).
pub fn predict_mass(_hadron: &HadronContent) -> Result<f64> {
    Err(anyhow!(
        "predictors::constituent::predict_mass is not yet implemented; \
         pending hyperfine-coupling table"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_reports_stable_name() {
        assert_eq!(ConstituentPredictor.name(), "constituent_quark");
    }

    #[test]
    fn proton_composition_round_trips() {
        let proton = HadronContent {
            composition: vec![QuarkFlavour::Up, QuarkFlavour::Up, QuarkFlavour::Down],
        };
        assert_eq!(proton.composition.len(), 3);
    }

    #[test]
    fn predict_mass_returns_error_until_implemented() {
        let proton = HadronContent {
            composition: vec![QuarkFlavour::Up, QuarkFlavour::Up, QuarkFlavour::Down],
        };
        assert!(predict_mass(&proton).is_err());
    }
}
