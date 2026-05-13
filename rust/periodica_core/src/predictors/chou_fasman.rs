//====== periodica/rust/periodica_core/src/predictors/chou_fasman.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # chou_fasman
//!
//! Chou-Fasman secondary-structure propensity predictor.
//!
//! Returns per-residue α-helix / β-sheet / turn propensities along an amino
//! acid sequence. The Python reference is
//! `src/periodica/utils/predictors/chou_fasman.py`.

use anyhow::{anyhow, Result};

use super::Predictor;

/// Per-position propensity triple `(P_α, P_β, P_turn)`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PropensityTriple {
    pub alpha: f64,
    pub beta: f64,
    pub turn: f64,
}

/// Predictor handle.
#[derive(Debug, Default)]
pub struct ChouFasmanPredictor;

impl Predictor for ChouFasmanPredictor {
    fn name(&self) -> &'static str {
        "chou_fasman"
    }
}

/// Compute the Chou-Fasman propensity triple for every residue in `sequence`.
///
/// `sequence` is the canonical 20-letter amino-acid alphabet. Unknown letters
/// trigger an error (parity with Python `KeyError`).
pub fn propensity(_sequence: &str) -> Result<Vec<PropensityTriple>> {
    Err(anyhow!(
        "predictors::chou_fasman::propensity is not yet implemented; \
         pending propensity table import"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_reports_stable_name() {
        assert_eq!(ChouFasmanPredictor.name(), "chou_fasman");
    }

    #[test]
    fn propensity_default_triple_is_zero() {
        let p = PropensityTriple::default();
        assert_eq!(p.alpha, 0.0);
        assert_eq!(p.beta, 0.0);
        assert_eq!(p.turn, 0.0);
    }

    #[test]
    fn propensity_returns_error_until_implemented() {
        assert!(propensity("MTYKLILNGKTLKGE").is_err());
    }
}
