//====== periodica/rust/periodica_core/src/predictors/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # predictors
//!
//! Semi-empirical predictor catalogue. Each submodule implements one model
//! whose Python reference lives under `src/periodica/utils/predictors/`.
//!
//! All predictors share a thin trait — [`Predictor`] — so the engine plugin
//! and PyO3 facade can dispatch generically (Open/Closed compliance: new
//! predictors are added without changing call-sites).

use anyhow::Result;

pub mod chou_fasman;
pub mod constituent;
pub mod miedema;
pub mod semf;
pub mod slater;
pub mod vsepr;

/// Common predictor surface. Implementors take a typed input struct and
/// return a typed output; see each submodule for concrete signatures.
pub trait Predictor {
    /// Stable identifier used by the registry (`"semf"`, `"slater"`, ...).
    fn name(&self) -> &'static str;
}

/// Registry of every predictor name shipped in this build, in stable order.
pub fn registered_predictor_names() -> &'static [&'static str] {
    &[
        "semf",
        "slater",
        "vsepr",
        "miedema",
        "chou_fasman",
        "constituent_quark",
    ]
}

/// Convenience: re-export every predictor entry-point at the module root for
/// the PyO3 facade (which prefers flat namespaces).
pub use chou_fasman::propensity as chou_fasman_propensity;
pub use constituent::predict_mass as constituent_predict_mass;
pub use miedema::heat_of_formation as miedema_heat_of_formation;
pub use semf::{binding_energy, nuclear_radius};
pub use slater::effective_charge as slater_effective_charge;
pub use vsepr::predict_geometry as vsepr_predict_geometry;

/// Smoke wrapper used by tests and by the PyO3 facade to confirm at least one
/// predictor is reachable from the registry without panicking.
pub fn ping(name: &str) -> Result<&'static str> {
    if registered_predictor_names().contains(&name) {
        Ok(name_to_static(name))
    } else {
        Err(anyhow::anyhow!("unknown predictor: {name}"))
    }
}

fn name_to_static(name: &str) -> &'static str {
    for n in registered_predictor_names() {
        if *n == name {
            return n;
        }
    }
    "unknown"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_exposes_all_six_models() {
        assert_eq!(registered_predictor_names().len(), 6);
    }

    #[test]
    fn ping_resolves_known_predictor() {
        assert_eq!(ping("semf").unwrap(), "semf");
    }

    #[test]
    fn ping_rejects_unknown_predictor() {
        assert!(ping("nope").is_err());
    }
}
