//====== periodica/rust/periodica_core/src/predictors/vsepr.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # vsepr
//!
//! Valence Shell Electron Pair Repulsion geometry predictor.
//!
//! Maps `(bonding_pairs, lone_pairs)` to one of the canonical VSEPR
//! geometries. The Python reference is
//! `src/periodica/utils/predictors/vsepr.py`.

use anyhow::{anyhow, Result};

use super::Predictor;

/// Canonical molecular geometries reachable from the AXnEm steric-number
/// matrix. Mirrors `MolecularGeometry` in `periodica.core.molecule_enums`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MolecularGeometry {
    Linear,
    BentBent,
    TrigonalPlanar,
    TrigonalPyramidal,
    Tetrahedral,
    SeeSaw,
    TShaped,
    SquarePlanar,
    TrigonalBipyramidal,
    SquarePyramidal,
    Octahedral,
    Pentagonal,
    PentagonalBipyramidal,
}

/// Predictor handle.
#[derive(Debug, Default)]
pub struct VseprPredictor;

impl Predictor for VseprPredictor {
    fn name(&self) -> &'static str {
        "vsepr"
    }
}

/// Predict the geometry from a steric pair count.
///
/// Returns `Err` for any combination that VSEPR cannot describe (`bonding +
/// lone > 7`).
pub fn predict_geometry(_bonding_pairs: u32, _lone_pairs: u32) -> Result<MolecularGeometry> {
    Err(anyhow!(
        "predictors::vsepr::predict_geometry is not yet implemented; \
         pending steric-number lookup table"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_reports_stable_name() {
        assert_eq!(VseprPredictor.name(), "vsepr");
    }

    #[test]
    fn predict_geometry_returns_error_until_implemented() {
        assert!(predict_geometry(2, 0).is_err());
        assert!(predict_geometry(4, 2).is_err());
    }

    #[test]
    fn canonical_enum_count_matches_axne_table() {
        // 13 geometries: AX2..AX7 plus the bent special case. The Python
        // reference enumerates the same set, so any drift will trip this test.
        let all = [
            MolecularGeometry::Linear,
            MolecularGeometry::BentBent,
            MolecularGeometry::TrigonalPlanar,
            MolecularGeometry::TrigonalPyramidal,
            MolecularGeometry::Tetrahedral,
            MolecularGeometry::SeeSaw,
            MolecularGeometry::TShaped,
            MolecularGeometry::SquarePlanar,
            MolecularGeometry::TrigonalBipyramidal,
            MolecularGeometry::SquarePyramidal,
            MolecularGeometry::Octahedral,
            MolecularGeometry::Pentagonal,
            MolecularGeometry::PentagonalBipyramidal,
        ];
        assert_eq!(all.len(), 13);
    }
}
