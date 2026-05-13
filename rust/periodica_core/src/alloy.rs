//====== periodica/rust/periodica_core/src/alloy.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # alloy
//!
//! Hume-Rothery alloy optimisation. Mirrors `periodica.optimize.optimize_alloy`.
//!
//! Given a base element, a candidate set, and a list of property targets,
//! returns the top-k composition vectors (mole fractions) that best match
//! the targets under Hume-Rothery feasibility constraints.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

/// One scalar alloy-property target. Weights are unitless and used in the
/// objective; tolerances are absolute and used to gate Hume-Rothery search.
#[derive(Debug, Clone)]
pub struct AlloyTarget {
    pub property: String,
    pub target_value: f64,
    pub weight: f64,
    pub tolerance: f64,
}

/// One ranked optimisation result.
#[derive(Debug, Clone)]
pub struct AlloyCandidate {
    /// Mole fraction per element symbol (sums to 1.0).
    pub composition: HashMap<String, f64>,
    /// Total weighted residual against the targets (lower is better).
    pub score: f64,
}

/// Public optimiser. Returns up to `top_k` candidates ranked by score
/// (ascending); ties are broken by insertion order to keep the output
/// deterministic across calls.
pub fn optimize_alloy(
    targets: &[AlloyTarget],
    base: &str,
    candidates: &[String],
    top_k: usize,
) -> Result<Vec<AlloyCandidate>> {
    if targets.is_empty() {
        return Err(anyhow!(
            "alloy::optimize_alloy: at least one target is required"
        ));
    }
    if candidates.is_empty() {
        return Err(anyhow!(
            "alloy::optimize_alloy: candidate set must not be empty"
        ));
    }
    if base.is_empty() {
        return Err(anyhow!("alloy::optimize_alloy: base element name required"));
    }
    let _ = top_k;
    Err(anyhow!(
        "alloy::optimize_alloy is not yet implemented; pending Hume-Rothery search port"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> AlloyTarget {
        AlloyTarget {
            property: "density_kg_m3".into(),
            target_value: 7800.0,
            weight: 1.0,
            tolerance: 100.0,
        }
    }

    #[test]
    fn validates_empty_targets() {
        let res = optimize_alloy(&[], "Fe", &["C".into()], 5);
        assert!(res.is_err());
    }

    #[test]
    fn validates_empty_candidates() {
        let res = optimize_alloy(&[target()], "Fe", &[], 5);
        assert!(res.is_err());
    }

    #[test]
    fn validates_empty_base() {
        let res = optimize_alloy(&[target()], "", &["C".into()], 5);
        assert!(res.is_err());
    }

    #[test]
    fn returns_unimplemented_until_search_lands() {
        let res = optimize_alloy(&[target()], "Fe", &["C".into(), "Cr".into()], 3);
        assert!(res.is_err());
    }
}
