//====== periodica/rust/periodica-chem/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # periodica-chem
//!
//! The chemistry layer of periodica: turning the strings people type
//! (`"Ca(OH)2"`, `"CuSO4·5H2O"`, `"SO4^2-"`) into exact compositions, and
//! holding curated 3D molecular geometry that the quantum-mechanics and
//! rendering crates consume.
//!
//! **Status: skeleton.** The modules below are placeholders that fix the crate's
//! place in the workspace DAG and its public layout. The formula engine lands
//! with plan item M5 and the geometry model with B2.
//!
//! ## Module map
//!
//! - [`formula`] -- formula grammar and parser: element counts, `()`/`[]`
//!   nesting, hydrates (`·`, `*`, `.`) with coefficients, charges
//!   (`^2-`, `2-`, `+3`), isotopes (`D`, `T`, `13C`, `[13C]`, `C-13`), and
//!   canonical Hill-order output.
//! - [`geometry`] -- `Geometry3D`: curated atomic coordinates with source,
//!   structure type (r_e / r_0 / r_s) and uncertainty; ingest, validation and
//!   bond perception from covalent radii.
//!
//! ## Units
//!
//! Lengths are stored in ångström, as the primary literature quotes them;
//! conversion to bohr for the QM crate uses the CODATA constant from
//! `periodica-mat`, never a local copy.
//!
//! ## What this crate deliberately is not
//!
//! It does not depend on pyo3, does not read the filesystem and does not know
//! about the datasheet registry: callers pass in strings and parsed JSON. It
//! computes no electronic structure (that is `periodica-qm`) and draws nothing
//! (that is `periodica-render`). SMILES, distance geometry and force fields are
//! out of scope until deferred item B6.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod formula;
pub mod geometry;

/// Version of this crate, for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    /// Smoke test: the crate builds, and its one upstream edge in the DAG
    /// (`periodica-mat`) resolves.
    #[test]
    fn crate_builds_against_periodica_mat() {
        assert!(!super::CRATE_VERSION.is_empty());
        assert_eq!(
            periodica_mat::PropertyId::from_name("Density"),
            Some(periodica_mat::PropertyId::Density)
        );
    }
}
