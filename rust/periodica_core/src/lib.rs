//====== periodica/rust/periodica_core/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # periodica_core
//!
//! Rust core for the Periodica scientific-computation library. Mirrors the
//! Python public surface (`Get`, `Save`, `sample`, predictors, exporters)
//! while adding two upstream contributions the engine drives:
//!
//! - [`export::export_glsl`] -- GLSL 450+ analog of `export_hlsl`, emitting
//!   `vec4 SampleMaterial(vec3 pos, float scaleM)` for raymarch shaders.
//! - [`fourier_bake`] -- bakes material properties as truncated Fourier
//!   expansions on a 3D voxel grid for `sampler3D` upload.
//!
//! ## Dependency posture
//!
//! - [`pyfacade`] (feature `python`) -- PyO3 wrappers; the only entry point the
//!   Python wheel binds to (`periodica._periodica_core`).
//!
//! The `with-arithmos` / `with-physica` features were removed: they gated
//! `arithmos_bridge` / `physica_bridge` modules that were declared here but
//! never existed on disk, so enabling either feature failed to compile. If those
//! bridges are revived, add the source files first.
//!
//! ## Module map
//!
//! - [`data_loader`] -- `DataHub` registry of 12 tier-keyed datasheet maps.
//! - [`get`] -- `Get(spec, scope)` formula composer (`{u=2,d=1}`,
//!   `{H=2,O=1}`, bare names).
//! - [`sample`] -- Sub-µs `sample(name, prop, at, scale_m)` Voronoi/Worley
//!   phase dispatch.
//! - [`predictors`] -- Semi-empirical models (SEMF, Slater, VSEPR, Miedema,
//!   Chou-Fasman, constituent-quark).
//! - [`alloy`] -- Hume-Rothery alloy optimisation.
//! - [`protein`] -- Backbone construction, NeRF placement, Ramachandran
//!   classification (Kabsch RMSD pending portable SVD).
//! - [`export`] -- STL / OBJ / VTK / SDF / HLSL / GLSL emitters.
//! - [`fourier_bake`] -- Fourier-coefficient field baker.
//! - [`record`] -- [`material_record`]: spec string to a typed, SI-normalised
//!   [`periodica_mat::MaterialRecord`], for pure-Rust consumers (no Python).

#![allow(clippy::module_inception)]
#![allow(non_snake_case)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod alloy;
pub mod data_loader;
pub mod export;
pub mod fourier_bake;
pub mod get;
pub mod predictors;
pub mod protein;
pub mod record;
pub mod registry;
pub mod sample;
pub mod sample_models;

#[cfg(feature = "python")]
pub mod pyfacade;

// Re-export the most commonly used surface for ergonomic access.
pub use crate::data_loader::{DataHub, DATA};
pub use crate::export::{export_glsl, export_sdf_raw};
pub use crate::fourier_bake::{bake_fourier, FourierCoefficient, FourierFieldConfig};
pub use crate::get::{Get, Save, Scope};
pub use crate::record::material_record;
pub use crate::sample::sample;

// Re-exported so a pure-Rust consumer of `material_record` can name the types
// it returns without declaring its own `periodica-mat` dependency -- and so
// cannot end up linking a second, version-skewed copy of them.
pub use periodica_mat;
pub use periodica_mat::{CrystalSystem, MaterialRecord, PropertyId};

/// Crate-wide error type. Anyhow is used for the public API to keep the
/// surface compatible with PyO3's automatic Python-exception conversion.
pub type Result<T> = anyhow::Result<T>;

/// Library version reported to Python (`periodica._periodica_core.__version__`).
pub const PERIODICA_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_published() {
        assert!(!PERIODICA_CORE_VERSION.is_empty());
        // Major version must agree with the Python package on PyPI.
        assert!(PERIODICA_CORE_VERSION.starts_with('2'));
    }

    #[test]
    fn public_reexports_resolve() {
        // Ensure the type re-exports compile from the crate root.
        let _: Option<Scope> = None;
        let _: Option<FourierCoefficient> = None;
    }
}
