//====== periodica/rust/periodica-render/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # periodica-render
//!
//! Physically grounded volume rendering of atoms and molecules: probability
//! densities rendered by emission-absorption ray marching, atoms and bonds by
//! exact ray-sphere / ray-capsule intersection, and isosurfaces at physically
//! defined thresholds.
//!
//! The same [`field_program`] feeds three consumers -- the CPU reference
//! renderer, the generated GLSL ES 1.00 shaders and the accuracy tests -- so
//! a GPU frame can always be checked against a CPU golden image.
//!
//! **Status: skeleton.** The modules below fix the crate's place in the
//! workspace DAG and its public layout; plan items C1-C8 fill them in.
//!
//! ## Module map
//!
//! - [`scene`] -- what is drawn: field sources, atom spheres, bond capsules,
//!   transfer functions, and the visual-mode state machines (real-time,
//!   averaged, step, collapse).
//! - [`camera`] -- camera basis, field of view and the per-pixel ray.
//! - [`field_program`] -- `FieldProgram`, a straight-line f32 register IR
//!   (`add`, `sub`, `mul`, `div`, `exp`, `sqrt`, `select`, `const`, `px`,
//!   `py`, `pz`) with a CPU interpreter.
//! - [`glsl`] -- lowering of a `FieldProgram` and a scene to specialised
//!   GLSL ES 1.00 (`#version 100`) fragment shaders, constants baked as literals.
//! - [`cpu`] -- the CPU reference renderer and golden-image comparison.
//!
//! ## Determinism
//!
//! Transcendentals go through `libm` (workspace dependency) rather than the
//! platform maths library, so a golden image is bit-identical on every OS.
//!
//! ## What this crate deliberately is not
//!
//! It does not depend on pyo3 and talks to no GPU API: it *generates* shader
//! source and renders reference images on the CPU; uploading and drawing is the
//! app's job (Kivy). It computes no physics of its own -- orbitals and
//! densities come from `periodica-qm`, geometry from `periodica-chem`.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod camera;
pub mod cpu;
pub mod field_program;
pub mod glsl;
pub mod scene;

/// Version of this crate, for diagnostics.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    /// Smoke test: the crate builds, and its upstream edge in the DAG
    /// (`periodica-chem`) resolves.
    #[test]
    fn crate_builds_against_periodica_chem() {
        assert!(!super::CRATE_VERSION.is_empty());
        assert_eq!(periodica_chem::CRATE_VERSION, super::CRATE_VERSION);
    }
}
