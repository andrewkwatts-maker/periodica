//====== periodica/rust/periodica-runtime/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # periodica-runtime
//!
//! Spatial evaluation over periodica materials: the code a game engine or
//! simulator calls every frame (or every bake) once a material has been
//! ingested into a [`periodica_mat::MaterialRecord`].
//!
//! Behaviour that acts on geometry (mass integration, fracture search,
//! resting orientation) lives in Dynameis; this crate keeps material data and
//! crystal lattices.
//!
//! ## Module map
//!
//! - [`crystal`] -- [`LatticeKind`] (the one lattice taxonomy, convertible to
//!   and from [`periodica_mat::CrystalSystem`]) and [`CrystalPrimitive`], a
//!   periodic "distance to nearest atom" SDF with FCC / BCC / HCP / diamond
//!   constructors.
//! - [`sampling`] -- deterministic low-discrepancy points: a Halton sequence
//!   with a seeded Cranley-Patterson rotation.
//! - [`fields`] -- [`MaterialFields`], which lifts uniform density, toughness and
//!   crystal system off a datasheet. Mass integration and fracture search read
//!   these and live in Dynameis.
//!
//! ## Units
//!
//! SI throughout, matching `periodica-mat`: metres, kg/m^3, kg*m^2 and Pa.
//! Fracture toughness is Pa*m^0.5 (not the MPa*m^0.5 that datasheets quote;
//! ingest has already converted it).
//!
//! ## Determinism
//!
//! Every function here is a pure function of its arguments, so a replay or a
//! networked peer reproduces the same results.
//!
//! ## What this crate deliberately is not
//!
//! It does not depend on pyo3, does not read the filesystem, and does not know
//! about the datasheet registry: callers hand it a [`periodica_mat::MaterialRecord`]
//! It owns no engine types (SDF trees, expressions, GPU
//! handles) and runs no rigid-body simulation; it computes the physical
//! quantities an engine feeds into its own solver. Registry and Python live in
//! `periodica_core`; asset I/O lives in `periodica-fmt`.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod crystal;
pub mod fields;
pub mod sampling;

pub use crystal::{
    make_bcc, make_diamond, make_fcc, make_hcp, CrystalError, CrystalPrimitive, LatticeKind,
};
pub use fields::MaterialFields;
pub use sampling::HaltonSampler;

/// Version of this crate, for diagnostics alongside `PERIODICA_MAT_VERSION`.
pub const PERIODICA_RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");
