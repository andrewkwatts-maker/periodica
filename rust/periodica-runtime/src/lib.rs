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
//! Geometry is passed in as plain closures ([`SdfFn`], [`DensityFn`],
//! [`ToughnessFn`]) so this crate never needs an engine's SDF or expression
//! types, and an engine never needs this crate's internals.
//!
//! ```
//! use periodica_runtime::{find_fracture_paths, integrate_mass, WeakpointGrid};
//!
//! // A 0.4 m iron sphere centred in a unit box.
//! let sdf = |p: [f64; 3]| {
//!     ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2) + (p[2] - 0.5).powi(2)).sqrt() - 0.4
//! };
//! let iron = |_p: [f64; 3]| 7874.0; // kg/m^3
//! let bounds = ([0.0; 3], [1.0; 3]);
//!
//! let m = integrate_mass(&sdf, &iron, bounds, 1 << 14, 7).unwrap();
//! let exact = 7874.0 * 4.0 / 3.0 * std::f64::consts::PI * 0.4_f64.powi(3);
//! assert!((m.total_mass_kg / exact - 1.0).abs() < 0.01);
//!
//! // A uniform bar under the default (vertical) load cracks horizontally.
//! let toughness = |_p: [f64; 3]| 50.0e6; // Pa*m^0.5
//! let grid = WeakpointGrid::build(bounds, (8, 8, 8), &toughness).unwrap();
//! let planes = find_fracture_paths(&grid, None).unwrap();
//! assert_eq!(planes[0].normal, [0.0, 1.0, 0.0]);
//! ```
//!
//! ## Module map
//!
//! - [`crystal`] -- [`LatticeKind`] (the one lattice taxonomy, convertible to
//!   and from [`periodica_mat::CrystalSystem`]) and [`CrystalPrimitive`], a
//!   periodic "distance to nearest atom" SDF with FCC / BCC / HCP / diamond
//!   constructors.
//! - [`sampling`] -- deterministic low-discrepancy points: a Halton sequence
//!   with a seeded Cranley-Patterson rotation.
//! - [`mass`] -- [`integrate_mass`]: total mass, centre of mass and inertia
//!   tensor of an SDF volume, plus a real symmetric eigensolve for the
//!   principal frame.
//! - [`fracture`] -- [`WeakpointGrid`] and [`find_fracture_paths`]: A* through
//!   a fracture-toughness grid to place a crack plane where the material is
//!   weakest.
//! - [`fields`] -- the closure sampler types, and [`MaterialFields`], which
//!   lifts uniform density, toughness and crystal system off a datasheet.
//!
//! ## Units
//!
//! SI throughout, matching `periodica-mat`: metres, kg/m^3, kg*m^2 and Pa.
//! Fracture toughness is Pa*m^0.5 (not the MPa*m^0.5 that datasheets quote;
//! ingest has already converted it).
//!
//! ## Determinism
//!
//! Every function here is a pure function of its arguments. The same inputs
//! (including `rng_seed`) give bit-identical outputs on every run, and A*
//! breaks cost ties by a fixed key, so a replay or a networked peer reproduces
//! the same mass properties and the same crack.
//!
//! ## What this crate deliberately is not
//!
//! It does not depend on pyo3, does not read the filesystem, and does not know
//! about the datasheet registry: callers hand it a [`periodica_mat::MaterialRecord`]
//! or plain closures. It owns no engine types (SDF trees, expressions, GPU
//! handles) and runs no rigid-body simulation; it computes the physical
//! quantities an engine feeds into its own solver. Registry and Python live in
//! `periodica_core`; asset I/O lives in `periodica-fmt`.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod crystal;
pub mod fields;
pub mod fracture;
pub mod mass;
pub mod sampling;

pub use crystal::{
    make_bcc, make_diamond, make_fcc, make_hcp, CrystalError, CrystalPrimitive, LatticeKind,
};
pub use fields::{DensityFn, MaterialFields, SdfFn, ToughnessFn};
pub use fracture::{
    find_fracture_paths, FractureError, FracturePlane, StressField, WeakpointGrid,
    DEFAULT_FRACTURE_GRID_RESOLUTION,
};
pub use mass::{integrate_mass, predict_resting_orientation, MassDistribution, MassError};
pub use sampling::HaltonSampler;

/// Version of this crate, for diagnostics alongside `PERIODICA_MAT_VERSION`.
pub const PERIODICA_RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");
