//====== periodica/rust/periodica-qm/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # periodica-qm
//!
//! Verified quantum mechanics for periodica's renderer and data: exact
//! hydrogen-like orbitals, their superpositions and time evolution, and
//! Born-rule samplers whose hit histograms converge to the rendered density.
//!
//! ```
//! use periodica_qm::{
//!     Complex, HydrogenLike, Orbital, SplitMix64, State, SuperpositionSampler, Term,
//! };
//!
//! // Hydrogen (CODATA 2022 proton mass, reduced-mass corrected).
//! let h = HydrogenLike::hydrogen();
//!
//! // The 1s + 2p_z superposition: a dipole sloshing at the Lyman-alpha beat.
//! let state = State::new(
//!     h,
//!     vec![
//!         Term::new(Complex::ONE, Orbital::real(1, 0, 0)?),
//!         Term::new(Complex::ONE, Orbital::real(2, 1, 0)?),
//!     ],
//! )?;
//! let beat_fs = periodica_qm::au_to_seconds(state.beat_periods()[0]) * 1e15;
//! assert!((beat_fs - 0.4055).abs() < 1e-4);
//!
//! // Born-rule measurements at t = 0, reproducible from the seed.
//! let sampler = SuperpositionSampler::new(&state, 0.0);
//! let mut rng = SplitMix64::new(42);
//! let hit = sampler.sample(&mut rng)?;
//! assert!(state.density(hit, 0.0) > 0.0);
//!
//! // Every result says what model produced it and how wrong that model is.
//! let label = state.model_label();
//! assert!(label.physical_error_vs_reality < 1e-3);
//! # Ok::<(), periodica_qm::QmError>(())
//! ```
//!
//! ## Module map
//!
//! - [`special`] -- Laguerre (recurrence, zeros), normalised associated
//!   Legendre, complex `Y_lm` (Condon-Shortley) and real `S_lm` (chemistry
//!   convention) harmonics, `ln Gamma`, integer-order incomplete gamma, and
//!   the Boys function `F_m(T)`.
//! - [`hydrogenic`] -- [`HydrogenLike`] atoms (charge, reduced mass),
//!   validated [`Orbital`] quantum numbers, [`HydrogenicOrbital`] evaluation
//!   (`R_nl`, `psi`, analytic `<r>`, `<r^2>`, `<1/r>`, radial nodes) and the
//!   closed-form [`RadialDistribution`].
//! - [`state`] -- [`State`]: normalised superpositions, `psi(x, t)` with
//!   phases reduced mod `2 pi` in `f64`, degenerate-group time averages, beat
//!   periods.
//! - [`sampling`] -- [`SplitMix64`] streams, the exact separable
//!   [`HydrogenicSampler`], the mixture-rejection [`SuperpositionSampler`] and
//!   the Marsaglia-Tsang [`GammaSampler`].
//! - [`label`] -- [`ModelLabel`]: model, numerical error, physical error,
//!   reference.
//! - [`quadrature`] -- Gauss-Legendre rules.
//! - [`complex`] -- the small [`Complex`] type (and why it is not
//!   `num-complex`).
//!
//! ## Units and constants
//!
//! Hartree atomic units: bohr, hartree, `hbar / E_h` for time; `hbar = m_e =
//! e = 1`. Conversions use [`periodica_mat::constants`] (CODATA 2022), the
//! single source of physical constants for the stack.
//!
//! ## Accuracy and determinism
//!
//! Everything is `f64`. Every transcendental is from `libm` (a pure-Rust musl
//! port), every random number from `periodica-runtime`'s `splitmix64`, and the
//! root finders and quadrature rules use only correctly rounded IEEE
//! operations, so results are **bit-identical across operating systems**:
//! golden fixtures and sampler streams are portable. The accuracy gates
//! (normalisation and `<r^k>` to `1e-12`, Boys to `1e-14`, harmonics
//! orthonormal to `1e-13` and cross-checked against SciPy, Kolmogorov-Smirnov
//! tests of every sampler) live in `tests/`.
//!
//! ## What this crate deliberately is not
//!
//! It has no pyo3 and no filesystem access: Python bindings live in
//! `periodica_core`, data loading in the registry. It does no rendering, owns
//! no GPU or engine types and builds no shaders (`periodica-render` turns its
//! states into field programs). It is not a general quantum-chemistry
//! package: the one-electron hydrogen-like tier here is exact; many-electron
//! atoms and molecules arrive as separately labelled model tiers (tabulated
//! RHF Slater orbitals, Gaussian-basis Hartree-Fock), never as silent
//! approximations. It does not depend on `rand`; its samplers own their
//! algorithms so streams cannot change under a dependency upgrade.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod complex;
pub mod error;
pub mod hydrogenic;
pub mod label;
mod numeric;
pub mod quadrature;
pub mod sampling;
pub mod special;
pub mod state;

pub use complex::Complex;
pub use error::QmError;
pub use hydrogenic::{HydrogenLike, HydrogenicOrbital, NuclearMass, Orbital, RadialDistribution};
pub use label::{hydrogenic_relativistic_error, ModelLabel};
pub use sampling::{
    AzimuthalDistribution, GammaSampler, HydrogenicSampler, PolarDistribution, SplitMix64,
    SuperpositionSampler,
};
pub use special::{Basis, Direction};
pub use state::{au_to_seconds, phase, seconds_to_au, EnergyGroup, State, Term};

/// Version of this crate, for diagnostics.
pub const PERIODICA_QM_VERSION: &str = env!("CARGO_PKG_VERSION");
