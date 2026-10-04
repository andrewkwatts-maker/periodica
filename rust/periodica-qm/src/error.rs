//====== periodica/rust/periodica-qm/src/error.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! The crate's error type. Invalid input is reported, never silently clamped
//! and never a panic.

use thiserror::Error;

/// Everything that can go wrong constructing or sampling a quantum state.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum QmError {
    /// Quantum numbers outside `n >= 1, 0 <= l < n, |m| <= l`.
    #[error("invalid quantum numbers n={n}, l={l}, m={m}: need n >= 1, 0 <= l < n, |m| <= l")]
    InvalidQuantumNumbers {
        /// Principal quantum number as given.
        n: u32,
        /// Orbital angular momentum as given.
        l: u32,
        /// Magnetic quantum number as given.
        m: i32,
    },
    /// Nuclear charge not finite and positive.
    #[error("nuclear charge must be finite and > 0, got {0}")]
    InvalidCharge(f64),
    /// Nuclear mass not finite and positive.
    #[error("nuclear mass must be finite and > 0 (in electron masses), got {0}")]
    InvalidNuclearMass(f64),
    /// A superposition was built with no terms.
    #[error("a state needs at least one term")]
    EmptyState,
    /// A coefficient was NaN or infinite.
    #[error("coefficient {index} is not finite")]
    NonFiniteCoefficient {
        /// Index of the offending term.
        index: usize,
    },
    /// The terms cancel (or are all zero), so the state cannot be normalised.
    #[error("the state has zero norm and cannot be normalised")]
    ZeroNorm,
    /// Gamma-distribution parameters not finite and positive.
    #[error("gamma distribution needs finite shape > 0 and scale > 0, got shape {shape}, scale {scale}")]
    InvalidGammaParameters {
        /// Shape `k` as given.
        shape: f64,
        /// Scale `theta` as given.
        scale: f64,
    },
    /// The superposition rejection sampler hit its attempt cap.
    #[error("rejection sampler made {0} attempts without accepting a sample")]
    RejectionLimit(u64),
}
