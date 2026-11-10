//! # optics
//!
//! Spectral optics for material records: dispersion models giving n(lambda)
//! and k(lambda), black-body emission, emission lines, and the one
//! spectrum-to-RGB conversion.
//!
//! Every model is an Arithma expression built once and evaluated on demand,
//! so no formula here is hand-written float arithmetic.

pub mod cauchy;
pub mod dispersion;
pub mod drude_lorentz;
pub mod lines;
pub mod planck;
pub mod sellmeier;
pub mod spectrum;
mod xyz;

use arithma_core::expression::{ArithmaBindings, Evaluable};
use arithma_core::ArithmaExpression as E;

pub use cauchy::Cauchy;
pub use dispersion::{Dispersion, Nk};
pub use spectrum::{spectrum_to_rgb, Spectrum};

/// Why a dispersion model could not give a value.
#[derive(Debug, Clone, PartialEq)]
pub enum OpticsError {
    /// The wavelength lies outside the range the model was fitted over.
    OutOfRange {
        wavelength_m: f64,
        lo_m: f64,
        hi_m: f64,
    },
    /// The model evaluated to NaN or infinity (for example at a pole).
    NotFinite { wavelength_m: f64 },
    /// A record lacks a parameter the model needs.
    MissingParameter(&'static str),
}

impl std::fmt::Display for OpticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange {
                wavelength_m,
                lo_m,
                hi_m,
            } => write!(
                f,
                "wavelength {wavelength_m} m is outside the fitted range [{lo_m}, {hi_m}] m"
            ),
            Self::NotFinite { wavelength_m } => {
                write!(f, "model is not finite at {wavelength_m} m")
            }
            Self::MissingParameter(name) => write!(f, "missing parameter {name}"),
        }
    }
}

impl std::error::Error for OpticsError {}

/// Fails unless `lambda_m` is finite and inside `range_m` (inclusive).
pub(crate) fn check_range(lambda_m: f64, range_m: (f64, f64)) -> Result<(), OpticsError> {
    if !lambda_m.is_finite() || lambda_m < range_m.0 || lambda_m > range_m.1 {
        return Err(OpticsError::OutOfRange {
            wavelength_m: lambda_m,
            lo_m: range_m.0,
            hi_m: range_m.1,
        });
    }
    Ok(())
}

/// Passes `v` through when finite, else a `NotFinite` error.
pub(crate) fn finite(v: f64, lambda_m: f64) -> Result<f64, OpticsError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(OpticsError::NotFinite {
            wavelength_m: lambda_m,
        })
    }
}

pub(crate) fn num(v: f64) -> E {
    E::constant("", None, Some(v), false)
}

pub(crate) fn var(name: &str) -> E {
    E::var(name)
}

pub(crate) fn sum(terms: Vec<E>) -> E {
    terms.into_iter().reduce(E::add).unwrap_or_else(|| num(0.0))
}

pub(crate) fn eval(expr: &E, vars: &[(&str, f64)]) -> f64 {
    let bindings: ArithmaBindings = vars.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    expr.evaluate(&bindings).unwrap_or(f64::NAN)
}

pub(crate) fn square(x: E) -> E {
    E::mul(x.clone(), x)
}
