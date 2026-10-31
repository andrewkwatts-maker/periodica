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

pub use dispersion::{Dispersion, Nk};
pub use spectrum::{spectrum_to_rgb, Spectrum};

pub(crate) fn num(v: f64) -> E {
    E::constant("", None, Some(v), false)
}

pub(crate) fn var(name: &str) -> E {
    E::var(name)
}

pub(crate) fn sum(terms: Vec<E>) -> E {
    terms
        .into_iter()
        .reduce(E::add)
        .unwrap_or_else(|| num(0.0))
}

pub(crate) fn eval(expr: &E, vars: &[(&str, f64)]) -> f64 {
    let bindings: ArithmaBindings = vars.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    expr.evaluate(&bindings).unwrap_or(f64::NAN)
}

pub(crate) fn square(x: E) -> E {
    E::mul(x.clone(), x)
}
