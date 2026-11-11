//! Consistency of stored elastic constants.
//!
//! An isotropic solid has two independent elastic constants. A sheet that
//! states three of E, nu, G, K can contradict itself. [`elastic_mismatch`]
//! returns the relative difference and judges nothing: it has no tolerance.

use crate::property::PropertyId as P;
use crate::table::{PropertyTable, Source};

/// Relative disagreement of a third stated constant with the E, nu pair.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ElasticMismatch {
    /// `|G - E / (2 (1 + nu))| / G`; `None` when an input is absent or nu is out of range.
    pub shear: Option<f64>,
    /// `|K - E / (3 (1 - 2 nu))| / K`; `None` when an input is absent or nu is out of range.
    pub bulk: Option<f64>,
}

/// A value the sheet itself states: a value derived at load agrees exactly
/// with the pair it came from, so it is not compared.
fn stated(t: &PropertyTable, prop: P) -> Option<f64> {
    match t.source(prop) {
        Source::Absent | Source::Derived => None,
        _ => t.get(prop),
    }
}

/// How far the stated G and K disagree with the stated E and nu.
///
/// Call it on the table as ingested: only constants the sheet states are compared.
#[must_use]
pub fn elastic_mismatch(t: &PropertyTable) -> ElasticMismatch {
    let (Some(e), Some(nu)) = (stated(t, P::YoungsModulus), stated(t, P::PoissonsRatio)) else {
        return ElasticMismatch::default();
    };
    if nu <= -1.0 || nu >= 0.5 {
        return ElasticMismatch::default();
    }
    let rel = |given: f64, from_pair: f64| (given - from_pair).abs() / given;
    ElasticMismatch {
        shear: stated(t, P::ShearModulus).map(|g| rel(g, e / (2.0 * (1.0 + nu)))),
        bulk: stated(t, P::BulkModulus).map(|k| rel(k, e / (3.0 * (1.0 - 2.0 * nu)))),
    }
}
