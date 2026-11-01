//! # composition
//!
//! Estimates `E`, `nu` and `gamma` of a sheet that states a composition but
//! not the property, from the element sheets of its constituents. Every
//! value is [`Source::Estimated`] and goes through
//! [`PropertyTable::set_if_better`], so a stated value is never replaced.
//!
//! The mixing is [`crate::mix`]: `E` and `nu` by the Hill average over volume
//! fractions (mass fraction over element density), `gamma` by the number
//! average over atom fractions (mass fraction over atomic mass). A constituent
//! that lacks the property fails the estimate and is named in the error; there
//! is no default and no branch on what kind of material the sheet is.

use std::collections::HashMap;

use crate::mix::{mix, MixError};
use crate::property::PropertyId as P;
use crate::record::MaterialRecord;
use crate::table::Source;

/// Properties the composition chain estimates.
pub const CHAIN_PROPERTIES: [P; 3] = [P::YoungsModulus, P::PoissonsRatio, P::SurfaceEnergy];

/// Why a composition estimate could not be made.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CompositionError {
    /// The sheet states no composition.
    #[error("`{0}` has no composition")]
    NoComposition(String),
    /// A constituent has no element sheet in the table.
    #[error("no element sheet for constituent `{0}`")]
    UnknownElement(String),
    /// A constituent's element sheet lacks the property.
    #[error("constituent `{element}` has no {property:?}")]
    MissingConstituent {
        /// The element symbol.
        element: String,
        /// The property its sheet lacks.
        property: P,
    },
    /// The mix itself failed.
    #[error(transparent)]
    Mix(#[from] MixError),
}

/// Element sheets by symbol.
#[derive(Debug, Default)]
pub struct ElementTable(HashMap<String, MaterialRecord>);

impl ElementTable {
    /// Build from `(symbol, record)` pairs.
    pub fn from_records(items: impl IntoIterator<Item = (String, MaterialRecord)>) -> Self {
        Self(items.into_iter().collect())
    }

    fn get(&self, symbol: &str) -> Result<&MaterialRecord, CompositionError> {
        self.0
            .get(symbol)
            .ok_or_else(|| CompositionError::UnknownElement(symbol.to_string()))
    }
}

/// How the constituent fractions are weighted for a property.
#[derive(Clone, Copy)]
enum Weight {
    Volume,
    Atom,
}

fn weight_of(prop: P) -> Weight {
    match prop {
        P::SurfaceEnergy => Weight::Atom,
        _ => Weight::Volume,
    }
}

/// The constituents with their normalised weights; every element must carry
/// `prop` and the property its weighting needs.
fn constituents<'a>(
    rec: &MaterialRecord,
    elements: &'a ElementTable,
    prop: P,
) -> Result<Vec<(&'a MaterialRecord, f64)>, CompositionError> {
    if rec.composition.is_empty() {
        return Err(CompositionError::NoComposition(rec.name.clone()));
    }
    let divisor = match weight_of(prop) {
        Weight::Volume => P::Density,
        Weight::Atom => P::AtomicMass,
    };
    let mut parts = Vec::with_capacity(rec.composition.len());
    for (sym, mass_fraction) in &rec.composition {
        let el = elements.get(sym)?;
        let missing = |property| CompositionError::MissingConstituent { element: sym.clone(), property };
        let d = el.bulk.get(divisor).filter(|d| *d > 0.0).ok_or_else(|| missing(divisor))?;
        el.bulk.get(prop).ok_or_else(|| missing(prop))?;
        parts.push((el, mass_fraction / d));
    }
    let total: f64 = parts.iter().map(|(_, w)| w).sum();
    Ok(parts.into_iter().map(|(el, w)| (el, w / total)).collect())
}

/// The estimate of `prop` for `rec` from its constituents.
///
/// # Errors
/// No composition, an unknown constituent, a constituent lacking `prop`
/// (named), or a failed mix.
pub fn estimate(rec: &MaterialRecord, elements: &ElementTable, prop: P) -> Result<f64, CompositionError> {
    let parts = constituents(rec, elements, prop)?;
    Ok(mix(&parts, prop)?)
}

/// Fill each of [`CHAIN_PROPERTIES`] that `rec` lacks, as `Estimated`.
/// Returns the failure for each property that could not be estimated.
pub fn fill(rec: &mut MaterialRecord, elements: &ElementTable) -> Vec<(P, CompositionError)> {
    let mut failures = Vec::new();
    for prop in CHAIN_PROPERTIES {
        if rec.bulk.has(prop) {
            continue;
        }
        match estimate(rec, elements, prop) {
            Ok(v) => {
                rec.bulk.set_if_better(prop, v, Source::Estimated);
            }
            Err(e) => failures.push((prop, e)),
        }
    }
    failures
}
