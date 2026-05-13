//====== periodica/rust/periodica_core/src/get.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # get
//!
//! Generic composer for periodica's data-driven scientific objects. This is
//! the Rust analog of `periodica.get.Get` / `Save` / `Scope` and accepts
//! the same three call shapes:
//!
//! 1. **Bare name** — `Get("Fe")` resolves through the tier-fallback order in
//!    [`data_loader::TIER_NAMES`].
//! 2. **Scoped name** — `Get("P", Some(Scope::Atom))` disambiguates Phosphorus
//!    from Proton.
//! 3. **Formula** — `Get("{u=2,d=1}")` (proton from quarks),
//!    `Get("{H=2,O=1}")` (water from atoms), `Get("{P=1,N=0,E=1}")`
//!    (hydrogen from subatomic constituents).
//!
//! Implementation status: scope enum and parser scaffold are complete; the
//! resolver bodies stub out until the data_loader walker lands.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use thiserror::Error;

/// The 12 tiers that periodica indexes. Each variant maps 1:1 to a folder under
/// `<package>/data/active/<tier>/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    SubAtomic,
    Atom,
    Molecule,
    Alloy,
    Ceramic,
    Composite,
    AminoAcid,
    Protein,
    Cell,
    CellComponent,
    NucleicAcid,
    BiomaterialType,
}

impl Scope {
    /// Canonical lower-snake name used as a tier key in the data hub.
    pub fn tier_name(self) -> &'static str {
        match self {
            Scope::SubAtomic => "subatomic",
            Scope::Atom => "atom",
            Scope::Molecule => "molecule",
            Scope::Alloy => "alloy",
            Scope::Ceramic => "ceramic",
            Scope::Composite => "composite",
            Scope::AminoAcid => "amino_acid",
            Scope::Protein => "protein",
            Scope::Cell => "cell",
            Scope::CellComponent => "cell_component",
            Scope::NucleicAcid => "nucleic_acid",
            Scope::BiomaterialType => "biomaterial",
        }
    }

    /// Iterate every variant in declaration order.
    pub fn iter() -> impl Iterator<Item = Scope> {
        [
            Scope::SubAtomic,
            Scope::Atom,
            Scope::Molecule,
            Scope::Alloy,
            Scope::Ceramic,
            Scope::Composite,
            Scope::AminoAcid,
            Scope::Protein,
            Scope::Cell,
            Scope::CellComponent,
            Scope::NucleicAcid,
            Scope::BiomaterialType,
        ]
        .into_iter()
    }
}

/// Domain errors mirrored from the Python public API.
#[derive(Debug, Error)]
pub enum GetError {
    #[error("unknown name: {0}")]
    UnknownName(String),
    #[error("unknown constituent {0:?} in spec")]
    UnknownConstituent(String),
    #[error("unknown tier: {0}")]
    UnknownTier(String),
    #[error("registry collision: {name} present in tiers {tiers:?}")]
    RegistryCollision { name: String, tiers: Vec<String> },
    #[error("invalid formula: {0}")]
    InvalidFormula(String),
}

/// Decomposed `{a=2,b=1}` formula. Internal value used by [`Get`] before tier
/// resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    /// Constituent name → integer count (no half-quantities at any tier).
    pub parts: Vec<(String, u32)>,
}

/// Parse a string of the form `"{name=count,name=count,...}"`.
///
/// Returns `Ok(None)` for inputs that are not formulas (i.e. bare names);
/// returns `Err` for malformed `{...}` literals.
pub fn parse_formula(spec: &str) -> Result<Option<Formula>> {
    let trimmed = spec.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return Ok(None);
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut parts: Vec<(String, u32)> = Vec::new();
    if inner.trim().is_empty() {
        return Err(anyhow!(GetError::InvalidFormula(spec.to_string())));
    }
    for raw in inner.split(',') {
        let mut kv = raw.splitn(2, '=');
        let key = kv
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!(GetError::InvalidFormula(spec.to_string())))?;
        let val = kv
            .next()
            .map(str::trim)
            .ok_or_else(|| anyhow!(GetError::InvalidFormula(spec.to_string())))?;
        let count: u32 = val
            .parse()
            .with_context(|| format!("formula count '{val}' is not a non-negative integer"))?;
        parts.push((key.to_string(), count));
    }
    Ok(Some(Formula { parts }))
}

/// Public composer. Resolves a periodica spec into its JSON datasheet form.
///
/// Status: stubbed. The resolver path will plug into `data_loader::DATA`
/// once the JSON walker lands; today the function delegates to the formula
/// parser so callers can already validate input syntax.
pub fn Get(spec: &str, scope: Option<Scope>) -> Result<Value> {
    if let Some(formula) = parse_formula(spec)? {
        return resolve_formula(&formula, scope);
    }
    resolve_named(spec, scope)
}

/// Bare-name resolver (used when the input is not a `{...}` formula).
fn resolve_named(_name: &str, _scope: Option<Scope>) -> Result<Value> {
    Err(anyhow!(
        "get::resolve_named is not yet implemented; pending data_loader walker"
    ))
}

/// Formula resolver: walk the parsed parts, resolve each constituent in the
/// scope above it, then synthesise the parent datasheet.
fn resolve_formula(_formula: &Formula, _scope: Option<Scope>) -> Result<Value> {
    Err(anyhow!(
        "get::resolve_formula is not yet implemented; pending tier composer"
    ))
}

/// Persist a fresh datasheet under `(tier, name)`. Returns the previous value
/// at that key, if any (mirrors Python's `Save(..., overwrite=False)`).
///
/// TODO: write-through to disk via a tier-specific JSON writer. For now this
/// only mutates the in-memory hub and is gated behind tests.
pub fn Save(_name: &str, _data: Value, _tier: Scope) -> Result<Option<Value>> {
    Err(anyhow!("get::Save is not yet implemented"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_tier_names_match_canonical_order() {
        let names: Vec<_> = Scope::iter().map(|s| s.tier_name()).collect();
        assert_eq!(
            names,
            vec![
                "subatomic",
                "atom",
                "molecule",
                "alloy",
                "ceramic",
                "composite",
                "amino_acid",
                "protein",
                "cell",
                "cell_component",
                "nucleic_acid",
                "biomaterial",
            ]
        );
    }

    #[test]
    fn parse_formula_accepts_proton() {
        let f = parse_formula("{u=2,d=1}").unwrap().expect("formula");
        assert_eq!(
            f.parts,
            vec![("u".to_string(), 2), ("d".to_string(), 1)]
        );
    }

    #[test]
    fn parse_formula_accepts_water() {
        let f = parse_formula("{H=2,O=1}").unwrap().expect("formula");
        assert_eq!(
            f.parts,
            vec![("H".to_string(), 2), ("O".to_string(), 1)]
        );
    }

    #[test]
    fn parse_formula_rejects_empty_braces() {
        assert!(parse_formula("{}").is_err());
    }

    #[test]
    fn parse_formula_returns_none_for_bare_name() {
        let f = parse_formula("Fe").unwrap();
        assert!(f.is_none());
    }

    #[test]
    fn parse_formula_rejects_non_integer_count() {
        assert!(parse_formula("{H=1.5}").is_err());
    }

    #[test]
    fn get_propagates_unimplemented_until_loader_lands() {
        // Scope-disambiguation path: not yet wired up — must error gracefully.
        assert!(Get("Fe", None).is_err());
        assert!(Get("{H=2,O=1}", None).is_err());
        assert!(Get("P", Some(Scope::Atom)).is_err());
    }
}
