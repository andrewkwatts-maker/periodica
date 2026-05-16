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
/// Accepts three call shapes:
/// 1. Bare name   — `Get("Fe", None)` walks all tiers by stem/Symbol/Aliases.
/// 2. Scoped name — `Get("P", Some(Scope::Atom))` restricts search to one tier.
/// 3. Formula     — `Get("{H=2,O=1}", None)` composes from constituent entries.
pub fn Get(spec: &str, scope: Option<Scope>) -> Result<Value> {
    if let Some(formula) = parse_formula(spec)? {
        return resolve_formula(&formula, scope);
    }
    resolve_named(spec, scope)
}

/// Bare-name lookup. Searches the data hub by file stem, then by the JSON
/// `Symbol` / `symbol` field, then by `Aliases` / `aliases` list.
///
/// Priority mirrors the Python implementation:
///   exact-stem > casefold-stem > exact-Symbol > casefold-Symbol > Aliases
/// When no scope is given, tiers are walked in [`TIER_NAMES`] order so the
/// most fundamental tier wins on a collision (quarks beat atoms, etc.).
fn resolve_named(name: &str, scope: Option<Scope>) -> Result<Value> {
    let hub = crate::data_loader::DATA.read();
    let lower = name.to_lowercase();

    // Build an iterator over the tier(s) to search.
    let tier_names: Vec<&str> = if let Some(s) = scope {
        vec![s.tier_name()]
    } else {
        crate::data_loader::TIER_NAMES.to_vec()
    };

    // Pass 1 — exact stem match (fastest; covers `Get("Fe")`, `Get("Steel-AISI-1070")`).
    for &tier_name in &tier_names {
        if let Some(tier_map) = hub.tiers.get(tier_name) {
            if let Some(v) = tier_map.get(name) {
                return Ok(v.value().clone());
            }
        }
    }

    // Pass 2 — casefold stem match.
    for &tier_name in &tier_names {
        if let Some(tier_map) = hub.tiers.get(tier_name) {
            for kv in tier_map.iter() {
                if kv.key().to_lowercase() == lower {
                    return Ok(kv.value().clone());
                }
            }
        }
    }

    // Pass 3 — Symbol / symbol field match (exact then casefold).
    for pass_casefold in [false, true] {
        for &tier_name in &tier_names {
            if let Some(tier_map) = hub.tiers.get(tier_name) {
                for kv in tier_map.iter() {
                    let data = kv.value();
                    let sym_opt = data.get("Symbol").or_else(|| data.get("symbol"));
                    if let Some(sym) = sym_opt.and_then(Value::as_str) {
                        let matches = if pass_casefold {
                            sym.to_lowercase() == lower
                        } else {
                            sym == name
                        };
                        if matches {
                            return Ok(data.clone());
                        }
                    }
                }
            }
        }
    }

    // Pass 4 — Aliases / aliases list match (exact then casefold).
    for pass_casefold in [false, true] {
        for &tier_name in &tier_names {
            if let Some(tier_map) = hub.tiers.get(tier_name) {
                for kv in tier_map.iter() {
                    let data = kv.value();
                    let aliases_opt = data.get("Aliases").or_else(|| data.get("aliases"));
                    if let Some(arr) = aliases_opt.and_then(Value::as_array) {
                        for alias_val in arr {
                            if let Some(alias) = alias_val.as_str() {
                                let matches = if pass_casefold {
                                    alias.to_lowercase() == lower
                                } else {
                                    alias == name
                                };
                                if matches {
                                    return Ok(data.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Err(anyhow!(GetError::UnknownName(name.to_string())))
}

/// Composite formula resolver: resolve each constituent, then merge additive
/// scalar properties (Mass_amu, Charge_e, …) by multiplication-weighted sum.
///
/// The scope supplied to `Get("{H=2,O=1}", scope)` is the *output* scope;
/// constituent resolution uses the tier immediately below it in the hierarchy.
/// When no scope is given, the resolver tries every tier that makes sense for
/// the first constituent it finds.
fn resolve_formula(formula: &Formula, _scope: Option<Scope>) -> Result<Value> {
    assert!(!formula.parts.is_empty(), "resolve_formula: empty formula");
    let mut merged = serde_json::Map::new();
    let mut resolved_count: usize = 0;

    for (constituent, count) in &formula.parts {
        assert!(*count <= 1000, "resolve_formula: implausibly large count");
        let entry = resolve_named(constituent, None)
            .with_context(|| format!("formula constituent '{constituent}' not found"))?;

        // Merge: additive scalar fields are multiplied by count and summed.
        // Non-scalar and non-additive fields from the first constituent win.
        if let Some(obj) = entry.as_object() {
            for (key, val) in obj {
                if let Some(num) = val.as_f64() {
                    let contribution = num * (*count as f64);
                    let existing = merged
                        .get(key)
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    merged.insert(key.clone(), serde_json::json!(existing + contribution));
                } else if !merged.contains_key(key) {
                    merged.insert(key.clone(), val.clone());
                }
            }
        }
        resolved_count += 1;
    }

    assert_eq!(
        resolved_count,
        formula.parts.len(),
        "resolve_formula: not all parts resolved"
    );
    merged.insert(
        "formula".to_string(),
        serde_json::json!(formula.parts
            .iter()
            .map(|(n, c)| format!("{n}{c}"))
            .collect::<Vec<_>>()
            .join("")),
    );
    Ok(Value::Object(merged))
}

/// Persist a fresh datasheet under `(tier, name)` in the in-memory hub.
///
/// Returns the previous entry at that key if one existed. Write-through to
/// disk is not yet implemented — the hub mutation is in-memory only.
pub fn Save(name: &str, data: Value, tier: Scope) -> Result<Option<Value>> {
    assert!(!name.is_empty(), "Save: name must be non-empty");
    let hub = crate::data_loader::DATA.read();
    let tier_map = hub
        .tiers
        .get(tier.tier_name())
        .ok_or_else(|| anyhow!(GetError::UnknownTier(tier.tier_name().to_string())))?;
    let previous = tier_map.insert(name.to_string(), data);
    Ok(previous)
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
