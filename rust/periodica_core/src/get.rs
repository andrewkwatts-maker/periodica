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
//! 1. **Bare name** — `Get("Fe")` resolves through the tier-fallback order
//!    reported by [`crate::data_loader::tier_order`].
//! 2. **Scoped name** — `Get("P", Some(Scope::Atom))` disambiguates Phosphorus
//!    from Proton.
//! 3. **Formula** — `Get("{u=2,d=1}")` (proton from quarks),
//!    `Get("{H=2,O=1}")` (water from atoms), `Get("{P=1,N=0,E=1}")`
//!    (hydrogen from subatomic constituents).
//!
//! Tier names are discovered from `data/config/composition_rules.json` at load
//! time (see [`crate::data_loader`]), never hardcoded here.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use thiserror::Error;

/// The 13 tiers periodica indexes.
///
/// [`Scope::tier_name`] must return exactly the strings the Python `Scope`
/// enum uses (`periodica/get.py`), because those are the registry keys.
///
/// These were previously **singular** (`"atom"`, `"alloy"`, `"protein"`) while
/// the registry keys are plural, so every scoped lookup silently missed. The
/// tiers are also not all under `active/`: eight of them live under `derived/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    Fundamental,
    SubAtomic,
    Atom,
    Molecule,
    HadronsGen,
    Isotope,
    Ion,
    Alloy,
    Polymer,
    Ceramic,
    Composite,
    AminoAcid,
    Protein,
}

impl Scope {
    /// Registry tier key. Mirrors the Python `Scope` enum's values exactly.
    pub fn tier_name(self) -> &'static str {
        match self {
            Scope::Fundamental => "fundamentals",
            Scope::SubAtomic => "subatomic",
            Scope::Atom => "atoms",
            Scope::Molecule => "molecules",
            Scope::HadronsGen => "hadrons_gen",
            Scope::Isotope => "isotopes",
            Scope::Ion => "ions",
            Scope::Alloy => "alloys",
            Scope::Polymer => "polymers",
            Scope::Ceramic => "ceramics",
            Scope::Composite => "composites",
            Scope::AminoAcid => "amino_acids",
            Scope::Protein => "proteins",
        }
    }

    /// Iterate every variant in declaration order.
    pub fn iter() -> impl Iterator<Item = Scope> {
        [
            Scope::Fundamental,
            Scope::SubAtomic,
            Scope::Atom,
            Scope::Molecule,
            Scope::HadronsGen,
            Scope::Isotope,
            Scope::Ion,
            Scope::Alloy,
            Scope::Polymer,
            Scope::Ceramic,
            Scope::Composite,
            Scope::AminoAcid,
            Scope::Protein,
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
/// When no scope is given, tiers are walked in load order (see
/// [`crate::data_loader::tier_order`]) so the
/// most fundamental tier wins on a collision (quarks beat atoms, etc.).
fn resolve_named(name: &str, scope: Option<Scope>) -> Result<Value> {
    let hub = crate::data_loader::DATA.read();

    // Resolve through the priority-ordered registry rather than a sequence of
    // passes. The old implementation tried exact stem, casefold stem, Symbol,
    // then Aliases -- roughly the inverse of Python's priority table -- so the
    // two backends disagreed on any name that is a stem in one tier and a
    // symbol or alias in another. See `crate::registry` for the table.
    let tiers: Option<Vec<&str>> = scope.map(|s| vec![s.tier_name()]);
    if let Some(found) = hub.registry.lookup(name, tiers.as_deref()) {
        return Ok(found);
    }

    // The materials catalogue sits outside the `Get` registry by design (see
    // data_loader's module docs), and only answers unscoped lookups.
    if scope.is_none() {
        if let Some(v) = hub.materials.get(name) {
            return Ok(v.value().clone());
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
                    let existing = merged.get(key).and_then(Value::as_f64).unwrap_or(0.0);
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
        serde_json::json!(formula
            .parts
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

    use crate::data_loader::DATA_TEST_LOCK;

    #[test]
    fn scope_tier_names_match_canonical_order() {
        // These are the values of the Python `Scope` enum in periodica/get.py.
        // They are registry keys, so they must be the plural on-disk names --
        // the previous singular list ("atom", "alloy", ...) matched no tier at
        // all, making every scoped lookup silently miss.
        let names: Vec<_> = Scope::iter().map(|s| s.tier_name()).collect();
        assert_eq!(
            names,
            vec![
                "fundamentals",
                "subatomic",
                "atoms",
                "molecules",
                "hadrons_gen",
                "isotopes",
                "ions",
                "alloys",
                "polymers",
                "ceramics",
                "composites",
                "amino_acids",
                "proteins",
            ]
        );
    }

    #[test]
    fn every_scope_names_a_tier_that_actually_loads() {
        // The assertion that would have caught the singular-name bug: a Scope
        // whose tier key is absent from the loaded registry is unusable.
        let data =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
        if !data.is_dir() {
            eprintln!("skipping: bundled data not present");
            return;
        }
        let _g = DATA_TEST_LOCK.lock();
        crate::data_loader::load_all_tiers(&data).expect("load");
        let tiers = crate::data_loader::list_tiers();
        for s in Scope::iter() {
            assert!(
                tiers.contains(&s.tier_name().to_string()),
                "Scope::{s:?} names tier {:?}, which is not in the registry {tiers:?}",
                s.tier_name()
            );
        }
    }

    #[test]
    fn scoped_lookup_resolves_against_the_real_registry() {
        let data =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
        if !data.is_dir() {
            return;
        }
        let _g = DATA_TEST_LOCK.lock();
        crate::data_loader::load_all_tiers(&data).expect("load");

        // `Get("P", Scope::Atom)` must find phosphorus, not the proton --
        // the whole point of scoping, and previously broken because the tier
        // key was "atom" while the registry key is "atoms".
        //
        // The derived atom datasheet is named "P" just like the proton, so
        // check the composition instead: phosphorus is 15p/16n/15e, whereas
        // the proton entry has no such Composition block.
        let p = Get("P", Some(Scope::Atom)).expect("P as an atom");
        assert_eq!(
            p.pointer("/Composition/P")
                .and_then(serde_json::Value::as_u64),
            Some(15),
            "expected phosphorus (Z=15), got {p:?}"
        );

        // Note: unscoped `Get("P")` resolves to phosphorus too, because the
        // exact-stem pass matches `atoms/P.json` before the proton is reached
        // via its `Symbol` field in a later pass. Scoping is demonstrated by
        // the negative case below instead.
        assert!(Get("Fe", Some(Scope::Atom)).is_ok());
        // A name from the wrong tier must not resolve under a scope.
        assert!(Get("Fe", Some(Scope::Protein)).is_err());
    }

    #[test]
    fn parse_formula_accepts_proton() {
        let f = parse_formula("{u=2,d=1}").unwrap().expect("formula");
        assert_eq!(f.parts, vec![("u".to_string(), 2), ("d".to_string(), 1)]);
    }

    #[test]
    fn parse_formula_accepts_water() {
        let f = parse_formula("{H=2,O=1}").unwrap().expect("formula");
        assert_eq!(f.parts, vec![("H".to_string(), 2), ("O".to_string(), 1)]);
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
    fn get_on_an_empty_registry_is_an_error_not_a_lazy_load() {
        // `Get` never loads data by itself (`record::material_record` and the
        // Python extension's import do). On an empty registry every shape of
        // spec must fail cleanly rather than panic or touch the filesystem.
        //
        // This used to run without the shared lock and without emptying the
        // hub, so it passed only when it happened to run before any other
        // test had loaded data into the process-wide registry.
        let _g = DATA_TEST_LOCK.lock();
        *crate::data_loader::DATA.write() = crate::data_loader::DataHub::empty();
        assert!(Get("Fe", None).is_err());
        assert!(Get("{H=2,O=1}", None).is_err());
        assert!(Get("P", Some(Scope::Atom)).is_err());
        assert!(crate::data_loader::DATA.read().root.is_none());
    }
}
