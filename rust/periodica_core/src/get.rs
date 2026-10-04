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
//! `Get(spec)`: look up a registered datasheet by name, or compose a new
//! result from registered constituents. This is the single implementation of
//! `periodica.Get`; the pure-Python original is frozen as a test oracle in
//! `tests/oracles/get_oracle.py`.
//!
//! ## Specs
//!
//! - **Bare name** -- any text without `{ } = : ,`: `"Fe"`, `"H2O"`, `"K+"`.
//!   Resolved through the priority index (see [`crate::registry`]).
//! - **Brace spec** -- `"{H=2,O=1}"`, `"{u:2,d:1}"`, `"Fe=0.99,C=0.01"`
//!   (braces optional, `=` or `:`, non-negative integer or decimal counts), or
//!   the same as a map of symbol to count.
//!
//! ## Composition
//!
//! A brace spec resolves every symbol to a registered entry, then sums the
//! rules' `additive_properties` weighted by count. The result is
//!
//! ```json
//! {"Composition": {...spec...},
//!  "Constituents": [{"Symbol": .., "Count": .., "Resolved": <Name>}, ..],
//!  "Charge_e": .., "Mass_MeVc2": .., ...}
//! ```
//!
//! Only `integral_properties` (charge, baryon and lepton number) are snapped
//! to an integer, and only within 1e-9 of one. The Python composer snapped
//! every total, which zeroed `Mass_kg` (~1e-26) in every composed entry (D1).
//!
//! ## Which entry a symbol means
//!
//! With a scope (`from_=`), each symbol is looked up in the named tiers in
//! order -- exactly as before, except that an unknown tier name is an error
//! (D8) instead of being silently ignored.
//!
//! Without a scope, the spec is resolved **as a whole** (D4): the first of the
//! rules' `resolution_levels` in which every symbol matches exactly wins. So
//! `{N=1,H=3}` is ammonia from atoms (17.031 u), where the Python composer
//! took `N` to be the neutron (4.03 u), while `{P=1,N=0,E=1}` -- `E` is not an
//! element -- is still hydrogen from particles. When no level covers the spec,
//! each symbol must resolve on its own to a single entry; a symbol that means
//! different entries in different tiers raises
//! [`GetError::AmbiguousConstituent`] rather than guessing.
//!
//! ## Errors
//!
//! Bad input is a typed [`GetError`], never a panic and never a silently
//! empty or negative composition (D7). [`GetError::kind`] names the Python
//! exception class the wrapper raises.

use std::sync::Arc;

use serde_json::{Map, Value};
use thiserror::Error;

use crate::data_loader::DataHub;
use crate::registry::{Entry, Registry};
use crate::rules::INTEGRAL_SNAP_TOLERANCE;

/// The standard registry tiers, as the Python `Scope` enum names them.
///
/// [`Scope::tier_name`] returns exactly the registry keys (plural on-disk
/// names). Scoping by an arbitrary tier name -- including ones created by
/// `Save(..., tier=...)` -- goes through [`get`] with a list of names.
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

    /// The variant whose tier is `tier` (exact registry key), if any.
    pub fn from_tier_name(tier: &str) -> Option<Scope> {
        Scope::iter().find(|s| s.tier_name() == tier)
    }
}

/// Every way `Get` can fail. [`GetError::kind`] is the Python class name.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GetError {
    /// A bare name matched nothing.
    #[error("UnknownName: {name:?}{}", scope_note(.scope))]
    UnknownName {
        name: String,
        scope: Option<Vec<String>>,
    },
    /// A spec symbol matched nothing.
    #[error("UnknownConstituent: {symbol:?}{}", scope_note(.scope))]
    UnknownConstituent {
        symbol: String,
        scope: Option<Vec<String>>,
    },
    /// A scope named a tier the registry does not have.
    #[error("UnknownTier: {tier:?} is not a registry tier; known tiers: {known:?}")]
    UnknownTier { tier: String, known: Vec<String> },
    /// Several files of one tier claim the same key.
    #[error("RegistryCollision: {key:?} is claimed by several files in tier {tier:?}: {files:?}")]
    RegistryCollision {
        key: String,
        tier: String,
        files: Vec<String>,
    },
    /// An unscoped spec symbol means different entries in different tiers,
    /// and no single resolution level covers the whole spec.
    #[error(
        "AmbiguousConstituent: {symbol:?} resolves to different entries in different tiers \
         ({candidates:?}); pass a scope (from_=...) to choose"
    )]
    AmbiguousConstituent {
        symbol: String,
        candidates: Vec<String>,
    },
    /// Malformed, empty or out-of-range input.
    #[error("InvalidSpec: {0}")]
    InvalidSpec(String),
}

fn scope_note(scope: &Option<Vec<String>>) -> String {
    match scope {
        Some(tiers) => format!(" not found in tier(s) {tiers:?}"),
        None => String::new(),
    }
}

impl GetError {
    /// The Python exception class the wrapper raises for this error:
    /// `UnknownName`, `UnknownConstituent`, `UnknownTier`, `RegistryCollision`,
    /// `AmbiguousConstituent` (all `periodica.get`) or `ValueError`.
    pub fn kind(&self) -> &'static str {
        match self {
            GetError::UnknownName { .. } => "UnknownName",
            GetError::UnknownConstituent { .. } => "UnknownConstituent",
            GetError::UnknownTier { .. } => "UnknownTier",
            GetError::RegistryCollision { .. } => "RegistryCollision",
            GetError::AmbiguousConstituent { .. } => "AmbiguousConstituent",
            GetError::InvalidSpec(_) => "ValueError",
        }
    }
}

/// A constituent count, keeping the int/float distinction of the input so
/// `Composition` echoes it back (`{H=2}` gives `2`, `{H: 2.0}` gives `2.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Count {
    Int(u64),
    Real(f64),
}

impl Count {
    pub fn value(self) -> f64 {
        match self {
            Count::Int(n) => n as f64,
            Count::Real(x) => x,
        }
    }

    fn to_json(self) -> Value {
        match self {
            Count::Int(n) => Value::from(n),
            Count::Real(x) => number(x),
        }
    }

    /// Validate a count given as a JSON number (the map form of a spec).
    pub fn from_json(symbol: &str, v: &Value) -> Result<Count, GetError> {
        let bad = |why: &str| GetError::InvalidSpec(format!("count for {symbol:?} {why}, got {v}"));
        match v {
            Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    Ok(Count::Int(u))
                } else if n.as_i64().is_some() {
                    Err(bad("must not be negative"))
                } else {
                    Count::real(n.as_f64().unwrap_or(f64::NAN)).map_err(|why| bad(why))
                }
            }
            _ => Err(bad("must be a number")),
        }
    }

    fn real(x: f64) -> Result<Count, &'static str> {
        if !x.is_finite() {
            Err("must be finite")
        } else if x < 0.0 {
            Err("must not be negative")
        } else {
            Ok(Count::Real(x))
        }
    }
}

/// A parsed brace spec: constituent symbols with their counts, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    pub parts: Vec<(String, Count)>,
}

impl Formula {
    /// Build from `(symbol, count)` pairs, rejecting empty specs, empty
    /// symbols and repeated symbols.
    pub fn new(parts: Vec<(String, Count)>) -> Result<Formula, GetError> {
        if parts.is_empty() {
            return Err(GetError::InvalidSpec(
                "spec has no constituents".to_string(),
            ));
        }
        for (i, (sym, _)) in parts.iter().enumerate() {
            if sym.is_empty() {
                return Err(GetError::InvalidSpec("empty constituent symbol".to_string()));
            }
            if parts[..i].iter().any(|(s, _)| s == sym) {
                return Err(GetError::InvalidSpec(format!(
                    "constituent {sym:?} is given more than once"
                )));
            }
        }
        Ok(Formula { parts })
    }

    /// Build from a symbol -> JSON-number map (the dict form of a spec).
    pub fn from_map(map: &[(String, Value)]) -> Result<Formula, GetError> {
        let parts = map
            .iter()
            .map(|(s, v)| Count::from_json(s, v).map(|c| (s.clone(), c)))
            .collect::<Result<Vec<_>, _>>()?;
        Formula::new(parts)
    }
}

/// Whether `spec` is a bare name rather than a brace spec: non-empty and free
/// of the spec punctuation `{ } = : ,`.
pub fn is_bare_name(spec: &str) -> bool {
    let s = spec.trim();
    !s.is_empty() && !s.contains(['{', '}', '=', ':', ','])
}

/// Parse a brace spec.
///
/// `Ok(None)` for a bare name. Accepts what the Python composer accepted --
/// optional braces, `name=count` or `name:count` pairs separated by commas,
/// integer or decimal counts -- and rejects what it silently swallowed:
/// empty specs, negative counts and repeated symbols.
pub fn parse_formula(spec: &str) -> Result<Option<Formula>, GetError> {
    let s = spec.trim();
    if s.is_empty() {
        return Err(GetError::InvalidSpec("spec is empty".to_string()));
    }
    if is_bare_name(s) {
        return Ok(None);
    }
    let inner = s
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .unwrap_or(s)
        .trim();
    let mut parts = Vec::new();
    for chunk in inner.split(',') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        parts.push(parse_chunk(chunk)?);
    }
    Formula::new(parts).map(Some)
}

fn parse_chunk(chunk: &str) -> Result<(String, Count), GetError> {
    let malformed = || {
        GetError::InvalidSpec(format!(
            "cannot parse spec fragment {chunk:?}; expected 'Symbol=count' or 'Symbol:count'"
        ))
    };
    let split = chunk.find(['=', ':']).ok_or_else(malformed)?;
    let (sym, count) = (chunk[..split].trim(), chunk[split + 1..].trim());
    if sym.is_empty() || sym.contains(|c: char| c.is_whitespace() || "{}=:,".contains(c)) {
        return Err(malformed());
    }
    let (negative, digits) = match count.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, count),
    };
    let (whole, frac) = match digits.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (digits, None),
    };
    let all_digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(whole) || frac.is_some_and(|f| !all_digits(f)) {
        return Err(malformed());
    }
    let x: f64 = digits.parse().map_err(|_| malformed())?;
    if negative && x != 0.0 {
        return Err(GetError::InvalidSpec(format!(
            "count for {sym:?} must not be negative, got {count}"
        )));
    }
    // Python: `n_int if int(n) == n else n`, so "2.0" is the integer 2.
    let c = if x.fract() == 0.0 && x < 9.007_199_254_740_992e15 {
        Count::Int(x as u64)
    } else {
        Count::real(x).map_err(|why| GetError::InvalidSpec(format!("count for {sym:?} {why}")))?
    };
    Ok((sym.to_string(), c))
}

/// The two shapes a spec can take.
#[derive(Debug, Clone, Copy)]
pub enum Spec<'a> {
    /// A bare name or a brace-spec string.
    Text(&'a str),
    /// Symbol -> count, as a Python dict passes it.
    Map(&'a [(String, Value)]),
}

/// Resolve a spec against the loaded registry and return an owned copy.
///
/// `scope` restricts resolution to the named tiers, tried in order. This is
/// the full `periodica.Get` behaviour; see the module docs.
pub fn get(spec: Spec<'_>, scope: Option<&[&str]>) -> Result<Value, GetError> {
    get_shared(spec, scope).map(|v| (*v).clone())
}

/// [`get`] without the final copy. Named lookups share the registry's
/// allocation; callers must not (and cannot) mutate it.
pub fn get_shared(spec: Spec<'_>, scope: Option<&[&str]>) -> Result<Arc<Value>, GetError> {
    let hub = crate::data_loader::DATA.read();
    if let Some(tiers) = scope {
        hub.registry.validate_scope(tiers)?;
    }
    match spec {
        Spec::Text(text) => match parse_formula(text)? {
            None => resolve_named(&hub, text.trim(), scope),
            Some(formula) => compose(&hub, &formula, scope).map(Arc::new),
        },
        Spec::Map(map) => compose(&hub, &Formula::from_map(map)?, scope).map(Arc::new),
    }
}

/// Public composer with the original Rust signature, used by
/// `material_record`, the sampler and the Python extension's `py_get`.
///
/// Errors are [`GetError`]s inside the `anyhow::Error` (downcastable).
pub fn Get(spec: &str, scope: Option<Scope>) -> anyhow::Result<Value> {
    let tiers = scope.map(|s| [s.tier_name()]);
    get(Spec::Text(spec), tiers.as_ref().map(|t| &t[..])).map_err(anyhow::Error::new)
}

/// Bare-name lookup.
fn resolve_named(hub: &DataHub, name: &str, scope: Option<&[&str]>) -> Result<Arc<Value>, GetError> {
    if let Some(found) = hub.registry.lookup(name, scope)? {
        return Ok(Arc::clone(&found.data));
    }
    // The materials catalogue sits outside the `Get` registry by design (see
    // data_loader's module docs), and only answers unscoped lookups.
    if scope.is_none() {
        if let Some(v) = hub.materials.get(name) {
            return Ok(Arc::new(v.value().clone()));
        }
    }
    Err(GetError::UnknownName {
        name: name.to_string(),
        scope: owned_scope(scope),
    })
}

fn owned_scope(scope: Option<&[&str]>) -> Option<Vec<String>> {
    scope.map(|t| t.iter().map(|s| s.to_string()).collect())
}

/// Pick the registry entry each constituent symbol means.
fn resolve_constituents(
    reg: &Registry,
    levels: &[Vec<String>],
    formula: &Formula,
    scope: Option<&[&str]>,
) -> Result<Vec<Entry>, GetError> {
    let unknown = |sym: &str| GetError::UnknownConstituent {
        symbol: sym.to_string(),
        scope: owned_scope(scope),
    };

    if let Some(tiers) = scope {
        return formula
            .parts
            .iter()
            .map(|(sym, _)| reg.lookup(sym, Some(tiers))?.ok_or_else(|| unknown(sym)))
            .collect();
    }

    // Whole-spec resolution: the first level that covers every symbol exactly.
    'levels: for level in levels {
        let tiers: Vec<&str> = level
            .iter()
            .map(String::as_str)
            .filter(|t| reg.has_tier(t))
            .collect();
        if tiers.is_empty() {
            continue;
        }
        let mut picked = Vec::with_capacity(formula.parts.len());
        for (sym, _) in &formula.parts {
            match reg.lookup_exact(sym, &tiers)? {
                Some(e) => picked.push(e),
                None => continue 'levels,
            }
        }
        return Ok(picked);
    }

    // No level covers it: every symbol must mean exactly one entry.
    formula
        .parts
        .iter()
        .map(|(sym, _)| {
            let mut found = reg.candidates(sym)?;
            match found.len() {
                0 => Err(unknown(sym)),
                1 => Ok(found.remove(0)),
                _ => Err(GetError::AmbiguousConstituent {
                    symbol: sym.clone(),
                    candidates: found
                        .iter()
                        .map(|e| format!("{} ({})", e.label(), e.display_name()))
                        .collect(),
                }),
            }
        })
        .collect()
}

/// Sum the additive properties of the resolved constituents.
fn compose(hub: &DataHub, formula: &Formula, scope: Option<&[&str]>) -> Result<Value, GetError> {
    let rules = &hub.rules;
    let entries = resolve_constituents(&hub.registry, &rules.resolution_levels, formula, scope)?;

    let additive = &rules.additive_properties;
    let mut totals = vec![0.0_f64; additive.len()];
    let mut constituents = Vec::with_capacity(entries.len());
    let mut composition = Map::new();

    for ((sym, count), entry) in formula.parts.iter().zip(&entries) {
        let data = entry.data.as_ref();
        let mut c = Map::new();
        c.insert("Symbol".into(), Value::String(sym.clone()));
        c.insert("Count".into(), count.to_json());
        c.insert(
            "Resolved".into(),
            data.get("Name")
                .cloned()
                .unwrap_or_else(|| Value::String(sym.clone())),
        );
        constituents.push(Value::Object(c));
        composition.insert(sym.clone(), count.to_json());

        let n = count.value();
        for (total, prop) in totals.iter_mut().zip(additive) {
            if let Some(v) = data.get(prop).and_then(additive_value) {
                *total += v * n;
            }
        }
    }

    let mut out = Map::new();
    out.insert("Composition".into(), Value::Object(composition));
    out.insert("Constituents".into(), Value::Array(constituents));
    for (prop, total) in additive.iter().zip(totals) {
        if !total.is_finite() {
            return Err(GetError::InvalidSpec(format!(
                "the {prop} total is not finite; counts are too large"
            )));
        }
        let snapped = (total - total.round()).abs() < INTEGRAL_SNAP_TOLERANCE;
        let v = if rules.is_integral(prop) && snapped {
            Value::from(total.round() as i64)
        } else {
            number(total)
        };
        out.insert(prop.clone(), v);
    }
    Ok(Value::Object(out))
}

/// A datasheet value as a summand. Mirrors Python's `float(v)`: numbers,
/// booleans and numeric strings count; anything else is skipped.
fn additive_value(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn number(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// Persist a fresh datasheet under `(tier, name)` in the in-memory hub only.
///
/// Legacy entry point kept for the Python extension's `py_save`. Returns the
/// previous entry at that key if one existed. It does not write to disk and
/// does not update the name index; `periodica.Save` writes files.
pub fn Save(name: &str, data: Value, tier: Scope) -> anyhow::Result<Option<Value>> {
    if name.trim().is_empty() {
        return Err(GetError::InvalidSpec("Save: name must be a non-empty string".into()).into());
    }
    let hub = crate::data_loader::DATA.read();
    let tier_map = hub
        .tiers
        .get(tier.tier_name())
        .ok_or_else(|| GetError::UnknownTier {
            tier: tier.tier_name().to_string(),
            known: hub.registry.tier_names(),
        })?;
    Ok(tier_map.insert(name.to_string(), data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::{DataHub, DATA, DATA_TEST_LOCK};
    use serde_json::json;
    use std::path::{Path, PathBuf};

    fn bundled() -> Option<PathBuf> {
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
        data.is_dir().then_some(data)
    }

    /// Run `f` against the shipped corpus, holding the shared test lock.
    fn with_corpus(f: impl FnOnce()) {
        let Some(data) = bundled() else {
            eprintln!("skipping: bundled data not present");
            return;
        };
        let _g = DATA_TEST_LOCK.lock();
        crate::data_loader::load_all_tiers(&data).expect("load");
        f();
    }

    fn text(spec: &str) -> Result<Value, GetError> {
        get(Spec::Text(spec), None)
    }

    fn scoped(spec: &str, tiers: &[&str]) -> Result<Value, GetError> {
        get(Spec::Text(spec), Some(tiers))
    }

    fn f(v: &Value, key: &str) -> f64 {
        v[key].as_f64().unwrap_or_else(|| panic!("{key} missing in {v}"))
    }

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    // ── Scope ──────────────────────────────────────────────────────────────

    #[test]
    fn scope_tier_names_match_the_python_enum() {
        let names: Vec<_> = Scope::iter().map(|s| s.tier_name()).collect();
        assert_eq!(
            names,
            vec![
                "fundamentals", "subatomic", "atoms", "molecules", "hadrons_gen", "isotopes",
                "ions", "alloys", "polymers", "ceramics", "composites", "amino_acids", "proteins",
            ]
        );
        assert_eq!(Scope::from_tier_name("atoms"), Some(Scope::Atom));
        assert_eq!(Scope::from_tier_name("atom"), None);
    }

    #[test]
    fn every_scope_names_a_tier_that_actually_loads() {
        with_corpus(|| {
            let tiers = crate::data_loader::list_tiers();
            for s in Scope::iter() {
                assert!(tiers.contains(&s.tier_name().to_string()), "{s:?} -> {tiers:?}");
            }
        });
    }

    // ── Parsing (D7) ───────────────────────────────────────────────────────

    #[test]
    fn parse_accepts_every_python_spelling() {
        let f = parse_formula("{u=2,d=1}").unwrap().unwrap();
        assert_eq!(f.parts, vec![("u".into(), Count::Int(2)), ("d".into(), Count::Int(1))]);
        let g = parse_formula(" { u : 2 , d:1 } ").unwrap().unwrap();
        assert_eq!(f, g);
        let h = parse_formula("H=2,O=1").unwrap().unwrap();
        assert_eq!(h.parts.len(), 2);
        // Trailing commas are tolerated, as before.
        assert!(parse_formula("{H=2,}").unwrap().is_some());
    }

    #[test]
    fn fractional_counts_are_accepted_and_integral_decimals_become_ints() {
        let f = parse_formula("{Fe=0.99,C=0.01}").unwrap().unwrap();
        assert_eq!(f.parts[0].1, Count::Real(0.99));
        let g = parse_formula("{H=2.0}").unwrap().unwrap();
        assert_eq!(g.parts[0].1, Count::Int(2));
        let z = parse_formula("{N=0}").unwrap().unwrap();
        assert_eq!(z.parts[0].1, Count::Int(0), "zero counts are legitimate (hydrogen has N=0)");
    }

    #[test]
    fn bare_names_are_not_specs() {
        for name in ["Fe", "H2O", "K+", "SO4^2-", "Higgs Boson", "Upsilon(1S)"] {
            assert_eq!(parse_formula(name).unwrap(), None, "{name}");
        }
    }

    #[test]
    fn bad_input_is_a_typed_error_not_a_silent_result() {
        for bad in ["", "   ", "{}", "{ }", "{,}", "{H=-2}", "{H=2,H=3}", "{H}", "{=2}",
                    "{H=two}", "{H=1e3}", "{H=.5}", "{H=1.}", "{not valid spec}"] {
            match parse_formula(bad) {
                Err(GetError::InvalidSpec(_)) => {}
                other => panic!("{bad:?} -> {other:?}"),
            }
        }
        // "-0" is zero, not negative.
        assert!(parse_formula("{H=-0}").is_ok());
    }

    #[test]
    fn map_specs_validate_their_counts() {
        let ok = Formula::from_map(&[("H".into(), json!(2)), ("O".into(), json!(1.5))]).unwrap();
        assert_eq!(ok.parts[1].1, Count::Real(1.5));
        for bad in [json!(null), json!(-1), json!(-0.5), json!("2"), json!(true), json!([1])] {
            assert!(
                matches!(Formula::from_map(&[("H".into(), bad.clone())]), Err(GetError::InvalidSpec(_))),
                "{bad}"
            );
        }
        assert!(Formula::from_map(&[]).is_err());
    }

    #[test]
    fn large_counts_are_values_not_panics() {
        // `{H=2001}` used to trip an `assert!(count <= 1000)` -- a panic that
        // reached Python as PanicException.
        with_corpus(|| {
            let v = scoped("{H=2001}", &["atoms"]).unwrap();
            assert_eq!(v["Composition"]["H"], 2001);
            assert!(f(&v, "Mass_amu") > 2000.0);
        });
    }

    // ── Composition semantics (D8, D1) ─────────────────────────────────────

    #[test]
    fn composition_reports_constituents_and_only_additive_properties() {
        with_corpus(|| {
            let v = scoped("{u=2,d=1}", &["fundamentals"]).unwrap();
            let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                vec!["Composition", "Constituents", "Charge_e", "Mass_MeVc2", "Mass_kg",
                     "Mass_amu", "BaryonNumber_B", "LeptonNumber_L"],
                "only the rules' additive properties, in rules order; no `formula` key"
            );
            assert_eq!(v["Composition"], json!({"u": 2, "d": 1}));
            assert_eq!(v["Constituents"][0], json!({"Symbol": "u", "Count": 2, "Resolved": "Up Quark"}));
            assert_eq!(v["Charge_e"], json!(1), "integral total snapped to an int");
            assert_eq!(v["BaryonNumber_B"], json!(1));
            assert_eq!(v["LeptonNumber_L"], json!(0));
            assert!((f(&v, "Mass_MeVc2") - 9.1).abs() < 1e-12);
        });
    }

    #[test]
    fn masses_are_never_snapped() {
        // D1: Mass_kg (~1e-26) is within 1e-9 of 0, so snapping every total
        // zeroed it in every composed entry.
        with_corpus(|| {
            let w = scoped("{H=2,O=1}", &["atoms"]).unwrap();
            let kg = f(&w, "Mass_kg");
            assert!(kg > 2.9e-26 && kg < 3.0e-26, "{kg}");
            assert!(rel(kg / f(&w, "Mass_amu"), 1.66053906660e-27) < 1e-9);
            assert!(w["Mass_amu"].is_f64());
        });
    }

    #[test]
    fn fractional_alloy_specs_compose() {
        with_corpus(|| {
            let a = scoped("{Fe:0.99, C:0.01}", &["atoms"]).unwrap();
            assert_eq!(a["Composition"], json!({"Fe": 0.99, "C": 0.01}));
            let want = 0.99 * 55.845 + 0.01 * 12.011;
            assert!(rel(f(&a, "Mass_amu"), want) < 1e-12);
        });
    }

    #[test]
    fn scoped_specs_resolve_only_in_the_named_tiers() {
        with_corpus(|| {
            let h = scoped("{P=1,N=0,E=1}", &["subatomic", "fundamentals"]).unwrap();
            assert_eq!(h["Charge_e"], json!(0));
            assert!(rel(f(&h, "Mass_amu"), 1.007825) < 1e-5);
            match scoped("{u=2,d=1}", &["atoms"]) {
                Err(GetError::UnknownConstituent { symbol, scope }) => {
                    assert_eq!(symbol, "d");
                    assert_eq!(scope, Some(vec!["atoms".to_string()]));
                }
                other => panic!("{other:?}"),
            }
        });
    }

    #[test]
    fn an_unknown_scope_is_an_error_not_an_unscoped_search() {
        with_corpus(|| {
            for spec in ["{H=2,O=1}", "Fe"] {
                match scoped(spec, &["atom"]) {
                    Err(GetError::UnknownTier { tier, known }) => {
                        assert_eq!(tier, "atom");
                        assert!(known.contains(&"atoms".to_string()));
                    }
                    other => panic!("{spec}: {other:?}"),
                }
            }
            assert!(matches!(scoped("Fe", &["atoms", "nope"]), Err(GetError::UnknownTier { .. })));
        });
    }

    // ── Whole-spec resolution (D4) ─────────────────────────────────────────

    #[test]
    fn element_symbols_in_unscoped_specs_mean_elements() {
        with_corpus(|| {
            let nh3 = text("{N=1,H=3}").unwrap();
            assert!(rel(f(&nh3, "Mass_amu"), 17.031) < 1e-4, "{nh3}");
            assert_eq!(nh3["Constituents"][0]["Resolved"], "N");
            let acid = text("{H=3,P=1,O=4}").unwrap();
            assert_eq!(acid["Charge_e"], json!(0), "phosphoric acid, not 3 H + a proton");
            assert!(rel(f(&acid, "Mass_amu"), 97.994) < 1e-4);
        });
    }

    #[test]
    fn particle_specs_still_resolve_to_particles() {
        with_corpus(|| {
            let h = text("{P=1,N=0,E=1}").unwrap();
            assert_eq!(h["Constituents"][0]["Resolved"], "Proton");
            assert_eq!(h["Constituents"][2]["Resolved"], "Electron");
            let p = text("{u=2,d=1}").unwrap();
            assert_eq!(p["Charge_e"], json!(1));
        });
    }

    #[test]
    fn an_ambiguous_symbol_in_a_mixed_spec_is_reported() {
        with_corpus(|| {
            // H2O lives only in molecules; P is a proton and phosphorus.
            match text("{P=1,H2O=1}") {
                Err(GetError::AmbiguousConstituent { symbol, candidates }) => {
                    assert_eq!(symbol, "P");
                    assert!(candidates.len() >= 2, "{candidates:?}");
                }
                other => panic!("{other:?}"),
            }
            // A symbol that means one thing composes fine in a mixed spec.
            let ok = text("{H2O=2,Na+=1}").unwrap();
            assert_eq!(ok["Charge_e"], json!(1));
            assert!(matches!(text("{Xxxxxxx=1}"), Err(GetError::UnknownConstituent { .. })));
        });
    }

    // ── Names ──────────────────────────────────────────────────────────────

    #[test]
    fn named_lookups_resolve_like_python() {
        with_corpus(|| {
            assert_eq!(text("P").unwrap()["Name"], "Proton");
            assert_eq!(text("N").unwrap()["Name"], "Neutron");
            assert_eq!(text("E").unwrap()["Name"], "Electron");
            assert_eq!(text("H").unwrap()["Symbol"], "H");
            assert_eq!(text("Higgs").unwrap()["Name"], "Higgs Boson");
            let p = scoped("P", &["atoms"]).unwrap();
            assert_eq!(p["Composition"]["P"], 15, "scoped P is phosphorus");
            assert!(scoped("Fe", &["proteins"]).is_err());
        });
    }

    #[test]
    fn k_plus_is_the_potassium_ion_and_the_kaon_is_still_reachable() {
        with_corpus(|| {
            let k = text("K+").unwrap();
            assert_eq!(k["Composition"]["P"], 19, "{k}");
            assert_eq!(k["Charge_e"], json!(1));
            let kaon = scoped("K+", &["subatomic"]).unwrap();
            assert_eq!(kaon["Name"], "Kaon+");
            assert_eq!(text("Upsilon(1S)").unwrap()["Name"], "Upsilon");
            assert_eq!(text("Y").unwrap()["Composition"]["P"], 39, "Y is yttrium");
        });
    }

    #[test]
    fn results_are_fresh_copies() {
        // D5: mutating a result must not reach the registry.
        with_corpus(|| {
            let mut a = text("Fe").unwrap();
            a["Composition"]["P"] = json!(999);
            assert_eq!(text("Fe").unwrap()["Composition"]["P"], 26);
        });
    }

    #[test]
    fn unknown_names_and_bad_specs_carry_their_python_class() {
        with_corpus(|| {
            let e = text("Imaginary_Element_Q3").unwrap_err();
            assert_eq!(e.kind(), "UnknownName");
            assert!(e.to_string().starts_with("UnknownName:"));
            assert_eq!(text("{}").unwrap_err().kind(), "ValueError");
            assert_eq!(text("").unwrap_err().kind(), "ValueError");
            let map: Vec<(String, Value)> = vec![("H".into(), Value::Null)];
            assert_eq!(get(Spec::Map(&map), None).unwrap_err().kind(), "ValueError");
        });
    }

    #[test]
    fn the_legacy_signature_wraps_the_typed_error() {
        with_corpus(|| {
            let err = Get("{H=-2}", Some(Scope::Atom)).unwrap_err();
            assert!(matches!(err.downcast_ref::<GetError>(), Some(GetError::InvalidSpec(_))));
            assert!(Get("P", Some(Scope::Atom)).is_ok());
        });
    }

    #[test]
    fn a_map_spec_matches_the_string_spec() {
        with_corpus(|| {
            let map: Vec<(String, Value)> = vec![("H".into(), json!(2)), ("O".into(), json!(1))];
            let a = get(Spec::Map(&map), Some(&["atoms"])).unwrap();
            let b = scoped("{H=2,O=1}", &["atoms"]).unwrap();
            assert_eq!(a, b);
        });
    }

    #[test]
    fn get_on_an_empty_registry_is_an_error_not_a_lazy_load() {
        // `Get` never loads data by itself (`record::material_record` and the
        // Python extension's import do). On an empty registry every shape of
        // spec must fail cleanly rather than panic or touch the filesystem.
        let _g = DATA_TEST_LOCK.lock();
        *DATA.write() = DataHub::empty();
        assert!(Get("Fe", None).is_err());
        assert!(Get("{H=2,O=1}", None).is_err());
        assert!(Get("P", Some(Scope::Atom)).is_err());
        assert!(Get("", None).is_err());
        assert!(DATA.read().root.is_none());
    }

    #[test]
    fn the_shipped_corpus_has_no_in_tier_collisions() {
        // D6: AsparticAcid/Aspartic_Acid and GlutamicAcid/Glutamic_Acid both
        // claimed their one-letter code; which won depended on the platform.
        with_corpus(|| {
            let collisions = DATA.read().registry.collisions();
            assert!(collisions.is_empty(), "{collisions:?}");
        });
    }
}
