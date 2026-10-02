//====== periodica/rust/periodica_core/src/registry.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # registry
//!
//! Name resolution for `Get`. Originally a faithful port of the index in the
//! pure-Python `periodica/get.py` (frozen as `tests/oracles/get_oracle.py`);
//! it is now the only implementation.
//!
//! ## The priority table
//!
//! Every datasheet contributes several keys at different priorities, and the
//! highest-priority claimant of a key wins:
//!
//! | Key source | active tier | derived tier |
//! |---|---:|---:|
//! | `Aliases` / `aliases` | 5 | 4 |
//! | `Symbol` / `symbol` | 1 | **3** |
//! | file stem | 2 | 0 |
//!
//! Note the inversion: a *derived* symbol (3) outranks an *active* stem (2),
//! which outranks an *active* symbol (1). That is what lets the generated
//! hydrogen atom (`Symbol: "H"`) shadow the Higgs boson's `H`, and the
//! generated potassium ion (`Symbol: "K+"`) shadow the kaon's.
//!
//! Ties are resolved **first-writer-wins** (`<`, not `<=`). Insertion order is
//! tier order (the rules' `tier_definitions`, then the `derived/`
//! subdirectories by name), then the file listing sorted by file name --
//! byte order, so the result no longer depends on the platform. Python sorted
//! `Path` objects, which compare case-insensitively on Windows only, so the
//! same tree resolved differently on Windows and Linux (D6).
//!
//! ## Collisions
//!
//! Two files *in the same tier* claiming the same key is a data error, not a
//! priority contest. The first file (in sorted order) keeps the key -- the
//! rule Python applied silently -- but the key is recorded as contested, and
//! any lookup that lands on it raises [`GetError::RegistryCollision`] naming
//! every claimant. The rest of the tier stays usable: one bad datasheet does
//! not take the registry down.
//!
//! ## Sharing
//!
//! Each datasheet is parsed once and held as an `Arc<Value>`; every key it
//! claims points at the same allocation. Callers that need an owned value
//! (the Python boundary, `Get`) clone it, so nothing a caller does to a result
//! can reach back into the registry (D5).

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::get::GetError;

/// Key priorities. See the module docs for the table.
pub const PRIORITY_ALIAS_ACTIVE: i32 = 5;
pub const PRIORITY_ALIAS_DERIVED: i32 = 4;
pub const PRIORITY_SYMBOL_DERIVED: i32 = 3;
pub const PRIORITY_STEM_ACTIVE: i32 = 2;
pub const PRIORITY_SYMBOL_ACTIVE: i32 = 1;
pub const PRIORITY_STEM_DERIVED: i32 = 0;

/// One registered datasheet.
#[derive(Debug)]
pub struct Record {
    /// The tier the datasheet belongs to.
    pub tier: String,
    /// Its file stem, which identifies it within the tier.
    pub stem: String,
    /// The parsed datasheet, shared by every key it claims.
    pub data: Arc<Value>,
}

impl Record {
    /// `tier:stem`, for error messages.
    pub fn label(&self) -> String {
        format!("{}:{}", self.tier, self.stem)
    }

    /// Display name: `Name`, else `name`, else the stem.
    pub fn display_name(&self) -> String {
        self.data
            .get("Name")
            .or_else(|| self.data.get("name"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| self.stem.clone())
    }
}

/// A shared handle to a registered datasheet.
pub type Entry = Arc<Record>;

#[derive(Debug, Clone)]
struct Claim {
    priority: i32,
    /// The exact key this claim was made under (relevant for casefold claims).
    key: String,
    entry: Entry,
}

/// Exact and casefold lookup tables for one tier (or the merged index).
#[derive(Debug, Default, Clone)]
pub struct TierIndex {
    exact: HashMap<String, Claim>,
    casefold: HashMap<String, Claim>,
}

impl TierIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Claim `key` for `entry` at priority `prio`. First writer wins a tie.
    fn claim(&mut self, key: &str, prio: i32, entry: &Entry) {
        let claim = || Claim {
            priority: prio,
            key: key.to_string(),
            entry: Arc::clone(entry),
        };
        if self.exact.get(key).map_or(true, |c| c.priority < prio) {
            self.exact.insert(key.to_string(), claim());
        }
        let cf = casefold(key);
        if self.casefold.get(&cf).map_or(true, |c| c.priority < prio) {
            self.casefold.insert(cf, claim());
        }
    }

    fn exact_claim(&self, name: &str) -> Option<&Claim> {
        self.exact.get(name)
    }

    fn casefold_claim(&self, name: &str) -> Option<&Claim> {
        self.casefold.get(&casefold(name))
    }

    /// Exact match, then casefold.
    fn claim_for(&self, name: &str) -> Option<&Claim> {
        self.exact_claim(name).or_else(|| self.casefold_claim(name))
    }

    /// Exact match, then casefold, ignoring collisions. Diagnostics only.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.claim_for(name).map(|c| c.entry.data.as_ref())
    }

    /// Every exactly-registered key, sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.exact.keys().cloned().collect();
        k.sort();
        k
    }

    pub fn len(&self) -> usize {
        self.exact.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exact.is_empty()
    }
}

/// Keys claimed by more than one file within a tier.
#[derive(Debug, Default, Clone)]
struct TierState {
    index: TierIndex,
    /// Key -> stem of the file that holds it, for duplicate detection.
    seen: HashMap<String, String>,
    /// Key -> every stem claiming it, in sorted file order.
    contested: HashMap<String, Vec<String>>,
}

/// Per-tier indices plus the merged cross-tier index.
#[derive(Debug, Default, Clone)]
pub struct Registry {
    tiers: HashMap<String, TierState>,
    /// Tier names in insertion (= resolution) order.
    order: Vec<String>,
    /// Used for unscoped name lookups: a priority contest across every tier.
    merged: TierIndex,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a tier, so it exists (and is listed) even with no entries.
    pub fn add_tier(&mut self, tier: &str) {
        if !self.tiers.contains_key(tier) {
            self.tiers.insert(tier.to_string(), TierState::default());
            self.order.push(tier.to_string());
        }
    }

    /// Register one datasheet under `tier`. Returns the shared handle.
    ///
    /// Must be called in sorted file order within a tier: a key already held
    /// by an earlier file of the same tier is not re-claimed, and is recorded
    /// as contested instead.
    pub fn add_entry(&mut self, tier: &str, stem: &str, data: Arc<Value>, is_active: bool) -> Entry {
        self.add_tier(tier);
        let entry = Arc::new(Record {
            tier: tier.to_string(),
            stem: stem.to_string(),
            data,
        });
        let state = self.tiers.get_mut(tier).expect("tier was just added");
        for (key, prio) in entry_keys(stem, &entry.data, is_active) {
            match state.seen.get(&key) {
                Some(holder) if holder != stem => {
                    let claimants = state
                        .contested
                        .entry(key.clone())
                        .or_insert_with(|| vec![holder.clone()]);
                    if !claimants.iter().any(|s| s == stem) {
                        claimants.push(stem.to_string());
                    }
                    continue;
                }
                _ => {
                    state.seen.insert(key.clone(), stem.to_string());
                }
            }
            state.index.claim(&key, prio, &entry);
            self.merged.claim(&key, prio, &entry);
        }
        entry
    }

    /// Whether `tier` is registered.
    pub fn has_tier(&self, tier: &str) -> bool {
        self.tiers.contains_key(tier)
    }

    /// Registered tier names, in resolution order.
    pub fn tier_order(&self) -> &[String] {
        &self.order
    }

    /// Registered tier names, sorted (what `list_tiers()` reports).
    pub fn tier_names(&self) -> Vec<String> {
        let mut names = self.order.clone();
        names.sort();
        names
    }

    /// The index of one tier.
    pub fn tier_index(&self, tier: &str) -> Option<&TierIndex> {
        self.tiers.get(tier).map(|s| &s.index)
    }

    /// Every in-tier collision: `(tier, key, claimants)`, sorted.
    pub fn collisions(&self) -> Vec<(String, String, Vec<String>)> {
        let mut out: Vec<_> = self
            .tiers
            .iter()
            .flat_map(|(t, s)| {
                s.contested
                    .iter()
                    .map(move |(k, v)| (t.clone(), k.clone(), v.clone()))
            })
            .collect();
        out.sort();
        out
    }

    /// Fail if `claim` landed on a key that several files of its tier claim.
    fn checked(&self, claim: &Claim) -> Result<Entry, GetError> {
        let tier = &claim.entry.tier;
        if let Some(files) = self.tiers.get(tier).and_then(|s| s.contested.get(&claim.key)) {
            return Err(GetError::RegistryCollision {
                key: claim.key.clone(),
                tier: tier.clone(),
                files: files.clone(),
            });
        }
        Ok(Arc::clone(&claim.entry))
    }

    /// Resolve a name: unscoped through the merged priority index, or scoped
    /// by trying each named tier in order (exact, then casefold, per tier).
    ///
    /// Unknown tiers in `scope` are the caller's to reject (see
    /// [`Self::validate_scope`]); here they simply match nothing.
    pub fn lookup(&self, name: &str, scope: Option<&[&str]>) -> Result<Option<Entry>, GetError> {
        match scope {
            None => self.merged.claim_for(name).map(|c| self.checked(c)).transpose(),
            Some(tiers) => {
                for tier in tiers {
                    if let Some(claim) = self.tiers.get(*tier).and_then(|s| s.index.claim_for(name)) {
                        return self.checked(claim).map(Some);
                    }
                }
                Ok(None)
            }
        }
    }

    /// Exact-case match in the first of `tiers` that has one.
    pub fn lookup_exact(&self, name: &str, tiers: &[&str]) -> Result<Option<Entry>, GetError> {
        for tier in tiers {
            if let Some(claim) = self.tiers.get(*tier).and_then(|s| s.index.exact_claim(name)) {
                return self.checked(claim).map(Some);
            }
        }
        Ok(None)
    }

    /// Every tier's resolution of `name`, in tier order: exact matches if any
    /// tier has one, otherwise casefold matches. Used to detect a constituent
    /// symbol that means different things in different tiers.
    pub fn candidates(&self, name: &str) -> Result<Vec<Entry>, GetError> {
        let collect = |pick: &dyn Fn(&TierIndex) -> Option<&Claim>| -> Result<Vec<Entry>, GetError> {
            let mut out: Vec<Entry> = Vec::new();
            for tier in &self.order {
                if let Some(claim) = self.tiers.get(tier).and_then(|s| pick(&s.index)) {
                    let entry = self.checked(claim)?;
                    if !out.iter().any(|e| Arc::ptr_eq(e, &entry)) {
                        out.push(entry);
                    }
                }
            }
            Ok(out)
        };
        let exact = collect(&|idx| idx.exact_claim(name))?;
        if !exact.is_empty() {
            return Ok(exact);
        }
        collect(&|idx| idx.casefold_claim(name))
    }

    /// Reject a scope naming a tier that is not registered (D8: these used to
    /// be ignored, so a typo silently searched nothing or everything).
    pub fn validate_scope(&self, scope: &[&str]) -> Result<(), GetError> {
        if scope.is_empty() {
            return Err(GetError::InvalidSpec(
                "scope is an empty list of tiers".to_string(),
            ));
        }
        for tier in scope {
            if !self.has_tier(tier) {
                return Err(GetError::UnknownTier {
                    tier: (*tier).to_string(),
                    known: self.tier_names(),
                });
            }
        }
        Ok(())
    }

    /// Number of distinct keys across the corpus.
    pub fn total_keys(&self) -> usize {
        self.merged.len()
    }

    /// The merged index (diagnostics).
    pub fn merged(&self) -> &TierIndex {
        &self.merged
    }
}

/// Case-insensitive key. Lower-casing, as Python's `str.casefold` does for
/// every symbol in the corpus.
fn casefold(key: &str) -> String {
    key.to_lowercase()
}

/// The `(key, priority)` pairs one entry contributes.
///
/// Both PascalCase and snake_case field names are accepted, because the amino
/// acid datasheets use lowercase `symbol` / `aliases`. Duplicate keys within
/// one entry (an alias equal to the stem) keep their highest priority.
pub fn entry_keys(stem: &str, data: &Value, is_active: bool) -> Vec<(String, i32)> {
    let mut out: Vec<(String, i32)> = Vec::new();
    let mut push = |key: &str, prio: i32| match out.iter_mut().find(|(k, _)| k == key) {
        Some(existing) => existing.1 = existing.1.max(prio),
        None => out.push((key.to_string(), prio)),
    };

    if let Some(sym) = data
        .get("Symbol")
        .or_else(|| data.get("symbol"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        push(
            sym,
            if is_active {
                PRIORITY_SYMBOL_ACTIVE
            } else {
                PRIORITY_SYMBOL_DERIVED
            },
        );
    }

    if let Some(aliases) = data
        .get("Aliases")
        .or_else(|| data.get("aliases"))
        .and_then(Value::as_array)
    {
        let prio = if is_active {
            PRIORITY_ALIAS_ACTIVE
        } else {
            PRIORITY_ALIAS_DERIVED
        };
        for a in aliases {
            if let Some(s) = a.as_str().filter(|s| !s.is_empty()) {
                push(s, prio);
            }
        }
    }

    push(
        stem,
        if is_active {
            PRIORITY_STEM_ACTIVE
        } else {
            PRIORITY_STEM_DERIVED
        },
    );
    out
}

/// Whether a file stem is a placeholder that the registry skips.
///
/// `prefixes` are the rules' `placeholder_prefixes`, already lower-cased.
pub fn is_placeholder(stem: &str, prefixes: &[String]) -> bool {
    let lower = stem.to_lowercase();
    prefixes.iter().any(|p| lower.starts_with(p.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn add(reg: &mut Registry, tier: &str, stem: &str, data: Value, active: bool) -> Entry {
        reg.add_entry(tier, stem, Arc::new(data), active)
    }

    fn name_of(e: &Entry) -> String {
        e.display_name()
    }

    #[test]
    fn priority_table_is_pinned() {
        // The inversion -- derived symbol (3) above active stem (2) above
        // active symbol (1) -- is deliberate and easy to get wrong.
        assert_eq!(PRIORITY_ALIAS_ACTIVE, 5);
        assert_eq!(PRIORITY_ALIAS_DERIVED, 4);
        assert_eq!(PRIORITY_SYMBOL_DERIVED, 3);
        assert_eq!(PRIORITY_STEM_ACTIVE, 2);
        assert_eq!(PRIORITY_SYMBOL_ACTIVE, 1);
        assert_eq!(PRIORITY_STEM_DERIVED, 0);
    }

    #[test]
    fn entry_keys_cover_symbol_aliases_and_stem() {
        let data = json!({"Symbol": "Fe", "Aliases": ["Iron", "Ferrum"]});
        let keys = entry_keys("Fe-56", &data, false);
        let map: HashMap<&str, i32> = keys.iter().map(|(k, p)| (k.as_str(), *p)).collect();
        assert_eq!(map["Fe"], PRIORITY_SYMBOL_DERIVED);
        assert_eq!(map["Iron"], PRIORITY_ALIAS_DERIVED);
        assert_eq!(map["Ferrum"], PRIORITY_ALIAS_DERIVED);
        assert_eq!(map["Fe-56"], PRIORITY_STEM_DERIVED);
    }

    #[test]
    fn a_key_repeated_within_one_entry_is_not_a_collision() {
        // The generated atom "H" claims "H" as its stem and as its Symbol.
        let mut reg = Registry::new();
        add(&mut reg, "atoms", "H", json!({"Symbol": "H"}), false);
        assert!(reg.collisions().is_empty());
        assert!(reg.lookup("H", None).unwrap().is_some());
        let keys = entry_keys("H", &json!({"Symbol": "H"}), false);
        assert_eq!(keys, vec![("H".to_string(), PRIORITY_SYMBOL_DERIVED)]);
    }

    #[test]
    fn snake_case_fields_are_accepted() {
        let data = json!({"symbol": "E", "aliases": ["Glu"]});
        let keys = entry_keys("GlutamicAcid", &data, true);
        let map: HashMap<&str, i32> = keys.iter().map(|(k, p)| (k.as_str(), *p)).collect();
        assert_eq!(map["E"], PRIORITY_SYMBOL_ACTIVE);
        assert_eq!(map["Glu"], PRIORITY_ALIAS_ACTIVE);
    }

    #[test]
    fn higher_priority_claims_a_contested_key_across_tiers() {
        let mut reg = Registry::new();
        add(&mut reg, "a", "X", json!({"Name": "stem"}), false);
        add(&mut reg, "b", "Other", json!({"Name": "alias", "Aliases": ["X"]}), true);
        let got = reg.lookup("X", None).unwrap().unwrap();
        assert_eq!(name_of(&got), "alias");
    }

    #[test]
    fn equal_priority_across_tiers_is_first_writer_wins() {
        let mut reg = Registry::new();
        add(&mut reg, "a", "One", json!({"Name": "first", "Symbol": "X"}), true);
        add(&mut reg, "b", "Two", json!({"Name": "second", "Symbol": "X"}), true);
        assert_eq!(name_of(&reg.lookup("X", None).unwrap().unwrap()), "first");
    }

    #[test]
    fn casefold_lookup_falls_back_after_exact() {
        let mut reg = Registry::new();
        add(&mut reg, "atoms", "Fe", json!({"n": 1}), false);
        for q in ["Fe", "fe", "FE"] {
            assert!(reg.lookup(q, None).unwrap().is_some(), "{q}");
        }
        assert!(reg.lookup("Zz", None).unwrap().is_none());
    }

    #[test]
    fn the_e_collision_resolves_to_the_electron() {
        // Electron (Symbol "e-", alias "E", fundamentals) against glutamic
        // acid (symbol "E", amino_acids): the active alias outranks the
        // active symbol, and fundamentals is registered first anyway.
        let mut reg = Registry::new();
        add(&mut reg, "fundamentals", "Electron", json!({"Name": "Electron", "Aliases": ["E"]}), true);
        add(&mut reg, "amino_acids", "GlutamicAcid", json!({"name": "Glutamic acid", "symbol": "E"}), true);
        assert_eq!(name_of(&reg.lookup("E", None).unwrap().unwrap()), "Electron");
        let scoped = reg.lookup("E", Some(&["amino_acids"])).unwrap().unwrap();
        assert_eq!(name_of(&scoped), "Glutamic acid");
    }

    #[test]
    fn scoped_lookup_only_searches_named_tiers() {
        let mut reg = Registry::new();
        add(&mut reg, "atoms", "Fe", json!({"Symbol": "Fe"}), false);
        reg.add_tier("proteins");
        assert!(reg.lookup("Fe", Some(&["atoms"])).unwrap().is_some());
        assert!(reg.lookup("Fe", Some(&["proteins"])).unwrap().is_none());
    }

    #[test]
    fn in_tier_duplicates_are_reported_on_lookup_not_silently_resolved() {
        // D6: AsparticAcid.json and Aspartic_Acid.json both claimed "D".
        let mut reg = Registry::new();
        add(&mut reg, "amino_acids", "AsparticAcid", json!({"symbol": "D", "n": 1}), true);
        add(&mut reg, "amino_acids", "Aspartic_Acid", json!({"symbol": "D", "n": 2}), true);
        add(&mut reg, "amino_acids", "Glycine", json!({"symbol": "G"}), true);

        let collisions = reg.collisions();
        assert_eq!(
            collisions,
            vec![(
                "amino_acids".to_string(),
                "D".to_string(),
                vec!["AsparticAcid".to_string(), "Aspartic_Acid".to_string()]
            )]
        );
        for scope in [None, Some(&["amino_acids"][..])] {
            match reg.lookup("D", scope) {
                Err(GetError::RegistryCollision { key, tier, files }) => {
                    assert_eq!(key, "D");
                    assert_eq!(tier, "amino_acids");
                    assert_eq!(files.len(), 2);
                }
                other => panic!("expected a collision, got {other:?}"),
            }
        }
        // The rest of the tier, and each file's own stem, still resolve.
        assert!(reg.lookup("G", None).unwrap().is_some());
        assert!(reg.lookup("Aspartic_Acid", None).unwrap().is_some());
    }

    #[test]
    fn the_same_key_in_two_tiers_is_a_priority_contest_not_a_collision() {
        let mut reg = Registry::new();
        add(&mut reg, "atoms", "P", json!({"Symbol": "P"}), false);
        add(&mut reg, "subatomic", "Proton", json!({"Aliases": ["P"]}), true);
        assert!(reg.collisions().is_empty());
        assert_eq!(reg.candidates("P").unwrap().len(), 2);
    }

    #[test]
    fn candidates_prefer_exact_matches_over_casefold() {
        let mut reg = Registry::new();
        add(&mut reg, "atoms", "U", json!({"Symbol": "U"}), false);
        add(&mut reg, "fundamentals", "UpQuark", json!({"Symbol": "u"}), true);
        let c = reg.candidates("u").unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].stem, "UpQuark");
        let c = reg.candidates("Ux").unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn unknown_and_empty_scopes_are_rejected() {
        let mut reg = Registry::new();
        reg.add_tier("atoms");
        assert!(reg.validate_scope(&["atoms"]).is_ok());
        assert!(matches!(
            reg.validate_scope(&["atom"]),
            Err(GetError::UnknownTier { .. })
        ));
        assert!(matches!(reg.validate_scope(&[]), Err(GetError::InvalidSpec(_))));
    }

    #[test]
    fn every_key_of_an_entry_shares_one_allocation() {
        let mut reg = Registry::new();
        let e = add(&mut reg, "atoms", "Fe", json!({"Symbol": "Fe", "Aliases": ["Iron"]}), false);
        let a = reg.lookup("Iron", None).unwrap().unwrap();
        let b = reg.lookup("Fe", None).unwrap().unwrap();
        assert!(Arc::ptr_eq(&a, &b) && Arc::ptr_eq(&a, &e));
    }

    #[test]
    fn placeholder_prefixes_are_recognised() {
        let prefixes = vec!["demo_".to_string(), "example_".to_string()];
        assert!(is_placeholder("demo_silicon_carbide", &prefixes));
        assert!(is_placeholder("Example_Thing", &prefixes));
        assert!(!is_placeholder("Steel-1018", &prefixes));
    }
}
