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
//! Name resolution, ported faithfully from `periodica/get.py`.
//!
//! ## Why this exists
//!
//! `get.rs::resolve_named` used to resolve names with a sequence of passes --
//! exact stem, casefold stem, `Symbol`, then `Aliases`. Python does something
//! structurally different: it builds a **priority-resolved index** at load
//! time, where every entry contributes several keys at different priorities,
//! and the highest-priority claimant of a key wins.
//!
//! The two orderings are close to opposite. Python ranks aliases highest and
//! derived stems lowest; the Rust passes tried stems first and aliases last.
//! They therefore disagreed on ambiguous short names -- `Get("E")` resolved to
//! glutamic acid in Rust (stem match in `amino_acids`) and to the electron in
//! Python (`Symbol` match). Any name that is a stem in one tier and a symbol
//! or alias in another was at risk.
//!
//! ## Python's priority table (`get.py`)
//!
//! | Key source | active tier | derived tier |
//! |---|---:|---:|
//! | `Aliases` / `aliases` | 5 | 4 |
//! | `Symbol` / `symbol` | 1 | **3** |
//! | file stem | 2 | 0 |
//!
//! Note the inversion: a *derived* symbol (3) outranks an *active* stem (2),
//! which outranks an *active* symbol (1). That is not an obvious ordering, and
//! it is exactly the sort of thing a re-implementation gets wrong -- hence a
//! direct port rather than a reconstruction from principles.
//!
//! Ties are resolved **first-writer-wins**: Python compares with `<`, not
//! `<=`, so an equal-priority later claimant does not displace an earlier one.
//! Insertion order is tier order, then the sorted file listing.

use std::collections::HashMap;

use serde_json::Value;

/// Key priorities, mirroring the `_PRIORITY_*` constants in `get.py`.
pub const PRIORITY_ALIAS_ACTIVE: i32 = 5;
pub const PRIORITY_ALIAS_DERIVED: i32 = 4;
pub const PRIORITY_SYMBOL_DERIVED: i32 = 3;
pub const PRIORITY_STEM_ACTIVE: i32 = 2;
pub const PRIORITY_SYMBOL_ACTIVE: i32 = 1;
pub const PRIORITY_STEM_DERIVED: i32 = 0;

/// Exact and casefold lookup tables for one tier. Mirrors `_TierIndex`.
#[derive(Debug, Default, Clone)]
pub struct TierIndex {
    exact: HashMap<String, Value>,
    casefold: HashMap<String, Value>,
    priority_exact: HashMap<String, i32>,
    priority_casefold: HashMap<String, i32>,
}

impl TierIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Claim `key` for `data` at priority `prio`.
    ///
    /// Uses `<` so the first writer wins a tie, matching Python.
    pub fn add(&mut self, key: &str, prio: i32, data: &Value) {
        if *self.priority_exact.get(key).unwrap_or(&-1) < prio {
            self.exact.insert(key.to_string(), data.clone());
            self.priority_exact.insert(key.to_string(), prio);
        }
        let cf = key.to_lowercase();
        if *self.priority_casefold.get(&cf).unwrap_or(&-1) < prio {
            self.casefold.insert(cf.clone(), data.clone());
            self.priority_casefold.insert(cf, prio);
        }
    }

    /// Exact match, then casefold. Mirrors `_Registry.lookup`.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.exact
            .get(name)
            .or_else(|| self.casefold.get(&name.to_lowercase()))
    }

    /// Every exactly-registered key, sorted. Used by diagnostics and tests.
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

/// Per-tier indices plus a merged cross-tier index. Mirrors `_Registry`.
#[derive(Debug, Default, Clone)]
pub struct Registry {
    pub by_tier: HashMap<String, TierIndex>,
    /// Used for unscoped lookups. Note this is a *priority* contest across the
    /// whole corpus, not a tier-order walk -- another place the previous Rust
    /// implementation diverged.
    pub merged: TierIndex,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one datasheet under `tier`.
    ///
    /// `stem` is the file stem, `is_active` marks tiers declared with
    /// `is_active: true` in `composition_rules.json`.
    pub fn add_entry(&mut self, tier: &str, stem: &str, data: &Value, is_active: bool) {
        let index = self.by_tier.entry(tier.to_string()).or_default();
        for (key, prio) in entry_keys(stem, data, is_active) {
            index.add(&key, prio, data);
            self.merged.add(&key, prio, data);
        }
    }

    /// Resolve a name, optionally restricted to specific tiers.
    pub fn lookup(&self, name: &str, from: Option<&[&str]>) -> Option<Value> {
        match from {
            None => self.merged.get(name).cloned(),
            Some(tiers) => {
                for tier in tiers {
                    if let Some(found) = self.by_tier.get(*tier).and_then(|i| i.get(name)) {
                        return Some(found.clone());
                    }
                }
                None
            }
        }
    }

    pub fn total_keys(&self) -> usize {
        self.merged.len()
    }
}

/// The `(key, priority)` pairs one entry contributes. Mirrors `_entry_keys`.
///
/// Both PascalCase and snake_case field names are accepted, because the amino
/// acid datasheets use lowercase `symbol` / `aliases` while everything else
/// uses `Symbol` / `Aliases`.
pub fn entry_keys(stem: &str, data: &Value, is_active: bool) -> Vec<(String, i32)> {
    let mut out = Vec::new();

    if let Some(sym) = data
        .get("Symbol")
        .or_else(|| data.get("symbol"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        out.push((
            sym.to_string(),
            if is_active {
                PRIORITY_SYMBOL_ACTIVE
            } else {
                PRIORITY_SYMBOL_DERIVED
            },
        ));
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
                out.push((s.to_string(), prio));
            }
        }
    }

    out.push((
        stem.to_string(),
        if is_active {
            PRIORITY_STEM_ACTIVE
        } else {
            PRIORITY_STEM_DERIVED
        },
    ));
    out
}

/// Whether a file stem is a placeholder that the registry skips.
///
/// `composition_rules.json` declares `placeholder_prefixes` (`demo_`,
/// `example_`); Python drops those files before indexing.
pub fn is_placeholder(stem: &str, prefixes: &[String]) -> bool {
    let lower = stem.to_lowercase();
    prefixes.iter().any(|p| lower.starts_with(p.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn priority_table_matches_python() {
        // Pinned against get.py's _PRIORITY_* constants. The inversion --
        // derived symbol (3) above active stem (2) above active symbol (1) --
        // is deliberate and easy to get wrong.
        assert_eq!(PRIORITY_ALIAS_ACTIVE, 5);
        assert_eq!(PRIORITY_ALIAS_DERIVED, 4);
        assert_eq!(PRIORITY_SYMBOL_DERIVED, 3);
        assert_eq!(PRIORITY_STEM_ACTIVE, 2);
        assert_eq!(PRIORITY_SYMBOL_ACTIVE, 1);
        assert_eq!(PRIORITY_STEM_DERIVED, 0);
        assert!(PRIORITY_SYMBOL_DERIVED > PRIORITY_STEM_ACTIVE);
        assert!(PRIORITY_STEM_ACTIVE > PRIORITY_SYMBOL_ACTIVE);
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
    fn snake_case_fields_are_accepted() {
        // Amino acid datasheets use lowercase `symbol` / `aliases`.
        let data = json!({"symbol": "E", "aliases": ["Glu"]});
        let keys = entry_keys("GlutamicAcid", &data, true);
        let map: HashMap<&str, i32> = keys.iter().map(|(k, p)| (k.as_str(), *p)).collect();
        assert_eq!(map["E"], PRIORITY_SYMBOL_ACTIVE);
        assert_eq!(map["Glu"], PRIORITY_ALIAS_ACTIVE);
    }

    #[test]
    fn higher_priority_claims_a_contested_key() {
        let mut idx = TierIndex::new();
        let stem_entry = json!({"which": "stem"});
        let alias_entry = json!({"which": "alias"});
        idx.add("X", PRIORITY_STEM_DERIVED, &stem_entry);
        idx.add("X", PRIORITY_ALIAS_ACTIVE, &alias_entry);
        assert_eq!(idx.get("X").unwrap()["which"], "alias");
    }

    #[test]
    fn lower_priority_cannot_displace() {
        let mut idx = TierIndex::new();
        let alias_entry = json!({"which": "alias"});
        let stem_entry = json!({"which": "stem"});
        idx.add("X", PRIORITY_ALIAS_ACTIVE, &alias_entry);
        idx.add("X", PRIORITY_STEM_DERIVED, &stem_entry);
        assert_eq!(idx.get("X").unwrap()["which"], "alias");
    }

    #[test]
    fn equal_priority_is_first_writer_wins() {
        // Python compares with `<`, not `<=`.
        let mut idx = TierIndex::new();
        let first = json!({"n": 1});
        let second = json!({"n": 2});
        idx.add("X", PRIORITY_STEM_ACTIVE, &first);
        idx.add("X", PRIORITY_STEM_ACTIVE, &second);
        assert_eq!(idx.get("X").unwrap()["n"], 1);
    }

    #[test]
    fn casefold_lookup_falls_back_after_exact() {
        let mut idx = TierIndex::new();
        let data = json!({"n": 1});
        idx.add("Fe", PRIORITY_STEM_ACTIVE, &data);
        assert!(idx.get("Fe").is_some());
        assert!(idx.get("fe").is_some());
        assert!(idx.get("FE").is_some());
        assert!(idx.get("Zz").is_none());
    }

    #[test]
    fn the_e_collision_resolves_the_way_python_does() {
        // The collision that motivated this module. Two real entries claim "E":
        //
        //   defaults/quarks/Electron.json        Symbol "E"  (fundamentals, active)
        //   active/amino_acids/GlutamicAcid.json symbol "E"  (amino_acids,  active)
        //
        // Neither claims it as a *stem* -- the amino acid's stem is
        // "GlutamicAcid" -- so both sit at PRIORITY_SYMBOL_ACTIVE. The winner
        // is decided purely by insertion order, and `fundamentals` is the
        // first entry in `composition_rules.json`'s tier list.
        //
        // Verified against CPython: `Get("E")` returns the Electron.
        let mut reg = Registry::new();
        let electron = json!({"Symbol": "E", "Name": "Electron", "Charge_e": -1});
        let glutamic = json!({"symbol": "E", "Name": "Glutamic acid"});
        reg.add_entry("fundamentals", "Electron", &electron, true);
        reg.add_entry("amino_acids", "GlutamicAcid", &glutamic, true);

        let got = reg.lookup("E", None).expect("E resolves");
        assert_eq!(
            got["Name"], "Electron",
            "unscoped `E` must resolve to the electron, as CPython does"
        );

        // Scoped lookup still reaches the amino acid.
        let scoped = reg.lookup("E", Some(&["amino_acids"])).expect("scoped E");
        assert_eq!(scoped["Name"], "Glutamic acid");

        // ...and the old pass-based resolver would have gone the other way: it
        // tried stems before symbols, so "GlutamicAcid" losing its stem claim
        // is precisely what makes the priority model necessary.
        assert!(reg.lookup("GlutamicAcid", None).is_some());
    }

    #[test]
    fn scoped_lookup_only_searches_named_tiers() {
        let mut reg = Registry::new();
        let fe = json!({"Symbol": "Fe", "Name": "Iron"});
        reg.add_entry("atoms", "Fe", &fe, false);
        assert!(reg.lookup("Fe", Some(&["atoms"])).is_some());
        assert!(reg.lookup("Fe", Some(&["proteins"])).is_none());
        assert!(reg.lookup("Fe", None).is_some());
    }

    #[test]
    fn placeholder_prefixes_are_recognised() {
        let prefixes = vec!["demo_".to_string(), "example_".to_string()];
        assert!(is_placeholder("demo_silicon_carbide", &prefixes));
        assert!(is_placeholder("Example_Thing", &prefixes));
        assert!(!is_placeholder("Steel-1018", &prefixes));
    }
}
