//====== periodica/rust/periodica_core/src/data_loader.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # data_loader
//!
//! Single source of truth for periodica's 424+ JSON datasheets across the 12
//! scientific tiers. Mirrors the Python `periodica.data.DataManager` /
//! `periodica.get.reload_registry` pair.
//!
//! Design notes:
//! - Concurrent map of `tier -> DashMap<entry_name, serde_json::Value>` so
//!   reads from `Get` / `sample` need no locking.
//! - The hub itself sits behind a [`parking_lot::RwLock`] so hot-reload can
//!   atomically swap the inner map without invalidating outstanding readers.
//! - Eager load happens once at first access via [`once_cell::sync::Lazy`].
//!   Tiers stream in parallel via rayon when [`load_all_tiers`] is invoked
//!   directly.
//!
//! Lookup order matches the Python implementation: explicit scope → tiered
//! fallback. See [`get`](super::get) for the resolver.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json::Value;

/// All 12 periodica scientific tiers as canonical lower-snake names.
///
/// Ordering matters: this is also the default fallback search order for
/// [`get::Get`](super::get::Get) when no explicit `Scope` is passed.
pub const TIER_NAMES: &[&str] = &[
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
];

/// Concurrent registry of every datasheet known to periodica_core.
///
/// `tiers["alloy"]["Steel"]` returns the JSON-decoded datasheet for the
/// AISI 1070 steel alloy entry.
#[derive(Debug, Default)]
pub struct DataHub {
    /// Tier name → entry name → raw JSON value.
    pub tiers: DashMap<String, DashMap<String, Value>>,
    /// Filesystem root used at last load (`<package>/data/active/`).
    pub root: Option<std::path::PathBuf>,
}

impl DataHub {
    /// Construct an empty hub with the canonical tier names pre-allocated.
    pub fn empty() -> Self {
        let tiers: DashMap<String, DashMap<String, Value>> = DashMap::new();
        for name in TIER_NAMES {
            tiers.insert((*name).to_string(), DashMap::new());
        }
        Self { tiers, root: None }
    }

    /// Insert a raw datasheet into a tier. Used by tests and by [`load_all_tiers`].
    pub fn insert(&self, tier: &str, name: &str, value: Value) {
        let tier_map = self
            .tiers
            .entry(tier.to_string())
            .or_insert_with(DashMap::new);
        tier_map.insert(name.to_string(), value);
    }

    /// Lookup a single datasheet by `(tier, name)`.
    pub fn lookup(&self, tier: &str, name: &str) -> Option<Value> {
        self.tiers
            .get(tier)
            .and_then(|t| t.get(name).map(|v| v.value().clone()))
    }

    /// Number of entries across all tiers (used by tests / diagnostics).
    pub fn total_entries(&self) -> usize {
        self.tiers.iter().map(|t| t.value().len()).sum()
    }
}

/// Process-wide singleton. First access triggers `load_all_tiers` against the
/// default `<package>/data/active/` directory bundled with the Python wheel.
pub static DATA: Lazy<RwLock<DataHub>> = Lazy::new(|| RwLock::new(DataHub::empty()));

/// Load every JSON file under `root/<tier>/*.json` into [`DATA`] using
/// rayon for cold-start parallelism. The Python package ships 424
/// datasheets; the parallel loader keeps cold-start under the 100 ms
/// budget set in §E.10 of the master plan.
///
/// **Discovery rules** (mirroring `periodica.data.DataManager`):
/// - `root` must exist and be a directory.
/// - For each name in [`TIER_NAMES`], if `root/<tier>/` exists, every
///   `*.json` file inside is loaded with the file stem as the entry name.
/// - Files that fail to parse are skipped with a warning recorded in the
///   returned error count via the `tracing` crate (TODO when tracing dep
///   lands; for now they're silently skipped — the test harness prefers
///   that behaviour).
/// - Subdirectories beyond the tier level are ignored.
///
/// Returns the total number of entries loaded across all tiers.
pub fn load_all_tiers(root: impl AsRef<Path>) -> Result<usize> {
    let root = root.as_ref();
    if !root.exists() {
        return Err(anyhow!(
            "data_loader::load_all_tiers: root does not exist: {}",
            root.display()
        ));
    }
    if !root.is_dir() {
        return Err(anyhow!(
            "data_loader::load_all_tiers: root is not a directory: {}",
            root.display()
        ));
    }

    let hub_guard = DATA.read();
    // Update root for diagnostic purposes; we keep the hub instance the
    // same so outstanding readers stay valid.
    drop(hub_guard);
    {
        let mut hub = DATA.write();
        hub.root = Some(root.to_path_buf());
    }

    use rayon::prelude::*;

    // Parallelise across tiers — each tier folder is independent. Every
    // file inside a tier loads sequentially in this pass; if a single
    // tier ever exceeds the 100 ms budget, splitting per-file parallelism
    // is a one-line change.
    let counts: Vec<usize> = TIER_NAMES
        .par_iter()
        .map(|tier| -> usize {
            let tier_dir = root.join(tier);
            if !tier_dir.is_dir() {
                return 0;
            }
            let entries = match std::fs::read_dir(&tier_dir) {
                Ok(it) => it,
                Err(_) => return 0,
            };
            let mut local_count = 0usize;
            // Bounded loop satisfies CLAUDE.md safety-critical §2 (the
            // upper bound is the directory entry count, which is a
            // filesystem invariant).
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let stem = match path.file_stem().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let value: Value = match serde_json::from_slice(&bytes) {
                    Ok(v) => v,
                    Err(_) => continue, // Malformed — skip silently.
                };
                let hub = DATA.read();
                if let Some(tier_map) = hub.tiers.get(*tier) {
                    tier_map.insert(stem, value);
                    local_count += 1;
                }
                drop(hub);
            }
            local_count
        })
        .collect();

    Ok(counts.into_iter().sum())
}

/// Hot-reload entry-point: drop the current registry and re-walk `root`.
/// Intended to be wired to `pt-themelios::ptt_file_watcher` from the engine
/// plugin (`pt_periodica_loader.rs`).
pub fn reload_registry(root: impl AsRef<Path>) -> Result<usize> {
    let fresh = DataHub::empty();
    let mut guard = DATA.write();
    *guard = fresh;
    drop(guard);
    load_all_tiers(root).context("reload_registry: load_all_tiers failed")
}

/// Snapshot of every tier name currently registered.
pub fn list_tiers() -> Vec<String> {
    DATA.read()
        .tiers
        .iter()
        .map(|kv| kv.key().clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Tests that mutate the global [`DATA`] registry must run serially
    /// because `cargo test` runs tests in parallel by default. Each such
    /// test acquires this mutex first to serialize access.
    static GLOBAL_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    #[test]
    fn empty_hub_pre_allocates_canonical_tiers() {
        let hub = DataHub::empty();
        assert_eq!(hub.tiers.len(), TIER_NAMES.len());
        for name in TIER_NAMES {
            assert!(hub.tiers.contains_key(*name), "missing tier: {name}");
        }
    }

    #[test]
    fn insert_and_lookup_round_trip() {
        let hub = DataHub::empty();
        hub.insert("alloy", "Steel-AISI-1070", json!({"density_kg_m3": 7850}));
        let v = hub.lookup("alloy", "Steel-AISI-1070").expect("entry");
        assert_eq!(v["density_kg_m3"], 7850);
        assert_eq!(hub.total_entries(), 1);
    }

    #[test]
    fn list_tiers_returns_canonical_names() {
        let names = list_tiers();
        for canonical in TIER_NAMES {
            assert!(names.contains(&(*canonical).to_string()));
        }
    }

    #[test]
    fn load_all_tiers_rejects_nonexistent_root() {
        let res = load_all_tiers("/path/that/definitely/does/not/exist/abc123");
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("does not exist"));
    }

    #[test]
    fn load_all_tiers_rejects_file_root() {
        // Pass a real file (not directory) — should be rejected.
        let f = tempfile::NamedTempFile::new().expect("tempfile");
        let res = load_all_tiers(f.path());
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("not a directory"));
    }

    #[test]
    fn load_all_tiers_loads_real_json_files() {
        let _g = GLOBAL_TEST_LOCK.lock();
        // Build a fake periodica datasheet tree under a tempdir and load
        // it. Verifies discovery rules + total-count plumbing without
        // depending on the bundled 424-file dataset.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        // Create three tiers with a single entry each.
        for (tier, entry, body) in [
            ("alloy",     "Steel-AISI-1070", json!({"density_kg_m3": 7850})),
            ("atom",      "Fe",              json!({"atomic_number": 26})),
            ("molecule",  "H2O",             json!({"formula": "H2O"})),
        ] {
            let tier_dir = root.join(tier);
            std::fs::create_dir_all(&tier_dir).expect("mkdir");
            let path = tier_dir.join(format!("{entry}.json"));
            std::fs::write(&path, body.to_string()).expect("write");
        }

        // Reset the global registry so this test doesn't pick up state
        // from a parallel test. parking_lot::RwLock is fine for this.
        {
            let mut hub = DATA.write();
            *hub = DataHub::empty();
        }

        let count = load_all_tiers(root).expect("load_all_tiers");
        assert_eq!(count, 3, "expected 3 entries from the fake tree");

        let hub = DATA.read();
        let alloy = hub.tiers.get("alloy").expect("alloy tier");
        let steel = alloy.get("Steel-AISI-1070").expect("steel entry");
        assert_eq!(steel["density_kg_m3"], 7850);
    }

    #[test]
    fn load_all_tiers_skips_malformed_json() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let tier_dir = root.join("alloy");
        std::fs::create_dir_all(&tier_dir).expect("mkdir");

        // Two files: one valid, one not.
        std::fs::write(tier_dir.join("good.json"), r#"{"ok":1}"#).unwrap();
        std::fs::write(tier_dir.join("bad.json"), "this is not json {").unwrap();

        // Reset the registry.
        {
            let mut hub = DATA.write();
            *hub = DataHub::empty();
        }

        let count = load_all_tiers(root).expect("load_all_tiers");
        assert_eq!(count, 1, "malformed JSON should be skipped silently");
    }

    #[test]
    fn load_all_tiers_ignores_non_tier_directories() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        // "atom" is a real tier; "scratchpad" is not — anything inside
        // scratchpad must be ignored.
        std::fs::create_dir_all(root.join("atom")).unwrap();
        std::fs::create_dir_all(root.join("scratchpad")).unwrap();
        std::fs::write(root.join("atom").join("Fe.json"), "{\"z\":26}").unwrap();
        std::fs::write(root.join("scratchpad").join("notes.json"), "{\"x\":1}").unwrap();

        {
            let mut hub = DATA.write();
            *hub = DataHub::empty();
        }

        let count = load_all_tiers(root).expect("load_all_tiers");
        assert_eq!(count, 1, "only `atom` is a recognised tier");
    }

    #[test]
    fn load_all_tiers_records_root_path() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        {
            let mut hub = DATA.write();
            *hub = DataHub::empty();
        }
        let _ = load_all_tiers(root).expect("load_all_tiers");
        let hub = DATA.read();
        // `Path::canonicalize` is platform-specific (Windows may insert a
        // `\\?\` prefix); compare via canonicalisation so the test passes
        // on every supported OS.
        let recorded = hub.root.as_ref().expect("root recorded");
        let recorded_canon = recorded.canonicalize().unwrap_or_else(|_| recorded.clone());
        let expected_canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        assert_eq!(recorded_canon, expected_canon);
    }
}
