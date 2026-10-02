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
//! Registry of periodica's bundled JSON datasheets, mirroring the Python
//! `periodica.get` registry so that both backends resolve the same names.
//!
//! ## What was wrong before
//!
//! This module previously hardcoded twelve **singular** tier names
//! (`"atom"`, `"alloy"`, `"protein"`, ...) and looked for them as direct
//! children of `data/active/`. Three things were wrong with that:
//!
//! 1. The on-disk folders are plural and differently named (`elements`,
//!    `alloys`, `proteins`, `materials`). Only `subatomic/` matched, so the
//!    loader found 24 files out of 949.
//! 2. Most tiers do not live under `active/` at all -- `atoms`, `molecules`,
//!    `ceramics`, `composites`, `polymers`, `ions`, `isotopes` and
//!    `hadrons_gen` are all under `derived/`, which was never scanned.
//! 3. 36 datasheets are JSONC, not JSON. `serde_json::from_slice` rejected
//!    every one of them (see [`periodica_mat::jsonc`]).
//!
//! ## What it does now
//!
//! The tier list is read from `data/config/composition_rules.json` -- the same
//! file `periodica.get._tier_sources` reads -- rather than being hardcoded a
//! second time. Tiers are the four `tier_definitions` entries plus every
//! subdirectory of `derived_root`, which reproduces Python's `list_tiers()`
//! exactly.
//!
//! ## The materials catalogue
//!
//! `active/materials/` holds the 45 richest datasheets in the corpus -- the
//! ones carrying `FractureMechanics`, `Microstructure` and `FailureCriteria`.
//! They are **not** part of the `Get()` registry in Python either
//! (`Get("Aluminum_6061_T6")` raises `UnknownName`), so adding them to
//! [`DataHub::tiers`] would break name-resolution parity.
//!
//! They are instead loaded into a separate [`DataHub::materials`] catalogue
//! that the runtime and the designer app consume. Registry behaviour is
//! unchanged; the rich data stops being unreachable.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use periodica_mat::jsonc;
use serde_json::Value;

/// Data roots scanned for the materials catalogue, in precedence order.
/// Earlier roots win when the same stem appears twice.
pub const MATERIAL_ROOTS: &[&str] = &["active", "derived", "reference", "defaults"];

/// Subdirectories of a root that carry material-like datasheets.
pub const MATERIAL_DIRS: &[&str] = &[
    "materials",
    "alloys",
    "ceramics",
    "composites",
    "polymers",
    "elements",
    "biological_materials",
    "tissues",
];

/// Concurrent registry of every datasheet known to periodica_core.
#[derive(Debug, Default)]
pub struct DataHub {
    /// Tier name -> entry name -> raw JSON. Mirrors the Python `Get()`
    /// registry, so tier names here are the plural on-disk names
    /// (`"alloys"`, `"atoms"`, `"proteins"`, ...).
    pub tiers: DashMap<String, DashMap<String, Value>>,
    /// Material-bearing datasheets from every root, keyed by file stem.
    /// Deliberately outside [`Self::tiers`] -- see the module docs.
    pub materials: DashMap<String, Value>,
    /// Tier names in *load* order, not alphabetical order.
    ///
    /// Name resolution walks this so the most fundamental tier wins a
    /// collision (quarks before atoms). That order comes from the config's
    /// `tier_definitions` followed by the `derived/` subdirectories, matching
    /// Python's `_tier_sources()`.
    pub order: Vec<String>,
    /// Priority-resolved name index, mirroring Python's `_Registry`.
    /// This is what `Get()` resolves through -- see [`crate::registry`].
    pub registry: crate::registry::Registry,
    /// The `data/` directory used at last load.
    pub root: Option<PathBuf>,
}

impl DataHub {
    /// Construct an empty hub.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Insert a raw datasheet into a tier.
    pub fn insert(&self, tier: &str, name: &str, value: Value) {
        self.tiers
            .entry(tier.to_string())
            .or_default()
            .insert(name.to_string(), value);
    }

    /// Lookup a single datasheet by `(tier, name)`.
    pub fn lookup(&self, tier: &str, name: &str) -> Option<Value> {
        self.tiers
            .get(tier)
            .and_then(|t| t.get(name).map(|v| v.value().clone()))
    }

    /// Find an entry by name across every tier, then the materials catalogue.
    ///
    /// Walks [`Self::order`] rather than iterating `tiers` directly: `DashMap`
    /// iteration order is not deterministic, so a name present in two tiers
    /// would otherwise resolve differently between runs. Prefer
    /// [`crate::get::Get`] where full name resolution (Symbol, Aliases,
    /// casefold) is wanted -- this is the fast exact-stem path only.
    pub fn find(&self, name: &str) -> Option<Value> {
        for tier in &self.order {
            if let Some(map) = self.tiers.get(tier) {
                if let Some(v) = map.get(name) {
                    return Some(v.value().clone());
                }
            }
        }
        // Tests may populate `tiers` directly without going through
        // `load_all_tiers`, leaving `order` empty; fall back to a scan.
        if self.order.is_empty() {
            for kv in self.tiers.iter() {
                if let Some(v) = kv.value().get(name) {
                    return Some(v.value().clone());
                }
            }
        }
        self.materials.get(name).map(|v| v.value().clone())
    }

    /// Number of entries across all tiers (excluding the materials catalogue).
    pub fn total_entries(&self) -> usize {
        self.tiers.iter().map(|t| t.value().len()).sum()
    }
}

/// Process-wide singleton.
pub static DATA: Lazy<RwLock<DataHub>> = Lazy::new(|| RwLock::new(DataHub::empty()));

/// Serialises tests that load into [`DATA`].
///
/// `cargo test` runs tests in parallel within a binary, and [`DATA`] is
/// process-wide, so a test that swaps in a fixture registry will corrupt any
/// concurrently-running test that expects the real corpus. This lock must be
/// shared by **every** module whose tests touch [`DATA`] -- a per-module mutex
/// provides no mutual exclusion between modules.
#[cfg(test)]
pub(crate) static DATA_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// One tier as declared in `composition_rules.json`.
#[derive(Debug, Clone)]
pub struct TierSource {
    pub name: String,
    /// Path relative to the `data/` directory, e.g. `active/quarks`.
    pub source: String,
    /// `is_active` from `composition_rules.json`. Feeds the name-resolution
    /// priority table (see [`crate::registry`]) -- an active tier's stem
    /// outranks a derived tier's stem, and so on.
    pub is_active: bool,
}

/// Read the tier list from `data/config/composition_rules.json`.
///
/// This is the same file the Python registry reads. Keeping one source of
/// truth is what guarantees `list_tiers()` agrees across the two backends.
pub fn tier_sources(data_root: &Path) -> Result<Vec<TierSource>> {
    let cfg_path = data_root.join("config").join("composition_rules.json");
    let cfg = jsonc::from_path(&cfg_path)
        .with_context(|| format!("data_loader: reading {}", cfg_path.display()))?;

    let mut out = Vec::new();
    if let Some(defs) = cfg.get("tier_definitions").and_then(Value::as_array) {
        for e in defs {
            let (Some(name), Some(source)) = (
                e.get("name").and_then(Value::as_str),
                e.get("source").and_then(Value::as_str),
            ) else {
                continue;
            };
            out.push(TierSource {
                name: name.to_string(),
                source: source.to_string(),
                is_active: e.get("is_active").and_then(Value::as_bool).unwrap_or(false),
            });
        }
    }

    // Every subdirectory of `derived_root` is a tier, discovered rather than
    // hardcoded so regenerated data cannot silently fall out of the registry.
    let derived = cfg
        .get("derived_root")
        .and_then(Value::as_str)
        .unwrap_or("derived");
    let derived_dir = data_root.join(derived);
    if let Ok(entries) = std::fs::read_dir(&derived_dir) {
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        names.sort();
        for n in names {
            out.push(TierSource {
                source: format!("{derived}/{n}"),
                name: n,
                // Everything discovered under `derived_root` is, by
                // definition, not an active tier.
                is_active: false,
            });
        }
    }

    if out.is_empty() {
        return Err(anyhow!(
            "data_loader: no tiers discovered under {}",
            data_root.display()
        ));
    }
    Ok(out)
}

/// `placeholder_prefixes` from `composition_rules.json` (`demo_`, `example_`).
/// Files whose stem starts with one of these are excluded from the registry,
/// matching `get.py::_index_files`.
pub fn placeholder_prefixes(data_root: &Path) -> Vec<String> {
    let cfg_path = data_root.join("config").join("composition_rules.json");
    jsonc::from_path(&cfg_path)
        .ok()
        .and_then(|cfg| {
            cfg.get("placeholder_prefixes")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                        .collect()
                })
        })
        .unwrap_or_default()
}

/// Load every datasheet under `data_root` into [`DATA`].
///
/// `data_root` is the package's `data/` directory (the one containing
/// `config/`, `active/` and `derived/`). For backwards compatibility a path
/// pointing directly at `active/` is accepted and silently corrected, since
/// that is what earlier builds of `pyfacade` passed.
///
/// Returns the number of tier entries loaded (the materials catalogue is
/// counted separately by [`DataHub::materials`]).
pub fn load_all_tiers(data_root: impl AsRef<Path>) -> Result<usize> {
    let given = data_root.as_ref();
    if !given.exists() {
        return Err(anyhow!(
            "data_loader::load_all_tiers: root does not exist: {}",
            given.display()
        ));
    }
    if !given.is_dir() {
        return Err(anyhow!(
            "data_loader::load_all_tiers: root is not a directory: {}",
            given.display()
        ));
    }

    // Accept either `data/` or `data/active/`.
    let root: PathBuf = if given.join("config").is_dir() || given.join("derived").is_dir() {
        given.to_path_buf()
    } else if given
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| MATERIAL_ROOTS.contains(&s))
    {
        given
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| given.to_path_buf())
    } else {
        given.to_path_buf()
    };

    let sources = tier_sources(&root)?;

    use rayon::prelude::*;
    // Tiers are independent, so load them in parallel and merge afterwards.
    // Building per-tier maps first (rather than inserting into the shared hub
    // inside the loop) keeps the write lock uncontended.
    let loaded: Vec<(String, Vec<(String, Value)>)> = sources
        .par_iter()
        .map(|ts| {
            let dir = root.join(&ts.source);
            (ts.name.clone(), read_dir_entries(&dir))
        })
        .collect();

    // Borrow rather than move: `root` is still needed after the parallel scan.
    let root_ref = &root;
    let materials: Vec<(String, Value)> = MATERIAL_ROOTS
        .par_iter()
        .flat_map(|r| {
            MATERIAL_DIRS
                .par_iter()
                .map(move |d| read_dir_entries(&root_ref.join(r).join(d)))
        })
        .flatten()
        .collect();

    let mut count = 0usize;
    {
        let mut hub = DATA.write();
        hub.tiers.clear();
        hub.materials.clear();
        for (tier, entries) in loaded {
            let map = hub.tiers.entry(tier).or_default();
            for (name, value) in entries {
                map.insert(name, value);
                count += 1;
            }
        }
        // MATERIAL_ROOTS is in precedence order and `flat_map` preserves it,
        // so the first occurrence of a stem must win.
        for (name, value) in materials {
            hub.materials.entry(name).or_insert(value);
        }
        hub.order = sources.iter().map(|t| t.name.clone()).collect();

        // Build the name-resolution index in tier order, then sorted file
        // order. Both matter: ties are first-writer-wins, so insertion order
        // decides collisions like "E" (electron vs glutamic acid).
        let prefixes = placeholder_prefixes(&root);
        let mut registry = crate::registry::Registry::new();
        for ts in &sources {
            if let Some(map) = hub.tiers.get(&ts.name) {
                let mut stems: Vec<String> = map.iter().map(|kv| kv.key().clone()).collect();
                stems.sort();
                for stem in stems {
                    if crate::registry::is_placeholder(&stem, &prefixes) {
                        continue;
                    }
                    if let Some(v) = map.get(&stem) {
                        registry.add_entry(&ts.name, &stem, v.value(), ts.is_active);
                    }
                }
            }
        }
        hub.registry = registry;
        hub.root = Some(root.clone());
    }
    Ok(count)
}

/// Tier names in resolution order (most fundamental first).
///
/// Falls back to sorted names if the registry was populated directly by tests
/// rather than by [`load_all_tiers`].
pub fn tier_order() -> Vec<String> {
    let hub = DATA.read();
    if !hub.order.is_empty() {
        return hub.order.clone();
    }
    let mut names: Vec<String> = hub.tiers.iter().map(|kv| kv.key().clone()).collect();
    names.sort();
    names
}

/// Read every `*.json` in `dir` as JSONC, keyed by file stem.
///
/// Unreadable or malformed files are skipped rather than failing the whole
/// load: one bad datasheet must not take down the registry.
fn read_dir_entries(dir: &Path) -> Vec<(String, Value)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // JSONC, not JSON: 36 datasheets carry `//` comments.
        if let Ok(value) = jsonc::from_path(&path) {
            out.push((stem.to_string(), value));
        }
    }
    out
}

/// Environment variable naming the `data/` directory [`ensure_loaded`] loads
/// when nothing has been loaded yet.
pub const DATA_DIR_ENV: &str = "PERIODICA_DATA_DIR";

/// The `data/` directory [`ensure_loaded`] uses.
///
/// `$PERIODICA_DATA_DIR` when set and non-empty, otherwise the corpus bundled
/// with this crate's source tree, `CARGO_MANIFEST_DIR/../../src/periodica/data`.
/// That second path is fixed at compile time, which is what makes a vendored
/// copy (`<repo>/rust/periodica_core`) find its own `<repo>/src/periodica/data`
/// with no configuration. A binary shipped to a machine without that tree must
/// set the variable or call [`reload_registry`] itself.
pub fn default_data_root() -> PathBuf {
    match std::env::var_os(DATA_DIR_ENV) {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data"),
    }
}

/// Serialises lazy loads so concurrent first callers walk the corpus once.
static LAZY_LOAD: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Load [`default_data_root`] into [`DATA`] unless a root is already loaded.
///
/// "Loaded" means [`load_all_tiers`] or [`reload_registry`] has succeeded at
/// least once; an explicitly loaded root is never replaced. The Python
/// extension loads the installed package's data at import, so this is a no-op
/// there. Pure-Rust callers get the bundled corpus on first use.
///
/// A failed load is not cached: the next call tries again, so setting
/// [`DATA_DIR_ENV`] after a failure takes effect.
pub fn ensure_loaded() -> Result<()> {
    let _serial = LAZY_LOAD.lock();
    let loaded = DATA.read().root.is_some();
    if loaded {
        return Ok(());
    }
    let root = default_data_root();
    load_all_tiers(&root).with_context(|| {
        format!(
            "data_loader::ensure_loaded: cannot load {} (set {DATA_DIR_ENV} to \
             the periodica data/ directory, or call reload_registry(path))",
            root.display()
        )
    })?;
    Ok(())
}

/// Drop the current registry and re-walk `data_root`.
pub fn reload_registry(data_root: impl AsRef<Path>) -> Result<usize> {
    {
        let mut guard = DATA.write();
        *guard = DataHub::empty();
    }
    load_all_tiers(data_root).context("reload_registry: load_all_tiers failed")
}

/// Snapshot of every tier name currently registered, sorted.
///
/// Matches Python's `periodica.list_tiers()`.
pub fn list_tiers() -> Vec<String> {
    let mut names: Vec<String> = DATA
        .read()
        .tiers
        .iter()
        .map(|kv| kv.key().clone())
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use super::DATA_TEST_LOCK as GLOBAL_TEST_LOCK;

    fn bundled_data_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
    }

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    /// A minimal data tree with the same shape as the real one.
    fn fake_tree(root: &Path) {
        write(
            &root.join("config/composition_rules.json"),
            r#"{
                // tiers, as the Python registry declares them
                "tier_definitions": [
                    {"name": "fundamentals", "source": "active/quarks", "is_active": true},
                    {"name": "subatomic", "source": "active/subatomic", "is_active": true}
                ],
                "derived_root": "derived"
            }"#,
        );
        write(
            &root.join("active/quarks/UpQuark.json"),
            r#"{"Symbol":"u"}"#,
        );
        write(
            &root.join("active/subatomic/Proton.json"),
            r#"{"Charge_e":1}"#,
        );
        write(
            &root.join("derived/alloys/Steel-1018.json"),
            r#"{"Properties":{"Density_kgm3":7870}}"#,
        );
        write(
            &root.join("derived/atoms/Fe.json"),
            r#"{"atomic_number":26}"#,
        );
        write(
            &root.join("active/materials/Al6061.json"),
            r#"{"Name":"Al6061"}"#,
        );
    }

    #[test]
    fn empty_hub_starts_empty() {
        let hub = DataHub::empty();
        assert_eq!(hub.total_entries(), 0);
        assert!(hub.tiers.is_empty());
        assert!(hub.materials.is_empty());
    }

    #[test]
    fn insert_and_lookup_round_trip() {
        let hub = DataHub::empty();
        hub.insert("alloys", "Steel-1018", json!({"density_kg_m3": 7850}));
        assert_eq!(
            hub.lookup("alloys", "Steel-1018").unwrap()["density_kg_m3"],
            7850
        );
        assert_eq!(hub.total_entries(), 1);
        assert!(hub.find("Steel-1018").is_some());
        assert!(hub.find("Nonexistent").is_none());
    }

    #[test]
    fn load_all_tiers_rejects_bad_roots() {
        let e = load_all_tiers("/path/that/does/not/exist/abc123").unwrap_err();
        assert!(format!("{e}").contains("does not exist"));

        let f = tempfile::NamedTempFile::new().unwrap();
        let e = load_all_tiers(f.path()).unwrap_err();
        assert!(format!("{e}").contains("not a directory"));
    }

    #[test]
    fn tier_sources_come_from_the_config_not_a_hardcoded_list() {
        let dir = tempfile::tempdir().unwrap();
        fake_tree(dir.path());
        let sources = tier_sources(dir.path()).unwrap();
        let names: Vec<&str> = sources.iter().map(|t| t.name.as_str()).collect();
        // Two declared tiers plus both discovered `derived/` subdirectories.
        assert!(names.contains(&"fundamentals"), "{names:?}");
        assert!(names.contains(&"subatomic"), "{names:?}");
        assert!(names.contains(&"alloys"), "{names:?}");
        assert!(names.contains(&"atoms"), "{names:?}");
    }

    #[test]
    fn loads_a_fake_tree_across_all_roots() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().unwrap();
        fake_tree(dir.path());

        let n = load_all_tiers(dir.path()).unwrap();
        assert_eq!(n, 4, "two active + two derived entries");

        let hub = DATA.read();
        assert!(
            hub.lookup("alloys", "Steel-1018").is_some(),
            "derived/ must be scanned"
        );
        assert!(hub.lookup("atoms", "Fe").is_some());
        assert!(hub.lookup("fundamentals", "UpQuark").is_some());

        // active/materials is catalogued but deliberately out of the registry,
        // matching Python, where Get("Aluminum_6061_T6") raises UnknownName.
        assert!(hub.materials.contains_key("Al6061"));
        assert!(hub.lookup("materials", "Al6061").is_none());
        assert!(hub.find("Al6061").is_some(), "still reachable via find()");
    }

    #[test]
    fn accepts_a_path_pointing_at_active_for_backwards_compatibility() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().unwrap();
        fake_tree(dir.path());
        // Earlier builds of pyfacade passed `data/active`.
        let n = load_all_tiers(dir.path().join("active")).unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn jsonc_datasheets_load() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().unwrap();
        fake_tree(dir.path());
        write(
            &dir.path().join("derived/atoms/Cu.json"),
            "{\n \"atomic_number\": 29, // copper\n \"mass\": 63.546\n}",
        );
        load_all_tiers(dir.path()).unwrap();
        let hub = DATA.read();
        let cu = hub.lookup("atoms", "Cu").expect("JSONC entry must load");
        assert_eq!(cu["atomic_number"], 29);
    }

    #[test]
    fn malformed_json_is_skipped_without_failing_the_load() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().unwrap();
        fake_tree(dir.path());
        write(
            &dir.path().join("derived/atoms/bad.json"),
            "this is not json {",
        );
        // The good entries still load.
        let n = load_all_tiers(dir.path()).unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn reload_replaces_rather_than_accumulates() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let dir = tempfile::tempdir().unwrap();
        fake_tree(dir.path());
        let first = load_all_tiers(dir.path()).unwrap();
        let second = reload_registry(dir.path()).unwrap();
        assert_eq!(first, second);
        assert_eq!(DATA.read().total_entries(), first);
    }

    // ── Against the real bundled corpus ──────────────────────────────────

    #[test]
    fn loads_the_real_corpus() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let data = bundled_data_dir();
        if !data.is_dir() {
            eprintln!("skipping: {} not present", data.display());
            return;
        }
        let n = load_all_tiers(&data).unwrap();

        // Python's registry scope is 392 files: the four `tier_definitions`
        // sources (19+24+24+20) plus the nine `derived/` subdirectories
        // (118+73+40+30+15+10+9+6+4). The Rust loader must cover the same set.
        //
        // The old loader found 24 -- only `subatomic/` matched its hardcoded
        // singular tier names. A count anywhere near that means tier discovery
        // has broken again.
        assert!(
            n > 300,
            "only {n} entries loaded -- tier discovery is broken"
        );

        let tiers = list_tiers();
        // Exactly the 13 tiers Python's list_tiers() reports.
        let expected = [
            "alloys",
            "amino_acids",
            "atoms",
            "ceramics",
            "composites",
            "fundamentals",
            "hadrons_gen",
            "ions",
            "isotopes",
            "molecules",
            "polymers",
            "proteins",
            "subatomic",
        ];
        assert_eq!(tiers, expected, "tier set must match Python's list_tiers()");

        let hub_tiers = DATA.read();
        for t in expected {
            let count = hub_tiers.tiers.get(t).map(|m| m.len()).unwrap_or(0);
            assert!(count > 0, "tier {t} loaded no entries");
        }
        drop(hub_tiers);

        let hub = DATA.read();
        // Names that Python's Get() resolves must resolve here too.
        assert!(hub.find("Steel-1018").is_some());
        assert!(hub.find("Fe").is_some());
        // And the rich materials are now reachable, where before they were not.
        assert!(
            hub.materials.len() > 100,
            "materials catalogue has only {} entries",
            hub.materials.len()
        );
        assert!(hub.materials.contains_key("Aluminum_6061_T6"));
    }

    #[test]
    fn jsonc_quark_datasheets_load_from_the_real_corpus() {
        let _g = GLOBAL_TEST_LOCK.lock();
        let data = bundled_data_dir();
        if !data.is_dir() {
            return;
        }
        load_all_tiers(&data).unwrap();
        let hub = DATA.read();
        // active/quarks/*.json all carry `//` comments; serde_json alone
        // rejects every one of them.
        let up = hub
            .lookup("fundamentals", "UpQuark")
            .expect("UpQuark is JSONC and must still load");
        assert_eq!(up["Symbol"], "u");
    }
}
