//====== periodica/rust/periodica_core/src/record.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # record
//!
//! One call from a periodica spec string to a typed, SI-normalised
//! [`MaterialRecord`], for pure-Rust consumers (game engines) that have no
//! Python runtime and should not have to know how the registry, the data root
//! or the datasheet schemas work.
//!
//! This module adds no name resolution of its own: names go through
//! [`crate::get::Get`], datasheets go through [`periodica_mat::ingest()`]. The
//! only extra step is the element join described on [`material_record`].

use anyhow::{anyhow, bail, Context, Result};
use periodica_mat::{ingest, MaterialRecord, PropertyId, Tier};
use serde_json::Value;

use crate::data_loader::{self, DATA};
use crate::get::{parse_formula, Get, Scope};

/// Resolve `spec` to a [`MaterialRecord`].
///
/// # Resolution
///
/// 1. `spec` is resolved with [`Get`]`(spec, scope)` -- the same priority
///    registry the Python `periodica.Get` uses, plus the unscoped fallback to
///    the materials catalogue (`active/materials`, ...), so `"Steel-1018"`,
///    `"Aluminum_6061_T6"` and `"Granite_Westerly"` all work.
/// 2. The datasheet is ingested by [`periodica_mat::ingest()`] as
///    `ingest(&value, spec)`.
/// 3. **Element join.** A bare element symbol such as `"Fe"` resolves to the
///    `atoms` tier, which carries particle-physics data (mass, charge,
///    constituents) and no bulk properties. When the resolved datasheet has a
///    root `Symbol` but no density, the element datasheet with that exact
///    symbol (`active/elements/026_Fe.json`) is ingested instead, giving the
///    bulk solid: density, melting point and, for elements solid at STP, the
///    crystal structure. That record has [`Tier::Element`].
///
/// # Errors
///
/// - the name is unknown (the [`Get`] error is preserved as the cause);
/// - the data root cannot be loaded (see below);
/// - **the resolved datasheet carries no density.** A record without one is
///   not usable as a material, and returning it would let a caller silently
///   fall back to defaults. This is the case for particles, amino acids,
///   proteins and the generated `molecules` tier. In particular:
///   - `"H2O"` resolves to `derived/molecules/H2O.json`, a constituent
///     composition with no bulk properties, and is an **error**;
///   - formula specs (`"{H=2,O=1}"`) compose additive particle quantities
///     (mass, charge, baryon number), never bulk properties, and are always an
///     **error**. The element join is deliberately not applied to them: the
///     merged value keeps the first constituent's `Symbol`, so joining would
///     turn `"{Fe=1,C=1}"` into pure iron.
///
/// # Data root
///
/// `Get` does not load anything by itself; the registry starts empty. This
/// function calls [`data_loader::ensure_loaded`], so no prior load call is
/// needed. When nothing has been loaded yet, the root is:
///
/// 1. `$PERIODICA_DATA_DIR` ([`data_loader::DATA_DIR_ENV`]) if set; otherwise
/// 2. `CARGO_MANIFEST_DIR/../../src/periodica/data`, fixed at compile time --
///    for a crate vendored at `<repo>/rust/periodica_core` that is the
///    vendored `<repo>/src/periodica/data`.
///
/// To use a different corpus, set the environment variable before the first
/// call, or call [`data_loader::reload_registry`]`(path)` (or
/// [`data_loader::load_all_tiers`]) explicitly: once a root is loaded it is
/// used as-is and never replaced behind the caller's back.
///
/// # Example
///
/// ```no_run
/// use periodica_core::{material_record, PropertyId};
///
/// let fe = material_record("Fe", None)?;
/// let density = fe.bulk.get(PropertyId::Density); // Some(7874.0), kg/m^3
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn material_record(spec: &str, scope: Option<Scope>) -> Result<MaterialRecord> {
    if spec.trim().is_empty() {
        bail!("material_record: spec must be non-empty");
    }
    data_loader::ensure_loaded().context("material_record: loading the periodica data")?;

    let resolved =
        Get(spec, scope).with_context(|| format!("material_record: cannot resolve {spec:?}"))?;
    let record = ingest(&resolved, spec).context("material_record: ingest")?;
    if has_density(&record) {
        return Ok(record);
    }

    if parse_formula(spec)?.is_some() {
        bail!(
            "material_record: formula spec {spec:?} composes particle quantities \
             (mass, charge), not bulk material properties; name a material instead"
        );
    }

    if let Some(symbol) = root_symbol(&resolved) {
        if let Some((stem, element)) = element_datasheet(symbol) {
            let mut joined = ingest(&element, &stem).context("material_record: ingest")?;
            if has_density(&joined) {
                joined.tier = Tier::Element;
                return Ok(joined);
            }
        }
    }

    Err(anyhow!(
        "material_record: {spec:?} resolved to a datasheet with no bulk density \
         (a particle, amino acid, protein or generated molecule), so it cannot \
         be used as a material"
    ))
}

fn has_density(record: &MaterialRecord) -> bool {
    record.bulk.has(PropertyId::Density)
}

/// The resolved entry's own root-level symbol (`Symbol`, or `symbol` as the
/// amino acid and element datasheets spell it). Nested constituent symbols are
/// deliberately ignored.
fn root_symbol(entry: &Value) -> Option<&str> {
    entry
        .get("Symbol")
        .or_else(|| entry.get("symbol"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// The element datasheet whose root symbol is exactly `symbol`, with its
/// catalogue stem.
///
/// Element datasheets live in the materials catalogue (`*/elements/`), keyed by
/// stem (`026_Fe`), so they are found by content. The match is case-sensitive
/// on purpose -- `Co` is cobalt, `CO` is not an element -- and requires an
/// atomic number so an alloy or compound carrying a `Symbol` cannot match. On
/// the (unexpected) event of several matches the smallest stem wins, keeping
/// the result independent of `DashMap` iteration order.
fn element_datasheet(symbol: &str) -> Option<(String, Value)> {
    let hub = DATA.read();
    hub.materials
        .iter()
        .filter(|kv| {
            let v = kv.value();
            root_symbol(v) == Some(symbol)
                && (v.get("atomic_number").is_some() || v.get("AtomicNumber").is_some())
        })
        .map(|kv| (kv.key().clone(), kv.value().clone()))
        .min_by(|a, b| a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::{DataHub, DATA_TEST_LOCK};
    use periodica_mat::CrystalSystem;

    fn bundled_data_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
    }

    /// Empty the registry so the call under test has to load lazily, exactly
    /// as it does in a downstream crate that never calls a loader. Other
    /// modules' tests swap fixture trees into the shared hub, so starting from
    /// whatever they left behind would make these tests order-dependent.
    fn with_empty_registry<T>(f: impl FnOnce() -> T) -> Option<T> {
        if !bundled_data_dir().is_dir() {
            eprintln!("skipping: bundled data not present");
            return None;
        }
        let _g = DATA_TEST_LOCK.lock();
        *DATA.write() = DataHub::empty();
        Some(f())
    }

    fn assert_close(got: f64, want: f64, rel: f64) {
        assert!(
            ((got - want) / want).abs() <= rel,
            "got {got}, want {want} (+/- {}%)",
            rel * 100.0
        );
    }

    #[test]
    fn iron_loads_lazily_and_is_a_bcc_solid() {
        with_empty_registry(|| {
            let fe = material_record("Fe", None).expect("Fe");
            let rho = fe.bulk.get(PropertyId::Density).expect("density");
            assert_close(rho, 7874.0, 0.01);
            assert_eq!(fe.crystal_system(), CrystalSystem::Bcc);
            assert_eq!(fe.tier, Tier::Element);
            // The registry really was populated by the call itself.
            assert!(DATA.read().root.is_some());
        });
    }

    #[test]
    fn aluminium_is_fcc() {
        with_empty_registry(|| {
            let al = material_record("Al", None).expect("Al");
            assert_eq!(al.crystal_system(), CrystalSystem::Fcc);
            assert_close(al.bulk.get(PropertyId::Density).unwrap(), 2700.0, 0.01);
        });
    }

    #[test]
    fn scoped_atom_lookup_gets_the_same_element_join() {
        with_empty_registry(|| {
            let fe = material_record("Fe", Some(Scope::Atom)).expect("Fe as an atom");
            assert_eq!(fe.crystal_system(), CrystalSystem::Bcc);
        });
    }

    #[test]
    fn fracture_toughness_is_si() {
        // active/materials/Aluminum_6061_T6.json:
        //   "FractureToughness_KIC_MPa_sqrt_m": 29, "Density_kg_m3": 2700
        with_empty_registry(|| {
            let r = material_record("Aluminum_6061_T6", None).expect("Aluminum_6061_T6");
            let kic = r
                .bulk
                .get(PropertyId::FractureToughnessKic)
                .expect("K_IC present");
            assert_close(kic, 29.0e6, 1e-12); // Pa*m^0.5
            assert_close(r.bulk.get(PropertyId::Density).unwrap(), 2700.0, 1e-12);
        });
    }

    #[test]
    fn registry_and_catalogue_materials_resolve_directly() {
        with_empty_registry(|| {
            // derived/alloys (a registry tier).
            let r = material_record("Steel-1018", None).expect("Steel-1018");
            assert_close(r.bulk.get(PropertyId::Density).unwrap(), 7870.0, 0.01);
            // active/materials (catalogue only, reached by Get's fallback).
            // PhysicalProperties.Density_kg_m3 is 2640; the per-mineral
            // densities nested under MineralComposition must not win.
            let g = material_record("Granite_Westerly", None).expect("Granite_Westerly");
            assert_close(g.bulk.get(PropertyId::Density).unwrap(), 2640.0, 1e-9);
        });
    }

    #[test]
    fn unknown_name_is_an_error() {
        with_empty_registry(|| {
            let e = material_record("Unobtainium_9000", None).unwrap_err();
            assert!(format!("{e:#}").contains("Unobtainium_9000"), "{e:#}");
        });
    }

    #[test]
    fn empty_spec_is_an_error() {
        assert!(material_record("  ", None).is_err());
    }

    #[test]
    fn molecule_and_formula_specs_are_clear_errors() {
        with_empty_registry(|| {
            // derived/molecules/H2O.json: constituents and mass only.
            let e = material_record("H2O", None).unwrap_err().to_string();
            assert!(e.contains("no bulk density"), "{e}");

            let e = material_record("{H=2,O=1}", None).unwrap_err().to_string();
            assert!(e.contains("formula"), "{e}");

            // The join must not turn a composition into its first constituent.
            let e = material_record("{Fe=1,C=1}", None).unwrap_err().to_string();
            assert!(e.contains("formula"), "{e}");
        });
    }

    #[test]
    fn particles_are_not_materials() {
        with_empty_registry(|| {
            let e = material_record("Proton", None).unwrap_err().to_string();
            assert!(e.contains("no bulk density"), "{e}");
        });
    }

    #[test]
    fn an_explicitly_loaded_root_is_not_replaced() {
        with_empty_registry(|| {
            let dir = tempfile::tempdir().unwrap();
            let w = |rel: &str, body: &str| {
                let p = dir.path().join(rel);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(p, body).unwrap();
            };
            w(
                "config/composition_rules.json",
                r#"{"tier_definitions": [], "derived_root": "derived"}"#,
            );
            w(
                "derived/alloys/Fixture.json",
                r#"{"Name": "Fixture", "Properties": {"Density_kgm3": 1234}}"#,
            );
            data_loader::reload_registry(dir.path()).unwrap();

            let r = material_record("Fixture", None).expect("fixture");
            assert_eq!(r.bulk.get(PropertyId::Density), Some(1234.0));
            // The bundled corpus was not loaded over the explicit root.
            assert!(material_record("Fe", None).is_err());
        });
    }
}
