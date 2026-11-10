//! F2a: gates on the isotropic elastic identities.

use std::path::{Path, PathBuf};

use periodica_mat::derive::{fill_derived_with, gates_for, DeriveGates};
use periodica_mat::{
    fill_derived, ingest, jsonc, MaterialRecord, Orthotropy, PropertyId as P, PropertyTable, Source,
};

fn table(pairs: &[(P, f64)]) -> PropertyTable {
    let mut t = PropertyTable::new();
    for (p, v) in pairs {
        t.set(*p, *v, Source::Datasheet);
    }
    t
}

fn data_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
}

/// Loads one sheet by its path under the data root. Never by file name alone: names repeat
/// across `derived/` and `reference/`, and `read_dir` order differs between platforms.
fn sheet(rel: &str) -> MaterialRecord {
    let path = data_root().join(rel);
    let text = std::fs::read_to_string(&path).expect("sheet readable");
    let value: serde_json::Value = jsonc::from_str(&text).expect("sheet parses");
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("file name");
    ingest(&value, name).expect("sheet ingests")
}

#[test]
fn closed_gate_derives_no_shear_or_bulk() {
    let mut t = table(&[(P::YoungsModulus, 100e9), (P::PoissonsRatio, 0.3)]);
    let gates = DeriveGates {
        isotropic_elastic: false,
    };
    fill_derived_with(&mut t, gates);
    assert!(t.get(P::ShearModulus).is_none() && t.get(P::BulkModulus).is_none());
}

#[test]
fn orthotropic_record_gets_no_derived_g_or_k() {
    let mut r = MaterialRecord::default();
    r.bulk.set(P::YoungsModulus, 11e9, Source::Datasheet);
    r.bulk.set(P::PoissonsRatio, 0.3, Source::Datasheet);
    r.orthotropy = Some(Orthotropy::default());
    let gates = gates_for(&r);
    fill_derived_with(&mut r.bulk, gates);
    assert!(r.bulk.get(P::ShearModulus).is_none() && r.bulk.get(P::BulkModulus).is_none());
}

#[test]
fn timber_keeps_e_and_nu_but_gets_no_g_or_k() {
    let r = sheet("derived/composites/Timber-C24.json");
    assert!(r.bulk.get(P::YoungsModulus).is_some() && r.bulk.get(P::PoissonsRatio).is_some());
    assert!(r.bulk.get(P::ShearModulus).is_none(), "G derived");
    assert!(r.bulk.get(P::BulkModulus).is_none(), "K derived");
}

#[test]
fn steel_1018_still_gets_g_and_k() {
    let r = sheet("derived/alloys/Steel-1018.json");
    assert!(r.bulk.get(P::ShearModulus).is_some(), "G missing");
    assert!(r.bulk.get(P::BulkModulus).is_some(), "K missing");
}

#[test]
fn near_incompressible_k_is_estimated_g_is_derived() {
    let mut t = table(&[(P::YoungsModulus, 2.0993e6), (P::PoissonsRatio, 0.4998)]);
    fill_derived(&mut t);
    assert_eq!(t.source(P::BulkModulus), Source::Estimated);
    assert_eq!(t.source(P::ShearModulus), Source::Derived);
    let mut s = table(&[(P::YoungsModulus, 200e9), (P::PoissonsRatio, 0.29)]);
    fill_derived(&mut s);
    assert_eq!(s.source(P::BulkModulus), Source::Derived);
}

/// Lists the corpus sheets the class predicate matches (run with --nocapture).
#[test]
fn class_predicate_matches_are_listed() {
    let mut hits = Vec::new();
    let mut stack = vec![data_root()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let Ok(v): Result<serde_json::Value, _> = jsonc::from_str(&text) else {
                continue;
            };
            let Ok(r) = ingest(&v, "x") else { continue };
            let closed = !gates_for(&r).isotropic_elastic;
            if closed && r.orthotropy.is_none() {
                hits.push(format!("{} [{:?}]", p.display(), r.category));
            }
        }
    }
    hits.sort();
    println!("CLASS MATCHES ({}):\n{}", hits.len(), hits.join("\n"));
    assert!(hits.iter().any(|h| h.contains("Timber-C24")));
}
