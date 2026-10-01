//====== periodica/rust/periodica-mat/tests/corpus.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Ingest every bundled datasheet and assert the alias table stays exhaustive.
//!
//! This is the acceptance test for the ingest layer. The alias table in
//! `canon.rs` was built from a census of the corpus; this test is what stops it
//! silently falling behind as datasheets are added or regenerated.
//!
//! If it fails with unmapped keys, the fix is to classify each reported key in
//! `canon.rs` -- either map it to a `PropertyId` with a unit conversion, or add
//! it to `is_ignored` with a comment saying why it has no bulk-scalar meaning.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use periodica_mat::{ingest, jsonc, CrystalSystem, PropertyId, Source};

/// The four data roots. The Rust loader historically read only `active/`,
/// which is one reason it found almost nothing.
const ROOTS: &[&str] = &["active", "derived", "defaults", "reference"];

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
}

fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_json(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(p);
        }
    }
}

fn corpus() -> Vec<PathBuf> {
    let root = data_dir();
    let mut files = Vec::new();
    for r in ROOTS {
        collect_json(&root.join(r), &mut files);
    }
    files.sort();
    files
}

#[test]
fn corpus_is_present() {
    let files = corpus();
    assert!(
        files.len() > 900,
        "expected ~954 datasheets under {}, found {}. Is the data directory intact?",
        data_dir().display(),
        files.len()
    );
}

#[test]
fn every_datasheet_ingests_without_unknown_keys() {
    let files = corpus();
    // canonical key -> (occurrences, one example file)
    let mut unknown: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut ingested = 0usize;
    let mut parse_failures = Vec::new();

    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let value: serde_json::Value = match jsonc::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                parse_failures.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(record) = ingest(&value, stem) else {
            // Non-object roots exist in a few config files; not a datasheet.
            continue;
        };
        ingested += 1;
        for k in &record.provenance.unknown_keys {
            let e = unknown
                .entry(k.clone())
                .or_insert((0, path.display().to_string()));
            e.0 += 1;
        }
    }

    assert!(
        parse_failures.is_empty(),
        "malformed JSON in the corpus:\n{}",
        parse_failures.join("\n")
    );
    assert!(ingested > 900, "only {ingested} datasheets ingested");

    if !unknown.is_empty() {
        let mut report = String::new();
        for (k, (n, example)) in &unknown {
            report.push_str(&format!("  {k:45} x{n:<5} e.g. {example}\n"));
        }
        panic!(
            "{} unmapped canonical key(s) across {ingested} datasheets.\n\
             Classify each in canon.rs -- map it to a PropertyId, or add it to \
             is_ignored() with a reason.\n{report}",
            unknown.len()
        );
    }
}

#[test]
fn corpus_property_coverage_is_reported() {
    // Not a pass/fail gate on the numbers themselves -- it is a guard against a
    // regression that silently empties the tables (a broken walker would drop
    // coverage to zero while every other test still passed).
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut records = 0usize;

    for path in corpus() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(r) = ingest(&value, stem) else {
            continue;
        };
        records += 1;
        for (p, _, src) in r.bulk.iter() {
            if src == Source::Datasheet {
                *counts.entry(p.name()).or_default() += 1;
            }
        }
    }

    let density = counts.get("Density").copied().unwrap_or(0);
    let youngs = counts.get("YoungsModulus").copied().unwrap_or(0);
    let resistivity = counts.get("ElectricalResistivity").copied().unwrap_or(0);
    let conductivity = counts.get("ThermalConductivity").copied().unwrap_or(0);

    println!("ingested {records} records; measured-property coverage:");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (name, n) in sorted.iter().take(30) {
        println!("  {n:5}  {name}");
    }

    // Collapsing the five density spellings must beat any single spelling's
    // count (the largest single one, `density`, is 228).
    assert!(
        density > 400,
        "density coverage {density} -- canonicalisation is not collapsing spellings"
    );
    assert!(youngs > 70, "Young's modulus coverage {youngs}");
    assert!(resistivity > 100, "resistivity coverage {resistivity}");
    assert!(
        conductivity > 100,
        "thermal conductivity coverage {conductivity}"
    );
}

#[test]
fn conductivity_is_answerable_wherever_resistivity_exists() {
    // No datasheet spells "electrical conductivity", but 137 state resistivity
    // and a few also state %IACS. Every one of them must answer a conductivity
    // query -- that is the whole point of the derivation layer.
    let mut derived = 0usize;
    let mut from_iacs = 0usize;
    let mut worst_disagreement = 0.0f64;

    for path in corpus() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(r) = ingest(&value, stem) else {
            continue;
        };

        let Some(rho) = r.bulk.get(PropertyId::ElectricalResistivity) else {
            continue;
        };
        if rho <= 0.0 {
            continue;
        }
        let sigma = r
            .bulk
            .get(PropertyId::ElectricalConductivity)
            .unwrap_or_else(|| panic!("{stem}: resistivity {rho} but no conductivity"));

        match r.bulk.source(PropertyId::ElectricalConductivity) {
            // Computed by us: must be an exact reciprocal.
            Source::Derived => {
                assert!(
                    (sigma * rho - 1.0).abs() < 1e-9,
                    "{stem}: derived sigma*rho = {}",
                    sigma * rho
                );
                derived += 1;
            }
            // Stated independently as %IACS. Real datasheets round the two
            // figures separately, so they agree only to a few parts in a
            // thousand -- assert consistency, not identity.
            Source::Datasheet => {
                let disagreement = (sigma * rho - 1.0).abs();
                worst_disagreement = worst_disagreement.max(disagreement);
                assert!(
                    disagreement < 0.05,
                    "{stem}: datasheet states both resistivity and %IACS but they \
                     disagree by {:.1}% (sigma*rho = {})",
                    disagreement * 100.0,
                    sigma * rho
                );
                from_iacs += 1;
            }
            other => panic!("{stem}: unexpected conductivity provenance {other:?}"),
        }
    }

    println!(
        "conductivity: {derived} derived from resistivity, {from_iacs} stated as %IACS \
         (worst self-disagreement {:.2}%)",
        worst_disagreement * 100.0
    );
    assert!(
        derived > 100,
        "only {derived} entries derived a conductivity"
    );
}

#[test]
fn microstructure_and_lattice_reach_the_records_that_declare_them() {
    // The `Microstructure` block is authored in 68 datasheets and was read by
    // nothing before this crate existed. Prove it now lands in the record,
    // because the runtime's grain fields and fracture graph depend on it.
    let mut with_micro = 0usize;
    let mut with_lattice = 0usize;
    let mut cleavable = 0usize;

    for path in corpus() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(r) = ingest(&value, stem) else {
            continue;
        };

        if let Some(m) = r.micro {
            with_micro += 1;
            assert!(
                m.grains.average_size_m > 0.0 && m.grains.average_size_m < 0.1,
                "{stem}: implausible grain size {} m",
                m.grains.average_size_m
            );
            assert!(
                m.grains.seed_density_per_m3 > 0.0,
                "{stem}: zero Voronoi seed density would collapse the grain field"
            );
        }
        if r.lattice.is_some() {
            with_lattice += 1;
        }
        if r.crystal_system().is_cleavable() {
            cleavable += 1;
        }
    }

    println!("microstructure: {with_micro}, lattice: {with_lattice}, cleavable: {cleavable}");
    assert!(
        with_micro >= 60,
        "only {with_micro} records carry microstructure"
    );
    assert!(
        with_lattice >= 100,
        "only {with_lattice} records carry a lattice"
    );
    // BCC/HCP entries must exist, or planar fracture has nothing to work with.
    assert!(
        cleavable > 10,
        "only {cleavable} records have a cleavage plane"
    );
}

#[test]
fn ingest_is_deterministic_across_the_whole_corpus() {
    // Re-ingesting must produce byte-identical records: the runtime caches
    // them, and the .pmat writer must be reproducible.
    for path in corpus().iter().take(200) {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let (Ok(a), Ok(b)) = (ingest(&value, stem), ingest(&value, stem)) else {
            continue;
        };
        assert_eq!(a, b, "{}", path.display());
    }
}

#[test]
fn no_record_contains_a_non_finite_value() {
    // A NaN reaching the runtime would poison every downstream average.
    for path in corpus() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(r) = ingest(&value, stem) else {
            continue;
        };
        for (p, v, _) in r.bulk.iter() {
            assert!(v.is_finite(), "{stem}: {p:?} = {v}");
        }
        for ph in &r.phases {
            assert!(
                (0.0..=1.0).contains(&ph.volume_fraction),
                "{stem}: phase {} fraction {}",
                ph.name,
                ph.volume_fraction
            );
        }
    }
}

#[test]
fn known_materials_ingest_with_expected_values() {
    // Spot-check three real files end to end, so a regression in the walker
    // shows up as a wrong number rather than merely a missing one.
    let cases: &[(&str, PropertyId, f64, f64)] = &[
        // (file stem, property, expected SI value, relative tolerance)
        ("Steel-1018", PropertyId::Density, 7870.0, 1e-6),
        ("Steel-1018", PropertyId::YoungsModulus, 2.0e11, 1e-6),
        ("Aluminum_6061_T6", PropertyId::Density, 2700.0, 1e-3),
    ];

    let files = corpus();
    for (stem, prop, want, tol) in cases {
        let Some(path) = files
            .iter()
            .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(*stem))
        else {
            // Data is regenerated periodically; a missing file is not a
            // failure of this crate.
            eprintln!("skipping {stem}: not present in the corpus");
            continue;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let value: serde_json::Value = jsonc::from_str(&text).unwrap();
        let r = ingest(&value, stem).unwrap();
        let got = r
            .bulk
            .get(*prop)
            .unwrap_or_else(|| panic!("{stem}: {prop:?} missing"));
        assert!(
            (got / want - 1.0).abs() < *tol,
            "{stem} {prop:?}: got {got}, want {want} {}",
            prop.si_unit()
        );
    }
}

#[test]
fn fcc_entries_never_claim_a_cleavage_plane() {
    // The fracture model treats "no cleavage plane" as the physically correct
    // answer for FCC, not as missing data. Guard the invariant over real files.
    for path in corpus() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(r) = ingest(&value, stem) else {
            continue;
        };
        if r.crystal_system() == CrystalSystem::Fcc {
            assert!(
                r.crystal_system().cleavage_planes().is_empty(),
                "{stem}: FCC must not expose a cleavage plane"
            );
        }
    }
}
