//! Column export: format, round trip, determinism and the real corpus.

use std::path::{Path, PathBuf};

use periodica_mat::export::*;
use periodica_mat::record::MaterialRecord;
use periodica_mat::{ingest, PropertyId, Source};
use serde_json::json;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("s5_export_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn rec(name: &str, density: f64) -> (String, MaterialRecord) {
    let v = json!({"Name": name, "Properties": {"Density_kgm3": density}});
    (name.to_string(), ingest(&v, name).unwrap())
}

fn small() -> ColumnSet {
    let mut empty = rec("zeta", 1.0);
    empty.1.bulk.clear(PropertyId::Density);
    let recs = vec![rec("mid", 7870.0), empty, rec("alpha", 2700.0)];
    columns(&recs, &[PropertyId::Density, PropertyId::YoungsModulus]).unwrap()
}

fn corpus_set() -> (ColumnSet, Corpus) {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
    let corpus = load_corpus(&data);
    let props: Vec<PropertyId> = PropertyId::ALL
        .iter()
        .copied()
        .filter(|&p| corpus.records.iter().any(|(_, r)| r.bulk.has(p)))
        .collect();
    let mut set = columns(&corpus.records, &props).unwrap();
    set.shadowed = corpus.shadowed.clone();
    (set, corpus)
}

fn bits(c: &Column) -> Vec<u64> {
    c.values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn round_trip_is_bit_equal() {
    let dir = tmp("rt");
    let set = small();
    write_dir(&set, &dir).unwrap();
    let back = read_dir(&dir).unwrap();
    assert_eq!(back.sheet_ids, set.sheet_ids);
    for (a, b) in set.columns.iter().zip(&back.columns) {
        assert_eq!(a.prop, b.prop);
        assert_eq!(bits(a), bits(b));
        assert_eq!(a.sources, b.sources);
    }
}

#[test]
fn missing_value_is_nan_with_source_zero() {
    let set = small();
    let d = &set.columns[0];
    assert_eq!(set.sheet_ids, ["alpha", "mid", "zeta"]);
    assert_eq!(d.values[0], 2700.0);
    assert_eq!(d.sources[0], Source::Datasheet as u8);
    assert!(d.values[2].is_nan());
    assert_eq!(d.sources[2], 0);
    assert!(set.columns[1].values.iter().all(|v| v.is_nan()));
}

#[test]
fn file_sizes_match_rows() {
    let dir = tmp("sizes");
    let m = write_dir(&small(), &dir).unwrap();
    for p in &m.properties {
        let f = std::fs::metadata(dir.join(&p.values_file)).unwrap().len();
        let s = std::fs::metadata(dir.join(&p.source_file)).unwrap().len();
        assert_eq!((f, s), (m.rows as u64 * 8, m.rows as u64));
    }
}

#[test]
fn duplicate_id_and_short_file_fail_early() {
    let recs = vec![rec("a", 1.0), rec("a", 2.0)];
    assert!(matches!(
        columns(&recs, &[PropertyId::Density]),
        Err(ExportError::DuplicateSheetId(_))
    ));
    let dir = tmp("short");
    write_dir(&small(), &dir).unwrap();
    let f = dir.join("Density.f64");
    let b = std::fs::read(&f).unwrap();
    std::fs::write(&f, &b[..b.len() - 1]).unwrap();
    assert!(matches!(read_dir(&dir), Err(ExportError::Length { .. })));
}

#[test]
fn unknown_property_in_manifest_is_an_error() {
    let dir = tmp("unk");
    write_dir(&small(), &dir).unwrap();
    let p = dir.join("manifest.json");
    let t = std::fs::read_to_string(&p)
        .unwrap()
        .replacen("\"Density\"", "\"Bogus\"", 1);
    std::fs::write(&p, t).unwrap();
    assert!(matches!(
        read_dir(&dir),
        Err(ExportError::UnknownProperty(_))
    ));
}

#[test]
fn newer_manifest_with_extras_still_reads() {
    let dir = tmp("v2");
    write_dir(&small(), &dir).unwrap();
    let p = dir.join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    m["format_version"] = json!(2);
    m["future_field"] = json!({"x": 1});
    std::fs::write(&p, serde_json::to_string(&m).unwrap()).unwrap();
    std::fs::write(dir.join("Density.unc"), [0u8; 3]).unwrap();
    assert_eq!(read_dir(&dir).unwrap().sheet_ids.len(), 3);
}

#[test]
fn corpus_export_is_sorted_deterministic_and_hashed() {
    let (set, _) = corpus_set();
    let (d1, d2) = (tmp("c1"), tmp("c2"));
    let m1 = write_dir(&set, &d1).unwrap();
    let m2 = write_dir(&set, &d2).unwrap();
    assert_eq!(m1, m2);
    assert_eq!(
        std::fs::read(d1.join("manifest.json")).unwrap(),
        std::fs::read(d2.join("manifest.json")).unwrap()
    );
    let mut sorted = m1.sheet_ids.clone();
    sorted.sort();
    assert_eq!(sorted, m1.sheet_ids);
    for p in &m1.properties {
        for (f, h) in [
            (&p.values_file, &p.values_hash),
            (&p.source_file, &p.source_hash),
        ] {
            let bytes = std::fs::read(d1.join(f)).unwrap();
            assert_eq!(&fnv1a_hex(&bytes), h, "{f}");
            assert_eq!(h.len(), 16);
        }
    }
    assert_eq!(read_dir(&d1).unwrap().sheet_ids, m1.sheet_ids);
}

#[test]
fn sheet_ids_are_unique_and_shadowed_copies_are_listed() {
    let (set, corpus) = corpus_set();
    let mut ids = set.sheet_ids.clone();
    ids.dedup();
    assert_eq!(ids.len(), set.sheet_ids.len());
    assert_eq!(
        set.sheet_ids.iter().filter(|i| *i == "Steel-1018").count(),
        1
    );
    for name in ["Steel-1018", "Aluminum-6061"] {
        assert!(
            corpus.shadowed.iter().any(|s| s.name == name),
            "{name} not listed as shadowed"
        );
    }
}

#[test]
fn playtow_sheets_and_iron_density_are_rows() {
    let (set, _) = corpus_set();
    for n in ALWAYS_INCLUDE {
        assert!(set.sheet_ids.iter().any(|i| i == n), "{n} missing");
    }
    let dir = tmp("iron");
    let m = write_dir(&set, &dir).unwrap();
    let row = m
        .sheet_ids
        .iter()
        .position(|i| i == "026_Fe")
        .expect("iron row");
    let d = m.properties.iter().find(|p| p.name == "Density").unwrap();
    let raw = std::fs::read(dir.join(&d.values_file)).unwrap();
    let v = f64::from_le_bytes(raw[row * 8..row * 8 + 8].try_into().unwrap());
    assert!((v - 7874.0).abs() < 1e-9, "iron density {v}");
}
