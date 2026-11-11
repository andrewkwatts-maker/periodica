//! P2: no computed value is presented as measured. Generated sheets ingest as `Estimated`,
//! water content is a property, and sound speeds follow the stated-input rules.

use std::path::{Path, PathBuf};

use periodica_mat::record::MaterialRecord;
use periodica_mat::{ingest, jsonc, PropertyId as P, Source};
use serde_json::{json, Value};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
}

fn read(rel: &str) -> Value {
    let text = std::fs::read_to_string(data_dir().join(rel)).unwrap();
    jsonc::from_str(&text).unwrap()
}

fn load(rel: &str) -> MaterialRecord {
    ingest(&read(rel), "t").unwrap()
}

fn with_marker(marker: Option<(&str, Value)>) -> MaterialRecord {
    let mut v = json!({
        "Name": "X",
        "Density_kg_m3": 1000.0,
        "MechanicalProperties": { "YoungsModulus_GPa": 2.0, "PoissonsRatio": 0.3 },
        "PhaseComposition": { "Phases": [ { "Name": "p", "Density_kg_m3": 900.0 } ] }
    });
    if let Some((k, m)) = marker {
        v[k] = m;
    }
    ingest(&v, "x").unwrap()
}

fn all_sources(r: &MaterialRecord) -> Vec<Source> {
    let s: Vec<Source> = P::ALL
        .iter()
        .filter(|p| r.bulk.get(**p).is_some())
        .map(|p| r.bulk.source(*p))
        .collect();
    s
}

#[test]
fn auto_generated_marker_makes_every_value_estimated() {
    let marked = with_marker(Some(("_derivation", json!({"source": "auto_generated"}))));
    assert!(!all_sources(&marked).is_empty());
    assert_eq!(marked.bulk.source(P::YoungsModulus), Source::Estimated);
    assert_eq!(marked.bulk.source(P::Density), Source::Estimated);
    let phase = &marked.phases[0].properties;
    assert_eq!(phase.source(P::Density), Source::Estimated);
    let plain = with_marker(None);
    assert_eq!(plain.bulk.source(P::YoungsModulus), Source::Datasheet);
    assert_eq!(
        plain.phases[0].properties.source(P::Density),
        Source::Datasheet
    );
    let other = with_marker(Some(("_derivation", json!({"source": "handbook"}))));
    assert_eq!(other.bulk.source(P::Density), Source::Datasheet);
}

#[test]
fn model_used_marks_a_sheet_unless_empty() {
    let m = with_marker(Some((
        "derived_properties",
        json!({"model_used": "Voigt-Reuss"}),
    )));
    assert_eq!(m.bulk.source(P::Density), Source::Estimated);
    let e = with_marker(Some(("derived_properties", json!({"model_used": ""}))));
    assert_eq!(e.bulk.source(P::Density), Source::Datasheet);
}

#[test]
fn blood_is_estimated_throughout_and_has_no_sound_speed() {
    let r = load("active/biological_materials/Blood.json");
    for p in P::ALL {
        if r.bulk.get(*p).is_some() {
            let s = r.bulk.source(*p);
            assert!(
                s != Source::Datasheet && s != Source::Derived,
                "{p:?} {s:?}"
            );
        }
    }
    assert!(r.bulk.get(P::YoungsModulus).is_some());
    assert_eq!(r.bulk.get(P::SoundSpeedLongitudinal), None);
    assert_eq!(r.bulk.get(P::SoundSpeedShear), None);
}

#[test]
fn cartilage_water_content_is_stated_and_closes_the_longitudinal_speed() {
    let r = load("active/tissues/cartilage.json");
    assert!((r.bulk.get(P::WaterContent).unwrap() - 0.7).abs() < 1e-12);
    assert_eq!(r.bulk.source(P::WaterContent), Source::Datasheet);
    assert_eq!(r.bulk.get(P::SoundSpeedLongitudinal), None);
    let (e, nu, rho) = (
        r.bulk.get(P::YoungsModulus).unwrap(),
        r.bulk.get(P::PoissonsRatio).unwrap(),
        r.bulk.get(P::Density).unwrap(),
    );
    let g = e / (2.0 * (1.0 + nu));
    let vs = r.bulk.get(P::SoundSpeedShear).unwrap();
    assert!((vs / (g / rho).sqrt() - 1.0).abs() < 1e-12);
}

#[test]
fn steel_1018_is_a_datasheet_with_both_speeds() {
    let r = load("derived/alloys/Steel-1018.json");
    assert_eq!(r.bulk.source(P::YoungsModulus), Source::Datasheet);
    assert!(r.bulk.get(P::SoundSpeedLongitudinal).is_some());
    assert!(r.bulk.get(P::SoundSpeedShear).is_some());
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

fn is_generated(v: &Value) -> bool {
    let auto = v["_derivation"]["source"].as_str() == Some("auto_generated");
    let model = v["derived_properties"]["model_used"]
        .as_str()
        .is_some_and(|m| !m.is_empty());
    auto || model
}

#[test]
fn corpus_has_no_false_sound_speeds() {
    let mut files = Vec::new();
    for r in ["active", "derived", "defaults", "reference"] {
        collect_json(&data_dir().join(r), &mut files);
    }
    files.sort();
    let mut checked = 0;
    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(v): Result<Value, _> = jsonc::from_str(&text) else {
            continue;
        };
        let Ok(r) = ingest(&v, "t") else { continue };
        checked += 1;
        let lon = r.bulk.get(P::SoundSpeedLongitudinal).is_some();
        let sheet = path.display();
        if is_generated(&v) {
            assert!(!lon && r.bulk.get(P::SoundSpeedShear).is_none(), "{sheet}");
        }
        let pores = [P::WaterContent, P::Porosity]
            .iter()
            .any(|p| r.bulk.get(*p).is_some_and(|x| x > 0.0));
        let solid_form = r.bulk.get(P::ShearModulus).is_some();
        let derived = r.bulk.source(P::SoundSpeedLongitudinal) == Source::Estimated;
        assert!(!(pores && lon && solid_form && derived), "{sheet}");
    }
    assert!(checked > 100);
}
