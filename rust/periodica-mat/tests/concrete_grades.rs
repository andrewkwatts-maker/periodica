//! Grade records ingest as concrete and receive the MC2010 derived values.

use std::path::Path;

use periodica_mat::{ingest, jsonc, MaterialRecord, PropertyId as P, Source};

fn load(rel: &str) -> MaterialRecord {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/periodica/data")
        .join(rel);
    let text = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&jsonc::strip_comments(&text)).unwrap();
    ingest(&value, rel).unwrap()
}

#[test]
fn grades_are_flagged_and_derived() {
    for (g, fck) in [("C25", 25.0), ("C30", 30.0), ("C40", 40.0)] {
        let r = load(&format!("derived/composites/Concrete-{g}.json"));
        assert_eq!(r.class.as_deref(), Some("Concrete"));
        assert_eq!(r.bulk.get(P::CompressiveStrength), Some(fck * 1.0e6));
        assert_eq!(r.bulk.get(P::AggregateMaxSize), Some(0.016));
        assert_ne!(r.bulk.source(P::YoungsModulus), Source::Datasheet);
        assert!(r.bulk.get(P::YoungsModulus).unwrap() > 20.0e9);
        for p in [
            P::MeanCompressiveStrength,
            P::UltimateTensileStrength,
            P::FractureEnergy,
            P::PeakCompressiveStrain,
            P::UltimateCompressiveStrain,
        ] {
            assert!(
                r.bulk.has(p) && r.bulk.source(p) != Source::Datasheet,
                "{g} {p:?}"
            );
        }
    }
}

#[test]
fn mix_datasheet_values_win() {
    let r = load("derived/composites/Concrete-Mix.json");
    assert_eq!(r.bulk.get(P::YoungsModulus), Some(30.0e9));
    assert_eq!(r.bulk.source(P::YoungsModulus), Source::Datasheet);
    assert!(r.bulk.has(P::MeanCompressiveStrength));
    assert_eq!(r.bulk.source(P::MeanCompressiveStrength), Source::Estimated);
}

#[test]
fn class_label_alone_selects_no_fib_fills() {
    let v = serde_json::json!({
        "Name": "Labelled", "Class": "Concrete",
        "Properties": { "CompressiveStrength_MPa": 30 }
    });
    let r = ingest(&v, "Labelled").unwrap();
    assert_eq!(r.class.as_deref(), Some("Concrete"));
    assert!(!r.bulk.has(P::MeanCompressiveStrength));
    assert!(!r.bulk.has(P::FractureEnergy));
}

#[test]
fn characteristic_strength_alone_selects_fib_fills() {
    let v = serde_json::json!({
        "Name": "Unlabelled",
        "Properties": { "CharacteristicCompressiveStrength_MPa": 30 }
    });
    let r = ingest(&v, "Unlabelled").unwrap();
    assert!(r.class.is_none());
    assert_eq!(r.bulk.get(P::MeanCompressiveStrength), Some(38.0e6));
    assert_eq!(r.bulk.get(P::CompressiveStrength), Some(30.0e6));
    assert_eq!(r.bulk.source(P::CompressiveStrength), Source::Estimated);
    assert!(r.bulk.has(P::FractureEnergy));
}

#[test]
fn rebar_has_class_inputs() {
    let r = load("derived/alloys/Steel-B500B.json");
    assert!(r.class.is_none());
    for p in [
        P::ElongationPct,
        P::UltimateTensileStrength,
        P::YieldStrength,
        P::YoungsModulus,
    ] {
        assert!(r.bulk.has(p), "{p:?}");
    }
}
