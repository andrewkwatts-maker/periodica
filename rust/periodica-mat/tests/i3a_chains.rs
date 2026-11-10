//! I3a: sheet fields and chains for contact laws, over the bundled sheets.

use std::path::{Path, PathBuf};

use periodica_mat::{ingest, jsonc, MaterialRecord, PropertyId as P, Source};

const DIRS: [&str; 5] = [
    "derived/alloys",
    "derived/ceramics",
    "derived/composites",
    "derived/polymers",
    "active/materials",
];

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
}

fn load(rel: &str) -> MaterialRecord {
    let text = std::fs::read_to_string(data_dir().join(rel)).unwrap();
    let value: serde_json::Value = serde_json::from_str(&jsonc::strip_comments(&text)).unwrap();
    ingest(&value, rel).unwrap()
}

fn sheets() -> Vec<String> {
    let mut out = Vec::new();
    for d in DIRS {
        let mut names: Vec<String> = std::fs::read_dir(data_dir().join(d))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".json"))
            .collect();
        names.sort();
        out.extend(names.into_iter().map(|n| format!("{d}/{n}")));
    }
    out
}

fn tag(r: &MaterialRecord, p: P) -> String {
    let src = match r.bulk.source(p) {
        Source::Absent => return "absent".into(),
        Source::Datasheet => "datasheet",
        Source::Derived => "derived",
        Source::Estimated => "estimated",
        Source::UserOverride => "override",
    };
    format!("{:.6e} {src}", r.bulk.get(p).unwrap())
}

#[test]
fn concrete_estimates_stay_estimated() {
    for g in ["C25", "C30", "C40"] {
        let r = load(&format!("derived/composites/Concrete-{g}.json"));
        for p in [
            P::YoungsModulus,
            P::ShearModulus,
            P::BulkModulus,
            P::UltimateTensileStrength,
            P::TensileStrength,
            P::FractureEnergy,
            P::Jic,
        ] {
            if g == "C30" && p == P::TensileStrength {
                continue;
            }
            assert_eq!(r.bulk.source(p), Source::Estimated, "{g} {p:?}");
        }
    }
}

#[test]
fn datasheet_values_are_never_restamped() {
    let r = load("derived/composites/Concrete-Mix.json");
    assert_eq!(r.bulk.source(P::YoungsModulus), Source::Datasheet);
}

#[test]
fn c30_tensile_strength_is_datasheet() {
    let r = load("derived/composites/Concrete-C30.json");
    assert_eq!(r.bulk.get(P::TensileStrength), Some(2.9e6));
    assert_eq!(r.bulk.source(P::TensileStrength), Source::Datasheet);
}

#[test]
fn steel_chains_fill_hardness_and_f_t() {
    let r = load("derived/alloys/Steel-B500B.json");
    let sy = r.bulk.get(P::YieldStrength).unwrap();
    assert_eq!(r.bulk.get(P::Hardness), Some(3.0 * sy));
    assert_eq!(r.bulk.source(P::Hardness), Source::Estimated);
    assert_eq!(
        r.bulk.get(P::TensileStrength),
        r.bulk.get(P::UltimateTensileStrength)
    );
    assert_eq!(r.bulk.source(P::TensileStrength), Source::Estimated);
}

#[test]
fn new_fields_ingest_from_a_sheet() {
    let v: serde_json::Value = serde_json::json!({
    "Name": "Probe", "Properties": {
        "SurfaceEnergy_Jm2": 1.1, "SurfaceSlopeRms_rad": 0.07,
        "KineticStaticRatio": 0.8, "TensileStrength_ft_MPa": 3.0
    }});
    let r = ingest(&v, "probe").unwrap();
    assert_eq!(r.bulk.get(P::SurfaceEnergy), Some(1.1));
    assert_eq!(r.bulk.get(P::SurfaceSlopeRms), Some(0.07));
    assert_eq!(r.bulk.get(P::KineticStaticRatio), Some(0.8));
    assert_eq!(r.bulk.get(P::TensileStrength), Some(3.0e6));
    assert_eq!(r.bulk.source(P::SurfaceSlopeRms), Source::Datasheet);
}

/// Writes the I3a diary CSV when `I3A_CSV` names an output path.
#[test]
fn chains_csv() {
    let cols = [
        ("E", P::YoungsModulus),
        ("sigma_y", P::YieldStrength),
        ("H", P::Hardness),
        ("f_t", P::TensileStrength),
        ("G_F", P::FractureEnergy),
        ("theta_q", P::SurfaceSlopeRms),
        ("gamma", P::SurfaceEnergy),
    ];
    let mut csv = String::from("sheet,name");
    for (n, _) in &cols {
        csv.push_str(&format!(",{n}"));
    }
    csv.push('\n');
    for rel in sheets() {
        let r = load(&rel);
        csv.push_str(&format!("{rel},{}", r.name));
        for (_, p) in &cols {
            csv.push_str(&format!(",{}", tag(&r, *p)));
        }
        csv.push('\n');
    }
    assert!(csv.lines().count() > 40);
    if let Ok(out) = std::env::var("I3A_CSV") {
        std::fs::write(out, csv).unwrap();
    }
}
