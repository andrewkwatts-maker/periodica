//! F2b: Tabor's rule for ductile metals only; plane-strain Irwin.

use std::path::Path;

use periodica_mat::{
    fill_derived, ingest, jsonc, MaterialRecord, PropertyId as P, PropertyTable, Source,
};

fn table(pairs: &[(P, f64)]) -> PropertyTable {
    let mut t = PropertyTable::new();
    for (p, v) in pairs {
        t.set(*p, *v, Source::Datasheet);
    }
    t
}

/// Loads one sheet by its path under the data root, never by file name alone.
fn sheet(rel: &str) -> MaterialRecord {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/periodica/data")
        .join(rel);
    let text = std::fs::read_to_string(&path).expect("sheet readable");
    let value: serde_json::Value = jsonc::from_str(&text).expect("sheet parses");
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("file name");
    ingest(&value, name).expect("sheet ingests")
}

#[test]
fn ceramic_like_table_gets_no_uts_from_hardness() {
    let mut t = table(&[(P::HardnessVickers, 1500.0)]);
    fill_derived(&mut t);
    assert!(t.get(P::UltimateTensileStrength).is_none());
}

#[test]
fn steel_like_table_gets_both_tabor_directions() {
    let mut a = table(&[
        (P::UltimateTensileStrength, 440e6),
        (P::ElectricalResistivity, 1.4e-7),
    ]);
    fill_derived(&mut a);
    assert!((a.get(P::HardnessVickers).unwrap() - 440.0 / 3.3).abs() < 1e-9);
    let mut b = table(&[(P::HardnessVickers, 200.0), (P::ElongationPct, 20.0)]);
    fill_derived(&mut b);
    assert!((b.get(P::UltimateTensileStrength).unwrap() - 660e6).abs() < 1.0);
}

#[test]
fn elastomer_elongation_is_not_ductile_metal() {
    let mut t = table(&[(P::YieldStrength, 16e6), (P::ElongationPct, 500.0)]);
    fill_derived(&mut t);
    assert!(t.get(P::Hardness).is_none());
}

#[test]
fn timber_and_rubber_get_no_hardness() {
    for rel in [
        "derived/composites/Timber-C24.json",
        "derived/polymers/Rubber-Natural.json",
    ] {
        let r = sheet(rel);
        assert!(r.bulk.get(P::Hardness).is_none(), "{rel} Hardness");
        assert!(r.bulk.get(P::HardnessVickers).is_none(), "{rel} HV");
    }
}

#[test]
fn masonry_hardness_is_three_times_compressive_strength() {
    for rel in [
        "derived/ceramics/Mortar-CementLime.json",
        "derived/ceramics/Brick-Clay.json",
    ] {
        let r = sheet(rel);
        let fc = r.bulk.get(P::CompressiveStrength).expect("f_c");
        assert!(
            (r.bulk.get(P::Hardness).expect("H") - 3.0 * fc).abs() < 1.0,
            "{rel}"
        );
        assert!(r.bulk.get(P::HardnessVickers).is_none(), "{rel} HV");
    }
}

#[test]
fn metal_sheets_keep_their_hardness_values_and_tags() {
    // (path, Hardness, HV, TensileStrength), recorded at 5c70977 before the change.
    let cases = [
        (
            "derived/alloys/Steel-1018.json",
            1235637900.0,
            133.33333333333334,
            440e6,
        ),
        (
            "derived/alloys/Aluminum-6061.json",
            931631750.0,
            93.93939393939394,
            310e6,
        ),
        (
            "derived/alloys/Steel-B500B.json",
            1500e6,
            163.63636363636365,
            540e6,
        ),
    ];
    for (rel, h, hv, ts) in cases {
        let t = sheet(rel).bulk;
        for (p, want) in [
            (P::Hardness, h),
            (P::HardnessVickers, hv),
            (P::TensileStrength, ts),
        ] {
            assert!(
                (t.get(p).unwrap() - want).abs() < 1e-6 * want,
                "{rel} {p:?}"
            );
            assert_eq!(t.source(p), Source::Estimated, "{rel} {p:?} tag");
        }
    }
}

#[test]
fn irwin_is_plane_strain_with_poisson_and_estimated_without() {
    let (k, e, nu) = (29.0e6_f64, 68.9e9_f64, 0.33_f64);
    let mut with = table(&[
        (P::FractureToughnessKic, k),
        (P::YoungsModulus, e),
        (P::PoissonsRatio, nu),
    ]);
    fill_derived(&mut with);
    let want = k * k * (1.0 - nu * nu) / e;
    assert!((with.get(P::Jic).unwrap() - want).abs() < 1e-9 * want);
    let mut without = table(&[(P::FractureToughnessKic, k), (P::YoungsModulus, e)]);
    fill_derived(&mut without);
    assert!((without.get(P::Jic).unwrap() - k * k / e).abs() < 1e-9 * (k * k / e));
    assert_eq!(without.source(P::Jic), Source::Estimated);
}
