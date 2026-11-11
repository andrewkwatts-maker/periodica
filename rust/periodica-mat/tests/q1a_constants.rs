//! Q1a: the constants in `chains.rs` and `derive.rs` have a derivation or a reference.
//! `rq_over_ra` and `nu_is_stable` are crate-private; their unit tests are in `chains.rs`
//! and `derive.rs`.

use approx::assert_relative_eq;
use periodica_mat::{fill_derived, ingest, jsonc, PropertyId as P, PropertyTable, Source};
use std::f64::consts::FRAC_PI_2;
use std::path::Path;

fn table(pairs: &[(P, f64)]) -> PropertyTable {
    let mut t = PropertyTable::new();
    for (p, v) in pairs {
        t.set(*p, *v, Source::Datasheet);
    }
    t
}

#[test]
fn slope_applies_the_rms_factor_and_the_source_range() {
    let mut t = table(&[(P::SurfaceRoughnessRa, 1.0e-6)]);
    fill_derived(&mut t);
    let r = FRAC_PI_2.sqrt();
    let want = r * 0.125 * (r * 1.0).powf(0.402);
    assert_relative_eq!(
        t.get(P::SurfaceSlopeRms).unwrap(),
        want,
        max_relative = 1e-12
    );
    assert_eq!(t.source(P::SurfaceSlopeRms), Source::Estimated);
    for ra in [0.1e-6, 10.0e-6] {
        let mut o = table(&[(P::SurfaceRoughnessRa, ra)]);
        fill_derived(&mut o);
        assert!(!o.has(P::SurfaceSlopeRms), "Ra {ra}");
    }
    let mut s = table(&[(P::SurfaceRoughnessRa, 1.0e-6), (P::SurfaceSlopeRms, 0.2)]);
    fill_derived(&mut s);
    assert_eq!(s.get(P::SurfaceSlopeRms), Some(0.2));
    assert_eq!(s.source(P::SurfaceSlopeRms), Source::Datasheet);
}

#[test]
fn slope_correlation_reproduces_the_source_worked_example() {
    // Teertstra (1997) works the correlation for rms roughness 0.4 um and 1.3 um and
    // prints m = 0.0865 and m = 0.139, three significant figures each. R_a is chosen so
    // that R_q is the source's sigma; the rms slope over sqrt(pi / 2) is the source's m.
    let r = FRAC_PI_2.sqrt();
    for (sigma_um, m, half_last_digit) in [(0.4, 0.0865, 0.00005), (1.3, 0.139, 0.0005)] {
        let mut t = table(&[(P::SurfaceRoughnessRa, sigma_um * 1.0e-6 / r)]);
        fill_derived(&mut t);
        let got = t.get(P::SurfaceSlopeRms).unwrap() / r;
        assert!(
            (got - m).abs() < half_last_digit,
            "sigma {sigma_um} um: m {got}"
        );
    }
}

#[test]
fn one_tabor_factor_for_hardness_and_uts() {
    let mut t = table(&[
        (P::UltimateTensileStrength, 400.0e6),
        (P::ElongationPct, 20.0),
    ]);
    fill_derived(&mut t);
    assert_relative_eq!(
        t.get(P::HardnessVickers).unwrap(),
        3.0 * 400.0e6 / 9.80665e6,
        max_relative = 1e-12
    );
    let mut r = table(&[(P::HardnessVickers, 150.0), (P::ElongationPct, 20.0)]);
    fill_derived(&mut r);
    let uts = r.get(P::UltimateTensileStrength).unwrap();
    let mut back = table(&[(P::UltimateTensileStrength, uts), (P::ElongationPct, 20.0)]);
    fill_derived(&mut back);
    assert_relative_eq!(
        back.get(P::HardnessVickers).unwrap(),
        150.0,
        max_relative = 1e-12
    );
    let mut y = table(&[(P::YieldStrength, 250.0e6), (P::ElongationPct, 20.0)]);
    fill_derived(&mut y);
    assert_relative_eq!(y.get(P::Hardness).unwrap(), 750.0e6, max_relative = 1e-12);
}

#[test]
fn no_hardness_from_compressive_strength() {
    let mut t = table(&[(P::CompressiveStrength, 30.0e6)]);
    fill_derived(&mut t);
    assert!(!t.has(P::Hardness));
    let rel = "derived/composites/Concrete-C30.json";
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
    let text = std::fs::read_to_string(dir.join(rel)).unwrap();
    let v: serde_json::Value = serde_json::from_str(&jsonc::strip_comments(&text)).unwrap();
    assert!(!ingest(&v, rel).unwrap().bulk.has(P::Hardness));
}

#[test]
fn poisson_near_half_is_estimated_and_the_limits_give_nothing() {
    let mut t = table(&[(P::YoungsModulus, 3.0e9), (P::PoissonsRatio, 0.49995)]);
    fill_derived(&mut t);
    let k = t.get(P::BulkModulus).unwrap();
    assert!(k.is_finite() && k > 0.0);
    // K = E / (3 (1 - 2 nu)) and G = E / (2 (1 + nu)).
    let (e, nu) = (3.0e9, 0.49995);
    assert_relative_eq!(k, e / (3.0 * (1.0 - 2.0 * nu)), max_relative = 1e-9);
    assert_relative_eq!(
        t.get(P::ShearModulus).unwrap(),
        e / (2.0 * (1.0 + nu)),
        max_relative = 1e-12
    );
    assert_eq!(t.source(P::BulkModulus), Source::Estimated);
    for nu in [0.5, -1.0] {
        let mut o = table(&[(P::YoungsModulus, 3.0e9), (P::PoissonsRatio, nu)]);
        fill_derived(&mut o);
        assert!(!o.has(P::BulkModulus), "nu {nu}");
        assert!(!o.has(P::ShearModulus), "nu {nu}");
    }
}
