//! F1: derived Reflectance honours k; Emissivity is not mirrored from Absorptance.

use periodica_mat::{fill_derived, PropertyId as P, PropertyTable, Source};

fn table(pairs: &[(P, f64)]) -> PropertyTable {
    let mut t = PropertyTable::new();
    for (p, v) in pairs {
        t.set(*p, *v, Source::Datasheet);
    }
    t
}

#[test]
fn reflectance_uses_extinction_k() {
    let mut t = table(&[(P::RefractiveIndexN, 1.2), (P::ExtinctionK, 7.26)]);
    fill_derived(&mut t);
    let r = t.get(P::Reflectance).unwrap();
    assert!((r - 0.917).abs() < 0.002, "R = {r}");
    assert_eq!(t.source(P::Reflectance), Source::Derived);
}

#[test]
fn reflectance_k_zero_form_is_estimated_with_transmittance() {
    let mut t = table(&[(P::RefractiveIndexN, 1.52), (P::Transmittance, 0.9)]);
    fill_derived(&mut t);
    let r = t.get(P::Reflectance).unwrap();
    assert!((r - 0.0426).abs() < 0.0005, "R = {r}");
    assert_eq!(t.source(P::Reflectance), Source::Estimated);
}

#[test]
fn index_alone_gives_no_reflectance() {
    let mut t = table(&[(P::RefractiveIndexN, 1.52)]);
    fill_derived(&mut t);
    assert!(t.get(P::Reflectance).is_none());
}

#[test]
fn absorptance_does_not_become_emissivity() {
    let mut t = table(&[(P::Reflectance, 0.95), (P::Transmittance, 0.0)]);
    fill_derived(&mut t);
    assert!((t.get(P::Absorptance).unwrap() - 0.05).abs() < 1e-9);
    assert!(t.get(P::Emissivity).is_none());
}

#[test]
fn emissivity_does_not_become_absorptance() {
    let mut t = table(&[(P::Emissivity, 0.8)]);
    fill_derived(&mut t);
    assert!(t.get(P::Absorptance).is_none());
}
