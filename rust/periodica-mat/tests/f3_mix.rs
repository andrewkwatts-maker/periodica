//! F3: mixing-rule fixes (Turner, elastic consistency, Lorentz-Lorenz).

use periodica_mat::{mix, mix_tagged, MaterialRecord, MixError, PropertyId as P, Source};

fn rec(props: &[(P, f64)]) -> MaterialRecord {
    let mut r = MaterialRecord::default();
    for (p, v) in props {
        r.bulk.set(*p, *v, Source::Datasheet);
    }
    r
}

fn hill(a: f64, b: f64, fa: f64) -> f64 {
    let fb = 1.0 - fa;
    0.5 * (fa * a + fb * b) + 0.5 / (fa / a + fb / b)
}

#[test]
fn turner_thermal_expansion_al_steel() {
    let al = rec(&[(P::ThermalExpansion, 23e-6), (P::BulkModulus, 68.6e9)]);
    let st = rec(&[(P::ThermalExpansion, 12e-6), (P::BulkModulus, 167e9)]);
    let (a, src) = mix_tagged(&[(&al, 0.5), (&st, 0.5)], P::ThermalExpansion).unwrap();
    assert!((a - 15.3e-6).abs() / 15.3e-6 < 0.01, "{a}");
    assert_eq!(src, Source::Derived);
    let nok = rec(&[(P::ThermalExpansion, 12e-6)]);
    let (v, src) = mix_tagged(&[(&al, 0.5), (&nok, 0.5)], P::ThermalExpansion).unwrap();
    assert!((v - 17.5e-6).abs() < 1e-12);
    assert_eq!(src, Source::Estimated);
}

#[test]
fn mixed_elastic_constants_are_isotropic_consistent() {
    let a = rec(&[(P::YoungsModulus, 200e9), (P::PoissonsRatio, 0.30)]);
    let b = rec(&[(P::YoungsModulus, 70e9), (P::PoissonsRatio, 0.33)]);
    let e = [(&a, 0.3), (&b, 0.7)];
    let k = hill(200e9 / (3.0 * 0.4), 70e9 / (3.0 * 0.34), 0.3);
    let g = hill(200e9 / 2.6, 70e9 / 2.66, 0.3);
    let em = mix(&e, P::YoungsModulus).unwrap();
    let nu = mix(&e, P::PoissonsRatio).unwrap();
    assert!((em - 9.0 * k * g / (3.0 * k + g)).abs() / em < 1e-12);
    assert!((nu - (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))).abs() < 1e-12);
    assert!((em - 3.0 * k * (1.0 - 2.0 * nu)).abs() / em < 1e-12);
}

#[test]
fn refractive_index_lorentz_lorenz() {
    let a = rec(&[(P::RefractiveIndexN, 1.5)]);
    let b = rec(&[(P::RefractiveIndexN, 1.0)]);
    let n = mix(&[(&a, 0.5), (&b, 0.5)], P::RefractiveIndexN).unwrap();
    assert!((n - 1.232).abs() < 1e-3, "{n}");
    let m = rec(&[(P::RefractiveIndexN, 2.0), (P::ExtinctionK, 3.0)]);
    assert!(matches!(
        mix(&[(&a, 0.5), (&m, 0.5)], P::RefractiveIndexN),
        Err(MixError::NotMixable(_))
    ));
}
