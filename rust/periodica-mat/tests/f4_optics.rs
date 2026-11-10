//! F4: gold optical model against Johnson and Christy (1972), and wavelength validity ranges.

use periodica_mat::optics::drude_lorentz::DrudeLorentz;
use periodica_mat::optics::planck::planck;
use periodica_mat::optics::sellmeier::Sellmeier;
use periodica_mat::optics::{Cauchy, Dispersion, OpticsError};
use periodica_mat::property::PropertyId as P;
use periodica_mat::table::{PropertyTable, Source};

/// Johnson and Christy, Phys. Rev. B 6, 4370 (1972), gold: (wavelength um, n, k).
const JC_AU: [(f64, f64, f64); 8] = [
    (0.4133, 1.46, 1.958),
    (0.4509, 1.38, 1.914),
    (0.4959, 1.04, 1.833),
    (0.5209, 0.62, 2.081),
    (0.5486, 0.43, 2.455),
    (0.5821, 0.29, 2.863),
    (0.6168, 0.21, 3.272),
    (0.6595, 0.14, 3.697),
];

fn fresnel(n: f64, k: f64) -> f64 {
    ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k)
}

fn gold() -> Dispersion {
    Dispersion::DrudeLorentz(DrudeLorentz::gold())
}

#[test]
fn gold_matches_johnson_christy_reflectance() {
    let g = gold();
    for (um, n0, k0) in JC_AU {
        let got = g.reflectance(um * 1e-6).unwrap();
        let want = fresnel(n0, k0);
        println!("{um} um: R {got:.3} vs JC {want:.3}");
        assert!((got - want).abs() < 0.05, "{um} um: {got} vs {want}");
        let nk = g.nk(um * 1e-6).unwrap();
        assert!((nk.k - k0).abs() / k0 < 0.1, "{um} um k {} vs {k0}", nk.k);
    }
    for nm in [450.0, 550.0, 650.0] {
        println!("R({nm} nm) = {:.4}", g.reflectance(nm * 1e-9).unwrap());
    }
}

#[test]
fn gold_is_yellow_not_silver() {
    let g = gold();
    assert!(g.reflectance(450e-9).unwrap() < 0.6);
    assert!(g.reflectance(600e-9).unwrap() > 0.9);
    let [r, _, b] = g.base_color().unwrap();
    assert!(r / b > 1.5, "r/b = {}", r / b);
}

#[test]
fn drude_record_without_epsilon_infinity_is_an_error() {
    let mut t = PropertyTable::new();
    t.set(P::DrudePlasmaFrequency, 1.3e16, Source::Datasheet);
    t.set(P::DrudeDamping, 1.0e14, Source::Datasheet);
    assert_eq!(
        Dispersion::from_table(&t),
        Err(OpticsError::MissingParameter("EpsilonInfinity"))
    );
    t.set(P::EpsilonInfinity, 1.54, Source::Datasheet);
    assert!(Dispersion::from_table(&t).unwrap().is_some());
}

#[test]
fn bk7_outside_its_range_is_an_error() {
    let bk7 = Sellmeier::bk7();
    assert!(matches!(
        bk7.refractive_index(9e-6),
        Err(OpticsError::OutOfRange { .. })
    ));
    assert!(bk7.refractive_index(0.1e-6).is_err());
    let n = bk7.refractive_index(0.5876e-6).unwrap();
    assert!((n - 1.5168).abs() < 1e-4, "{n}");
}

#[test]
fn every_model_rejects_out_of_range_and_non_finite_wavelengths() {
    let cauchy = Cauchy {
        a: 1.5,
        b: 0.004,
        c: 0.0001,
        range_m: (0.4e-6, 0.8e-6),
    };
    assert!(cauchy.refractive_index(0.2e-6).is_err());
    assert!(cauchy.refractive_index(f64::NAN).is_err());
    assert!(cauchy.refractive_index(0.5e-6).is_ok());
    assert!(DrudeLorentz::gold().nk(5e-6).is_err());
    assert!(gold().reflectance(2e-6).is_err());
    assert!(gold().reflectance_spectrum(300e-9, 830e-9, 10).is_err());
}

#[test]
fn a_pole_inside_the_range_is_an_error_not_nan() {
    let s = Sellmeier {
        b: vec![1.0],
        c: vec![0.25],
        range_m: (0.1e-6, 1e-6),
    };
    let r = s.refractive_index(0.5e-6);
    assert!(
        matches!(r, Err(OpticsError::NotFinite { .. })) || r.clone().is_ok_and(f64::is_finite),
        "{r:?}"
    );
}

#[test]
fn planck_is_zero_not_nan_for_tiny_lambda_t() {
    for (t, l) in [(300.0, 1e-9), (1.0, 1e-7), (10.0, 5e-9)] {
        let v = planck(t, l);
        assert!(v == 0.0, "T={t} l={l}: {v}");
    }
    assert!(planck(5800.0, 500e-9) > 0.0);
}
