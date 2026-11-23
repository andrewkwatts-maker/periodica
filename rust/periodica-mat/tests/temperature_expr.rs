//! MP-PER-2: temperature laws are Arithma expressions with validity ranges.
//! Coefficients below are test fixtures chosen for closed forms, not material data.

use periodica_mat::temperature::{Provenance, TemperatureError, ThermalLaw};

fn cited() -> Provenance {
    Provenance::Cited("fixture".into())
}

#[test]
fn linear_modulus_follows_t() {
    let l = ThermalLaw::linear(200e9, 293.0, -50e6, (200.0, 800.0), cited());
    assert!((l.at(293.0).unwrap() - 200e9).abs() < 1.0);
    assert!((l.at(493.0).unwrap() - 190e9).abs() < 1.0);
}

#[test]
fn polynomial_matches_closed_form() {
    let l = ThermalLaw::polynomial(&[1e-5, 2e-8, 1e-11], 300.0, (250.0, 900.0), cited());
    let d: f64 = 100.0;
    let want = 1e-5 + 2e-8 * d + 1e-11 * d * d;
    assert!((l.at(400.0).unwrap() - want).abs() < 1e-15);
}

#[test]
fn out_of_range_is_an_error_not_extrapolation() {
    let l = ThermalLaw::linear(1.0, 300.0, 0.1, (250.0, 350.0), cited());
    assert!(matches!(l.at(400.0), Err(TemperatureError::OutOfRange { .. })));
    assert!(l.at(f64::NAN).is_err());
}

#[test]
fn table_interpolates_and_hits_nodes() {
    let pts = [(300.0, 10.0), (400.0, 12.0), (600.0, 20.0)];
    let l = ThermalLaw::table(&pts, cited()).unwrap();
    for (t, y) in pts {
        assert!((l.at(t).unwrap() - y).abs() < 1e-9, "node {t}");
    }
    assert!((l.at(350.0).unwrap() - 11.0).abs() < 1e-9);
    assert!((l.at(500.0).unwrap() - 16.0).abs() < 1e-9);
}

#[test]
fn bad_tables_are_refused() {
    let one = ThermalLaw::table(&[(1.0, 1.0)], cited());
    assert_eq!(one.unwrap_err(), TemperatureError::BadTable);
    let dup = ThermalLaw::table(&[(1.0, 1.0), (1.0, 2.0)], cited());
    assert_eq!(dup.unwrap_err(), TemperatureError::BadTable);
}

#[test]
fn norton_scales_with_stress_power_and_arrhenius() {
    let l = ThermalLaw::norton(1e-30, 5.0, 3.0e5, (600.0, 900.0), cited());
    let lo = l.at_stress(700.0, 50e6).unwrap();
    let hi = l.at_stress(700.0, 100e6).unwrap();
    assert!((hi / lo - 32.0).abs() < 1e-9);
    let r = 6.022_140_76e23 * 1.380_649e-23;
    let want = 1e-30 * (50e6f64).powi(5) * (-3.0e5_f64 / (r * 700.0)).exp();
    assert!((lo / want - 1.0).abs() < 1e-9);
    assert!(l.at_stress(900.0, 50e6).unwrap() > lo);
}

#[test]
fn uncited_law_is_not_testable() {
    let l = ThermalLaw::linear(1.0, 300.0, 0.0, (0.0, 1.0e3), Provenance::NotTestable);
    assert!(!l.testable());
    assert!(ThermalLaw::linear(1.0, 300.0, 0.0, (0.0, 1.0e3), cited()).testable());
}
