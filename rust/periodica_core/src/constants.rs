//! Physical constants for game modules: G, c, h, k_B, sigma_SB and the flat
//! LCDM cosmology set (H0, Omega_m, Omega_Lambda).
//!
//! Values come from the checked-in datasheet
//! `src/periodica/data/config/physical_constants.json`, each with a unit, a
//! source and an uncertainty. Game modules read constants here, never from
//! metaphysica.

use std::collections::BTreeMap;

use arithma_core::ArithmaExpression;
use once_cell::sync::Lazy;
use serde::Deserialize;

const DATASHEET: &str = include_str!("../../../src/periodica/data/config/physical_constants.json");

/// One constant with its provenance.
#[derive(Debug, Clone, Deserialize)]
pub struct Constant {
    /// The value in `unit`.
    pub value: f64,
    /// One standard uncertainty in `unit` (0 for exact SI-defined values).
    pub uncertainty: f64,
    /// The unit as text.
    pub unit: String,
    /// Where the value comes from.
    pub source: String,
}

impl Constant {
    /// The value as an Arithma number.
    pub fn expr(&self) -> ArithmaExpression {
        ArithmaExpression::from_f64(self.value)
    }
}

/// A constant that is missing from the datasheet.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("periodica constants: no entry named `{0}`")]
pub struct MissingConstant(pub String);

static TABLE: Lazy<BTreeMap<String, Constant>> = Lazy::new(|| {
    serde_json::from_str(DATASHEET).expect("physical_constants.json is checked in and valid")
});

/// Parse a datasheet body; for tests and alternative sheets.
pub fn parse(json: &str) -> Result<BTreeMap<String, Constant>, serde_json::Error> {
    serde_json::from_str(json)
}

/// Look up a constant by name.
///
/// # Errors
/// [`MissingConstant`] naming the entry when it is absent.
pub fn get(name: &str) -> Result<&'static Constant, MissingConstant> {
    TABLE.get(name).ok_or_else(|| MissingConstant(name.to_owned()))
}

/// Look up a constant's value in its stored unit.
///
/// # Errors
/// [`MissingConstant`] when the entry is absent.
pub fn value(name: &str) -> Result<f64, MissingConstant> {
    get(name).map(|c| c.value)
}

/// Look up a constant as an Arithma number.
///
/// # Errors
/// [`MissingConstant`] when the entry is absent.
pub fn expr(name: &str) -> Result<ArithmaExpression, MissingConstant> {
    get(name).map(Constant::expr)
}

/// Resolve a name that the checked-in sheet is known to hold.
fn known(name: &str) -> &'static Constant {
    get(name).unwrap_or_else(|e| panic!("{e}"))
}

/// Newton's G, m^3 kg^-1 s^-2.
pub fn g() -> f64 { known("G").value }
/// Speed of light, m/s.
pub fn c() -> f64 { known("c").value }
/// Planck constant, J s.
pub fn h() -> f64 { known("h").value }
/// Boltzmann constant, J/K.
pub fn k_b() -> f64 { known("k_B").value }
/// Stefan-Boltzmann constant, W m^-2 K^-4.
pub fn sigma_sb() -> f64 { known("sigma_SB").value }
/// Hubble constant, km/s/Mpc.
pub fn h0() -> f64 { known("H0").value }
/// Matter density parameter.
pub fn omega_m() -> f64 { known("Omega_m").value }
/// Dark-energy density parameter.
pub fn omega_lambda() -> f64 { known("Omega_Lambda").value }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_match_codata_2018_and_planck_2018() {
        assert_eq!(g(), 6.674_30e-11);
        assert_eq!(c(), 299_792_458.0);
        assert_eq!(h(), 6.626_070_15e-34);
        assert_eq!(k_b(), 1.380_649e-23);
        assert_eq!(sigma_sb(), 5.670_374_419e-8);
        assert_eq!(h0(), 70.0);
        assert_eq!(omega_m(), 0.3153);
        assert_eq!(omega_lambda(), 0.6847);
    }

    #[test]
    fn every_entry_has_source_and_unit() {
        for (name, c) in TABLE.iter() {
            assert!(!c.source.is_empty() && !c.unit.is_empty(), "{name}");
        }
    }

    #[test]
    fn a_missing_entry_is_an_error_naming_it() {
        let err = get("nope").unwrap_err();
        assert_eq!(err, MissingConstant("nope".into()));
        assert!(err.to_string().contains("nope"));
        assert!(parse("{\"G\": 1}").is_err());
    }

    #[test]
    fn expr_evaluates_to_the_value() {
        assert_eq!(expr("G").unwrap().to_f64(), Some(6.674_30e-11));
    }
}
