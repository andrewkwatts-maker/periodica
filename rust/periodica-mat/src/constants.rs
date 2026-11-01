//====== periodica/rust/periodica-mat/src/constants.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # constants
//!
//! The single source of physical constants for the periodica stack: the
//! CODATA 2022 recommended values.
//!
//! > P. J. Mohr, D. B. Newell, B. N. Taylor and E. Tiesinga, "CODATA
//! > recommended values of the fundamental physical constants: 2022",
//! > Rev. Mod. Phys. **97**, 025002 (2025); J. Phys. Chem. Ref. Data **54**,
//! > 033105 (2025). Values transcribed from
//! > <https://physics.nist.gov/cuu/Constants/Table/allascii.txt>.
//!
//! Every value here is mirrored, with its unit and standard uncertainty, in
//! `data/constants/codata2022.json` for non-Rust consumers. The test
//! `json_mirror_matches_rust_constants` asserts the two agree entry by entry,
//! so neither copy can drift.
//!
//! ## Exact and measured values
//!
//! Since the 2019 SI redefinition `c`, `h`, `e`, `N_A` and `k_B` are exact by
//! definition (standard uncertainty 0). `hbar = h / 2pi` is therefore exact as
//! well; it is computed here rather than transcribed, because NIST prints it
//! truncated (`1.054 571 817...`). Everything else is a measured value with the
//! standard uncertainty given in its doc comment.
//!
//! Where CODATA publishes a quantity directly (for example `m_p / m_e`, or the
//! energy equivalent of `u` in MeV) the published value is used rather than a
//! ratio of other constants: the direct value carries a smaller uncertainty
//! than the ratio would.

/// Speed of light in vacuum, `c` = 299 792 458 m s^-1 (exact).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Planck constant, `h` = 6.626 070 15e-34 J Hz^-1 (exact).
pub const PLANCK: f64 = 6.626_070_15e-34;

/// Reduced Planck constant, `hbar = h / 2pi` in J s (exact; derived from [`PLANCK`]).
pub const HBAR: f64 = PLANCK / (2.0 * core::f64::consts::PI);

/// Elementary charge, `e` = 1.602 176 634e-19 C (exact).
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

/// One electron volt in joules, numerically equal to [`ELEMENTARY_CHARGE`] (exact).
pub const ELECTRON_VOLT: f64 = ELEMENTARY_CHARGE;

/// Electron mass, `m_e` = 9.109 383 7139(28)e-31 kg.
pub const ELECTRON_MASS: f64 = 9.109_383_713_9e-31;

/// Proton mass, `m_p` = 1.672 621 925 95(52)e-27 kg.
pub const PROTON_MASS: f64 = 1.672_621_925_95e-27;

/// Neutron mass, `m_n` = 1.674 927 500 56(85)e-27 kg.
pub const NEUTRON_MASS: f64 = 1.674_927_500_56e-27;

/// Atomic mass constant, `m_u = 1 u` = 1.660 539 068 92(52)e-27 kg.
pub const ATOMIC_MASS_CONSTANT: f64 = 1.660_539_068_92e-27;

/// Kilograms per unified atomic mass unit (the same quantity as
/// [`ATOMIC_MASS_CONSTANT`], named for conversions).
pub const KG_PER_U: f64 = ATOMIC_MASS_CONSTANT;

/// Energy equivalent of 1 u, 931.494 103 72(29) MeV.
pub const MEV_PER_U: f64 = 931.494_103_72;

/// Electron mass in u, 5.485 799 090 441(97)e-4 u.
pub const ELECTRON_MASS_U: f64 = 5.485_799_090_441e-4;

/// Proton-electron mass ratio, `m_p / m_e` = 1836.152 673 426(32).
pub const PROTON_ELECTRON_MASS_RATIO: f64 = 1_836.152_673_426;

/// Bohr radius, `a0` = 5.291 772 105 44(82)e-11 m.
pub const BOHR_RADIUS: f64 = 5.291_772_105_44e-11;

/// Hartree energy, `E_h` = 4.359 744 722 2060(48)e-18 J.
pub const HARTREE_ENERGY: f64 = 4.359_744_722_206_0e-18;

/// Hartree energy in electron volts, 27.211 386 245 981(30) eV.
pub const HARTREE_ENERGY_EV: f64 = 27.211_386_245_981;

/// Atomic unit of time, `hbar / E_h` = 2.418 884 326 5864(26)e-17 s.
pub const ATOMIC_UNIT_OF_TIME: f64 = 2.418_884_326_586_4e-17;

/// Fine-structure constant, `alpha` = 7.297 352 5643(11)e-3 (dimensionless).
pub const FINE_STRUCTURE: f64 = 7.297_352_564_3e-3;

/// Rydberg constant, `R_inf` = 10 973 731.568 157(12) m^-1.
pub const RYDBERG: f64 = 10_973_731.568_157;

/// Avogadro constant, `N_A` = 6.022 140 76e23 mol^-1 (exact).
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// Boltzmann constant, `k_B` = 1.380 649e-23 J K^-1 (exact).
pub const BOLTZMANN: f64 = 1.380_649e-23;

/// One CODATA entry with its metadata, for tables, labels and the JSON mirror.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant {
    /// Stable key, identical to the key in `codata2022.json`.
    pub key: &'static str,
    /// CODATA name, as printed in the NIST table.
    pub name: &'static str,
    /// Recommended value in `unit`.
    pub value: f64,
    /// Standard uncertainty in `unit`; 0 for exact constants.
    pub standard_uncertainty: f64,
    /// Unit in NIST notation (`"J Hz^-1"`, `"m^-1"`); empty when dimensionless.
    pub unit: &'static str,
}

impl Constant {
    /// Whether the value is exact by definition of the SI.
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.standard_uncertainty == 0.0
    }

    /// Relative standard uncertainty (0 for exact constants).
    #[inline]
    pub fn relative_uncertainty(&self) -> f64 {
        self.standard_uncertainty / self.value.abs()
    }
}

/// Citation for every value in this module.
pub const CODATA_2022_CITATION: &str = "P. J. Mohr, D. B. Newell, B. N. Taylor and E. Tiesinga, \
CODATA recommended values of the fundamental physical constants: 2022, \
Rev. Mod. Phys. 97, 025002 (2025)";

/// Every constant in this module, in a fixed order.
pub const CODATA_2022: &[Constant] = &[
    entry(
        "c",
        "speed of light in vacuum",
        SPEED_OF_LIGHT,
        0.0,
        "m s^-1",
    ),
    entry("h", "Planck constant", PLANCK, 0.0, "J Hz^-1"),
    entry("hbar", "reduced Planck constant", HBAR, 0.0, "J s"),
    entry("e", "elementary charge", ELEMENTARY_CHARGE, 0.0, "C"),
    entry("eV", "electron volt", ELECTRON_VOLT, 0.0, "J"),
    entry(
        "m_e",
        "electron mass",
        ELECTRON_MASS,
        0.000_000_002_8e-31,
        "kg",
    ),
    entry(
        "m_p",
        "proton mass",
        PROTON_MASS,
        0.000_000_000_52e-27,
        "kg",
    ),
    entry(
        "m_n",
        "neutron mass",
        NEUTRON_MASS,
        0.000_000_000_85e-27,
        "kg",
    ),
    entry(
        "u",
        "atomic mass constant",
        ATOMIC_MASS_CONSTANT,
        0.000_000_000_52e-27,
        "kg",
    ),
    entry(
        "MeV_per_u",
        "atomic mass constant energy equivalent in MeV",
        MEV_PER_U,
        0.000_000_29,
        "MeV",
    ),
    entry(
        "m_e_u",
        "electron mass in u",
        ELECTRON_MASS_U,
        0.000_000_000_097e-4,
        "u",
    ),
    entry(
        "m_p_over_m_e",
        "proton-electron mass ratio",
        PROTON_ELECTRON_MASS_RATIO,
        0.000_000_032,
        "",
    ),
    entry("a0", "Bohr radius", BOHR_RADIUS, 0.000_000_000_82e-11, "m"),
    entry(
        "E_h",
        "Hartree energy",
        HARTREE_ENERGY,
        0.000_000_000_004_8e-18,
        "J",
    ),
    entry(
        "E_h_eV",
        "Hartree energy in eV",
        HARTREE_ENERGY_EV,
        0.000_000_000_030,
        "eV",
    ),
    entry(
        "t_au",
        "atomic unit of time",
        ATOMIC_UNIT_OF_TIME,
        0.000_000_000_002_6e-17,
        "s",
    ),
    entry(
        "alpha",
        "fine-structure constant",
        FINE_STRUCTURE,
        0.000_000_001_1e-3,
        "",
    ),
    entry("R_inf", "Rydberg constant", RYDBERG, 0.000_012, "m^-1"),
    entry("N_A", "Avogadro constant", AVOGADRO, 0.0, "mol^-1"),
    entry("k_B", "Boltzmann constant", BOLTZMANN, 0.0, "J K^-1"),
];

const fn entry(
    key: &'static str,
    name: &'static str,
    value: f64,
    standard_uncertainty: f64,
    unit: &'static str,
) -> Constant {
    Constant {
        key,
        name,
        value,
        standard_uncertainty,
        unit,
    }
}

/// Look a constant up by its key (`"a0"`, `"E_h"`, ...).
pub fn by_key(key: &str) -> Option<&'static Constant> {
    CODATA_2022.iter().find(|k| k.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const JSON: &str = include_str!("../data/constants/codata2022.json");

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    /// Distance in units in the last place between two finite, same-sign values.
    fn ulps_apart(a: f64, b: f64) -> u64 {
        assert!(a.is_finite() && b.is_finite() && (a >= 0.0) == (b >= 0.0));
        a.abs().to_bits().abs_diff(b.abs().to_bits())
    }

    #[test]
    fn json_mirror_matches_rust_constants() {
        let doc: Value = serde_json::from_str(JSON).expect("codata2022.json parses");
        let table = doc["constants"]
            .as_object()
            .expect("`constants` is an object");
        assert_eq!(
            table.len(),
            CODATA_2022.len(),
            "JSON and Rust list different constants"
        );
        for k in CODATA_2022 {
            let entry = &table
                .get(k.key)
                .unwrap_or_else(|| panic!("{} missing from JSON", k.key));
            let value = entry["value"].as_f64().unwrap();
            let unc = entry["standard_uncertainty"].as_f64().unwrap();
            // Equal to the last bit, up to serde_json's parser: without its
            // `float_roundtrip` feature it may land one ULP off on a
            // 17-significant-digit literal (it does on `hbar`). No transcription
            // error can hide inside one ULP (~1e-16 relative).
            assert!(
                ulps_apart(value, k.value) <= 1,
                "{}: value {value} vs {}",
                k.key,
                k.value
            );
            assert!(
                ulps_apart(unc, k.standard_uncertainty) <= 1,
                "{}: uncertainty",
                k.key
            );
            assert_eq!(entry["unit"].as_str().unwrap(), k.unit, "{}: unit", k.key);
            assert_eq!(entry["name"].as_str().unwrap(), k.name, "{}: name", k.key);
            assert_eq!(
                entry["exact"].as_bool().unwrap(),
                k.is_exact(),
                "{}: exact flag",
                k.key
            );
        }
    }

    #[test]
    fn keys_are_unique_and_lookup_works() {
        for (i, a) in CODATA_2022.iter().enumerate() {
            for b in &CODATA_2022[i + 1..] {
                assert_ne!(a.key, b.key);
            }
            assert_eq!(by_key(a.key), Some(a));
        }
        assert!(by_key("not-a-constant").is_none());
        assert!(by_key("c").unwrap().is_exact());
        assert!(!by_key("a0").unwrap().is_exact());
    }

    /// Transcription check: CODATA values satisfy the defining relations to
    /// within their stated uncertainties. A typo in any digit fails this.
    #[test]
    fn transcribed_values_satisfy_defining_relations() {
        let pi = core::f64::consts::PI;
        // a0 = hbar / (m_e c alpha)            (u_r 1.6e-10)
        let a0 = HBAR / (ELECTRON_MASS * SPEED_OF_LIGHT * FINE_STRUCTURE);
        assert!(rel(a0, BOHR_RADIUS) < 1e-9, "a0 {a0}");
        // E_h = alpha^2 m_e c^2                (u_r 1.1e-12, via m_e 3e-10)
        let eh = FINE_STRUCTURE * FINE_STRUCTURE * ELECTRON_MASS * SPEED_OF_LIGHT.powi(2);
        assert!(rel(eh, HARTREE_ENERGY) < 1e-9, "E_h {eh}");
        assert!(rel(HARTREE_ENERGY / ELECTRON_VOLT, HARTREE_ENERGY_EV) < 1e-12);
        // R_inf = alpha^2 m_e c / (2 h)
        let r = FINE_STRUCTURE * FINE_STRUCTURE * ELECTRON_MASS * SPEED_OF_LIGHT / (2.0 * PLANCK);
        assert!(rel(r, RYDBERG) < 1e-9, "R_inf {r}");
        // t_au = hbar / E_h
        assert!(rel(HBAR / HARTREE_ENERGY, ATOMIC_UNIT_OF_TIME) < 1e-11);
        // u c^2 in MeV
        let mev = ATOMIC_MASS_CONSTANT * SPEED_OF_LIGHT.powi(2) / (ELEMENTARY_CHARGE * 1e6);
        assert!(rel(mev, MEV_PER_U) < 1e-9, "MeV/u {mev}");
        // Mass ratios and masses in u.
        assert!(rel(PROTON_MASS / ELECTRON_MASS, PROTON_ELECTRON_MASS_RATIO) < 1e-9);
        assert!(rel(ELECTRON_MASS / ATOMIC_MASS_CONSTANT, ELECTRON_MASS_U) < 1e-9);
        // hbar is h / 2pi to the last bit.
        assert_eq!(HBAR, PLANCK / (2.0 * pi));
        assert_eq!(KG_PER_U, ATOMIC_MASS_CONSTANT);
        // 1/alpha as printed by NIST: 137.035 999 177(21).
        assert!((1.0 / FINE_STRUCTURE - 137.035_999_177).abs() < 3e-8);
    }

    #[test]
    fn relative_uncertainties_are_as_published() {
        let a0 = by_key("a0").unwrap();
        assert!((a0.relative_uncertainty() / 1.6e-10 - 1.0).abs() < 0.05);
        assert_eq!(by_key("h").unwrap().relative_uncertainty(), 0.0);
    }
}
