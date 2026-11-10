//====== periodica/rust/periodica-mat/src/concrete.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # concrete
//!
//! fib Model Code 2010 (MC2010) derivations for records flagged
//! `"Class": "Concrete"`. `CompressiveStrength` is taken as the characteristic
//! cylinder strength `f_ck`; every fill is [`Source::Estimated`], so a
//! datasheet value always wins through [`PropertyTable::set_if_better`].

use crate::property::PropertyId as P;
use crate::table::{PropertyTable, Source};

/// `f_cm = f_ck + 8 MPa` (MC2010 Eq. 5.1-1).
const DELTA_F_MPA: f64 = 8.0;
const MPA: f64 = 1.0e6;
/// `E_c0 = 21.5 GPa` (MC2010 Eq. 5.1-21, quartzite aggregate).
const EC0_PA: f64 = 21.5e9;
/// Above this `f_ck` MC2010 Eq. 5.1-3b replaces Eq. 5.1-3a.
const FCK_NORMAL_MAX_MPA: f64 = 50.0;

/// Grades of MC2010 Table 5.1-8 (`f_ck`, MPa), as implemented in the fib
/// `structuralcodes` library (`mc2010/_concrete_material_properties.py`).
const GRADE_MPA: [f64; 17] = [
    12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    110.0, 120.0,
];
/// `|eps_c1|` per grade, in per mille (Table 5.1-8).
const EPS_C1_PERMILLE: [f64; 17] = [
    1.9, 2.0, 2.1, 2.2, 2.3, 2.3, 2.4, 2.5, 2.6, 2.6, 2.7, 2.7, 2.8, 2.9, 3.0, 3.0, 3.0,
];
/// `|eps_c,lim|` per grade, in per mille (Table 5.1-8).
const EPS_CLIM_PERMILLE: [f64; 17] = [
    3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.4, 3.4, 3.3, 3.2, 3.1, 3.0, 3.0, 3.0, 3.0,
];

/// Mean tensile strength in MPa, MC2010 Eqs. 5.1-3a and 5.1-3b.
pub fn fctm_mpa(fck: f64) -> f64 {
    if fck <= FCK_NORMAL_MAX_MPA {
        0.3 * fck.powf(2.0 / 3.0)
    } else {
        2.12 * (1.0 + (fck + DELTA_F_MPA) / 10.0).ln()
    }
}

/// Tensile fracture energy in N/m, MC2010 Eq. 5.1-9.
pub fn gf_n_per_m(fcm: f64) -> f64 {
    73.0 * fcm.powf(0.18)
}

/// Tangent modulus in Pa, MC2010 Eq. 5.1-21.
pub fn eci_pa(fcm: f64) -> f64 {
    EC0_PA * (fcm / 10.0).cbrt()
}

/// Linear interpolation in a Table 5.1-8 column; `None` outside 12..=120 MPa.
fn interp(fck: f64, column: &[f64; 17]) -> Option<f64> {
    if !(GRADE_MPA[0]..=GRADE_MPA[16]).contains(&fck) {
        return None;
    }
    let hi = GRADE_MPA.iter().position(|g| *g >= fck)?;
    if hi == 0 {
        return Some(column[0]);
    }
    let (g0, g1) = (GRADE_MPA[hi - 1], GRADE_MPA[hi]);
    let t = (fck - g0) / (g1 - g0);
    Some(column[hi - 1] + t * (column[hi] - column[hi - 1]))
}

/// Fill the MC2010 concrete properties from `CompressiveStrength` (`f_ck`).
///
/// Returns the number of properties added. Absent `f_ck` fills nothing.
pub fn fill_concrete(t: &mut PropertyTable) -> usize {
    let Some(fck_pa) = t.get(P::CompressiveStrength).filter(|v| *v > 0.0) else {
        return 0;
    };
    let fck = fck_pa / MPA;
    let fcm = fck + DELTA_F_MPA;
    let mut fills = vec![
        (P::MeanCompressiveStrength, fcm * MPA),
        (P::YoungsModulus, eci_pa(fcm)),
        (P::UltimateTensileStrength, fctm_mpa(fck) * MPA),
        (P::FractureEnergy, gf_n_per_m(fcm)),
    ];
    if let Some(e) = interp(fck, &EPS_C1_PERMILLE) {
        fills.push((P::PeakCompressiveStrain, e * 1.0e-3));
    }
    if let Some(e) = interp(fck, &EPS_CLIM_PERMILLE) {
        fills.push((P::UltimateCompressiveStrain, e * 1.0e-3));
    }
    fills
        .into_iter()
        .filter(|(p, v)| t.set_if_better(*p, *v, Source::Estimated))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn c30() -> PropertyTable {
        let mut t = PropertyTable::new();
        t.set(P::CompressiveStrength, 30.0e6, Source::Datasheet);
        t
    }

    #[test]
    fn fctm_c30() {
        assert_relative_eq!(fctm_mpa(30.0), 2.90, max_relative = 0.01);
    }

    #[test]
    fn gf_fcm38() {
        assert_relative_eq!(gf_n_per_m(38.0), 140.5, max_relative = 0.01);
    }

    #[test]
    fn eci_fcm38() {
        assert_relative_eq!(eci_pa(38.0), 33.6e9, max_relative = 0.01);
    }

    #[test]
    fn fctm_switches_above_c50() {
        assert_relative_eq!(
            fctm_mpa(60.0),
            2.12 * (1.0f64 + 6.8).ln(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn strains_interpolate_between_grades() {
        let mut t = PropertyTable::new();
        t.set(P::CompressiveStrength, 27.5e6, Source::Datasheet);
        fill_concrete(&mut t);
        assert_relative_eq!(
            t.get(P::PeakCompressiveStrain).unwrap(),
            2.25e-3,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            t.get(P::UltimateCompressiveStrain).unwrap(),
            3.5e-3,
            epsilon = 1e-12
        );
    }

    #[test]
    fn datasheet_beats_derived() {
        let mut t = c30();
        t.set(P::YoungsModulus, 30.0e9, Source::Datasheet);
        fill_concrete(&mut t);
        assert_eq!(t.get(P::YoungsModulus), Some(30.0e9));
        assert_eq!(t.source(P::YoungsModulus), Source::Datasheet);
        assert_eq!(t.source(P::FractureEnergy), Source::Estimated);
    }

    #[test]
    fn no_strength_fills_nothing() {
        assert_eq!(fill_concrete(&mut PropertyTable::new()), 0);
    }
}
