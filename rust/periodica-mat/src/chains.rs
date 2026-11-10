//====== periodica/rust/periodica-mat/src/chains.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # chains
//!
//! Interface-relevant property chains: the fields a contact law needs
//! (`H`, `f_t`, `theta_q`) estimated from sheet fields that are commonly
//! present. Every step is [`Source::Estimated`], and every write goes through
//! [`PropertyTable::set_if_better`], so a datasheet or override value is never
//! replaced. `G_F` (from `K_IC^2/E` or `J_IC`) lives with Irwin's relation in
//! [`crate::derive`].
//!
//! A chain that finds no input writes nothing; the consumer then reports the
//! record and the missing fields. There is no default here.

use crate::property::PropertyId as P;
use crate::table::{PropertyTable, Source};

/// `Tabor: H ~ 3 sigma_y` (Tabor, *The Hardness of Metals*, 1951).
const TABOR_H_OVER_SIGMA_Y: f64 = 3.0;
/// Constraint factor on the compressive strength of a pressure-sensitive,
/// yield-free material: `H ~ 3 f_c` (Tabor's constraint factor applied to the
/// compressive yield of the Drucker-Prager cap).
const CONSTRAINT_H_OVER_FC: f64 = 3.0;
/// Vickers and Brinell numbers are kgf/mm^2: one kgf/mm^2 in Pa.
const KGF_MM2_PA: f64 = 9.80665e6;

/// Rockwell C to Vickers, ASTM E140 approximate conversion table
/// (`HRC`, `HV`), used by piecewise-linear interpolation.
const HRC_TO_HV: [(f64, f64); 10] = [
    (20.0, 238.0),
    (25.0, 266.0),
    (30.0, 302.0),
    (35.0, 345.0),
    (40.0, 392.0),
    (45.0, 446.0),
    (50.0, 513.0),
    (55.0, 595.0),
    (60.0, 697.0),
    (65.0, 832.0),
];

/// Antonetti, Whittle and Simons (1993) slope correlation for ground,
/// lapped and blasted metal surfaces: `m = 0.125 * sigma_um^0.402`.
const SLOPE_COEFF: f64 = 0.125;
const SLOPE_EXP: f64 = 0.402;
/// `sigma_rms = 1.25 R_a` for a Gaussian height distribution.
const RQ_OVER_RA: f64 = 1.25;

/// Run every chain once; returns the number of properties added.
pub(crate) fn interface_chains(t: &mut PropertyTable) -> usize {
    usize::from(hardness_pa(t)) + usize::from(tensile_f_t(t)) + usize::from(slope_from_ra(t))
}

/// Vickers number from Rockwell C by ASTM E140 interpolation; `None` outside
/// the table's 20..=65 range.
fn hrc_to_hv(hrc: f64) -> Option<f64> {
    let (lo, hi) = (HRC_TO_HV[0], HRC_TO_HV[HRC_TO_HV.len() - 1]);
    if !(lo.0..=hi.0).contains(&hrc) {
        return None;
    }
    HRC_TO_HV.windows(2).find(|w| hrc <= w[1].0).map(|w| {
        let f = (hrc - w[0].0) / (w[1].0 - w[0].0);
        w[0].1 + f * (w[1].1 - w[0].1)
    })
}

/// A hardness-scale value that was measured, not itself estimated from UTS.
fn measured(t: &PropertyTable, p: P) -> Option<f64> {
    let src = t.source(p);
    (src != Source::Estimated)
        .then(|| t.get(p))
        .flatten()
        .filter(|v| *v > 0.0)
}

/// Indentation hardness `H` in Pa: from a measured Vickers or Brinell number
/// (`H = HV * 9.80665e6 Pa`), from Rockwell C via ASTM E140, else from yield
/// by Tabor's `H ~ 3 sigma_y`, else (no yield, as for concrete) `H ~ 3 f_c`.
fn hardness_pa(t: &mut PropertyTable) -> bool {
    let from_scale = measured(t, P::HardnessVickers)
        .or_else(|| measured(t, P::HardnessBrinell))
        .or_else(|| measured(t, P::HardnessRockwellC).and_then(hrc_to_hv))
        .map(|hv| hv * KGF_MM2_PA);
    let from_yield = t
        .get(P::YieldStrength)
        .filter(|v| *v > 0.0)
        .map(|y| TABOR_H_OVER_SIGMA_Y * y);
    let from_fc = || measured_f_c(t).map(|fc| CONSTRAINT_H_OVER_FC * fc);
    match from_scale.or(from_yield).or_else(from_fc) {
        Some(h) => t.set_if_better(P::Hardness, h, Source::Estimated),
        None => false,
    }
}

/// Compressive strength for the `H ~ 3 f_c` chain: the mean value if the sheet
/// has one, else the characteristic value.
fn measured_f_c(t: &PropertyTable) -> Option<f64> {
    t.get(P::MeanCompressiveStrength)
        .or_else(|| t.get(P::CompressiveStrength))
        .filter(|v| *v > 0.0)
}

/// Tensile strength `f_t = UTS` where the sheet states no separate `f_t`.
fn tensile_f_t(t: &mut PropertyTable) -> bool {
    match t.get(P::UltimateTensileStrength).filter(|v| *v > 0.0) {
        Some(uts) => t.set_if_better(P::TensileStrength, uts, Source::Estimated),
        None => false,
    }
}

/// Rms asperity slope `theta_q` (rad) from `R_a`, using the Antonetti,
/// Whittle and Simons (1993) correlation `m = 0.125 (R_q[um])^0.402` with
/// `R_q = 1.25 R_a`; `theta_q ~ m` for small slopes.
fn slope_from_ra(t: &mut PropertyTable) -> bool {
    let Some(ra) = t.get(P::SurfaceRoughnessRa).filter(|v| *v > 0.0) else {
        return false;
    };
    let rq_um = RQ_OVER_RA * ra * 1.0e6;
    t.set_if_better(
        P::SurfaceSlopeRms,
        SLOPE_COEFF * rq_um.powf(SLOPE_EXP),
        Source::Estimated,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn table(pairs: &[(P, f64)]) -> PropertyTable {
        let mut t = PropertyTable::new();
        for (p, v) in pairs {
            t.set(*p, *v, Source::Datasheet);
        }
        t
    }

    #[test]
    fn hardness_from_yield_is_tabor() {
        let mut t = table(&[(P::YieldStrength, 250.0e6)]);
        interface_chains(&mut t);
        assert_relative_eq!(t.get(P::Hardness).unwrap(), 750.0e6, max_relative = 1e-12);
        assert_eq!(t.source(P::Hardness), Source::Estimated);
    }

    #[test]
    fn hardness_prefers_measured_vickers_over_yield() {
        let mut t = table(&[(P::HardnessVickers, 200.0), (P::YieldStrength, 250.0e6)]);
        interface_chains(&mut t);
        assert_relative_eq!(
            t.get(P::Hardness).unwrap(),
            200.0 * KGF_MM2_PA,
            max_relative = 1e-12
        );
    }

    #[test]
    fn hardness_from_rockwell_c_interpolates() {
        let mut t = table(&[(P::HardnessRockwellC, 42.5)]);
        interface_chains(&mut t);
        assert_relative_eq!(
            t.get(P::Hardness).unwrap(),
            419.0 * KGF_MM2_PA,
            max_relative = 1e-9
        );
        let mut out = table(&[(P::HardnessRockwellC, 10.0)]);
        interface_chains(&mut out);
        assert!(!out.has(P::Hardness));
    }

    #[test]
    fn hardness_without_yield_is_three_f_c() {
        let mut t = table(&[(P::CompressiveStrength, 30.0e6)]);
        interface_chains(&mut t);
        assert_relative_eq!(t.get(P::Hardness).unwrap(), 90.0e6, max_relative = 1e-12);
        assert_eq!(t.source(P::Hardness), Source::Estimated);
        let mut y = table(&[
            (P::CompressiveStrength, 30.0e6),
            (P::YieldStrength, 250.0e6),
        ]);
        interface_chains(&mut y);
        assert_relative_eq!(y.get(P::Hardness).unwrap(), 750.0e6, max_relative = 1e-12);
    }

    #[test]
    fn estimated_vickers_is_not_a_hardness_input() {
        let mut t = table(&[(P::UltimateTensileStrength, 400.0e6)]);
        crate::derive::fill_derived(&mut t);
        assert_eq!(t.source(P::HardnessVickers), Source::Estimated);
        assert!(!t.has(P::Hardness));
    }

    #[test]
    fn f_t_from_uts_is_estimated() {
        let mut t = table(&[(P::UltimateTensileStrength, 400.0e6)]);
        interface_chains(&mut t);
        assert_eq!(t.get(P::TensileStrength), Some(400.0e6));
        assert_eq!(t.source(P::TensileStrength), Source::Estimated);
    }

    #[test]
    fn stated_f_t_survives() {
        let mut t = table(&[
            (P::UltimateTensileStrength, 400.0e6),
            (P::TensileStrength, 350.0e6),
        ]);
        interface_chains(&mut t);
        assert_eq!(t.get(P::TensileStrength), Some(350.0e6));
        assert_eq!(t.source(P::TensileStrength), Source::Datasheet);
    }

    #[test]
    fn slope_from_ra_follows_the_correlation() {
        let mut t = table(&[(P::SurfaceRoughnessRa, 1.6e-6)]);
        interface_chains(&mut t);
        assert_relative_eq!(
            t.get(P::SurfaceSlopeRms).unwrap(),
            0.125 * 2.0_f64.powf(0.402),
            max_relative = 1e-12
        );
        assert_eq!(t.source(P::SurfaceSlopeRms), Source::Estimated);
    }

    #[test]
    fn gf_from_kic_squared_over_e_is_estimated() {
        let mut t = table(&[
            (P::FractureToughnessKic, 50.0e6),
            (P::YoungsModulus, 200.0e9),
        ]);
        crate::derive::fill_derived(&mut t);
        assert_relative_eq!(
            t.get(P::FractureEnergy).unwrap(),
            12500.0,
            max_relative = 1e-12
        );
        assert_eq!(t.source(P::FractureEnergy), Source::Estimated);
    }

    #[test]
    fn gf_from_jic_is_estimated_mirror() {
        let mut t = table(&[(P::Jic, 9000.0)]);
        crate::derive::fill_derived(&mut t);
        assert_eq!(t.get(P::FractureEnergy), Some(9000.0));
        assert_eq!(t.source(P::FractureEnergy), Source::Estimated);
    }
}
