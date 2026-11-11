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

use crate::derive::{KGF_MM2_PA, TABOR_CONSTRAINT_FACTOR};

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

/// Coefficient of the mean-absolute-slope correlation `m = 0.125 (sigma * 1e6)^0.402`,
/// `sigma` the rms roughness in metres. Source as read: P. Teertstra, "Calculating
/// interface resistance", Electronics Cooling, 1 May 1997
/// (electronics-cooling.com/1997/05/calculating-interface-resistance/), which credits
/// V. W. Antonetti, T. D. Whittle and R. E. Simons, "An approximate thermal contact
/// conductance correlation", ASME HTD-Vol. 170, 1991, pp. 35-42. `m` is the mean absolute
/// asperity slope. The 1991 paper itself was not read; the page does not say which
/// surfaces the fit covers and gives no scatter.
const SLOPE_COEFF: f64 = 0.125;
/// Exponent of the correlation above.
const SLOPE_EXP: f64 = 0.402;
/// The roughness range of the correlation, `R_q` in micrometres: Teertstra states it was
/// "developed for the surface roughness range 0.216 um < sigma < 9.6 um".
const SLOPE_RQ_RANGE_UM: (f64, f64) = (0.216, 9.6);

/// `R_q / R_a = sqrt(pi / 2) = 1.2533` for a Gaussian height distribution.
///
/// With rms height `R_q` the mean absolute height of a zero-mean Gaussian is
/// `R_a = R_q sqrt(2 / pi)`, so `R_q = R_a sqrt(pi / 2)`. The same ratio converts a
/// Gaussian mean absolute slope to an rms slope. (`sqrt` is not `const`, hence a function.)
pub(crate) fn rq_over_ra() -> f64 {
    std::f64::consts::FRAC_PI_2.sqrt()
}

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
/// (`H = HV * 9.80665e6 Pa`), from Rockwell C via ASTM E140, else, for a ductile metal,
/// from yield by Tabor's `H ~ 3 sigma_y` ([`TABOR_CONSTRAINT_FACTOR`]). A sheet with
/// none of these gets no `Hardness`: for a cohesive-frictional solid the ratio of
/// hardness to cohesion depends on the friction angle, so no single `H / f_c` exists.
fn hardness_pa(t: &mut PropertyTable) -> bool {
    let from_scale = measured(t, P::HardnessVickers)
        .or_else(|| measured(t, P::HardnessBrinell))
        .or_else(|| measured(t, P::HardnessRockwellC).and_then(hrc_to_hv))
        .map(|hv| hv * KGF_MM2_PA);
    let from_yield = t
        .get(P::YieldStrength)
        .filter(|v| *v > 0.0 && crate::derive::shows_ductile_metal(t))
        .map(|y| TABOR_CONSTRAINT_FACTOR * y);
    match from_scale.or(from_yield) {
        Some(h) => t.set_if_better(P::Hardness, h, Source::Estimated),
        None => false,
    }
}

/// Tensile strength `f_t = UTS` where the sheet states no separate `f_t`.
fn tensile_f_t(t: &mut PropertyTable) -> bool {
    match t.get(P::UltimateTensileStrength).filter(|v| *v > 0.0) {
        Some(uts) => t.set_if_better(P::TensileStrength, uts, Source::Estimated),
        None => false,
    }
}

/// Rms asperity slope `theta_q` (rad) from `R_a`: `R_q = sqrt(pi / 2) R_a`, then the
/// mean absolute slope `m = 0.125 (R_q[um])^0.402` (see [`SLOPE_COEFF`]), then
/// `theta_q = sqrt(pi / 2) m`, the rms of Gaussian slopes with mean absolute value `m`.
/// Writes nothing when `R_q` is outside the correlation's range 0.216..=9.6 um.
fn slope_from_ra(t: &mut PropertyTable) -> bool {
    let Some(ra) = t.get(P::SurfaceRoughnessRa).filter(|v| *v > 0.0) else {
        return false;
    };
    let rq_um = rq_over_ra() * ra * 1.0e6;
    if !(SLOPE_RQ_RANGE_UM.0..=SLOPE_RQ_RANGE_UM.1).contains(&rq_um) {
        return false;
    }
    let mean_abs = SLOPE_COEFF * rq_um.powf(SLOPE_EXP);
    t.set_if_better(
        P::SurfaceSlopeRms,
        rq_over_ra() * mean_abs,
        Source::Estimated,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rq_over_ra_is_sqrt_half_pi() {
        assert_eq!(rq_over_ra(), (std::f64::consts::PI / 2.0).sqrt());
    }

    fn table(pairs: &[(P, f64)]) -> PropertyTable {
        let mut t = PropertyTable::new();
        for (p, v) in pairs {
            t.set(*p, *v, Source::Datasheet);
        }
        t
    }

    #[test]
    fn hardness_from_yield_is_tabor() {
        let mut t = table(&[(P::YieldStrength, 250.0e6), (P::ElongationPct, 20.0)]);
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
    fn hardness_without_yield_is_absent() {
        let mut t = table(&[(P::CompressiveStrength, 30.0e6)]);
        interface_chains(&mut t);
        assert!(!t.has(P::Hardness));
        let mut y = table(&[
            (P::CompressiveStrength, 30.0e6),
            (P::YieldStrength, 250.0e6),
            (P::ElongationPct, 20.0),
        ]);
        interface_chains(&mut y);
        assert_relative_eq!(y.get(P::Hardness).unwrap(), 750.0e6, max_relative = 1e-12);
    }

    #[test]
    fn estimated_vickers_is_not_a_hardness_input() {
        let mut t = table(&[
            (P::UltimateTensileStrength, 400.0e6),
            (P::ElongationPct, 20.0),
        ]);
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
        let rq_um = rq_over_ra() * 1.6;
        assert_relative_eq!(
            t.get(P::SurfaceSlopeRms).unwrap(),
            rq_over_ra() * 0.125 * rq_um.powf(0.402),
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
