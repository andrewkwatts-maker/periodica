//====== periodica/rust/periodica-mat/src/derive.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # derive
//!
//! Fills gaps in a [`PropertyTable`] from exact physical relations between
//! properties that *are* present.
//!
//! ## Why this is needed
//!
//! The corpus is uneven. Electrical **resistivity** appears in 137 datasheets;
//! electrical **conductivity** in zero -- yet a game engine asking "how well
//! does this conduct" should not have to know which way round the data happens
//! to be stored. Likewise Young's modulus (81 files) and Poisson's ratio (90)
//! are common while bulk modulus (14) is rare, even though the first two
//! determine the third exactly for an isotropic solid.
//!
//! ## Rules of engagement
//!
//! - Every write goes through [`PropertyTable::set_if_better`], so a derived
//!   value can never overwrite a datasheet or user-override value. Derivation
//!   is therefore safe to re-run at any time.
//! - Each fill is stamped [`Source::Derived`] (an exact identity) or
//!   [`Source::Estimated`] (an empirical correlation) so the app can badge it
//!   and a careful consumer can reject correlations.
//! - Relations are applied to a fixpoint, because they cascade: `E` and `nu`
//!   give `G`, and `G` then satisfies other relations in turn.

use crate::property::PropertyId as P;
use crate::table::{PropertyTable, Source};

/// Poisson's ratio outside this range is thermodynamically impossible for an
/// isotropic solid; treat such data as corrupt rather than dividing by it.
const NU_MIN: f64 = -0.999;
const NU_MAX: f64 = 0.4999;

/// Tabor's correlation: for work-hardening metals the Vickers hardness in
/// MPa is roughly three times the ultimate tensile strength.
const TABOR_FACTOR: f64 = 3.3;

/// Upper bound on fixpoint passes. Every rule is monotone (it only ever fills
/// absent slots), so this terminates well before the bound; the cap exists to
/// satisfy the bounded-loop rule in CLAUDE.md.
const MAX_PASSES: usize = 8;

/// Apply every derivation rule until no further property can be filled.
///
/// Returns the number of properties added.
pub fn fill_derived(t: &mut PropertyTable) -> usize {
    let mut total = 0;
    for _ in 0..MAX_PASSES {
        let added = one_pass(t);
        total += added;
        if added == 0 {
            break;
        }
    }
    total
}

fn one_pass(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    n += electrical(t);
    n += elastic(t);
    n += fracture(t);
    n += thermal(t);
    n += optical(t);
    n += hardness(t);
    n
}

/// Conductivity and resistivity are exact reciprocals.
fn electrical(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    let rho = t.get(P::ElectricalResistivity);
    let sigma = t.get(P::ElectricalConductivity);

    // A stated conductivity of zero is not a measurement -- it is what %IACS
    // rounds to for any insulator. `demo_silicon_carbide` quotes both
    // `PercentIACS: 0` and a real resistivity; taking the zero at face value
    // would report an insulator as having no resistivity information at all,
    // and would divide by zero on the way back. Treat it as absent.
    if matches!(sigma, Some(s) if s <= 0.0) {
        t.clear(P::ElectricalConductivity);
    }

    if let Some(rho) = rho {
        if rho > 0.0 && t.set_if_better(P::ElectricalConductivity, 1.0 / rho, Source::Derived) {
            n += 1;
        }
    }
    if let Some(sigma) = sigma {
        if sigma > 0.0 && t.set_if_better(P::ElectricalResistivity, 1.0 / sigma, Source::Derived) {
            n += 1;
        }
    }
    n
}

/// Isotropic-elasticity identities. Any two of `E`, `G`, `K`, `nu` determine
/// the rest.
fn elastic(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    let e = t.get(P::YoungsModulus);
    let g = t.get(P::ShearModulus);
    let k = t.get(P::BulkModulus);
    let nu = t
        .get(P::PoissonsRatio)
        .filter(|v| (NU_MIN..=NU_MAX).contains(v));

    if let (Some(e), Some(nu)) = (e, nu) {
        if t.set_if_better(P::ShearModulus, e / (2.0 * (1.0 + nu)), Source::Derived) {
            n += 1;
        }
        if t.set_if_better(
            P::BulkModulus,
            e / (3.0 * (1.0 - 2.0 * nu)),
            Source::Derived,
        ) {
            n += 1;
        }
    }
    if let (Some(g), Some(nu)) = (g, nu) {
        if t.set_if_better(P::YoungsModulus, 2.0 * g * (1.0 + nu), Source::Derived) {
            n += 1;
        }
    }
    if let (Some(k), Some(nu)) = (k, nu) {
        if t.set_if_better(
            P::YoungsModulus,
            3.0 * k * (1.0 - 2.0 * nu),
            Source::Derived,
        ) {
            n += 1;
        }
    }
    if let (Some(e), Some(g)) = (e, g) {
        if g > 0.0 {
            let derived_nu = e / (2.0 * g) - 1.0;
            if (NU_MIN..=NU_MAX).contains(&derived_nu)
                && t.set_if_better(P::PoissonsRatio, derived_nu, Source::Derived)
            {
                n += 1;
            }
        }
    }
    if let (Some(e), Some(k)) = (e, k) {
        if k > 0.0 {
            let derived_nu = (1.0 - e / (3.0 * k)) * 0.5;
            if (NU_MIN..=NU_MAX).contains(&derived_nu)
                && t.set_if_better(P::PoissonsRatio, derived_nu, Source::Derived)
            {
                n += 1;
            }
        }
    }
    if let (Some(g), Some(k)) = (g, k) {
        let denom = 3.0 * k + g;
        if denom > 0.0 && t.set_if_better(P::YoungsModulus, 9.0 * k * g / denom, Source::Derived) {
            n += 1;
        }
    }
    n
}

/// Irwin's relation between the stress-intensity and energy formulations of
/// toughness. Plane stress: `G_IC = K_IC^2 / E`.
fn fracture(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    let e = t.get(P::YoungsModulus).filter(|v| *v > 0.0);
    if let (Some(kic), Some(e)) = (t.get(P::FractureToughnessKic), e) {
        if t.set_if_better(P::Jic, kic * kic / e, Source::Derived) {
            n += 1;
        }
    }
    if let (Some(jic), Some(e)) = (t.get(P::Jic), e) {
        if jic > 0.0 && t.set_if_better(P::FractureToughnessKic, (jic * e).sqrt(), Source::Derived)
        {
            n += 1;
        }
    }
    // J_IC and a measured fracture energy are the same quantity in the same
    // units; mirror whichever one is present.
    if let Some(jic) = t.get(P::Jic) {
        if t.set_if_better(P::FractureEnergy, jic, Source::Derived) {
            n += 1;
        }
    }
    if let Some(gf) = t.get(P::FractureEnergy) {
        if t.set_if_better(P::Jic, gf, Source::Derived) {
            n += 1;
        }
    }
    n
}

/// `alpha = k / (rho * c_p)`.
fn thermal(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    let (k, rho, cp) = (
        t.get(P::ThermalConductivity),
        t.get(P::Density),
        t.get(P::SpecificHeat),
    );
    if let (Some(k), Some(rho), Some(cp)) = (k, rho, cp) {
        let denom = rho * cp;
        if denom > 0.0 && t.set_if_better(P::ThermalDiffusivity, k / denom, Source::Derived) {
            n += 1;
        }
    }
    n
}

/// Energy conservation across a surface, plus Kirchhoff's law.
fn optical(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    let r = t.get(P::Reflectance);
    let tr = t.get(P::Transmittance);
    if let (Some(r), Some(tr)) = (r, tr) {
        let a = 1.0 - r - tr;
        if (0.0..=1.0).contains(&a) && t.set_if_better(P::Absorptance, a, Source::Derived) {
            n += 1;
        }
    }
    // Kirchhoff: at thermal equilibrium emissivity equals absorptance.
    if let Some(a) = t.get(P::Absorptance) {
        if t.set_if_better(P::Emissivity, a, Source::Derived) {
            n += 1;
        }
    }
    if let Some(e) = t.get(P::Emissivity) {
        if t.set_if_better(P::Absorptance, e, Source::Derived) {
            n += 1;
        }
    }
    // Normal-incidence Fresnel reflectance from the refractive index.
    if let Some(nidx) = t.get(P::RefractiveIndexN) {
        if nidx >= 1.0 {
            let r0 = ((nidx - 1.0) / (nidx + 1.0)).powi(2);
            if t.set_if_better(P::Reflectance, r0, Source::Estimated) {
                n += 1;
            }
        }
    }
    n
}

/// Tabor's hardness/strength correlation. Empirical, so `Estimated`.
fn hardness(t: &mut PropertyTable) -> usize {
    let mut n = 0;
    if let Some(uts) = t.get(P::UltimateTensileStrength) {
        // HV is conventionally a kgf/mm^2 number; UTS[MPa] ~ 3.3 * HV.
        let hv = uts / 1.0e6 / TABOR_FACTOR;
        if hv > 0.0 && t.set_if_better(P::HardnessVickers, hv, Source::Estimated) {
            n += 1;
        }
    }
    if let Some(hv) = t.get(P::HardnessVickers) {
        let uts = hv * TABOR_FACTOR * 1.0e6;
        if t.set_if_better(P::UltimateTensileStrength, uts, Source::Estimated) {
            n += 1;
        }
    }
    n
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
    fn conductivity_is_the_reciprocal_of_resistivity() {
        // Copper: 1.68e-8 Ohm*m -> 5.95e7 S/m.
        let mut t = table(&[(P::ElectricalResistivity, 1.68e-8)]);
        fill_derived(&mut t);
        assert_relative_eq!(
            t.get(P::ElectricalConductivity).unwrap(),
            5.952e7,
            max_relative = 1e-3
        );
        assert_eq!(t.source(P::ElectricalConductivity), Source::Derived);
    }

    #[test]
    fn elastic_identities_match_a_real_datasheet() {
        // Aluminium 6061-T6 quotes E=68.9 GPa, nu=0.33, G=26 GPa, K=67.6 GPa.
        // Derive G and K from E and nu and check against the published values.
        let mut t = table(&[(P::YoungsModulus, 68.9e9), (P::PoissonsRatio, 0.33)]);
        fill_derived(&mut t);
        assert_relative_eq!(t.get(P::ShearModulus).unwrap(), 25.9e9, max_relative = 0.01);
        assert_relative_eq!(t.get(P::BulkModulus).unwrap(), 67.5e9, max_relative = 0.01);
    }

    #[test]
    fn elastic_derivation_is_self_consistent_both_ways() {
        let mut t = table(&[(P::ShearModulus, 79.3e9), (P::PoissonsRatio, 0.29)]);
        fill_derived(&mut t);
        let e = t.get(P::YoungsModulus).unwrap();
        // Steel-1018 publishes 200 GPa.
        assert_relative_eq!(e, 2.046e11, max_relative = 0.01);

        // Feeding E and G back must recover the same nu.
        let mut t2 = table(&[(P::YoungsModulus, e), (P::ShearModulus, 79.3e9)]);
        fill_derived(&mut t2);
        assert_relative_eq!(t2.get(P::PoissonsRatio).unwrap(), 0.29, max_relative = 1e-6);
    }

    #[test]
    fn irwin_relation_round_trips() {
        // K_IC = 29 MPa*sqrt(m), E = 68.9 GPa  ->  J_IC = K^2/E.
        let mut t = table(&[
            (P::FractureToughnessKic, 29.0e6),
            (P::YoungsModulus, 68.9e9),
        ]);
        fill_derived(&mut t);
        let jic = t.get(P::Jic).unwrap();
        assert_relative_eq!(jic, 29.0e6_f64.powi(2) / 68.9e9, max_relative = 1e-9);

        let mut back = table(&[(P::Jic, jic), (P::YoungsModulus, 68.9e9)]);
        fill_derived(&mut back);
        assert_relative_eq!(
            back.get(P::FractureToughnessKic).unwrap(),
            29.0e6,
            max_relative = 1e-9
        );
    }

    #[test]
    fn thermal_diffusivity_from_k_rho_cp() {
        // Steel-1018: k=51.9, rho=7870, cp=486  ->  ~1.36e-5 m^2/s.
        let mut t = table(&[
            (P::ThermalConductivity, 51.9),
            (P::Density, 7870.0),
            (P::SpecificHeat, 486.0),
        ]);
        fill_derived(&mut t);
        assert_relative_eq!(
            t.get(P::ThermalDiffusivity).unwrap(),
            1.357e-5,
            max_relative = 1e-3
        );
    }

    #[test]
    fn fresnel_reflectance_from_refractive_index() {
        // Soda-lime glass n=1.52 -> ~4.3 % reflectance at normal incidence.
        let mut t = table(&[(P::RefractiveIndexN, 1.52)]);
        fill_derived(&mut t);
        assert_relative_eq!(t.get(P::Reflectance).unwrap(), 0.0426, max_relative = 0.01);
        assert_eq!(t.source(P::Reflectance), Source::Estimated);
    }

    #[test]
    fn optical_energy_balance_closes() {
        let mut t = table(&[(P::Reflectance, 0.08), (P::Transmittance, 0.90)]);
        fill_derived(&mut t);
        assert_relative_eq!(t.get(P::Absorptance).unwrap(), 0.02, max_relative = 1e-9);
        // Kirchhoff carries it into emissivity.
        assert_relative_eq!(t.get(P::Emissivity).unwrap(), 0.02, max_relative = 1e-9);
    }

    #[test]
    fn derivation_never_overwrites_datasheet_values() {
        // Al 6061-T6 publishes K=67.6 GPa; the identity would give ~67.5.
        // The published number must survive.
        let mut t = table(&[
            (P::YoungsModulus, 68.9e9),
            (P::PoissonsRatio, 0.33),
            (P::BulkModulus, 67.6e9),
        ]);
        fill_derived(&mut t);
        assert_eq!(t.get(P::BulkModulus), Some(67.6e9));
        assert_eq!(t.source(P::BulkModulus), Source::Datasheet);
    }

    #[test]
    fn unphysical_poisson_ratio_is_ignored() {
        // nu = 0.5 is incompressible: E/(3(1-2nu)) would divide by zero.
        let mut t = table(&[(P::YoungsModulus, 1.0e9), (P::PoissonsRatio, 0.5)]);
        fill_derived(&mut t);
        assert!(
            !t.has(P::BulkModulus),
            "must not produce an infinite bulk modulus"
        );
        for (p, v, _) in t.iter() {
            assert!(v.is_finite(), "{p:?} became non-finite");
        }
    }

    #[test]
    fn zero_resistivity_does_not_divide_by_zero() {
        let mut t = table(&[(P::ElectricalResistivity, 0.0)]);
        fill_derived(&mut t);
        assert!(!t.has(P::ElectricalConductivity));
    }

    #[test]
    fn derivation_is_idempotent() {
        let mut t = table(&[
            (P::YoungsModulus, 68.9e9),
            (P::PoissonsRatio, 0.33),
            (P::ElectricalResistivity, 1.856e-7),
            (P::Density, 2700.0),
            (P::SpecificHeat, 896.0),
            (P::ThermalConductivity, 167.0),
        ]);
        let first = fill_derived(&mut t);
        assert!(first > 0);
        let snapshot = t.clone();
        // A second run must be a no-op -- otherwise the fixpoint is unstable
        // and re-ingesting a record would keep mutating it.
        assert_eq!(fill_derived(&mut t), 0);
        assert_eq!(t, snapshot);
    }

    #[test]
    fn empty_table_derives_nothing() {
        let mut t = PropertyTable::new();
        assert_eq!(fill_derived(&mut t), 0);
        assert!(t.is_empty());
    }
}
