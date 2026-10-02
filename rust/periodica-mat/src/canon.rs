//====== periodica/rust/periodica-mat/src/canon.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # canon
//!
//! Key canonicalisation and the datasheet-spelling alias table.
//!
//! ## The problem this solves
//!
//! periodica's 954 JSON datasheets accreted over time and spell the same
//! quantity many different ways:
//!
//! | On disk | Files |
//! |---|---|
//! | `ThermalConductivity_W_mK` / `ThermalConductivity_WmK` | 149 |
//! | `Density_g_cm3` / `Density_kg_m3` / `Density_kgm3` / `density` | ~480 |
//! | `Hardness_HV` / `Vickers_HV` | 94 |
//! | `TensileStrength_MPa` / `UltimateTensileStrength_MPa` / `ultimate_strength_MPa` | 168 |
//!
//! Stripping every non-alphanumeric character and lowercasing collapses the
//! *unit-punctuation* variants exactly -- `ThermalConductivity_W_mK` and
//! `ThermalConductivity_WmK` both become `thermalconductivitywmk`. What remains
//! is genuine synonymy (`TensileStrength` vs `UltimateTensileStrength`), which
//! needs a real alias table.
//!
//! Restricted to the blocks that actually carry material properties, the whole
//! corpus contains **162 distinct canonical keys**. That is small enough to
//! enumerate exhaustively, which is what [`lookup`] and [`is_ignored`] do
//! between them -- and `ingest_reports_no_unknown_keys` in `tests/` asserts the
//! enumeration stays exhaustive as data is added.
//!
//! ## Ignored vs unknown
//!
//! A key is **ignored** when we deliberately do not map it: biological-tier
//! measurements (`hemoglobin_content_pg`), direction-resolved composite
//! entries (`YoungsModulus_0_GPa`, which belongs in an anisotropy field node
//! rather than the scalar bulk table), and quantities with no SI meaning for
//! the runtime (`MachinabilityRating_percent`).
//!
//! A key is **unknown** when nobody has classified it. Unknown keys are
//! reported by [`crate::ingest`] rather than silently dropped, so new data
//! cannot quietly lose properties.

use crate::constants;
use crate::property::PropertyId;

/// Canonicalise a datasheet key: strip everything that is not `[a-z0-9]`,
/// lowercasing as it goes.
///
/// `"ThermalConductivity_W_mK"` and `"ThermalConductivity_WmK"` both yield
/// `"thermalconductivitywmk"`.
pub fn canon(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for c in key.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        }
    }
    out
}

/// Affine conversion from a datasheet's unit into the property's SI unit.
///
/// `si = raw * scale + offset`. The offset exists for Celsius-style data; every
/// mapping in the current corpus uses `offset == 0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitConv {
    pub scale: f64,
    pub offset: f64,
}

impl UnitConv {
    #[inline]
    pub const fn scale(scale: f64) -> Self {
        Self { scale, offset: 0.0 }
    }
    #[inline]
    pub const fn identity() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
        }
    }
    #[inline]
    pub fn apply(self, raw: f64) -> f64 {
        raw * self.scale + self.offset
    }
}

/// A resolved datasheet key: which property it is, and how to get it into SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mapping {
    pub property: PropertyId,
    pub conv: UnitConv,
}

// ── Unit scale constants ──────────────────────────────────────────────────
const GPA: f64 = 1.0e9;
const MPA: f64 = 1.0e6;
const KJ: f64 = 1.0e3;
const G_CM3_TO_KG_M3: f64 = 1.0e3;
const UM: f64 = 1.0e-6;
const PM: f64 = 1.0e-12;
const MM: f64 = 1.0e-3;
const MM2_S: f64 = 1.0e-6;
const KV_MM_TO_V_M: f64 = 1.0e6;
const PCT_TO_FRAC: f64 = 1.0e-2;
/// 1 u in kg (CODATA 2022, from the single constants source).
const AMU_TO_KG: f64 = constants::KG_PER_U;
/// 1 eV in J (exact).
const EV_TO_J: f64 = constants::ELECTRON_VOLT;
/// 1 kJ/mol expressed as J per particle.
const KJ_MOL_TO_J: f64 = 1.0e3 / constants::AVOGADRO;
/// 100 % IACS is defined as 5.8e7 S/m, so one percentage point is 5.8e5 S/m.
const IACS_PCT_TO_S_M: f64 = 5.8e5;

macro_rules! map {
    ($prop:ident) => {
        Some(Mapping {
            property: PropertyId::$prop,
            conv: UnitConv::identity(),
        })
    };
    ($prop:ident, $scale:expr) => {
        Some(Mapping {
            property: PropertyId::$prop,
            conv: UnitConv::scale($scale),
        })
    };
}

/// Resolve a *canonical* key (output of [`canon`]) to a property + conversion.
///
/// Returns `None` for both ignored and unknown keys; use [`is_ignored`] to
/// tell them apart.
pub fn lookup(canon_key: &str) -> Option<Mapping> {
    match canon_key {
        // ── Density ──────────────────────────────────────────────────────
        // Element datasheets spell it bare and use g/cm^3 (H = 8.988e-05).
        "densitygcm3" | "density" => map!(Density, G_CM3_TO_KG_M3),
        "densitykgm3" => map!(Density),

        // ── Elastic ──────────────────────────────────────────────────────
        "youngsmodulusgpa" => map!(YoungsModulus, GPA),
        "youngsmodulusmpa" => map!(YoungsModulus, MPA),
        "shearmodulusgpa" => map!(ShearModulus, GPA),
        "shearmodulusmpa" => map!(ShearModulus, MPA),
        "bulkmodulusgpa" => map!(BulkModulus, GPA),
        "poissonsratio" => map!(PoissonsRatio),
        "storagemoduluspa" => map!(StorageModulus),
        "lossmoduluspa" => map!(LossModulus),

        // ── Strength ─────────────────────────────────────────────────────
        "yieldstrengthmpa" => map!(YieldStrength, MPA),
        // All three spell the same quantity; UTS is the canonical home.
        "tensilestrengthmpa" | "ultimatetensilestrengthmpa" | "ultimatestrengthmpa" => {
            map!(UltimateTensileStrength, MPA)
        }
        "compressivestrengthmpa" => map!(CompressiveStrength, MPA),
        "compressiveyieldstrengthmpa" => map!(CompressiveYieldStrength, MPA),
        "shearstrengthmpa" => map!(ShearStrength, MPA),
        "flexuralstrengthmpa" => map!(FlexuralStrength, MPA),
        "strengthcoefficientmpa" => map!(StrengthCoefficient, MPA),
        "strainhardeningexponent" => map!(StrainHardeningExponent),

        // ── Ductility (kept in percent by engineering convention) ─────────
        "elongationpercent" | "elongationatbreakpct" => map!(ElongationPct),
        "reductionofareapercent" => map!(ReductionOfAreaPct),

        // ── Hardness ─────────────────────────────────────────────────────
        "hardnesshv" | "vickershv" => map!(HardnessVickers),
        "brinellhardnesshb" | "brinellhb" | "hardnesshb" => map!(HardnessBrinell),
        "rockwellhrb" => map!(HardnessRockwellB),
        "rockwellhrc" => map!(HardnessRockwellC),
        "knoophk" | "knoophardnesshk" => map!(HardnessKnoop),
        "mohs" => map!(HardnessMohs),

        // ── Fracture ─────────────────────────────────────────────────────
        "fracturetoughnesskicmpasqrtm"
        | "fracturetoughnessmpasqrtm"
        | "fracturetoughnessmpam05" => map!(FractureToughnessKic, MPA),
        "jickjm2" => map!(Jic, KJ),
        // G_IC is a fracture energy in the same units; mode II and
        // delamination are separate quantities and stay unmapped.
        "fractureenergyjm2" | "mode1toughnessgicjm2" => map!(FractureEnergy),
        "parisc" => map!(ParisC),
        "parism" => map!(ParisM),
        "thresholddeltakmpasqrtm" => map!(ThresholdDeltaK, MPA),
        // Room-temperature Charpy is the reference point; `at_minus40C` is a
        // second sample of the same curve and is ignored.
        "at20c" | "impactstrengthj" => map!(CharpyImpact),
        "weibullmodulus" => map!(WeibullModulus),
        "dbttk" => map!(Dbtt),
        "criticalflawsizeum" => map!(CriticalFlawSize, UM),
        "criticalgriffithlengthmm" => map!(CriticalFlawSize, MM),

        // ── Fatigue ──────────────────────────────────────────────────────
        "fatiguestrengthmpa" => map!(FatigueStrength, MPA),
        // Rotating-bending is the standard fatigue-limit test, so an
        // unqualified "fatigue limit" and a "bending fatigue limit" are the
        // same quantity. Contact (Hertzian) fatigue is not, and stays ignored.
        "fatiguelimitmpa" | "bendingfatiguelimitmpa" => map!(FatigueLimit, MPA),
        "fatiguestrengthexponent" => map!(FatigueStrengthExponent),
        "endurancelimitcycles" => map!(EnduranceLimitCycles),

        // ── Failure criteria ─────────────────────────────────────────────
        "maxprincipalstressmpa" | "maxtensilestressmpa" => map!(MaxPrincipalStress, MPA),
        "maxshearstressmpa" => map!(MaxShearStress, MPA),
        "vonmisesstressmpa" | "maxvonmisesstressmpa" => map!(VonMisesStress, MPA),
        // Heat-deflection temperature is the practical service ceiling quoted
        // for polymers, which is what MaxTemperature means for the runtime.
        "maxtemperaturek"
        | "maxservicetempk"
        | "maxoperatingtemperaturek"
        | "heatdeflectiontempk" => map!(MaxTemperature),

        // ── Thermal ──────────────────────────────────────────────────────
        "thermalconductivitywmk" => map!(ThermalConductivity),
        "specificheatjkgk" => map!(SpecificHeat),
        "thermalexpansionperk" => map!(ThermalExpansion),
        "thermaldiffusivitymm2s" => map!(ThermalDiffusivity, MM2_S),
        "meltingpointk" | "meltingpoint" => map!(MeltingPoint),
        "boilingpointk" | "boilingpoint" => map!(BoilingPoint),
        "solidustemperaturek" => map!(SolidusTemperature),
        "liquidustemperaturek" => map!(LiquidusTemperature),
        "glasstransitionk" | "glasstransitiontempk" => map!(GlassTransition),

        // ── Electrical ───────────────────────────────────────────────────
        "electricalresistivityohmm" => map!(ElectricalResistivity),
        // %IACS is defined against 5.8e7 S/m, so this is an exact conversion.
        "percentiacs" => map!(ElectricalConductivity, IACS_PCT_TO_S_M),
        "dielectricconstant" => map!(RelativePermittivity),
        "dielectricstrengthkvmm" => map!(DielectricStrength, KV_MM_TO_V_M),
        "resistivitytemperaturecoefficientperk" => map!(ResistivityTempCoeff),

        // ── Optical / surface ────────────────────────────────────────────
        "refractiveindex" => map!(RefractiveIndexN),
        "transmittancevisiblepercent" => map!(Transmittance, PCT_TO_FRAC),
        "abbenumber" => map!(AbbeNumber),
        "emissivity" => map!(Emissivity),
        "surfaceroughnessum" => map!(SurfaceRoughnessRa, UM),

        // ── Misc physical ────────────────────────────────────────────────
        "porositypercent" => map!(Porosity, PCT_TO_FRAC),
        "permeabilitym2" => map!(Permeability),

        // ── Atomic (element-tier root scalars) ───────────────────────────
        // NOTE the mixed units *within a single element file*: ionization
        // energy is in eV but electron affinity is in kJ/mol. Verified against
        // hydrogen (13.598 eV, 72.8 kJ/mol). This is exactly the kind of trap
        // that makes a grounded alias table necessary.
        "atomicmass" => map!(AtomicMass, AMU_TO_KG),
        "atomicnumber" => map!(AtomicNumber),
        "electronegativity" => map!(Electronegativity),
        "atomicradius" => map!(AtomicRadius, PM),
        "covalentradius" => map!(CovalentRadius, PM),
        "ionizationenergy" => map!(IonizationEnergy, EV_TO_J),
        "electronaffinity" => map!(ElectronAffinity, KJ_MOL_TO_J),
        "valenceelectrons" => map!(ValenceElectrons),

        _ => None,
    }
}

/// Canonical keys we deliberately do not map into the scalar bulk table.
///
/// Kept separate from "unknown" so that [`crate::ingest`] can report genuinely
/// unclassified keys without drowning them in noise.
pub fn is_ignored(canon_key: &str) -> bool {
    matches!(
        canon_key,
        // Direction-resolved composite/laminate values. These describe
        // anisotropy and belong in a FieldNode::AnisotropyAxial, not in the
        // isotropic bulk table -- mapping them would silently overwrite the
        // bulk value with an off-axis one.
        "youngsmodulus0gpa"
            | "youngsmodulus90gpa"
            | "shearmodulus12gpa"
            | "poissonsratio12"
            | "poissonsratio21"
            | "quasiisotropicmodulusgpa"
            | "tensilestrength0mpa"
            | "tensilestrength90mpa"
            | "compressivestrength0mpa"
            | "compressivestrength90mpa"
            | "shearstrength12mpa"
            | "straintofailure0percent"
            | "straintofailure90percent"
            | "thermalconductivity0wmk"
            | "thermalconductivity90wmk"
            | "thermalexpansion0perk"
            | "thermalexpansion90perk"
            | "interlaminarshearstrengthmpa"
            | "maxfiberstrain"
            | "maxmatrixstrain"
            | "tsaiwuindex"
            // Per-constituent densities of a composite, not the bulk density.
            | "fiberdensitykgm3"
            | "matrixdensitykgm3"
            // Elastic bounds, not measured values; unit not stated on disk.
            | "youngsmodulusvoigt"
            | "youngsmodulusreuss"
            // Further samples of the Charpy temperature curve; `at_20C` is the
            // reference point the bulk table keeps.
            | "atminus40c"
            | "atminus196c"
            | "at650c"
            // Hertzian contact fatigue is a different failure mode from the
            // rotating-bending fatigue limit and must not be conflated with it.
            | "contactfatiguelimitmpa"
            // Case-hardening geometry and process parameters, not bulk
            // properties of the material.
            | "criticaldepthmm"
            | "rratio"
            // Drude-model inputs. These belong to the optical model in
            // `optical.rs`, not to the scalar bulk table.
            | "plasmafrequencyev"
            | "dampingev"
            // Strain-based failure criterion; the runtime works in stress.
            | "maxprincipalstrain"
            // Rockwell without a stated scale (B and C are not comparable),
            // and Shore D, which has no defined conversion to HV/HB.
            | "rockwellhr"
            | "shored"
            // Unqualified symbols in the rock datasheets: `alpha` could be the
            // Biot coefficient or a thermal expansion, and `K_MPa` could be a
            // bulk modulus or a strength coefficient. Guessing either would be
            // worse than declining.
            | "alpha"
            | "kmpa"
            | "thresholdmpa"
            // Polymer moisture behaviour, no runtime meaning.
            | "waterabsorptionpercent"
            // Protein-tier sequence statistics.
            | "gravy"
            | "helixpercent"
            | "sheetpercent"
            | "isoelectricpoint"
            | "molecularmass"
            | "length"
            // Tissue geometry.
            | "thicknessmm"
            // Distinct fracture modes with no scalar-bulk meaning.
            | "mode2toughnessgiicjm2"
            | "delaminationtoughnessjm2"
            // Fatigue curve-shape parameters.
            | "fatiguestrengthcoefficientmpa"
            | "fatiguestrengthratio"
            | "sncurveslope"
            // Rock/soil test-specific quantities.
            | "braziliantensilestrengthmpa"
            | "cohesionmpa"
            | "frictionangledeg"
            // Manufacturing / service ratings with no runtime meaning.
            | "machinabilityratingpercent"
            | "creepstrength100000hmpa"
            | "thermoelectricemfuvk"
            | "thermalshockresistancedeltatk"
            | "softeningpointk"
            | "quartztransitiontemperaturek"
            | "rockwellhrf"
            | "barcol"
            | "watercontentpercent"
            // Nested `{"value": ..., "wavelength": ...}` spectral records; the
            // spectral path handles these, not the scalar table.
            | "value"
            // Biological-tier measurements (cells, organelles, tissues).
            | "diameternm"
            | "diameterum"
            | "thicknessnm"
            | "thicknessum"
            | "volumefl"
            | "surfaceareaum2"
            | "masspg"
            | "hemoglobincontentpg"
            | "lifespandays"
            | "molecularmassmda"
            | "copynumber"
            | "copynumberpercell"
            | "proteincount"
            | "rnacontentpercent"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canon_collapses_both_unit_conventions() {
        // The exact collapse the ingest design depends on.
        assert_eq!(canon("ThermalConductivity_W_mK"), "thermalconductivitywmk");
        assert_eq!(canon("ThermalConductivity_WmK"), "thermalconductivitywmk");
        assert_eq!(
            canon("ElectricalResistivity_Ohm_m"),
            "electricalresistivityohmm"
        );
        assert_eq!(
            canon("ElectricalResistivity_ohmm"),
            "electricalresistivityohmm"
        );
        assert_eq!(canon("PoissonsRatio"), "poissonsratio");
        assert_eq!(canon("poissons_ratio"), "poissonsratio");
    }

    #[test]
    fn canon_is_idempotent() {
        for k in ["Density_g_cm3", "YoungsModulus_GPa", "at_20C", "Paris_C"] {
            let once = canon(k);
            assert_eq!(canon(&once), once);
        }
    }

    #[test]
    fn density_spellings_all_reach_si() {
        // 2.775 g/cm^3 and 2775 kg/m^3 are the same material.
        let a = lookup(&canon("Density_g_cm3")).unwrap();
        let b = lookup(&canon("Density_kg_m3")).unwrap();
        let c = lookup(&canon("Density_kgm3")).unwrap();
        assert_eq!(a.property, PropertyId::Density);
        assert_eq!(b.property, PropertyId::Density);
        assert_eq!(c.property, PropertyId::Density);
        assert!((a.conv.apply(2.775) - 2775.0).abs() < 1e-9);
        assert!((b.conv.apply(2775.0) - 2775.0).abs() < 1e-9);
        assert!((c.conv.apply(2775.0) - 2775.0).abs() < 1e-9);
    }

    #[test]
    fn modulus_units_convert_to_pascals() {
        let gpa = lookup(&canon("YoungsModulus_GPa")).unwrap();
        let mpa = lookup(&canon("youngs_modulus_MPa")).unwrap();
        assert_eq!(gpa.property, PropertyId::YoungsModulus);
        assert_eq!(mpa.property, PropertyId::YoungsModulus);
        // 68.9 GPa == 68900 MPa == 6.89e10 Pa (Aluminium 6061-T6).
        assert!((gpa.conv.apply(68.9) - 6.89e10).abs() < 1.0);
        assert!((mpa.conv.apply(68_900.0) - 6.89e10).abs() < 1.0);
    }

    #[test]
    fn tensile_synonyms_agree() {
        for k in [
            "TensileStrength_MPa",
            "UltimateTensileStrength_MPa",
            "ultimate_strength_MPa",
        ] {
            let m = lookup(&canon(k)).unwrap();
            assert_eq!(m.property, PropertyId::UltimateTensileStrength, "{k}");
            assert!((m.conv.apply(451.0) - 4.51e8).abs() < 1.0, "{k}");
        }
    }

    #[test]
    fn iacs_percentage_is_an_exact_conductivity() {
        // Pure annealed copper is 100 %IACS == 5.8e7 S/m by definition.
        let m = lookup(&canon("PercentIACS")).unwrap();
        assert_eq!(m.property, PropertyId::ElectricalConductivity);
        assert!((m.conv.apply(100.0) - 5.8e7).abs() < 1.0);
    }

    #[test]
    fn element_root_keys_use_their_own_units() {
        // Hydrogen: 8.988e-05 g/cm^3, 13.598 eV, 72.8 kJ/mol, 53 pm.
        let d = lookup("density").unwrap();
        assert!((d.conv.apply(8.988e-5) - 0.08988).abs() < 1e-9);

        // 13.598 eV * 1.602176634e-19 J/eV = 2.178640e-18 J.
        let ie = lookup("ionizationenergy").unwrap();
        let ie_j = ie.conv.apply(13.598);
        assert!(
            (ie_j / 2.178_640e-18 - 1.0).abs() < 1e-5,
            "ionization energy {ie_j} J"
        );

        // Mixed units inside one file: affinity is kJ/mol, not eV.
        // 72.8 kJ/mol / 6.02214076e23 = 1.208872e-19 J per atom.
        let ea = lookup("electronaffinity").unwrap();
        let ea_j = ea.conv.apply(72.8);
        assert!(
            (ea_j / 1.208_872e-19 - 1.0).abs() < 1e-5,
            "electron affinity {ea_j} J"
        );
        // Sanity: the two must not be off by the ~96 000x that confusing
        // eV with kJ/mol would produce.
        assert!(
            ea_j < ie_j,
            "affinity should be smaller than ionization energy"
        );

        let r = lookup("atomicradius").unwrap();
        assert!((r.conv.apply(53.0) - 53e-12).abs() < 1e-24);
    }

    #[test]
    fn percentages_that_should_become_fractions_do() {
        let p = lookup(&canon("Porosity_percent")).unwrap();
        assert!((p.conv.apply(12.0) - 0.12).abs() < 1e-12);
        let t = lookup(&canon("TransmittanceVisible_percent")).unwrap();
        assert!((t.conv.apply(92.0) - 0.92).abs() < 1e-12);
    }

    #[test]
    fn engineering_percentages_stay_percentages() {
        // Elongation is quoted in percent everywhere in materials engineering;
        // silently dividing by 100 would surprise every consumer.
        let e = lookup(&canon("Elongation_percent")).unwrap();
        assert_eq!(e.property, PropertyId::ElongationPct);
        assert!((e.conv.apply(26.0) - 26.0).abs() < 1e-12);
    }

    #[test]
    fn ignored_keys_are_not_mapped() {
        for k in [
            "YoungsModulus_0_GPa",
            "hemoglobin_content_pg",
            "TsaiWuIndex",
        ] {
            let c = canon(k);
            assert!(lookup(&c).is_none(), "{k} must not map to a bulk property");
            assert!(is_ignored(&c), "{k} must be explicitly ignored");
        }
    }

    #[test]
    fn a_key_is_never_both_mapped_and_ignored() {
        // Overlap would mean the ingest report is lying about coverage.
        for p in PropertyId::ALL {
            let c = canon(p.name());
            if lookup(&c).is_some() {
                assert!(!is_ignored(&c), "{} is both mapped and ignored", p.name());
            }
        }
    }
}
