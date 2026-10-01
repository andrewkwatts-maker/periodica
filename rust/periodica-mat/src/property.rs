//====== periodica/rust/periodica-mat/src/property.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # property
//!
//! The closed set of material properties periodica can evaluate, and the SI
//! unit each one is stored in.
//!
//! ## Why a closed enum and not a string map
//!
//! The runtime hot path (`MaterialInstance::eval`) has a sub-microsecond
//! budget because it is called from raymarch fragment callbacks. A string key
//! would mean hashing on every sample. [`PropertyId`] is `#[repr(u16)]` and
//! dense from zero, so [`crate::PropertyTable`] is a plain array and lookup is
//! a bounds-checked index -- no hashing, no allocation, no branching.
//!
//! ## Units policy
//!
//! **Every value stored against a `PropertyId` is in the SI unit named by
//! [`PropertyId::si_unit`], without exception.** Conversion happens exactly
//! once, at ingest (see [`crate::canon`]). Unit suffixes never appear in
//! variant names, because the on-disk data spells the same quantity five
//! different ways (`Density_g_cm3`, `Density_kg_m3`, `Density_kgm3`,
//! `Density_per_mm3`, bare `density`) and the whole point of this type is that
//! downstream code never has to care.
//!
//! Two deliberate exceptions, both dimensionless-by-convention quantities
//! where the percentage form *is* the engineering convention and converting to
//! a fraction would surprise every consumer:
//! - [`PropertyId::ElongationPct`] and [`PropertyId::ReductionOfAreaPct`] stay
//!   in percent. Their names say so.
//!
//! Hardness scales (HV/HB/HRC/...) are not inter-convertible by any exact
//! relation, so each scale is its own property rather than one "hardness".

use serde::{Deserialize, Serialize};

/// Defines the property set once, deriving the enum, the count, the iteration
/// order, the SI unit strings and the display names from a single table.
///
/// Keeping these in one place is what stops `si_unit()` drifting out of sync
/// with the variant list as properties are added.
macro_rules! define_properties {
    ($( $variant:ident => ($si:expr, $disp:expr) ),* $(,)?) => {
        /// A material property. Dense, `#[repr(u16)]`, indexable into
        /// [`crate::PropertyTable`].
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[repr(u16)]
        #[non_exhaustive]
        pub enum PropertyId {
            $( $variant ),*
        }

        impl PropertyId {
            /// Number of distinct properties. This is the width of every
            /// [`crate::PropertyTable`].
            pub const COUNT: usize = [ $( stringify!($variant) ),* ].len();

            /// Every property, in declaration order (== numeric order).
            pub const ALL: &'static [PropertyId] = &[ $( PropertyId::$variant ),* ];

            /// The SI unit this property is *always* stored in.
            #[inline]
            pub const fn si_unit(self) -> &'static str {
                match self { $( PropertyId::$variant => $si ),* }
            }

            /// Stable machine name (matches the Rust variant), used as the
            /// key in `.pmat` JSON and accepted by [`PropertyId::from_name`].
            #[inline]
            pub const fn name(self) -> &'static str {
                match self { $( PropertyId::$variant => stringify!($variant) ),* }
            }

            /// Human-facing label for the app's inspector.
            #[inline]
            pub const fn display_name(self) -> &'static str {
                match self { $( PropertyId::$variant => $disp ),* }
            }
        }
    };
}

define_properties! {
    // ── Bulk physical ────────────────────────────────────────────────────
    Density                  => ("kg/m^3",   "Density"),
    Porosity                 => ("1",        "Porosity"),
    Permeability             => ("m^2",      "Permeability"),

    // ── Elastic ──────────────────────────────────────────────────────────
    YoungsModulus            => ("Pa",       "Young's Modulus"),
    ShearModulus             => ("Pa",       "Shear Modulus"),
    BulkModulus              => ("Pa",       "Bulk Modulus"),
    PoissonsRatio            => ("1",        "Poisson's Ratio"),
    StorageModulus           => ("Pa",       "Storage Modulus"),
    LossModulus              => ("Pa",       "Loss Modulus"),

    // ── Strength ─────────────────────────────────────────────────────────
    YieldStrength            => ("Pa",       "Yield Strength"),
    UltimateTensileStrength  => ("Pa",       "Ultimate Tensile Strength"),
    CompressiveStrength      => ("Pa",       "Compressive Strength"),
    CompressiveYieldStrength => ("Pa",       "Compressive Yield Strength"),
    ShearStrength            => ("Pa",       "Shear Strength"),
    FlexuralStrength         => ("Pa",       "Flexural Strength"),
    StrengthCoefficient      => ("Pa",       "Strength Coefficient"),
    StrainHardeningExponent  => ("1",        "Strain-Hardening Exponent"),

    // ── Ductility (percent by engineering convention) ─────────────────────
    ElongationPct            => ("%",        "Elongation at Break"),
    ReductionOfAreaPct       => ("%",        "Reduction of Area"),

    // ── Hardness (each scale is distinct; no exact inter-conversion) ──────
    HardnessVickers          => ("HV",       "Hardness (Vickers)"),
    HardnessBrinell          => ("HB",       "Hardness (Brinell)"),
    HardnessRockwellB        => ("HRB",      "Hardness (Rockwell B)"),
    HardnessRockwellC        => ("HRC",      "Hardness (Rockwell C)"),
    HardnessKnoop            => ("HK",       "Hardness (Knoop)"),
    HardnessMohs             => ("1",        "Hardness (Mohs)"),

    // ── Fracture / fatigue ───────────────────────────────────────────────
    FractureToughnessKic     => ("Pa*m^0.5", "Fracture Toughness K_IC"),
    Jic                      => ("J/m^2",    "J-Integral J_IC"),
    FractureEnergy           => ("J/m^2",    "Fracture Energy"),
    ParisC                   => ("1",        "Paris Law C"),
    ParisM                   => ("1",        "Paris Law m"),
    ThresholdDeltaK          => ("Pa*m^0.5", "Threshold Stress Intensity"),
    CharpyImpact             => ("J",        "Charpy Impact Energy"),
    WeibullModulus           => ("1",        "Weibull Modulus"),
    Dbtt                     => ("K",        "Ductile-Brittle Transition"),
    CriticalFlawSize         => ("m",        "Critical Flaw Size"),
    FatigueStrength          => ("Pa",       "Fatigue Strength"),
    FatigueLimit             => ("Pa",       "Fatigue Limit"),
    FatigueStrengthExponent  => ("1",        "Fatigue Strength Exponent"),
    EnduranceLimitCycles     => ("cycles",   "Endurance Limit"),

    // ── Failure criteria ─────────────────────────────────────────────────
    MaxPrincipalStress       => ("Pa",       "Max Principal Stress"),
    MaxShearStress           => ("Pa",       "Max Shear Stress"),
    VonMisesStress           => ("Pa",       "Von Mises Stress"),
    MaxTemperature           => ("K",        "Max Service Temperature"),

    // ── Thermal ──────────────────────────────────────────────────────────
    ThermalConductivity      => ("W/(m*K)",  "Thermal Conductivity"),
    SpecificHeat             => ("J/(kg*K)", "Specific Heat"),
    ThermalExpansion         => ("1/K",      "Thermal Expansion"),
    ThermalDiffusivity       => ("m^2/s",    "Thermal Diffusivity"),
    MeltingPoint             => ("K",        "Melting Point"),
    BoilingPoint             => ("K",        "Boiling Point"),
    SolidusTemperature       => ("K",        "Solidus Temperature"),
    LiquidusTemperature      => ("K",        "Liquidus Temperature"),
    GlassTransition          => ("K",        "Glass Transition"),

    // ── Electrical ───────────────────────────────────────────────────────
    ElectricalResistivity    => ("Ohm*m",    "Electrical Resistivity"),
    ElectricalConductivity   => ("S/m",      "Electrical Conductivity"),
    RelativePermittivity     => ("1",        "Relative Permittivity"),
    DielectricStrength       => ("V/m",      "Dielectric Strength"),
    ResistivityTempCoeff     => ("1/K",      "Resistivity Temp. Coefficient"),

    // ── Optical / surface ────────────────────────────────────────────────
    RefractiveIndexN         => ("1",        "Refractive Index n"),
    ExtinctionK              => ("1",        "Extinction Coefficient k"),
    Transmittance            => ("1",        "Transmittance"),
    Reflectance              => ("1",        "Reflectance"),
    Absorptance              => ("1",        "Absorptance"),
    Emissivity               => ("1",        "Emissivity"),
    AbbeNumber               => ("1",        "Abbe Number"),
    SurfaceRoughnessRa       => ("m",        "Surface Roughness Ra"),

    // ── Atomic (element-tier datasheets) ─────────────────────────────────
    AtomicMass               => ("kg",       "Atomic Mass"),
    AtomicNumber             => ("1",        "Atomic Number"),
    Electronegativity        => ("1",        "Electronegativity"),
    AtomicRadius             => ("m",        "Atomic Radius"),
    CovalentRadius           => ("m",        "Covalent Radius"),
    IonizationEnergy         => ("J",        "Ionization Energy"),
    ElectronAffinity         => ("J",        "Electron Affinity"),
    ValenceElectrons         => ("1",        "Valence Electrons"),
}

impl PropertyId {
    /// Dense index into a [`crate::PropertyTable`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`PropertyId::index`].
    ///
    /// Returns `None` when `i` is out of range, so a corrupt `.pmat` cannot
    /// produce an invalid discriminant.
    #[inline]
    pub fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }

    /// Parse a machine name as produced by [`PropertyId::name`].
    ///
    /// Matching is case-insensitive and ignores punctuation, so the Python
    /// boundary accepts `"YoungsModulus"`, `"youngs_modulus"` and
    /// `"Young's Modulus"` alike.
    pub fn from_name(s: &str) -> Option<Self> {
        let want = crate::canon::canon(s);
        Self::ALL
            .iter()
            .copied()
            .find(|p| crate::canon::canon(p.name()) == want)
            // A caller is far more likely to pass a datasheet spelling than the
            // Rust variant name, so fall back to the full ingest alias table.
            .or_else(|| crate::canon::lookup(&want).map(|m| m.property))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_matches_all_slice() {
        assert_eq!(PropertyId::COUNT, PropertyId::ALL.len());
    }

    #[test]
    fn indices_are_dense_and_ordered() {
        // PropertyTable is a plain array indexed by `index()`. If the
        // discriminants were ever sparse or reordered, every table would
        // silently misalign -- so pin it.
        for (i, p) in PropertyId::ALL.iter().enumerate() {
            assert_eq!(
                p.index(),
                i,
                "{p:?} has index {} but sits at {i}",
                p.index()
            );
            assert_eq!(PropertyId::from_index(i), Some(*p));
        }
        assert_eq!(PropertyId::from_index(PropertyId::COUNT), None);
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = PropertyId::ALL.iter().map(|p| p.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate PropertyId name");
    }

    #[test]
    fn every_property_has_a_unit_and_label() {
        for p in PropertyId::ALL {
            assert!(!p.si_unit().is_empty(), "{p:?} has no unit");
            assert!(!p.display_name().is_empty(), "{p:?} has no display name");
        }
    }

    #[test]
    fn from_name_round_trips_and_tolerates_spelling() {
        for p in PropertyId::ALL {
            assert_eq!(PropertyId::from_name(p.name()), Some(*p));
        }
        assert_eq!(
            PropertyId::from_name("youngs_modulus"),
            Some(PropertyId::YoungsModulus)
        );
        assert_eq!(
            PropertyId::from_name("Young's Modulus"),
            Some(PropertyId::YoungsModulus)
        );
        // A datasheet spelling, resolved via the alias table.
        assert_eq!(
            PropertyId::from_name("Density_g_cm3"),
            Some(PropertyId::Density)
        );
        assert_eq!(PropertyId::from_name("not_a_property"), None);
    }
}
