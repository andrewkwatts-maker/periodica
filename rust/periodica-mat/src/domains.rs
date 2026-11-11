//! The PlayTow domain table: which `PropertyId`s belong to which domain, defined once.
//!
//! Domains follow PlayTow's REQ-MAT, REQ-THERMAL, REQ-FIELDS and REQ-SPECTRAL items and the
//! asks in `docs/plan/playtow-sheets.md`. A domain with no id today keeps an empty list, so the
//! gap shows in the census. Used by the `domain_census` example and its test.

use crate::property::PropertyId as P;

/// One census domain and the property ids that count towards it.
#[derive(Debug)]
pub struct Domain {
    pub name: &'static str,
    pub ids: &'static [P],
}

/// The domains, in report order. Every id appears in exactly one domain or in [`EXCLUDED`].
pub const DOMAINS: &[Domain] = &[
    Domain {
        name: "mass",
        // Viscosity sits with the flow property `Permeability`: the table has no
        // transport domain, and these describe how a fluid moves through or past matter.
        ids: &[
            P::Density,
            P::Porosity,
            P::Permeability,
            P::DynamicViscosity,
            P::ViscosityReferenceTemperature,
        ],
    },
    Domain {
        name: "elastic",
        ids: &[
            P::YoungsModulus,
            P::ShearModulus,
            P::BulkModulus,
            P::PoissonsRatio,
            P::StorageModulus,
            P::LossModulus,
        ],
    },
    Domain {
        name: "failure",
        ids: &[
            P::YieldStrength,
            P::UltimateTensileStrength,
            P::CompressiveStrength,
            P::MeanCompressiveStrength,
            P::PeakCompressiveStrain,
            P::UltimateCompressiveStrain,
            P::AggregateMaxSize,
            P::CompressiveYieldStrength,
            P::ShearStrength,
            P::FlexuralStrength,
            P::StrengthCoefficient,
            P::StrainHardeningExponent,
            P::ElongationPct,
            P::ReductionOfAreaPct,
            P::HardnessVickers,
            P::HardnessBrinell,
            P::HardnessRockwellB,
            P::HardnessRockwellC,
            P::HardnessKnoop,
            P::HardnessMohs,
            P::FractureToughnessKic,
            P::Jic,
            P::FractureEnergy,
            P::ParisC,
            P::ParisM,
            P::ThresholdDeltaK,
            P::CharpyImpact,
            P::WeibullModulus,
            P::Dbtt,
            P::CriticalFlawSize,
            P::FatigueStrength,
            P::FatigueLimit,
            P::FatigueStrengthExponent,
            P::EnduranceLimitCycles,
            P::MaxPrincipalStress,
            P::MaxShearStress,
            P::VonMisesStress,
            P::TensileStrength,
            P::Hardness,
            P::CharacteristicCompressiveStrength,
        ],
    },
    // Friction and restitution have no id; only surface descriptors exist.
    Domain {
        name: "contact",
        ids: &[
            P::SurfaceRoughnessRa,
            P::SurfaceEnergy,
            P::SurfaceSlopeRms,
            P::KineticStaticRatio,
        ],
    },
    Domain {
        name: "optical",
        ids: &[
            P::RefractiveIndexN,
            P::ExtinctionK,
            P::Transmittance,
            P::Reflectance,
            P::Absorptance,
            P::Emissivity,
            P::AbbeNumber,
            P::SellmeierB1,
            P::SellmeierB2,
            P::SellmeierB3,
            P::SellmeierC1,
            P::SellmeierC2,
            P::SellmeierC3,
            P::DrudePlasmaFrequency,
            P::DrudeDamping,
            P::EpsilonInfinity,
            P::EmissionTemperature,
            P::StressOptic,
            P::PrimaryEmissionWavelength,
            P::VisibleEmissionWavelength,
        ],
    },
    Domain {
        name: "thermal",
        ids: &[
            P::MaxTemperature,
            P::ThermalConductivity,
            P::SpecificHeat,
            P::ThermalExpansion,
            P::ThermalDiffusivity,
            P::MeltingPoint,
            P::BoilingPoint,
            P::SolidusTemperature,
            P::LiquidusTemperature,
            P::GlassTransition,
            P::DebyeTemperature,
            P::LatentHeatFusion,
            P::LatentHeatVaporization,
            P::LatentHeatSublimation,
        ],
    },
    Domain {
        name: "electrical",
        ids: &[
            P::ElectricalResistivity,
            P::ElectricalConductivity,
            P::RelativePermittivity,
            P::DielectricStrength,
            P::ResistivityTempCoeff,
        ],
    },
    Domain {
        name: "magnetic",
        ids: &[
            P::Remanence,
            P::IntrinsicCoercivity,
            P::RelativePermeability,
        ],
    },
    Domain {
        name: "acoustic",
        ids: &[
            P::SoundSpeedLongitudinal,
            P::SoundSpeedShear,
            P::MechanicalLossFactor,
            P::AcousticAttenuationOverF2,
        ],
    },
];

/// Ids left out of every domain on purpose: element-tier atomic data, not a PlayTow domain.
pub const EXCLUDED: &[P] = &[
    P::AtomicMass,
    P::AtomicNumber,
    P::Electronegativity,
    P::AtomicRadius,
    P::CovalentRadius,
    P::IonizationEnergy,
    P::ElectronAffinity,
    P::ValenceElectrons,
];

/// The domain holding `id`, if any.
pub fn domain_of(id: P) -> Option<&'static str> {
    DOMAINS.iter().find(|d| d.ids.contains(&id)).map(|d| d.name)
}
