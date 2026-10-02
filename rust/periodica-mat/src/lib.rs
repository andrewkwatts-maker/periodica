//====== periodica/rust/periodica-mat/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # periodica-mat
//!
//! The canonical typed material model for periodica.
//!
//! periodica's 954 bundled datasheets accreted three different schemas and five
//! different spellings of "density". This crate turns any of them into one
//! SI-normalised [`MaterialRecord`] that the runtime can evaluate without ever
//! parsing a unit suffix or hashing a string.
//!
//! ```
//! use periodica_mat::{ingest, PropertyId};
//! use serde_json::json;
//!
//! // The nested `active/materials` schema...
//! let a = ingest(&json!({"PhysicalProperties": {"Density_g_cm3": 7.87}}), "a").unwrap();
//! // ...and the flat `derived/alloys` schema with a different unit spelling.
//! let b = ingest(&json!({"Properties": {"Density_kgm3": 7870.0}}), "b").unwrap();
//!
//! assert_eq!(a.bulk.get(PropertyId::Density), b.bulk.get(PropertyId::Density));
//! ```
//!
//! ## Module map
//!
//! - [`property`] -- [`PropertyId`], the closed property set and its SI units.
//! - [`canon`] -- key canonicalisation and the datasheet-spelling alias table.
//! - [`table`] -- [`PropertyTable`], a dense array with per-entry provenance.
//! - [`record`] -- [`MaterialRecord`] and the structured blocks (lattice,
//!   phases, microstructure, texture).
//! - [`ingest`] -- the JSON walker that reconciles all three schemas.
//! - [`derive`] -- exact physical relations that fill gaps in the data.
//!
//! ## A note on JSON float fidelity
//!
//! `serde_json` is not bit-exact for `f64` in every case: `25.0 * 1e-6` writes
//! as `0.000024999999999999998` and parses back one ULP away, as `2.5e-5`.
//! That is ~1e-16 relative -- far below the 3-4 significant figures any
//! datasheet actually carries -- so it is harmless for scalar properties.
//!
//! It is *not* harmless for bulk numeric data, which is why baked volumes,
//! heightmaps and graphs live in binary blobs in the `.pmat` container rather
//! than in its JSON header. Tests here compare property values with a relative
//! tolerance rather than `==` for the same reason.
//!
//! ## What this crate deliberately is not
//!
//! It does not read the filesystem, does not know about tiers-as-folders, and
//! does not depend on pyo3. Registry and discovery stay in `periodica_core`;
//! spatial evaluation lives in `periodica-runtime`. This crate is the boundary
//! where messy JSON becomes typed, SI-normalised data, and nothing more.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod canon;
pub mod derive;
pub mod ingest;
pub mod jsonc;
pub mod property;
pub mod record;
pub mod table;

pub use canon::{canon, Mapping, UnitConv};
pub use derive::fill_derived;
pub use ingest::{ingest, is_material_like, IngestError};
pub use jsonc::strip_comments;
pub use property::PropertyId;
pub use record::{
    CrystalSystem, Defects, GrainStructure, Inclusions, Lattice, MaterialRecord, Microstructure,
    NoiseKind, Phase, PhaseDistribution, Provenance, Texture, Tier,
};
pub use table::{PropertyTable, Source};

/// Version of this crate, reported through `.pmat` headers for diagnostics.
pub const PERIODICA_MAT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn end_to_end_ingest_of_a_realistic_alloy() {
        // Shaped like `active/materials/Aluminum_6061_T6.json`, the richest
        // schema in the corpus.
        let v = json!({
            "Name": "Aluminum_6061_T6",
            "Category": "Aluminium Alloy",
            "Color": "#C0C0C0",
            "Components": [
                {"Element": "Al", "MinPercent": 95.8, "MaxPercent": 98.6},
                {"Element": "Mg", "MinPercent": 0.8,  "MaxPercent": 1.2}
            ],
            "PhysicalProperties": {"Density_kg_m3": 2700},
            "ElasticProperties": {
                "YoungsModulus_GPa": 68.9, "ShearModulus_GPa": 26.0,
                "BulkModulus_GPa": 67.6, "PoissonsRatio": 0.33
            },
            "StrengthProperties": {
                "YieldStrength_MPa": 276, "UltimateTensileStrength_MPa": 310,
                "StrainHardeningExponent": 0.085
            },
            "Hardness": {"Brinell_HB": 95, "Vickers_HV": 107},
            "Ductility": {"Elongation_percent": 12, "ReductionOfArea_percent": 25},
            "FractureMechanics": {
                "FractureToughness_KIC_MPa_sqrt_m": 29, "JIC_kJ_m2": 18,
                "CrackGrowthParameters": {"Paris_C": 1.2e-11, "Paris_m": 3.5}
            },
            "ThermalProperties": {"ThermalConductivity_W_mK": 167, "SpecificHeat_J_kgK": 896},
            "ElectricalProperties": {"ElectricalResistivity_Ohm_m": 3.99e-8},
            "FailureCriteria": {"VonMisesStress_MPa": 276, "MaxTemperature_K": 423},
            "LatticeProperties": {"PrimaryStructure": "FCC"}
        });

        let r = ingest(&v, "Aluminum_6061_T6").unwrap();

        assert!(
            r.is_fully_ingested(),
            "unmapped keys: {:?}",
            r.provenance.unknown_keys
        );

        // Everything in SI. Unit conversion is a float multiply, so compare
        // relatively: 67.6 * 1e9 is not bit-identical to the literal 67.6e9.
        let close = |p: PropertyId, want: f64| {
            let got = r.bulk.get(p).unwrap_or_else(|| panic!("{p:?} missing"));
            assert!(
                (got / want - 1.0).abs() < 1e-12,
                "{p:?}: got {got} want {want} {}",
                p.si_unit()
            );
        };
        close(PropertyId::Density, 2700.0);
        close(PropertyId::YoungsModulus, 68.9e9);
        close(PropertyId::YieldStrength, 276.0e6);
        close(PropertyId::FractureToughnessKic, 29.0e6);
        close(PropertyId::Jic, 18.0e3);
        close(PropertyId::ParisC, 1.2e-11);
        close(PropertyId::MaxTemperature, 423.0);
        close(PropertyId::HardnessBrinell, 95.0);
        close(PropertyId::HardnessVickers, 107.0);
        // Elongation stays in percent by engineering convention.
        close(PropertyId::ElongationPct, 12.0);

        // Published elastic constants survive derivation.
        assert_eq!(r.bulk.source(PropertyId::BulkModulus), Source::Datasheet);
        close(PropertyId::BulkModulus, 67.6e9);

        // Conductivity is derived because no datasheet ever states it.
        assert_eq!(
            r.bulk.source(PropertyId::ElectricalConductivity),
            Source::Derived
        );

        // FCC: ductile, no cleavage plane. Cracks must go intergranular.
        assert_eq!(r.crystal_system(), CrystalSystem::Fcc);
        assert!(!r.crystal_system().is_cleavable());

        assert!(is_material_like(&r));
    }

    #[test]
    fn record_survives_a_json_round_trip_after_ingest() {
        let v = json!({
            "Name": "Steel-1018",
            "Properties": {"Density_kgm3": 7870, "YoungsModulus_GPa": 200, "PoissonsRatio": 0.29},
            "LatticeProperties": {"PrimaryStructure": "BCC"},
            "Microstructure": {"GrainStructure": {"AverageGrainSize_um": 25.0}}
        });
        let r = ingest(&v, "Steel-1018").unwrap();
        let s = serde_json::to_string(&r).unwrap();
        let back: MaterialRecord = serde_json::from_str(&s).unwrap();

        // Structure and provenance must survive exactly.
        assert_eq!(r.name, back.name);
        assert_eq!(r.lattice.map(|l| l.system), back.lattice.map(|l| l.system));
        assert_eq!(r.provenance, back.provenance);
        assert_eq!(r.bulk.len(), back.bulk.len());

        // Values to within a float ULP -- see the crate docs on serde_json
        // float fidelity. `25.0 * 1e-6` is the canonical offender here.
        for (p, v, src) in r.bulk.iter() {
            let got = back.bulk.get(p).unwrap();
            assert!((got / v - 1.0).abs() < 1e-12, "{p:?}: {got} vs {v}");
            assert_eq!(back.bulk.source(p), src, "{p:?} provenance");
        }
        let (a, b) = (r.micro.unwrap().grains, back.micro.unwrap().grains);
        assert!((a.average_size_m / b.average_size_m - 1.0).abs() < 1e-12);
    }
}
