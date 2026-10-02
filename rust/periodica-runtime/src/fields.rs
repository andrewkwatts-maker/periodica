//====== periodica/rust/periodica-runtime/src/fields.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # fields
//!
//! Scalar fields over space, and where their values come from.
//!
//! The runtime samples geometry and material properties through plain
//! closures ([`SdfFn`], [`DensityFn`], [`ToughnessFn`]) so it never needs an
//! engine's SDF or expression types. [`MaterialFields`] lifts the uniform bulk
//! values off an ingested datasheet so those closures can be fed from real
//! data:
//!
//! ```
//! use periodica_mat::{ingest, CrystalSystem};
//! use periodica_runtime::{integrate_mass, MaterialFields, WeakpointGrid};
//! use serde_json::json;
//!
//! let record = ingest(
//!     &json!({
//!         "PhysicalProperties": {"Density_kg_m3": 7870},
//!         "FractureMechanics": {"FractureToughness_KIC_MPa_sqrt_m": 50},
//!         "LatticeProperties": {"PrimaryStructure": "BCC"}
//!     }),
//!     "steel",
//! )
//! .unwrap();
//! let fields = MaterialFields::from_record(&record);
//! assert_eq!(fields.crystal_system, CrystalSystem::Bcc);
//!
//! let rho = fields.density_kg_m3.unwrap();
//! let k_ic = fields.fracture_toughness_pa_sqrt_m.unwrap(); // 50e6, SI
//!
//! let cube = |_p: [f64; 3]| -1.0;
//! let bounds = ([0.0; 3], [0.1; 3]);
//! let m = integrate_mass(&cube, &|_p| rho, bounds, 4096, 0).unwrap();
//! assert!((m.total_mass_kg - 7.87).abs() < 1e-9);
//! let grid = WeakpointGrid::build(bounds, (4, 4, 4), &|_p| k_ic).unwrap();
//! assert_eq!(grid.len(), 64);
//! ```

use periodica_mat::{CrystalSystem, MaterialRecord, PropertyId};

use crate::crystal::LatticeKind;

/// Signed-distance sampler: distance in metres at a point in metres.
/// Negative inside, positive outside.
pub type SdfFn<'a> = &'a dyn Fn([f64; 3]) -> f64;

/// Density sampler: local density in kg/m^3 at a point in metres.
pub type DensityFn<'a> = &'a dyn Fn([f64; 3]) -> f64;

/// Fracture-toughness sampler: local mode-I fracture toughness `K_IC` in
/// Pa*m^0.5 (SI) at a point in metres.
///
/// SI is a deliberate change from the engine original, which documented
/// MPa*m^0.5: `periodica-mat` stores every property in SI, so a value read
/// straight off a [`MaterialRecord`] is already in Pa*m^0.5. Only the
/// *relative* toughness across a grid affects the crack path; the absolute
/// scale only moves `FracturePlane::weakpoint_confidence`.
pub type ToughnessFn<'a> = &'a dyn Fn([f64; 3]) -> f64;

/// The uniform bulk values of a datasheet that the runtime's fields need.
///
/// A `None` means the datasheet does not state the property (or states a
/// non-positive or non-finite value, which is physically meaningless for
/// both). The caller decides whether to fall back or to refuse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialFields {
    /// Bulk density in kg/m^3.
    pub density_kg_m3: Option<f64>,
    /// Plane-strain fracture toughness `K_IC` in Pa*m^0.5 (SI; datasheets
    /// quote MPa*m^0.5 and ingest has already multiplied by `1e6`).
    pub fracture_toughness_pa_sqrt_m: Option<f64>,
    /// Crystal system of the primary phase (see
    /// [`MaterialRecord::crystal_system`]).
    pub crystal_system: CrystalSystem,
}

impl MaterialFields {
    /// Read the uniform bulk fields off an ingested record.
    pub fn from_record(record: &MaterialRecord) -> Self {
        let positive = |p: PropertyId| record.bulk.get(p).filter(|v| v.is_finite() && *v > 0.0);
        Self {
            density_kg_m3: positive(PropertyId::Density),
            fracture_toughness_pa_sqrt_m: positive(PropertyId::FractureToughnessKic),
            crystal_system: record.crystal_system(),
        }
    }

    /// The lattice family for [`crate::CrystalPrimitive`] or an engine's
    /// lattice tag; `None` for amorphous or unknown structure.
    pub fn lattice_kind(&self) -> Option<LatticeKind> {
        LatticeKind::from_crystal_system(self.crystal_system)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use periodica_mat::{ingest, PropertyTable, Source};
    use serde_json::json;

    #[test]
    fn reads_si_values_from_an_ingested_datasheet() {
        let r = ingest(
            &json!({
                "PhysicalProperties": {"Density_g_cm3": 2.70},
                "FractureMechanics": {"FractureToughness_KIC_MPa_sqrt_m": 29},
                "LatticeProperties": {"PrimaryStructure": "FCC"}
            }),
            "Aluminum",
        )
        .unwrap();
        let f = MaterialFields::from_record(&r);
        let rho = f.density_kg_m3.unwrap();
        let k = f.fracture_toughness_pa_sqrt_m.unwrap();
        assert!((rho / 2700.0 - 1.0).abs() < 1e-12, "{rho}");
        // MPa*m^0.5 on the datasheet, Pa*m^0.5 here.
        assert!((k / 29.0e6 - 1.0).abs() < 1e-12, "{k}");
        assert_eq!(f.crystal_system, CrystalSystem::Fcc);
        assert_eq!(f.lattice_kind(), Some(LatticeKind::FCC));
    }

    #[test]
    fn absent_or_meaningless_values_are_none() {
        let mut bulk = PropertyTable::new();
        bulk.set(PropertyId::Density, 0.0, Source::Datasheet);
        bulk.set(PropertyId::FractureToughnessKic, -5.0, Source::Datasheet);
        let r = MaterialRecord {
            name: "Nothing".into(),
            bulk,
            ..Default::default()
        };
        let f = MaterialFields::from_record(&r);
        assert_eq!(f.density_kg_m3, None);
        assert_eq!(f.fracture_toughness_pa_sqrt_m, None);
        assert_eq!(f.crystal_system, CrystalSystem::Unknown);
        assert_eq!(f.lattice_kind(), None);
    }
}
