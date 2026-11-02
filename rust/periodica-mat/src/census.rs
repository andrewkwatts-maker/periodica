//! Volume-readiness census: what a record lacks before a volumetric solver can use it.

use crate::property::PropertyId;
use crate::record::MaterialRecord;

/// Outcome of [`volume_readiness`]; `volume_ready` is true exactly when `missing` is empty.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Readiness {
    /// Bulk, microstructure and phase data are all present.
    pub volume_ready: bool,
    /// The sheet carries orthotropic data (complete or not).
    pub orthotropic: bool,
    /// Keys the sheet does not give: `bulk.Density`, `micro.grains`, `phases`, `orthotropy.E1`, ...
    pub missing: Vec<String>,
}

/// Reports what the record lacks for a fine volume: bulk density and Young's modulus,
/// grains, phases, and every orthotropy key where orthotropy is present. Nothing is filled.
pub fn volume_readiness(record: &MaterialRecord) -> Readiness {
    let mut missing = Vec::new();
    if !record.bulk.has(PropertyId::Density) {
        missing.push("bulk.Density".to_string());
    }
    if !record.bulk.has(PropertyId::YoungsModulus) {
        missing.push("bulk.YoungsModulus".to_string());
    }
    if record.micro.is_none() {
        missing.push("micro.grains".to_string());
    }
    if record.phases.is_empty() {
        missing.push("phases".to_string());
    }
    if let Some(o) = &record.orthotropy {
        let keys: [(&str, bool); 7] = [
            ("E1", o.e[0].is_some()),
            ("E2", o.e[1].is_some()),
            ("G12", o.g[0].is_some()),
            ("nu12", o.nu[0].is_some()),
            ("X1", o.tensile[0].is_some()),
            ("X2", o.tensile[1].is_some()),
            ("S", o.shear.is_some()),
        ];
        missing.extend(
            keys.iter().filter(|(_, ok)| !ok).map(|(k, _)| format!("orthotropy.{k}")),
        );
    }
    Readiness {
        volume_ready: missing.is_empty(),
        orthotropic: record.orthotropy.is_some(),
        missing,
    }
}
