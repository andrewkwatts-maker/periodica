//! The volume probe: phase fractions over a cube, mixed and calibrated to the bulk record.
//!
//! direct-float: periodica has no Arithma dependency.

use super::error::VolumeError;
use super::grains::GrainField;
use crate::sampling::{splitmix64, HaltonSampler};
use periodica_mat::{mix, MaterialRecord, Phase, PropertyId, Source};

/// Halton points per probe.
pub const PROBE_SAMPLES: u64 = 64;
/// Representative volume edge in mean grain diameters (Kanit et al. 2003).
pub const RVE_GRAIN_DIAMETERS: f64 = 10.0;

/// A cube of material in material coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probe {
    /// Cube centre in metres.
    pub point: [f64; 3],
    /// Half the cube edge in metres.
    pub half_extent_m: f64,
    /// Load direction; ignored until orthotropy lands.
    pub direction: Option<[f64; 3]>,
}

/// One material record with its grains and per-phase records.
#[derive(Debug, Clone)]
pub struct MaterialVolume {
    record: MaterialRecord,
    phase_records: Vec<MaterialRecord>,
    fractions: Vec<f64>,
    grains: Option<GrainField>,
    /// Representative volume edge in metres; probes at least this large return bulk values.
    pub rve_m: f64,
}

impl MaterialVolume {
    /// Builds the volume for `record`; grains exist only when it has both micro data and phases.
    pub fn new(record: &MaterialRecord, seed: u64) -> Self {
        let fractions: Vec<f64> = record.phases.iter().map(|p| p.volume_fraction).collect();
        let micro = record.micro.filter(|_| !record.phases.is_empty());
        let axis = record.texture.and_then(|t| t.axis);
        Self {
            record: record.clone(),
            phase_records: record.phases.iter().map(|p| phase_record(record, p)).collect(),
            grains: micro.map(|m| GrainField::new(&m.grains, axis, &fractions, seed)),
            rve_m: micro.map_or(0.0, |m| RVE_GRAIN_DIAMETERS * m.grains.average_size_m),
            fractions,
        }
    }

    /// The grain field, when the record has one.
    pub fn grains(&self) -> Option<&GrainField> {
        self.grains.as_ref()
    }

    /// The record's bulk values for `props`, mixed and calibrated when the probe is below the RVE.
    pub fn probe(&self, probe: &Probe, props: &[PropertyId]) -> Result<Vec<f64>, VolumeError> {
        validate(probe)?;
        let bulk = props
            .iter()
            .map(|&p| self.record.bulk.get(p).ok_or(VolumeError::MissingProperty(p)))
            .collect::<Result<Vec<f64>, _>>()?;
        let Some(grains) = self.grains.as_ref().filter(|_| 2.0 * probe.half_extent_m < self.rve_m)
        else {
            return Ok(bulk);
        };
        let sampled = fractions_from(&self.count_phases(grains, probe));
        let at_record = normalised(&self.fractions);
        props
            .iter()
            .zip(&bulk)
            .map(|(&p, &b)| {
                let reference = self.mixed(&at_record, p)?;
                let scale = if reference != 0.0 && reference.is_finite() { b / reference } else { 1.0 };
                Ok(self.mixed(&sampled, p)? * scale)
            })
            .collect()
    }

    fn count_phases(&self, grains: &GrainField, probe: &Probe) -> Vec<u64> {
        let sampler = HaltonSampler::new(probe_seed(probe));
        let mut counts = vec![0u64; self.phase_records.len()];
        for i in 0..PROBE_SAMPLES {
            let u = sampler.point(i);
            let p = [0, 1, 2].map(|k| probe.point[k] + (2.0 * u[k] - 1.0) * probe.half_extent_m);
            let phase = grains.grain_at(p).phase.min(counts.len() - 1);
            counts[phase] += 1;
        }
        counts
    }

    fn mixed(&self, fractions: &[f64], prop: PropertyId) -> Result<f64, VolumeError> {
        let entries: Vec<(&MaterialRecord, f64)> =
            self.phase_records.iter().zip(fractions.iter().copied()).collect();
        Ok(mix(&entries, prop)?)
    }
}

fn fractions_from(counts: &[u64]) -> Vec<f64> {
    let n = counts.iter().sum::<u64>() as f64;
    counts.iter().map(|&c| c as f64 / n).collect()
}

fn normalised(fractions: &[f64]) -> Vec<f64> {
    let total: f64 = fractions.iter().sum();
    fractions.iter().map(|f| f / total).collect()
}

fn phase_record(record: &MaterialRecord, phase: &Phase) -> MaterialRecord {
    let mut out = record.clone();
    out.name = format!("{}/{}", record.name, phase.name);
    for (prop, value, _) in phase.properties.iter() {
        out.bulk.set(prop, value, Source::Datasheet);
    }
    out
}

fn validate(probe: &Probe) -> Result<(), VolumeError> {
    if !probe.point.iter().all(|v| v.is_finite()) {
        return Err(VolumeError::BadProbe("point is not finite"));
    }
    if !(probe.half_extent_m.is_finite() && probe.half_extent_m > 0.0) {
        return Err(VolumeError::BadProbe("half extent must be finite and positive"));
    }
    Ok(())
}

fn probe_seed(probe: &Probe) -> u64 {
    let mut s = probe.half_extent_m.to_bits();
    for v in probe.point {
        let mut t = s ^ v.to_bits();
        s = splitmix64(&mut t);
    }
    s
}
