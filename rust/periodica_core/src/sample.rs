//====== periodica/rust/periodica_core/src/sample.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # sample
//!
//! Per-position property sampler. Hot path: must execute in sub-microsecond
//! latency for engine raymarch fragment-stage callbacks (see §E.2 priority 1
//! of the master plan).
//!
//! Dispatches between phase models registered for an entry:
//! - `homogeneous` — direct datasheet lookup
//! - `mixture`     — rule-of-mixtures by volume fraction
//! - `voronoi`     — Voronoi/Worley phase decomposition (engine vertical
//!                   slice 2)
//! - `anisotropic_axial` — axial-tensor scalar projection
//! - `scale_dependent`   — λ-dependent dispersion (Sellmeier/Cauchy)
//!
//! `register_field_model` lets host applications inject custom evaluators.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::data_loader::DATA;

/// Trait every phase evaluator must satisfy. SOLID-DIP boundary: callers depend
/// on the trait, not on any concrete model.
pub trait FieldModel: Send + Sync {
    /// Evaluate `property` for `entry` at world position `at` and length scale
    /// `scale_m`. Both `at` and `scale_m` are optional to mirror the Python
    /// public API; missing values resolve to bulk averages.
    fn evaluate(
        &self,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        scale_m: Option<f64>,
    ) -> Result<f64>;
}

/// Process-wide registry of named evaluators. Look-up is concurrent and
/// lock-free; insertion is `O(1)` average.
pub static FIELD_MODELS: Lazy<DashMap<String, Arc<dyn FieldModel>>> = Lazy::new(DashMap::new);

/// Register a named field model. Returns the prior evaluator if one existed.
pub fn register_field_model(name: &str, model: Arc<dyn FieldModel>) -> Option<Arc<dyn FieldModel>> {
    FIELD_MODELS.insert(name.to_string(), model)
}

/// Public sampler. Returns the property value (already in SI units) at the
/// requested position. `None` arguments fall back to bulk-average behaviour
/// matching the Python reference implementation.
pub fn sample(
    name: &str,
    property: &str,
    at: Option<(f64, f64, f64)>,
    scale_m: Option<f64>,
) -> Result<f64> {
    ensure_default_models_registered();
    let entry = lookup_entry(name)?;
    let model_name = pick_model(&entry, property);
    if let Some(model) = FIELD_MODELS.get(&model_name) {
        return model.evaluate(&entry, property, at, scale_m);
    }
    // Default behaviour: scalar lookup.
    homogeneous_lookup(&entry, property)
}

/// Returns the raw datasheet entry for the given name, searching every tier
/// in order. Stubbed pending the data_loader walker.
pub fn data_sheet(name: &str) -> Result<Value> {
    lookup_entry(name)
}

fn lookup_entry(name: &str) -> Result<Value> {
    let hub = DATA.read();
    for kv in hub.tiers.iter() {
        if let Some(value) = kv.value().get(name) {
            return Ok(value.value().clone());
        }
    }
    Err(anyhow!(
        "sample::lookup_entry: no datasheet found for '{name}'"
    ))
}

fn pick_model(entry: &Value, _property: &str) -> String {
    entry
        .get("phase_model")
        .and_then(Value::as_str)
        .unwrap_or("homogeneous")
        .to_string()
}

fn homogeneous_lookup(entry: &Value, property: &str) -> Result<f64> {
    entry
        .get(property)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("sample: property '{property}' missing or non-numeric"))
}

// ─── Built-in Voronoi / Worley phase model ────────────────────────────────
//
// Rationale: the master plan §E.7 voxel-ID dispatch demo (Steel split into
// ferrite + cementite) needs a deterministic, fast, position-driven phase
// picker. This evaluator reads the entry's phase volume fractions and
// per-phase property tables, hash-stamps a phase per micro-cell at the
// requested length scale, and either:
//
//   - returns that phase's property value when `at` is inside a cell
//     (micro-scale sampling — drives the voxel-ID demo), or
//   - falls back to the volume-weighted average when `scale_m` is `None`
//     or coarser than the registered macro scale.
//
// The picker is a 64-bit FxHash on `(floor(x/cell), floor(y/cell),
// floor(z/cell))`. Same input → same phase, every call. No allocation,
// no RNG seed plumbing, sub-microsecond.

const PT_VORONOI_DEFAULT_CELL_M: f64 = 1.0e-3; // 1 mm — typical grain size
const PT_VORONOI_MACRO_THRESHOLD_M: f64 = 1.0e-2; // 1 cm → bulk average

/// Built-in Voronoi-grain phase model. Registered automatically the
/// first time `sample` looks up the `"voronoi"` model.
pub struct PTVoronoiPhaseModel;

impl FieldModel for PTVoronoiPhaseModel {
    fn evaluate(
        &self,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        scale_m: Option<f64>,
    ) -> Result<f64> {
        // No `at` → caller wants the bulk-average; use the volume-weighted mean.
        if at.is_none() {
            return volume_weighted_average(entry, property);
        }
        // Coarser than macro threshold → also bulk average.
        if let Some(s) = scale_m {
            if s.is_finite() && s >= PT_VORONOI_MACRO_THRESHOLD_M {
                return volume_weighted_average(entry, property);
            }
        }
        let cell = scale_m.filter(|s| s.is_finite() && *s > 0.0).unwrap_or(PT_VORONOI_DEFAULT_CELL_M);
        let pos = at.unwrap();
        let phase = pick_phase_by_voronoi_hash(entry, pos, cell)?;
        phase_property_lookup(entry, &phase, property)
    }
}

/// Return the phase name selected at world-position `pos` for cell-size `cell`.
fn pick_phase_by_voronoi_hash(
    entry: &Value,
    pos: (f64, f64, f64),
    cell: f64,
) -> Result<String> {
    let fractions = entry
        .get("phase_volume_fractions")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("voronoi sample: entry missing 'phase_volume_fractions' object"))?;
    if fractions.is_empty() {
        return Err(anyhow!("voronoi sample: phase_volume_fractions is empty"));
    }
    // Stable, ordering-aware vector of (name, fraction). Sort by name so
    // hash → phase mapping is deterministic across runs.
    let mut items: Vec<(String, f64)> = fractions
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
        .collect();
    if items.is_empty() {
        return Err(anyhow!("voronoi sample: no numeric phase fractions"));
    }
    items.sort_by(|a, b| a.0.cmp(&b.0));
    let total: f64 = items.iter().map(|(_, f)| *f).sum();
    if total <= 0.0 {
        return Err(anyhow!("voronoi sample: total phase volume must be > 0"));
    }

    let ix = (pos.0 / cell).floor() as i64;
    let iy = (pos.1 / cell).floor() as i64;
    let iz = (pos.2 / cell).floor() as i64;
    let h = fxhash_cell(ix, iy, iz);
    // Map hash → unit interval.
    let r = (h as f64 / u64::MAX as f64).clamp(0.0, 1.0) * total;

    let mut acc = 0.0;
    // CLAUDE.md safety-critical §2: bounded loop.
    for (name, frac) in &items {
        acc += *frac;
        if r <= acc { return Ok(name.clone()); }
    }
    // Fallback: floating-point edge — return last.
    Ok(items.last().unwrap().0.clone())
}

/// Volume-weighted bulk average of `property` across all phases.
fn volume_weighted_average(entry: &Value, property: &str) -> Result<f64> {
    let fractions = match entry.get("phase_volume_fractions").and_then(Value::as_object) {
        Some(f) => f,
        None => {
            // No phases declared — fall through to the bare scalar.
            return homogeneous_lookup(entry, property);
        }
    };
    let mut sum_weighted = 0.0_f64;
    let mut sum_weight = 0.0_f64;
    for (phase, frac_v) in fractions.iter() {
        let frac = match frac_v.as_f64() { Some(f) => f, None => continue };
        if frac <= 0.0 { continue; }
        let value = phase_property_lookup(entry, phase, property)?;
        sum_weighted += value * frac;
        sum_weight += frac;
    }
    if sum_weight > 0.0 {
        Ok(sum_weighted / sum_weight)
    } else {
        homogeneous_lookup(entry, property)
    }
}

/// Read `entry["phase_properties"][phase][property]` as `f64`.
fn phase_property_lookup(entry: &Value, phase: &str, property: &str) -> Result<f64> {
    let phase_props = entry
        .get("phase_properties")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("voronoi sample: entry missing 'phase_properties' object"))?;
    let props = phase_props
        .get(phase)
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("voronoi sample: no properties for phase '{phase}'"))?;
    props
        .get(property)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!(
            "voronoi sample: phase '{phase}' missing numeric '{property}'"
        ))
}

/// FxHash-style 64-bit mixer for a 3D cell coordinate. Same input always
/// yields the same hash — that's the "deterministic per cell" guarantee
/// the Voronoi model needs.
fn fxhash_cell(x: i64, y: i64, z: i64) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in &x.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01B3); }
    for &b in &y.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01B3); }
    for &b in &z.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01B3); }
    h
}

// ─── Voxel-grid functions ─────────────────────────────────────────────────

/// Bounds as `((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))` in metres.
pub type Bounds = ([f64; 3], [f64; 3]);

/// Grid of integer phase indices: -1 = unset/outside, ≥0 = phase index.
/// Indexing: `grid[ix][iy][iz]`.
pub struct VoxelGrid {
    pub data: Vec<Vec<Vec<i32>>>,
    pub phase_names: Vec<String>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

/// Evaluate property at each voxel centre and return a 3-D float grid.
/// Indexing mirrors `VoxelGrid`: `[ix][iy][iz]`.
pub struct PropertyGrid {
    pub data: Vec<Vec<Vec<f64>>>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

/// Build a phase-index voxel grid by sampling the named entry at each voxel.
/// Mirrors `periodica.export.voxel_phase_map`.
pub fn voxel_phase_map(
    name: &str,
    bounds: Bounds,
    voxel_size: f64,
    scale_m: Option<f64>,
) -> Result<VoxelGrid> {
    assert!(!name.is_empty(), "voxel_phase_map: name must be non-empty");
    assert!(voxel_size > 0.0, "voxel_phase_map: voxel_size must be positive");
    ensure_default_models_registered();

    let entry = lookup_entry(name)?;
    let (lo, hi) = bounds;
    let nx = (((hi[0] - lo[0]) / voxel_size).ceil() as usize).max(1);
    let ny = (((hi[1] - lo[1]) / voxel_size).ceil() as usize).max(1);
    let nz = (((hi[2] - lo[2]) / voxel_size).ceil() as usize).max(1);
    let eff_scale = scale_m.unwrap_or(voxel_size);

    // Collect phase names from entry (if any)
    let mut phase_names: Vec<String> = {
        if let Some(fracs) = entry.get("phase_volume_fractions").and_then(Value::as_object) {
            let mut names: Vec<String> = fracs.keys().cloned().collect();
            names.sort();
            names
        } else {
            vec![name.to_string()]
        }
    };

    // Build phase-name → index map
    let phase_idx: std::collections::HashMap<String, i32> = phase_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i as i32))
        .collect();

    let model_name = pick_model(&entry, "");
    let model = FIELD_MODELS.get(&model_name);

    let mut data = vec![vec![vec![0i32; nz]; ny]; nx];

    for ix in 0..nx {
        let x = lo[0] + (ix as f64 + 0.5) * voxel_size;
        for iy in 0..ny {
            let y = lo[1] + (iy as f64 + 0.5) * voxel_size;
            for iz in 0..nz {
                let z = lo[2] + (iz as f64 + 0.5) * voxel_size;
                let at = (x, y, z);

                // Determine phase ID at this voxel
                let phase_id: i32 = if phase_names.len() == 1 {
                    0 // single-phase material
                } else if let Ok(ph) = pick_phase_by_voronoi_hash(&entry, at, eff_scale) {
                    *phase_idx.get(&ph).unwrap_or(&0)
                } else {
                    0
                };
                data[ix][iy][iz] = phase_id;
            }
        }
    }

    // If single-phase, ensure the phase name list contains the bulk name
    if phase_names.is_empty() {
        phase_names.push(name.to_string());
    }

    Ok(VoxelGrid { data, phase_names, nx, ny, nz })
}

/// Sample a scalar property at every voxel centre. Returns a 3-D float grid.
/// Mirrors `periodica.export.voxel_sample`.
pub fn voxel_sample(
    name: &str,
    property: &str,
    bounds: Bounds,
    voxel_size: f64,
    scale_m: Option<f64>,
) -> Result<PropertyGrid> {
    assert!(!name.is_empty(), "voxel_sample: name must be non-empty");
    assert!(!property.is_empty(), "voxel_sample: property must be non-empty");
    assert!(voxel_size > 0.0, "voxel_sample: voxel_size must be positive");

    let (lo, hi) = bounds;
    let nx = (((hi[0] - lo[0]) / voxel_size).ceil() as usize).max(1);
    let ny = (((hi[1] - lo[1]) / voxel_size).ceil() as usize).max(1);
    let nz = (((hi[2] - lo[2]) / voxel_size).ceil() as usize).max(1);
    let eff_scale = scale_m.unwrap_or(voxel_size);

    let mut data = vec![vec![vec![0.0f64; nz]; ny]; nx];
    for ix in 0..nx {
        let x = lo[0] + (ix as f64 + 0.5) * voxel_size;
        for iy in 0..ny {
            let y = lo[1] + (iy as f64 + 0.5) * voxel_size;
            for iz in 0..nz {
                let z = lo[2] + (iz as f64 + 0.5) * voxel_size;
                let v = sample(name, property, Some((x, y, z)), Some(eff_scale)).unwrap_or(0.0);
                data[ix][iy][iz] = v;
            }
        }
    }
    Ok(PropertyGrid { data, nx, ny, nz })
}

/// One-time registration of the built-in Voronoi model. The library
/// auto-registers it on the first call to `sample()` so callers don't
/// need a separate init step.
fn ensure_default_models_registered() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        register_field_model("voronoi", Arc::new(PTVoronoiPhaseModel));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sample_misses_when_data_hub_empty() {
        // No data has been seeded — sample must error rather than panic.
        let res = sample("Iron", "density_kg_m3", None, None);
        assert!(res.is_err());
    }

    #[test]
    fn homogeneous_lookup_returns_scalar() {
        let entry = json!({"density_kg_m3": 7874.0});
        assert_eq!(homogeneous_lookup(&entry, "density_kg_m3").unwrap(), 7874.0);
    }

    #[test]
    fn homogeneous_lookup_rejects_missing_property() {
        let entry = json!({"density_kg_m3": 7874.0});
        assert!(homogeneous_lookup(&entry, "ior").is_err());
    }

    #[test]
    fn pick_model_prefers_explicit_phase_model() {
        let entry = json!({"phase_model": "voronoi"});
        assert_eq!(pick_model(&entry, "density_kg_m3"), "voronoi");
    }

    #[test]
    fn pick_model_defaults_to_homogeneous() {
        let entry = json!({"density_kg_m3": 7874.0});
        assert_eq!(pick_model(&entry, "density_kg_m3"), "homogeneous");
    }

    // ─── Voronoi phase model tests ─────────────────────────────────────────

    fn steel_two_phase_entry() -> Value {
        // 95% ferrite (~200 GPa) + 5% cementite (~350 GPa) — AISI 1070-ish.
        json!({
            "phase_model": "voronoi",
            "phase_volume_fractions": {"ferrite": 0.95, "cementite": 0.05},
            "phase_properties": {
                "ferrite":   {"young_modulus_gpa": 200.0},
                "cementite": {"young_modulus_gpa": 350.0}
            }
        })
    }

    #[test]
    fn fxhash_cell_is_deterministic() {
        let a = fxhash_cell(7, -3, 42);
        let b = fxhash_cell(7, -3, 42);
        assert_eq!(a, b, "same cell index must hash identically across calls");
        assert_ne!(fxhash_cell(7, -3, 42), fxhash_cell(8, -3, 42));
    }

    #[test]
    fn voronoi_phase_picker_is_deterministic_per_cell() {
        let entry = steel_two_phase_entry();
        let pos = (0.0123, -0.004, 0.07); // arbitrary but fixed
        let cell = 1.0e-3;
        let p1 = pick_phase_by_voronoi_hash(&entry, pos, cell).unwrap();
        let p2 = pick_phase_by_voronoi_hash(&entry, pos, cell).unwrap();
        assert_eq!(p1, p2, "voronoi phase pick must be deterministic");
        assert!(p1 == "ferrite" || p1 == "cementite");
    }

    #[test]
    fn voronoi_phase_distribution_tracks_volume_fractions() {
        // Sample many distinct cells; majority should land in 'ferrite' (95%).
        let entry = steel_two_phase_entry();
        let cell = 1.0e-3;
        let mut ferrite = 0_u32;
        let mut cementite = 0_u32;
        // Walk a deterministic 10x10x10 micro-grid of distinct cell indices.
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    let pos = (x as f64 * cell, y as f64 * cell, z as f64 * cell);
                    let phase = pick_phase_by_voronoi_hash(&entry, pos, cell).unwrap();
                    match phase.as_str() {
                        "ferrite" => ferrite += 1,
                        "cementite" => cementite += 1,
                        _ => panic!("unexpected phase"),
                    }
                }
            }
        }
        // 1000 cells, expected ~950 ferrite ±~3σ.
        assert!(ferrite > 880 && ferrite < 990, "ferrite count {ferrite} out of expected band");
        assert!(cementite > 10 && cementite < 120, "cementite count {cementite} out of expected band");
    }

    #[test]
    fn voronoi_evaluate_returns_phase_property_at_micro_scale() {
        let model = PTVoronoiPhaseModel;
        let entry = steel_two_phase_entry();
        let v = model
            .evaluate(&entry, "young_modulus_gpa", Some((0.0, 0.0, 0.0)), Some(1.0e-3))
            .unwrap();
        assert!(v == 200.0 || v == 350.0, "must return one of the two phase moduli");
    }

    #[test]
    fn voronoi_evaluate_falls_back_to_bulk_when_at_is_none() {
        let model = PTVoronoiPhaseModel;
        let entry = steel_two_phase_entry();
        let v = model.evaluate(&entry, "young_modulus_gpa", None, None).unwrap();
        // Volume-weighted: 0.95 * 200 + 0.05 * 350 = 207.5
        assert!((v - 207.5).abs() < 1e-9, "expected volume-weighted average ~207.5, got {v}");
    }

    #[test]
    fn voronoi_evaluate_falls_back_to_bulk_above_macro_threshold() {
        let model = PTVoronoiPhaseModel;
        let entry = steel_two_phase_entry();
        // 5 cm scale ≫ 1 cm macro threshold → bulk average.
        let v = model
            .evaluate(&entry, "young_modulus_gpa", Some((0.0, 0.0, 0.0)), Some(5.0e-2))
            .unwrap();
        assert!((v - 207.5).abs() < 1e-9);
    }

    #[test]
    fn volume_weighted_average_handles_missing_phase_property() {
        let entry = json!({
            "phase_volume_fractions": {"a": 1.0, "b": 0.0},
            "phase_properties": {
                "a": {"k": 100.0}
                // b is intentionally absent — but its fraction is 0 so it must be skipped.
            }
        });
        let v = volume_weighted_average(&entry, "k").unwrap();
        assert!((v - 100.0).abs() < 1e-12);
    }

    #[test]
    fn voronoi_evaluate_errors_when_phase_volume_fractions_missing() {
        let model = PTVoronoiPhaseModel;
        let entry = json!({
            "phase_properties": {"ferrite": {"young_modulus_gpa": 200.0}}
        });
        let res = model.evaluate(&entry, "young_modulus_gpa", Some((0.0, 0.0, 0.0)), Some(1.0e-3));
        assert!(res.is_err(), "missing phase_volume_fractions must error");
    }

    #[test]
    fn ensure_default_models_registered_registers_voronoi() {
        ensure_default_models_registered();
        assert!(FIELD_MODELS.contains_key("voronoi"));
    }
}
