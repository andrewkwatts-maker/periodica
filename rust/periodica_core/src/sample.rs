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
//! Per-position property sampler -- the Rust fast path behind
//! `periodica.sample(name, prop, at=, scale_m=)`.
//!
//! ## What was wrong before
//!
//! This module read a schema that **no datasheet uses**. It looked for
//! `entry["phase_model"]`, `entry["phase_volume_fractions"]` and
//! `entry["phase_properties"]`. The shipped data nests everything under
//! `Field` and `Properties` instead:
//!
//! ```json
//! { "Properties": { "YoungsModulus_GPa": 200 },
//!   "Field": { "model": "microstructure_voronoi", "macro_scale_m": 1e-4,
//!              "grain_density": 1e15,
//!              "phases": { "ferrite": {"fraction": 0.85, "YoungsModulus_GPa": 195} } } }
//! ```
//!
//! So `sample()` returned `Err` for every real entry -- and `sample.py` wrapped
//! the call in `except Exception: pass`, which made the failure invisible.
//! Python quietly did all the work while the Rust backend appeared to be
//! enabled. That silent fallback is what `PERIODICA_BACKEND=rust` exists to
//! make impossible.
//!
//! ## Field model coverage
//!
//! Of the 953 bundled entries, 924 declare no `Field` block at all and take the
//! plain bulk path; 24 declare `homogeneous`, 3 `anisotropic_axial`, 2
//! `microstructure_voronoi`, and exactly 1 carries a `scale_dependent` block.
//! All five Python models live in [`crate::sample_models`], which also
//! documents the four details that must match CPython bit-for-bit.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::sample_models::{
    AnisotropicAxialModel, BackbonePathModel, HomogeneousModel, MicrostructureVoronoiModel,
    MixtureModel,
};

/// Trait every phase evaluator must satisfy. SOLID-DIP boundary: callers
/// depend on the trait, not on any concrete model.
pub trait FieldModel: Send + Sync {
    /// Evaluate `property` for `entry` at world position `at` and length scale
    /// `scale_m`. `field` is the entry's `Field` block (an empty object when
    /// the entry declares none). Both `at` and `scale_m` are optional, to
    /// mirror the Python signature; missing values mean "bulk".
    fn evaluate(
        &self,
        field: &Value,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        scale_m: Option<f64>,
    ) -> Result<f64>;
}

/// The [`sample`] failures a caller must be able to tell apart.
///
/// The Python facade used to map *every* sampler error to `KeyError`, which the
/// dispatcher treats as "Rust declined, ask Python". So a property the entry
/// simply lacks -- for which Python's `sample()` returns `None` -- cost a
/// silent fallback in `auto` mode and raised `RustBackendUnavailable` in
/// `rust` mode, and a genuine Rust bug was indistinguishable from a decline.
/// Each variant now maps to its own Python outcome (see `pyfacade::py_sample`);
/// any other error is a bug and propagates.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SampleError {
    /// No datasheet resolves under this name in the Rust registry.
    #[error("sample: no datasheet for '{name}': {reason}")]
    UnknownName { name: String, reason: String },
    /// The entry has no value for this property here. Python returns `None`.
    #[error("sample: property '{0}' missing")]
    MissingProperty(String),
    /// A valid request the Rust backend does not implement: a non-numeric
    /// property value, `backbone_path` at a 3D point, or a field model that
    /// was registered from Python at runtime.
    #[error("sample: {0}")]
    Unsupported(String),
}

/// The error for "no numeric value for `property`".
///
/// [`SampleError::MissingProperty`] unless the bulk table holds a non-numeric
/// value under that key: Python returns that raw value (a string, a list),
/// which an `f64` sampler cannot, so that case is
/// [`SampleError::Unsupported`] and the Python path answers it.
pub(crate) fn no_value(entry: &Value, property: &str) -> anyhow::Error {
    match props_of(entry).get(property) {
        Some(v) if !v.is_null() && v.as_f64().is_none() && !v.is_boolean() => anyhow::Error::new(
            SampleError::Unsupported(format!("property '{property}' has a non-numeric value")),
        ),
        _ => anyhow::Error::new(SampleError::MissingProperty(property.to_string())),
    }
}

/// Process-wide registry of named evaluators. Look-up is concurrent and
/// lock-free; insertion is `O(1)` average.
pub static FIELD_MODELS: Lazy<DashMap<String, Arc<dyn FieldModel>>> = Lazy::new(DashMap::new);

/// Register a named field model. Returns the prior evaluator if one existed.
pub fn register_field_model(name: &str, model: Arc<dyn FieldModel>) -> Option<Arc<dyn FieldModel>> {
    FIELD_MODELS.insert(name.to_string(), model)
}

/// Bulk property table of an entry.
///
/// Mirrors `sample.py::_props_of`: prefer the `Properties` object, otherwise
/// promote the entry's own numeric root scalars, so atoms and molecules --
/// which store `Mass_amu` / `Charge_e` at the root -- stay samplable.
fn props_of(entry: &Value) -> Value {
    if let Some(p) = entry.get("Properties") {
        if p.is_object() {
            return p.clone();
        }
    }
    let mut map = serde_json::Map::new();
    if let Some(obj) = entry.as_object() {
        for (k, v) in obj {
            // Booleans are included on purpose. Python promotes root scalars
            // with `isinstance(v, (int, float))`, and in Python `bool` is a
            // subclass of `int`, so `"can_form_disulfide": true` lands in the
            // property dict there. Excluding it here made `data_sheet()`
            // disagree with Python on every amino-acid entry.
            //
            // This mirrors a Python quirk rather than endorsing it. Dropping
            // booleans would be the cleaner design, but `data_sheet()` is a
            // published API, so that is a deliberate breaking change for both
            // implementations at once -- not a unilateral Rust divergence.
            if v.is_number() || v.is_boolean() {
                map.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(map)
}

/// Look a property up in the bulk table.
///
/// Booleans read as 1.0 / 0.0 so a property promoted from a root scalar stays
/// samplable; Python returns `True`, and `True == 1.0` there, so the two agree.
pub(crate) fn bulk_lookup(entry: &Value, property: &str) -> Option<f64> {
    let v = props_of(entry).get(property).cloned()?;
    match v {
        Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
        other => other.as_f64(),
    }
}

/// The entry's `Field` block, or an empty object.
fn field_of(entry: &Value) -> Value {
    match entry.get("Field") {
        Some(f) if f.is_object() => f.clone(),
        // The legacy shape this module used to assume. Kept so hand-written
        // fixtures and any caller that adopted it keep working.
        _ => match entry.get("phase_model").and_then(Value::as_str) {
            Some(m) => serde_json::json!({ "model": m }),
            None => Value::Object(serde_json::Map::new()),
        },
    }
}

/// Public sampler. Returns the property value at the requested position.
///
/// `None` arguments fall back to bulk behaviour, matching Python.
pub fn sample(
    name: &str,
    property: &str,
    at: Option<(f64, f64, f64)>,
    scale_m: Option<f64>,
) -> Result<f64> {
    let entry = lookup_entry(name)?;
    sample_entry(&entry, property, at, scale_m)
}

/// Sample a pre-fetched entry, so a grid walk need not re-resolve the name for
/// every voxel.
pub fn sample_entry(
    entry: &Value,
    property: &str,
    at: Option<(f64, f64, f64)>,
    scale_m: Option<f64>,
) -> Result<f64> {
    ensure_default_models_registered();
    let field = field_of(entry);
    let model_name = field
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("homogeneous");

    let model = FIELD_MODELS.get(model_name).ok_or_else(|| {
        // Never silently return the bulk value. This is a decline rather than
        // a hard error because Python's `register_field_model` can add models
        // at runtime that Rust has never heard of; the Python path either
        // knows the model or raises its own ValueError.
        let mut known: Vec<String> = FIELD_MODELS.iter().map(|k| k.key().clone()).collect();
        known.sort();
        anyhow::Error::new(SampleError::Unsupported(format!(
            "unknown field model {model_name:?}; registered: {known:?}"
        )))
    })?;
    model.evaluate(&field, entry, property, at, scale_m)
}

/// The bulk `Properties` dict for an entry.
pub fn data_sheet(name: &str) -> Result<Value> {
    Ok(props_of(lookup_entry(name)?.as_ref()))
}

fn lookup_entry(name: &str) -> Result<Arc<Value>> {
    // Resolve through `Get`, the one resolution algorithm the Python API
    // uses too, so `sample()` and `data_sheet()` see exactly the entry
    // `Get(name)` returns.
    //
    // There is deliberately no fallback to the materials catalogue (R2-3):
    // those sheets are not registry entries -- `Get` raises UnknownName for
    // them in both languages -- and they carry no `Properties` block, so the
    // fallback turned every catalogue property into a silent `None`. Only an
    // unknown name maps to `SampleError::UnknownName`; every other `Get`
    // failure (a collision, a malformed spec) propagates as itself.
    crate::get::get_shared(crate::get::Spec::Text(name), None).map_err(|e| match e {
        crate::get::GetError::UnknownName { .. } => anyhow::Error::new(SampleError::UnknownName {
            name: name.to_string(),
            reason: e.to_string(),
        }),
        other => anyhow::Error::new(other),
    })
}

/// One-time registration of the built-in models.
fn ensure_default_models_registered() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        register_field_model("homogeneous", Arc::new(HomogeneousModel));
        register_field_model("mixture", Arc::new(MixtureModel));
        register_field_model("anisotropic_axial", Arc::new(AnisotropicAxialModel));
        register_field_model(
            "microstructure_voronoi",
            Arc::new(MicrostructureVoronoiModel),
        );
        register_field_model("backbone_path", Arc::new(BackbonePathModel));
    });
}

/// Names of the registered field models, sorted. Used by diagnostics and by
/// the Python-parity tests.
pub fn registered_models() -> Vec<String> {
    ensure_default_models_registered();
    let mut v: Vec<String> = FIELD_MODELS.iter().map(|k| k.key().clone()).collect();
    v.sort();
    v
}

// ── Voxel grids ───────────────────────────────────────────────────────────

/// Bounds as `((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))` in metres.
pub type Bounds = ([f64; 3], [f64; 3]);

/// Grid of integer phase indices: `-1` = unset/outside, `>= 0` = index into
/// [`VoxelGrid::phase_names`]. Indexing is `grid[ix][iy][iz]`.
#[derive(Debug)]
pub struct VoxelGrid {
    pub data: Vec<Vec<Vec<i32>>>,
    pub phase_names: Vec<String>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

/// Scalar property sampled at every voxel centre. Indexing matches
/// [`VoxelGrid`].
#[derive(Debug)]
pub struct PropertyGrid {
    pub data: Vec<Vec<Vec<f64>>>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

/// Voxel counts along each axis for `bounds` at `voxel_size`.
fn grid_dims(bounds: Bounds, voxel_size: f64) -> (usize, usize, usize) {
    let (lo, hi) = bounds;
    let n = |i: usize| (((hi[i] - lo[i]) / voxel_size).ceil() as usize).max(1);
    (n(0), n(1), n(2))
}

/// Build a phase-index voxel grid by sampling the named entry at each voxel.
/// Mirrors `periodica.export.voxel_phase_map`.
pub fn voxel_phase_map(
    name: &str,
    bounds: Bounds,
    voxel_size: f64,
    scale_m: Option<f64>,
) -> Result<VoxelGrid> {
    if name.is_empty() {
        return Err(anyhow!("voxel_phase_map: name must be non-empty"));
    }
    if !(voxel_size > 0.0) {
        return Err(anyhow!("voxel_phase_map: voxel_size must be positive"));
    }
    ensure_default_models_registered();

    let entry = lookup_entry(name)?;
    let field = field_of(&entry);
    let (lo, _) = bounds;
    let (nx, ny, nz) = grid_dims(bounds, voxel_size);
    let eff_scale = scale_m.unwrap_or(voxel_size);

    // Phase names in declared order, so an index means the same thing here as
    // in the shader that consumes the grid.
    let phases = field.get("phases").and_then(Value::as_object);
    let phase_names: Vec<String> = match phases {
        Some(p) if !p.is_empty() => p.keys().cloned().collect(),
        _ => vec![name.to_string()],
    };

    // A single-phase material is uniform; skip the per-voxel work entirely.
    if phase_names.len() == 1 {
        return Ok(VoxelGrid {
            data: vec![vec![vec![0i32; nz]; ny]; nx],
            phase_names,
            nx,
            ny,
            nz,
        });
    }

    let fractions = crate::sample_models::ordered_fractions(phases.expect("checked above"));
    let macro_scale = field.get("macro_scale_m").and_then(Value::as_f64);
    let g_density = field
        .get("grain_density")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let grain_size = (1.0f64 / g_density.max(1e-12)).max(1e-12).cbrt();

    let index_of: std::collections::HashMap<&str, i32> = phase_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i as i32))
        .collect();

    let mut data = vec![vec![vec![0i32; nz]; ny]; nx];
    for (ix, plane) in data.iter_mut().enumerate().take(nx) {
        let x = lo[0] + (ix as f64 + 0.5) * voxel_size;
        for (iy, row) in plane.iter_mut().enumerate().take(ny) {
            let y = lo[1] + (iy as f64 + 0.5) * voxel_size;
            for (iz, cell) in row.iter_mut().enumerate().take(nz) {
                let z = lo[2] + (iz as f64 + 0.5) * voxel_size;
                // Above the macro scale the material reads as homogeneous, so
                // every voxel is phase 0 -- the same rule the sampler applies.
                let phase = match macro_scale {
                    Some(m) if eff_scale < m => {
                        let key = format!(
                            "({}, {}, {})",
                            crate::sample_models::py_round(x / grain_size) as i64,
                            crate::sample_models::py_round(y / grain_size) as i64,
                            crate::sample_models::py_round(z / grain_size) as i64
                        );
                        crate::sample_models::pick_by_cumulative(
                            &fractions,
                            crate::sample_models::sha1_unit_interval(&key),
                        )
                    }
                    _ => None,
                };
                *cell = phase
                    .and_then(|p| index_of.get(p.as_str()).copied())
                    .unwrap_or(0);
            }
        }
    }

    Ok(VoxelGrid {
        data,
        phase_names,
        nx,
        ny,
        nz,
    })
}

/// Sample a scalar property at every voxel centre.
/// Mirrors `periodica.export.voxel_sample`.
pub fn voxel_sample(
    name: &str,
    property: &str,
    bounds: Bounds,
    voxel_size: f64,
    scale_m: Option<f64>,
) -> Result<PropertyGrid> {
    if name.is_empty() || property.is_empty() {
        return Err(anyhow!(
            "voxel_sample: name and property must both be non-empty"
        ));
    }
    if !(voxel_size > 0.0) {
        return Err(anyhow!("voxel_sample: voxel_size must be positive"));
    }

    // Resolve once, then sample the entry directly: the old code re-resolved
    // the name for every voxel, which dominated the cost of a large grid.
    let entry = lookup_entry(name)?;
    let (lo, _) = bounds;
    let (nx, ny, nz) = grid_dims(bounds, voxel_size);
    let eff_scale = scale_m.unwrap_or(voxel_size);

    let mut data = vec![vec![vec![0.0f64; nz]; ny]; nx];
    for (ix, plane) in data.iter_mut().enumerate().take(nx) {
        let x = lo[0] + (ix as f64 + 0.5) * voxel_size;
        for (iy, row) in plane.iter_mut().enumerate().take(ny) {
            let y = lo[1] + (iy as f64 + 0.5) * voxel_size;
            for (iz, cell) in row.iter_mut().enumerate().take(nz) {
                let z = lo[2] + (iz as f64 + 0.5) * voxel_size;
                *cell =
                    sample_entry(&entry, property, Some((x, y, z)), Some(eff_scale)).unwrap_or(0.0);
            }
        }
    }
    Ok(PropertyGrid { data, nx, ny, nz })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn steel_entry() -> Value {
        // The real shape of derived/alloys/Steel-1018.json.
        json!({
            "Name": "Steel-1018",
            "Properties": {
                "Density_kgm3": 7870.0,
                "YoungsModulus_GPa": 200.0
            },
            "Field": {
                "model": "microstructure_voronoi",
                "macro_scale_m": 1.0e-4,
                "grain_density": 1.0e15,
                "phases": {
                    "ferrite":  {"fraction": 0.85, "YoungsModulus_GPa": 195.0},
                    "pearlite": {"fraction": 0.15, "YoungsModulus_GPa": 220.0}
                }
            }
        })
    }

    #[test]
    fn reads_the_real_properties_block() {
        // The regression this rewrite exists for: the old code looked for bare
        // root scalars and returned Err for every shipped datasheet.
        let e = steel_entry();
        assert_eq!(
            sample_entry(&e, "Density_kgm3", None, None).unwrap(),
            7870.0
        );
    }

    #[test]
    fn entry_without_a_field_block_takes_the_bulk_path() {
        // 924 of 953 bundled entries look like this.
        let e = json!({"Properties": {"Density_kg_m3": 2700.0}});
        assert_eq!(
            sample_entry(&e, "Density_kg_m3", Some((1.0, 2.0, 3.0)), Some(1e-6)).unwrap(),
            2700.0
        );
    }

    #[test]
    fn root_scalars_are_promoted_for_atoms_and_molecules() {
        let e = json!({"Name": "Fe", "Mass_amu": 55.845, "Charge_e": 0});
        assert_eq!(sample_entry(&e, "Mass_amu", None, None).unwrap(), 55.845);
    }

    #[test]
    fn boolean_root_scalars_are_promoted_like_python_does() {
        // Python's `isinstance(v, (int, float))` is True for bools, so they
        // reach the property dict there. Rust must agree or `data_sheet()`
        // diverges on every amino-acid entry.
        let e = json!({"Name": "Cys", "can_form_disulfide": true, "mass": 121.16});
        assert_eq!(
            sample_entry(&e, "can_form_disulfide", None, None).unwrap(),
            1.0
        );
        assert_eq!(sample_entry(&e, "mass", None, None).unwrap(), 121.16);
    }

    fn kind(err: anyhow::Error) -> SampleError {
        err.downcast::<SampleError>()
            .expect("sampler errors the facade maps must be typed")
    }

    #[test]
    fn catalogue_only_names_are_unknown_not_silently_empty() {
        // R2-3: a name the registry does not know used to fall back to the
        // materials catalogue, whose sheets have no `Properties` block, so
        // every property came back as a silent None where Python raises
        // UnknownName.
        let data =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
        if !data.is_dir() {
            return;
        }
        let _g = crate::data_loader::DATA_TEST_LOCK.lock();
        crate::data_loader::load_all_tiers(&data).unwrap();
        let err = sample("Aluminum_6061_T6", "Density_kg_m3", None, None).unwrap_err();
        assert!(matches!(kind(err), SampleError::UnknownName { .. }));
        assert_eq!(sample("Aluminum-6061", "Density_kgm3", None, None).unwrap(), 2700.0);
        // Any other resolution failure is reported as itself, not as an
        // unknown name.
        let err = sample("{H=-2}", "Mass_amu", None, None).unwrap_err();
        assert!(matches!(
            err.downcast_ref::<crate::get::GetError>(),
            Some(crate::get::GetError::InvalidSpec(_))
        ));
    }

    #[test]
    fn missing_property_is_an_error_not_a_zero() {
        let e = json!({"Properties": {"Density_kgm3": 7870.0}});
        let err = sample_entry(&e, "NoSuchProperty", None, None).unwrap_err();
        // Typed, so the Python facade can return None as Python does instead
        // of turning it into a fallback.
        assert_eq!(
            kind(err),
            SampleError::MissingProperty("NoSuchProperty".into())
        );
    }

    #[test]
    fn missing_property_is_typed_in_every_model() {
        let p = "NoSuchProperty";
        let at = Some((1e-5, 2e-5, 3e-5));
        for e in [steel_entry(), fibre_entry()] {
            for (at, scale) in [(None, None), (at, Some(1e-9)), (at, Some(1.0))] {
                let err = sample_entry(&e, p, at, scale).unwrap_err();
                assert_eq!(kind(err), SampleError::MissingProperty(p.into()));
            }
        }
    }

    #[test]
    fn non_numeric_property_is_a_decline_not_a_missing_value() {
        // Python returns the raw string; an f64 sampler cannot, so it must
        // defer rather than claim the property is absent.
        let e = json!({"Properties": {"Grade": "A36", "x": 1.0}});
        let err = sample_entry(&e, "Grade", None, None).unwrap_err();
        assert!(matches!(kind(err), SampleError::Unsupported(_)));
    }

    #[test]
    fn unknown_field_model_errors_and_names_the_alternatives() {
        let e = json!({"Properties": {"x": 1.0}, "Field": {"model": "no_such_model"}});
        let err = sample_entry(&e, "x", None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no_such_model"), "{msg}");
        assert!(msg.contains("homogeneous"), "{msg}");
        // A decline: Python may have registered this model at runtime.
        assert!(matches!(kind(err), SampleError::Unsupported(_)));
    }

    #[test]
    fn legacy_phase_model_key_still_dispatches() {
        let e = json!({"phase_model": "homogeneous", "Properties": {"x": 5.0}});
        assert_eq!(sample_entry(&e, "x", None, None).unwrap(), 5.0);
    }

    #[test]
    fn all_five_python_models_are_registered() {
        assert_eq!(
            registered_models(),
            vec![
                "anisotropic_axial",
                "backbone_path",
                "homogeneous",
                "microstructure_voronoi",
                "mixture",
            ]
        );
    }

    // ── microstructure_voronoi ───────────────────────────────────────────

    #[test]
    fn voronoi_returns_bulk_above_the_macro_scale() {
        let e = steel_entry();
        // 1 mm sampling of a 0.1 mm microstructure sees the bulk.
        let v = sample_entry(&e, "YoungsModulus_GPa", Some((0.0, 0.0, 0.0)), Some(1.0e-3)).unwrap();
        assert_eq!(v, 200.0);
    }

    #[test]
    fn voronoi_returns_bulk_when_no_point_or_scale_is_given() {
        let e = steel_entry();
        assert_eq!(
            sample_entry(&e, "YoungsModulus_GPa", None, None).unwrap(),
            200.0
        );
        assert_eq!(
            sample_entry(&e, "YoungsModulus_GPa", Some((0.0, 0.0, 0.0)), None).unwrap(),
            200.0
        );
    }

    #[test]
    fn voronoi_returns_a_phase_value_below_the_macro_scale() {
        let e = steel_entry();
        let v = sample_entry(&e, "YoungsModulus_GPa", Some((0.0, 0.0, 0.0)), Some(1.0e-6)).unwrap();
        assert!(
            v == 195.0 || v == 220.0,
            "expected a phase modulus, got {v}"
        );
    }

    #[test]
    fn voronoi_is_deterministic_per_cell() {
        let e = steel_entry();
        let at = Some((1.234e-5, -4.0e-6, 7.7e-6));
        let a = sample_entry(&e, "YoungsModulus_GPa", at, Some(1.0e-6)).unwrap();
        let b = sample_entry(&e, "YoungsModulus_GPa", at, Some(1.0e-6)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn voronoi_phase_mix_tracks_declared_fractions() {
        // 85 % ferrite / 15 % pearlite. Walk a grid of distinct grain cells and
        // check the sampled mix is close to the declaration.
        //
        // The pitch MUST come from `grain_size_from_density`, not from a raw
        // cube root: `grain_density` 1e15 is clamped to a 1e-4 m grain, so
        // sampling on a 1e-5 m pitch would revisit ~8 cells and report a wildly
        // skewed mix. That mistake is what this comment exists to prevent.
        let e = steel_entry();
        let grain = crate::sample_models::grain_size_from_density(1.0e15);
        let mut ferrite = 0;
        let mut total = 0;
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..12 {
                    let at = Some((i as f64 * grain, j as f64 * grain, k as f64 * grain));
                    let v = sample_entry(&e, "YoungsModulus_GPa", at, Some(1.0e-9)).unwrap();
                    if v == 195.0 {
                        ferrite += 1;
                    }
                    total += 1;
                }
            }
        }
        let frac = ferrite as f64 / total as f64;
        assert!(
            (0.78..0.92).contains(&frac),
            "ferrite fraction {frac} is far from the declared 0.85"
        );
    }

    #[test]
    fn voronoi_falls_back_to_bulk_when_the_phase_lacks_the_property() {
        let mut e = steel_entry();
        e["Field"]["phases"]["ferrite"] = json!({"fraction": 0.85});
        e["Field"]["phases"]["pearlite"] = json!({"fraction": 0.15});
        let v = sample_entry(&e, "YoungsModulus_GPa", Some((0.0, 0.0, 0.0)), Some(1e-9)).unwrap();
        assert_eq!(v, 200.0, "should fall back to the bulk modulus");
    }

    // ── anisotropic_axial ────────────────────────────────────────────────

    fn fibre_entry() -> Value {
        json!({
            "Properties": {"E": 50.0},
            "Field": {
                "model": "anisotropic_axial",
                "fiber_direction": [1.0, 0.0, 0.0],
                "axial": {"E": 130.0},
                "transverse": {"E": 10.0}
            }
        })
    }

    #[test]
    fn anisotropic_returns_axial_along_the_fibre() {
        let v = sample_entry(&fibre_entry(), "E", Some((1.0, 0.0, 0.0)), None).unwrap();
        assert!((v - 130.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn anisotropic_returns_transverse_across_the_fibre() {
        let v = sample_entry(&fibre_entry(), "E", Some((0.0, 1.0, 0.0)), None).unwrap();
        assert!((v - 10.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn anisotropic_blends_at_45_degrees() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let v = sample_entry(&fibre_entry(), "E", Some((s, s, 0.0)), None).unwrap();
        // cos = 1/sqrt(2): 130*cos + 10*(1-cos)
        let want = 130.0 * s + 10.0 * (1.0 - s);
        assert!((v - want).abs() < 1e-9, "got {v}, want {want}");
    }

    #[test]
    fn anisotropic_at_the_origin_returns_axial() {
        // pnorm == 0 has no direction; Python picks the axial value.
        let v = sample_entry(&fibre_entry(), "E", Some((0.0, 0.0, 0.0)), None).unwrap();
        assert_eq!(v, 130.0);
    }

    #[test]
    fn anisotropic_zero_direction_does_not_produce_nan() {
        let mut e = fibre_entry();
        e["Field"]["fiber_direction"] = json!([0.0, 0.0, 0.0]);
        let v = sample_entry(&e, "E", Some((1.0, 1.0, 1.0)), None).unwrap();
        assert!(v.is_finite(), "got {v}");
    }

    // ── homogeneous with scale_dependent ─────────────────────────────────

    #[test]
    fn scale_dependent_dispatches_below_the_macro_scale() {
        let e = json!({
            "Properties": {"YoungsModulus_GPa": 200.0},
            "Field": {
                "model": "homogeneous",
                "macro_scale_m": 1.0e-3,
                "scale_dependent": {
                    "matrix":      {"fraction": 0.95, "YoungsModulus_GPa": 200.0},
                    "precipitate": {"fraction": 0.05, "YoungsModulus_GPa": 350.0}
                }
            }
        });
        // Coarse sampling sees bulk.
        assert_eq!(
            sample_entry(&e, "YoungsModulus_GPa", Some((0.0, 0.0, 0.0)), Some(1.0)).unwrap(),
            200.0
        );
        // Fine sampling picks one of the declared phases.
        let v = sample_entry(&e, "YoungsModulus_GPa", Some((0.0, 0.0, 0.0)), Some(1e-9)).unwrap();
        assert!(v == 200.0 || v == 350.0, "got {v}");
    }

    // ── voxel grids ──────────────────────────────────────────────────────

    // ── Cross-language parity ────────────────────────────────────────────
    //
    // The expected values below were produced by calling periodica's own
    // Python `sample()` (not a reimplementation of it) on the identical entry
    // and points. They are the contract: if the Rust backend stops agreeing
    // with Python here, the "pure Rust backend" claim is false.

    #[test]
    fn voronoi_matches_python_sample_point_for_point() {
        let e = steel_entry();
        // grain_density 1e15 clamps to a 1e-4 m grain pitch.
        let g = crate::sample_models::grain_size_from_density(1.0e15);
        let cases: &[((i32, i32, i32), f64)] = &[
            ((0, 0, 0), 220.0),
            ((1, 0, 0), 220.0),
            ((1, 2, 3), 195.0),
            ((11, 11, 11), 195.0),
            ((5, 7, 2), 195.0),
            ((-3, 4, -9), 195.0),
            ((2, 2, 2), 220.0),
        ];
        for ((i, j, k), want) in cases {
            let at = Some((*i as f64 * g, *j as f64 * g, *k as f64 * g));
            let got = sample_entry(&e, "YoungsModulus_GPa", at, Some(1.0e-9)).unwrap();
            assert_eq!(
                got, *want,
                "cell ({i}, {j}, {k}): Rust says {got}, CPython says {want}"
            );
        }
    }

    #[test]
    fn scale_dependent_matches_python_sample_point_for_point() {
        // Exercises the float-repr hash path, where Python formats 0.0 as
        // "0.0" and 1e-5 as "1e-05". Getting either wrong changes the digest
        // and therefore the phase.
        let e = json!({
            "Properties": {"E": 200.0},
            "Field": {
                "model": "homogeneous",
                "macro_scale_m": 1.0e-3,
                "scale_dependent": {
                    "matrix":      {"fraction": 0.95, "E": 200.0},
                    "precipitate": {"fraction": 0.05, "E": 350.0}
                }
            }
        });
        let cases: &[((f64, f64, f64), f64)] = &[
            ((0.0, 0.0, 0.0), 350.0),
            ((0.0123, -0.004, 0.07), 200.0),
            ((1e-5, 2e-5, 3e-5), 200.0),
            ((1.0, 2.0, 3.0), 200.0),
            ((-0.5, 0.25, 1e-7), 200.0),
        ];
        for (at, want) in cases {
            let got = sample_entry(&e, "E", Some(*at), Some(1.0e-9)).unwrap();
            assert_eq!(
                got, *want,
                "at {at:?}: Rust says {got}, CPython says {want}"
            );
        }
    }

    #[test]
    fn grid_dims_round_up_and_never_collapse() {
        let b = ([0.0, 0.0, 0.0], [0.005, 0.0025, 0.0]);
        assert_eq!(grid_dims(b, 0.001), (5, 3, 1));
    }

    #[test]
    fn voxel_sample_rejects_bad_arguments() {
        let b = ([0.0; 3], [1.0; 3]);
        assert!(voxel_sample("", "x", b, 0.1, None).is_err());
        assert!(voxel_sample("Steel", "", b, 0.1, None).is_err());
        assert!(voxel_sample("Steel", "x", b, 0.0, None).is_err());
        assert!(voxel_sample("Steel", "x", b, -1.0, None).is_err());
    }

    #[test]
    fn voxel_phase_map_rejects_bad_arguments() {
        let b = ([0.0; 3], [1.0; 3]);
        assert!(voxel_phase_map("", b, 0.1, None).is_err());
        assert!(voxel_phase_map("Steel", b, 0.0, None).is_err());
    }
}
