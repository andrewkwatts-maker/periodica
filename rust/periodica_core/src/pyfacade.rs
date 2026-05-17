//====== periodica/rust/periodica_core/src/pyfacade.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade exposed to Python as the `periodica._periodica_core` extension
//! module. Gated behind the `python` feature so the engine path (no Python
//! runtime) compiles cleanly without PyO3.
//!
//! ## Module init
//!
//! On first import, the module resolves the `periodica/data/active/` path
//! relative to the installed Python package and calls
//! [`data_loader::load_all_tiers`] to eagerly load all 424 datasheets into the
//! process-wide [`data_loader::DATA`] hub.
//!
//! ## API surface
//!
//! Every exported function is the Rust-accelerated twin of the same-named
//! Python function in `periodica._dispatch`. The Python `@rust_accelerated`
//! decorator routes calls here when the wheel is present.

// Python fallback: src/periodica/{get,sample,export}.py

#![cfg(feature = "python")]

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Convert a scope string (e.g. `"atom"`, `"Atom"`) to a [`crate::get::Scope`].
/// Returns `None` for unrecognised strings (caller falls back to no-scope search).
fn str_to_scope(s: &str) -> Option<crate::get::Scope> {
    use crate::get::Scope;
    match s.to_lowercase().trim_end_matches('s') {
        "subatomic" | "subatom" => Some(Scope::SubAtomic),
        "atom"                  => Some(Scope::Atom),
        "molecule"              => Some(Scope::Molecule),
        "alloy"                 => Some(Scope::Alloy),
        "ceramic"               => Some(Scope::Ceramic),
        "composite"             => Some(Scope::Composite),
        "amino_acid"            => Some(Scope::AminoAcid),
        "protein"               => Some(Scope::Protein),
        "cell"                  => Some(Scope::Cell),
        "cell_component"        => Some(Scope::CellComponent),
        "nucleic_acid"          => Some(Scope::NucleicAcid),
        "biomaterial"           => Some(Scope::BiomaterialType),
        _                       => None,
    }
}

/// Convert a `serde_json::Value` to a Python `dict` / `list` / scalar by
/// round-tripping through a JSON string and Python's `json.loads`.
fn value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    let s = serde_json::to_string(val)
        .map_err(|e| PyValueError::new_err(format!("json serialise: {e}")))?;
    let json_mod = py.import_bound("json")?;
    json_mod.call_method1("loads", (s,))?.extract()
}

// ─── exported functions ───────────────────────────────────────────────────────

/// Rust-accelerated twin of `periodica.get.Get`.
///
/// `scope_str` is the optional tier name (e.g. `"atom"`, `"alloy"`) or
/// `None` for cross-tier search. Returns a Python `dict`.
#[pyfunction]
#[pyo3(signature = (spec, scope_str=None))]
fn py_get(py: Python<'_>, spec: &str, scope_str: Option<&str>) -> PyResult<PyObject> {
    assert!(!spec.is_empty(), "py_get: spec must be non-empty");
    let scope = scope_str.and_then(str_to_scope);
    let val = crate::get::Get(spec, scope)
        .map_err(|e| PyKeyError::new_err(e.to_string()))?;
    value_to_py(py, &val)
}

/// Rust-accelerated twin of `periodica.get.Save`.
///
/// `data_json` is a JSON-encoded string (the dict to store);
/// `tier_str` is the tier name (e.g. `"alloy"`).
/// Returns the previous entry dict if one was displaced, else `None`.
#[pyfunction]
fn py_save(
    py: Python<'_>,
    name: &str,
    data_json: &str,
    tier_str: &str,
) -> PyResult<PyObject> {
    assert!(!name.is_empty(), "py_save: name must be non-empty");
    assert!(!tier_str.is_empty(), "py_save: tier_str must be non-empty");
    let data: serde_json::Value = serde_json::from_str(data_json)
        .map_err(|e| PyValueError::new_err(format!("data_json is not valid JSON: {e}")))?;
    let scope = str_to_scope(tier_str)
        .ok_or_else(|| PyValueError::new_err(format!("unknown tier: {tier_str}")))?;
    let prev = crate::get::Save(name, data, scope)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    match prev {
        Some(v) => value_to_py(py, &v),
        None => Ok(py.None()),
    }
}

/// Rust-accelerated twin of `periodica.sample.sample`.
///
/// `at` is `(x, y, z)` or `None`; `scale_m` is a float or `None`.
#[pyfunction]
#[pyo3(signature = (name, property, at=None, scale_m=None))]
fn py_sample(
    name: &str,
    property: &str,
    at: Option<(f64, f64, f64)>,
    scale_m: Option<f64>,
) -> PyResult<f64> {
    assert!(!name.is_empty(), "py_sample: name must be non-empty");
    assert!(!property.is_empty(), "py_sample: property must be non-empty");
    crate::sample::sample(name, property, at, scale_m)
        .map_err(|e| PyKeyError::new_err(e.to_string()))
}

/// Rust-accelerated twin of `periodica.sample.data_sheet`.
///
/// Returns the bulk Properties dict for the named entry.
#[pyfunction]
fn py_data_sheet(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    assert!(!name.is_empty(), "py_data_sheet: name must be non-empty");
    let val = crate::sample::data_sheet(name)
        .map_err(|e| PyKeyError::new_err(e.to_string()))?;
    value_to_py(py, &val)
}

/// Rust-accelerated twin of `periodica.get.list_tiers`.
#[pyfunction]
fn py_list_tiers() -> Vec<String> {
    crate::data_loader::list_tiers()
}

/// Rust-accelerated twin of `periodica.export.export_glsl`.
///
/// Returns a GLSL 450+ source string for the named material.
#[pyfunction]
#[pyo3(signature = (name, include_density=true, include_ior=true, include_sss=true, include_caustic=true))]
fn py_export_glsl(
    name: &str,
    include_density: bool,
    include_ior: bool,
    include_sss: bool,
    include_caustic: bool,
) -> PyResult<String> {
    assert!(!name.is_empty(), "py_export_glsl: name must be non-empty");
    crate::export::export_glsl(name, include_density, include_ior, include_sss, include_caustic)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ─── protein functions ────────────────────────────────────────────────────────

/// Compute Kabsch RMSD (Å) between two Nx3 point sets.
/// `a` and `b` are Python lists of [x, y, z] triples.
#[pyfunction]
fn py_kabsch_rmsd(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<f64> {
    use ndarray::Array2;
    if a.is_empty() || b.is_empty() {
        return Ok(0.0);
    }
    let n = a.len();
    if n != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("a and b must have same length"));
    }
    let mut arr_a = Array2::<f64>::zeros((n, 3));
    let mut arr_b = Array2::<f64>::zeros((n, 3));
    for (i, row) in a.iter().enumerate() {
        if row.len() < 3 { return Err(pyo3::exceptions::PyValueError::new_err("each row must have 3 elements")); }
        arr_a[[i, 0]] = row[0]; arr_a[[i, 1]] = row[1]; arr_a[[i, 2]] = row[2];
    }
    for (i, row) in b.iter().enumerate() {
        if row.len() < 3 { return Err(pyo3::exceptions::PyValueError::new_err("each row must have 3 elements")); }
        arr_b[[i, 0]] = row[0]; arr_b[[i, 1]] = row[1]; arr_b[[i, 2]] = row[2];
    }
    crate::protein::kabsch_rmsd(&arr_a, &arr_b)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Build a protein Cα backbone from phi/psi angles.
/// `sequence` — 1-letter amino acid codes. `phi_psi_deg` — list of (phi, psi) in degrees.
/// Returns list of dicts: {name, res_idx, x, y, z}.
#[pyfunction]
fn py_build_backbone(
    py: Python<'_>,
    sequence: &str,
    phi_psi_deg: Vec<(f64, f64)>,
) -> PyResult<PyObject> {
    let atoms = crate::protein::build_backbone(sequence, &phi_psi_deg)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let list = pyo3::types::PyList::empty_bound(py);
    for a in &atoms {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("name", &a.atom_name)?;
        d.set_item("res_idx", a.residue_index)?;
        d.set_item("x", a.position[0])?;
        d.set_item("y", a.position[1])?;
        d.set_item("z", a.position[2])?;
        list.append(&d)?;
    }
    Ok(list.into())
}

/// Build backbone from a periodica protein datasheet entry name.
#[pyfunction]
fn py_build_backbone_from_entry(py: Python<'_>, entry_name: &str) -> PyResult<PyObject> {
    let atoms = crate::protein::build_backbone_from_entry(entry_name)
        .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
    let list = pyo3::types::PyList::empty_bound(py);
    for a in &atoms {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("name", &a.atom_name)?;
        d.set_item("res_idx", a.residue_index)?;
        d.set_item("x", a.position[0])?;
        d.set_item("y", a.position[1])?;
        d.set_item("z", a.position[2])?;
        list.append(&d)?;
    }
    Ok(list.into())
}

/// Classify (phi_deg, psi_deg) into Ramachandran region.
/// Returns a string: "alpha_helix", "beta_sheet", "left_alpha", "polyproline_ii", or "other".
#[pyfunction]
fn py_ramachandran_region(phi_deg: f64, psi_deg: f64) -> &'static str {
    use crate::protein::{PhiPsi, RamachandranRegion};
    match crate::protein::ramachandran_region(PhiPsi { phi: phi_deg, psi: psi_deg }) {
        RamachandranRegion::AlphaHelix       => "alpha_helix",
        RamachandranRegion::BetaSheet        => "beta_sheet",
        RamachandranRegion::LeftHandedAlpha  => "left_alpha",
        RamachandranRegion::PolyprolineII    => "polyproline_ii",
        RamachandranRegion::Disallowed       => "other",
    }
}

// ─── alloy functions ─────────────────────────────────────────────────────────

/// Rust-accelerated `periodica.optimize.optimize_alloy`.
///
/// `targets` — list of dicts with keys: property (str), min_value (float|None),
///   max_value (float|None), weight (float).
/// Returns list of dicts: {composition, estimated_properties, score}.
#[pyfunction]
#[pyo3(signature = (targets, base, alloying_pool=None, n_candidates=1000, top_k=5, seed=None))]
fn py_optimize_alloy(
    py: Python<'_>,
    targets: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
    base: &str,
    alloying_pool: Option<Vec<String>>,
    n_candidates: usize,
    top_k: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    use crate::alloy::AlloyTarget;

    let rust_targets: Vec<AlloyTarget> = targets.iter().map(|d| {
        let property: String = d.get_item("property")
            .ok().flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let min_value: Option<f64> = d.get_item("min_value")
            .ok().flatten()
            .and_then(|v| v.extract::<f64>().ok());
        let max_value: Option<f64> = d.get_item("max_value")
            .ok().flatten()
            .and_then(|v| v.extract::<f64>().ok());
        let weight: f64 = d.get_item("weight")
            .ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(1.0);
        AlloyTarget { property, min_value, max_value, weight }
    }).collect();

    let pool = alloying_pool.unwrap_or_default();
    let results = crate::alloy::optimize_alloy(&rust_targets, base, &pool, n_candidates, top_k, seed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let list = pyo3::types::PyList::empty_bound(py);
    for c in &results {
        let d = pyo3::types::PyDict::new_bound(py);
        let comp = pyo3::types::PyDict::new_bound(py);
        for (k, v) in &c.composition { comp.set_item(k, v)?; }
        let props = pyo3::types::PyDict::new_bound(py);
        for (k, v) in &c.estimated_properties { props.set_item(k, v)?; }
        d.set_item("composition", &comp)?;
        d.set_item("estimated_properties", &props)?;
        d.set_item("score", c.score)?;
        list.append(&d)?;
    }
    Ok(list.into())
}

// ─── export functions ─────────────────────────────────────────────────────────

/// Write an HLSL shader file for the named material. Returns the output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, properties=None))]
fn py_export_hlsl(name: &str, out_path: &str, properties: Option<Vec<String>>) -> PyResult<String> {
    let path = std::path::Path::new(out_path);
    crate::export::export_hlsl(name, path, properties.as_deref())
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write a raw float32 SDF voxel volume + JSON sidecar. Returns output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, bounds, voxel_size, scale_m=None, mode="phase"))]
fn py_export_sdf_raw(
    name: &str,
    out_path: &str,
    bounds: ((f64,f64,f64),(f64,f64,f64)),
    voxel_size: f64,
    scale_m: Option<f64>,
    mode: &str,
) -> PyResult<String> {
    let (lo, hi) = bounds;
    let b: crate::sample::Bounds = ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2]);
    crate::export::export_sdf_raw(name, std::path::Path::new(out_path), b, voxel_size, scale_m, mode)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write an ASCII VTK STRUCTURED_POINTS file. Returns output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, bounds, voxel_size, properties=None, scale_m=None))]
fn py_export_vtk_legacy(
    name: &str,
    out_path: &str,
    bounds: ((f64,f64,f64),(f64,f64,f64)),
    voxel_size: f64,
    properties: Option<Vec<String>>,
    scale_m: Option<f64>,
) -> PyResult<String> {
    let (lo, hi) = bounds;
    let b: crate::sample::Bounds = ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2]);
    let props = properties.unwrap_or_default();
    crate::export::export_vtk_legacy(name, std::path::Path::new(out_path), b, voxel_size, &props, scale_m)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write a binary or ASCII STL surface mesh. Returns output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, bounds, voxel_size, scale_m=None, binary=true))]
fn py_export_stl(
    name: &str,
    out_path: &str,
    bounds: ((f64,f64,f64),(f64,f64,f64)),
    voxel_size: f64,
    scale_m: Option<f64>,
    binary: bool,
) -> PyResult<String> {
    let (lo, hi) = bounds;
    let b: crate::sample::Bounds = ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2]);
    crate::export::export_stl(name, std::path::Path::new(out_path), b, voxel_size, scale_m, binary)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write OBJ + MTL files. Returns (obj_path, mtl_path).
#[pyfunction]
#[pyo3(signature = (name, obj_path, bounds, voxel_size, scale_m=None))]
fn py_export_obj(
    name: &str,
    obj_path: &str,
    bounds: ((f64,f64,f64),(f64,f64,f64)),
    voxel_size: f64,
    scale_m: Option<f64>,
) -> PyResult<(String, String)> {
    let (lo, hi) = bounds;
    let b: crate::sample::Bounds = ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2]);
    crate::export::export_obj(name, std::path::Path::new(obj_path), b, voxel_size, scale_m)
        .map(|(o, m)| (o.to_string_lossy().to_string(), m.to_string_lossy().to_string()))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// ─── fourier bake ─────────────────────────────────────────────────────────────

/// Bake a material property into a Fourier expansion.
/// Returns a dict: {property_name, base_value, domain_size_m, coefficients, boundary_condition}
/// where coefficients is a list of {n, m, l, amplitude, phase}.
#[pyfunction]
#[pyo3(signature = (entry_name, property, bounds, grid_size, truncate_threshold=0.01))]
fn py_bake_fourier(
    py: Python<'_>,
    entry_name: &str,
    property: &str,
    bounds: ((f64,f64,f64),(f64,f64,f64)),
    grid_size: (usize, usize, usize),
    truncate_threshold: f64,
) -> PyResult<PyObject> {
    let cfg = crate::fourier_bake::bake_fourier(entry_name, property, bounds, grid_size, truncate_threshold)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let coeffs = pyo3::types::PyList::empty_bound(py);
    for c in &cfg.coefficients {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("n", c.n)?; d.set_item("m", c.m)?; d.set_item("l", c.l)?;
        d.set_item("amplitude", c.amplitude)?; d.set_item("phase", c.phase)?;
        coeffs.append(&d)?;
    }
    let out = pyo3::types::PyDict::new_bound(py);
    out.set_item("property_name", &cfg.property_name)?;
    out.set_item("base_value", cfg.base_value)?;
    out.set_item("domain_size_m", vec![cfg.domain_size_m.0, cfg.domain_size_m.1, cfg.domain_size_m.2])?;
    out.set_item("coefficients", &coeffs)?;
    out.set_item("boundary_condition", &cfg.boundary_condition)?;
    Ok(out.into())
}

// ─── sentinels ────────────────────────────────────────────────────────────────

/// Sentinel that the Python facade can probe to confirm the Rust backend
/// loaded successfully.
#[pyfunction]
fn is_rust_backend() -> bool {
    true
}

/// Returns the underlying Rust crate version.
#[pyfunction]
fn version_rust() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ─── module entry point ───────────────────────────────────────────────────────

/// `periodica._periodica_core` module entry point.
///
/// On import, resolves the `periodica/data/active/` data path from the
/// installed Python package and eagerly loads all datasheets into
/// [`crate::data_loader::DATA`].
#[pymodule]
fn _periodica_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Discover the installed package root via `periodica.__file__` so the
    // Rust loader can find `data/active/` regardless of install path.
    let init_result = (|| -> PyResult<()> {
        let pkg_file: String = py
            .import_bound("periodica")?
            .getattr("__file__")?
            .extract()?;
        let data_path = Path::new(&pkg_file)
            .parent()
            .unwrap_or(Path::new("."))
            .join("data")
            .join("active");
        if let Err(e) = crate::data_loader::load_all_tiers(&data_path) {
            // Non-fatal: Python code will fall back to its own loaders.
            eprintln!("periodica_core: data load warning: {e}");
        }
        Ok(())
    })();
    if let Err(e) = init_result {
        eprintln!("periodica_core: init warning: {e}");
    }

    m.add_function(wrap_pyfunction!(py_get, m)?)?;
    m.add_function(wrap_pyfunction!(py_save, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_sheet, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_tiers, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_glsl, m)?)?;
    // protein
    m.add_function(wrap_pyfunction!(py_kabsch_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_backbone, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_backbone_from_entry, m)?)?;
    m.add_function(wrap_pyfunction!(py_ramachandran_region, m)?)?;
    // alloy
    m.add_function(wrap_pyfunction!(py_optimize_alloy, m)?)?;
    // export
    m.add_function(wrap_pyfunction!(py_export_hlsl, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_sdf_raw, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_vtk_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_stl, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_obj, m)?)?;
    // fourier
    m.add_function(wrap_pyfunction!(py_bake_fourier, m)?)?;
    // sentinels
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add("__version__", crate::PERIODICA_CORE_VERSION)?;
    Ok(())
}
