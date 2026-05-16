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
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add("__version__", crate::PERIODICA_CORE_VERSION)?;
    Ok(())
}
