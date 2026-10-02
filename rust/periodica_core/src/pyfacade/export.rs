//====== periodica/rust/periodica_core/src/pyfacade/export.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Export bindings: shader sources (GLSL / HLSL), voxel and mesh writers
//! (SDF / VTK / STL / OBJ) and the Fourier field bake.

// Python fallback: src/periodica/export.py

use super::convert::{to_bounds, PyBounds};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::Path;

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
    crate::export::export_glsl(
        name,
        include_density,
        include_ior,
        include_sss,
        include_caustic,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Write an HLSL shader file for the named material. Returns the output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, properties=None))]
fn py_export_hlsl(name: &str, out_path: &str, properties: Option<Vec<String>>) -> PyResult<String> {
    let path = Path::new(out_path);
    crate::export::export_hlsl(name, path, properties.as_deref())
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Write a raw float32 SDF voxel volume + JSON sidecar. Returns output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, bounds, voxel_size, scale_m=None, mode="phase"))]
fn py_export_sdf_raw(
    name: &str,
    out_path: &str,
    bounds: PyBounds,
    voxel_size: f64,
    scale_m: Option<f64>,
    mode: &str,
) -> PyResult<String> {
    crate::export::export_sdf_raw(
        name,
        Path::new(out_path),
        to_bounds(bounds),
        voxel_size,
        scale_m,
        mode,
    )
    .map(|p| p.to_string_lossy().to_string())
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Write an ASCII VTK STRUCTURED_POINTS file. Returns output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, bounds, voxel_size, properties=None, scale_m=None))]
fn py_export_vtk_legacy(
    name: &str,
    out_path: &str,
    bounds: PyBounds,
    voxel_size: f64,
    properties: Option<Vec<String>>,
    scale_m: Option<f64>,
) -> PyResult<String> {
    let props = properties.unwrap_or_default();
    crate::export::export_vtk_legacy(
        name,
        Path::new(out_path),
        to_bounds(bounds),
        voxel_size,
        &props,
        scale_m,
    )
    .map(|p| p.to_string_lossy().to_string())
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Write a binary or ASCII STL surface mesh. Returns output path.
#[pyfunction]
#[pyo3(signature = (name, out_path, bounds, voxel_size, scale_m=None, binary=true))]
fn py_export_stl(
    name: &str,
    out_path: &str,
    bounds: PyBounds,
    voxel_size: f64,
    scale_m: Option<f64>,
    binary: bool,
) -> PyResult<String> {
    crate::export::export_stl(
        name,
        Path::new(out_path),
        to_bounds(bounds),
        voxel_size,
        scale_m,
        binary,
    )
    .map(|p| p.to_string_lossy().to_string())
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Write OBJ + MTL files. Returns (obj_path, mtl_path).
#[pyfunction]
#[pyo3(signature = (name, obj_path, bounds, voxel_size, scale_m=None))]
fn py_export_obj(
    name: &str,
    obj_path: &str,
    bounds: PyBounds,
    voxel_size: f64,
    scale_m: Option<f64>,
) -> PyResult<(String, String)> {
    crate::export::export_obj(
        name,
        Path::new(obj_path),
        to_bounds(bounds),
        voxel_size,
        scale_m,
    )
    .map(|(o, m)| {
        (
            o.to_string_lossy().to_string(),
            m.to_string_lossy().to_string(),
        )
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    bounds: PyBounds,
    grid_size: (usize, usize, usize),
    truncate_threshold: f64,
) -> PyResult<PyObject> {
    let cfg = crate::fourier_bake::bake_fourier(
        entry_name,
        property,
        bounds,
        grid_size,
        truncate_threshold,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let coeffs = PyList::empty_bound(py);
    for c in &cfg.coefficients {
        let d = PyDict::new_bound(py);
        d.set_item("n", c.n)?;
        d.set_item("m", c.m)?;
        d.set_item("l", c.l)?;
        d.set_item("amplitude", c.amplitude)?;
        d.set_item("phase", c.phase)?;
        coeffs.append(&d)?;
    }
    let out = PyDict::new_bound(py);
    out.set_item("property_name", &cfg.property_name)?;
    out.set_item("base_value", cfg.base_value)?;
    out.set_item(
        "domain_size_m",
        vec![
            cfg.domain_size_m.0,
            cfg.domain_size_m.1,
            cfg.domain_size_m.2,
        ],
    )?;
    out.set_item("coefficients", &coeffs)?;
    out.set_item("boundary_condition", &cfg.boundary_condition)?;
    Ok(out.into())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_export_glsl, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_hlsl, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_sdf_raw, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_vtk_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_stl, m)?)?;
    m.add_function(wrap_pyfunction!(py_export_obj, m)?)?;
    m.add_function(wrap_pyfunction!(py_bake_fourier, m)?)?;
    Ok(())
}
