//====== periodica/rust/periodica_core/src/pyfacade/mod.rs ======//
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
//! ## Features
//!
//! - `python` -- compiles these bindings (pyo3 + numpy) and links against
//!   libpython, so `cargo test -p periodica_core --features python` can run the
//!   `#[cfg(test)]` tests in this module tree.
//! - `extension-module` -- `python` plus `pyo3/extension-module`, which tells
//!   the linker *not* to link libpython (the host interpreter provides it).
//!   This is what maturin builds the wheel with (see `[tool.maturin]` in
//!   `pyproject.toml`); a test binary cannot link with it on Linux/macOS.
//!
//! ## Module init
//!
//! On first import, the module resolves the `periodica/data/active/` path
//! relative to the installed Python package and calls
//! [`crate::data_loader::load_all_tiers`] to eagerly load all datasheets into
//! the process-wide [`crate::data_loader::DATA`] hub.
//!
//! ## Module map
//!
//! Each submodule owns one area of the Python surface and contributes its
//! functions through a `register` function, so adding an area means adding a
//! file and one `register` call below -- nothing else changes.
//!
//! - [`convert`] -- shared Rust -> Python value conversion.
//! - [`registry`] -- `py_get`, `py_save`, `py_list_tiers`.
//! - [`sample`] -- `py_sample`, `py_data_sheet`.
//! - [`export`] -- GLSL / HLSL / SDF / VTK / STL / OBJ writers and the Fourier bake.
//! - [`protein`] -- Kabsch RMSD, backbone construction, Ramachandran regions.
//! - [`alloy`] -- `py_optimize_alloy`.
//!
//! ## API surface
//!
//! Every exported function is the Rust twin of the same-named Python function
//! in `periodica._dispatch`. The Python `@rust_accelerated` decorator routes
//! calls here when the wheel is present. `src/periodica/_periodica_core.pyi`
//! declares this surface; a drift test keeps the two in sync.

// Python fallback: src/periodica/{get,sample,export,folding,optimize}.py
//
// Feature-gated by `#[cfg(feature = "python")]` on the `mod` item in lib.rs.

// PyO3 0.22 macro expansion calls `unwrap_required_argument` (unsafe) without
// an explicit unsafe block. Allow the lint for this module tree only so the
// deny in lib.rs does not cascade into the generated code.
#![allow(unsafe_op_in_unsafe_fn)]
// `#[pyfunction]` expansion converts every returned `PyErr` into itself
// (`From<PyErr> for PyErr`), which clippy reports on the signature line of
// each binding. Generated code, not ours.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use std::path::Path;

mod alloy;
mod convert;
mod export;
mod numpy_probe;
mod protein;
mod registry;
mod sample;

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

fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add("__version__", crate::PERIODICA_CORE_VERSION)?;
    Ok(())
}

// ─── module entry point ───────────────────────────────────────────────────────

/// Load every datasheet tier from the installed `periodica` package.
///
/// Discovers the package root via `periodica.__file__` so the Rust loader can
/// find `data/active/` regardless of install path. Failures are reported but
/// not fatal: the import must still succeed so the caller can diagnose.
fn load_installed_data(py: Python<'_>) {
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
}

/// Add every exported function and attribute to `m`.
///
/// Split from the `#[pymodule]` entry point so tests can build the module
/// surface without importing the `periodica` package (and loading its data).
fn add_surface(m: &Bound<'_, PyModule>) -> PyResult<()> {
    registry::register(m)?;
    sample::register(m)?;
    export::register(m)?;
    protein::register(m)?;
    alloy::register(m)?;
    numpy_probe::register(m)?;
    register(m)
}

/// `periodica._periodica_core` module entry point.
///
/// On import, resolves the `periodica/data/active/` data path from the
/// installed Python package and eagerly loads all datasheets into
/// [`crate::data_loader::DATA`].
#[pymodule]
fn _periodica_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    load_installed_data(py);
    add_surface(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Python-visible names, in sorted order. Mirrors
    /// `src/periodica/_periodica_core.pyi`; the Python drift test checks the
    /// stub against the built extension, this one checks the registration.
    const EXPECTED: &[&str] = &[
        "__version__",
        "is_rust_backend",
        "py_bake_fourier",
        "py_build_backbone",
        "py_build_backbone_from_entry",
        "py_data_sheet",
        "py_export_glsl",
        "py_export_hlsl",
        "py_export_obj",
        "py_export_sdf_raw",
        "py_export_stl",
        "py_export_vtk_legacy",
        "py_get",
        "py_kabsch_rmsd",
        "py_list_tiers",
        "py_optimize_alloy",
        "py_ramachandran_region",
        "py_sample",
        "py_save",
        "version_rust",
    ];

    #[test]
    fn surface_registers_exactly_the_expected_names() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let m = PyModule::new_bound(py, "_periodica_core").unwrap();
            add_surface(&m).unwrap();
            let mut names: Vec<String> = m
                .dict()
                .keys()
                .iter()
                .map(|k| k.extract::<String>().unwrap())
                // Drop the interpreter-supplied module dunders (`__name__`,
                // `__doc__`, `__spec__`, ...); `__version__` is ours.
                .filter(|k| !k.starts_with("__") || k == "__version__")
                .collect();
            names.sort();
            assert_eq!(names, EXPECTED);
        });
    }

    #[test]
    fn sentinels_report_the_crate() {
        assert!(is_rust_backend());
        assert_eq!(version_rust(), crate::PERIODICA_CORE_VERSION);
    }
}
