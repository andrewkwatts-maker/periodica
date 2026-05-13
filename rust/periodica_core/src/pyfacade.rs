//====== periodica/rust/periodica_core/src/pyfacade.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade exposed to Python as the `periodica._periodica_core`
//! extension module.
//!
//! Gated behind the `python` feature so the engine path (no Python
//! interpreter on the build host) compiles cleanly without PyO3.
//! Maturin invokes the `_periodica_core` `#[pymodule]` entry point through
//! the `[tool.maturin] module-name = "periodica._periodica_core"` setting in
//! `SubModules/periodica/pyproject.toml`.
//!
//! ## Status
//!
//! Wave-2 minimal viable bindings: just enough to verify the wheel build
//! succeeds and the Python-side `_HAS_RUST` guard flips to True. The full
//! Get/sample/bake_fourier/export_glsl wrapper surface populates as those
//! Rust functions migrate from stub to real implementation.

#![cfg(feature = "python")]

use pyo3::prelude::*;

/// Returns the underlying Rust crate version. Mirrors `periodica.__version__`
/// so users can detect maturin/wheel/python desyncs.
#[pyfunction]
fn version_rust() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Sentinel that the Python facade can probe to confirm the Rust backend
/// loaded successfully.
#[pyfunction]
fn is_rust_backend() -> bool {
    true
}

/// `periodica._periodica_core` module entry point.
#[pymodule]
fn _periodica_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    Ok(())
}
