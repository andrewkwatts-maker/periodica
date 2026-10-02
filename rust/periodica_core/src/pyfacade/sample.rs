//====== periodica/rust/periodica_core/src/pyfacade/sample.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Sampling bindings: `sample` and `data_sheet`.

// Python fallback: src/periodica/sample.py

use super::convert::value_to_py;
use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;

/// Rust-accelerated twin of `periodica.sample.sample`.
///
/// `at` is `(x, y, z)` or `None`; `scale_m` is a float or `None`.
#[pyfunction]
#[pyo3(signature = (name, property, at=None, scale_m=None))]
fn py_sample(
    py: Python<'_>,
    name: &str,
    property: &str,
    at: Option<(f64, f64, f64)>,
    scale_m: Option<f64>,
) -> PyResult<PyObject> {
    assert!(!name.is_empty(), "py_sample: name must be non-empty");
    assert!(
        !property.is_empty(),
        "py_sample: property must be non-empty"
    );
    use crate::sample::SampleError;
    // Each sampler outcome maps to what Python's `sample()` would do, so the
    // dispatcher can tell a decline from a bug (see `SampleError`):
    //   MissingProperty -> None, exactly as Python returns;
    //   UnknownName     -> KeyError, a decline (e.g. an entry Saved from Python);
    //   Unsupported     -> NotImplementedError, a decline;
    //   anything else   -> RuntimeError, a Rust bug, which must propagate.
    let value = match crate::sample::sample(name, property, at, scale_m) {
        Ok(v) => v,
        Err(e) => {
            return match e.downcast_ref::<SampleError>() {
                Some(SampleError::MissingProperty(_)) => Ok(py.None()),
                Some(SampleError::UnknownName { .. }) => Err(PyKeyError::new_err(e.to_string())),
                Some(SampleError::Unsupported(_)) => {
                    Err(PyNotImplementedError::new_err(e.to_string()))
                }
                None => Err(PyRuntimeError::new_err(format!("{e:#}"))),
            };
        }
    };

    // Return a Python `int` for integral results.
    //
    // The Python implementation hands back the raw JSON value, so
    // `sample("Steel-1018", "Density_kgm3")` is the *int* 7870 there while the
    // Rust core computes in f64 and would give 7870.0. That difference is
    // visible in the CLI's output and in any caller that formats the result,
    // so the boundary restores the integer form.
    //
    // Tradeoff: a datasheet that stores `200.0` yields a Python float from the
    // Python path and an int from this one. Both compare equal, and the
    // alternative -- threading JSON number types through the whole evaluator
    // just to preserve a literal's spelling -- is not worth it.
    if value.fract() == 0.0 && value.abs() < 9.007_199_254_740_992e15 {
        Ok((value as i64).to_object(py))
    } else {
        Ok(value.to_object(py))
    }
}

/// Rust-accelerated twin of `periodica.sample.data_sheet`.
///
/// Returns the bulk Properties dict for the named entry.
#[pyfunction]
fn py_data_sheet(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    assert!(!name.is_empty(), "py_data_sheet: name must be non-empty");
    let val = crate::sample::data_sheet(name).map_err(|e| PyKeyError::new_err(e.to_string()))?;
    value_to_py(py, &val)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sample, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_sheet, m)?)?;
    Ok(())
}
