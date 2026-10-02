//====== periodica/rust/periodica_core/src/pyfacade/alloy.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Alloy bindings: composition search (`optimize_alloy`).

// Python fallback: src/periodica/optimize.py

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Read an optional key from a target dict, treating a missing key, a Python
/// `None` and a value of the wrong type alike as absent.
fn opt_item<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> Option<T> {
    d.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<T>().ok())
}

/// Rust-accelerated `periodica.optimize.optimize_alloy`.
///
/// `targets` — list of dicts with keys: property (str), min_value (float|None),
///   max_value (float|None), weight (float).
/// Returns list of dicts: {composition, estimated_properties, score}.
#[pyfunction]
#[pyo3(signature = (targets, base, alloying_pool=None, n_candidates=1000, top_k=5, seed=None))]
fn py_optimize_alloy(
    py: Python<'_>,
    targets: Vec<Bound<'_, PyDict>>,
    base: &str,
    alloying_pool: Option<Vec<String>>,
    n_candidates: usize,
    top_k: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    use crate::alloy::AlloyTarget;

    let rust_targets: Vec<AlloyTarget> = targets
        .iter()
        .map(|d| AlloyTarget {
            property: opt_item::<String>(d, "property").unwrap_or_default(),
            min_value: opt_item::<f64>(d, "min_value"),
            max_value: opt_item::<f64>(d, "max_value"),
            weight: opt_item::<f64>(d, "weight").unwrap_or(1.0),
        })
        .collect();

    let pool = alloying_pool.unwrap_or_default();
    let results =
        crate::alloy::optimize_alloy(&rust_targets, base, &pool, n_candidates, top_k, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let list = PyList::empty_bound(py);
    for c in &results {
        let d = PyDict::new_bound(py);
        let comp = PyDict::new_bound(py);
        for (k, v) in &c.composition {
            comp.set_item(k, v)?;
        }
        let props = PyDict::new_bound(py);
        for (k, v) in &c.estimated_properties {
            props.set_item(k, v)?;
        }
        d.set_item("composition", &comp)?;
        d.set_item("estimated_properties", &props)?;
        d.set_item("score", c.score)?;
        list.append(&d)?;
    }
    Ok(list.into())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_optimize_alloy, m)?)?;
    Ok(())
}
