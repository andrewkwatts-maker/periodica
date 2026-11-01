//! TEMPORARY (P0a spike): zero-copy numpy round-trip probes. Removed before commit.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;

/// Address of the first element as Rust sees it, plus the sum (proves we read the data).
#[pyfunction]
fn _probe_input_ptr(a: PyReadonlyArray2<'_, f64>) -> (usize, f64) {
    let v = a.as_array();
    (v.as_ptr() as usize, v.sum())
}

/// Scale in place through a writable borrow.
#[pyfunction]
fn _probe_scale_inplace(mut a: PyReadwriteArray1<'_, f64>, k: f64) {
    a.as_array_mut().mapv_inplace(|x| x * k);
}

/// Hand a Rust-allocated buffer to numpy without copying; return its address too.
#[pyfunction]
fn _probe_output<'py>(py: Python<'py>, n: usize) -> (usize, Bound<'py, PyArray1<f64>>) {
    let v: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
    let ptr = v.as_ptr() as usize;
    let arr = v.into_pyarray_bound(py);
    let _ = arr.readonly();
    (ptr, arr)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_probe_input_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(_probe_scale_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(_probe_output, m)?)?;
    Ok(())
}
