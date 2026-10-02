//====== periodica/rust/periodica_core/src/pyfacade/convert.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Rust -> Python value conversion shared by every binding area.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convert a `serde_json::Value` to a Python `dict` / `list` / scalar by
/// round-tripping through a JSON string and Python's `json.loads`.
pub(super) fn value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    let s = serde_json::to_string(val)
        .map_err(|e| PyValueError::new_err(format!("json serialise: {e}")))?;
    let json_mod = py.import_bound("json")?;
    json_mod.call_method1("loads", (s,))?.extract()
}

/// A Python `((x0, y0, z0), (x1, y1, z1))` box as the core's [`crate::sample::Bounds`].
pub(super) type PyBounds = ((f64, f64, f64), (f64, f64, f64));

/// Convert a [`PyBounds`] tuple pair to the core's array form.
pub(super) fn to_bounds(bounds: PyBounds) -> crate::sample::Bounds {
    let (lo, hi) = bounds;
    ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn to_bounds_keeps_axis_order() {
        let b = to_bounds(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)));
        assert_eq!(b, ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]));
    }

    #[test]
    fn value_to_py_round_trips_nested_json() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let v = json!({"a": [1, 2.5, "x"], "b": {"c": null, "d": true}});
            let obj = value_to_py(py, &v).unwrap();
            let back: String = py
                .import_bound("json")
                .unwrap()
                .call_method1("dumps", (obj,))
                .unwrap()
                .extract()
                .unwrap();
            let reparsed: serde_json::Value = serde_json::from_str(&back).unwrap();
            assert_eq!(reparsed, v);
        });
    }
}
