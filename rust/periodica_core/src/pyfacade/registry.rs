//====== periodica/rust/periodica_core/src/pyfacade/registry.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Registry bindings: `Get`, `Save` and `list_tiers`.

// Python fallback: src/periodica/get.py

use super::convert::value_to_py;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Convert a scope string (e.g. `"atom"`, `"atoms"`, `"Atom"`) to a
/// [`crate::get::Scope`]. Returns `None` for unrecognised strings, and the
/// caller falls back to an unscoped search.
///
/// Accepts both singular and plural spellings because the Python `Scope` enum
/// exposes both (`Scope.Atom` and `Scope.Atoms` alike resolve to `"atoms"`),
/// and because the tier keys themselves are plural.
///
/// The `cell` / `cell_component` / `nucleic_acid` / `biomaterial` arms were
/// removed with the corresponding `Scope` variants: those tiers are not in the
/// registry that `composition_rules.json` declares, so they could never
/// resolve. `None` is the honest answer for them.
fn str_to_scope(s: &str) -> Option<crate::get::Scope> {
    use crate::get::Scope;
    let lowered = s.to_lowercase();
    // Normalise a trailing plural, but keep `subatomic` intact.
    let key = match lowered.as_str() {
        "subatomic" | "subatom" => "subatomic",
        other => other.trim_end_matches('s'),
    };
    match key {
        "subatomic" => Some(Scope::SubAtomic),
        "fundamental" => Some(Scope::Fundamental),
        "atom" => Some(Scope::Atom),
        "molecule" => Some(Scope::Molecule),
        "hadrons_gen" | "hadron_gen" | "hadrons_gen_" => Some(Scope::HadronsGen),
        "isotope" => Some(Scope::Isotope),
        "ion" => Some(Scope::Ion),
        "alloy" => Some(Scope::Alloy),
        "polymer" => Some(Scope::Polymer),
        "ceramic" => Some(Scope::Ceramic),
        "composite" => Some(Scope::Composite),
        "amino_acid" => Some(Scope::AminoAcid),
        "protein" => Some(Scope::Protein),
        _ => None,
    }
}

/// Rust-accelerated twin of `periodica.get.Get`.
///
/// `scope_str` is the optional tier name (e.g. `"atom"`, `"alloy"`) or
/// `None` for cross-tier search. Returns a Python `dict`.
#[pyfunction]
#[pyo3(signature = (spec, scope_str=None))]
fn py_get(py: Python<'_>, spec: &str, scope_str: Option<&str>) -> PyResult<PyObject> {
    assert!(!spec.is_empty(), "py_get: spec must be non-empty");
    let scope = scope_str.and_then(str_to_scope);
    let val = crate::get::Get(spec, scope).map_err(|e| PyKeyError::new_err(e.to_string()))?;
    value_to_py(py, &val)
}

/// Rust-accelerated twin of `periodica.get.Save`.
///
/// `data_json` is a JSON-encoded string (the dict to store);
/// `tier_str` is the tier name (e.g. `"alloy"`).
/// Returns the previous entry dict if one was displaced, else `None`.
#[pyfunction]
fn py_save(py: Python<'_>, name: &str, data_json: &str, tier_str: &str) -> PyResult<PyObject> {
    assert!(!name.is_empty(), "py_save: name must be non-empty");
    assert!(!tier_str.is_empty(), "py_save: tier_str must be non-empty");
    let data: serde_json::Value = serde_json::from_str(data_json)
        .map_err(|e| PyValueError::new_err(format!("data_json is not valid JSON: {e}")))?;
    let scope = str_to_scope(tier_str)
        .ok_or_else(|| PyValueError::new_err(format!("unknown tier: {tier_str}")))?;
    let prev =
        crate::get::Save(name, data, scope).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    match prev {
        Some(v) => value_to_py(py, &v),
        None => Ok(py.None()),
    }
}

/// Rust-accelerated twin of `periodica.get.list_tiers`.
#[pyfunction]
fn py_list_tiers() -> Vec<String> {
    crate::data_loader::list_tiers()
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_get, m)?)?;
    m.add_function(wrap_pyfunction!(py_save, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_tiers, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::get::Scope;

    #[test]
    fn scope_accepts_singular_plural_and_case() {
        assert!(matches!(str_to_scope("atom"), Some(Scope::Atom)));
        assert!(matches!(str_to_scope("Atoms"), Some(Scope::Atom)));
        assert!(matches!(
            str_to_scope("amino_acids"),
            Some(Scope::AminoAcid)
        ));
        assert!(matches!(str_to_scope("subatomic"), Some(Scope::SubAtomic)));
        assert!(matches!(
            str_to_scope("hadrons_gen"),
            Some(Scope::HadronsGen)
        ));
    }

    #[test]
    fn scope_rejects_unregistered_tiers() {
        assert!(str_to_scope("cell").is_none());
        assert!(str_to_scope("biomaterial").is_none());
        assert!(str_to_scope("").is_none());
    }
}
