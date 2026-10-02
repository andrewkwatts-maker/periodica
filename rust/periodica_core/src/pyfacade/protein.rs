//====== periodica/rust/periodica_core/src/pyfacade/protein.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Protein bindings: Kabsch RMSD, Cα backbone construction and Ramachandran
//! classification.

// Python fallback: src/periodica/folding.py

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Copy an Nx3 list of `[x, y, z]` rows into an `ndarray`, rejecting short rows.
fn rows_to_array(rows: &[Vec<f64>]) -> PyResult<ndarray::Array2<f64>> {
    let mut arr = ndarray::Array2::<f64>::zeros((rows.len(), 3));
    for (i, row) in rows.iter().enumerate() {
        if row.len() < 3 {
            return Err(PyValueError::new_err("each row must have 3 elements"));
        }
        arr[[i, 0]] = row[0];
        arr[[i, 1]] = row[1];
        arr[[i, 2]] = row[2];
    }
    Ok(arr)
}

/// Compute Kabsch RMSD (Å) between two Nx3 point sets.
/// `a` and `b` are Python lists of [x, y, z] triples.
#[pyfunction]
fn py_kabsch_rmsd(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<f64> {
    if a.is_empty() || b.is_empty() {
        return Ok(0.0);
    }
    if a.len() != b.len() {
        return Err(PyValueError::new_err("a and b must have same length"));
    }
    let arr_a = rows_to_array(&a)?;
    let arr_b = rows_to_array(&b)?;
    crate::protein::kabsch_rmsd(&arr_a, &arr_b).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Backbone atoms as a Python list of `{name, res_idx, x, y, z}` dicts.
fn atoms_to_py(py: Python<'_>, atoms: &[crate::protein::BackboneAtom]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for a in atoms {
        let d = PyDict::new_bound(py);
        d.set_item("name", &a.atom_name)?;
        d.set_item("res_idx", a.residue_index)?;
        d.set_item("x", a.position[0])?;
        d.set_item("y", a.position[1])?;
        d.set_item("z", a.position[2])?;
        list.append(&d)?;
    }
    Ok(list.into())
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
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    atoms_to_py(py, &atoms)
}

/// Build backbone from a periodica protein datasheet entry name.
#[pyfunction]
fn py_build_backbone_from_entry(py: Python<'_>, entry_name: &str) -> PyResult<PyObject> {
    let atoms = crate::protein::build_backbone_from_entry(entry_name)
        .map_err(|e| PyKeyError::new_err(e.to_string()))?;
    atoms_to_py(py, &atoms)
}

/// Classify (phi_deg, psi_deg) into Ramachandran region.
/// Returns a string: "alpha_helix", "beta_sheet", "left_alpha", "polyproline_ii", or "other".
#[pyfunction]
fn py_ramachandran_region(phi_deg: f64, psi_deg: f64) -> &'static str {
    use crate::protein::{PhiPsi, RamachandranRegion};
    match crate::protein::ramachandran_region(PhiPsi {
        phi: phi_deg,
        psi: psi_deg,
    }) {
        RamachandranRegion::AlphaHelix => "alpha_helix",
        RamachandranRegion::BetaSheet => "beta_sheet",
        RamachandranRegion::LeftHandedAlpha => "left_alpha",
        RamachandranRegion::PolyprolineII => "polyproline_ii",
        RamachandranRegion::Disallowed => "other",
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_kabsch_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_backbone, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_backbone_from_entry, m)?)?;
    m.add_function(wrap_pyfunction!(py_ramachandran_region, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_to_array_rejects_short_rows() {
        assert!(rows_to_array(&[vec![1.0, 2.0]]).is_err());
        let arr = rows_to_array(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(arr.shape(), &[1, 3]);
    }

    #[test]
    fn kabsch_of_identical_sets_is_zero() {
        let pts = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let rmsd = py_kabsch_rmsd(pts.clone(), pts).unwrap();
        assert!(rmsd.abs() < 1e-9);
    }

    #[test]
    fn ramachandran_alpha_helix() {
        assert_eq!(py_ramachandran_region(-57.0, -47.0), "alpha_helix");
    }
}
