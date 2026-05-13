//====== periodica/rust/periodica_core/src/protein.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # protein
//!
//! Protein backbone construction, NeRF placement, Ramachandran classification,
//! and Kabsch RMSD. Mirrors `periodica.folding`.
//!
//! ## SVD blocker
//!
//! TODO(kabsch): Kabsch alignment requires a portable SVD. The plan called
//! for `ndarray-linalg` with the `openblas-static` feature, which fails to
//! build on Windows out of the box (vcpkg / mingw / Fortran prerequisites).
//! Until a pure-Rust SVD backend lands (candidates: `nalgebra`, `linfa-linalg`)
//! [`kabsch_rmsd`] returns an "unimplemented" error.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};

/// Backbone atom record.
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneAtom {
    pub residue_index: u32,
    pub residue_name: String,
    pub atom_name: String,
    /// Angstrom Cartesian coordinates.
    pub position: [f64; 3],
}

/// Phi/psi angle pair (radians).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhiPsi {
    pub phi: f64,
    pub psi: f64,
}

/// Ramachandran allowed-region tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamachandranRegion {
    AlphaHelix,
    BetaSheet,
    LeftHandedAlpha,
    Disallowed,
}

/// Build a backbone atom list from a periodica protein datasheet entry.
pub fn build_backbone_from_entry(_entry_name: &str) -> Result<Vec<BackboneAtom>> {
    Err(anyhow!(
        "protein::build_backbone_from_entry is not yet implemented; \
         pending data_loader walker + sequence-to-residue lookup"
    ))
}

/// Natural Extension of Reference Frames (NeRF) placement of a single atom
/// from three preceding atoms plus a bond length, bond angle, and dihedral.
pub fn nerf_place(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    bond_length_a: f64,
    bond_angle_rad: f64,
    dihedral_rad: f64,
) -> [f64; 3] {
    let (sa, ca) = bond_angle_rad.sin_cos();
    let (sd, cd) = dihedral_rad.sin_cos();
    // Local-frame target.
    let d2 = [
        bond_length_a * ca,
        bond_length_a * sa * cd,
        bond_length_a * sa * sd,
    ];
    // Build the orthonormal frame from (a, b, c).
    let bc = sub3(c, b);
    let ab = sub3(b, a);
    let bc_n = normalize3(bc);
    let n = normalize3(cross3(ab, bc_n));
    let m_x = bc_n;
    let m_y = cross3(n, m_x);
    let m_z = n;
    [
        c[0] + m_x[0] * d2[0] + m_y[0] * d2[1] + m_z[0] * d2[2],
        c[1] + m_x[1] * d2[0] + m_y[1] * d2[1] + m_z[1] * d2[2],
        c[2] + m_x[2] * d2[0] + m_y[2] * d2[1] + m_z[2] * d2[2],
    ]
}

/// Classify a (φ, ψ) pair into a Ramachandran region.
pub fn ramachandran_region(_pair: PhiPsi) -> RamachandranRegion {
    // Placeholder: real classifier reads polygon tables from JSON.
    RamachandranRegion::Disallowed
}

/// Kabsch root-mean-square deviation between two equally-shaped point sets
/// `(N, 3)`. Returns an Angstrom RMSD on success.
///
/// **Currently unimplemented** — see module-level SVD blocker.
pub fn kabsch_rmsd(_a: &Array2<f64>, _b: &Array2<f64>) -> Result<f64> {
    Err(anyhow!(
        "protein::kabsch_rmsd is blocked on a portable SVD backend \
         (ndarray-linalg+openblas-static fails on Windows). \
         TODO: switch to nalgebra::SVD or linfa-linalg."
    ))
}

/// Centroid-aligned RMSD without rotation. Useful as a smoke check while
/// Kabsch is unavailable.
pub fn translation_only_rmsd(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "protein::translation_only_rmsd: shape mismatch ({:?} vs {:?})",
            a.shape(),
            b.shape()
        ));
    }
    if a.nrows() == 0 {
        return Err(anyhow!("protein::translation_only_rmsd: empty input"));
    }
    let centroid_a = mean_row(a);
    let centroid_b = mean_row(b);
    let mut acc = 0.0_f64;
    for r in 0..a.nrows() {
        let mut sq = 0.0_f64;
        for c in 0..3 {
            let d = (a[[r, c]] - centroid_a[c]) - (b[[r, c]] - centroid_b[c]);
            sq += d * d;
        }
        acc += sq;
    }
    Ok((acc / a.nrows() as f64).sqrt())
}

fn mean_row(m: &Array2<f64>) -> Array1<f64> {
    let n = m.nrows() as f64;
    let mut s = Array1::<f64>::zeros(3);
    for r in 0..m.nrows() {
        for c in 0..3 {
            s[c] += m[[r, c]];
        }
    }
    s / n
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn build_backbone_from_entry_returns_unimplemented() {
        assert!(build_backbone_from_entry("Lysozyme").is_err());
    }

    #[test]
    fn nerf_place_round_trip_keeps_bond_length() {
        // Use a non-colinear (a, b, c) triple so the NeRF reference frame is
        // well-defined. Three colinear atoms produce a degenerate frame
        // (cross-product is zero) and can't be used to place a fourth atom.
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [1.5, 1.0, 0.0];
        let placed = nerf_place(a, b, c, 1.5, std::f64::consts::FRAC_PI_2, 0.0);
        let dx = placed[0] - c[0];
        let dy = placed[1] - c[1];
        let dz = placed[2] - c[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!((len - 1.5).abs() < 1e-9, "got {len}");
    }

    #[test]
    fn translation_only_rmsd_zero_for_identity() {
        let a = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = a.clone();
        let r = translation_only_rmsd(&a, &b).unwrap();
        assert!(r < 1e-12);
    }

    #[test]
    fn translation_only_rmsd_invariant_under_translation() {
        let a = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = array![[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]];
        let r = translation_only_rmsd(&a, &b).unwrap();
        assert!(r < 1e-12);
    }

    #[test]
    fn translation_only_rmsd_rejects_shape_mismatch() {
        let a = array![[0.0, 0.0, 0.0]];
        let b = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(translation_only_rmsd(&a, &b).is_err());
    }

    #[test]
    fn kabsch_rmsd_blocked_until_svd_lands() {
        let a = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = a.clone();
        assert!(kabsch_rmsd(&a, &b).is_err());
    }

    #[test]
    fn ramachandran_region_default_path_is_disallowed() {
        let r = ramachandran_region(PhiPsi {
            phi: 0.0,
            psi: 0.0,
        });
        assert_eq!(r, RamachandranRegion::Disallowed);
    }
}
