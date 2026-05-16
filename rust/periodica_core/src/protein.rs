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
//! ## Design decisions
//!
//! Kabsch SVD uses `nalgebra` (pure-Rust, no OpenBLAS). All folding constants
//! (bond lengths, bond angles, Ramachandran bounds) are embedded from
//! `folding_rules.json` version 1 — the same values the Python path reads.
//
// Python fallback: src/periodica/folding.py

use anyhow::{anyhow, Result};
use nalgebra::{Matrix3, Vector3};
use ndarray::{Array1, Array2};

// ─── Bond geometry constants (from folding_rules.json v1) ─────────────────
const BOND_N_CA: f64 = 1.458; // Å
const BOND_CA_C: f64 = 1.525;
const BOND_C_N: f64 = 1.329;
const ANGLE_N_CA_C: f64 = 111.2; // degrees
const ANGLE_CA_C_N: f64 = 116.2;
const ANGLE_C_N_CA: f64 = 121.7;
const OMEGA_DEG: f64 = 180.0;

// ─── Ramachandran region bounds (from folding_rules.json v1, in degrees) ──
// Format: (phi_lo, phi_hi, psi_lo, psi_hi)
const REGION_ALPHA_HELIX: (f64, f64, f64, f64) = (-100.0, -30.0, -80.0, -10.0);
const REGION_BETA_SHEET: (f64, f64, f64, f64) = (-180.0, -90.0, 90.0, 180.0);
const REGION_LEFT_ALPHA: (f64, f64, f64, f64) = (30.0, 90.0, 0.0, 90.0);
const REGION_POLYPROLINE_II: (f64, f64, f64, f64) = (-90.0, -30.0, 110.0, 180.0);

/// Backbone atom record.
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneAtom {
    pub residue_index: u32,
    pub residue_name: String,
    pub atom_name: String,
    /// Angstrom Cartesian coordinates.
    pub position: [f64; 3],
}

/// Phi/psi angle pair — stored as **degrees** to match the JSON datasheets.
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
    PolyprolineII,
    Disallowed,
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Build a backbone atom list from a periodica protein datasheet entry.
///
/// Reads `"residues"` array from the entry (loaded via `data_loader::DATA`),
/// extracts per-residue phi/psi in degrees, then calls `build_backbone`.
pub fn build_backbone_from_entry(entry_name: &str) -> Result<Vec<BackboneAtom>> {
    assert!(!entry_name.is_empty(), "build_backbone_from_entry: name must be non-empty");
    let hub = crate::data_loader::DATA.read();
    let entry = {
        let mut found = None;
        for kv in hub.tiers.iter() {
            if let Some(v) = kv.value().get(entry_name) {
                found = Some(v.value().clone());
                break;
            }
        }
        found.ok_or_else(|| anyhow!("build_backbone_from_entry: entry '{entry_name}' not found"))?
    };

    let residues = entry
        .get("residues")
        .or_else(|| entry.get("Residues"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("build_backbone_from_entry: entry '{entry_name}' has no 'residues' array"))?;

    if residues.is_empty() {
        return Ok(Vec::new());
    }

    let sequence: String = residues
        .iter()
        .map(|r| r.get("residue").and_then(|v| v.as_str()).unwrap_or("X").chars().next().unwrap_or('X'))
        .collect();

    let phi_psi: Vec<(f64, f64)> = residues
        .iter()
        .map(|r| {
            let phi = r.get("phi").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let psi = r.get("psi").and_then(|v| v.as_f64()).unwrap_or(0.0);
            (phi, psi)
        })
        .collect();

    build_backbone(&sequence, &phi_psi)
}

/// NeRF backbone builder. Returns one `BackboneAtom` per (residue, atom) for
/// N, CA, C in each residue — shape analogous to Python's `(n, 3, 3)` array.
///
/// `phi_psi` is in **degrees** (matches folding_rules.json convention).
pub fn build_backbone(sequence: &str, phi_psi: &[(f64, f64)]) -> Result<Vec<BackboneAtom>> {
    assert!(!sequence.is_empty(), "build_backbone: sequence must be non-empty");
    let n = sequence.len().min(phi_psi.len());
    if n == 0 {
        return Ok(Vec::new());
    }

    // Degrees → radians for bond angles (constant)
    let a_nca_c = ANGLE_N_CA_C.to_radians();
    let a_cac_n = ANGLE_CA_C_N.to_radians();
    let a_cn_ca = ANGLE_C_N_CA.to_radians();
    let omega = OMEGA_DEG.to_radians();

    // coords[i] = ([N_xyz], [CA_xyz], [C_xyz])
    let mut n_coords: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut ca_coords: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut c_coords: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    // Place residue 0: N at origin, CA along +x, C in xy-plane.
    n_coords[0] = [0.0, 0.0, 0.0];
    ca_coords[0] = [BOND_N_CA, 0.0, 0.0];
    let ca_c_angle_r = a_nca_c;
    c_coords[0] = [
        BOND_N_CA + BOND_CA_C * (-ca_c_angle_r.cos()),
        BOND_CA_C * ca_c_angle_r.sin(),
        0.0,
    ];

    for i in 1..n {
        let psi_prev = phi_psi[i - 1].1.to_radians();
        let phi_curr = phi_psi[i].0.to_radians();

        // Place N[i] using C[i-1], CA[i-1], C[i-1] reference frame + psi_{i-1}
        n_coords[i] = nerf_place(
            ca_coords[i - 1],
            c_coords[i - 1],
            c_coords[i - 1],
            BOND_C_N,
            a_cac_n,
            psi_prev,
        );
        // Place CA[i]
        ca_coords[i] = nerf_place(
            c_coords[i - 1],
            n_coords[i],
            n_coords[i],
            BOND_N_CA,
            a_cn_ca,
            omega,
        );
        // Place C[i]
        c_coords[i] = nerf_place(
            n_coords[i],
            ca_coords[i],
            ca_coords[i],
            BOND_CA_C,
            a_nca_c,
            phi_curr,
        );
    }

    let chars: Vec<char> = sequence.chars().collect();
    let mut atoms = Vec::with_capacity(n * 3);
    for i in 0..n {
        let res = chars[i].to_string();
        atoms.push(BackboneAtom { residue_index: i as u32, residue_name: res.clone(), atom_name: "N".into(),  position: n_coords[i]  });
        atoms.push(BackboneAtom { residue_index: i as u32, residue_name: res.clone(), atom_name: "CA".into(), position: ca_coords[i] });
        atoms.push(BackboneAtom { residue_index: i as u32, residue_name: res.clone(), atom_name: "C".into(),  position: c_coords[i]  });
    }
    Ok(atoms)
}

/// Classify a (φ, ψ) pair into a Ramachandran region.
/// `pair.phi` and `pair.psi` are in **degrees**.
pub fn ramachandran_region(pair: PhiPsi) -> RamachandranRegion {
    let regions: &[(&str, (f64, f64, f64, f64), RamachandranRegion)] = &[
        ("alpha_helix",     REGION_ALPHA_HELIX,     RamachandranRegion::AlphaHelix),
        ("beta_sheet",      REGION_BETA_SHEET,      RamachandranRegion::BetaSheet),
        ("left_alpha",      REGION_LEFT_ALPHA,      RamachandranRegion::LeftHandedAlpha),
        ("polyproline_ii",  REGION_POLYPROLINE_II,  RamachandranRegion::PolyprolineII),
    ];
    for (_, (phi_lo, phi_hi, psi_lo, psi_hi), region) in regions {
        if pair.phi >= *phi_lo && pair.phi <= *phi_hi
            && pair.psi >= *psi_lo && pair.psi <= *psi_hi
        {
            return *region;
        }
    }
    RamachandranRegion::Disallowed
}

/// True when (φ, ψ) falls inside any allowed region.
pub fn ramachandran_in_allowed(pair: PhiPsi) -> bool {
    !matches!(ramachandran_region(pair), RamachandranRegion::Disallowed)
}

/// Kabsch root-mean-square deviation between two equally-shaped point clouds
/// `(N, 3)`. Returns the Angstrom RMSD after optimal rotation alignment.
///
/// Uses `nalgebra` SVD — pure-Rust, no external LAPACK/OpenBLAS required.
pub fn kabsch_rmsd(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64> {
    assert!(a.ncols() == 3, "kabsch_rmsd: a must be (N, 3)");
    assert!(b.ncols() == 3, "kabsch_rmsd: b must be (N, 3)");
    if a.shape() != b.shape() {
        return Err(anyhow!("kabsch_rmsd: shape mismatch ({:?} vs {:?})", a.shape(), b.shape()));
    }
    let n = a.nrows();
    if n == 0 {
        return Err(anyhow!("kabsch_rmsd: empty input"));
    }

    // Centroids
    let ca = centroid_row(a);
    let cb = centroid_row(b);

    // Centered point clouds as Vec<[f64;3]>
    let a_c = center_rows(a, &ca);
    let b_c = center_rows(b, &cb);

    // Covariance H = A_centered^T @ B_centered (3×3)
    let h = covariance_3x3(&a_c, &b_c);

    // SVD via nalgebra
    let svd = h.svd(true, true);
    let u = svd.u.ok_or_else(|| anyhow!("kabsch_rmsd: SVD did not produce U"))?;
    let vt = svd.v_t.ok_or_else(|| anyhow!("kabsch_rmsd: SVD did not produce V^T"))?;

    // Determinant correction to prevent improper rotation (reflection)
    let det = (vt.transpose() * u.transpose()).determinant();
    let d = if det < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let d_mat = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d));

    // Rotation matrix R = V @ diag(1,1,d) @ U^T
    let rot = vt.transpose() * d_mat * u.transpose();

    // Rotate b_c, compute RMSD
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let bv = Vector3::new(b_c[i][0], b_c[i][1], b_c[i][2]);
        let rb = rot * bv;
        let av = Vector3::new(a_c[i][0], a_c[i][1], a_c[i][2]);
        let diff = av - rb;
        sum_sq += diff.norm_squared();
    }
    Ok((sum_sq / n as f64).sqrt())
}

/// Centroid-aligned RMSD without rotation. Useful as a fast sanity check.
pub fn translation_only_rmsd(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64> {
    assert!(a.ncols() == 3, "translation_only_rmsd: a must be (N, 3)");
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "protein::translation_only_rmsd: shape mismatch ({:?} vs {:?})", a.shape(), b.shape()
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

/// Natural Extension of Reference Frames (NeRF) placement of a single atom.
/// All angles in radians; bond length in Angstroms.
pub fn nerf_place(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    bond_length_a: f64,
    bond_angle_rad: f64,
    dihedral_rad: f64,
) -> [f64; 3] {
    assert!(bond_length_a > 0.0, "nerf_place: bond_length must be positive");
    let (sa, ca) = bond_angle_rad.sin_cos();
    let (sd, cd) = dihedral_rad.sin_cos();
    let d2 = [bond_length_a * ca, bond_length_a * sa * cd, bond_length_a * sa * sd];
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

// ─── Private helpers ────────────────────────────────────────────────────────

fn centroid_row(m: &Array2<f64>) -> [f64; 3] {
    let n = m.nrows() as f64;
    let mut s = [0.0f64; 3];
    for r in 0..m.nrows() {
        for c in 0..3 {
            s[c] += m[[r, c]];
        }
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

fn center_rows(m: &Array2<f64>, centroid: &[f64; 3]) -> Vec<[f64; 3]> {
    (0..m.nrows()).map(|r| [m[[r, 0]] - centroid[0], m[[r, 1]] - centroid[1], m[[r, 2]] - centroid[2]]).collect()
}

fn covariance_3x3(a_c: &[[f64; 3]], b_c: &[[f64; 3]]) -> Matrix3<f64> {
    // H = A^T B
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..a_c.len() {
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += a_c[i][r] * b_c[i][c];
            }
        }
    }
    Matrix3::new(h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2])
}

fn mean_row(m: &Array2<f64>) -> Array1<f64> {
    let n = m.nrows() as f64;
    let mut s = Array1::<f64>::zeros(3);
    for r in 0..m.nrows() {
        for c in 0..3 { s[c] += m[[r, c]]; }
    }
    s / n
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if n == 0.0 { [0.0,0.0,0.0] } else { [v[0]/n, v[1]/n, v[2]/n] }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nerf_place_preserves_bond_length() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [1.5, 1.0, 0.0];
        let placed = nerf_place(a, b, c, 1.5, std::f64::consts::FRAC_PI_2, 0.0);
        let dx = placed[0]-c[0]; let dy = placed[1]-c[1]; let dz = placed[2]-c[2];
        assert!((( dx*dx+dy*dy+dz*dz).sqrt()-1.5).abs()<1e-9);
    }

    #[test]
    fn ramachandran_alpha_helix() {
        let r = ramachandran_region(PhiPsi { phi: -60.0, psi: -40.0 });
        assert_eq!(r, RamachandranRegion::AlphaHelix);
    }

    #[test]
    fn ramachandran_beta_sheet() {
        let r = ramachandran_region(PhiPsi { phi: -120.0, psi: 130.0 });
        assert_eq!(r, RamachandranRegion::BetaSheet);
    }

    #[test]
    fn ramachandran_left_alpha() {
        let r = ramachandran_region(PhiPsi { phi: 60.0, psi: 45.0 });
        assert_eq!(r, RamachandranRegion::LeftHandedAlpha);
    }

    #[test]
    fn ramachandran_polyproline_ii() {
        let r = ramachandran_region(PhiPsi { phi: -60.0, psi: 140.0 });
        assert_eq!(r, RamachandranRegion::PolyprolineII);
    }

    #[test]
    fn ramachandran_disallowed() {
        let r = ramachandran_region(PhiPsi { phi: 0.0, psi: 0.0 });
        assert_eq!(r, RamachandranRegion::Disallowed);
    }

    #[test]
    fn ramachandran_in_allowed_alpha() {
        assert!(ramachandran_in_allowed(PhiPsi { phi: -60.0, psi: -40.0 }));
        assert!(!ramachandran_in_allowed(PhiPsi { phi: 0.0, psi: 0.0 }));
    }

    #[test]
    fn kabsch_rmsd_identity_is_zero() {
        let a = array![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]];
        let r = kabsch_rmsd(&a, &a.clone()).unwrap();
        assert!(r < 1e-10, "got {r}");
    }

    #[test]
    fn kabsch_rmsd_translation_invariant() {
        let a = array![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]];
        let b = array![[5.0,5.0,5.0],[6.0,5.0,5.0],[5.0,6.0,5.0]];
        let r = kabsch_rmsd(&a, &b).unwrap();
        assert!(r < 1e-10, "got {r}");
    }

    #[test]
    fn kabsch_rmsd_shape_mismatch_errors() {
        let a = array![[0.0,0.0,0.0]];
        let b = array![[0.0,0.0,0.0],[1.0,0.0,0.0]];
        assert!(kabsch_rmsd(&a, &b).is_err());
    }

    #[test]
    fn kabsch_rmsd_empty_errors() {
        let a: Array2<f64> = Array2::zeros((0, 3));
        assert!(kabsch_rmsd(&a, &a.clone()).is_err());
    }

    #[test]
    fn build_backbone_five_residues() {
        let seq = "ACDEF";
        let phi_psi: Vec<(f64, f64)> = vec![
            (-60.0, -40.0), (-120.0, 130.0), (-60.0, -40.0), (-60.0, -40.0), (-60.0, -40.0)
        ];
        let atoms = build_backbone(seq, &phi_psi).unwrap();
        // 5 residues × 3 atoms = 15
        assert_eq!(atoms.len(), 15);
        // All positions finite
        for a in &atoms { assert!(a.position.iter().all(|x| x.is_finite())); }
    }

    #[test]
    fn translation_only_rmsd_zero_for_identity() {
        let a = array![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]];
        let r = translation_only_rmsd(&a, &a.clone()).unwrap();
        assert!(r < 1e-12);
    }
}
