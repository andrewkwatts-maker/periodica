//====== periodica/rust/periodica-runtime/src/crystal.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # crystal
//!
//! The lattice taxonomy and a periodic crystal SDF.
//!
//! - [`LatticeKind`] is the single lattice-family type for the whole stack. It
//!   converts losslessly to and from [`periodica_mat::CrystalSystem`] for every
//!   crystalline system, so a datasheet's `PrimaryStructure` and an engine's
//!   lattice tag never drift apart.
//! - [`CrystalPrimitive`] evaluates "distance to the nearest atom, minus the
//!   atom radius" for an infinite periodic crystal: negative inside an atom,
//!   positive between atoms.
//!
//! ## Cells
//!
//! Every constructor uses an *orthogonal* conventional cell, which is what
//! lets [`CrystalPrimitive::evaluate`] fold a point into one cell and check
//! only the 27 surrounding cells. Hexagonal close packing therefore uses the
//! orthohexagonal cell `(a, a*sqrt(3), c)` with four atoms rather than the
//! two-atom primitive cell with its 120 degree angle.

use periodica_mat::{CrystalSystem, Lattice};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Crystal-lattice family.
///
/// The common metallic and covalent packings get their own variants; any other
/// periodic structure is `Custom(name)`. The serde form is the default
/// externally-tagged one (`"FCC"`, `{"Custom": "zincblende"}`), so data
/// serialised by earlier engine builds still loads.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LatticeKind {
    /// Face-centred cubic (Cu, Au, Al, gamma-Fe, NaCl).
    FCC,
    /// Body-centred cubic (alpha-Fe, W, Cr, K).
    BCC,
    /// Hexagonal close-packed (Mg, Zn, alpha-Ti, Co).
    HCP,
    /// Diamond cubic (C-diamond, Si, Ge).
    Diamond,
    /// 8-dimensional E8 root lattice (theoretical / specialty).
    E8,
    /// 24-dimensional Leech lattice (specialty).
    Leech,
    /// Any other periodic structure, keyed by name.
    Custom(String),
}

impl LatticeKind {
    /// Canonical lower-case name used for lookup and GLSL identifiers.
    pub fn canonical_name(&self) -> &str {
        match self {
            LatticeKind::FCC => "fcc",
            LatticeKind::BCC => "bcc",
            LatticeKind::HCP => "hcp",
            LatticeKind::Diamond => "diamond",
            LatticeKind::E8 => "e8",
            LatticeKind::Leech => "leech",
            LatticeKind::Custom(name) => name.as_str(),
        }
    }

    /// The lattice family of a datasheet crystal system.
    ///
    /// The four packings with their own variant map onto them; every other
    /// crystalline system becomes `Custom` with its kebab-case name
    /// (`"simple-cubic"`, `"tetragonal"`, ...), which [`Self::crystal_system`]
    /// reads back. Returns `None` for [`CrystalSystem::Amorphous`] (no
    /// lattice) and [`CrystalSystem::Unknown`] (no data).
    pub fn from_crystal_system(system: CrystalSystem) -> Option<Self> {
        let custom = |name: &str| Some(LatticeKind::Custom(name.to_owned()));
        match system {
            CrystalSystem::Fcc => Some(LatticeKind::FCC),
            CrystalSystem::Bcc => Some(LatticeKind::BCC),
            CrystalSystem::Hcp => Some(LatticeKind::HCP),
            CrystalSystem::DiamondCubic => Some(LatticeKind::Diamond),
            CrystalSystem::SimpleCubic => custom("simple-cubic"),
            CrystalSystem::Tetragonal => custom("tetragonal"),
            CrystalSystem::Orthorhombic => custom("orthorhombic"),
            CrystalSystem::Rhombohedral => custom("rhombohedral"),
            CrystalSystem::Monoclinic => custom("monoclinic"),
            CrystalSystem::Triclinic => custom("triclinic"),
            CrystalSystem::Amorphous | CrystalSystem::Unknown => None,
        }
    }

    /// The datasheet crystal system of this lattice family.
    ///
    /// `Custom(name)` is resolved with [`CrystalSystem::parse`], so any
    /// spelling the datasheets use (`"Body-Centered Cubic"`, `"sc"`, ...) is
    /// understood. Returns `None` for the higher-dimensional `E8` and `Leech`
    /// lattices, and for a custom name that is not a 3-D crystal system.
    pub fn crystal_system(&self) -> Option<CrystalSystem> {
        match self {
            LatticeKind::FCC => Some(CrystalSystem::Fcc),
            LatticeKind::BCC => Some(CrystalSystem::Bcc),
            LatticeKind::HCP => Some(CrystalSystem::Hcp),
            LatticeKind::Diamond => Some(CrystalSystem::DiamondCubic),
            LatticeKind::E8 | LatticeKind::Leech => None,
            LatticeKind::Custom(name) => match CrystalSystem::parse(name) {
                // A lattice kind always names a periodic structure.
                CrystalSystem::Unknown | CrystalSystem::Amorphous => None,
                system => Some(system),
            },
        }
    }
}

/// Errors produced when building a crystal primitive.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CrystalError {
    /// Lattice has no basis atoms.
    #[error("crystal must have at least one basis atom")]
    NoBasisAtoms,
    /// Period vector is non-positive (or non-finite) in some dimension.
    #[error("period vector must be positive in all dimensions; got {0:?}")]
    InvalidPeriod([f64; 3]),
    /// [`CrystalPrimitive::from_lattice`] has no orthogonal cell for this
    /// crystal system.
    #[error("no crystal primitive for the {0:?} crystal system")]
    UnsupportedSystem(CrystalSystem),
}

/// A periodic crystal SDF.
///
/// [`Self::evaluate`] returns the distance to the nearest atom centre minus
/// [`Self::atom_radius`], so atoms are the negative region.
#[derive(Debug, Clone, PartialEq)]
pub struct CrystalPrimitive {
    /// Lattice family.
    pub lattice_kind: LatticeKind,
    /// Atom positions inside one fundamental cell, in cell-relative
    /// (fractional) coordinates.
    pub basis_atoms: Vec<[f64; 3]>,
    /// Orthogonal cell edge lengths `(a, b, c)` in metres.
    pub periods: [f64; 3],
    /// Atom radius in metres (for distance evaluation).
    pub atom_radius: f64,
}

impl CrystalPrimitive {
    /// Construct a crystal primitive with parameter validation.
    ///
    /// The atom radius is clamped to at least `1e-12` m so the SDF always has
    /// a non-empty negative region.
    pub fn new(
        lattice_kind: LatticeKind,
        basis_atoms: Vec<[f64; 3]>,
        periods: [f64; 3],
        atom_radius: f64,
    ) -> Result<Self, CrystalError> {
        if basis_atoms.is_empty() {
            return Err(CrystalError::NoBasisAtoms);
        }
        if periods.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err(CrystalError::InvalidPeriod(periods));
        }
        let atom_radius = atom_radius.max(1.0e-12);
        Ok(Self {
            lattice_kind,
            basis_atoms,
            periods,
            atom_radius,
        })
    }

    /// Build the primitive for a datasheet unit cell.
    ///
    /// Supported systems and the cell edge they read from `lattice.abc_m`:
    ///
    /// | system | constructor | edges used |
    /// |---|---|---|
    /// | FCC | [`make_fcc`] | `a` |
    /// | BCC | [`make_bcc`] | `a` |
    /// | diamond cubic | [`make_diamond`] | `a` |
    /// | simple cubic | one atom per cell, `Custom("simple-cubic")` | `a` |
    /// | HCP | [`make_hcp`] | `a`, `c` |
    ///
    /// Any other system returns [`CrystalError::UnsupportedSystem`]; a missing
    /// (zero) edge returns [`CrystalError::InvalidPeriod`]. A missing HCP `c`
    /// is reported rather than replaced by the ideal ratio: real HCP metals
    /// deviate from it (Zn ~1.86, Ti ~1.59) and the datasheets carry `c`.
    pub fn from_lattice(lattice: &Lattice, atom_radius: f64) -> Result<Self, CrystalError> {
        let [a, _, c] = lattice.abc_m;
        match lattice.system {
            CrystalSystem::Fcc => make_fcc(a, atom_radius),
            CrystalSystem::Bcc => make_bcc(a, atom_radius),
            CrystalSystem::DiamondCubic => make_diamond(a, atom_radius),
            CrystalSystem::Hcp => make_hcp(a, c, atom_radius),
            CrystalSystem::SimpleCubic => CrystalPrimitive::new(
                LatticeKind::Custom("simple-cubic".to_owned()),
                vec![[0.0, 0.0, 0.0]],
                [a, a, a],
                atom_radius,
            ),
            other => Err(CrystalError::UnsupportedSystem(other)),
        }
    }

    /// Evaluate the SDF at `point` (in metres). Folds the point to the
    /// fundamental cell, computes distance to nearest basis atom in 27
    /// neighbouring cells, returns `dist - atom_radius`.
    pub fn evaluate(&self, point: [f64; 3]) -> f64 {
        let cell = [
            modulo_positive(point[0], self.periods[0]),
            modulo_positive(point[1], self.periods[1]),
            modulo_positive(point[2], self.periods[2]),
        ];
        let mut min_d_sq = f64::MAX;
        // The cell is orthogonal and the folded point lies inside it, so the
        // nearest atom image is always in one of the 27 surrounding cells.
        // Bounded loop: 27 * basis_atoms.len() iterations.
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    for atom in &self.basis_atoms {
                        let ax = (atom[0] + f64::from(dx)) * self.periods[0];
                        let ay = (atom[1] + f64::from(dy)) * self.periods[1];
                        let az = (atom[2] + f64::from(dz)) * self.periods[2];
                        let d_sq = (cell[0] - ax).powi(2)
                            + (cell[1] - ay).powi(2)
                            + (cell[2] - az).powi(2);
                        if d_sq < min_d_sq {
                            min_d_sq = d_sq;
                        }
                    }
                }
            }
        }
        min_d_sq.sqrt() - self.atom_radius
    }
}

/// Modulo for f64 that always returns a non-negative result.
#[inline]
fn modulo_positive(x: f64, period: f64) -> f64 {
    let r = x % period;
    if r < 0.0 {
        r + period
    } else {
        r
    }
}

/// Construct an FCC primitive given a lattice parameter and atom radius.
///
/// Four atoms per cubic cell; nearest-neighbour distance `a / sqrt(2)`.
pub fn make_fcc(a: f64, atom_radius: f64) -> Result<CrystalPrimitive, CrystalError> {
    CrystalPrimitive::new(
        LatticeKind::FCC,
        vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        [a, a, a],
        atom_radius,
    )
}

/// Construct a BCC primitive given a lattice parameter and atom radius.
///
/// Two atoms per cubic cell; nearest-neighbour distance `a * sqrt(3) / 2`.
pub fn make_bcc(a: f64, atom_radius: f64) -> Result<CrystalPrimitive, CrystalError> {
    CrystalPrimitive::new(
        LatticeKind::BCC,
        vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        [a, a, a],
        atom_radius,
    )
}

/// Construct a diamond-cubic primitive given a lattice parameter and atom
/// radius.
///
/// Eight atoms per cubic cell: the FCC sites plus the same set displaced by
/// `(1/4, 1/4, 1/4)`. Each atom has four tetrahedral neighbours at
/// `a * sqrt(3) / 4`.
pub fn make_diamond(a: f64, atom_radius: f64) -> Result<CrystalPrimitive, CrystalError> {
    CrystalPrimitive::new(
        LatticeKind::Diamond,
        vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        [a, a, a],
        atom_radius,
    )
}

/// Construct a hexagonal close-packed primitive from the in-plane spacing
/// `a`, the stacking period `c` and an atom radius.
///
/// Uses the orthohexagonal cell `(a, a*sqrt(3), c)`: x along a basal lattice
/// vector, y perpendicular to it in the basal plane, z along the c axis. The
/// four atoms are the A layer at `(0, 0, 0)` and `(1/2, 1/2, 0)` and the
/// B layer at `(1/2, 1/6, 1/2)` and `(0, 2/3, 1/2)` (fractional).
///
/// At the ideal ratio `c / a = sqrt(8/3)` every atom has twelve nearest
/// neighbours at distance `a`, the same coordination as FCC.
pub fn make_hcp(a: f64, c: f64, atom_radius: f64) -> Result<CrystalPrimitive, CrystalError> {
    CrystalPrimitive::new(
        LatticeKind::HCP,
        vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 1.0 / 6.0, 0.5],
            [0.0, 2.0 / 3.0, 0.5],
        ],
        [a, a * 3.0_f64.sqrt(), c],
        atom_radius,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Transplanted from the engine (pt_periodica_sdf.rs) ----------------

    #[test]
    fn empty_basis_rejected() {
        assert!(matches!(
            CrystalPrimitive::new(LatticeKind::FCC, vec![], [1.0, 1.0, 1.0], 0.1),
            Err(CrystalError::NoBasisAtoms)
        ));
    }

    #[test]
    fn zero_period_rejected() {
        let r = CrystalPrimitive::new(LatticeKind::FCC, vec![[0.0; 3]], [0.0, 1.0, 1.0], 0.1);
        assert!(matches!(r, Err(CrystalError::InvalidPeriod(_))));
    }

    #[test]
    fn modulo_is_always_non_negative() {
        assert_eq!(modulo_positive(-0.3, 1.0), 0.7);
        assert_eq!(modulo_positive(1.7, 1.0), 0.7);
    }

    #[test]
    fn fcc_at_atom_position_is_negative() {
        let c = make_fcc(1.0, 0.1).unwrap();
        let d = c.evaluate([0.0, 0.0, 0.0]);
        // Right at an atom, distance = -atom_radius = -0.1.
        assert!((d + 0.1).abs() < 1e-9, "expected -0.1, got {d}");
    }

    #[test]
    fn bcc_centre_atom_is_negative() {
        let c = make_bcc(1.0, 0.1).unwrap();
        let d = c.evaluate([0.5, 0.5, 0.5]);
        assert!((d + 0.1).abs() < 1e-9, "expected -0.1, got {d}");
    }

    #[test]
    fn fcc_far_from_atom_is_positive() {
        // 1/4-cell offset in all axes -- definitely outside any atom radius=0.05.
        let c = make_fcc(1.0, 0.05).unwrap();
        let d = c.evaluate([0.25, 0.0, 0.0]);
        assert!(d > 0.0, "expected positive distance, got {d}");
    }

    // --- Lattice taxonomy --------------------------------------------------

    const ALL_SYSTEMS: [CrystalSystem; 12] = [
        CrystalSystem::Bcc,
        CrystalSystem::Fcc,
        CrystalSystem::Hcp,
        CrystalSystem::SimpleCubic,
        CrystalSystem::DiamondCubic,
        CrystalSystem::Tetragonal,
        CrystalSystem::Orthorhombic,
        CrystalSystem::Rhombohedral,
        CrystalSystem::Monoclinic,
        CrystalSystem::Triclinic,
        CrystalSystem::Amorphous,
        CrystalSystem::Unknown,
    ];

    #[test]
    fn lattice_canonical_names() {
        assert_eq!(LatticeKind::FCC.canonical_name(), "fcc");
        assert_eq!(LatticeKind::BCC.canonical_name(), "bcc");
        assert_eq!(LatticeKind::HCP.canonical_name(), "hcp");
        assert_eq!(LatticeKind::Diamond.canonical_name(), "diamond");
        assert_eq!(LatticeKind::E8.canonical_name(), "e8");
        assert_eq!(LatticeKind::Leech.canonical_name(), "leech");
        assert_eq!(
            LatticeKind::Custom("zincblende".into()).canonical_name(),
            "zincblende"
        );
    }

    #[test]
    fn every_crystalline_system_round_trips_through_lattice_kind() {
        for system in ALL_SYSTEMS {
            match LatticeKind::from_crystal_system(system) {
                Some(kind) => assert_eq!(kind.crystal_system(), Some(system), "{kind:?}"),
                None => assert!(
                    matches!(system, CrystalSystem::Amorphous | CrystalSystem::Unknown),
                    "{system:?} should map to a lattice kind"
                ),
            }
        }
        assert_eq!(
            LatticeKind::from_crystal_system(CrystalSystem::Bcc),
            Some(LatticeKind::BCC)
        );
        assert_eq!(
            LatticeKind::from_crystal_system(CrystalSystem::DiamondCubic),
            Some(LatticeKind::Diamond)
        );
    }

    #[test]
    fn custom_names_resolve_through_datasheet_spellings() {
        let custom = |s: &str| LatticeKind::Custom(s.into()).crystal_system();
        assert_eq!(custom("Body-Centered Cubic"), Some(CrystalSystem::Bcc));
        assert_eq!(custom("sc"), Some(CrystalSystem::SimpleCubic));
        assert_eq!(custom("zincblende"), None);
        assert_eq!(custom("glass"), None, "amorphous is not a lattice");
        assert_eq!(LatticeKind::E8.crystal_system(), None);
        assert_eq!(LatticeKind::Leech.crystal_system(), None);
    }

    #[test]
    fn serde_form_is_externally_tagged_and_round_trips() {
        // The engine persisted this enum with default serde derives; the
        // upstream type must read that data unchanged.
        let fcc = serde_json::to_string(&LatticeKind::FCC).unwrap();
        assert_eq!(fcc, "\"FCC\"");
        let custom = LatticeKind::Custom("zincblende".into());
        let s = serde_json::to_string(&custom).unwrap();
        assert_eq!(s, r#"{"Custom":"zincblende"}"#);
        for kind in [
            LatticeKind::FCC,
            LatticeKind::BCC,
            LatticeKind::HCP,
            LatticeKind::Diamond,
            LatticeKind::E8,
            LatticeKind::Leech,
            custom,
        ] {
            let back: LatticeKind =
                serde_json::from_str(&serde_json::to_string(&kind).unwrap()).unwrap();
            assert_eq!(back, kind);
        }
    }

    // --- Lattice geometry --------------------------------------------------

    /// Sorted distances from basis atom `atom` to every other atom image in
    /// the surrounding 5x5x5 block of cells (enough for the first two shells).
    fn neighbour_distances(c: &CrystalPrimitive, atom: usize) -> Vec<f64> {
        let origin = c.basis_atoms[atom];
        let mut out = Vec::new();
        for i in -2..=2i32 {
            for j in -2..=2i32 {
                for k in -2..=2i32 {
                    for b in &c.basis_atoms {
                        let d = [
                            (b[0] + f64::from(i) - origin[0]) * c.periods[0],
                            (b[1] + f64::from(j) - origin[1]) * c.periods[1],
                            (b[2] + f64::from(k) - origin[2]) * c.periods[2],
                        ];
                        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if r > 1e-12 {
                            out.push(r);
                        }
                    }
                }
            }
        }
        out.sort_by(f64::total_cmp);
        out
    }

    /// Assert every basis atom has exactly `count` nearest neighbours at
    /// `expected`, and that the next shell is clearly further out.
    fn assert_first_shell(c: &CrystalPrimitive, count: usize, expected: f64) {
        for atom in 0..c.basis_atoms.len() {
            let d = neighbour_distances(c, atom);
            for (n, r) in d.iter().take(count).enumerate() {
                assert!(
                    (r - expected).abs() < 1e-12,
                    "{:?} atom {atom} neighbour {n}: {r} != {expected}",
                    c.lattice_kind
                );
            }
            assert!(
                d[count] > expected * 1.05,
                "{:?} atom {atom}: shell 2 at {} too close to {expected}",
                c.lattice_kind,
                d[count]
            );
        }
    }

    #[test]
    fn fcc_nearest_neighbours_are_twelve_at_a_over_root_two() {
        let a = 3.615e-10; // Cu
        assert_first_shell(&make_fcc(a, 1e-11).unwrap(), 12, a / 2.0_f64.sqrt());
    }

    #[test]
    fn bcc_nearest_neighbours_are_eight_at_half_body_diagonal() {
        let a = 2.866e-10; // alpha-Fe
        assert_first_shell(&make_bcc(a, 1e-11).unwrap(), 8, a * 3.0_f64.sqrt() / 2.0);
    }

    #[test]
    fn diamond_nearest_neighbours_are_four_at_quarter_body_diagonal() {
        let a = 5.431e-10; // Si
        assert_first_shell(
            &make_diamond(a, 1e-11).unwrap(),
            4,
            a * 3.0_f64.sqrt() / 4.0,
        );
    }

    #[test]
    fn ideal_hcp_has_twelve_nearest_neighbours_at_a() {
        let a = 3.21e-10; // Mg
        let c = a * (8.0_f64 / 3.0).sqrt();
        let hcp = make_hcp(a, c, 1e-11).unwrap();
        assert_eq!(hcp.basis_atoms.len(), 4);
        assert_first_shell(&hcp, 12, a);
        // The second shell of ideal HCP is at a*sqrt(2), six atoms.
        let d = neighbour_distances(&hcp, 0);
        assert!((d[12] - a * 2.0_f64.sqrt()).abs() < 1e-12, "{}", d[12]);
    }

    #[test]
    fn sdf_is_minus_radius_on_every_atom_and_half_gap_between_neighbours() {
        let a = 1.0;
        let r = 0.05;
        let hcp_c = a * (8.0_f64 / 3.0).sqrt();
        let cases = [
            (make_fcc(a, r).unwrap(), a / 2.0_f64.sqrt()),
            (make_bcc(a, r).unwrap(), a * 3.0_f64.sqrt() / 2.0),
            (make_diamond(a, r).unwrap(), a * 3.0_f64.sqrt() / 4.0),
            (make_hcp(a, hcp_c, r).unwrap(), a),
        ];
        for (crystal, nn) in cases {
            for atom in &crystal.basis_atoms {
                let p = [0, 1, 2].map(|k| atom[k] * crystal.periods[k]);
                assert!((crystal.evaluate(p) + r).abs() < 1e-12);
                // Shifted by whole periods (including negative ones) too.
                let shifted = [
                    p[0] - 3.0 * crystal.periods[0],
                    p[1] + crystal.periods[1],
                    p[2],
                ];
                assert!((crystal.evaluate(shifted) + r).abs() < 1e-9);
            }
            // Midpoint between atom 0 and its nearest neighbour.
            let target = nn;
            let mut best = None;
            for b in &crystal.basis_atoms {
                for i in -1..=1i32 {
                    for j in -1..=1i32 {
                        for k in -1..=1i32 {
                            let q = [
                                (b[0] + f64::from(i)) * crystal.periods[0],
                                (b[1] + f64::from(j)) * crystal.periods[1],
                                (b[2] + f64::from(k)) * crystal.periods[2],
                            ];
                            let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                            if (len - target).abs() < 1e-12 {
                                best = Some(q);
                            }
                        }
                    }
                }
            }
            let q = best.expect("a nearest neighbour image");
            let mid = q.map(|v| v * 0.5);
            let d = crystal.evaluate(mid);
            assert!(
                (d - (nn / 2.0 - r)).abs() < 1e-12,
                "{:?}: {d} vs {}",
                crystal.lattice_kind,
                nn / 2.0 - r
            );
        }
    }

    #[test]
    fn from_lattice_builds_supported_systems() {
        let cubic = |system| Lattice {
            system,
            abc_m: [4.0e-10, 4.0e-10, 4.0e-10],
            ..Default::default()
        };
        let fcc = CrystalPrimitive::from_lattice(&cubic(CrystalSystem::Fcc), 1e-10).unwrap();
        assert_eq!(fcc, make_fcc(4.0e-10, 1e-10).unwrap());
        let bcc = CrystalPrimitive::from_lattice(&cubic(CrystalSystem::Bcc), 1e-10).unwrap();
        assert_eq!(bcc.lattice_kind, LatticeKind::BCC);
        let dia =
            CrystalPrimitive::from_lattice(&cubic(CrystalSystem::DiamondCubic), 1e-10).unwrap();
        assert_eq!(dia.basis_atoms.len(), 8);
        let sc = CrystalPrimitive::from_lattice(&cubic(CrystalSystem::SimpleCubic), 1e-10).unwrap();
        assert_eq!(sc.basis_atoms.len(), 1);
        assert_eq!(
            sc.lattice_kind.crystal_system(),
            Some(CrystalSystem::SimpleCubic)
        );

        // Ti6Al4V's datasheet cell: a = 295 pm, c = 468 pm.
        let ti = Lattice {
            system: CrystalSystem::Hcp,
            abc_m: [295.0e-12, 295.0e-12, 468.0e-12],
            ..Default::default()
        };
        let hcp = CrystalPrimitive::from_lattice(&ti, 1.4e-10).unwrap();
        assert_eq!(hcp.lattice_kind, LatticeKind::HCP);
        assert!((hcp.periods[1] - 295.0e-12 * 3.0_f64.sqrt()).abs() < 1e-24);
        assert_eq!(hcp.periods[2], 468.0e-12);
    }

    #[test]
    fn from_lattice_rejects_unsupported_or_missing_data() {
        let t = Lattice {
            system: CrystalSystem::Tetragonal,
            abc_m: [1.0, 1.0, 2.0],
            ..Default::default()
        };
        assert_eq!(
            CrystalPrimitive::from_lattice(&t, 0.1),
            Err(CrystalError::UnsupportedSystem(CrystalSystem::Tetragonal))
        );
        for system in [CrystalSystem::Amorphous, CrystalSystem::Unknown] {
            let l = Lattice {
                system,
                ..Default::default()
            };
            assert_eq!(
                CrystalPrimitive::from_lattice(&l, 0.1),
                Err(CrystalError::UnsupportedSystem(system))
            );
        }
        // FCC with no lattice parameter on the datasheet.
        let missing = Lattice {
            system: CrystalSystem::Fcc,
            ..Default::default()
        };
        assert!(matches!(
            CrystalPrimitive::from_lattice(&missing, 0.1),
            Err(CrystalError::InvalidPeriod(_))
        ));
        // HCP with `a` but no `c`.
        let no_c = Lattice {
            system: CrystalSystem::Hcp,
            abc_m: [3.0e-10, 3.0e-10, 0.0],
            ..Default::default()
        };
        assert!(matches!(
            CrystalPrimitive::from_lattice(&no_c, 0.1),
            Err(CrystalError::InvalidPeriod(_))
        ));
    }
}
