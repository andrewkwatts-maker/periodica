//====== periodica/rust/periodica-runtime/src/mass.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # mass
//!
//! Mass properties of an SDF volume: integrates `density(x) * H(-sdf(x))`
//! over a bounding box to produce total mass, centre of mass and the inertia
//! tensor, and diagonalises that tensor into principal moments and axes.
//!
//! ## Estimator
//!
//! [`integrate_mass`] is a quasi-Monte-Carlo estimate over the points of a
//! [`HaltonSampler`] scaled into the bounds. Each point inside the SDF carries
//! weight `density * box_volume / N`.
//!
//! It runs in a **single pass**. Mass, first moments and second moments are
//! accumulated about the *bounds centre* `c`; once the centre of mass
//! `c + d` is known, the second moments are moved onto it with the
//! parallel-axis theorem, `C_ij = S_ij - M * d_i * d_j`. That evaluates the
//! SDF and density once per sample instead of twice, and accumulating about
//! `c` rather than the world origin keeps the subtraction well conditioned
//! for an object far from the origin.
//!
//! ## Determinism
//!
//! The result is a pure function of the arguments: the same `rng_seed` gives a
//! bit-identical [`MassDistribution`]; a different seed gives a different
//! Cranley-Patterson rotation and therefore an independent estimate.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use thiserror::Error;

use crate::fields::{DensityFn, SdfFn};
use crate::sampling::HaltonSampler;

/// Error type for mass integration.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MassError {
    /// Caller asked for zero samples.
    #[error("integration_steps must be > 0; got {0}")]
    NoSamples(u64),
    /// Bounds are inverted, empty or non-finite in some axis.
    #[error("bounds are inverted in some axis: lo={lo:?}, hi={hi:?}")]
    InvalidBounds {
        /// Lower corner as passed.
        lo: [f64; 3],
        /// Upper corner as passed.
        hi: [f64; 3],
    },
    /// Integration produced a non-finite mass -- the density closure returned
    /// NaN/Inf, or the sum overflowed.
    #[error("integration produced a non-finite mass: {0}")]
    NonFiniteMass(f64),
}

/// Bundled outputs of a mass integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassDistribution {
    /// Total integrated mass (kg).
    pub total_mass_kg: f64,
    /// Centre of mass in the same coordinate frame as `bounds` (metres).
    pub center_of_mass: [f64; 3],
    /// Symmetric 3x3 inertia tensor about the centre of mass (kg*m^2).
    pub inertia_tensor: [[f64; 3]; 3],
}

/// Iteration cap for the eigensolver. A symmetric 3x3 converges in a handful
/// of implicit QR sweeps; the cap only guarantees termination on garbage.
const EIGEN_MAX_ITERATIONS: usize = 128;

impl MassDistribution {
    /// The principal moments of inertia (eigenvalues of the inertia tensor),
    /// sorted ascending, in kg*m^2.
    ///
    /// Entry `i` is the moment about axis `i` of [`Self::principal_axes`].
    pub fn principal_moments(&self) -> [f64; 3] {
        self.principal_frame().0
    }

    /// The principal axes (eigenvectors of the inertia tensor), as **rows**.
    ///
    /// Row `i` is the unit axis whose moment is entry `i` of
    /// [`Self::principal_moments`]. The rows are orthonormal and right-handed
    /// (`row2 = row0 x row1`), so the returned matrix `R` is a proper
    /// rotation taking world vectors into the principal frame:
    /// `R * I * R^T = diag(moments)`.
    ///
    /// Signs are canonical so the frame is reproducible: the largest
    /// component of rows 0 and 1 is positive, and row 2 follows from
    /// handedness. For a diagonal tensor with ascending entries this is the
    /// identity. Where moments coincide (a sphere, a cube) any orthonormal
    /// basis of the degenerate subspace is a correct answer.
    pub fn principal_axes(&self) -> [[f64; 3]; 3] {
        self.principal_frame().1
    }

    /// [`Self::principal_moments`] and [`Self::principal_axes`] from a single
    /// eigensolve.
    ///
    /// The tensor is symmetrised (`(I + I^T) / 2`) before solving. A tensor
    /// with non-finite entries falls back to its diagonal and the coordinate
    /// axes, still sorted, rather than looping or panicking.
    pub fn principal_frame(&self) -> ([f64; 3], [[f64; 3]; 3]) {
        let t = &self.inertia_tensor;
        let m = Matrix3::from_fn(|i, j| 0.5 * (t[i][j] + t[j][i]));
        let solved = if m.iter().all(|v| v.is_finite()) {
            SymmetricEigen::try_new(m, f64::EPSILON, EIGEN_MAX_ITERATIONS)
        } else {
            None
        };
        let (values, vectors) = match solved {
            Some(eig) => (
                [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]],
                [0, 1, 2].map(|k| eig.eigenvectors.column(k).into_owned()),
            ),
            None => (
                [m[(0, 0)], m[(1, 1)], m[(2, 2)]],
                [Vector3::x(), Vector3::y(), Vector3::z()],
            ),
        };
        sorted_right_handed(values, vectors)
    }
}

/// Sort eigenpairs ascending and turn the vectors into a canonical,
/// orthonormal, right-handed set of rows.
fn sorted_right_handed(values: [f64; 3], vectors: [Vector3<f64>; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let moments = order.map(|k| values[k]);

    let a0 = canonical_sign(vectors[order[0]].normalize());
    // Re-orthogonalise against a0 (one Gram-Schmidt step) so the frame is
    // orthonormal to rounding even if the solver's vectors are not quite.
    let v1 = vectors[order[1]];
    let a1 = canonical_sign((v1 - a0 * a0.dot(&v1)).normalize());
    let a2 = a0.cross(&a1);
    (moments, [a0, a1, a2].map(|v| [v.x, v.y, v.z]))
}

/// Flip `v` so its largest-magnitude component is positive.
fn canonical_sign(v: Vector3<f64>) -> Vector3<f64> {
    if v[v.iamax()] < 0.0 {
        -v
    } else {
        v
    }
}

/// Integrate mass distribution over `bounds` for the given `sdf` and `density`.
///
/// `integration_steps` is the number of quasi-random sample points. Error
/// falls faster than plain Monte Carlo's `N^-1/2` (about `N^-0.8` for a
/// sphere, whose mean relative mass error is ~0.25% at 2^14 samples). The
/// same `rng_seed` always produces a bit-identical result.
///
/// When no sample lands inside the SDF (or the integrated mass is not
/// positive) the centre of mass is undefined; it and the inertia tensor are
/// then reported as zero, alongside the integrated `total_mass_kg`.
///
/// # Errors
///
/// - [`MassError::NoSamples`] when `integration_steps == 0`.
/// - [`MassError::InvalidBounds`] when `lo >= hi` or either is non-finite in
///   any axis.
/// - [`MassError::NonFiniteMass`] when `density` returns NaN/Inf at a point
///   inside the SDF, or the total overflows.
pub fn integrate_mass(
    sdf: SdfFn<'_>,
    density: DensityFn<'_>,
    bounds: ([f64; 3], [f64; 3]),
    integration_steps: u64,
    rng_seed: u64,
) -> Result<MassDistribution, MassError> {
    if integration_steps == 0 {
        return Err(MassError::NoSamples(integration_steps));
    }
    let (lo, hi) = bounds;
    if (0..3).any(|k| !lo[k].is_finite() || !hi[k].is_finite() || lo[k] >= hi[k]) {
        return Err(MassError::InvalidBounds { lo, hi });
    }

    let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let centre = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let sample_volume = extent[0] * extent[1] * extent[2] / integration_steps as f64;

    // Moments about the bounds centre: zeroth, first, and the six unique
    // second moments. Accumulating each off-diagonal once keeps the final
    // tensor exactly symmetric.
    let mut mass = 0.0_f64;
    let mut first = [0.0_f64; 3];
    let (mut sxx, mut syy, mut szz) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut sxy, mut sxz, mut syz) = (0.0_f64, 0.0_f64, 0.0_f64);

    let sampler = HaltonSampler::new(rng_seed);
    // Bounded loop: exactly `integration_steps` iterations.
    for i in 0..integration_steps {
        let u = sampler.point(i);
        let p = [
            lo[0] + extent[0] * u[0],
            lo[1] + extent[1] * u[1],
            lo[2] + extent[2] * u[2],
        ];
        if sdf(p) < 0.0 {
            let rho = density(p);
            if !rho.is_finite() {
                return Err(MassError::NonFiniteMass(rho));
            }
            let w = rho * sample_volume;
            let (x, y, z) = (p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
            mass += w;
            first[0] += w * x;
            first[1] += w * y;
            first[2] += w * z;
            sxx += w * x * x;
            syy += w * y * y;
            szz += w * z * z;
            sxy += w * x * y;
            sxz += w * x * z;
            syz += w * y * z;
        }
    }

    if !mass.is_finite() {
        return Err(MassError::NonFiniteMass(mass));
    }
    if mass <= 0.0 {
        return Ok(MassDistribution {
            total_mass_kg: mass,
            center_of_mass: [0.0; 3],
            inertia_tensor: [[0.0; 3]; 3],
        });
    }

    // Centre-of-mass offset from the bounds centre.
    let d = first.map(|s| s / mass);
    let center_of_mass = [centre[0] + d[0], centre[1] + d[1], centre[2] + d[2]];

    // Parallel-axis theorem: second moments about the centre of mass.
    let cxx = sxx - mass * d[0] * d[0];
    let cyy = syy - mass * d[1] * d[1];
    let czz = szz - mass * d[2] * d[2];
    let cxy = sxy - mass * d[0] * d[1];
    let cxz = sxz - mass * d[0] * d[2];
    let cyz = syz - mass * d[1] * d[2];

    // I = tr(C) * Id - C.
    let inertia_tensor = [
        [cyy + czz, -cxy, -cxz],
        [-cxy, cxx + czz, -cyz],
        [-cxz, -cyz, cxx + cyy],
    ];

    Ok(MassDistribution {
        total_mass_kg: mass,
        center_of_mass,
        inertia_tensor,
    })
}

/// Predicts which axis a body will rest on, given an SDF's bounds and its
/// computed centre of mass. Returns the index of the down-axis
/// (0 = x, 1 = y, 2 = z): the axis along which the centre of mass sits
/// furthest from the geometric centre of `bounds` (first axis on ties).
///
/// This is a cheap heuristic for spawning bodies in a plausible pose, not a
/// stability analysis.
pub fn predict_resting_orientation(bounds: ([f64; 3], [f64; 3]), com: [f64; 3]) -> usize {
    let (lo, hi) = bounds;
    let offsets = [0, 1, 2].map(|k| (com[k] - 0.5 * (lo[k] + hi[k])).abs());
    let mut best = 0usize;
    for (axis, &offset) in offsets.iter().enumerate().skip(1) {
        if offset > offsets[best] {
            best = axis;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use std::f64::consts::PI;

    // --- Transplanted from the engine (pt_periodica_mass.rs) ---------------

    #[test]
    fn rejects_zero_steps() {
        let sdf = |_p: [f64; 3]| 1.0;
        let rho = |_p: [f64; 3]| 1.0;
        let r = integrate_mass(&sdf, &rho, ([0.0; 3], [1.0; 3]), 0, 42);
        assert!(matches!(r, Err(MassError::NoSamples(_))));
    }

    #[test]
    fn rejects_inverted_bounds() {
        let sdf = |_p: [f64; 3]| 1.0;
        let rho = |_p: [f64; 3]| 1.0;
        let r = integrate_mass(&sdf, &rho, ([1.0; 3], [0.0; 3]), 100, 42);
        assert!(matches!(r, Err(MassError::InvalidBounds { .. })));
    }

    #[test]
    fn empty_sdf_yields_zero_mass() {
        let sdf = |_p: [f64; 3]| 1.0; // always outside
        let rho = |_p: [f64; 3]| 7874.0;
        let r = integrate_mass(&sdf, &rho, ([0.0; 3], [1.0; 3]), 1000, 42).unwrap();
        assert_eq!(r.total_mass_kg, 0.0);
    }

    #[test]
    fn uniform_box_mass_is_density_times_volume() {
        // A unit cube SDF (entirely inside the bounds) with 1 kg/m^3 density
        // should integrate to ~1 kg.
        let sdf = |_p: [f64; 3]| -1.0;
        let rho = |_p: [f64; 3]| 1.0;
        let r = integrate_mass(&sdf, &rho, ([0.0; 3], [1.0; 3]), 4096, 42).unwrap();
        assert!(
            (r.total_mass_kg - 1.0).abs() < 0.05,
            "got {}",
            r.total_mass_kg
        );
    }

    #[test]
    fn deterministic_seed_repeats() {
        let sdf =
            |p: [f64; 3]| (p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2) + (p[2] - 0.5).powi(2) - 0.16;
        let rho = |_p: [f64; 3]| 7874.0;
        let a = integrate_mass(&sdf, &rho, ([0.0; 3], [1.0; 3]), 2048, 42).unwrap();
        let b = integrate_mass(&sdf, &rho, ([0.0; 3], [1.0; 3]), 2048, 42).unwrap();
        assert_eq!(a.total_mass_kg, b.total_mass_kg);
        assert_eq!(a.center_of_mass, b.center_of_mass);
        // Bit-identical across the board, not just the two engine fields.
        assert_eq!(a, b);
    }

    #[test]
    fn principal_moments_returns_diagonal() {
        let m = MassDistribution {
            total_mass_kg: 1.0,
            center_of_mass: [0.0; 3],
            inertia_tensor: [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        };
        assert_eq!(m.principal_moments(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn rest_axis_picks_max_offset() {
        let bounds = ([0.0; 3], [1.0; 3]);
        // CoM offset in y is largest.
        assert_eq!(predict_resting_orientation(bounds, [0.5, 0.9, 0.5]), 1);
    }

    // --- Analytic solids ---------------------------------------------------

    fn sphere(centre: [f64; 3], r: f64) -> impl Fn([f64; 3]) -> f64 {
        move |p: [f64; 3]| {
            ((p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2) + (p[2] - centre[2]).powi(2))
                .sqrt()
                - r
        }
    }

    fn rel(got: f64, want: f64) -> f64 {
        (got / want - 1.0).abs()
    }

    #[test]
    fn uniform_solid_sphere_matches_closed_form() {
        let (r, rho) = (0.4, 7874.0);
        let sdf = sphere([0.5; 3], r);
        let m = integrate_mass(&sdf, &|_p| rho, ([0.0; 3], [1.0; 3]), 1 << 15, 3).unwrap();

        let mass = rho * 4.0 / 3.0 * PI * r.powi(3);
        let moment = 0.4 * mass * r * r;
        assert!(
            rel(m.total_mass_kg, mass) < 0.01,
            "mass {}",
            m.total_mass_kg
        );
        for k in 0..3 {
            assert!(
                (m.center_of_mass[k] - 0.5).abs() < 1e-3,
                "{:?}",
                m.center_of_mass
            );
            assert!(
                rel(m.inertia_tensor[k][k], moment) < 0.02,
                "I{k}{k} {} vs {moment}",
                m.inertia_tensor[k][k]
            );
            for j in (0..3).filter(|&j| j != k) {
                assert!(m.inertia_tensor[k][j].abs() < 0.01 * moment);
            }
        }
        // Isotropic: all three principal moments agree.
        let pm = m.principal_moments();
        assert!(
            rel(pm[0], moment) < 0.02 && rel(pm[2], moment) < 0.02,
            "{pm:?}"
        );
    }

    #[test]
    fn off_centre_box_has_analytic_com_and_inertia() {
        // Box [0.1,0.3] x [0.5,0.9] x [0.2,0.4] inside the unit cube.
        let (c, h) = ([0.2, 0.7, 0.3], [0.1, 0.2, 0.1]);
        let sdf = move |p: [f64; 3]| {
            (0..3)
                .map(|k| (p[k] - c[k]).abs() - h[k])
                .fold(f64::MIN, f64::max)
        };
        let rho = 2700.0;
        let m = integrate_mass(&sdf, &|_p| rho, ([0.0; 3], [1.0; 3]), 1 << 16, 11).unwrap();

        let mass = rho * 8.0 * h[0] * h[1] * h[2];
        assert!(rel(m.total_mass_kg, mass) < 0.01, "{}", m.total_mass_kg);
        for (got, want) in m.center_of_mass.iter().zip(c) {
            assert!((got - want).abs() < 2e-3, "{:?}", m.center_of_mass);
        }
        // Solid cuboid with edges (2h): I_xx = M/12 * ((2hy)^2 + (2hz)^2).
        let e = h.map(|v| 2.0 * v);
        let want = [
            mass / 12.0 * (e[1] * e[1] + e[2] * e[2]),
            mass / 12.0 * (e[0] * e[0] + e[2] * e[2]),
            mass / 12.0 * (e[0] * e[0] + e[1] * e[1]),
        ];
        for (k, want) in want.into_iter().enumerate() {
            let got = m.inertia_tensor[k][k];
            assert!(rel(got, want) < 0.03, "I{k}{k} {got} vs {want}");
        }
        // x is the axis with the largest CoM offset from the bounds centre.
        assert_eq!(
            predict_resting_orientation(([0.0; 3], [1.0; 3]), m.center_of_mass),
            0
        );
    }

    #[test]
    fn different_seed_gives_a_different_but_equally_good_estimate() {
        let sdf = sphere([0.5; 3], 0.4);
        let mass = 4.0 / 3.0 * PI * 0.4_f64.powi(3);
        let a = integrate_mass(&sdf, &|_p| 1.0, ([0.0; 3], [1.0; 3]), 4096, 1).unwrap();
        let b = integrate_mass(&sdf, &|_p| 1.0, ([0.0; 3], [1.0; 3]), 4096, 2).unwrap();
        assert_ne!(a.total_mass_kg, b.total_mass_kg);
        assert!(rel(a.total_mass_kg, mass) < 0.01);
        assert!(rel(b.total_mass_kg, mass) < 0.01);
    }

    #[test]
    fn non_finite_density_inside_is_an_error() {
        let sdf = |_p: [f64; 3]| -1.0;
        let r = integrate_mass(&sdf, &|_p| f64::NAN, ([0.0; 3], [1.0; 3]), 16, 0);
        assert!(matches!(r, Err(MassError::NonFiniteMass(_))));
    }

    // --- Eigensolver -------------------------------------------------------

    fn assert_proper_rotation(axes: &[[f64; 3]; 3]) {
        let r = Matrix3::from_fn(|i, j| axes[i][j]);
        let should_be_id = r * r.transpose();
        assert!((should_be_id - Matrix3::identity()).norm() < 1e-12, "{r}");
        assert!(
            (r.determinant() - 1.0).abs() < 1e-12,
            "det {}",
            r.determinant()
        );
    }

    fn with_tensor(t: [[f64; 3]; 3]) -> MassDistribution {
        MassDistribution {
            total_mass_kg: 1.0,
            center_of_mass: [0.0; 3],
            inertia_tensor: t,
        }
    }

    #[test]
    fn rotated_tensor_recovers_its_eigen_decomposition() {
        let rot = Rotation3::from_euler_angles(0.3, -0.7, 1.1);
        let q = rot.matrix();
        let moments = [1.0, 2.0, 3.0];
        // I = Q diag(moments) Q^T, so column k of Q is the axis of moment k.
        let t = q * Matrix3::from_diagonal(&Vector3::from(moments)) * q.transpose();
        let m = with_tensor(std::array::from_fn(|i| std::array::from_fn(|j| t[(i, j)])));
        assert!(
            m.inertia_tensor[0][1].abs() > 0.1,
            "test needs off-diagonals"
        );

        let (pm, axes) = m.principal_frame();
        for k in 0..3 {
            assert!((pm[k] - moments[k]).abs() < 1e-12, "{pm:?}");
            let expected = q.column(k);
            let dot: f64 = (0..3).map(|i| axes[k][i] * expected[i]).sum();
            assert!((dot.abs() - 1.0).abs() < 1e-12, "axis {k}: dot {dot}");
        }
        assert_proper_rotation(&axes);

        // R I R^T is diagonal with the sorted moments.
        let r = Matrix3::from_fn(|i, j| axes[i][j]);
        let d = r * t * r.transpose();
        assert!((d - Matrix3::from_diagonal(&Vector3::from(pm))).norm() < 1e-12);
    }

    #[test]
    fn unsorted_diagonal_is_sorted_with_matching_axes() {
        let m = with_tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
        let (pm, axes) = m.principal_frame();
        assert_eq!(pm, [1.0, 2.0, 3.0]);
        let want = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        for k in 0..3 {
            for i in 0..3 {
                assert!((axes[k][i] - want[k][i]).abs() < 1e-12, "{axes:?}");
            }
        }
        assert_proper_rotation(&axes);
    }

    #[test]
    fn diagonal_ascending_tensor_has_identity_axes() {
        let m = with_tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        let axes = m.principal_axes();
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for k in 0..3 {
            for i in 0..3 {
                assert!((axes[k][i] - id[k][i]).abs() < 1e-12, "{axes:?}");
            }
        }
    }

    #[test]
    fn degenerate_and_non_finite_tensors_stay_well_formed() {
        let iso = with_tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
        let (pm, axes) = iso.principal_frame();
        assert_eq!(pm, [2.0, 2.0, 2.0]);
        assert_proper_rotation(&axes);

        let zero = with_tensor([[0.0; 3]; 3]);
        assert_eq!(zero.principal_moments(), [0.0; 3]);
        assert_proper_rotation(&zero.principal_axes());

        // Garbage in must terminate and keep a usable frame.
        let nan = with_tensor([[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]);
        let (pm, axes) = nan.principal_frame();
        assert_eq!(&pm[..2], &[0.5, 1.0]);
        assert!(pm[2].is_nan());
        assert_proper_rotation(&axes);
    }

    // --- Estimator quality -------------------------------------------------

    /// The engine's original estimator, kept verbatim as the reference: a
    /// 64-bit LCG with the Numerical Recipes constants, three draws per point.
    fn old_lcg_volume(sdf: SdfFn<'_>, steps: u64, seed: u64) -> f64 {
        fn lcg_step(state: u64) -> (u64, f64) {
            let new_state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let unit = (new_state >> 8) as f64 / (1u64 << 56) as f64;
            (new_state, unit)
        }
        let mut state = seed | 1;
        let mut inside = 0u64;
        for _ in 0..steps {
            let (s1, x) = lcg_step(state);
            let (s2, y) = lcg_step(s1);
            let (s3, z) = lcg_step(s2);
            state = s3;
            if sdf([x, y, z]) < 0.0 {
                inside += 1;
            }
        }
        inside as f64 / steps as f64
    }

    #[test]
    fn halton_beats_the_old_lcg_at_equal_sample_count() {
        let sdf = sphere([0.5; 3], 0.4);
        let exact = 4.0 / 3.0 * PI * 0.4_f64.powi(3);
        // Odd seeds: the old sampler ORed the seed with 1, so 2k and 2k+1
        // were the same stream.
        let seeds: Vec<u64> = (0..16).map(|k| 2 * k + 1).collect();
        let mean_err = |f: &dyn Fn(u64) -> f64| -> f64 {
            seeds.iter().map(|&s| rel(f(s), exact)).sum::<f64>() / seeds.len() as f64
        };

        // Measured mean relative error over the 16 seeds:
        //   N        Halton    old LCG
        //   1024     1.4e-2    5.8e-2
        //   4096     8.8e-3    1.9e-2
        //   16384    2.4e-3    6.9e-3
        //   65536    8.6e-4    6.5e-3
        // A sphere's indicator is discontinuous, so at small N the error is
        // floored by the boundary cells and the gain is modest; it widens
        // with N because the convergence rate itself is better.
        let mut halton_err = Vec::new();
        for steps in [1024u64, 4096, 16384, 65536] {
            let halton = mean_err(&|s| {
                integrate_mass(&sdf, &|_p| 1.0, ([0.0; 3], [1.0; 3]), steps, s)
                    .unwrap()
                    .total_mass_kg
            });
            let lcg = mean_err(&|s| old_lcg_volume(&sdf, steps, s));
            assert!(
                halton < lcg,
                "N={steps}: Halton mean relative error {halton:.2e} should beat the LCG's {lcg:.2e}"
            );
            halton_err.push(halton);
        }
        // 16x the samples buys 4x accuracy at the Monte-Carlo rate N^-1/2.
        // Low discrepancy must do strictly better than that.
        assert!(
            halton_err[3] * 4.0 < halton_err[1],
            "Halton error {:.2e} -> {:.2e} for 16x samples is no better than N^-1/2",
            halton_err[1],
            halton_err[3]
        );
    }
}
