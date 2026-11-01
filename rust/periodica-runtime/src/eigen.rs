//====== periodica/rust/periodica-runtime/src/eigen.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # eigen
//!
//! A cyclic Jacobi eigensolver for real symmetric 3x3 matrices.
//!
//! ## Why not `nalgebra::SymmetricEigen`
//!
//! nalgebra 0.33's implicit-QR solver returns wrong eigenvectors for nearly
//! diagonal input, and whether it does depends on the overall scale of the
//! matrix: with off-diagonals `1e-12` of the diagonal it is right at scales
//! 1 and 1e3 but leaves residuals of `1.7e-3 * |lambda|max` (axes ~1e-3 rad
//! off) at 1e-9 and 1e9, and random near-diagonal tensors tilt by up to 25
//! degrees. An inertia tensor is very often nearly diagonal (any body roughly
//! aligned with the world axes), so that is the common case, not an edge one.
//!
//! ## Properties
//!
//! - **Scale-free.** The matrix is first scaled by an exact power of two so
//!   its largest entry is in `[1, 2)`; the rotations depend only on ratios,
//!   so the eigenvectors are bit-identical for `A` and `2^k * A`.
//! - **Backward stable.** Each rotation zeroes its pivot exactly. A pivot is
//!   only dropped without a rotation when it is below 1% of an ulp of *both*
//!   diagonal entries it couples, so the result is the exact decomposition
//!   of a matrix within a few ulps of the input.
//! - **Deterministic across platforms.** Only `+ - * /` and `sqrt` (all
//!   correctly rounded by IEEE 754): no libm calls whose last bit could
//!   differ between operating systems, so a networked peer reproduces the
//!   same principal frame.
//! - **Bounded.** Quadratic convergence settles a 3x3 in about five sweeps;
//!   [`MAX_SWEEPS`] only guarantees termination.

/// Sweep cap. Each sweep visits the three off-diagonal pivots once.
const MAX_SWEEPS: usize = 32;

/// Past this `|theta|`, `theta^2` could overflow; `t = 1 / (2 theta)` is then
/// exact to working precision.
const THETA_ASYMPTOTIC: f64 = 1.0e150;

/// Eigen-decomposition of the real symmetric matrix `m`.
///
/// Only the upper triangle of `m` is read. Returns `(values, vectors)` with
/// `vectors[k]` the unit eigenvector of `values[k]`; the three vectors are
/// orthonormal. The order is the solver's (unsorted), and the caller fixes
/// order, signs and handedness.
///
/// The entries must be finite; the caller is responsible for that check.
pub(crate) fn symmetric_eigen3(m: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let max_abs = [m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]]
        .iter()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return ([0.0; 3], v);
    }
    let (down, up) = power_of_two_normaliser(max_abs);

    let mut a = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            a[i][j] = m[i][j] * down;
            a[j][i] = a[i][j];
        }
    }

    // Bounded loop: at most MAX_SWEEPS sweeps of three rotations.
    for _ in 0..MAX_SWEEPS {
        if a[0][1] == 0.0 && a[0][2] == 0.0 && a[1][2] == 0.0 {
            break;
        }
        for (p, q, r) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
            jacobi_rotate(&mut a, &mut v, p, q, r);
        }
    }

    let values = [a[0][0] * up, a[1][1] * up, a[2][2] * up];
    let vectors = [0, 1, 2].map(|k| [v[0][k], v[1][k], v[2][k]]);
    (values, vectors)
}

/// Exact powers of two `(down, up)` with `max_abs * down` in `[1, 2)` and
/// `down * up == 1`. The exponent is clamped to the normal range, so a
/// subnormal `max_abs` is only partly normalised (still correct, merely
/// without the bit-identical scale invariance).
fn power_of_two_normaliser(max_abs: f64) -> (f64, f64) {
    // Unbiased binary exponent of a positive finite double.
    let exponent = ((max_abs.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let e = exponent.clamp(-1022, 1023);
    (2.0_f64.powi(-e), 2.0_f64.powi(e))
}

/// One Jacobi rotation in the `(p, q)` plane (`r` is the third index),
/// zeroing `a[p][q]` and accumulating the rotation into the columns of `v`.
///
/// The update is the numerically stable form of Rutishauser (Numerical
/// Recipes section 11.1): the smaller rotation angle, with `tau = s / (1 + c)`
/// so every update is a small correction to the old value.
fn jacobi_rotate(a: &mut [[f64; 3]; 3], v: &mut [[f64; 3]; 3], p: usize, q: usize, r: usize) {
    let apq = a[p][q];
    if apq == 0.0 {
        return;
    }
    let (app, aqq) = (a[p][p], a[q][q]);

    // Negligible pivot: below 1% of an ulp of both diagonal entries it couples.
    // Dropping it perturbs the matrix by less than its own rounding error.
    let g = 100.0 * apq.abs();
    if app.abs() + g == app.abs() && aqq.abs() + g == aqq.abs() {
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        return;
    }

    let theta = (aqq - app) / (2.0 * apq);
    let t = if theta.abs() > THETA_ASYMPTOTIC {
        0.5 / theta
    } else {
        let t = 1.0 / (theta.abs() + (theta * theta + 1.0).sqrt());
        if theta < 0.0 {
            -t
        } else {
            t
        }
    };
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;
    let tau = s / (1.0 + c);
    let h = t * apq;

    a[p][p] = app - h;
    a[q][q] = aqq + h;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    let (arp, arq) = (a[r][p], a[r][q]);
    let new_rp = arp - s * (arq + arp * tau);
    let new_rq = arq + s * (arp - arq * tau);
    a[r][p] = new_rp;
    a[p][r] = new_rp;
    a[r][q] = new_rq;
    a[q][r] = new_rq;

    for row in v.iter_mut() {
        let (vp, vq) = (row[p], row[q]);
        row[p] = vp - s * (vq + vp * tau);
        row[q] = vq + s * (vp - vq * tau);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual(m: &[[f64; 3]; 3], value: f64, vector: &[f64; 3]) -> f64 {
        (0..3)
            .map(|i| {
                let mv: f64 = (0..3).map(|j| m[i][j] * vector[j]).sum();
                (mv - value * vector[i]).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn diagonal_input_is_returned_unrotated() {
        let m = [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]];
        let (values, vectors) = symmetric_eigen3(&m);
        assert_eq!(values, [3.0, -1.0, 2.0]);
        assert_eq!(
            vectors,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn zero_matrix_is_zero_with_identity_vectors() {
        let (values, vectors) = symmetric_eigen3(&[[0.0; 3]; 3]);
        assert_eq!(values, [0.0; 3]);
        assert_eq!(
            vectors,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn dense_matrix_decomposes_to_working_precision() {
        let m = [[4.0, 1.0, -2.0], [1.0, 2.0, 0.5], [-2.0, 0.5, 3.0]];
        let (values, vectors) = symmetric_eigen3(&m);
        let trace: f64 = values.iter().sum();
        assert!((trace - 9.0).abs() < 1e-14);
        for k in 0..3 {
            assert!(residual(&m, values[k], &vectors[k]) < 1e-14, "{values:?}");
            for j in 0..3 {
                let dot: f64 = (0..3).map(|i| vectors[k][i] * vectors[j][i]).sum();
                let want = if j == k { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn eigenvectors_are_bit_identical_under_power_of_two_scaling() {
        let m = [[1.0, 1e-12, 3e-13], [1e-12, 2.0, -2e-12], [3e-13, -2e-12, 3.0]];
        let (values, vectors) = symmetric_eigen3(&m);
        for k in [-40, -3, 7, 40] {
            let s = 2.0_f64.powi(k);
            let scaled = m.map(|row| row.map(|x| x * s));
            let (sv, svec) = symmetric_eigen3(&scaled);
            assert_eq!(svec, vectors, "scale 2^{k}");
            assert_eq!(sv, values.map(|x| x * s), "scale 2^{k}");
        }
    }

    #[test]
    fn subnormal_and_huge_pivots_terminate_cleanly() {
        let tiny = f64::from_bits(1); // smallest subnormal
        let m = [[0.0, tiny, 0.0], [tiny, 1.0, 0.0], [0.0, 0.0, 1.5]];
        let (values, vectors) = symmetric_eigen3(&m);
        assert!(values.iter().all(|v| v.is_finite()));
        for k in 0..3 {
            assert!(residual(&m, values[k], &vectors[k]) < 1e-300);
        }
        let big = 1.0e300;
        let m = [[big, big, 0.0], [big, big, 0.0], [0.0, 0.0, -big]];
        let (values, vectors) = symmetric_eigen3(&m);
        for k in 0..3 {
            assert!(residual(&m, values[k], &vectors[k]) <= 1e-15 * 2.0 * big);
        }
    }
}
