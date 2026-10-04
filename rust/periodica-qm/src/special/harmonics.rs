//====== periodica/rust/periodica-qm/src/special/harmonics.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Complex and real spherical harmonics.
//!
//! ## Complex `Y_lm` (physics convention)
//!
//! ```text
//! Y_lm(theta, phi) = Pbar_l^m(cos theta) e^{i m phi} / sqrt(2 pi),   m >= 0
//! Y_l,-m           = (-1)^m conj(Y_lm)
//! ```
//!
//! with `Pbar` the normalised associated Legendre function **including** the
//! Condon-Shortley phase (see [`super::legendre`]). These are orthonormal on
//! the unit sphere and agree with `scipy.special.sph_harm_y` (cross-checked in
//! `tests/special_accuracy.rs`), Jackson, Sakurai and DLMF 14.30.1.
//!
//! ## Real `S_lm` (chemistry convention)
//!
//! ```text
//! S_l0  = Y_l0
//! S_lm  = sqrt(2) (-1)^m Re Y_lm   = (1/sqrt 2)   [(-1)^m Y_lm + Y_l,-m],  m > 0
//! S_l-m = sqrt(2) (-1)^m Im Y_lm   = (1/(i sqrt 2))[(-1)^m Y_lm - Y_l,-m],  m > 0
//! ```
//!
//! The `(-1)^m` cancels the Condon-Shortley phase, so every real harmonic is a
//! positive multiple of a polynomial in `x/r, y/r, z/r` with the textbook
//! sign: `S_11 = sqrt(3/4pi) x/r` (p_x, positive along +x), `S_1,-1` is p_y,
//! `S_10` is p_z, `S_22 ~ x^2 - y^2`, `S_2,-2 ~ xy`. This is the convention of
//! quantum-chemistry codes (Helgaker, Jorgensen and Olsen, *Molecular
//! Electronic-Structure Theory*, eq. 6.4.19; PySCF; Wikipedia's "real
//! spherical harmonics" table). Some physics texts define real harmonics
//! *without* the `(-1)^m`, which flips the sign of every odd-`m` function --
//! p_x would point along -x.

use core::f64::consts::{FRAC_1_SQRT_2, PI};

use super::legendre::legendre_normalised_cs;
use crate::complex::Complex;

/// `1 / sqrt(2 pi)`.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// `1 / sqrt(pi)`.
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Which angular basis an orbital uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Basis {
    /// Real harmonics `S_lm` (chemistry convention): `p_x, p_y, p_z, d_xy, ...`.
    Real,
    /// Complex harmonics `Y_lm` (Condon-Shortley): eigenfunctions of `L_z`.
    Complex,
}

/// A direction on the unit sphere, held as the four numbers every harmonic
/// needs: `cos theta, sin theta, cos phi, sin phi` (with `sin theta >= 0`).
///
/// Building it from a Cartesian vector needs only `sqrt` and division -- no
/// inverse trigonometry -- so evaluating orbitals on a grid is cheap and
/// bit-reproducible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction {
    cos_theta: f64,
    sin_theta: f64,
    cos_phi: f64,
    sin_phi: f64,
}

impl Direction {
    /// From polar angle `theta` (from +z) and azimuth `phi` (from +x toward +y).
    pub fn from_angles(theta: f64, phi: f64) -> Self {
        let (st, ct) = libm::sincos(theta);
        let (sp, cp) = libm::sincos(phi);
        // A negative sin(theta) (theta outside [0, pi]) is the same point as
        // (|theta|, phi + pi).
        if st < 0.0 {
            Self {
                cos_theta: ct,
                sin_theta: -st,
                cos_phi: -cp,
                sin_phi: -sp,
            }
        } else {
            Self {
                cos_theta: ct,
                sin_theta: st,
                cos_phi: cp,
                sin_phi: sp,
            }
        }
    }

    /// The direction of `v` from the origin. The zero vector maps to +z (where
    /// every `m != 0` harmonic vanishes anyway); on the z axis `phi = 0`.
    pub fn from_cartesian(v: [f64; 3]) -> Self {
        let rho = libm::hypot(v[0], v[1]);
        let r = libm::hypot(rho, v[2]);
        if r == 0.0 {
            return Self::from_angles(0.0, 0.0);
        }
        let (cos_phi, sin_phi) = if rho > 0.0 {
            (v[0] / rho, v[1] / rho)
        } else {
            (1.0, 0.0)
        };
        Self {
            cos_theta: v[2] / r,
            sin_theta: rho / r,
            cos_phi,
            sin_phi,
        }
    }

    /// `cos theta`.
    #[inline]
    pub fn cos_theta(&self) -> f64 {
        self.cos_theta
    }

    /// `sin theta` (non-negative).
    #[inline]
    pub fn sin_theta(&self) -> f64 {
        self.sin_theta
    }

    /// `e^{i |m| phi}` by repeated complex multiplication (no transcendental).
    fn azimuthal(&self, m: u32) -> Complex {
        let base = Complex::new(self.cos_phi, self.sin_phi);
        let mut acc = Complex::ONE;
        let mut b = base;
        let mut k = m;
        while k > 0 {
            if k & 1 == 1 {
                acc *= b;
            }
            b *= b;
            k >>= 1;
        }
        acc
    }

    /// `Pbar_l^{|m|}(cos theta)` (Condon-Shortley phase included).
    #[inline]
    fn legendre(&self, l: u32, m_abs: u32) -> f64 {
        legendre_normalised_cs(l, m_abs, self.cos_theta, self.sin_theta)
    }

    /// Complex `Y_lm` at this direction; `0` when `|m| > l`.
    pub fn ylm(&self, l: u32, m: i32) -> Complex {
        let ma = m.unsigned_abs();
        if ma > l {
            return Complex::ZERO;
        }
        let p = self.legendre(l, ma) * FRAC_1_SQRT_2PI;
        let e = self.azimuthal(ma);
        if m >= 0 {
            e * p
        } else {
            // (-1)^m conj(Y_l|m|).
            e.conj() * (parity(ma) * p)
        }
    }

    /// Real `S_lm` (chemistry convention) at this direction; `0` when `|m| > l`.
    pub fn slm(&self, l: u32, m: i32) -> f64 {
        let ma = m.unsigned_abs();
        if ma > l {
            return 0.0;
        }
        if m == 0 {
            return self.legendre(l, 0) * FRAC_1_SQRT_2PI;
        }
        // (-1)^m Pbar cancels the Condon-Shortley phase.
        let p = parity(ma) * self.legendre(l, ma) * FRAC_1_SQRT_PI;
        let e = self.azimuthal(ma);
        if m > 0 {
            p * e.re
        } else {
            p * e.im
        }
    }

    /// The angular function of `basis` at this direction, as a complex value
    /// (real harmonics have zero imaginary part).
    #[inline]
    pub fn harmonic(&self, l: u32, m: i32, basis: Basis) -> Complex {
        match basis {
            Basis::Complex => self.ylm(l, m),
            Basis::Real => Complex::real(self.slm(l, m)),
        }
    }
}

/// `(-1)^m` as an `f64`.
#[inline]
fn parity(m: u32) -> f64 {
    if m % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Complex spherical harmonic `Y_lm(theta, phi)` with the Condon-Shortley
/// phase; `0` when `|m| > l`.
pub fn spherical_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> Complex {
    Direction::from_angles(theta, phi).ylm(l, m)
}

/// Real spherical harmonic `S_lm(theta, phi)` in the chemistry convention
/// (p_x positive along +x); `0` when `|m| > l`.
pub fn real_spherical_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
    Direction::from_angles(theta, phi).slm(l, m)
}

/// The expansion of an angular basis function in complex harmonics of the
/// same `l`: `f = sum_k c_k Y_{l, m_k}`, as up to two `(m_k, c_k)` pairs.
///
/// Complex `Y_lm` is itself; real `S_lm` follows the relations in the module
/// docs.
pub fn complex_expansion(m: i32, basis: Basis) -> ([(i32, Complex); 2], usize) {
    let none = (0, Complex::ZERO);
    if basis == Basis::Complex || m == 0 {
        return ([(m, Complex::ONE), none], 1);
    }
    let sign = parity(m.unsigned_abs());
    let ma = m.abs();
    if m > 0 {
        // (1/sqrt 2)[(-1)^m Y_m + Y_-m]
        (
            [
                (ma, Complex::real(sign * FRAC_1_SQRT_2)),
                (-ma, Complex::real(FRAC_1_SQRT_2)),
            ],
            2,
        )
    } else {
        // (-i/sqrt 2)[(-1)^m Y_m - Y_-m]
        (
            [
                (ma, Complex::new(0.0, -sign * FRAC_1_SQRT_2)),
                (-ma, Complex::new(0.0, FRAC_1_SQRT_2)),
            ],
            2,
        )
    }
}

/// The overlap `<f_a | f_b>` on the unit sphere of two angular basis
/// functions of the **same** `l` (functions of different `l` are orthogonal).
///
/// Exact, from [`complex_expansion`]: `sum conj(a_k) b_k` over matching `m`.
pub fn angular_overlap(m_a: i32, basis_a: Basis, m_b: i32, basis_b: Basis) -> Complex {
    let (ea, na) = complex_expansion(m_a, basis_a);
    let (eb, nb) = complex_expansion(m_b, basis_b);
    let mut acc = Complex::ZERO;
    for &(ma, ca) in &ea[..na] {
        for &(mb, cb) in &eb[..nb] {
            if ma == mb {
                acc += ca.conj() * cb;
            }
        }
    }
    acc
}

/// `(CDF, PDF)` at `phi` in `[0, 2 pi]` of the azimuth of a real harmonic,
/// whose density is `cos^2(m phi)/pi` (`m > 0`) or `sin^2(|m| phi)/pi` (`m < 0`):
/// `CDF = phi/2pi +- sin(2|m| phi)/(4 pi |m|)`. `m` must be non-zero.
pub(crate) fn real_azimuthal_cdf(m: i32, phi: f64) -> (f64, f64) {
    let ma = f64::from(m.unsigned_abs());
    let (s2, c2) = libm::sincos(2.0 * ma * phi);
    let sign = if m > 0 { 1.0 } else { -1.0 };
    let cdf = phi / (2.0 * PI) + sign * s2 / (4.0 * PI * ma);
    let pdf = (1.0 + sign * c2) / (2.0 * PI);
    (cdf, pdf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_p_orbitals_point_along_their_axes() {
        let c = (3.0 / (4.0 * PI)).sqrt();
        // p_x at theta = pi/2, phi = 0 is +sqrt(3/4pi).
        let px = real_spherical_harmonic(1, 1, PI / 2.0, 0.0);
        assert!((px - c).abs() < 1e-15 && px > 0.0, "{px}");
        let py = real_spherical_harmonic(1, -1, PI / 2.0, PI / 2.0);
        assert!((py - c).abs() < 1e-15, "{py}");
        let pz = real_spherical_harmonic(1, 0, 0.0, 0.0);
        assert!((pz - c).abs() < 1e-15, "{pz}");
        // Cartesian construction agrees.
        let d = Direction::from_cartesian([2.0, 0.0, 0.0]);
        assert!((d.slm(1, 1) - c).abs() < 1e-15);
        assert!(d.slm(1, -1).abs() < 1e-15 && d.slm(1, 0).abs() < 1e-15);
        // d_xy positive in the first quadrant, d_x2-y2 positive on +x.
        let q = Direction::from_cartesian([1.0, 1.0, 0.0]);
        assert!(q.slm(2, -2) > 0.0);
        assert!(Direction::from_cartesian([1.0, 0.0, 0.0]).slm(2, 2) > 0.0);
    }

    #[test]
    fn real_harmonics_are_the_stated_combinations_of_complex_ones() {
        let d = Direction::from_angles(1.1, 2.3);
        for l in 0..=5u32 {
            for m in -(l as i32)..=(l as i32) {
                let (e, n) = complex_expansion(m, Basis::Real);
                let mut via = Complex::ZERO;
                for &(mk, ck) in &e[..n] {
                    via += ck * d.ylm(l, mk);
                }
                let s = d.slm(l, m);
                assert!(
                    (via.re - s).abs() < 1e-14 && via.im.abs() < 1e-14,
                    "l={l} m={m}: {via:?} vs {s}"
                );
            }
        }
    }

    #[test]
    fn overlaps_are_unitary() {
        // Within each l, the real basis is an orthonormal rotation of the
        // complex one.
        let l = 3i32;
        for a in -l..=l {
            for b in -l..=l {
                for (ba, bb) in [
                    (Basis::Real, Basis::Real),
                    (Basis::Complex, Basis::Complex),
                ] {
                    let o = angular_overlap(a, ba, b, bb);
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!((o.re - want).abs() < 1e-15 && o.im.abs() < 1e-15);
                }
            }
            // A real function's weights over the complex basis sum to 1.
            let w: f64 = (-l..=l)
                .map(|b| angular_overlap(b, Basis::Complex, a, Basis::Real).norm_sqr())
                .sum();
            assert!((w - 1.0).abs() < 1e-15);
        }
        // <p_+1 | p_x> = -1/sqrt 2 (Condon-Shortley).
        let o = angular_overlap(1, Basis::Complex, 1, Basis::Real);
        assert!((o.re + FRAC_1_SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn out_of_range_m_is_zero_and_poles_are_finite() {
        assert_eq!(spherical_harmonic(1, 2, 0.4, 0.1), Complex::ZERO);
        assert_eq!(real_spherical_harmonic(1, -2, 0.4, 0.1), 0.0);
        let north = Direction::from_cartesian([0.0, 0.0, 3.0]);
        assert_eq!(north.ylm(3, 1), Complex::ZERO);
        let origin = Direction::from_cartesian([0.0; 3]);
        assert_eq!(origin.cos_theta(), 1.0);
        // theta outside [0, pi] is folded onto the same point.
        let a = spherical_harmonic(2, 1, -0.7, 0.4);
        let b = spherical_harmonic(2, 1, 0.7, 0.4 + PI);
        assert!((a - b).abs() < 1e-14);
        let d = Direction::from_angles(0.5, 0.0);
        assert!((d.sin_theta() - 0.5f64.sin()).abs() < 1e-16);
    }

    #[test]
    fn azimuthal_cdf_is_consistent() {
        for m in [-3, -1, 1, 2] {
            let (c0, _) = real_azimuthal_cdf(m, 0.0);
            let (c1, _) = real_azimuthal_cdf(m, 2.0 * PI);
            assert!(c0.abs() < 1e-15 && (c1 - 1.0).abs() < 1e-15);
            let h = 1e-6;
            let (a, _) = real_azimuthal_cdf(m, 1.0 - h);
            let (b, _) = real_azimuthal_cdf(m, 1.0 + h);
            let (_, pdf) = real_azimuthal_cdf(m, 1.0);
            assert!(((b - a) / (2.0 * h) - pdf).abs() < 1e-8);
        }
    }
}
