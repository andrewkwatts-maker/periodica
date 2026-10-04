//====== periodica/rust/periodica-qm/src/complex.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # complex
//!
//! A minimal `f64` complex number for wavefunction amplitudes.
//!
//! ## Why not `num-complex`
//!
//! Wavefunctions need a dozen operations: `+ - *`, conjugate, `|z|^2`, and
//! `e^{i theta}`. `num-complex` would supply them, but its transcendental
//! methods (`exp`, `from_polar`, `arg`) dispatch through `num-traits::Float`
//! to std's `f64::sin`/`cos`/`atan2`, which call the platform C library and
//! differ in the last bit between Windows, macOS and Linux. Every phase
//! here goes through `libm` instead, so `psi(x, t)` -- and every golden image
//! and sampler stream built on it -- is bit-identical across operating
//! systems. The arithmetic itself is plain IEEE-754 and needs no crate.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number `re + i im`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// `0 + 0i`.
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    /// `1 + 0i`.
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    /// `0 + 1i`.
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    /// `re + i im`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// A purely real number.
    #[inline]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// `e^{i theta} = cos(theta) + i sin(theta)`, via `libm`.
    #[inline]
    pub fn cis(theta: f64) -> Self {
        let (s, c) = libm::sincos(theta);
        Self { re: c, im: s }
    }

    /// `r e^{i theta}`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::cis(theta) * r
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// `|z|^2`, without a square root.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// `|z|`, overflow-safe (`libm::hypot`).
    #[inline]
    pub fn abs(self) -> f64 {
        libm::hypot(self.re, self.im)
    }

    /// The argument in `(-pi, pi]` (`libm::atan2`).
    #[inline]
    pub fn arg(self) -> f64 {
        libm::atan2(self.im, self.re)
    }

    /// Multiply by a real scalar.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Whether both parts are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl From<f64> for Complex {
    #[inline]
    fn from(re: f64) -> Self {
        Self::real(re)
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, s: f64) -> Complex {
        self.scale(s)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn mul(self, z: Complex) -> Complex {
        z.scale(self)
    }
}

impl Div<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn div(self, s: f64) -> Complex {
        Complex::new(self.re / s, self.im / s)
    }
}

impl Neg for Complex {
    type Output = Complex;
    #[inline]
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, o: Complex) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, o: Complex) {
        self.re -= o.re;
        self.im -= o.im;
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, o: Complex) {
        *self = *self * o;
    }
}

impl MulAssign<f64> for Complex {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        self.re *= s;
        self.im *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn arithmetic_matches_hand_values() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        assert_eq!(a + b, Complex::new(4.0, 1.0));
        assert_eq!(a - b, Complex::new(-2.0, 3.0));
        assert_eq!(a * b, Complex::new(5.0, 5.0));
        assert_eq!(a.conj(), Complex::new(1.0, -2.0));
        assert_eq!(a.norm_sqr(), 5.0);
        assert_eq!(2.0 * a, Complex::new(2.0, 4.0));
        assert_eq!(a / 2.0, Complex::new(0.5, 1.0));
        assert_eq!(-a, Complex::new(-1.0, -2.0));
        assert_eq!(Complex::I * Complex::I, -Complex::ONE);
        let mut c = a;
        c += b;
        c -= b;
        c *= Complex::ONE;
        c *= 1.0;
        assert_eq!(c, a);
        assert_eq!(Complex::from(3.0), Complex::real(3.0));
    }

    #[test]
    fn polar_form_round_trips() {
        let z = Complex::from_polar(2.0, FRAC_PI_2);
        assert!(z.re.abs() < 1e-15 && (z.im - 2.0).abs() < 1e-15);
        assert!((z.abs() - 2.0).abs() < 1e-15);
        assert!((z.arg() - FRAC_PI_2).abs() < 1e-15);
        assert!((Complex::cis(PI).re + 1.0).abs() < 1e-15);
        assert!(Complex::cis(0.3).is_finite());
        assert!(!Complex::new(f64::NAN, 0.0).is_finite());
    }
}
