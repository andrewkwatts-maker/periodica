//====== periodica/rust/periodica-qm/src/hydrogenic/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # hydrogenic
//!
//! Exact eigenstates of one electron bound to a point nucleus of charge `Z`:
//! the hydrogen atom, He+, Li2+, ... and any screened-hydrogenic model with a
//! fractional effective charge.
//!
//! ## Units
//!
//! Hartree atomic units throughout: lengths in bohr (`a0`), energies in
//! hartree (`E_h`), time in `hbar / E_h`. [`HydrogenLike::transition_wavelength`]
//! converts to SI with CODATA 2022 constants.
//!
//! ## Model
//!
//! ```text
//! psi_nlm(r, theta, phi) = R_nl(r) A_lm(theta, phi),       A = Y_lm or S_lm
//! R_nl(r) = N rho^l e^{-rho/2} L_{n-l-1}^{2l+1}(rho),     rho = 2 Z r / (n a)
//! N       = sqrt((2Z/(n a))^3 (n-l-1)! / (2n (n+l)!))
//! E_n     = -(mu / m_e) Z^2 / (2 n^2) E_h
//! ```
//!
//! `a = a0 m_e / mu` is the reduced-mass Bohr radius: with a finite nuclear
//! mass `M`, `mu = m_e M / (m_e + M)` and the electron orbits the centre of
//! mass. Supplying a nuclear mass switches the correction on (it shifts
//! hydrogen's energies by 5.4e-4, far more than relativity); [`NuclearMass::Infinite`]
//! gives the textbook `a = a0`.
//!
//! The normaliser is evaluated through `ln Gamma`, so it neither overflows nor
//! underflows for large `n`. (The legacy Python `orbital_clouds.py` used
//! `((n+l)!)^3`, a convention-mixing error that made `integral R^2 r^2 dr`
//! equal 0.25 for 2s; it is not ported.)

mod radial;

pub use radial::RadialDistribution;

use periodica_mat::constants::{
    ELECTRON_MASS_U, HARTREE_ENERGY, PLANCK, PROTON_ELECTRON_MASS_RATIO, SPEED_OF_LIGHT,
};

use crate::complex::Complex;
use crate::error::QmError;
use crate::label::ModelLabel;
use crate::numeric::pow_u;
use crate::special::{laguerre, laguerre_roots, ln_gamma, Basis, Direction};

/// The nucleus mass model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NuclearMass {
    /// Infinitely heavy nucleus: `mu = m_e`, `a = a0`.
    Infinite,
    /// Nuclear mass in electron masses (`m_p / m_e = 1836.15...` for hydrogen).
    ElectronMasses(f64),
}

/// A one-electron atom or ion: nuclear charge and reduced mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrogenLike {
    z: f64,
    nucleus: NuclearMass,
    /// `mu / m_e`.
    mass_ratio: f64,
}

impl HydrogenLike {
    /// A nucleus of charge `z` (> 0; fractional for screened models) and the
    /// given mass model.
    ///
    /// # Errors
    ///
    /// [`QmError::InvalidCharge`] or [`QmError::InvalidNuclearMass`] for
    /// non-finite or non-positive values.
    pub fn new(z: f64, nucleus: NuclearMass) -> Result<Self, QmError> {
        if !z.is_finite() || z <= 0.0 {
            return Err(QmError::InvalidCharge(z));
        }
        let mass_ratio = match nucleus {
            NuclearMass::Infinite => 1.0,
            NuclearMass::ElectronMasses(m) => {
                if !m.is_finite() || m <= 0.0 {
                    return Err(QmError::InvalidNuclearMass(m));
                }
                // mu/m_e = M/(M + 1) = 1/(1 + 1/M), the form that keeps every bit.
                1.0 / (1.0 + 1.0 / m)
            }
        };
        Ok(Self {
            z,
            nucleus,
            mass_ratio,
        })
    }

    /// Hydrogen (protium): `Z = 1` with the CODATA 2022 proton mass.
    pub fn hydrogen() -> Self {
        Self {
            z: 1.0,
            nucleus: NuclearMass::ElectronMasses(PROTON_ELECTRON_MASS_RATIO),
            mass_ratio: 1.0 / (1.0 + 1.0 / PROTON_ELECTRON_MASS_RATIO),
        }
    }

    /// Charge `z` with an infinitely heavy nucleus.
    ///
    /// # Errors
    ///
    /// [`QmError::InvalidCharge`] for non-finite or non-positive `z`.
    pub fn infinite_mass(z: f64) -> Result<Self, QmError> {
        Self::new(z, NuclearMass::Infinite)
    }

    /// Charge `z` with a nuclear mass given in unified atomic mass units.
    /// Pass the **nuclear** mass (atomic mass minus `Z` electron masses and
    /// their binding energy), not the atomic mass.
    ///
    /// # Errors
    ///
    /// As [`HydrogenLike::new`].
    pub fn with_nuclear_mass_u(z: f64, nuclear_mass_u: f64) -> Result<Self, QmError> {
        Self::new(
            z,
            NuclearMass::ElectronMasses(nuclear_mass_u / ELECTRON_MASS_U),
        )
    }

    /// Nuclear charge `Z`.
    #[inline]
    pub fn z(&self) -> f64 {
        self.z
    }

    /// The nucleus mass model.
    #[inline]
    pub fn nucleus(&self) -> NuclearMass {
        self.nucleus
    }

    /// `mu / m_e` (1 for an infinite nuclear mass).
    #[inline]
    pub fn reduced_mass_ratio(&self) -> f64 {
        self.mass_ratio
    }

    /// The length unit of the wavefunctions, `a = a0 m_e / mu`, in bohr.
    #[inline]
    pub fn length_scale(&self) -> f64 {
        1.0 / self.mass_ratio
    }

    /// Bound-state energy `E_n = -(mu/m_e) Z^2 / (2 n^2)` in hartree; `NaN`
    /// for `n = 0`.
    pub fn energy(&self, n: u32) -> f64 {
        if n == 0 {
            return f64::NAN;
        }
        let nf = f64::from(n);
        -self.mass_ratio * self.z * self.z / (2.0 * nf * nf)
    }

    /// Photon energy `E_upper - E_lower` in hartree (positive for emission).
    pub fn transition_energy(&self, n_upper: u32, n_lower: u32) -> f64 {
        self.energy(n_upper) - self.energy(n_lower)
    }

    /// Vacuum wavelength `h c / (E_upper - E_lower)` in metres of the photon
    /// emitted in `n_upper -> n_lower`. Hydrogen 2 -> 1 is Lyman-alpha,
    /// 121.567 nm (NIST ASD, fine-structure-weighted).
    pub fn transition_wavelength(&self, n_upper: u32, n_lower: u32) -> f64 {
        PLANCK * SPEED_OF_LIGHT / (self.transition_energy(n_upper, n_lower) * HARTREE_ENERGY)
    }

    /// The eigenstate `orbital` of this atom, ready to evaluate.
    pub fn orbital(&self, orbital: Orbital) -> HydrogenicOrbital {
        HydrogenicOrbital::new(*self, orbital)
    }

    /// Model label for the `(n, l)` eigenstate: model text, numerical gate,
    /// and the `(Z alpha)^2` relativistic error (see [`ModelLabel::hydrogenic`]).
    pub fn model_label(&self, n: u32, l: u32) -> ModelLabel {
        let reduced = !matches!(self.nucleus, NuclearMass::Infinite);
        ModelLabel::hydrogenic(self.z, n, l, reduced)
    }
}

/// Validated quantum numbers `(n, l, m)` and the angular basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Orbital {
    n: u32,
    l: u32,
    m: i32,
    basis: Basis,
}

impl Orbital {
    /// `n >= 1`, `0 <= l < n`, `|m| <= l`. For [`Basis::Real`], `m` labels the
    /// real harmonic: `m > 0` the `cos(m phi)` member (p_x, d_xz, d_x2-y2),
    /// `m < 0` the `sin(|m| phi)` member (p_y, d_yz, d_xy).
    ///
    /// # Errors
    ///
    /// [`QmError::InvalidQuantumNumbers`] when the constraints fail.
    pub fn new(n: u32, l: u32, m: i32, basis: Basis) -> Result<Self, QmError> {
        if n == 0 || l >= n || m.unsigned_abs() > l {
            return Err(QmError::InvalidQuantumNumbers { n, l, m });
        }
        Ok(Self { n, l, m, basis })
    }

    /// A complex (`L_z` eigenstate) orbital.
    ///
    /// # Errors
    ///
    /// As [`Orbital::new`].
    pub fn complex(n: u32, l: u32, m: i32) -> Result<Self, QmError> {
        Self::new(n, l, m, Basis::Complex)
    }

    /// A real (chemistry convention) orbital.
    ///
    /// # Errors
    ///
    /// As [`Orbital::new`].
    pub fn real(n: u32, l: u32, m: i32) -> Result<Self, QmError> {
        Self::new(n, l, m, Basis::Real)
    }

    /// Principal quantum number.
    #[inline]
    pub fn n(&self) -> u32 {
        self.n
    }
    /// Orbital angular momentum quantum number.
    #[inline]
    pub fn l(&self) -> u32 {
        self.l
    }
    /// Magnetic (complex) or real-harmonic index.
    #[inline]
    pub fn m(&self) -> i32 {
        self.m
    }
    /// Angular basis.
    #[inline]
    pub fn basis(&self) -> Basis {
        self.basis
    }
}

/// The radial function `R_nl` with its constants precomputed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RadialFunction {
    pub n: u32,
    pub l: u32,
    /// `2 Z / (n a)`, so `rho = scale * r`.
    pub scale: f64,
    /// `ln N`.
    pub ln_norm: f64,
    /// `N`.
    pub norm: f64,
}

impl RadialFunction {
    pub(crate) fn new(atom: &HydrogenLike, n: u32, l: u32) -> Self {
        let nf = f64::from(n);
        let scale = 2.0 * atom.z / (nf * atom.length_scale());
        let ln_norm = 0.5
            * (3.0 * libm::log(scale) + ln_gamma(f64::from(n - l)) - libm::log(2.0 * nf)
                - ln_gamma(f64::from(n + l + 1)));
        Self {
            n,
            l,
            scale,
            ln_norm,
            norm: libm::exp(ln_norm),
        }
    }

    #[inline]
    pub(crate) fn laguerre_order(&self) -> u32 {
        self.n - self.l - 1
    }

    #[inline]
    pub(crate) fn alpha(&self) -> f64 {
        f64::from(2 * self.l + 1)
    }

    /// `R_nl(r)`; `r < 0` is treated as `|r|`.
    ///
    /// Formed directly as `N rho^l L(rho) e^{-rho/2}`, each factor good to an
    /// ulp or so. Only when an intermediate leaves the normal range -- the
    /// far tail, where `e^{-rho/2}` goes subnormal, or `rho^l` underflowing
    /// next to the nucleus for large `l` -- is it re-formed in log space,
    /// whose error grows like `|ln R| * eps`.
    pub(crate) fn eval(&self, r: f64) -> f64 {
        let rho = self.scale * r.abs();
        if rho.is_nan() {
            return f64::NAN;
        }
        let lag = laguerre(self.laguerre_order(), self.alpha(), rho);
        if lag == 0.0 {
            return 0.0;
        }
        // e^{-rho/2} as the square of e^{-rho/4}: that stays a normal number
        // out to rho = 2830, where e^{-rho/2} alone would already be
        // subnormal (and imprecise) beyond rho = 1416.
        let quarter = libm::exp(-0.25 * rho);
        let head = self.norm * pow_u(rho, self.l);
        let partial = head * lag * quarter;
        let direct = partial * quarter;
        let normal = |v: f64| v.is_finite() && v.abs() >= f64::MIN_POSITIVE;
        if normal(direct) && normal(partial) && normal(head) && normal(quarter) {
            return direct;
        }
        if rho == 0.0 {
            // R(0) = 0 for l > 0 (head is exactly zero).
            return direct;
        }
        if !lag.is_finite() {
            // A polynomial large enough to overflow sits where e^{-rho/2}
            // underflowed long ago: the true value is 0 to f64 precision.
            return 0.0;
        }
        let ln_mag = self.ln_norm + f64::from(self.l) * libm::log(rho) - 0.5 * rho
            + libm::log(lag.abs());
        libm::exp(ln_mag).copysign(lag)
    }
}

/// One hydrogen-like eigenstate, ready to evaluate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrogenicOrbital {
    atom: HydrogenLike,
    orbital: Orbital,
    radial: RadialFunction,
}

impl HydrogenicOrbital {
    /// Precompute the normaliser and length scale of `orbital` in `atom`.
    pub fn new(atom: HydrogenLike, orbital: Orbital) -> Self {
        Self {
            radial: RadialFunction::new(&atom, orbital.n, orbital.l),
            atom,
            orbital,
        }
    }

    /// The atom this orbital belongs to.
    #[inline]
    pub fn atom(&self) -> &HydrogenLike {
        &self.atom
    }

    /// The quantum numbers.
    #[inline]
    pub fn orbital(&self) -> Orbital {
        self.orbital
    }

    /// `E_n` in hartree.
    #[inline]
    pub fn energy(&self) -> f64 {
        self.atom.energy(self.orbital.n)
    }

    /// The radial function `R_nl(r)`, `r` in bohr, in bohr^{-3/2}.
    #[inline]
    pub fn radial(&self, r: f64) -> f64 {
        self.radial.eval(r)
    }

    /// The radial probability density `P(r) = R_nl(r)^2 r^2` (per bohr).
    #[inline]
    pub fn radial_probability_density(&self, r: f64) -> f64 {
        let v = self.radial.eval(r) * r;
        v * v
    }

    /// The angular factor (`Y_lm` or `S_lm`) in `direction`.
    #[inline]
    pub fn angular(&self, direction: &Direction) -> Complex {
        direction.harmonic(self.orbital.l, self.orbital.m, self.orbital.basis)
    }

    /// `psi(p)` at Cartesian `p` (bohr, nucleus at the origin), in bohr^{-3/2}.
    /// Real orbitals return a zero imaginary part.
    pub fn psi(&self, p: [f64; 3]) -> Complex {
        let r = libm::hypot(libm::hypot(p[0], p[1]), p[2]);
        self.psi_polar(r, &Direction::from_cartesian(p))
    }

    /// `psi` at distance `r` (bohr) in `direction`: lets a superposition share
    /// one direction between all its terms.
    #[inline]
    pub fn psi_polar(&self, r: f64, direction: &Direction) -> Complex {
        self.angular(direction) * self.radial.eval(r)
    }

    /// Probability density `|psi(p)|^2` in bohr^{-3}.
    #[inline]
    pub fn density(&self, p: [f64; 3]) -> f64 {
        self.psi(p).norm_sqr()
    }

    /// `<r> = (a / 2Z) (3 n^2 - l(l+1))`, bohr.
    pub fn mean_radius(&self) -> f64 {
        let (n, l, a, z) = self.nlaz();
        a / (2.0 * z) * (3.0 * n * n - l * (l + 1.0))
    }

    /// `<r^2> = (a^2 n^2 / 2 Z^2) (5 n^2 + 1 - 3 l(l+1))`, bohr^2.
    pub fn mean_radius_squared(&self) -> f64 {
        let (n, l, a, z) = self.nlaz();
        a * a * n * n / (2.0 * z * z) * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1.0))
    }

    /// `<1/r> = Z / (a n^2)`, bohr^{-1} (the virial theorem in disguise).
    pub fn mean_inverse_radius(&self) -> f64 {
        let (n, _, a, z) = self.nlaz();
        z / (a * n * n)
    }

    fn nlaz(&self) -> (f64, f64, f64, f64) {
        (
            f64::from(self.orbital.n),
            f64::from(self.orbital.l),
            self.atom.length_scale(),
            self.atom.z,
        )
    }

    /// The `n - l - 1` radial nodes (zeros of `R_nl` at `r > 0`), ascending,
    /// in bohr: the zeros of `L_{n-l-1}^{2l+1}(rho)` mapped by `r = rho / scale`.
    pub fn radial_nodes(&self) -> Vec<f64> {
        laguerre_roots(self.radial.laguerre_order(), self.radial.alpha())
            .into_iter()
            .map(|rho| rho / self.radial.scale)
            .collect()
    }

    /// The radial distribution of `r` (closed-form CDF, quantile function).
    pub fn radial_distribution(&self) -> RadialDistribution {
        RadialDistribution::new(self.radial)
    }

    /// The model label of this eigenstate.
    pub fn model_label(&self) -> ModelLabel {
        self.atom.model_label(self.orbital.n, self.orbital.l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn construction_validates() {
        assert!(Orbital::complex(0, 0, 0).is_err());
        assert!(Orbital::complex(2, 2, 0).is_err());
        assert!(Orbital::real(3, 1, -2).is_err());
        let e = Orbital::complex(1, 1, 0).unwrap_err();
        assert!(e.to_string().contains("n=1, l=1"));
        assert!(HydrogenLike::infinite_mass(0.0).is_err());
        assert!(HydrogenLike::infinite_mass(f64::NAN).is_err());
        assert!(HydrogenLike::new(1.0, NuclearMass::ElectronMasses(-3.0)).is_err());
        assert!(HydrogenLike::new(1.0, NuclearMass::ElectronMasses(f64::INFINITY)).is_err());
        let o = Orbital::real(3, 2, -1).unwrap();
        assert_eq!((o.n(), o.l(), o.m(), o.basis()), (3, 2, -1, Basis::Real));
    }

    #[test]
    fn hydrogen_ground_state_closed_form() {
        // R_10 = 2 (Z/a)^{3/2} e^{-Zr/a} ; psi_100 = e^{-r}/sqrt(pi) at a = Z = 1.
        let h = HydrogenLike::infinite_mass(1.0).unwrap();
        let o = h.orbital(Orbital::complex(1, 0, 0).unwrap());
        for &r in &[0.0, 0.5, 1.0, 3.0, 10.0] {
            assert!((o.radial(r) - 2.0 * libm::exp(-r)).abs() < 1e-15);
            let p = o.psi([r, 0.0, 0.0]);
            assert!((p.re - libm::exp(-r) / PI.sqrt()).abs() < 1e-15 && p.im == 0.0);
        }
        assert_eq!(o.energy(), -0.5);
        assert_eq!(o.mean_radius(), 1.5);
        // 2p_z = (1/(4 sqrt(2 pi))) z e^{-r/2}.
        let pz = h.orbital(Orbital::real(2, 1, 0).unwrap());
        let v = pz.psi([0.3, -0.2, 1.1]).re;
        let r = (0.09f64 + 0.04 + 1.21).sqrt();
        let want = 1.1 * (-r / 2.0).exp() / (4.0 * (2.0 * PI).sqrt());
        assert!((v - want).abs() < 1e-15, "{v} vs {want}");
    }

    #[test]
    fn reduced_mass_scales_lengths_and_energies() {
        let h = HydrogenLike::hydrogen();
        let mu = h.reduced_mass_ratio();
        assert!((mu - 0.999_455_679_4).abs() < 1e-10, "{mu}");
        assert!((h.length_scale() - 1.0 / mu).abs() < 1e-16);
        assert!((h.energy(1) + 0.5 * mu).abs() < 1e-16);
        assert!(h.energy(0).is_nan());
        // Deuterium nucleus 2.013553212544 u.
        let d = HydrogenLike::with_nuclear_mass_u(1.0, 2.013_553_212_544).unwrap();
        assert!(d.reduced_mass_ratio() > mu && d.reduced_mass_ratio() < 1.0);
        assert!(matches!(h.nucleus(), NuclearMass::ElectronMasses(_)));
        assert!(h.model_label(1, 0).model.contains("reduced mass"));
    }

    #[test]
    fn far_tail_is_finite_and_tiny() {
        let h = HydrogenLike::infinite_mass(1.0).unwrap();
        for (n, l) in [(1, 0), (6, 2), (40, 39), (60, 3)] {
            let o = h.orbital(Orbital::complex(n, l, 0).unwrap());
            let n2 = f64::from(n * n);
            for r in [100.0 * n2, 1e4 * n2, 1e300] {
                let v = o.radial(r);
                assert!(v.is_finite() && v.abs() < 1e-30, "n={n} l={l} r={r}: {v}");
            }
        }
        // At rho = 1440, e^{-rho/2} is subnormal but R_{30,2} is a normal
        // number: the log-space branch must recover it. Reference: the same
        // product with the exponential split in two normal halves.
        let o = h.orbital(Orbital::complex(30, 2, 0).unwrap());
        let rf = o.radial;
        let rho = 1440.0;
        let r = rho / rf.scale;
        let reference = rf.norm
            * pow_u(rho, rf.l)
            * laguerre(rf.laguerre_order(), rf.alpha(), rho)
            * libm::exp(-0.25 * rho)
            * libm::exp(-0.25 * rho);
        let v = o.radial(r);
        assert!(reference.abs() > f64::MIN_POSITIVE, "{reference}");
        assert!((v / reference - 1.0).abs() < 1e-12, "{v} vs {reference}");
        // R(0) = 0 for l > 0 and finite for l = 0; NaN propagates.
        assert_eq!(o.radial(0.0), 0.0);
        assert!(h.orbital(Orbital::complex(3, 0, 0).unwrap()).radial(0.0) > 0.0);
        assert!(o.radial(f64::NAN).is_nan());
    }
}
