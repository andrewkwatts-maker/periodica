//====== periodica/rust/periodica-qm/src/state.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # state
//!
//! Superpositions of hydrogen-like eigenstates and their exact time evolution
//!
//! ```text
//! psi(x, t) = sum_k c_k psi_k(x) e^{-i E_k t / hbar}
//! ```
//!
//! in atomic units (`hbar = 1`, `t` in `hbar / E_h = 24.188 843 265 864 as`).
//!
//! ## Phases in f64, reduced mod 2 pi
//!
//! After a few thousand periods `E_k t` is large and `cos(E_k t)` would lose
//! digits to argument reduction inside `cos`, differently on each platform.
//! [`phase`] reduces `E t` into `[0, 2 pi)` with a two-word `2 pi` and fused
//! multiply-adds (`libm::fma`, exactly rounded in software), then
//! `libm::sincos` sees a small argument. The only error left is the rounding
//! of the product `E t` itself.
//!
//! ## Time average and degeneracy
//!
//! Over long times the cross terms between states of different energy
//! average to zero, but terms of *equal* energy keep a fixed relative phase
//! and their interference survives. [`State::time_averaged_density`] groups
//! terms whose energies agree to [`DEGENERACY_TOLERANCE`] and sums
//! `|sum_{k in g} c_k psi_k|^2` over groups. A degenerate superposition such
//! as 2s + 2p_z (the hybrid that the Stark effect selects) is therefore
//! stationary: its density is lopsided and does not move.
//!
//! In the real atom fine structure splits 2s from 2p_{3/2} by ~0.365 cm^-1,
//! so that "stationary" hybrid actually beats with a ~90 ps period; the
//! [`crate::ModelLabel`] states the relativistic error that this model omits.

use core::f64::consts::TAU;

use periodica_mat::constants::ATOMIC_UNIT_OF_TIME;

use crate::complex::Complex;
use crate::error::QmError;
use crate::hydrogenic::{HydrogenLike, HydrogenicOrbital, Orbital};
use crate::label::ModelLabel;
use crate::special::{angular_overlap, Direction};

/// Energies closer than this (hartree) are treated as degenerate.
pub const DEGENERACY_TOLERANCE: f64 = 1e-12;

/// Coefficients whose squared norm is below this fraction of `sum |c_k|^2`
/// after interference are rejected as cancelling to zero.
const ZERO_NORM_FRACTION: f64 = 1e-24;

/// The low word of `2 pi`: `TAU_LO = 2 pi - TAU` to double precision.
const TAU_LO: f64 = 2.449_293_598_294_706_4e-16;

/// `E t mod 2 pi`, in `[0, 2 pi)`: the phase angle of `e^{-i E t}` (atomic
/// units). See the module docs for why this is done explicitly.
///
/// Non-finite products return `NaN`.
pub fn phase(energy: f64, t: f64) -> f64 {
    let x = energy * t;
    if !x.is_finite() {
        return f64::NAN;
    }
    let k = libm::round(x / TAU);
    let r = libm::fma(-k, TAU, x);
    let r = libm::fma(-k, TAU_LO, r);
    if r < 0.0 {
        r + TAU
    } else if r >= TAU {
        r - TAU
    } else {
        r
    }
}

/// Atomic units of time to seconds.
#[inline]
pub fn au_to_seconds(t_au: f64) -> f64 {
    t_au * ATOMIC_UNIT_OF_TIME
}

/// Seconds to atomic units of time.
#[inline]
pub fn seconds_to_au(t_s: f64) -> f64 {
    t_s / ATOMIC_UNIT_OF_TIME
}

/// One term `c_k psi_k` of a superposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term {
    /// Complex amplitude `c_k` at `t = 0`.
    pub coefficient: Complex,
    /// The eigenstate.
    pub orbital: Orbital,
}

impl Term {
    /// `coefficient * |orbital>`.
    pub fn new(coefficient: Complex, orbital: Orbital) -> Self {
        Self {
            coefficient,
            orbital,
        }
    }
}

/// Terms sharing one energy (to [`DEGENERACY_TOLERANCE`]).
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyGroup {
    /// The group's energy, hartree (that of its first term).
    pub energy: f64,
    /// Indices into [`State::terms`].
    pub terms: Vec<usize>,
}

/// A normalised superposition of eigenstates of one hydrogen-like atom.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    atom: HydrogenLike,
    terms: Vec<Term>,
    orbitals: Vec<HydrogenicOrbital>,
    energies: Vec<f64>,
    groups: Vec<EnergyGroup>,
}

impl State {
    /// Build and normalise `sum_k c_k |psi_k>`.
    ///
    /// The relative amplitudes and phases are kept; all coefficients are
    /// divided by `||psi||`, computed exactly from the overlaps (terms may
    /// repeat, or mix real and complex harmonics of the same `n, l`, which
    /// are not orthogonal).
    ///
    /// # Errors
    ///
    /// [`QmError::EmptyState`], [`QmError::NonFiniteCoefficient`], or
    /// [`QmError::ZeroNorm`] when the terms cancel.
    pub fn new(atom: HydrogenLike, terms: Vec<Term>) -> Result<Self, QmError> {
        if terms.is_empty() {
            return Err(QmError::EmptyState);
        }
        if let Some(index) = terms.iter().position(|t| !t.coefficient.is_finite()) {
            return Err(QmError::NonFiniteCoefficient { index });
        }
        let orbitals: Vec<HydrogenicOrbital> =
            terms.iter().map(|t| atom.orbital(t.orbital)).collect();
        let energies: Vec<f64> = orbitals.iter().map(HydrogenicOrbital::energy).collect();
        let groups = group_energies(&energies);
        let mut state = Self {
            atom,
            terms,
            orbitals,
            energies,
            groups,
        };
        let raw: f64 = state.terms.iter().map(|t| t.coefficient.norm_sqr()).sum();
        let norm2 = state.norm_squared();
        if norm2.is_nan() || norm2 <= ZERO_NORM_FRACTION * raw {
            return Err(QmError::ZeroNorm);
        }
        let inv = 1.0 / libm::sqrt(norm2);
        for t in &mut state.terms {
            t.coefficient = t.coefficient * inv;
        }
        Ok(state)
    }

    /// The single eigenstate `|orbital>` with coefficient 1.
    pub fn eigenstate(atom: HydrogenLike, orbital: Orbital) -> Self {
        Self::new(atom, vec![Term::new(Complex::ONE, orbital)])
            .expect("a single eigenstate has unit norm")
    }

    /// The atom.
    #[inline]
    pub fn atom(&self) -> &HydrogenLike {
        &self.atom
    }

    /// The normalised terms (coefficients at `t = 0`).
    #[inline]
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// The evaluators of each term's eigenstate, in term order.
    #[inline]
    pub fn orbitals(&self) -> &[HydrogenicOrbital] {
        &self.orbitals
    }

    /// `E_k` of each term, hartree.
    #[inline]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// The degenerate energy groups, ascending in energy.
    #[inline]
    pub fn energy_groups(&self) -> &[EnergyGroup] {
        &self.groups
    }

    /// `<psi_j | psi_k>` between two terms' eigenstates (not including the
    /// coefficients): `delta_{n n'} delta_{l l'}` times the angular overlap.
    pub fn term_overlap(&self, j: usize, k: usize) -> Complex {
        let (a, b) = (self.terms[j].orbital, self.terms[k].orbital);
        if a.n() != b.n() || a.l() != b.l() {
            return Complex::ZERO;
        }
        angular_overlap(a.m(), a.basis(), b.m(), b.basis())
    }

    /// `<psi|psi>` from the coefficients and exact overlaps; 1 after
    /// construction, and constant in time (overlapping terms are degenerate).
    pub fn norm_squared(&self) -> f64 {
        let mut acc = Complex::ZERO;
        for (j, tj) in self.terms.iter().enumerate() {
            for (k, tk) in self.terms.iter().enumerate() {
                let o = self.term_overlap(j, k);
                if o != Complex::ZERO {
                    acc += tj.coefficient.conj() * tk.coefficient * o;
                }
            }
        }
        acc.re
    }

    /// `<H> = sum_g E_g ||P_g psi||^2`, hartree.
    pub fn mean_energy(&self) -> f64 {
        let mut e = 0.0;
        for (j, tj) in self.terms.iter().enumerate() {
            for (k, tk) in self.terms.iter().enumerate() {
                let o = self.term_overlap(j, k);
                if o != Complex::ZERO {
                    e += self.energies[k] * (tj.coefficient.conj() * tk.coefficient * o).re;
                }
            }
        }
        e
    }

    /// `c_k e^{-i E_k t}` for every term (`t` in atomic units).
    pub fn coefficients_at(&self, t: f64) -> Vec<Complex> {
        self.terms
            .iter()
            .zip(&self.energies)
            .map(|(term, &e)| term.coefficient * Complex::cis(-phase(e, t)))
            .collect()
    }

    /// `psi(p, t)` at Cartesian `p` (bohr), `t` in atomic units.
    pub fn psi(&self, p: [f64; 3], t: f64) -> Complex {
        let coeffs = self.coefficients_at(t);
        self.psi_with(p, &coeffs)
    }

    /// `sum_k coeffs[k] psi_k(p)`: `psi` with precomputed time-dependent
    /// coefficients (see [`State::coefficients_at`]), for evaluating many
    /// points at one time.
    pub fn psi_with(&self, p: [f64; 3], coeffs: &[Complex]) -> Complex {
        let r = libm::hypot(libm::hypot(p[0], p[1]), p[2]);
        let dir = Direction::from_cartesian(p);
        let mut acc = Complex::ZERO;
        for (orb, &c) in self.orbitals.iter().zip(coeffs) {
            acc += c * orb.psi_polar(r, &dir);
        }
        acc
    }

    /// `|psi(p, t)|^2`, bohr^{-3}.
    pub fn density(&self, p: [f64; 3], t: f64) -> f64 {
        self.psi(p, t).norm_sqr()
    }

    /// The infinite-time average of `|psi(p, t)|^2`: interference survives
    /// only within degenerate groups (see the module docs).
    pub fn time_averaged_density(&self, p: [f64; 3]) -> f64 {
        let r = libm::hypot(libm::hypot(p[0], p[1]), p[2]);
        let dir = Direction::from_cartesian(p);
        self.groups
            .iter()
            .map(|g| {
                let mut acc = Complex::ZERO;
                for &k in &g.terms {
                    acc += self.terms[k].coefficient * self.orbitals[k].psi_polar(r, &dir);
                }
                acc.norm_sqr()
            })
            .sum()
    }

    /// Whether the density is time-independent (one energy group).
    pub fn is_stationary(&self) -> bool {
        self.groups.len() == 1
    }

    /// Beat periods `2 pi / |E_g - E_h|` (atomic units of time) for every pair
    /// of energy groups, ascending. Empty for a stationary state. Hydrogen
    /// 1s + 2p beats at `16.76 au = 0.4055 fs`.
    pub fn beat_periods(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for (i, a) in self.groups.iter().enumerate() {
            for b in &self.groups[i + 1..] {
                out.push(TAU / (b.energy - a.energy).abs());
            }
        }
        out.sort_by(f64::total_cmp);
        out
    }

    /// The least accurate term's [`ModelLabel`].
    pub fn model_label(&self) -> ModelLabel {
        let labels: Vec<ModelLabel> = self
            .orbitals
            .iter()
            .map(HydrogenicOrbital::model_label)
            .collect();
        ModelLabel::worst(&labels).expect("a state has at least one term")
    }
}

/// Group energies that agree to [`DEGENERACY_TOLERANCE`], ascending.
fn group_energies(energies: &[f64]) -> Vec<EnergyGroup> {
    let mut order: Vec<usize> = (0..energies.len()).collect();
    order.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]).then(a.cmp(&b)));
    let mut groups: Vec<EnergyGroup> = Vec::new();
    for k in order {
        match groups.last_mut() {
            Some(g) if (energies[k] - g.energy).abs() < DEGENERACY_TOLERANCE => g.terms.push(k),
            _ => groups.push(EnergyGroup {
                energy: energies[k],
                terms: vec![k],
            }),
        }
    }
    for g in &mut groups {
        g.terms.sort_unstable();
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::Basis;

    fn h() -> HydrogenLike {
        HydrogenLike::hydrogen()
    }

    fn orb(n: u32, l: u32, m: i32, b: Basis) -> Orbital {
        Orbital::new(n, l, m, b).unwrap()
    }

    #[test]
    fn phase_reduction_is_exact_for_whole_turns() {
        assert_eq!(phase(1.0, 0.0), 0.0);
        assert!((phase(1.0, 1.0) - 1.0).abs() < 1e-16);
        assert!((phase(-1.0, 1.0) - (TAU - 1.0)).abs() < 1e-15);
        // A million turns plus 0.5 rad: only the rounding of E*t remains.
        let t = 1e6 * TAU + 0.5;
        let p = phase(1.0, t);
        assert!((p - 0.5).abs() < 1e-9, "{p}");
        assert!(phase(f64::INFINITY, 1.0).is_nan());
        for x in [1e-3, 3.0, 7.0, 1e5, -42.0] {
            let p = phase(x, 1.0);
            assert!((0.0..TAU).contains(&p));
        }
    }

    #[test]
    fn construction_normalises_and_validates() {
        assert_eq!(State::new(h(), vec![]), Err(QmError::EmptyState));
        let bad = Term::new(Complex::new(f64::NAN, 0.0), orb(1, 0, 0, Basis::Complex));
        assert_eq!(
            State::new(h(), vec![bad]),
            Err(QmError::NonFiniteCoefficient { index: 0 })
        );
        let s = State::new(
            h(),
            vec![
                Term::new(Complex::real(3.0), orb(1, 0, 0, Basis::Complex)),
                Term::new(Complex::new(0.0, 4.0), orb(2, 1, 1, Basis::Complex)),
            ],
        )
        .unwrap();
        assert!((s.norm_squared() - 1.0).abs() < 1e-15);
        assert!((s.terms()[0].coefficient.re - 0.6).abs() < 1e-15);
        // p_x = (Y_-1 - Y_1)/sqrt 2 cancels exactly against its expansion.
        let r = core::f64::consts::FRAC_1_SQRT_2;
        let cancel = State::new(
            h(),
            vec![
                Term::new(Complex::ONE, orb(2, 1, 1, Basis::Real)),
                Term::new(Complex::real(r), orb(2, 1, 1, Basis::Complex)),
                Term::new(Complex::real(-r), orb(2, 1, -1, Basis::Complex)),
            ],
        );
        assert_eq!(cancel, Err(QmError::ZeroNorm));
        // Repeating a term is fine: it just doubles that amplitude.
        let rep = State::new(
            h(),
            vec![
                Term::new(Complex::ONE, orb(2, 0, 0, Basis::Real)),
                Term::new(Complex::ONE, orb(2, 0, 0, Basis::Real)),
            ],
        )
        .unwrap();
        assert!((rep.terms()[0].coefficient.re - 0.5).abs() < 1e-15);
    }

    #[test]
    fn degenerate_2s_2pz_is_stationary() {
        let s = State::new(
            h(),
            vec![
                Term::new(Complex::ONE, orb(2, 0, 0, Basis::Real)),
                Term::new(Complex::ONE, orb(2, 1, 0, Basis::Real)),
            ],
        )
        .unwrap();
        assert!(s.is_stationary());
        assert!(s.beat_periods().is_empty());
        let p = [0.4, -0.3, 1.7];
        let rho0 = s.density(p, 0.0);
        for t in [0.37, 10.0, 1234.5, 1e7] {
            let rho = s.density(p, t);
            assert!((rho / rho0 - 1.0).abs() < 1e-13, "t={t}: {rho} vs {rho0}");
        }
        // The average keeps the 2s-2p cross term: lopsided along z.
        let avg = s.time_averaged_density(p);
        assert!((avg / rho0 - 1.0).abs() < 1e-13);
        let up = s.time_averaged_density([0.0, 0.0, 2.0]);
        let down = s.time_averaged_density([0.0, 0.0, -2.0]);
        assert!((up - down).abs() > 1e-3 * up.max(down));
    }

    #[test]
    fn hydrogen_1s_2p_beats_at_0_405_fs() {
        let s = State::new(
            h(),
            vec![
                Term::new(Complex::ONE, orb(1, 0, 0, Basis::Real)),
                Term::new(Complex::ONE, orb(2, 1, 0, Basis::Real)),
            ],
        )
        .unwrap();
        let periods = s.beat_periods();
        assert_eq!(periods.len(), 1);
        let t_beat = periods[0];
        let mu = h().reduced_mass_ratio();
        assert!((t_beat - TAU / (0.375 * mu)).abs() < 1e-12);
        let fs = au_to_seconds(t_beat) * 1e15;
        assert!((fs - 0.405).abs() < 1e-3, "{fs} fs");
        assert!((seconds_to_au(au_to_seconds(3.0)) - 3.0).abs() < 1e-15);
        // The density is periodic with that period, and moves within it.
        let p = [0.2, 0.1, 1.3];
        let d0 = s.density(p, 0.0);
        assert!((s.density(p, 7.0 * t_beat) / d0 - 1.0).abs() < 1e-12);
        assert!((s.density(p, 0.5 * t_beat) / d0 - 1.0).abs() > 1e-2);
        // <H> is the weighted mean of the two levels.
        let want = 0.5 * (h().energy(1) + h().energy(2));
        assert!((s.mean_energy() - want).abs() < 1e-15);
        assert!(!s.is_stationary());
        assert!(s.model_label().physical_error_vs_reality > 1e-5);
    }

    #[test]
    fn grouping_and_overlaps() {
        let s = State::new(
            h(),
            vec![
                Term::new(Complex::ONE, orb(3, 2, 1, Basis::Real)),
                Term::new(Complex::ONE, orb(1, 0, 0, Basis::Complex)),
                Term::new(Complex::ONE, orb(3, 0, 0, Basis::Complex)),
                Term::new(Complex::I, orb(2, 1, -1, Basis::Complex)),
            ],
        )
        .unwrap();
        let g = s.energy_groups();
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].terms, vec![1]);
        assert_eq!(g[1].terms, vec![3]);
        assert_eq!(g[2].terms, vec![0, 2]);
        assert_eq!(s.beat_periods().len(), 3);
        assert_eq!(s.term_overlap(0, 2), Complex::ZERO);
        assert_eq!(s.term_overlap(1, 1), Complex::ONE);
        let e = State::eigenstate(h(), orb(2, 1, 1, Basis::Real));
        assert!(e.is_stationary());
        assert_eq!(e.orbitals().len(), 1);
        assert_eq!(e.energies()[0], h().energy(2));
        assert_eq!(e.atom(), &h());
        // psi_with matches psi.
        let p = [0.5, 0.5, -0.25];
        let c = s.coefficients_at(2.5);
        assert_eq!(s.psi(p, 2.5), s.psi_with(p, &c));
    }
}
