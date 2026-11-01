//====== periodica/rust/periodica-qm/src/sampling.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # sampling
//!
//! Born-rule measurement sampling: positions drawn with probability density
//! `|psi(x, t)|^2`. These drive the app's "Collapse" mode, where the hit
//! histogram provably converges to the rendered density.
//!
//! ## Reproducibility
//!
//! All randomness comes from [`SplitMix64`], a thin stream over
//! [`periodica_runtime::sampling::splitmix64`]; uniforms use its top 53 bits.
//! No `rand` distributions are used (their algorithms are allowed to change
//! between versions) and every transcendental is `libm`, so the same seed
//! gives the same points, bit for bit, on every platform.
//!
//! ## Samplers
//!
//! * [`HydrogenicSampler`] -- exact and separable for one eigenstate:
//!   `r` by inverting the closed-form radial CDF (safeguarded Newton),
//!   `cos theta` by inverting the polynomial CDF of `Pbar_l^m(cos theta)^2`
//!   (evaluated exactly with an `(l+1)`-point Gauss-Legendre rule), and `phi`
//!   uniform for complex orbitals or by inverting
//!   `phi/2pi +- sin(2|m|phi)/(4 pi |m|)` for real ones. Three uniforms per
//!   point, no rejection.
//! * [`SuperpositionSampler`] -- mixture rejection for `sum c_k psi_k`: pick
//!   `k` with probability `|c_k|^2 / sum |c|^2`, draw `x ~ |psi_k|^2` exactly,
//!   accept with probability `|psi(x)|^2 / (K sum_k |c_k|^2 |psi_k(x)|^2)`.
//!   Cauchy-Schwarz (`|sum_k a_k|^2 <= K sum_k |a_k|^2`) bounds that ratio by
//!   1, so accepted points are independent exact draws from `|psi|^2`; for
//!   orthonormal terms the acceptance rate is `1/K`.
//! * [`GammaSampler`] -- Marsaglia and Tsang (2000) gamma variates, for
//!   Slater-type radial densities (`r ~ Gamma(2n + 1, 1/(2 zeta))`) and
//!   per-axis Gaussian-type densities in later physics tiers.

use core::f64::consts::TAU;

use periodica_runtime::sampling::{splitmix64, unit_from_bits};

use crate::complex::Complex;
use crate::error::QmError;
use crate::hydrogenic::{HydrogenicOrbital, RadialDistribution};
use crate::numeric::solve_bracketed;
use crate::quadrature::gauss_legendre;
use crate::special::harmonics::real_azimuthal_cdf;
use crate::special::{legendre_normalised, Basis, Direction};
use crate::state::State;

/// Cap on proposals per accepted superposition sample. With orthonormal
/// terms the expected count is `K <= ~20`; reaching this cap means the
/// terms nearly cancel.
pub const MAX_REJECTION_ATTEMPTS: u64 = 1 << 24;

/// A reproducible stream of uniforms and normals over the runtime's
/// `splitmix64` (Steele, Lea and Flood 2014).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A stream seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }

    /// Uniform on `[0, 1)`: a multiple of `2^-53`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        unit_from_bits(self.next_u64())
    }

    /// Uniform on `(0, 1]` (never 0, so safe under `ln` and as a tail
    /// probability).
    #[inline]
    pub fn next_f64_open_closed(&mut self) -> f64 {
        1.0 - self.next_f64()
    }

    /// A standard normal variate (Marsaglia polar method; one of each pair
    /// is used, so the stream position stays a simple function of draws).
    pub fn next_standard_normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * libm::sqrt(-2.0 * libm::log(s) / s);
            }
        }
    }
}

/// The distribution of `x = cos theta` for angular momentum `(l, |m|)`:
/// density `Pbar_l^{|m|}(x)^2` on `[-1, 1]` (the same for complex and real
/// harmonics, whose `theta` dependence is identical).
#[derive(Debug, Clone, PartialEq)]
pub struct PolarDistribution {
    l: u32,
    m_abs: u32,
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl PolarDistribution {
    /// For orbital angular momentum `l` and `|m| <= l`.
    pub fn new(l: u32, m: i32) -> Self {
        let (nodes, weights) = gauss_legendre(l as usize + 1);
        Self {
            l,
            m_abs: m.unsigned_abs().min(l),
            nodes,
            weights,
        }
    }

    /// The density `Pbar_l^{|m|}(x)^2`; 0 outside `[-1, 1]`.
    pub fn pdf(&self, x: f64) -> f64 {
        if !(-1.0..=1.0).contains(&x) {
            return 0.0;
        }
        let p = legendre_normalised(self.l, self.m_abs, x);
        p * p
    }

    /// `P(cos theta <= x)`. Exact up to rounding: the `(l+1)`-point rule
    /// integrates the degree-`2l` density without error. The density is even,
    /// so `x > 0` is evaluated as `1 - cdf(-x)` to keep the small side exact.
    pub fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= -1.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        if x > 0.0 {
            1.0 - self.lower(-x)
        } else {
            self.lower(x)
        }
    }

    /// `integral_{-1}^{x} pdf`, for `x` in `[-1, 0]`.
    fn lower(&self, x: f64) -> f64 {
        let half = 0.5 * (x + 1.0);
        let mut acc = 0.0;
        for (&xi, &w) in self.nodes.iter().zip(&self.weights) {
            let t = -1.0 + half * (xi + 1.0);
            let p = legendre_normalised(self.l, self.m_abs, t);
            acc += w * p * p;
        }
        acc * half
    }

    /// The `x` with `cdf(x) = u`, `u` in `[0, 1]`.
    pub fn quantile(&self, u: f64) -> f64 {
        if self.l == 0 {
            return 2.0 * u - 1.0;
        }
        if u <= 0.0 {
            return -1.0;
        }
        if u >= 1.0 {
            return 1.0;
        }
        let g = |x: f64| (self.cdf(x) - u, self.pdf(x));
        solve_bracketed(g, -1.0, 1.0, 2.0 * u - 1.0)
    }
}

/// The distribution of the azimuth `phi` in `[0, 2 pi)`: uniform for complex
/// harmonics and `m = 0`; `cos^2(m phi)/pi` for real `m > 0`;
/// `sin^2(|m| phi)/pi` for real `m < 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AzimuthalDistribution {
    /// `0` encodes the uniform case.
    m_real: i32,
}

impl AzimuthalDistribution {
    /// For magnetic index `m` in `basis`.
    pub fn new(m: i32, basis: Basis) -> Self {
        Self {
            m_real: if basis == Basis::Real { m } else { 0 },
        }
    }

    /// Whether `phi` is uniform.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.m_real == 0
    }

    /// The density at `phi` in `[0, 2 pi)`.
    pub fn pdf(&self, phi: f64) -> f64 {
        if self.is_uniform() {
            1.0 / TAU
        } else {
            real_azimuthal_cdf(self.m_real, phi).1
        }
    }

    /// `P(Phi <= phi)` for `phi` in `[0, 2 pi]`.
    pub fn cdf(&self, phi: f64) -> f64 {
        let phi = phi.clamp(0.0, TAU);
        if self.is_uniform() {
            phi / TAU
        } else {
            real_azimuthal_cdf(self.m_real, phi).0
        }
    }

    /// The `phi` with `cdf(phi) = u`, `u` in `[0, 1)`.
    pub fn quantile(&self, u: f64) -> f64 {
        if self.is_uniform() {
            return TAU * u;
        }
        let g = |phi: f64| {
            let (c, p) = real_azimuthal_cdf(self.m_real, phi);
            (c - u, p)
        };
        solve_bracketed(g, 0.0, TAU, TAU * u)
    }
}

/// Exact, separable Born-rule sampler for one hydrogen-like eigenstate.
#[derive(Debug, Clone, PartialEq)]
pub struct HydrogenicSampler {
    radial: RadialDistribution,
    polar: PolarDistribution,
    azimuth: AzimuthalDistribution,
}

impl HydrogenicSampler {
    /// Precompute the radial CDF table and the angular rules for `orbital`.
    pub fn new(orbital: &HydrogenicOrbital) -> Self {
        let o = orbital.orbital();
        Self {
            radial: orbital.radial_distribution(),
            polar: PolarDistribution::new(o.l(), o.m()),
            azimuth: AzimuthalDistribution::new(o.m(), o.basis()),
        }
    }

    /// The radial distribution.
    #[inline]
    pub fn radial(&self) -> &RadialDistribution {
        &self.radial
    }

    /// The `cos theta` distribution.
    #[inline]
    pub fn polar(&self) -> &PolarDistribution {
        &self.polar
    }

    /// The `phi` distribution.
    #[inline]
    pub fn azimuth(&self) -> &AzimuthalDistribution {
        &self.azimuth
    }

    /// One point as `(r, cos theta, phi)`; consumes exactly three uniforms
    /// (radius, polar, azimuth, in that order).
    pub fn sample_spherical(&self, rng: &mut SplitMix64) -> (f64, f64, f64) {
        let r = self.radial.inverse_survival(rng.next_f64_open_closed());
        let x = self.polar.quantile(rng.next_f64());
        let phi = self.azimuth.quantile(rng.next_f64());
        (r, x, phi)
    }

    /// One point in Cartesian coordinates (bohr, nucleus at the origin).
    pub fn sample(&self, rng: &mut SplitMix64) -> [f64; 3] {
        let (r, x, phi) = self.sample_spherical(rng);
        let s = libm::sqrt((1.0 - x) * (1.0 + x));
        let (sp, cp) = libm::sincos(phi);
        [r * s * cp, r * s * sp, r * x]
    }
}

/// Born-rule sampler for a superposition at a fixed time, by mixture
/// rejection (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct SuperpositionSampler {
    orbitals: Vec<HydrogenicOrbital>,
    samplers: Vec<HydrogenicSampler>,
    /// `c_k e^{-i E_k t}`.
    coefficients: Vec<Complex>,
    /// `|c_k|^2`.
    weights: Vec<f64>,
    /// Running sums of `weights`.
    cumulative: Vec<f64>,
}

impl SuperpositionSampler {
    /// Sample `|psi(x, t)|^2` of `state` at time `t` (atomic units). Terms
    /// with a zero coefficient are dropped.
    pub fn new(state: &State, t: f64) -> Self {
        let coeffs = state.coefficients_at(t);
        let mut orbitals = Vec::new();
        let mut samplers = Vec::new();
        let mut coefficients = Vec::new();
        let mut weights = Vec::new();
        for (orb, c) in state.orbitals().iter().zip(coeffs) {
            let w = c.norm_sqr();
            if w > 0.0 {
                orbitals.push(*orb);
                samplers.push(HydrogenicSampler::new(orb));
                coefficients.push(c);
                weights.push(w);
            }
        }
        let mut acc = 0.0;
        let cumulative = weights
            .iter()
            .map(|w| {
                acc += w;
                acc
            })
            .collect();
        Self {
            orbitals,
            samplers,
            coefficients,
            weights,
            cumulative,
        }
    }

    /// Number of mixture components `K`.
    #[inline]
    pub fn components(&self) -> usize {
        self.samplers.len()
    }

    /// One independent draw from `|psi|^2`.
    ///
    /// # Errors
    ///
    /// [`QmError::RejectionLimit`] after [`MAX_REJECTION_ATTEMPTS`] rejected
    /// proposals (only possible when the terms nearly cancel), and
    /// [`QmError::EmptyState`] when every coefficient is zero.
    pub fn sample(&self, rng: &mut SplitMix64) -> Result<[f64; 3], QmError> {
        let k_terms = self.samplers.len();
        let Some(&total) = self.cumulative.last() else {
            return Err(QmError::EmptyState);
        };
        if k_terms == 1 {
            return Ok(self.samplers[0].sample(rng));
        }
        let k_f = k_terms as f64;
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let pick = rng.next_f64() * total;
            let k = self
                .cumulative
                .iter()
                .position(|&c| pick < c)
                .unwrap_or(k_terms - 1);
            let p = self.samplers[k].sample(rng);
            let accept = rng.next_f64();

            let r = libm::hypot(libm::hypot(p[0], p[1]), p[2]);
            let dir = Direction::from_cartesian(p);
            let mut psi = Complex::ZERO;
            let mut mixture = 0.0;
            for ((orb, &c), &w) in self.orbitals.iter().zip(&self.coefficients).zip(&self.weights)
            {
                let v = orb.psi_polar(r, &dir);
                psi += c * v;
                mixture += w * v.norm_sqr();
            }
            // mixture > 0 wherever the proposal can land.
            if accept * k_f * mixture < psi.norm_sqr() {
                return Ok(p);
            }
        }
        Err(QmError::RejectionLimit(MAX_REJECTION_ATTEMPTS))
    }

    /// `count` independent draws.
    ///
    /// # Errors
    ///
    /// As [`SuperpositionSampler::sample`].
    pub fn sample_many(
        &self,
        count: usize,
        rng: &mut SplitMix64,
    ) -> Result<Vec<[f64; 3]>, QmError> {
        (0..count).map(|_| self.sample(rng)).collect()
    }
}

/// Gamma(`shape`, `scale`) variates by Marsaglia and Tsang, "A simple method
/// for generating gamma variables", ACM TOMS 26, 363 (2000).
///
/// For `shape >= 1`: with `d = shape - 1/3`, `c = 1/sqrt(9d)`, draw a normal
/// `x`, set `v = (1 + c x)^3`, and accept `d v` if `u < 1 - 0.0331 x^4`
/// (squeeze) or `ln u < x^2/2 + d (1 - v + ln v)`. For `shape < 1` the
/// standard boost `Gamma(a) = Gamma(a + 1) U^{1/a}` is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaSampler {
    shape: f64,
    scale: f64,
    d: f64,
    c: f64,
}

impl GammaSampler {
    /// # Errors
    ///
    /// [`QmError::InvalidGammaParameters`] unless both are finite and `> 0`.
    pub fn new(shape: f64, scale: f64) -> Result<Self, QmError> {
        if !(shape.is_finite() && scale.is_finite() && shape > 0.0 && scale > 0.0) {
            return Err(QmError::InvalidGammaParameters { shape, scale });
        }
        let d = if shape < 1.0 { shape + 1.0 } else { shape } - 1.0 / 3.0;
        Ok(Self {
            shape,
            scale,
            d,
            c: 1.0 / libm::sqrt(9.0 * d),
        })
    }

    /// The shape `k`.
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// The scale `theta` (mean = `k theta`).
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// One variate.
    pub fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let g = loop {
            let x = rng.next_standard_normal();
            let t = 1.0 + self.c * x;
            if t <= 0.0 {
                continue;
            }
            let v = t * t * t;
            let u = rng.next_f64();
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                break self.d * v;
            }
            if libm::log(u) < 0.5 * x2 + self.d * (1.0 - v + libm::log(v)) {
                break self.d * v;
            }
        };
        let g = if self.shape < 1.0 {
            g * libm::pow(rng.next_f64_open_closed(), 1.0 / self.shape)
        } else {
            g
        };
        g * self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrogenic::{HydrogenLike, Orbital};
    use crate::state::Term;
    use core::f64::consts::PI;

    #[test]
    fn stream_is_reproducible_and_in_range() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut s = SplitMix64::new(1);
        for _ in 0..10_000 {
            let u = s.next_f64();
            let v = s.next_f64_open_closed();
            assert!((0.0..1.0).contains(&u) && v > 0.0 && v <= 1.0);
        }
        // Normal moments.
        let mut s = SplitMix64::new(3);
        let n = 200_000;
        let (mut m1, mut m2) = (0.0, 0.0);
        for _ in 0..n {
            let x = s.next_standard_normal();
            m1 += x;
            m2 += x * x;
        }
        assert!((m1 / n as f64).abs() < 0.01 && (m2 / n as f64 - 1.0).abs() < 0.01);
    }

    #[test]
    fn polar_cdf_endpoints_and_inverse() {
        for (l, m) in [(0, 0), (1, 0), (1, 1), (2, 1), (5, 3), (8, 0)] {
            let d = PolarDistribution::new(l, m);
            assert_eq!(d.cdf(-1.0), 0.0);
            assert_eq!(d.cdf(1.0), 1.0);
            assert!((d.cdf(0.0) - 0.5).abs() < 1e-15, "symmetric l={l} m={m}");
            for &u in &[1e-9, 0.1, 0.5, 0.93] {
                let x = d.quantile(u);
                assert!((d.cdf(x) - u).abs() < 1e-14, "l={l} m={m} u={u}");
            }
        }
        // p_z: density (3/2) x^2, CDF (x^3 + 1)/2.
        let d = PolarDistribution::new(1, 0);
        assert!((d.cdf(0.4) - 0.5 * (0.064 + 1.0)).abs() < 1e-15);
        assert_eq!(d.pdf(2.0), 0.0);
        assert!(d.cdf(f64::NAN).is_nan());
        assert_eq!(d.quantile(0.0), -1.0);
        assert_eq!(d.quantile(1.0), 1.0);
    }

    #[test]
    fn azimuth_quantiles() {
        let u = AzimuthalDistribution::new(2, Basis::Complex);
        assert!(u.is_uniform());
        assert_eq!(u.quantile(0.25), TAU * 0.25);
        assert_eq!(u.cdf(PI), 0.5);
        assert!((u.pdf(1.0) - 1.0 / TAU).abs() < 1e-16);
        for m in [-2, -1, 1, 3] {
            let a = AzimuthalDistribution::new(m, Basis::Real);
            assert!(!a.is_uniform());
            for &q in &[0.0, 0.1, 0.37, 0.5, 0.99] {
                let phi = a.quantile(q);
                assert!((a.cdf(phi) - q).abs() < 1e-14, "m={m} q={q}");
            }
        }
    }

    #[test]
    fn samplers_are_deterministic() {
        let h = HydrogenLike::hydrogen();
        let o = h.orbital(Orbital::real(3, 2, -1).unwrap());
        let s = HydrogenicSampler::new(&o);
        let a: Vec<[f64; 3]> = {
            let mut rng = SplitMix64::new(11);
            (0..100).map(|_| s.sample(&mut rng)).collect()
        };
        let b: Vec<[f64; 3]> = {
            let mut rng = SplitMix64::new(11);
            (0..100).map(|_| s.sample(&mut rng)).collect()
        };
        assert_eq!(a, b);
        assert!(a.iter().all(|p| p.iter().all(|c| c.is_finite())));
        assert!(s.radial().cdf(1.0) > 0.0 && s.polar().cdf(0.0) > 0.0);
        assert!(!s.azimuth().is_uniform());
    }

    #[test]
    fn superposition_sampler_edge_cases() {
        let h = HydrogenLike::hydrogen();
        let single = State::eigenstate(h, Orbital::complex(2, 1, 1).unwrap());
        let s = SuperpositionSampler::new(&single, 3.0);
        assert_eq!(s.components(), 1);
        let mut rng = SplitMix64::new(5);
        assert!(s.sample(&mut rng).is_ok());
        let mixed = State::new(
            h,
            vec![
                Term::new(Complex::ONE, Orbital::real(1, 0, 0).unwrap()),
                Term::new(Complex::ZERO, Orbital::real(2, 1, 0).unwrap()),
                Term::new(Complex::I, Orbital::real(2, 1, 1).unwrap()),
            ],
        )
        .unwrap();
        let s = SuperpositionSampler::new(&mixed, 0.0);
        assert_eq!(s.components(), 2);
        let pts = s.sample_many(50, &mut rng).unwrap();
        assert_eq!(pts.len(), 50);
    }

    #[test]
    fn gamma_parameters_and_moments() {
        assert!(GammaSampler::new(0.0, 1.0).is_err());
        assert!(GammaSampler::new(1.0, f64::NAN).is_err());
        for (shape, scale) in [(0.4, 2.0), (1.0, 1.0), (3.0, 0.5), (11.5, 0.1)] {
            let g = GammaSampler::new(shape, scale).unwrap();
            assert_eq!((g.shape(), g.scale()), (shape, scale));
            let mut rng = SplitMix64::new(99);
            let n = 200_000;
            let (mut m1, mut m2) = (0.0, 0.0);
            for _ in 0..n {
                let x = g.sample(&mut rng);
                assert!(x > 0.0);
                m1 += x;
                m2 += x * x;
            }
            let mean = m1 / n as f64;
            let var = m2 / n as f64 - mean * mean;
            let (want_mean, want_var) = (shape * scale, shape * scale * scale);
            assert!((mean / want_mean - 1.0).abs() < 0.01, "mean {mean}");
            assert!((var / want_var - 1.0).abs() < 0.03, "var {var}");
        }
    }
}
