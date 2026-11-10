//! Drude (free electron) and Lorentz (bound electron) dielectric response.
//!
//! Photon energy `e` is in eV and `epsilon = e1 + i e2`. The complex index
//! follows from `n + ik = sqrt(epsilon)` through the real and imaginary parts.

use arithma_core::ArithmaExpression as E;

use super::{check_range, eval, finite, num, square, sum, var, OpticsError};

/// eV nm: hc/lambda in eV for lambda in nm (the constant the Python prototype uses).
pub const HC_EV_NM: f64 = 1239.84;

/// One Lorentz oscillator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Oscillator {
    pub strength: f64,
    pub resonance_ev: f64,
    pub damping_ev: f64,
}

/// One interband critical-point term (Etchegoin et al. 2006):
/// `A Ec [e^{i phi} / (Ec - E - i g) + e^{-i phi} / (Ec + E + i g)]`, energies in eV.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriticalPoint {
    pub amplitude: f64,
    pub phase_rad: f64,
    pub energy_ev: f64,
    pub damping_ev: f64,
}

/// Drude plus Lorentz oscillators and critical-point terms.
/// `range_m` is the wavelength range (metres) the parameters were fitted over.
#[derive(Debug, Clone, PartialEq)]
pub struct DrudeLorentz {
    pub epsilon_inf: f64,
    pub plasma_ev: f64,
    pub damping_ev: f64,
    pub oscillators: Vec<Oscillator>,
    pub critical_points: Vec<CriticalPoint>,
    pub range_m: (f64, f64),
}

impl DrudeLorentz {
    /// Gold: Drude plus two interband critical points, the three-term fit of
    /// P. G. Etchegoin, E. C. Le Ru and M. Meyer, J. Chem. Phys. 125, 164705 (2006),
    /// to the Johnson and Christy (1972) data: eps_inf = 1.54, lambda_p = 143 nm,
    /// gamma_p = 14500 nm; A1 = 1.27, phi1 = -pi/4, lambda1 = 470 nm, gamma1 = 1900 nm;
    /// A2 = 1.1, phi2 = -pi/4, lambda2 = 325 nm, gamma2 = 1060 nm (wavelengths converted
    /// to eV with `HC_EV_NM`). The fit holds over the visible and near infrared; the
    /// range 350 to 1000 nm is this crate's choice, inside the fitted data.
    pub fn gold() -> Self {
        let cp = |amplitude, nm, gamma_nm| CriticalPoint {
            amplitude,
            phase_rad: -std::f64::consts::FRAC_PI_4,
            energy_ev: HC_EV_NM / nm,
            damping_ev: HC_EV_NM / gamma_nm,
        };
        Self {
            epsilon_inf: 1.54,
            plasma_ev: HC_EV_NM / 143.0,
            damping_ev: HC_EV_NM / 14500.0,
            oscillators: Vec::new(),
            critical_points: vec![cp(1.27, 470.0, 1900.0), cp(1.1, 325.0, 1060.0)],
            range_m: (350e-9, 1000e-9),
        }
    }

    fn critical_parts(c: &CriticalPoint) -> (E, E) {
        let (cos, sin) = (c.phase_rad.cos(), c.phase_rad.sin());
        let g = c.damping_ev;
        let a = E::sub(num(c.energy_ev), var("e"));
        let d = E::add(num(c.energy_ev), var("e"));
        let d1 = E::add(square(a.clone()), num(g * g));
        let d2 = E::add(square(d.clone()), num(g * g));
        let amp = num(c.amplitude * c.energy_ev);
        let re1 = E::div(
            E::sub(E::mul(num(cos), a.clone()), num(sin * g)),
            d1.clone(),
        );
        let im1 = E::div(E::add(E::mul(num(sin), a), num(cos * g)), d1);
        let re2 = E::div(
            E::sub(E::mul(num(cos), d.clone()), num(sin * g)),
            d2.clone(),
        );
        let im2 = E::div(E::sub(E::mul(num(-sin), d), num(cos * g)), d2);
        (
            E::mul(amp.clone(), E::add(re1, re2)),
            E::mul(amp, E::add(im1, im2)),
        )
    }

    fn drude_parts(&self) -> (E, E) {
        let e2 = square(var("e"));
        let g = num(self.damping_ev);
        let wp2 = num(self.plasma_ev * self.plasma_ev);
        let den = E::add(e2, square(g.clone()));
        let re = E::sub(num(self.epsilon_inf), E::div(wp2.clone(), den.clone()));
        let im = E::div(E::mul(wp2, g), E::mul(var("e"), den));
        (re, im)
    }

    fn lorentz_parts(o: &Oscillator) -> (E, E) {
        let w0sq = o.resonance_ev * o.resonance_ev;
        let diff = E::sub(num(w0sq), square(var("e")));
        let g = num(o.damping_ev);
        let den = E::add(square(diff.clone()), square(E::mul(g.clone(), var("e"))));
        let amp = num(o.strength * w0sq);
        let re = E::div(E::mul(amp.clone(), diff), den.clone());
        let im = E::div(E::mul(E::mul(amp, g), var("e")), den);
        (re, im)
    }

    /// Real and imaginary parts of epsilon as expressions in `e` (eV).
    pub fn epsilon_exprs(&self) -> (E, E) {
        let (mut res, mut ims) = (Vec::new(), Vec::new());
        let (dr, di) = self.drude_parts();
        res.push(dr);
        ims.push(di);
        let lorentz = self.oscillators.iter().map(Self::lorentz_parts);
        let critical = self.critical_points.iter().map(Self::critical_parts);
        for (r, i) in lorentz.chain(critical) {
            res.push(r);
            ims.push(i);
        }
        (sum(res), sum(ims))
    }

    /// `(n, k)` as expressions in `e`.
    pub fn nk_exprs(&self) -> (E, E) {
        let (re, im) = self.epsilon_exprs();
        let modulus = E::sqrt(E::add(square(re.clone()), square(im)));
        let half = num(0.5);
        let n = E::sqrt(E::mul(half.clone(), E::add(modulus.clone(), re.clone())));
        let k = E::sqrt(E::mul(half, E::sub(modulus, re)));
        (n, k)
    }

    /// `(n, k)` at `wavelength_m` metres; an error outside the fitted range.
    pub fn nk(&self, wavelength_m: f64) -> Result<(f64, f64), OpticsError> {
        check_range(wavelength_m, self.range_m)?;
        let e = HC_EV_NM / (wavelength_m * 1e9);
        let (n, k) = self.nk_exprs();
        let n = finite(eval(&n, &[("e", e)]), wavelength_m)?;
        let k = finite(eval(&k, &[("e", e)]), wavelength_m)?;
        Ok((n, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gold_is_a_strong_reflector_in_the_red() {
        let (n, k) = DrudeLorentz::gold().nk(600e-9).unwrap();
        let r = ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k);
        assert!(r > 0.9, "{r}");
    }
}
