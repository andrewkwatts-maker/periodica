//! Drude (free electron) and Lorentz (bound electron) dielectric response.
//!
//! Photon energy `e` is in eV and `epsilon = e1 + i e2`. The complex index
//! follows from `n + ik = sqrt(epsilon)` through the real and imaginary parts.

use arithma_core::ArithmaExpression as E;

use super::{eval, num, square, sum, var};

/// eV nm: hc/lambda in eV for lambda in nm (the constant the Python prototype uses).
pub const HC_EV_NM: f64 = 1239.84;

/// One Lorentz oscillator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Oscillator {
    pub strength: f64,
    pub resonance_ev: f64,
    pub damping_ev: f64,
}

/// Drude plus any number of Lorentz oscillators.
#[derive(Debug, Clone, PartialEq)]
pub struct DrudeLorentz {
    pub epsilon_inf: f64,
    pub plasma_ev: f64,
    pub damping_ev: f64,
    pub oscillators: Vec<Oscillator>,
}

impl DrudeLorentz {
    /// Drude gold.
    pub fn gold() -> Self {
        Self { epsilon_inf: 1.0, plasma_ev: 9.03, damping_ev: 0.07, oscillators: Vec::new() }
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
        for o in &self.oscillators {
            let (r, i) = Self::lorentz_parts(o);
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

    /// `(n, k)` at `wavelength_m` metres.
    pub fn nk(&self, wavelength_m: f64) -> (f64, f64) {
        let e = HC_EV_NM / (wavelength_m * 1e9);
        let (n, k) = self.nk_exprs();
        (eval(&n, &[("e", e)]), eval(&k, &[("e", e)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gold_is_a_strong_reflector_in_the_red() {
        let (n, k) = DrudeLorentz::gold().nk(600e-9);
        let r = ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k);
        assert!(r > 0.9, "{r}");
    }
}
