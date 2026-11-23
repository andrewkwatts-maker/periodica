//! # temperature
//!
//! Temperature dependence of material properties as Arithma expressions in `T` (kelvin),
//! each with a validity range and a provenance. Serves the gallery thermal scenes
//! S14 (free and clamped bar: alpha(T), E(T)), S15 (bimetal strip) and S16 (Norton creep).
//!
//! No value is invented here: every law is built from caller-supplied coefficients, and a
//! law without a cited source carries [`Provenance::NotTestable`]. Outside its range a law
//! returns an error rather than extrapolating.

use arithma_core::expression::{ArithmaBindings, Evaluable};
use arithma_core::{ArithmaExpression as E, ArithmaFunction};

use crate::constants::{AVOGADRO, BOLTZMANN};

/// Where the coefficients of a law come from.
#[derive(Debug, Clone, PartialEq)]
pub enum Provenance {
    /// Fitted to or tabulated from the named published source.
    Cited(String),
    /// No source exists, so no test may score this law against experiment.
    NotTestable,
}

/// Why a law could not give a value.
#[derive(Debug, Clone, PartialEq)]
pub enum TemperatureError {
    /// `T` lies outside the validity range.
    OutOfRange { t_k: f64, lo_k: f64, hi_k: f64 },
    /// The expression was not finite at this input.
    NotFinite { t_k: f64 },
    /// A table needs at least two points with strictly increasing temperatures.
    BadTable,
}

impl std::fmt::Display for TemperatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange { t_k, lo_k, hi_k } => {
                write!(f, "T = {t_k} K is outside the valid range [{lo_k}, {hi_k}] K")
            }
            Self::NotFinite { t_k } => write!(f, "law is not finite at T = {t_k} K"),
            Self::BadTable => write!(f, "table needs two or more strictly increasing T points"),
        }
    }
}

impl std::error::Error for TemperatureError {}

fn num(v: f64) -> E {
    E::constant("", None, Some(v), false)
}

fn t_minus(t_ref: f64) -> E {
    E::sub(E::var("T"), num(t_ref))
}

fn run(expr: &E, vars: &[(&str, f64)]) -> f64 {
    let b: ArithmaBindings = vars.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    expr.evaluate(&b).unwrap_or(f64::NAN)
}

/// `max(0, x)` as an expression: `x (1 + sign x) / 2`.
fn relu(x: E) -> E {
    let s = E::func(ArithmaFunction::Sign, vec![x.clone()]);
    E::mul(x, E::mul(num(0.5), E::add(num(1.0), s)))
}

/// A property as an expression in `T` (and optional extra variables) with its validity range.
#[derive(Debug, Clone)]
pub struct ThermalLaw {
    pub expr: E,
    /// Inclusive validity range in kelvin.
    pub range_k: (f64, f64),
    pub provenance: Provenance,
}

impl ThermalLaw {
    /// `y = y_ref + slope (T - t_ref)`, the form used for E(T) over a modest range.
    pub fn linear(y_ref: f64, t_ref: f64, slope: f64, range_k: (f64, f64), p: Provenance) -> Self {
        let expr = E::add(num(y_ref), E::mul(num(slope), t_minus(t_ref)));
        Self { expr, range_k, provenance: p }
    }

    /// `y = sum c_i (T - t_ref)^i`, coefficients lowest order first (alpha(T) fits).
    pub fn polynomial(coeffs: &[f64], t_ref: f64, range_k: (f64, f64), p: Provenance) -> Self {
        let terms = coeffs
            .iter()
            .enumerate()
            .map(|(i, c)| E::mul(num(*c), E::pow(t_minus(t_ref), num(i as f64))));
        let expr = terms.reduce(E::add).unwrap_or_else(|| num(0.0));
        Self { expr, range_k, provenance: p }
    }

    /// Piecewise-linear interpolation of tabulated `(T, y)` points; the range is the table span.
    pub fn table(points: &[(f64, f64)], p: Provenance) -> Result<Self, TemperatureError> {
        if points.len() < 2 || points.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(TemperatureError::BadTable);
        }
        let mut expr = num(points[0].1);
        let mut prev_slope = 0.0;
        for w in points.windows(2) {
            // direct-float: slope of a tabulated segment, the table's own data reduction
            let slope = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
            let kink = E::mul(num(slope - prev_slope), relu(t_minus(w[0].0)));
            expr = E::add(expr, kink);
            prev_slope = slope;
        }
        let range_k = (points[0].0, points[points.len() - 1].0);
        Ok(Self { expr, range_k, provenance: p })
    }

    /// Norton creep rate `A sigma^n exp(-Q / (R T))`, in variables `T` (K) and `sigma` (Pa).
    /// `a` in Pa^-n s^-1, `q` in J/mol; evaluate with [`ThermalLaw::at_stress`].
    pub fn norton(a: f64, n: f64, q: f64, range_k: (f64, f64), p: Provenance) -> Self {
        // direct-float: molar gas constant, the exact product of two SI-defined constants
        let r = AVOGADRO * BOLTZMANN;
        let arrhenius = E::exp(E::neg(E::div(num(q), E::mul(num(r), E::var("T")))));
        let power = E::mul(num(a), E::pow(E::var("sigma"), num(n)));
        Self { expr: E::mul(power, arrhenius), range_k, provenance: p }
    }

    /// Value at `t_k`, or an error outside the range or when not finite.
    pub fn at(&self, t_k: f64) -> Result<f64, TemperatureError> {
        self.at_stress(t_k, 0.0)
    }

    /// Value at `t_k` with a stress `sigma_pa` bound (used by [`ThermalLaw::norton`]).
    pub fn at_stress(&self, t_k: f64, sigma_pa: f64) -> Result<f64, TemperatureError> {
        let (lo_k, hi_k) = self.range_k;
        if !t_k.is_finite() || t_k < lo_k || t_k > hi_k {
            return Err(TemperatureError::OutOfRange { t_k, lo_k, hi_k });
        }
        let v = run(&self.expr, &[("T", t_k), ("sigma", sigma_pa)]);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(TemperatureError::NotFinite { t_k })
        }
    }

    /// True when a test may score this law against a published experiment.
    pub fn testable(&self) -> bool {
        matches!(self.provenance, Provenance::Cited(_))
    }
}
