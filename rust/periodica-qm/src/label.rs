//====== periodica/rust/periodica-qm/src/label.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # label
//!
//! [`ModelLabel`]: what a computed quantity *is*. Every view of an orbital
//! carries one, so a picture never claims more than its physics: which model
//! produced it, how faithfully the code evaluates that model, and how far the
//! model itself is from the real atom.

use core::fmt;

use periodica_mat::constants::{FINE_STRUCTURE, PROTON_ELECTRON_MASS_RATIO};

/// Relative accuracy to which this crate evaluates the hydrogen-like model:
/// the `1e-12` gate on normalisation, orthogonality and `<r^k>` in
/// `tests/hydrogenic_accuracy.rs`.
pub const HYDROGENIC_NUMERICAL_ERROR: f64 = 1e-12;

/// Model provenance and error budget for a computed quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelLabel {
    /// The physical model, in words.
    pub model: String,
    /// Relative error of the computed numbers against the exact model (what
    /// the tests gate).
    pub numerical_error_vs_model: f64,
    /// Estimated relative error of the model against the real system (for an
    /// energy-like quantity), from the leading neglected physics.
    pub physical_error_vs_reality: f64,
    /// Where the model and its error estimate come from.
    pub reference: String,
}

impl ModelLabel {
    /// The label of a hydrogen-like eigenstate `(n, l)` of nuclear charge `z`.
    ///
    /// `physical_error_vs_reality` is the leading relativistic correction,
    /// the Dirac fine-structure shift relative to the Schrodinger energy,
    /// maximised over the two `j = l +- 1/2` levels
    /// (the order-`(Z alpha)^2` term of the Dirac energy; Bethe and Salpeter 1957):
    ///
    /// ```text
    /// |dE / E_n| = (Z alpha)^2 / n^2 * (n / (j + 1/2) - 3/4),   j + 1/2 = max(l, 1)
    /// ```
    ///
    /// i.e. `alpha^2 / 4 = 1.3e-5` for hydrogen 1s. With an infinitely heavy
    /// nucleus (`reduced_mass = false`) it adds the neglected reduced-mass
    /// correction, bounded above by `m_e / (Z m_p)` because a nucleus of
    /// charge `Z` holds at least `Z` nucleons. Lamb shift and hyperfine
    /// structure (order `alpha (Z alpha)^2` and smaller) are not included.
    pub fn hydrogenic(z: f64, n: u32, l: u32, reduced_mass: bool) -> ModelLabel {
        let mut physical = hydrogenic_relativistic_error(z, n, l);
        if !reduced_mass {
            physical += 1.0 / (z.max(1.0) * PROTON_ELECTRON_MASS_RATIO);
        }
        let mass = if reduced_mass {
            "reduced mass"
        } else {
            "infinite nuclear mass"
        };
        ModelLabel {
            model: format!(
                "hydrogen-like Z={z} n={n} l={l}: non-relativistic Schrodinger, point nucleus, {mass}"
            ),
            numerical_error_vs_model: HYDROGENIC_NUMERICAL_ERROR,
            physical_error_vs_reality: physical,
            reference: "Bethe & Salpeter, Quantum Mechanics of One- and Two-Electron Atoms (1957); \
                        error = leading Dirac fine-structure term; constants CODATA 2022"
                .to_string(),
        }
    }

    /// The least accurate of several labels (for a superposition): the model
    /// text of the first, and the largest of each error.
    ///
    /// Returns `None` for an empty iterator.
    pub fn worst<'a, I>(labels: I) -> Option<ModelLabel>
    where
        I: IntoIterator<Item = &'a ModelLabel>,
    {
        let mut it = labels.into_iter();
        let mut out = it.next()?.clone();
        for l in it {
            out.numerical_error_vs_model = out
                .numerical_error_vs_model
                .max(l.numerical_error_vs_model);
            out.physical_error_vs_reality = out
                .physical_error_vs_reality
                .max(l.physical_error_vs_reality);
        }
        Some(out)
    }
}

impl fmt::Display for ModelLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} | numerical <= {:.0e} | vs reality ~{:.1e} | {}",
            self.model, self.numerical_error_vs_model, self.physical_error_vs_reality, self.reference
        )
    }
}

/// The leading relativistic error of the Schrodinger hydrogen-like energy
/// `E_n`, relative: `(Z alpha)^2 / n^2 * (n / max(l, 1) - 3/4)`.
///
/// Returns `NaN` for `n = 0`.
pub fn hydrogenic_relativistic_error(z: f64, n: u32, l: u32) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    let za = z * FINE_STRUCTURE;
    let nf = f64::from(n);
    let j_half = f64::from(l.max(1));
    za * za / (nf * nf) * (nf / j_half - 0.75)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrogen_1s_is_alpha_squared_over_four() {
        let e = hydrogenic_relativistic_error(1.0, 1, 0);
        let want = FINE_STRUCTURE * FINE_STRUCTURE / 4.0;
        assert!((e / want - 1.0).abs() < 1e-15);
        assert!((e - 1.33e-5).abs() < 1e-7);
        // Scales as Z^2.
        let u = hydrogenic_relativistic_error(92.0, 1, 0);
        assert!((u / e / (92.0 * 92.0) - 1.0).abs() < 1e-14);
        assert!(hydrogenic_relativistic_error(1.0, 0, 0).is_nan());
        // 2p: j = 1/2 level, n/(j+1/2) = 2.
        let p = hydrogenic_relativistic_error(1.0, 2, 1);
        assert!((p / (FINE_STRUCTURE.powi(2) / 4.0 * 1.25) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn label_mentions_the_model_and_the_worst_case_wins() {
        let a = ModelLabel::hydrogenic(1.0, 1, 0, true);
        let b = ModelLabel::hydrogenic(1.0, 2, 1, false);
        assert!(a.model.contains("reduced mass"));
        assert!(b.model.contains("infinite nuclear mass"));
        // Neglecting the reduced mass dominates for hydrogen (5.4e-4).
        assert!(b.physical_error_vs_reality > 5e-4);
        let w = ModelLabel::worst([&a, &b]).unwrap();
        assert_eq!(w.model, a.model);
        assert_eq!(w.physical_error_vs_reality, b.physical_error_vs_reality);
        assert!(ModelLabel::worst(core::iter::empty()).is_none());
        let text = a.to_string();
        assert!(text.contains("CODATA 2022") && text.contains("1e-12"), "{text}");
    }
}
