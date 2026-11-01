//====== periodica/rust/periodica-qm/src/special/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # special
//!
//! Special functions for atomic and molecular quantum mechanics, in `f64`,
//! with every transcendental from `libm` so results are bit-identical across
//! operating systems.
//!
//! | Function | Method | Gate (tests) |
//! |---|---|---|
//! | [`laguerre`] `L_k^alpha(x)` | three-term recurrence | vs explicit sum `1e-12` rel |
//! | [`laguerre_roots`] | interlacing brackets + safeguarded Newton | sign change within `1e-13` rel |
//! | [`legendre_normalised`] `Pbar_l^m(x)` | normalised recurrences, CS phase | closed forms `1e-14` |
//! | [`spherical_harmonic`] `Y_lm` | `Pbar e^{i m phi}/sqrt(2 pi)` | orthonormality `1e-13`, scipy |
//! | [`real_spherical_harmonic`] `S_lm` | chemistry convention | p_x > 0 on +x |
//! | [`ln_gamma`] | `libm::lgamma_r` | `ln n!` to 4 ulp |
//! | [`gamma_p_int`], [`gamma_q_int`] | closed form + lower-tail series | `P + Q = 1` |
//! | [`boys`] `F_m(T)` | series + downward / `erf` + upward | GL(128) `1e-14` abs |

pub mod boys;
pub mod gamma;
pub mod harmonics;
pub mod laguerre;
pub mod legendre;

pub use boys::{boys, boys_array};
pub use gamma::{gamma_p_int, gamma_q_int, ln_factorial, ln_gamma};
pub use harmonics::{
    angular_overlap, complex_expansion, real_spherical_harmonic, spherical_harmonic, Basis,
    Direction,
};
pub use laguerre::{laguerre, laguerre_derivative, laguerre_roots};
pub use legendre::legendre_normalised;
