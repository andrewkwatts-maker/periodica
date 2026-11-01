//====== periodica/rust/periodica-render/src/scene.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Scene description and visual-mode state.
//!
//! **Skeleton** -- plan items C1 (scene, transfer functions) and C6 (mode
//! state machines). The intended contract:
//!
//! - A scene is field sources (analytic `FieldProgram`s or baked atlases),
//!   atom spheres and bond capsules, plus a bounding sphere chosen so the
//!   probability outside it is below 1e-6.
//! - Physical transfer: optical depth tau = kappa * integral of rho ds, with
//!   kappa set so the averaged central ray has tau_ref = 3; phase mapped to an
//!   OKLab cyclic hue at constant lightness.
//! - Visual modes: real-time psi(t), time-averaged |psi|^2, step, and collapse
//!   (Born-rule measurement samples converging to the rendered density).
