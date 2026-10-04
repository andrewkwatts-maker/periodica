//====== periodica/rust/periodica-render/src/field_program.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! `FieldProgram`: the scalar-field IR shared by CPU and GPU.
//!
//! **Skeleton** -- plan item C2. The intended contract:
//!
//! - A straight-line program over f32 registers -- no loops, no branches --
//!   with the opcodes `add`, `sub`, `mul`, `div`, `exp`, `sqrt`, `select`,
//!   `const`, `px`, `py`, `pz`.
//! - A CPU interpreter whose transcendentals go through `libm`, so results are
//!   bit-identical across platforms.
//! - A subset of Stereoma's opcodes by design (registered temporary exception
//!   in `modules.yaml`), so it can later be swapped for Stereoma + Phantasia
//!   mechanically.
