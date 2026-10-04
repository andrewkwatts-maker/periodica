//====== periodica/rust/periodica-render/src/glsl.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! GLSL ES 1.00 lowering.
//!
//! **Skeleton** -- plan items C2 (lowering) and C3 (templates, glslang CI).
//! The intended contract: lower a `FieldProgram` and a scene to a specialised
//! `#version 100` fragment shader with basis centres, exponents and
//! coefficients baked as literals in unrolled code (ES 1.00 has no const
//! arrays, and literals do not consume the uniform budget), using at most nine
//! `vec4` per-frame uniforms. Every generated variant must pass
//! `glslangValidator` for ES 1.00.
