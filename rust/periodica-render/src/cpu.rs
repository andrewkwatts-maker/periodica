//====== periodica/rust/periodica-render/src/cpu.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! CPU reference renderer.
//!
//! **Skeleton** -- plan item C4. The intended contract: the same ray setup,
//! `FieldProgram`, step count, jitter, front-to-back emission-absorption
//! compositing and sRGB encode as the generated shaders, so its images are the
//! goldens the GPU path is tested against (CPU vs golden <= 1e-5; GPU vs CPU
//! 99.9 % of pixels within 2/255).
