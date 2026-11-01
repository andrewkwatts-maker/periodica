//====== periodica/rust/periodica-render/src/camera.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Camera model and primary rays.
//!
//! **Skeleton** -- plan item C1. The intended contract: an orthonormal camera
//! basis, vertical field of view and aspect ratio, perspective and
//! orthographic ("quantitative", column-density) projections, and the
//! per-pixel ray that the CPU reference and the generated shaders both use.
