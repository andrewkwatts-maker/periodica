//====== periodica/rust/periodica-chem/src/geometry.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! `Geometry3D`: curated molecular geometry.
//!
//! **Skeleton** -- the model lands with plan item B2, fed by the Q3 curation
//! (`data/active/geometry/<name>.json`, deliberately not a registry tier).
//! The intended contract:
//!
//! - Atoms as element symbol + Cartesian position in ångström, with the
//!   structure type (r_e, r_0 or r_s), the primary-literature source and its
//!   stated uncertainty carried alongside -- never rounded on ingest.
//! - Validation: bond lengths consistent with the stated source to 1e-4 Å and
//!   angles to 0.05°.
//! - Bond perception from Cordero 2008 covalent radii.
