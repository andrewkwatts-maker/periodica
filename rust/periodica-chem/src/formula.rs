//====== periodica/rust/periodica-chem/src/formula.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Chemical formula grammar and parser.
//!
//! **Skeleton** -- implemented by plan item M5. The intended contract:
//!
//! - Grammar: element symbols with counts, repeated elements, `()` / `[]`
//!   nesting, hydrates joined by `·`, `*` or `.` with leading coefficients,
//!   charges written `^2-`, `2-`, `+3`, `Fe+3`, `OH-`, and isotopes `D`, `T`,
//!   `13C`, `[13C]`, `C-13`.
//! - Element symbols are case-sensitive; a charge is carried as a signed count
//!   of elementary charges.
//! - Output is an exact composition plus a canonical Hill-order string, and
//!   `parse(hill(parse(s)))` round-trips (checked by `proptest`).
//! - One golden corpus, `tests/data/formula_cases.json`, is run by both
//!   `cargo test` (`include_str!`) and pytest.
