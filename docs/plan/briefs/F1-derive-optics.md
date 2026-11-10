# F1: derived optical values (lane `derive-optics`)

Read `_common.md` first. Source: `docs/plan/sheets-formula-review.md`, BUGs 1 and 2. Deliverable:
two fixes in `rust/periodica-mat/src/derive.rs` (about lines 228-243), each with a test in a new
file `rust/periodica-mat/tests/f1_derive_optics.rs`, on `claude/derive-optics`.

1. Reflectance from n (derive.rs:235-243) ignores k. New rule:
   - `ExtinctionK` present: `R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)`, source `Derived`. Reuse
     `optics/dispersion.rs` `fresnel_expr` if it fits.
   - `ExtinctionK` absent and `Transmittance` present: the k = 0 form, source `Estimated`.
   - Otherwise derive nothing. A sheet with n alone doesn't say whether the material absorbs.

   Tests: n = 1.2, k = 7.26 gives R of about 0.917; n alone gives no Reflectance.
2. Emissivity (derive.rs:228-233) is mirrored from Absorptance, which is a visible-band value.
   Remove the mirror in both directions. Thermal emissivity then comes from the sheet only, until a
   band-integrated formula exists. Test: R = 0.95, T = 0 produces an Absorptance but no Emissivity.

Run `cargo test -p periodica-mat`. Count how many corpus sheets lose a derived Reflectance or
Emissivity because of this, and list the count in your report. Update oracles that change, with old
and new numbers.
