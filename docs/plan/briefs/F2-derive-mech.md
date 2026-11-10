# F2: derived mechanical values (lane `derive-mech`)

Read `_common.md` first. F1 is merged (it also edits `derive.rs`). Source:
`docs/plan/sheets-formula-review.md`, BUG 4, U3 (first half) and U13, plus items 4 and 5 below
(found 2026-11-11). Deliverable: five changes in `rust/periodica-mat/src/derive.rs`, each with a test in
a new file `rust/periodica-mat/tests/f2_derive_mech.rs`. Line numbers below are from the review;
find the functions by name (`hardness`, `elastic`, `fracture`).

1. HV to UTS (derive.rs:250-264): Tabor's `UTS = 3.3 HV` holds for work-hardening metals only.
   Gate it on data that the record already carries and that shows ductile metal behaviour. Read
   `record.rs` and pick the gate; say which fields you used and why. Don't add a material-class
   enum. Test: a ceramic-like record (HV 1500, no ductility data) gets no UTS; a steel record still
   does.
2. Isotropic identities (derive.rs:122-170): skip them when the record has an `Orthotropy` field.
   Test: an orthotropic record gets no derived G or K.
3. Irwin (derive.rs:174-195): K_IC is a plane-strain value, so use `G = K^2 (1 - nu^2) / E` when
   Poisson's ratio is present; keep `K^2 / E` (tagged `Estimated`) when it isn't. Leave the
   J_IC and fracture-energy mirror as it is; that is an owner decision.
4. Near-incompressible solids (`elastic`): `K = E / (3 (1 - 2 nu))` is ill-conditioned as nu nears
   0.5. The relative error of K is `2 d_nu / (1 - 2 nu)`: at nu = 0.4998 (the Rubber-Natural
   sheet) an error of 0.0001 in nu moves K by 50 %. Add a named constant (0.45) with that
   sentence in its doc comment. Above it, `K` from `(E, nu)` and `E` from `(K, nu)` are written
   `Estimated`, never `Derived`; `G` from `(E, nu)` is unaffected. Test: E = 2.0993 MPa,
   nu = 0.4998, both `Datasheet`, gives K `Estimated` and G `Derived`; steel (200 GPa, 0.29)
   still gives K `Derived`.
5. Anisotropic by class, no directional data (`elastic`): the sheet Timber-C24 has one modulus
   (E along the grain, 11 GPa), one Poisson's ratio (0.372), category `Wood` and no `Field`, so
   its `orthotropy` is `None`, item 2 does not catch it, and the identities give G = 4.0 GPa
   `Derived` (the standard's value is about a sixth of that). No value is better than that one.
   Add one private predicate, fed by one named constant list of category words (wood, timber,
   laminate, fibre or fiber composite; match case-insensitively on `record.category`), and skip
   the isotropic identities when it is true. Don't test sheet names. Test, on the real corpus:
   Timber-C24 has E and nu and no G and no K after `fill_derived`; Steel-1018 is unchanged.
   List in the report every corpus sheet the predicate matches.

This lane may run beside a `periodica_core` lane: use `-p periodica-mat` for every cargo command
and `cargo fmt -p periodica-mat`. Run `cargo test -p periodica-mat --no-fail-fast` and
`cargo clippy -p periodica-mat --all-targets -- -D warnings`. Then rerun
`cargo run -p periodica-mat --example domain_census` and commit the two regenerated files under
`docs/dev/`. Update oracles that change, with old and new numbers in your report.
