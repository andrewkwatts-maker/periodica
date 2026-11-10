# F2: derived mechanical values (lane `derive-mech`)

Read `_common.md` first. F1 is merged (it also edits `derive.rs`). Source:
`docs/plan/sheets-formula-review.md`, BUG 4, U3 (first half) and U13, plus item 4 below (found
2026-11-11). Deliverable: four changes in `rust/periodica-mat/src/derive.rs`, each with a test in
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

Run `cargo test -p periodica-mat`. Update oracles that change, with old and new numbers in your
report.
