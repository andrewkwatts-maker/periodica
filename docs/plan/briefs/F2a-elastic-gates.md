# F2a: gates on the isotropic elastic identities (lane `derive-elastic`)

Read `_common.md` first. One deliverable: the `elastic` rule in
`rust/periodica-mat/src/derive.rs` stops writing wrong or ill-conditioned moduli. Tests go in a
new file `rust/periodica-mat/tests/f2a_elastic_gates.rs`. Source: U3 (first half) of
`docs/plan/sheets-formula-review.md`, plus two finds of 2026-11-11.

## Known (read 2026-11-11)
- `fill_derived(t: &mut PropertyTable)` sees only the table. `orthotropy` and `category` are
  fields of `MaterialRecord` (`record.rs`), so `elastic` cannot test them today.
- The one record-level caller is `ingest.rs`, near line 197:
  `fill_concrete(&mut record.bulk); fill_derived(&mut record.bulk);`. Other callers are tests
  and `chains.rs` tests.
- `elastic` applies the isotropic identities to every record: (E, nu) gives G and K; (G, nu) and
  (K, nu) give E; (E, G), (E, K) and (G, K) give the third.

## Shape
Add a small `Copy` struct of gates (`DeriveGates`, one `bool` field for now,
`isotropic_elastic`; default: everything allowed), a function `fill_derived_with(t, gates)`, and
one function that reads the gates from a `MaterialRecord`. `fill_derived(t)` keeps its signature
and means "all gates open". `ingest.rs` calls the gated form: that is the only line to change
there. Lane F2b adds a second field later, so keep the struct easy to extend.

## Changes
1. A record with an `Orthotropy` field: skip the isotropic identities.
   Test: an orthotropic record gets no derived G or K.
2. Anisotropic by class, no directional data. The sheet Timber-C24 has one modulus (E along the
   grain, 11 GPa), one Poisson's ratio (0.372), category `Wood` and no `Field`, so its
   `orthotropy` is `None` and the identities give G = 4.0 GPa `Derived`; the standard's value is
   about a sixth of that. No value is better than that one. Add one private predicate fed by one
   named constant list of category words (wood, timber, laminate, fibre composite, fiber
   composite; case-insensitive match on `record.category`), and close the gate when it is true.
   Don't test sheet names. Test, on the real corpus: Timber-C24 has E and nu, and no G and no K;
   Steel-1018 is unchanged. List in the report every corpus sheet the predicate matches.
3. Near-incompressible solids. `K = E / (3 (1 - 2 nu))` is ill-conditioned as nu nears 0.5: the
   relative error of K is `2 d_nu / (1 - 2 nu)`, so at nu = 0.4998 (the Rubber-Natural sheet) an
   error of 0.0001 in nu moves K by 50 %. Add a named constant (0.45) with that sentence in its
   doc comment. Above it, K from (E, nu) and E from (K, nu) are written `Estimated`, never
   `Derived`; G from (E, nu) is unaffected. Test: E = 2.0993 MPa, nu = 0.4998, both `Datasheet`,
   gives K `Estimated` and G `Derived`; steel (200 GPa, 0.29) still gives K `Derived`.

## Verify
This lane may run beside a `periodica_core` lane: use `-p periodica-mat` for every cargo command.
`cargo fmt -p periodica-mat`; `cargo test -p periodica-mat --no-fail-fast`;
`cargo clippy -p periodica-mat --all-targets -- -D warnings`. Then rerun
`cargo run -p periodica-mat --example domain_census` and commit the two regenerated files under
`docs/dev/`.

## Report (15 lines or fewer)
Commit hash; test counts; every oracle that moved, with old and new numbers; the sheets the class
predicate matches; how many corpus sheets lost a derived G or K; what you did not verify.
