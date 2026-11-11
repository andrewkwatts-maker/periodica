# P2: no computed value is presented as measured (generated sheets, sound speeds)

Read `_common.md` first, including "Real quantities only". One deliverable: the three changes
below in `periodica-mat`, their tests, and the regenerated census. No datasheet is edited. Use
`-p periodica-mat` for every cargo command. This lane sits on top of S3a (a2bf4a8), which is not
pushed.

## What was found (planner's review of S3a, 2026-11-11)
- 126 sheets carry `"_derivation": {"source": "auto_generated", ...}` (46 in `active/alloys`, 30 in
  `active/materials`, 18 amino acids, 8 nucleic acids, 7 proteins, 6 cells, 6 biological
  materials, 5 cell components). 20 sheets in `active/biological_materials` carry
  `"derived_properties": {"model_used": "..."}` (a Voigt-Reuss, Gibson-Ashby or rubber-elasticity
  model). Python generators wrote them (`src/periodica/utils/biological_generator.py` and
  others). `rust/periodica-mat/src/ingest.rs` reads neither marker and tags every value
  `Source::Datasheet` (lines 141, 228, 461, 503), so model output passes as measured.
  `Blood.json` is one: a liquid with a Young's modulus of 3.3 MPa.
- S3a then derived a longitudinal sound speed for 139 exported sheets. 21 are false by 10 to 250
  times: dermis 27 m/s, blood 160 m/s, liver 506 m/s, where the measured values are about 1500
  to 1600 m/s. 20 of the 21 are generated sheets.
- The other one is `active/tissues/cartilage.json`, hand-written: E 1 MPa, nu 0.45, 70 % water,
  59 m/s. In a fluid-saturated solid the E and nu of a slow test describe the drained frame, and
  `K = E / (3 (1 - 2 nu))` is the drained bulk modulus, not the one a sound wave sees (Biot 1956;
  Gassmann 1951). The shear modulus is the same drained or saturated, so the shear speed stays
  valid. The sheet states its water content, but `canon.rs` drops the key.

## 1. A generated sheet's values are estimates (`ingest.rs`)
A sheet is generated when its root object has `_derivation.source == "auto_generated"`, or
`derived_properties.model_used` is a non-empty string. Write one small function that answers
this from the root object. Every value that ingest reads from a generated sheet is tagged
`Source::Estimated`; every other sheet keeps `Source::Datasheet`. All four sites that tag today
must use it (pass the source down where a site doesn't see the root). Change no value and no
other key mapping. Update the doc comments of `Source::Datasheet` and `Source::Estimated` in
`table.rs` only if they now read wrong.

## 2. Water content is a property (`property.rs`, `canon.rs`, `domains.rs`)
Add one id the way S3a added its ten: `WaterContent`, unit `1`, mass of water per total mass of
the wet material (wet basis). Sheet key `watercontentpercent` maps to it with `PCT_TO_FRAC`;
take that key out of the ignore list in `canon.rs`. Put it in the domain that holds `Porosity`.
Update `tests/domain_census.rs` if it counts ids.

## 3. Sound speeds (`acoustic` in `derive.rs`)
Replace the three rules with these. Keep each function at 60 lines or fewer and leave every
other rule in the file alone.
- No input may be `Estimated`. A wave speed is not computed from an estimated modulus or density.
  A `Derived` input counts as known: G or K from a stated E and nu is an exact identity.
- Shear, `sqrt(G / rho)`: G and rho present and positive.
- Longitudinal, solid, `sqrt((K + 4G/3) / rho)`: K, G and rho present and positive, and the
  sheet states no fluid phase: closed when `WaterContent` or `Porosity` is present and above
  zero (the drained-modulus reason above). Zero is not a threshold here; it is "no pore fluid
  and no pores stated".
- Longitudinal, fluid, `sqrt(K / rho)`: K and rho present and positive, G absent,
  `DynamicViscosity` present. The water and porosity gate does not apply to this form.
- Every result is tagged `Estimated`: a sheet's moduli are in general quasi-static and
  isothermal, and the wave moves with the adiabatic moduli at its own frequency. They agree
  within a few percent in metals and dense ceramics and differ by tens of percent in polymers.
  Say this, and the drained-modulus reason, in the doc comment. Still `set_if_better`, so a
  measured speed in a sheet outranks the estimate.
Update the definitions S3a wrote in the module doc of `property.rs` where they now read wrong.

## Tests
In `rust/periodica-mat/tests/s3a_new_properties.rs`, change the tests that these rules move, and
add:
- Steel stated as E 200 GPa, nu 0.29, rho 7850: both speeds present and `Estimated`, each equal
  to its formula computed in the test.
- The same with E tagged `Estimated`: no sound speed at all.
- The same with `WaterContent` 0.7: shear present, longitudinal absent. With `Porosity` 0.1: the
  same. With `WaterContent` 0.0: both present.
- Water (K 2.2 GPa, rho 998, viscosity 1.0e-3 Pa s, `WaterContent` 1.0): the fluid form,
  `sqrt(K / rho)` computed in the test, `Estimated`, no shear.
- The gate test, the "K and rho alone" test and the "measured 5900 survives" test stay.
New file `rust/periodica-mat/tests/p2_generated_sheets.rs`, sheets loaded by path under
`src/periodica/data`:
- An in-memory sheet with `_derivation.source: "auto_generated"` ingests with every value
  `Estimated`; the same sheet without the marker is `Datasheet`. The same pair for
  `derived_properties.model_used`. An empty `model_used` string is not a marker.
- `active/biological_materials/Blood.json`: after ingest and derivation no value in `bulk` is
  `Datasheet` or `Derived`, and it has neither sound speed.
- `active/tissues/cartilage.json` (not generated): `WaterContent` is 0.7 and `Datasheet`; no
  longitudinal speed; the shear speed is `sqrt(G / rho)` with `G = E / (2 (1 + nu))`.
- `derived/alloys/Steel-1018.json` (not generated): E is `Datasheet`; both speeds are present.
- A corpus walk (the way `tests/corpus.rs` or `tests/ingest_census.rs` walks it): no generated
  sheet has a sound speed, and no sheet with `WaterContent` or `Porosity` above zero has a
  longitudinal speed from the solid form (a sheet that states its own speed is exempt).

## Checks before the commit
- Columns. The S3a baseline is in `H:/claude-scratch/periodica-sheets/review/export-s3a`
  (written by `cargo run -p periodica-mat --example export_columns -- <dir>`). Export again to
  `H:/claude-scratch/periodica-sheets/p2/export`. Compare every `.f64` file with a small Python
  script (`python -I`), row by row with the `.src` tags. Expected: the two sound-speed columns
  lose rows; a new `WaterContent` column; in every other column no value changes where both
  exports have one. A value that appears, disappears or changes elsewhere means another rule
  reads the tag (for instance one that asks for a stated input). Don't change that rule: list the
  property, the count and the rule's name in the report.
- Census. Rerun `cargo run -p periodica-mat --example domain_census` and commit the two files
  under `docs/dev/`. Diff the per-sheet `props` against
  `git show a2bf4a8:docs/dev/domain-census.json` (prefix `MSYS_NO_PATHCONV=1`), keyed on
  `folder/name`. Count: sheets with a tag change; sheets that lost each sound speed and sheets
  that keep each; sheets that gained `WaterContent`; any other property lost or gained.

## Report (20 lines or fewer)
The commit hash; the test commands and counts; `git diff --name-only a2bf4a8..HEAD`; the census
and column numbers above; every existing test whose expectation moved and why (change one only
when it is a tag on a generated sheet or a sound speed); every number you added that is not a
measured value or a unit conversion; what you did not verify.
