# Q1a: every constant in `chains.rs` and `derive.rs` has a derivation or a reference

Read `_common.md` first, including "Real quantities only". One deliverable: the eight changes
below in `rust/periodica-mat` (`src/chains.rs`, `src/derive.rs`, and `src/mix.rs` and
`src/consistency.rs` for item 6 only), their doc comments and tests, and the census rerun. No
datasheet changes. Use `-p periodica-mat` for every cargo command.

## Context
The owner's rule: no factor, threshold, clamp or range without a derivation or a reference. The
planner read these two files on 2026-11-11 and checked each source below on the web that day.
Write the doc comments from what this brief gives. Don't add a citation, a page number or a
range from memory; where the brief says a thing was not read or is not stated, the doc comment
says the same.

## 1. `RQ_OVER_RA` is `sqrt(pi / 2)` (`chains.rs`)
For a Gaussian height distribution with rms height `R_q`, the mean absolute height is
`R_a = R_q sqrt(2 / pi)`, so `R_q / R_a = sqrt(pi / 2) = 1.2533`. The code has 1.25. Replace the
constant with a small function that returns `std::f64::consts::FRAC_PI_2.sqrt()` (`sqrt` is not
`const`), and give the derivation in its doc comment.

## 2. The slope correlation: its source, its range, and rms from mean absolute (`chains.rs`)
Source as read: P. Teertstra, "Calculating interface resistance", Electronics Cooling, 1 May 1997
(electronics-cooling.com/1997/05/calculating-interface-resistance/). It gives
`m = 0.125 (sigma * 1e6)^0.402` with sigma the rms roughness in metres, "developed for the
surface roughness range 0.216 um < sigma < 9.6 um", and credits V. W. Antonetti, T. D. Whittle
and R. E. Simons, "An approximate thermal contact conductance correlation", ASME HTD-Vol. 170,
1991, pp. 35-42. There `m` is the mean absolute asperity slope. The 1991 paper itself was not
read; the page doesn't say which surfaces the fit covers and gives no scatter. Its worked example
checks the units: sigma 0.4 um gives m 0.0865, sigma 1.3 um gives m 0.139.
- Fix the doc comments: the year (1991, not 1993), the source as above, the range, and drop
  "ground, lapped and blasted metal surfaces" (not in the source read).
- `slope_from_ra` writes nothing when `R_q` in micrometres is outside `0.216..=9.6`. Those two
  numbers are the source's range.
- `SurfaceSlopeRms` is an rms slope and the correlation gives a mean absolute slope. For
  Gaussian slopes (the same assumption as item 1) `rms = sqrt(pi / 2) * mean absolute`. Apply
  that factor, with the function from item 1, and say so in the doc comment. Every estimated
  slope rises by about 25 %.

## 3. One Tabor factor (`chains.rs` 3.0, `derive.rs` 3.3)
Both are Tabor's relation `H ~ 3 sigma` (D. Tabor, *The Hardness of Metals*, Oxford, 1951): the
mean pressure under a fully plastic indentation of a ductile metal is about 3 times its flow
stress. Factors reported for metals run from 2.6 to 3.0 (secondary sources; the book was not
read). The flow stress is the yield stress for a metal that does not work harden and lies between
yield and UTS for one that does. `UTS[MPa] ~ 3.3 HV` is the same relation with HV in kgf/mm^2:
`9.80665 / 3 = 3.269`, rounded.
- One `pub(crate) const TABOR_CONSTRAINT_FACTOR: f64 = 3.0` in `derive.rs` with that doc
  comment; `chains.rs` uses it. Make `KGF_MM2_PA` visible to both files.
- `hardness` in `derive.rs`: `hv = TABOR_CONSTRAINT_FACTOR * uts / KGF_MM2_PA` and the inverse.
  Remove `TABOR_FACTOR`.
- Every HV estimated from UTS rises by 0.95 %; every UTS estimated from HV falls by 0.94 %.

## 4. Remove `H ~ 3 f_c` (`chains.rs`)
`CONSTRAINT_H_OVER_FC = 3.0` has no source. For a cohesive-frictional solid the ratio of
indentation hardness to cohesion depends on the friction angle (Ganneau, Constantinides and Ulm,
2006, as cited in the papers the planner found; not read), so no single `H / f_c` exists.
Remove the constant, `from_fc` and `measured_f_c`; fix the module doc and the doc comment of
`hardness_pa`. A sheet with no measured hardness scale and no ductile-metal yield then has no
`Hardness`. PlayTow agreed on 2026-11-11: it prefers a refused sheet to an uncited number.

## 5. `METALLIC_RESISTIVITY_MAX`: keep 2.0e-6, state the source (`derive.rs`)
Doc comment only. 2.0e-6 Ohm m is 200 micro-ohm cm, the top of the range (100 to 200
micro-ohm cm) where the resistivity of a metal saturates because the electron mean free path has
fallen to the interatomic spacing, the Mott-Ioffe-Regel limit: Mooij (1973) and Fisk and Webb
(1976), as reviewed by N. E. Hussey, K. Takenaka and H. Takagi, "Universality of the
Mott-Ioffe-Regel limit in metals" (2004), arXiv:cond-mat/0404263. That review says the number
is not sharply defined and that some correlated metals exceed it.

## 6. Poisson's ratio: the stability interval, one helper (`derive.rs`, `mix.rs`, `consistency.rs`)
`NU_MIN = -0.999` and `NU_MAX = 0.4999` are insets with no source. The strain energy of an
isotropic solid is positive definite when K > 0 and G > 0, which is `-1 < nu < 0.5` (Landau and
Lifshitz, *Theory of Elasticity*, section 5). Add `pub(crate) fn nu_is_stable(nu: f64) -> bool`
(the open interval) in `derive.rs`, use it at the three sites that use `NU_MIN..=NU_MAX`, and at
every place in `mix.rs` and `consistency.rs` that tests `-1 < nu < 0.5` by hand (`mix.rs` had
one before lane F2d, and F2d added more). Remove the two constants. Leave `fracture` in
`derive.rs` alone: its `-1 < nu <= 0.5` includes the incompressible limit on purpose, because
`1 - nu^2` has no singularity there; cite the same section in its comment.

## 7. `NU_ILL_CONDITIONED`: say what the number means (`derive.rs`)
Doc comment only; leave 0.45. The condition number of K with respect to nu is
`(nu / K) dK/dnu = 2 nu / (1 - 2 nu)`, which is 9 at nu = 0.45: an error of 1 % in nu is an
error of 9 % in K. The cut-off itself is a choice with no reference; say so, and that the owner
decides it (decision 15 in `docs/plan/playtow-sheets.md`).

## 8. Census
Run `cargo run -p periodica-mat --example domain_census` and commit the files it rewrites under
`docs/dev/`. Before that, copy `docs/dev/domain-census.json` to
`H:/claude-scratch/periodica-sheets/q1a/census-before.json`.

## Not in this lane
`DUCTILE_ELONGATION_PCT` (lane F2c), `HRC_TO_HV` (needs the standard), `MAX_PASSES` (a loop
bound), any id (lane Q1b), and constants in other files.

## Tests: `rust/periodica-mat/tests/q1a_constants.rs`, plus the unit tests the files already have
Build tables in memory; a test that reads a sheet names its path.
- Item 1: the function equals `(PI / 2.0).sqrt()`.
- Item 2: R_a 1e-6 m gives `sqrt(pi/2) * 0.125 * (sqrt(pi/2) * 1.0)^0.402`, computed in the
  test, within 1e-12 relative. R_a 0.1e-6 m and R_a 10e-6 m give no `SurfaceSlopeRms`. A stated
  slope is never replaced.
- Item 3: UTS 400e6 Pa on a ductile metal gives `HV = 3 * 400e6 / 9.80665e6`; the round trip
  HV to UTS to HV is exact within 1e-12 relative; yield 250e6 gives `Hardness = 750e6`.
- Item 4: a table with a compressive strength and nothing else gets no `Hardness`.
  `src/periodica/data/derived/composites/Concrete-C30.json` has no `Hardness` after ingest.
- Item 6: nu 0.49995 with E gives a finite positive K tagged `Estimated`; nu 0.5 and nu -1.0
  give none; `nu_is_stable` at -1.0, -0.9999, 0.0, 0.49999, 0.5.
- An existing test or oracle whose numbers move: change it only when one numbered item explains
  the move, and name the item in the report.

## Report (25 lines or fewer)
The commit hash; test counts and commands; every file touched (`git diff --name-only`); per
item, the number of sheets whose values changed, were lost or were gained, from a per-sheet diff
of the `props` in `docs/dev/domain-census.json` against `census-before.json` (the script
`H:/claude-scratch/periodica-sheets/review/census_diff.py OLD.json NEW.json` does this; don't
count from `domain-census.md`); the sheets that lost `Hardness` by item 4, by path; the sheets
that lost `SurfaceSlopeRms` by item 2 and the categories of the sheets that keep an estimated
slope; the sheets item 6 changed; every existing test or oracle whose numbers moved, with the
item; every number you added that is not a measured value or a unit conversion; anything in the
brief you found wrong or ambiguous (say so; don't change a rule on your own); what you did not
verify.
