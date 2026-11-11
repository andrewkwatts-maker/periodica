# Plan: periodica sheets and formulas for the PlayTow engine (2026-11-11)

Owner request: "review this project for bugs/improvements so it's offering clear, comprehensive and
usable data sheets/formulas to PlayTow engine needs; discuss with the PlayTow session what it would
like from you."

## What PlayTow asked for (playtowecosystem-16, 2026-11-11)
periodica owns the periodica API. The PlayTow session sends requests and doesn't edit periodica.
Requirements: `PlayTow/docs/plan/REQUIREMENTS.md` (REQ-MAT, REQ-UNITS, REQ-THERMAL, REQ-FIELDS,
REQ-SPECTRAL). The request list and the properties each PlayTow reader uses today:
`PlayTow/docs/plan/modules/MATTER.md:73-90`.

Owner ruling of 2026-11-11 (`PlayTow/docs/plan/EDITIONS.md`): PlayTow is a base engine plus two
paid plugins. The base engine doesn't depend on periodica. Only the Engineering plugin (fields,
heat, SI units) and the Full Spectrum plugin (emission spectra, light beyond RGB) read periodica
sheets. The needs below are unchanged.

The asks, in priority order:
- (a) One generic sheet schema for real and imagined materials. Every value carries its SI
  dimension, its uncertainty, and a provenance flag: measured, derived, authored, or no data.
- (b) Temperature and frequency dependence, never a constant with a hidden reference temperature.
- (c) No material classes or names that the engine must branch on.
- (d) Spectra over wavelength from IR to UV: n, k, emissivity and emission lines.
- (e) Phase data: melt, boil, sublime, the solid phases, and the latent heats.
- (f) Electrical, magnetic and acoustic columns.
- (g) A plain column export per build. periodica must not depend on Themelios or Topos.

The four requests, with PlayTow's answers to the planner's questions:
| ID | Request | Agreed form |
|---|---|---|
| MP-PER-1 | Remove the class branch at `ingest.rs:197` | Lane F6: f_ck is its own property and its presence triggers the fib relations |
| MP-PER-2 | Temperature and frequency dependence | Arithma expression strings, each with a validity range (min, max, unit) and its source points beside it. Tabulated data becomes a piecewise or interpolating expression, so one evaluator serves fits and tables |
| MP-PER-3 | New properties | Latent heats (fusion, vaporisation, sublimation), magnetic permeability, speed of sound, acoustic absorption, viscosity, emission lines |
| MP-PER-4 | Column export and uncertainty | Raw little-endian f64 column files plus one manifest (property, SI unit, row count, sheet ids in row order, missing-value mask). One row per sheet. Curves are sampled only on a grid the caller gives. One standard uncertainty per value; min and max too where the datasheet gives a range. PlayTow wraps the columns itself |

PlayTow's answers of 2026-11-11 on MP-PER-3 and MP-PER-4 (they set the S3 and S5a briefs):
- Acoustic loss, intrinsic quantities only. Solids: loss factor eta = tan delta (dimensionless), a
  scalar now. Fluids: bulk attenuation alpha(f) in Np/m as an MP-PER-2 expression; until then a
  scalar alpha/f^2 (Np s^2/m) at the reference state where the classical f^2 law holds, absent
  otherwise. No surface absorption coefficient: that belongs to a product, not a material.
- Speed of sound. Solids: two properties. Longitudinal is the bulk wave in an unbounded medium,
  sqrt((K + 4G/3)/rho), not the thin-rod sqrt(E/rho); the property doc must say so. Shear is
  sqrt(G/rho). Fluids: one value, and shear reads absent, not zero. Derive from moduli and density,
  flagged derived, when nothing is measured.
- Viscosity: dynamic, Pa s, a scalar now with its reference temperature stored beside it. Absent
  for solids. The MP-PER-2 expression replaces it later.
- Emission lines: vacuum wavelengths from 200 to 2500 nm, ion stages I and II, the 50 strongest per
  (element, stage). Carry A_ki (1/s), the upper-level energy E_k (J) and the statistical weight g_k
  where NIST has them, NaN otherwise; prefer lines with A_ki when intensities tie.
- Column export: the form in `briefs/S5a-column-export.md` fits. Asks: the source legend in the
  manifest, `.src` as the authority for presence, a content hash per column file, and later
  versions that only add files. One standard uncertainty per value in the next version.
- PlayTow will adapt its two optics readers to the `Result` API when it pulls 4191f64.

PlayTow's reply of 2026-11-11 on the ten S3a ids: all names and units accepted, no change asked.
The values it needs first (this orders lane S3b; a reader's needs, the source is owner decision 11):
1. `MechanicalLossFactor`: Steel-1018, Concrete-C30, Steel-B500B, Aluminum-6061, Timber-C24,
   Rubber-Natural, Brick-Clay, Mortar-CementLime (modal damping and sound; the widest need).
2. Sound speeds: longitudinal and shear for the same eight solids; longitudinal only for water
   and air.
3. `DynamicViscosity` with `ViscosityReferenceTemperature`, and `AcousticAttenuationOverF2`:
   air, water.
4. `RelativePermeability`: Steel-1018, Magnet-N42, Aluminum-6061 (about 1, the non-magnetic
   check). A soft-iron or electrical-steel sheet would be new; no reader needs it yet.
5. Latent heats: water (fusion, vaporization), then Steel-1018 and Aluminum-6061 (fusion).
   Sublimation has no reader yet.

The sheets PlayTow loads today, most used first: Steel-1018, Concrete-C30, Steel-B500B,
Concrete-C40, Concrete-C25, Aluminum-6061, Magnet-N42; planned: Timber-C24, Rubber-Natural,
Mortar-CementLime, Brick-Clay. All eleven exist (`data/derived/` or `data/reference/`).
Missing: a bulk water sheet and an air sheet. `molecules/Water.json` describes the molecule
(density in g/cm3, state at STP), not the liquid as a material; there is no air sheet at all.
S3b adds both as material sheets.

## How lanes run
- Lanes work in the main checkout, `H:/Projects/MathsPhysicsEcosystem/periodica`, on `main`. There
  is no lane branch and no second checkout: the worktree `../periodica-sheets` and its branch
  `claude/sheets` were removed on 2026-11-11, after Q1a (decision 10).
- No new worktrees: a TokenWatch hook bans `git worktree add` (worktrees took 108 GB). One Rust
  lane runs at a time; a lane that touches no Rust source may run beside it.
- An executor commits to local `main` with explicit paths and doesn't push. The planner reads the
  diff, runs the workspace tests, pushes `main`, then checks GitHub Actions.
- `sync.bat` commits and pushes `main` in every repo here, so lane work can reach `origin` before
  the planner has reviewed it. The planner then reviews what is on `origin/main` and fixes forward.
- At most 3 agents at once; one deliverable per brief (`briefs/_common.md`).

## Lanes
| Lane | Brief | Status |
|---|---|---|
| R1a formula review | `briefs/R1a-formula-review.md` | merged 30786c4: 36 OK, 6 BUG, 13 UNCLEAR in `sheets-formula-review.md` |
| M1 element bulk properties | `briefs/M1-merge-element-bulk.md` | merged 5a7a054. It also overwrote 28 ion sheets; F6 restores them |
| F3 mix (Turner expansion, Hill consistency, Lorentz-Lorenz n, `mix_tagged`) | `briefs/F3-mix.md` | merged bb3bd57 |
| R0 CI (Arithma clone, fmt, path-dependency guard) | `briefs/R0-ci-green.md` | merged 11b2a79. First run on GitHub (4191f64) failed in two places; see below |
| R0c reproduce the Python CI failure with the Rust extension | `briefs/R0c-python-ci-repro.md` | done, no repo change. With the extension live: 7 failed, 1512 passed, 99 skipped, the same on a second run. All 7 come from the Rust core lagging the Python code: 6 from the datasheet reader (R0d), 1 from the grain hash (R0e). Log `H:/claude-scratch/periodica-sheets/pyrepro/pytest.log` |
| R0d Rust core reads a datasheet as Python does | `briefs/R0d-rust-sheet-parity.md` | done 53f5f29, 0a8bc00 (on main). `props_of` is a port of `properties.flatten` and reads the group lists from `config/property_schema.json`; an unreadable schema is an error. Local run with the extension: 1 failed (the R0e test), 1519 passed, 99 skipped. Limit: literal keys only (R0f) |
| R0e Rust grain hash matches Python's | `briefs/R0e-grain-hash.md` | done, f43e075. Local run with the extension: 0 failed, 1520 passed, 99 skipped. CI on 0b2cb4a is green in all four jobs (run 42890111992). The unused `sha1` dependency is removed. Not checked: NaN or huge coordinates (numpy's cast gives the i64 minimum, Rust's `as i64` gives 0 or saturates), points with fewer than 3 columns |
| R0f Rust resolves aliases and units (a port of `DataSheet.resolve`) | to write | proposed, after R0e. Today `py_sample` declines any key that is not literal, and in strict `PERIODICA_BACKEND=rust` mode a decline raises `RustBackendUnavailable` (`_dispatch.py:165`). Ties to owner decision 13 |
| F5 ion keys, elastomer floor | `briefs/F5-corpus-green.md` | merged f9902fe; F6 revisits the ion keys |
| F4 optics (gold interband terms, validity ranges, Planck guard) | `briefs/F4-optics-models.md` | merged 7eedcac. The gold fit matches Johnson and Christy within 0.05 in R; its citation (Etchegoin 2006) is from memory and unchecked |
| F6 derived data green, MP-PER-1 | `briefs/F6-derived-green.md` | done dcfd0dc, 4191f64. f_ck is the property `CharacteristicCompressiveStrength` and triggers the fib relations; no code reads `Class`. 28 ion and 39 isotope sheets restored from 9890238 |
| R0b release workflow builds with Arithma | `briefs/R0b-release-workflow.md` | done 37020cb: wheels and sdist jobs fetch Arithma at commit 447a7c0. Not run on GitHub. The sdist can't build a wheel on its own (owner decision 9) |
| S5a column export of scalar properties | `briefs/S5a-column-export.md` | done, 9fb3908 and d35db37. `periodica_mat::export`: `load_corpus(data_dir, keep)`, `columns`, `write_dir`, `read_dir` (checks length and hash), `has_density`. Format 1: per property one `.f64` file (little-endian) and one `.src` file (the source tag, which also says whether a value is present), plus a manifest with `sheet_ids`, `shadowed`, `source_legend` and an `fnv1a-64` hash per file. Ids are file stems, sorted; the earlier root wins and the other copy is listed as `shadowed`. The row rule is the caller's predicate. With `has_density`: 328 rows, 87 properties, 299,214 bytes, 162 shadowed copies, 506 files skipped. 74 of the rows fail `is_material_like` (25 cells, 23 elements, 21 molecules, 3 composites, 1 material, 1 alloy); nobody has checked that the cells and molecules are bulk matter |
| F1 derived optics (Fresnel with k, no emissivity mirror) | `briefs/F1-derive-optics.md` | done 957f83a, 0500d1f. Reflectance from n needs k (`Derived`) or a known transmittance (k = 0, `Estimated`); with k it also covers metals with n below 1. Absorptance and emissivity no longer copy each other. Open: R is `Estimated` even when the transmittance is itself an estimate |
| P1 pytest must not rewrite package data | `briefs/P1-tests-leave-data-alone.md` | done 1247dc3, 826cc50. Only `tests/test_cli.py` wrote to the package; it now builds into a private copy of `derived/` and its fixture checks that the patch ended |
| F2a gates on the isotropic elastic identities (orthotropy, wood-class sheets without directional data, near-incompressible K) | `briefs/F2a-elastic-gates.md` | done, 8ed7d85. `DeriveGates` (`isotropic_elastic`), `fill_derived_with(t, gates)`, `gates_for(record)`; ingest uses the gated form. The class predicate matches one corpus sheet, Timber-C24, which now has E and nu and no G and no K. Above nu = 0.45 a K from (E, nu) is `Estimated`: Rubber-Natural and seven tissue sheets. No oracle moved. Not checked: whether any corpus sheet has `orthotropy` set (the gate has a unit test only). CI was red on ff3ae8c and 2dacb57 (Rust `Test` step): the F2a test found Steel-1018 by a directory walk, and `reference/alloys/Steel-1018.json` has no Poisson's ratio, so it failed wherever `read_dir` lists `reference/` first. Fixed in 2cd310e (sheets loaded by path) |
| F2b Tabor's rule for ductile metals only (three sites); plane-strain Irwin | `briefs/F2b-tabor-gate-irwin.md` | done, 8a08b5a and 351d5de (the second is the planner's review fix). `shows_ductile_metal(table)` in `derive.rs` gates all three Tabor sites: a stated resistivity of 2e-6 Ohm m or more says no; a metallic resistivity says yes unless a stated elongation is under 1 %; with no resistivity, an elongation of 1 to 60 % says yes; with neither field, no. Irwin is `G = K^2 (1 - nu^2) / E` for nu in (-1, 0.5], else `K^2 / E`, both `Estimated`; the reverse (J_IC to K_IC) uses the same factor, so an estimated K_IC from J_IC or G_F rises by `1 / sqrt(1 - nu^2)` where nu is stated (2 % at nu = 0.2). Census, per property: 53 sheets lost estimates, none gained or changed tag. 21 tissue sheets and 14 polymer sheets lost `HardnessVickers` (HDPE in `active/materials`, Polycarbonate_Makrolon and Rubber-Natural also `Hardness`); 6 ceramics lost the UTS and `TensileStrength` estimated from HV; Brick-Clay and Mortar-CementLime lost HV and now take `Hardness` from `3 * CompressiveStrength`; 4 concrete sheets lost HV; Timber-C24 lost both. Steel-1018, Aluminum-6061 and Steel-B500B in `derived/alloys` are unchanged (asserted with values). Five metal sheets lost theirs because they state neither field: see F2c. Not closed: a polymer sheet with an elongation of 1 to 60 % and no resistivity still passes; no corpus sheet does today |
| F2c metals with no ductility data | to write | proposed. `derived/alloys` Magnesium-AZ31B (lost `Hardness` and HV) and Solder-Sn63Pb37 (HV), and the thin `reference/alloys` copies of Steel-1018, Aluminum-6061 and Titanium-Ti6Al4V (both), get no Tabor estimate after F2b. None is one of PlayTow's eleven sheets. Two ways: add a sourced elongation or resistivity to each sheet (owner decision 11), or a `DeriveGates` field that `gates_for` sets from the sheet's `Composition`. Deleting the `reference/` copies that `derived/` shadows removes three of the five. Q1 adds one item here: `DUCTILE_ELONGATION_PCT = (1.0, 60.0)` in `derive.rs` has no reference for either bound. The convention the planner found on 2026-11-11 is "ductile at 5 % elongation or more" (Shigley, from secondary sources; the book was not read), and nothing gives an upper bound |
| F2d a one-part mix returns the part's stored values; a measure of stored E, nu, G, K disagreement | `briefs/F2d-one-part-mix.md` | done, ae7a52d, 2c7ac95, fac0ac9. From PlayTow on 2026-11-11. (1) `mix_tagged` returns the stored value and its tag for a mix of one part, or of parts that are the same record (same pointer or equal records), for every property; a property the part lacks is `Missing`. (2) A part that states E and nu with `-1 < nu < 0.5` gives the Hill average of E and nu its K and G from that pair; any other part gives its stored K and G. `BulkModulus` and `ShearModulus` still mix from the stored values, so every property is continuous at the one-part limit. Steel-1018 at fraction 1 returns E 200 GPa and nu 0.29 (before: 203.9 GPa and 0.286). (3) `elastic_mismatch(&PropertyTable)` returns the relative difference of the stated G and K from the stated E, nu pair. It judges nothing and has no tolerance. It needs the record as the sheet states it, so `ingest` is now `ingest_stated` (public, no computed value) plus the fills. The corpus record is `rust/periodica-mat/tests/oracles/elastic-mismatch.txt`: 177 lines over the four data roots (`active/` and `defaults/` element sheets are copies, so an element appears twice). Largest shear: Rh 27.3 %, U 23.8 %, demo_hydrogel 14.9 %, Solder_60_40 10.7 %, Copper-C11000 6.7 %; Steel-1018 2.25 %, Aluminum-6061 0.38 %. Largest bulk: HDPE 71.5 %, Ru 69.3 %, Yb 55.5 %. No sheet was changed: which constant is wrong needs a cited source (decision 11). Five generated tissue sheets read 0.00 because their generator computed G from E and nu. Known limit: for a part with nu near 0.5 the K behind the mixed E and nu is `E/(3(1-2nu))`, not the stated K; continuity at the one-part limit forces that. A one-part mix now carries the part's own tag (`Datasheet` for a stated value), not `Estimated`. The column export is byte for byte the one at aef7952 (181 files); 501 Rust tests pass |
| S3a new scalar properties (MP-PER-3): ten ids, sheet keys, sound-speed rules | `briefs/S3a-new-scalar-properties.md` | done, a2bf4a8, pushed with P2. No data values; the data and the emission lines are later lanes. Its sound-speed rule was replaced by P2 before the push |
| P2 no computed value passes as measured | `briefs/P2-generated-sheets-are-estimates.md` | done, aef7952. Found at the review of S3a: 21 of 139 exported longitudinal speeds were false by 10 to 250 times (dermis 27 m/s, blood 160 m/s; measured about 1500 to 1600). (1) A sheet a Python generator wrote (`_derivation.source == "auto_generated"`, 126 sheets, or a non-empty `derived_properties.model_used`, 20 sheets) now ingests as `Estimated`, not `Datasheet`: 96 census sheets, 76 of them alloy and material sheets, 20 tissue sheets. (2) New id `WaterContent` (wet-basis mass fraction; the key `water_content_percent` was dropped before): 21 sheets. (3) `acoustic` takes no `Estimated` input, is closed for the longitudinal speed where a sheet states `WaterContent` or `Porosity` above zero (drained moduli: Gassmann 1951, Biot 1956), and tags every result `Estimated` (quasi-static moduli, not the adiabatic ones the wave moves with). Census against a2bf4a8: 26 sheets lost the longitudinal speed (20 generated tissue sheets, cartilage, Granite_Westerly, three concretes, Rubber-Natural) and 198 keep it; 23 lost the shear speed and 201 keep it. Side effect, not a fault: `Hardness` changed on 74 generated alloy sheets and went missing on 2 (Al-Si_Alloy_4xxx and its hot-rolled copy), because `chains.rs` no longer reads a generated Vickers number as measured and falls back to Tabor from yield. No stated value changed in any column |
| Q1a every constant in `chains.rs` and `derive.rs` has a derivation or a reference | `briefs/Q1a-constants-cited.md` | done, 6e4649c and 3bdaf23 (the second is the planner's review fix: two stronger tests and one doc comment). The planner read both files and checked each source on the web on 2026-11-11. (1) `rq_over_ra()` is `sqrt(pi/2)`. (2) The slope fit cites Teertstra 1997, who credits Antonetti, Whittle and Simons 1991; it writes nothing outside 0.216 to 9.6 um rms roughness and returns an rms slope, `sqrt(pi/2)` times the fit's mean absolute slope. A test reproduces the source's worked example (0.4 um gives 0.0865, 1.3 um gives 0.139). (3) One `TABOR_CONSTRAINT_FACTOR = 3.0` serves both files. (4) `H ~ 3 f_c` is gone: no single ratio exists for a cohesive-frictional solid, and PlayTow prefers the refusal. (5) `METALLIC_RESISTIVITY_MAX` is the Mott-Ioffe-Regel saturation, 200 micro-ohm cm (Hussey, Takenaka and Takagi 2004). (6) `nu_is_stable` is the open interval `-1 < nu < 0.5`; `NU_MIN` and `NU_MAX` are gone. (7) `NU_ILL_CONDITIONED` keeps 0.45 with its meaning stated (decision 15). What moved in the data, from the column export and the per-sheet census diff: nine sheets lost `Hardness`, not the three the brief expected, each one `Estimated` from compressive strength alone: `derived/composites` Concrete-C25, Concrete-C30, Concrete-C40 and Concrete-Mix, `derived/ceramics` Brick-Clay and Mortar-CementLime, `active/tissues/cartilage`, and `reference/ceramics` Alumina and SiliconCarbide (their `derived/ceramics` twins keep `Hardness` from a stated Vickers number, so the export loses seven rows). `HardnessVickers` rose 0.952 % on the 29 sheets where it is estimated from UTS (Steel-1018 133.33 to 134.60, Aluminum-6061 93.94 to 94.83, Steel-B500B 163.64 to 165.19). One estimated slope remains, Steel_8620_CaseHardened, 0.0946 to 0.1187 rad; Granite_Westerly lost its slope (R_q 62.7 um). Item 6 changed no sheet (the nearest is Rubber-Natural at nu 0.4998), and no UTS in the corpus is estimated from HV. Still open: `HRC_TO_HV` (needs the standard); `chains.rs` takes `Hardness` from 3 times yield while `derive.rs` takes HV from 3 times UTS, which disagree for a metal that work hardens; the slope range test includes its end points where the source's range is strict. PlayTow was told on 2026-11-11 |
| Q1b ids that are not material properties | to write | proposed. PlayTow answered on 2026-11-11 (playtowecosystem-e0). Not read by PlayTow, free to remove: `MaxPrincipalStress`, `MaxShearStress`, `VonMisesStress` (stress states), `CriticalFlawSize`, `ReductionOfAreaPct`, `ParisC` and `ParisM` (`ParisC` has unit `1` but its unit depends on `ParisM`). Read, and PlayTow will change its side: `KineticStaticRatio` (Dynameis `sheet.rs:155`; no sheet states it; it belongs in `PairTable` as a pair value), `EmissionTemperature` (Phantasia `appearance.rs:90`, a fallback; no definition; remove), `ElongationPct` (one line that divides by 100; a fraction id is fine). `PropertyId` is `#[repr(u16)]` and dense, and its names are keys in `.pmat` files, so removing an id renumbers the rest: the brief must say what happens to stored `.pmat` files. PlayTow's checkout stays at 5c70977 until its next session pulls; send it the commit and the new id names |
| Q1c defaults and constants outside `chains.rs` and `derive.rs` | to write | proposed. A scout listed every numeric literal in `periodica-mat/src` except `optics/` on 2026-11-11: 17 with no source. `record.rs:233` default grain size 50e-6 m; `record.rs:236` seed density 8e12 per m^3; `record.rs:281-284` `PhaseDistribution` defaults 1.0, 3, 0.5, 0.9; `record.rs:403` and `ingest.rs:607` default colour [0.75, 0.75, 0.78]; `ingest.rs:380` an exponent 1.5 called a stereological relation with no author; `ingest.rs:415` `clamp(1.0, 8.0)` on octaves; `ingest.rs:336-338` and `record.rs:159` default cell angle 90 degrees; `ingest.rs:476` default axis [1, 0, 0]; `ingest.rs:633-636` sRGB constants with no standard named; `derive.rs:54` `nidx >= 1.0`. A default that stands in for a value no sheet states is an invented quantity under the owner's rule: each becomes an `Option`, or gets a reference. `optics/*.rs` was not read. Sheets, same rule: element emission wavelengths come from `utils/physics_calculator.py` and ingest as `Datasheet`; HDPE's stated moduli give 1372 m/s against a measured value near 2400 and disagree with each other by 71 % in K |
| R1b domain census | `briefs/R1b-domain-census.md` | done, 5f306e1 (on main). Findings below. Rerun the example after F2 and S3a |

State at 826cc50, pushed to `main` on 2026-11-11: 433 Rust tests pass and none fail; fmt, clippy
and `cargo check --workspace --all-features` pass.

GitHub Actions on 4191f64 (run 42866560822) failed in two places:
- Rust job, "Format check": `cargo fmt --all` also checks the Arithma path dependency, and three
  files on Arithma's `main` aren't formatted. periodica's own code is clean under rustfmt 1.88 and
  1.95. Fixed: CI runs `cargo fmt --check`, which covers the four workspace crates only.
- Python jobs (3.11, 3.12, 3.13), "Run tests": the runner builds the Rust extension; no local run
  had it, so local runs skipped the tests that need it. Lane R0c reproduces it.

The runs on dc131e7 and 826cc50 (42869060902, 42871652719) pass format, clippy and the workspace
tests on Linux. Two failures remain:
- Rust job, "Test the PyO3 bindings": `surface_registers_exactly_the_expected_names` fails
  because the module also registers `_probe_input_ptr`, `_probe_output` and
  `_probe_scale_inplace`. They come from `pyfacade/numpy_probe.rs`, a spike whose header says
  "Removed before commit"; a work-in-progress sync committed it (0ada509). Fixed by removing the
  module. The `numpy` crate is now unused in `periodica_core` and stays in the `python` feature
  for the zero-copy work the spike was testing.
  The Rust job is green on GitHub from 48f2b17 (run 42874255586).
- Python jobs, "Run tests": 7 tests fail once the extension is built (R0c). Since commit 02b6807
  Python flattens nested property groups and answers other spellings of a key
  (`periodica.properties`); the Rust twin still reads only a flat `Properties` block, so with the
  extension `data_sheet()` is empty for every curated sheet and `sample()` returns None. The
  newest release on PyPI is 2.0.2 (2026-05-17, wheels with the extension), which predates that
  commit, so no published wheel has the fault; the next release would. Lanes R0d and R0e fix it.
  Three vocabularies now name the same quantities: `property_schema.json` (Python),
  `periodica-mat/src/canon.rs` (Rust, what PlayTow reads) and the Rust core's reader. One of them
  should generate the others (owner decision 13).

To run the binding tests on this machine: the Python on PATH is the Windows Store build, whose
DLL a test binary can't load (exit 0xc0000022). Set `PYO3_PYTHON` to
`%LOCALAPPDATA%/Programs/Python/Python313/python.exe` and put that folder first on PATH.

Found by F6, not yet fixed:
- A pytest run deletes and rebuilds the ion and isotope sheets inside the package
  (`tests/test_cli.py:94-113`), and the Python builder writes about 26 more keys per sheet than
  the native composer. Lane P1 isolates the tests and reports on the builder difference.
- The emission wavelengths in the element sheets look computed
  (`utils/physics_calculator.py`, a Rydberg-style estimate clamped to 100-1000 nm), yet they
  ingest as `Source::Datasheet`. Their unit (nm) isn't stated in the key. For S1 and S3.

## Schema lanes (after the census; one brief each)
- S1 (MP-PER-4): one standard uncertainty per value, plus min and max where given. Map `Source` to
  PlayTow's flags: measured, derived, authored, no data. Open point: sheets under `data/derived/`
  are generated by code but ingest as `Source::Datasheet`.
- S2 (MP-PER-2): temperature and frequency dependence as Arithma expressions with a validity range
  and source points. The optics models already return an error outside their range (F4).
- S3 (MP-PER-3), in three lanes. S3a: the ids, sheet keys and sound-speed rules (brief ready).
  S3b: the values, each with a cited source, in PlayTow's order of need (above), plus new water
  and air material sheets. S3c: emission lines from the NIST Atomic Spectra
  Database, a new record field, not a scalar. S3b and S3c wait on owner decision 11.
- S4: phase data beyond the latent heats (sublimation point, solid phases).
- S5 (MP-PER-4): the column export.

## Gaps found on the first read (origin/main 9890238)
| Ask | Present | Missing or wrong |
|---|---|---|
| a | A unit string per property; `Source` (Absent/Datasheet/Derived/Estimated/UserOverride) | No uncertainty. The dimension is a string, not an Arithma dimension. `ParisC` is declared unit "1". |
| b | Optics dispersion models | Every other value is a scalar with no temperature. |
| d | `optics/` spectrum, lines, planck, xyz | No datasheet carries Drude or Sellmeier coefficients; gold is the only built-in metal. |
| e | Melting, Boiling, Solidus, Liquidus, GlassTransition | Latent heats, sublimation, solid-phase transitions. |
| f | Resistivity, conductivity, permittivity, dielectric strength, Remanence, Coercivity | Magnetic permeability, saturation, B-H, sound speed, acoustic loss. `Permeability` is the hydraulic one (m²). |
| g | Nothing | A column export. |

## Gaps in the sheets PlayTow loads (read 2026-11-11, c6a1e8b)
What each sheet gives the elastic and sound-speed rules today. Values to add are read from a
cited source in the data lane (S3b), never recalled.
- Timber-C24: wrong today. The sheet holds one modulus (E_0 = 11 GPa, along the grain) and one
  Poisson's ratio (0.372) and no `Field`, so `orthotropy` is `None` and the isotropic identities
  run: shear modulus 4.0 GPa `Derived` (EN 338 gives G_mean of about 0.69 GPa for C24, a factor of
  6) and bulk modulus 14.3 GPa. The F2 gate tests for an `Orthotropy` field, so it does not catch
  this sheet. Fix in the data lane: a `Field` with `axial` and `transverse` blocks (E_90, G, the
  strengths), read the way `active/materials/Carbon_Fiber_Epoxy.json` is. Its sound speed is
  directional (along the grain about `sqrt(E_0 / rho)`, far lower across), so one scalar is the
  along-grain value and the sheet must say so.
- Steel-B500B: Young's modulus and density only. No Poisson's ratio, so no shear or bulk modulus
  and no sound speed. Add Poisson's ratio from a steel standard.
- Rubber-Natural: Poisson's ratio 0.4998, so the bulk modulus from `(E, nu)` moves 50 % for an
  error of 0.0001 in nu, and the longitudinal speed with it. F2 item 4 marks it `Estimated`; S3b
  gives a measured speed.
- Magnet-N42: density, remanence and coercivity only. No elastic data. PlayTow asks only for its
  permeability.
- Concrete-C25, C30, C40: no modulus in the sheet; `fill_concrete` gives the fib tangent modulus
  from f_ck, so both speeds derive. A sound wave sees the tangent (dynamic) modulus, not the
  secant one: an argument for keeping the tangent value in owner decision 6, or for storing both.
- Steel-1018, Aluminum-6061: E, G, nu and density present; both speeds derive.

## Census findings (R1b, 5f306e1; `docs/dev/domain-census.md`)
Rerun with `cargo run -p periodica-mat --example domain_census`. The domain table is
`rust/periodica-mat/src/domains.rs`; a test fails when a new `PropertyId` is in no domain.
- Contact: no `PropertyId` for friction or restitution, and PlayTow wants none (its answer of
  2026-11-11). Both are pair properties: measured friction goes in the existing `PairTable`,
  keyed by the two record ids; restitution is not read yet, and when it is, its entries carry
  the impact speed. Pairs first, unchecked: timber-timber, timber-steel, steel-concrete,
  concrete-concrete, steel-steel.
- Contact inputs. PlayTow derives friction from `Hardness`, `SurfaceEnergy`, `SurfaceSlopeRms`
  and `KineticStaticRatio` (with E and nu) and refuses a sheet without the first two. In the
  eleven: `SurfaceEnergy`, `SurfaceSlopeRms` and `KineticStaticRatio` are in none (corpus: 18
  sheets, 2 sheets and none). `Hardness` is in ten, always `Estimated`: from a measured Brinell
  number (Steel-1018, Aluminum-6061), from `3 * yield` (Steel-B500B, and wrongly Timber-C24,
  Rubber-Natural, Mortar-CementLime, Brick-Clay), from `3 * f_c` (the concretes). F2b gates the
  Tabor rule; S3b adds surface energy for the eleven and a measured hardness for timber and
  rubber.
- PlayTow reads neither `ShearModulus` nor `BulkModulus`; its lattice is isotropic from E and nu.
  Timber's shear error is theirs to fix on their side. F2a's wood gate still matters here: the
  S3a sound speeds derive from K and G.
- Acoustic: empty until S3a.
- Thermal and electrical: nine of the eleven sheets have no value, all except Steel-1018 and
  Aluminum-6061 (table 2 of the census). S3b adds conductivity and heat capacity.
- Optical: none of the eleven sheets has n, k or emissivity.
- Magnet-N42: density, remanence and coercivity only, and `is_material_like` rejects the sheet.
  Check that S5a's export still includes it.
- Steel-1018 and Aluminum-6061 exist in both `derived/` and `reference/`. The Rust lookup returns
  `derived/` (`data_loader.rs:67`, `:294`). The `reference/` copies hold fewer values. Not
  checked: whether Python's lookup picks the same one. One of the two copies should go.
- 1827 temperature-dependent values in 374 sheets state no reference temperature (a key-name
  heuristic, so an upper bound). Input to S2.
- 129 sheets have no `bulk.YoungsModulus`. Ingest drops no keys.
- Not done by R1b: it took the domains from this plan, not from PlayTow's `MATERIALS.md`.

## Owner decisions open
From `sheets-formula-review.md` (the U numbers are its rows):
1. Licence: the source headers say "All rights reserved", `LICENSE` says MIT, and there is no
   LEGAL.txt. Which one holds?
2. U2: mixed thermal and electrical conductivity always takes the series (lowest) bound. Copper
   with 10 % air gives 0.26 W/mK, where a dispersed mix is about 320. This needs a microstructure
   tag or a Maxwell-Eucken default.
3. U5, U6: strengths, toughness, fatigue and the hardening exponent mix by volume average, which
   holds only along the fibres of a composite. Keep them as tagged estimates, or refuse to mix?
4. U7: Paris C has no unit scale. Which units do the sheets quote?
5. U8: the colour-matching functions are a fit, not the CIE 1931 table. Fine for rendering; rename
   it, or load the table?
6. U9: concrete Young's modulus is the fib Model Code 2010 tangent value, 11 % above the secant
   value and not EN 1992. Which one does the statics solver expect?
7. U3 (second half): the J_IC and fracture-energy mirror is up to 2x off for concrete.
8. Done 2026-11-11 (the owner asked for one line, a second session being open). The main
    checkout `../periodica` follows `origin/main` again: its 5 unpushed commits were already on
    `origin/main` in content, so d5e2d01 joins the two histories with no content change, and the
    checkout was fast-forwarded. Its modified `tests/deviation_report.txt` was byte-identical to
    the pushed version (copy in `H:/claude-scratch/periodica-sheets/main-checkout-backup/`). The
    stash there is untouched. (Until decision 10 closed, each push from the lane checkout was
    followed by a fast-forward of this one.)
9. (PyPI state on 2026-11-11: newest release 2.0.2 of 2026-05-17; the repo is at 2.3.0 with no
   tag and no release.) The PyPI source distribution can't build without the sibling repos (R0b, checked locally):
    - `rust/periodica-mat/src/pairs.rs:24` includes `../../../src/periodica/data/interfaces/pairs.json`,
      a path that the sdist layout breaks;
    - `arithma_core` declares `license-file = "LICENSE"` but Arithma has no such file beside the
      crate (the workflow copies the root one as a workaround);
    - the release pins an untagged Arithma commit, because `Cargo.lock` needs `arithma_core` 2.0.4
      and the newest tag carries 2.0.2.

    Publishing `arithma_core` on crates.io would remove all three. Otherwise: fix the include path
    and tag Arithma, or release wheels only.
10. Done 2026-11-11. With item 8 settled and Q1a landed, the worktree `../periodica-sheets` and
   its branch `claude/sheets` were removed. Lanes work in the main checkout on `main`, which is
   the owner's rule (no feature branches, no worktrees). See "How lanes run".
11. Source and licence of the new data (S3b values, S3c emission lines). The values PlayTow asked
    for (latent heats, viscosity, sound speed, loss factors, spectral lines) aren't in any sheet
    today. The candidates are NIST (Atomic Spectra Database, Chemistry WebBook; Standard Reference
    Data, free to use, NIST asks for a citation and holds the database copyright) and handbooks
    (CRC, ASM; the numbers are facts, the tables are copyrighted). Suggested: NIST values with a
    citation per value in the sheet and a LEGAL.txt entry; no handbook table copied wholesale.
12. The Python builder and the native composer write different ion and isotope sheets. The Python
    `Get` runs the `refinements` of `composition_rules.json` (binding energy, mass defect, the
    cross-reference copy), so `python -m periodica build ions` writes 11 more keys for Ca2+ and 8
    more for H-2 than the committed sheets, which the native composer wrote. Which one is the
    reference: the Python build drops the refinements, or the native composer gains them?
    `scripts/_mass_source.py` calls the Python mass source an interim twin of `mass_source.rs`.
13. One property vocabulary. `property_schema.json` (68 properties, 96 unit tokens, Python) and
    `canon.rs` (99 property ids, Rust) are kept by hand and already differ. Suggested: the schema
    file is the single source and a test fails when `canon.rs` and the schema disagree; or
    `canon.rs` is generated from it.
14. The generated sheets. 126 sheets are marked `auto_generated` and 20 carry a `model_used`: the
    Python generators (`utils/biological_generator.py`, `biological_component_factory.py`,
    `biological_derivation_chain.py`, `derivation_metadata.py`) computed them from a rule of
    mixtures or a porosity model. They include alloys nobody makes (`Fe-Ag-Bi_Ternary`) and a
    liquid with a Young's modulus (`Blood`, 3.3 MPa); `Dermis` states a water content of 0. Since
    P2 they ingest as `Estimated`, which is the default and loses nothing. The other choices:
    keep them out of the export PlayTow reads, or delete them and rebuild the real ones from
    cited data (ties to decision 11). The owner's rule of 2026-11-11 (sheets hold real, cited
    quantities; `briefs/_common.md`) points at the last one.
15. The cut-off for "nu too close to 0.5". `NU_ILL_CONDITIONED = 0.45` in `derive.rs` tags K
    from E and nu as `Estimated` above it. The number has no reference. What it means: the
    relative error of K is `2 nu / (1 - 2 nu)` times the relative error of nu, which is 9 at
    0.45. The owner picks the accepted factor c and the cut follows as `c / (2 (1 + c))`:
    c = 4 gives 0.40, c = 9 gives 0.45, c = 19 gives 0.475. Lane Q1a keeps 0.45 and states this.
