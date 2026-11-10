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

## How lanes run
- The integration branch is `claude/sheets`, in the one checkout `../periodica-sheets`.
- No new worktrees: a TokenWatch hook bans `git worktree add` (worktrees took 108 GB). Lanes commit
  straight to `claude/sheets` in that checkout. One Rust lane runs at a time; a lane that touches
  no Rust source may run beside it.
- The planner pushes `claude/sheets:main` when the workspace tests pass, then checks GitHub
  Actions.
- At most 3 agents at once; one deliverable per brief (`briefs/_common.md`).

## Lanes
| Lane | Brief | Status |
|---|---|---|
| R1a formula review | `briefs/R1a-formula-review.md` | merged 30786c4: 36 OK, 6 BUG, 13 UNCLEAR in `sheets-formula-review.md` |
| M1 element bulk properties | `briefs/M1-merge-element-bulk.md` | merged 5a7a054. It also overwrote 28 ion sheets; F6 restores them |
| F3 mix (Turner expansion, Hill consistency, Lorentz-Lorenz n, `mix_tagged`) | `briefs/F3-mix.md` | merged bb3bd57 |
| R0 CI (Arithma clone, fmt, path-dependency guard) | `briefs/R0-ci-green.md` | merged 11b2a79. First run on GitHub (4191f64) failed in two places; see below |
| R0c reproduce the Python CI failure with the Rust extension | `briefs/R0c-python-ci-repro.md` | running (works on an export, changes nothing). It stopped once while its build compiled and was resumed |
| F5 ion keys, elastomer floor | `briefs/F5-corpus-green.md` | merged f9902fe; F6 revisits the ion keys |
| F4 optics (gold interband terms, validity ranges, Planck guard) | `briefs/F4-optics-models.md` | merged 7eedcac. The gold fit matches Johnson and Christy within 0.05 in R; its citation (Etchegoin 2006) is from memory and unchecked |
| F6 derived data green, MP-PER-1 | `briefs/F6-derived-green.md` | done dcfd0dc, 4191f64. f_ck is the property `CharacteristicCompressiveStrength` and triggers the fib relations; no code reads `Class`. 28 ion and 39 isotope sheets restored from 9890238 |
| R0b release workflow builds with Arithma | `briefs/R0b-release-workflow.md` | done 37020cb: wheels and sdist jobs fetch Arithma at commit 447a7c0. Not run on GitHub. The sdist can't build a wheel on its own (owner decision 9) |
| S5a column export of scalar properties | `briefs/S5a-column-export.md` | next Rust lane (PlayTow waits on it) |
| F1 derived optics (Fresnel with k, no emissivity mirror) | `briefs/F1-derive-optics.md` | done 957f83a, 0500d1f. Reflectance from n needs k (`Derived`) or a known transmittance (k = 0, `Estimated`); with k it also covers metals with n below 1. Absorptance and emissivity no longer copy each other. Open: R is `Estimated` even when the transmittance is itself an estimate |
| P1 pytest must not rewrite package data | `briefs/P1-tests-leave-data-alone.md` | done 1247dc3, 826cc50. Only `tests/test_cli.py` wrote to the package; it now builds into a private copy of `derived/` and its fixture checks that the patch ended |
| F2 derived mechanics (HV gate, orthotropy gate, plane-strain Irwin) | `briefs/F2-derive-mech.md` | after S5a |
| S3a new scalar properties (MP-PER-3): ten ids, sheet keys, sound-speed rules | `briefs/S3a-new-scalar-properties.md` | ready; after F2. No data values; the data and the emission lines are later lanes |
| R1b domain census | `briefs/R1b-domain-census.md` | after S3a |

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
- Python jobs, "Run tests": still open (R0c).

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
  S3b: the values, each with a cited source. S3c: emission lines from the NIST Atomic Spectra
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
8. The main checkout `../periodica` can't follow `origin/main`:
    - it has an uncommitted `tests/deviation_report.txt`, which the pushed work also changes;
    - its local `main` (a23ed05) holds 5 commits that were never pushed, and merging it with
      `origin/main` conflicts in `canon.rs` and about 20 element sheets. Their content is already
      on `origin/main`: M1 cherry-picked d78bb77, both lane branches are merged there, and
      `CLAUDE.md` is identical.

    Suggested: discard or commit the report file, then `git reset --hard origin/main` there. The
    planner won't do that without the owner's word.
9. The PyPI source distribution can't build without the sibling repos (R0b, checked locally):
    - `rust/periodica-mat/src/pairs.rs:24` includes `../../../src/periodica/data/interfaces/pairs.json`,
      a path that the sdist layout breaks;
    - `arithma_core` declares `license-file = "LICENSE"` but Arithma has no such file beside the
      crate (the workflow copies the root one as a workaround);
    - the release pins an untagged Arithma commit, because `Cargo.lock` needs `arithma_core` 2.0.4
      and the newest tag carries 2.0.2.

    Publishing `arithma_core` on crates.io would remove all three. Otherwise: fix the include path
    and tag Arithma, or release wheels only.
10. The hook asks for branches in the main checkout, not worktrees. `../periodica-sheets` is still
   a worktree because of item 8. Once item 8 is settled, the work moves to the main checkout and
   the worktree goes.
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
