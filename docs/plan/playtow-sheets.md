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
| R0 CI (Arithma clone, fmt, path-dependency guard) | `briefs/R0-ci-green.md` | merged 11b2a79; not yet run on a GitHub runner |
| F5 ion keys, elastomer floor | `briefs/F5-corpus-green.md` | merged f9902fe; F6 revisits the ion keys |
| F4 optics (gold interband terms, validity ranges, Planck guard) | `briefs/F4-optics-models.md` | merged 7eedcac. The gold fit matches Johnson and Christy within 0.05 in R; its citation (Etchegoin 2006) is from memory and unchecked |
| F6 derived data green, MP-PER-1 | `briefs/F6-derived-green.md` | running |
| R0b release workflow builds with Arithma | `briefs/R0b-release-workflow.md` | done 37020cb: wheels and sdist jobs fetch Arithma at commit 447a7c0. Not run on GitHub. The sdist can't build a wheel on its own (owner decision 9) |
| S5a column export of scalar properties | `briefs/S5a-column-export.md` | ready; after F2 |
| F1 derived optics (Fresnel with k, no emissivity mirror) | `briefs/F1-derive-optics.md` | after F6 |
| F2 derived mechanics (HV gate, orthotropy gate, plane-strain Irwin) | `briefs/F2-derive-mech.md` | after F1 |
| R1b domain census | `briefs/R1b-domain-census.md` | after F2 |

State at 7eedcac: 423 Rust tests pass and one fails (`derived_regeneration`, which F6 fixes);
clippy and `cargo check --workspace --all-features` pass.

## Schema lanes (after the census; one brief each)
- S1 (MP-PER-4): one standard uncertainty per value, plus min and max where given. Map `Source` to
  PlayTow's flags: measured, derived, authored, no data. Open point: sheets under `data/derived/`
  are generated by code but ingest as `Source::Datasheet`.
- S2 (MP-PER-2): temperature and frequency dependence as Arithma expressions with a validity range
  and source points. The optics models already return an error outside their range (F4).
- S3 (MP-PER-3): the new property ids, sheet keys and formulas (for example sound speed from K and
  density), then the data, from sources with permissive licences.
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
8. The main checkout `../periodica` has an uncommitted `tests/deviation_report.txt`, which the
   pushed work also changes, so it can't fast-forward until that file is committed or discarded.
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
