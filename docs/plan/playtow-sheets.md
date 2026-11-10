# Plan: periodica sheets and formulas for the PlayTow engine (2026-11-11)

Owner request: "review this project for bugs/improvements so it's offering clear, comprehensive and
usable data sheets/formulas to PlayTow engine needs; discuss with the PlayTow session what it would
like from you."

## What PlayTow asked for (playtowecosystem-16, 2026-11-11)
periodica owns the periodica API. The PlayTow session sends requests and doesn't edit periodica.
Requirements: `PlayTow/docs/plan/REQUIREMENTS.md`, which defines REQ-MAT, REQ-UNITS, REQ-THERMAL,
REQ-FIELDS and REQ-SPECTRAL. Per-property needs with file:line will follow in
`PlayTow/docs/plan/modules/MATTER.md`.

The asks, in priority order:
- (a) One generic sheet schema for real and imagined materials. Every value carries its SI
  dimension, its uncertainty, and a provenance flag: measured, derived, authored, or no data.
- (b) Temperature and frequency dependence, as Arithma expressions or sampled curves with a stated
  range. Never a constant with a hidden reference temperature.
- (c) No material classes or names that the engine must branch on. Behaviour, and any A/B
  interface, derive from the two sheets.
- (d) Spectra over wavelength from IR to UV: n, k, emissivity and emission lines.
- (e) Phase data: melt, boil, sublime, the solid phases, and the latent heats.
- (f) Electrical, magnetic and acoustic columns: conductivity, permittivity, permeability or a B-H
  curve, sound speed, loss factor.
- (g) A plain column export per build (sheet id to SoA columns), which Hyle packs for the GPU.
  periodica must not depend on Themelios or Topos.

## First read of periodica-mat at origin/main 9890238 (planner, to be checked by R1a/R1b)
| Ask | Present | Missing or wrong |
|---|---|---|
| a | `PropertyId::si_unit` string per property (`property.rs`); `Source` enum Absent/Datasheet/Derived/Estimated/UserOverride (`table.rs:45`) | No uncertainty. The dimension is a string, not an Arithma dimension. `ParisC` is declared unit "1", but Paris C has dimensions. The hardness scales and `ElongationPct` are non-SI by design. |
| b | Optics dispersion models (Sellmeier, Cauchy, Drude-Lorentz) | Every other value is a scalar, with no temperature range or reference temperature. |
| c | One `MaterialRecord`, plus descriptive `Tier` and `CrystalSystem` | Check whether any consumer has to branch on them. |
| d | `optics/` spectrum, lines, planck, xyz | The stated validity range per model is unknown. |
| e | Melting, Boiling, Solidus, Liquidus, GlassTransition | Latent heats, sublimation, solid-phase transitions. |
| f | Resistivity, conductivity, permittivity, dielectric strength, Remanence, Coercivity | Magnetic permeability, saturation, B-H, sound speed, acoustic loss. `Permeability` is the hydraulic one (m²): name it apart. |
| g | Nothing | A column export. |

Also:
- The source headers say "All rights reserved ... strictly prohibited", but `LICENSE` is MIT.
- There is no LEGAL.txt.

Both are owner decisions.

## Lanes (one deliverable each; at most 3 agents at once)
| Lane | Brief | Branch / worktree | Status |
|---|---|---|---|
| R0 CI green | `briefs/R0-ci-green.md` | claude/ci, periodica-ci | running |
| M1 merge the element bulk properties | `briefs/M1-merge-element-bulk.md` | claude/merge, periodica-merge | running (d78bb77 only; 33a84fb nets to nothing) |
| R1a formula review | `briefs/R1a-formula-review.md` | merged 30786c4, branch closed | done: 36 OK, 6 BUG, 13 UNCLEAR in `sheets-formula-review.md` |
| F3 mix (Turner expansion, Hill consistency, Lorentz-Lorenz n) | `briefs/F3-mix.md` | claude/mix, periodica-mix | running |
| F1 derive optics (Fresnel with k, no emissivity mirror) | `briefs/F1-derive-optics.md` | claude/derive-optics | ready |
| F2 derive mechanics (HV gate, orthotropy gate, plane-strain Irwin) | `briefs/F2-derive-mech.md` | claude/derive-mech | after F1 |
| F4 optics models (gold interband, validity ranges) | `briefs/F4-optics-models.md` | claude/optics | ready |
| R1b domain census | `briefs/R1b-domain-census.md` | claude/census, periodica-census | after M1 |

The integration branch is `claude/sheets`, in worktree `../periodica-sheets`, from origin/main. The
planner merges the lanes there, then pushes `claude/sheets:main`.

## Later, after R1a, R1b and MATTER.md (schema design; one brief each)
- S1: uncertainty per value, and the provenance mapping to measured/derived/authored/no data.
- S2: temperature and frequency curves with a stated range.
- S3: phase data and latent heats.
- S4: the electrical, magnetic and acoustic columns.
- S5: the column export, sheet id to SoA columns.

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
