# R1a: periodica-mat formula review (lane `review`)

Read `_common.md` first. Deliverable: `docs/plan/sheets-formula-review.md`, committed on
`claude/review`. This is a read-only review: change no code, because the fixes become separate
briefs. Scratch scripts that check numbers go under your scratch directory.

## Scope (`rust/periodica-mat/src/`)
Review every formula that is applied to sheet values:
- `mix.rs`: the mixing rules. PlayTow's `docs/MATERIALS.md` expects density by volume fraction,
  n by Lorentz-Lorenz, and stiffness by Voigt/Reuss.
- `derive.rs`: the exact relations (E, G, K, ν; diffusivity; and so on) and when each fires.
- `optics/*`: Sellmeier (C in m² vs µm²), Cauchy, Drude-Lorentz, Planck, the CIE xyz tables,
  emission lines, and each model's validity range.
- `canon.rs`: every unit-conversion factor.
- `constants.rs`: the CODATA values and edition.
- `orthotropy.rs` (off-axis modulus, Tsai-Hill) and `concrete.rs` (the EN 1992 relations).
- `property.rs`: check that each declared SI unit is dimensionally right for the quantity.

## Output format
For each formula, give one table row:
- file:line;
- the formula as implemented;
- the reference formula and its source;
- the verdict, OK / BUG / UNCLEAR;
- for a BUG, a concrete input with the wrong and the right output, and the test that would catch
  it.

Rank the BUGs first. End with a short list of the formulas PlayTow will need that are missing,
for example temperature-dependent ones. Don't redesign the schema.
