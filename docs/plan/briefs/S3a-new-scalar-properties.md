# S3a: new scalar properties for PlayTow (MP-PER-3), ids, sheet keys, sound-speed rules

Read `_common.md` first. One deliverable: the ten property ids below, their sheet keys, the two
sound-speed derivations, their docs and tests. No values are added to any datasheet in this lane
(the data comes in a later lane, from cited sources). Emission lines are a later lane too.

## Context
PlayTow (the engine that reads periodica) asked for these on 2026-11-11. Its answers fix the
definitions, so don't change them:
- Sound speed in a solid is two properties. Longitudinal is the bulk wave in an unbounded medium,
  `sqrt((K + 4G/3) / rho)`. It is not the thin-rod speed `sqrt(E / rho)`. Shear is `sqrt(G / rho)`.
- A fluid has one sound speed, stored as the longitudinal one. Its shear speed is absent, not zero.
- Acoustic loss in a solid is the loss factor `eta = tan delta`, dimensionless. In a fluid it is the
  bulk attenuation divided by frequency squared, `alpha / f^2`, in Np s^2/m, at a reference state
  where the classical f^2 law holds.
- Viscosity is dynamic, in Pa s, a scalar with its reference temperature stored beside it. It is
  absent for solids.

## The ids
Append to the END of the table in `rust/periodica-mat/src/property.rs`, after
`CharacteristicCompressiveStrength`, in this order (stored indices must not move):

| Id | SI unit string | Display name |
|---|---|---|
| `LatentHeatFusion` | `J/kg` | Latent Heat of Fusion |
| `LatentHeatVaporization` | `J/kg` | Latent Heat of Vaporization |
| `LatentHeatSublimation` | `J/kg` | Latent Heat of Sublimation |
| `RelativePermeability` | `1` | Relative Magnetic Permeability mu_r |
| `SoundSpeedLongitudinal` | `m/s` | Longitudinal Sound Speed (unbounded medium) |
| `SoundSpeedShear` | `m/s` | Shear Sound Speed |
| `MechanicalLossFactor` | `1` | Mechanical Loss Factor eta |
| `AcousticAttenuationOverF2` | `Np s^2/m` | Acoustic Attenuation alpha/f^2 |
| `DynamicViscosity` | `Pa s` | Dynamic Viscosity |
| `ViscosityReferenceTemperature` | `K` | Viscosity Reference Temperature |

Add a section to the module doc of `property.rs` that defines each one in a sentence. It must say:
longitudinal is the unbounded-medium speed and not the thin-rod speed; `RelativePermeability` is
magnetic and unrelated to `Permeability` (m^2, flow through a porous solid);
`MechanicalLossFactor` is mechanical tan delta, not the dielectric loss tangent.

Then grep the repo for `CharacteristicCompressiveStrength` and `DebyeTemperature` (the last ids
added) and add the new ids wherever a list of ids is mirrored (docs, schema, stubs, tests that
count properties).

## Sheet keys (`rust/periodica-mat/src/canon.rs`)
Read how `lookup` normalises a key, then map these sheet keys (the unit suffix is part of the key):
- `LatentHeatFusion_J_kg`, `LatentHeatFusion_kJ_kg` (x 1e3); the same two for `Vaporization`
  (accept the spelling `Vaporisation` too) and `Sublimation`.
- `RelativePermeability` (no suffix).
- `SoundSpeedLongitudinal_m_s`, `SoundSpeedShear_m_s`, and `SoundSpeed_m_s` (a fluid's single
  value; it maps to `SoundSpeedLongitudinal`).
- `MechanicalLossFactor` (no suffix).
- `AcousticAttenuationOverF2_Np_s2_m`.
- `DynamicViscosity_Pa_s`, `DynamicViscosity_mPa_s` (x 1e-3), `DynamicViscosity_cP` (x 1e-3).
- `ViscosityReferenceTemperature_K`.

Don't map molar latent heats (kJ/mol): the conversion needs the molar mass, which a scale can't
give. Don't map `LossTangent` or `SpeedOfSound`: both are ambiguous.

## Derivations (`rust/periodica-mat/src/derive.rs`)
One new rule function `acoustic`, called from `one_pass` after `elastic`, 60 lines or fewer:
- `SoundSpeedShear = sqrt(G / rho)` when `ShearModulus` and `Density` are present and positive.
  Source: `inherit(t, &[ShearModulus, Density])`.
- `SoundSpeedLongitudinal = sqrt((K + 4G/3) / rho)` when `BulkModulus`, `ShearModulus` and
  `Density` are present and positive. Source: `inherit` over the three.
- Fluid form: `SoundSpeedLongitudinal = sqrt(K / rho)` when `BulkModulus` and `Density` are
  present, `ShearModulus` is absent and `DynamicViscosity` is present (viscosity is what marks a
  fluid; no class or name check). Source: `Estimated`, because the sheet's bulk modulus may be the
  isothermal one and the wave needs the adiabatic one. Say so in the doc comment.
- With `K` and `rho` only (no `G`, no viscosity), derive nothing.
- Use `set_if_better`, so a measured speed is never overwritten. Derive nothing in the other
  direction (no modulus from a speed), and nothing for the loss factor, attenuation, viscosity,
  latent heats or permeability.

Another lane (F2) changes `elastic`, `fracture` and `hardness` in the same file. If its commit is
already there, leave its code alone.

## Tests: `rust/periodica-mat/tests/s3a_new_properties.rs`
- The ten ids follow `CharacteristicCompressiveStrength` in the order above (compare indices), and
  each has the unit string above. `PropertyId::from_name` round-trips them.
- Each sheet key maps to its id with the right scale (kJ/kg, mPa s and cP included).
  `LossTangent`, `SpeedOfSound` and a kJ/mol key map to nothing.
- Steel, E = 200 GPa, nu = 0.29, rho = 7850 kg/m^3, after `fill_derived`: longitudinal is about
  5778 m/s and shear about 3142 m/s (0.5 % tolerance; compute the exact values in the test from
  the formulas). Longitudinal differs from `sqrt(E / rho)` (about 5048 m/s) by more than 10 %.
  Both are `Derived` when E, nu and rho are `Datasheet`, and `Estimated` when one input is.
- Water, K = 2.2 GPa, rho = 998, viscosity 1.0e-3 Pa s at 293.15 K: longitudinal is about
  1485 m/s and `Estimated`; shear is `None`.
- K and rho alone: no sound speed.
- A `Datasheet` longitudinal speed of 5900 survives `fill_derived` unchanged.
- An in-memory sheet with `DynamicViscosity_mPa_s` and `ViscosityReferenceTemperature_K` ingests
  to SI values (use whatever in-memory ingest the existing tests use).
- A corpus check (add to the corpus test file, or here): every record with `DynamicViscosity` has
  `ViscosityReferenceTemperature`. No sheet has either today, so it passes; it guards the data lane.

## Report (20 lines or fewer)
The commit hash; test counts; every file touched outside the three source files and the test; any
existing test whose expected numbers moved, with the reason; anything in the definitions above that
you found wrong or ambiguous (say so; don't change a definition on your own).
