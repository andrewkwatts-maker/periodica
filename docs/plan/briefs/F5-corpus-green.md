# F5: periodica-mat corpus tests green (lane `corpus`)

Read `_common.md` first. Deliverable: `cargo test -p periodica-mat` passes on `claude/corpus`
(branched from `claude/sheets` at 5a7a054 or later), in two commits.

## Known
Two tests fail on `claude/sheets` 5a7a054:
1. `corpus::every_datasheet_ingests_without_unknown_keys`: 5 unmapped keys in
   `src/periodica/data/derived/ions/` (28 sheets): `debyetemperaturek`, `group`, `period`,
   `primaryemissionwavelength`, `visibleemissionwavelength`. The M1 merge (797b938, the cherry-pick
   of d78bb77) added these keys to the ion sheets.
2. `i3c_composition::plausible_ranges`: `derived/polymers/Rubber-Natural.json` has E = 2.1 MPa,
   below the test's 0.01 GPa floor. The sheet value is right for natural rubber (1 to 5 MPa).

## Steps
1. Run both tests first and confirm the failures and their messages.
2. Ion keys. The element sheets carry the same kind of keys and pass, so find how `canon.rs` and
   `ingest.rs` handle them for elements and do the same for ions. Rules:
   - `group` and `period` are descriptive; handle them the way the element sheets' are handled.
   - Debye temperature is a real property (K). If no `PropertyId` exists for it, append
     `DebyeTemperature => ("K", "Debye Temperature")` at the END of the list in `property.rs`, so
     stored indices stay stable.
   - The emission wavelengths: look at the values and units in the sheets (scalar or list, nm or
     m). If they're scalars, append `PrimaryEmissionWavelength` and `VisibleEmissionWavelength`
     in metres, converted at ingest. If they're lists or the unit is unclear, don't force them
     into a scalar: stop and report what the sheets hold.

   Never drop a key silently and never invent a unit. Add a test that one ion sheet (name it)
   ingests with no unknown keys and the Debye temperature in K with `Source::Datasheet`.
3. Rubber floor. Lower the floor in `i3c_composition::plausible_ranges` so elastomers pass, with a
   comment citing the range for natural rubber, and keep an upper and lower bound that still
   catches unit errors (a Young's modulus in Pa read as GPa must still fail).
4. Run the full `cargo test -p periodica-mat --no-fail-fast` and report the totals.
