# F6: derived data matches the native composer; no class branch (lane `derived`)

Read `_common.md` first. Deliverable: commits on `claude/sheets` after which
`cargo test --workspace --no-fail-fast` has no failures. At 7eedcac one test fails:
`-p periodica_core --test derived_regeneration`, with 729 differences. Part 2 is also PlayTow's
request MP-PER-1.

Files you may edit: `src/periodica/data/derived/ions/*`, the four concrete sheets and the registry
row named below, and in `rust/periodica-mat`: `src/property.rs`, `src/canon.rs`, `src/concrete.rs`,
`src/ingest.rs`, `src/record.rs` (doc comment only), `tests/concrete_grades.rs`,
`tests/corpus.rs`.

## Part 1: the 28 ion sheets (728 of the differences)
The cause is known. Merge 5a7a054 (lane M1) took old generated copies of
`src/periodica/data/derived/ions/*.json`. Each gained 26 keys (`AtomicNumber_Z`, `Properties`,
`_provenance`, ...) that the native composer doesn't write. At 9890238 the sheets match it.

1. Run `git checkout 9890238 -- src/periodica/data/derived/ions`, then the test. Expect one
   difference, `composites/Concrete-Mix.Class`. If others remain, list them in your report and
   carry on with Part 2.
2. Lane F5 (48c2f8f) mapped keys for those copies: `debyetemperaturek`,
   `primaryemissionwavelength` and `visibleemissionwavelength` in `canon.rs`, three `PropertyId`s
   at the end of `property.rs`, and the test `ion_sheet_ingests_debye_and_emission_keys` in
   `tests/corpus.rs`. After step 1 that test fails. Find out whether any other sheet reaches those
   mappings through ingest. 118 element sheets and 39 isotope sheets carry the same keys.
   - If some do: point the test at one of them. Say where the wavelength unit is stated; F5 assumed
     nm.
   - If none do: remove the three mappings, the three `PropertyId`s and the test. Nothing has
     shipped, so the ids aren't fixed yet. Keep F5's elastomer modulus floor.
3. Report whether the emission wavelengths in the element sheets are measured or computed
   (`src/periodica/utils/physics_calculator.py` computes some). Don't change them.

## Part 2: MP-PER-1, remove the class branch (the last difference)
`ingest.rs:197` runs `fill_concrete` only when the sheet says `"Class": "Concrete"`. A class name
must not select code. Also, `Class` on `derived/composites/Concrete-Mix.json` was added by hand in
9813976, and the composer can't reproduce it.

The design is decided; don't redesign it. The fib MC2010 relations take the characteristic
cylinder strength f_ck, so f_ck becomes its own property and its presence is the trigger.
- Append `CharacteristicCompressiveStrength` (Pa) at the END of `property.rs`. Map the sheet key
  `CharacteristicCompressiveStrength_MPa` in `canon.rs`.
- `fill_concrete` reads that property, not `CompressiveStrength`. It also fills
  `CompressiveStrength` = f_ck as `Estimated`; a datasheet value still wins. It fills nothing when
  f_ck is absent.
- `ingest.rs` calls `fill_concrete` for every record, with no class test.
- Add `"CharacteristicCompressiveStrength_MPa"`, equal to the existing `CompressiveStrength_MPa`,
  beside it in every sheet whose `Class` is `"Concrete"` today. Every record that got fib fills
  before must get the same values after.
- Concrete-Mix is composed from a registry row. Put the new key in the row's `properties`, so the
  composer writes it; `rust/periodica_core/tests/derived_regeneration.rs` shows where the rows come
  from and that row `properties` become sheet `Properties`. Remove `Class` from the committed
  Concrete-Mix sheet.
- Leave `Class` on the other sheets and `MaterialRecord.class` as a descriptive label. Change the
  doc comments in `record.rs` and `concrete.rs` to say that no code branches on it.

Tests, in `tests/concrete_grades.rs`:
- new: a record with `"Class": "Concrete"` and `CompressiveStrength_MPa` but no f_ck gets no fib
  fills; a record with f_ck and no `Class` gets them;
- `mix_datasheet_values_win`: drop the class assert, keep Young's modulus 30e9 from the datasheet,
  and assert that an fib fill is present (for example `MeanCompressiveStrength`, `Estimated`);
- the grade tests keep their numbers. Say so if a value moved.

## Acceptance
- `cargo test --workspace --no-fail-fast`: 0 failed.
- `cargo fmt --all -- --check` and
  `cargo clippy -p periodica-mat -p periodica-runtime --all-targets -- -D warnings` pass.
- pytest, run the way `.github/workflows/ci.yml` runs it: no new failures (R0 saw 1608 passed).
- Your report lists `git diff --name-only 7eedcac..HEAD`.
