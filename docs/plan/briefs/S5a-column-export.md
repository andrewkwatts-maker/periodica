# S5a: column export of scalar properties (lane `export`)

Read `_common.md` first. This is the first part of PlayTow's request MP-PER-4
(`docs/plan/playtow-sheets.md`). Deliverable, on `claude/sheets`:
- a new module `rust/periodica-mat/src/export.rs`, with `pub mod export;` in `lib.rs`;
- tests in a new file `rust/periodica-mat/tests/s5_export.rs`;
- an example `rust/periodica-mat/examples/export_columns.rs` that exports every sheet of the corpus
  to a directory given on the command line.

Edit no other file. periodica-mat must not gain a dependency; `serde_json` is already there.

## Format (agreed with PlayTow; they wrap it in their own container)
One directory per export:
- `<PropertyName>.f64`: one little-endian f64 per row. A missing value is NaN.
- `<PropertyName>.src`: one byte per row, the `Source` of that value. 0 means no data, so this
  file is the missing-value mask and the provenance column in one.
- `manifest.json`: `format_version` 1, `endianness` "little", `rows`, `sheet_ids` in row order,
  the legend of the source codes, and per property: name, numeric id, SI unit string (from
  `PropertyId::si_unit`), the two file names, and the count of present values.

Rules:
- One row per sheet. Rows are sorted by sheet id, so two exports of the same corpus are
  byte-identical. Take the sheet id from the way `tests/corpus.rs` and `census.rs` already name
  sheets; say in your report which form you used.
- The caller gives the property list. The example's default is every property that has at least
  one value.
- Only `record.bulk` is exported here. Curves, uncertainty columns, orthotropy and phases are
  later parts; `format_version` lets the manifest grow.
- A property's unit is whatever `si_unit` declares, including the few non-SI ones (hardness scales,
  `ElongationPct`). Don't convert anything.

## API (keep it this small)
- `columns(records, props) -> ColumnSet`: pure, no I/O.
- `write_dir(&ColumnSet, dir) -> Result<Manifest>` and `read_dir(dir) -> Result<ColumnSet>`.
- Errors are a typed enum; fail early on a row-count mismatch, a short file or an unknown property
  name in a manifest.

## Tests
- Round trip: write, read back, every value bit-equal and every source equal.
- A missing value is NaN with source 0.
- The row order is sorted and the same on a second run.
- Each `.f64` file is `rows * 8` bytes and each `.src` file is `rows` bytes.
- From an export of the real corpus: the iron element sheet's Density column value is 7874 at the
  row that the manifest gives for it.

Run `cargo test -p periodica-mat --no-fail-fast`, clippy with `-D warnings`, and the example once
into `H:/claude-scratch/periodica-sheets/export/out`. Report the row count, the property count and
the total size of that export.
