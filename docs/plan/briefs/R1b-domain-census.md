# R1b: sheet coverage census by PlayTow domain

Read `_common.md` first. One deliverable: a rerunnable census example in `periodica-mat`, its
committed output, and one test. Read-only for everything else: no sheet, no rule and no id changes.

## Another Rust lane is running in this checkout
It edits `rust/periodica_core` and Python files. Keep out of its way:
- Run cargo only for this package: `cargo test -p periodica-mat --no-fail-fast`,
  `cargo run -p periodica-mat --example domain_census`, `cargo clippy -p periodica-mat --all-targets`.
  No `--workspace` commands.
- Format with `cargo fmt -p periodica-mat`, not the bare `cargo fmt` that `_common.md` names (the
  other lane's files may be half-edited). Check with `cargo fmt -p periodica-mat -- --check`.
- Cargo may print "Blocking waiting for file lock on build directory": the other lane is building.
  Wait; don't kill anything.
- `git status --short` shows the other lane's files as modified. Don't touch, stage or report them.

## Files
- new `rust/periodica-mat/examples/domain_census.rs`;
- new `rust/periodica-mat/tests/domain_census.rs` (the test below);
- new `docs/dev/domain-census.json` and `docs/dev/domain-census.md` (the example writes both);
- if the domain table must be shared by the example and the test, a new module
  `rust/periodica-mat/src/domains.rs` and its `pub mod` line in `lib.rs`. Nothing else in `src/`.

## What it measures
Ingest every bundled sheet through the real ingest path, as `tests/ingest_census.rs` and
`tests/corpus.rs` do (reuse their directory walk and `is_material_like`; read them first). For
each sheet report, per domain, the properties present and their `Source` mix
(Datasheet / Derived / Estimated / UserOverride):
mass; elastic; failure (strength, fracture, fatigue, hardness); contact (friction, restitution);
optical and spectral; thermal and phase; electrical; magnetic; acoustic.

Define the domain to `PropertyId` map once, in one table. Take the domains from PlayTow's docs
(read-only, another repo: `H:/Projects/PlayTowEcosystem/PlayTow/docs/MATERIALS.md` and
`docs/plan/REQUIREMENTS.md` there, the REQ-MAT, REQ-THERMAL, REQ-FIELDS and REQ-SPECTRAL items)
and from "What PlayTow asked for" in `docs/plan/playtow-sheets.md`. A domain with no `PropertyId`
today (acoustic) stays in the table with an empty list, so the gap shows.

It also lists:
- keys that ingest drops (`provenance.unknown_keys`), with the sheets that carry them;
- sheets that `census::volume_readiness` reports as not ready, with what they miss;
- records whose `orthotropy` is `None` but whose category or name says wood, timber, laminate or
  fibre composite (the isotropic rules run on them: see "Gaps in the sheets PlayTow loads" in the
  plan). A name test is fine here: this is a report, not library behaviour;
- temperature-dependent values (thermal and electrical conductivity, heat capacity, resistivity,
  moduli) in sheets that state no reference temperature. Count them; don't list every one.

## The markdown (`docs/dev/domain-census.md`, 150 lines or fewer)
1. Totals per data folder and category: sheets, and per domain how many have at least one
   `Datasheet` value and how many have only derived or estimated ones.
2. A table for the eleven sheets PlayTow loads, one row per sheet and one column per domain, each
   cell `datasheet/derived/estimated` counts: Steel-1018, Concrete-C30, Steel-B500B, Concrete-C40,
   Concrete-C25, Aluminum-6061, Magnet-N42, Timber-C24, Rubber-Natural, Mortar-CementLime,
   Brick-Clay. Where a name ingests from two folders (`derived/` and `reference/`), give both rows
   and say which one a lookup by name returns, if the code makes that decidable; otherwise say it
   is undecided.
3. The 20 worst gaps for PlayTow, most used sheets first.
The JSON holds the full per-sheet data, keys sorted, so a rerun gives a stable diff.

## Test (`tests/domain_census.rs`)
The domain table covers every `PropertyId` exactly once, or names each one it leaves out on
purpose in an explicit exclusion list. A new id then fails the test until someone places it.

## Verify before you report
`cargo test -p periodica-mat --no-fail-fast`; the two fmt and clippy commands above; run the
example twice and confirm the second run changes nothing (`git diff --stat` on the two outputs).

## Report (20 lines or fewer)
Commit hash; test counts; the table of PlayTow's eleven sheets (or its five worst cells); the
three findings you think matter most to PlayTow; what you did not verify.
