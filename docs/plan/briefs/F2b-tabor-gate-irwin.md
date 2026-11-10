# F2b: Tabor's rule for ductile metals only; plane-strain Irwin (lane `derive-strength`)

Read `_common.md` first. Runs after F2a (both edit `derive.rs`; F2a adds the `DeriveGates`
struct). One deliverable: two fixes in `rust/periodica-mat/src/derive.rs` and
`rust/periodica-mat/src/chains.rs`, with tests in a new file
`rust/periodica-mat/tests/f2b_tabor_irwin.rs`. Source: BUG 4 and U13 of
`docs/plan/sheets-formula-review.md`, plus a find of 2026-11-11.

## Known (read 2026-11-11)
Tabor's correlation holds for work-hardening metals. It runs in three places with no gate:
- `hardness` in `derive.rs`: UTS gives `HardnessVickers` (`UTS / 3.3`), and `HardnessVickers`
  gives UTS. Both `Estimated`.
- `hardness_pa` in `chains.rs`: with no measured hardness scale, `Hardness = 3 * YieldStrength`,
  tried before the `3 * f_c` branch.

Result in the sheets PlayTow loads (its contact model reads `Hardness`):
- Timber-C24, Rubber-Natural, Mortar-CementLime and Brick-Clay state a `YieldStrength`, so each
  gets `Hardness = 3 * yield`. Mortar and brick also have a `CompressiveStrength`.
- All ten sheets with a UTS get an estimated `HardnessVickers`, concrete included.
- Steel-1018 and Aluminum-6061 take `Hardness` from a measured Brinell number; Steel-B500B from
  `3 * yield` (it has `ElongationPct`). These three are right and must not change.

## Newer than the text above (2026-11-11, after F2a landed as 8ed7d85)
- `derive.rs` now has `DeriveGates { isotropic_elastic }`, `fill_derived_with(t, gates)` and
  `gates_for(record)`; `ingest.rs` calls the gated form. Read them before you design the predicate.
- Corpus tests load a sheet by its path under `src/periodica/data`; copy `sheet(rel)` from
  `tests/f2a_elastic_gates.rs`. Never find a sheet by file name with a directory walk:
  `reference/alloys/` holds thinner copies of Steel-1018 and Aluminum-6061, and `read_dir` order
  differs on Linux. That made CI red on ff3ae8c. The paths:
  `derived/composites/Timber-C24.json`, `derived/polymers/Rubber-Natural.json`,
  `derived/ceramics/Mortar-CementLime.json`, `derived/ceramics/Brick-Clay.json`,
  `derived/alloys/Steel-1018.json`, `derived/alloys/Aluminum-6061.json`,
  `derived/alloys/Steel-B500B.json`.
- "Unchanged" for the three steel and aluminium sheets means the same `Hardness`,
  `HardnessVickers` and `TensileStrength` values and source tags as before your change. Record
  them before you edit and assert them in the test.

## Changes
1. One predicate, "this table shows ductile-metal behaviour", used by all three Tabor sites. Build
   it from data the table already carries (read `property.rs` and a few alloy sheets; candidates:
   `ElongationPct`, a metallic resistivity, a measured hardness scale). Say which fields you used
   and why. Don't add a material-class enum. If the table alone can't decide, add a field to
   F2a's `DeriveGates` and say why.
   Tests: a ceramic-like table (measured HV 1500, no ductility data) gets no UTS; a steel table
   still gets both directions. On the real corpus: Timber-C24 and Rubber-Natural have no
   `Hardness` and no `HardnessVickers`; Mortar-CementLime and Brick-Clay have
   `Hardness = 3 * CompressiveStrength`; Steel-1018, Aluminum-6061 and Steel-B500B are unchanged.
2. Irwin (`fracture`): K_IC is a plane-strain value, so use `G = K^2 (1 - nu^2) / E` when
   Poisson's ratio is present; keep `K^2 / E`, tagged `Estimated`, when it isn't. Leave the J_IC
   and fracture-energy mirror as it is; that is an owner decision.

## Verify
Use `-p periodica-mat` for every cargo command. `cargo fmt -p periodica-mat`;
`cargo test -p periodica-mat --no-fail-fast`;
`cargo clippy -p periodica-mat --all-targets -- -D warnings`. Then rerun
`cargo run -p periodica-mat --example domain_census` and commit the two regenerated files under
`docs/dev/`.

## Report (15 lines or fewer)
Commit hash; test counts; the predicate's fields; how many corpus sheets lost `Hardness`,
`HardnessVickers` or an estimated UTS, by category; every oracle that moved, with old and new
numbers; what you did not verify.
