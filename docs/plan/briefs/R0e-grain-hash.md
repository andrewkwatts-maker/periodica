# R0e: the Rust sampler picks grains with the same hash as Python (1 failing CI test)

Read `_common.md` first. Runs after lane R0d and reuses its virtual environment
(`H:/claude-scratch/periodica-sheets/venv`; setup and rebuild commands are in
`R0d-rust-sheet-parity.md`, section Setup). One deliverable: the test below passes with the Rust
extension built.

## Known
- `tests/test_engine_sampling.py::TestStrategyEquivalence::test_vectorised_and_legacy_scalar_sampler_agree`
  fails when `periodica._HAS_RUST` is True: 31 of 216 points differ (Stainless-304; the vectorised
  sampler gives 195 where the scalar `sample()` gives 210).
- Python picks the grain with `src/periodica/_hash.py`: the grain cell is `rint(point / size)`
  as int64 per axis, and the unit value is SplitMix64's finalising mix over the packed cell
  (`grain_cells`, `unit_from_cells`, `unit_at_grains`). Its module docstring says why SHA-1 went.
- Rust still hashes a string key with SHA-1: `sha1_unit_interval` in
  `rust/periodica_core/src/sample_models.rs:114`, used at `sample_models.rs:149`, `:358` and
  `sample.rs:375`. R0c read this from the code; it did not run a patched build, so confirm it.

## Do
1. Port `grain_cells` and `unit_from_cells` to Rust, bit for bit: the same constants, wrapping
   64-bit arithmetic, numpy's `rint` (round half to even; `f64::round_ties_even` if the
   workspace `rust-version` allows it, else write it), the same integer-to-unit conversion, and
   the same handling of a missing axis and of a grain size near zero.
2. Replace the three SHA-1 call sites. Read each one's Python twin first (`sample.py`
   `_maybe_phase_at`, `_microstructure_voronoi`, and whatever `sample.rs:375` mirrors) and match
   what Python hashes there: the cell of the point, or a neighbour cell.
3. Remove `sha1_unit_interval` and its tests when nothing uses it. Remove the `sha1` dependency
   from `rust/periodica_core/Cargo.toml` only if no other code in the crate uses it; leave the
   workspace entry.
4. Grep `rust/` for other users of the SHA-1 grain hash (`periodica-runtime`, `periodica-mat`).
   Don't change them; list them in the report.

## Tests
- A Rust unit test pins at least 8 vectors: cells (including negative ones and zero) and their
  unit values, computed with `$V -c "from periodica import _hash; ..."`. Paste the Python command
  in a comment above the table.
- Rust: every point inside one cell gets the same value; a half-way coordinate rounds to even.
- The failing Python test passes. Add nothing to it.

## Verify before you report
As in R0d (cargo tests with and without `--features python`, fmt, all-features check, rebuild the
extension, the full pytest run as CI does, log to
`H:/claude-scratch/periodica-sheets/pytest-r0e.log`). Expected: no failures. `git status --short`
shows only your files.

## Report (15 lines or fewer)
Commit hash; pytest counts; the other SHA-1 users found; whether sampled values changed for any
existing Rust test, and which; what you did not verify.
