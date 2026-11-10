# R0d: the Rust core reads a datasheet the way Python does (6 failing CI tests)

Read `_common.md` first. One deliverable: with the Rust extension built, the six tests below pass
and nothing else breaks. This is a Rust lane (`rust/periodica_core`) plus two Python files.

## Known (lane R0c reproduced it; log `H:/claude-scratch/periodica-sheets/pyrepro/pytest.log`)
GitHub Actions builds the Rust extension, so `periodica._HAS_RUST` is True there. Local runs never
had it. With it, 7 tests fail; this lane fixes 6 (the seventh is lane R0e, the grain hash):
- `tests/test_properties.py::TestBundledData::test_curated_material_sheet_is_not_empty`
- `tests/test_properties.py::TestBundledData::test_sampling_a_curated_material_works`
- `tests/test_properties.py::TestBundledData::test_sampling_accepts_any_known_spelling`
- `tests/test_rust_python_parity.py::test_data_sheet_matches_between_backends`
- `tests/test_rust_python_parity.py::test_name_resolution_agrees_between_backends`
- `tests/test_rust_python_parity.py::test_sample_matches_between_backends_across_properties`

Cause: commit 02b6807 gave Python a property layer (`src/periodica/properties.py`, vocabulary in
`src/periodica/data/config/property_schema.json`). `sample.py:_props_of` now calls
`properties.sheet`, which flattens nested groups (`PhysicalProperties`, `ElasticProperties`, ...)
and answers other spellings and units of a key. The Rust twin was not updated:
`rust/periodica_core/src/sample.rs` `props_of` (about line 125) reads only a top-level
`Properties` object or root scalars, so `py_data_sheet` returns `{}` for every curated sheet and
`py_sample` returns `None`. It also keeps booleans, which Python now drops (`properties.py:309`).

## The design (decided; don't redesign)
1. `props_of` becomes a port of `properties.flatten` (`properties.py:313`), same order and same
   precedence: root scalars; then the `Properties` / `properties` block; then the schema's
   `property_group_keys` in schema order; then any other shallow dict. Depth limit
   `MAX_GROUP_DEPTH` (3). Skip keys in the schema's `excluded_group_keys` and keys that start with
   `_`. Keep numbers and strings, never booleans. The first key seen wins.
2. The two lists come from `property_schema.json`, read once (the crate already has `once_cell`).
   Don't copy the lists into Rust. Find how the crate locates its data directory (the registry
   loader does it) and how Python strips comments from that file (`_strip_comments`,
   `properties.py:70`); do the same.
3. Rust does not get the alias and unit vocabulary in this lane. When `py_sample` is asked for a
   property that is not a literal key of the flattened sheet, the Python-facing function (in
   `rust/periodica_core/src/pyfacade/`) raises `NotImplementedError`, which `sample.py` already
   treats as "Rust declines, Python answers". The core function (no Python) keeps returning `None`.
   A literal key must never decline.
4. `periodica.data_sheet(name)` must return a `DataSheet` whichever backend answered. If the Rust
   path returns a plain dict, wrap it (`sample.py:328`, `call_rust`).
5. `tests/test_rust_python_parity.py` `_resolution_divergences` (about line 282) compares Rust keys
   with the raw `Properties` block. Its Python side must use `periodica.properties.flatten`. In
   the sample-parity test, a decline (rule 3) is not a mismatch; assert that no literal key
   declines.
6. Replace the comment in `props_of` that explains why booleans are kept; it describes the old
   Python behaviour.

If a rule above can't be met as written, stop and report; don't pick another design.

## Setup (the extension must be built to see the failures)
Environment for every command: `TEMP=TMP=H:/claude-scratch/tmp`,
`CARGO_TARGET_DIR=H:/claude-scratch/periodica-sheets/target`,
`PIP_CACHE_DIR=H:/claude-scratch/pip-cache`.
- Virtual environment (never install into a global Python):
  `"C:/Users/Andrew/AppData/Local/Programs/Python/Python313/python.exe" -m venv H:/claude-scratch/periodica-sheets/venv`
  and use `V=H:/claude-scratch/periodica-sheets/venv/Scripts/python` for everything Python.
- In the checkout: `$V -m pip install -e ".[test]"`, in the background (several minutes). It
  writes `src/periodica/_periodica_core.*.pyd`, which git ignores. Check
  `$V -c "import periodica; print(periodica._HAS_RUST, periodica.__file__)"` prints True and a path
  in this checkout.
- After every Rust change, rebuild with `$V -m pip install -e . --no-deps` before running pytest.
- The binding unit tests need a loadable Python DLL:
  `PYO3_PYTHON=C:/Users/Andrew/AppData/Local/Programs/Python/Python313/python.exe` and that folder
  first on `PATH`, then `cargo test -p periodica_core --features python`.
- Leave the venv in place for lane R0e. Don't delete `target`.

## Tests
- Rust unit tests beside `props_of`: a nested sheet flattens (group names dropped); a root scalar
  beats the same key in a group; `Properties` beats a named group; a boolean and an excluded key
  are dropped; a string is kept; depth 4 is not reached.
- Python: `data_sheet("Stainless_Steel_316L")` is a `DataSheet`, has more than 10 keys, and
  `["Density_kgm3"]` is 7990.0 (put it where the existing bundled-data tests live).

## Verify before you report
1. `cargo test -p periodica_core` and, with the DLL setup, `--features python`.
2. `cargo fmt --check`; `cargo check --workspace --all-features` (with the DLL setup).
3. Rebuild the extension, then as CI does:
   `$V -m pytest tests/ --tb=short -q -m "not slow and not gemini" -p no:cacheprovider --basetemp=H:/claude-scratch/periodica-sheets/pytest-tmp`
   in the background, log to `H:/claude-scratch/periodica-sheets/pytest-r0d.log`. Expected: one
   failure only, `test_vectorised_and_legacy_scalar_sampler_agree` (lane R0e).
4. `git status --short` shows only your files: pytest must not have rewritten package data.

## Report (20 lines or fewer)
Commit hash; the pytest counts; each of the six tests and whether it passes; how many (entry,
property) pairs decline to Python in the sample-parity test; anything in the design you could not
meet; what you did not verify.
