# P1: the Python tests must not rewrite package data (lane `pytest-data`)

Read `_common.md` first. This lane touches no Rust source, so a Rust lane may be running in the
same checkout. Deliverable, on `claude/sheets`: after a full pytest run, `git status --short` is
empty.

## Known
- `tests/test_cli.py` calls the CLI against the package's real `data/derived/` folder:
  - line 94, `cli_main(["clear", "-y", "isotopes"])`, then line 96 builds them again;
  - line 109, `cli_main(["clear", "-y", "ions"])`; `test_ions_charge_calculation` builds them again;
  - lines 69-79 and 165 run a full `build` when a tier is missing;
  - line 127 runs a custom input into `derived/test_custom` and clears it afterwards.
- So a pytest run deletes and regenerates the 28 ion sheets and the 39 isotope sheets. The
  regenerated sheets differ from the committed ones: each gains about 26 keys (`AtomicNumber_Z`,
  `Properties`, `_provenance`, ...). The Rust test
  `cargo test -p periodica_core --test derived_regeneration` then fails with about 1150
  differences. This is how bad copies were committed once already (merge 5a7a054).
- Every path goes through `periodica.get._derived_dir()` (`src/periodica/get.py:117`), which reads
  `derived_root` from the config. `periodica/__main__.py:56` and `Save()` (`get.py:566`) both use
  it.

## Part 1: isolate the tests
1. Give the tests that build or clear a private derived folder: a fixture that copies
   `data/derived/` to `tmp_path_factory` and points `_derived_dir()` at the copy, then calls
   `reload_registry()` before and after. Put it in `tests/test_cli.py`, or in `tests/conftest.py`
   if other test files write there too.
2. Find every other test that writes under `src/periodica/data/`: run the full suite the way
   `.github/workflows/ci.yml` runs it and look at `git status --short`. Fix each one the same way.
3. If monkeypatching `_derived_dir` isn't enough (a cached path, a subprocess), you may add one
   small override to `get.py`: an environment variable `PERIODICA_DERIVED_ROOT` that, when set,
   replaces the derived folder. Document it in the function's docstring. Change nothing else in
   the package.
4. Until your fix works, each pytest run dirties the checkout and can break the other lane's Rust
   tests. Right after each run, restore the sheets with
   `git checkout -- src/periodica/data/derived/ions src/periodica/data/derived/isotopes` (these
   two explicit paths are the one allowed exception to the `_common.md` rule), and remove any
   untracked files your run left under `src/periodica/data/derived/`.

Files you may edit: `tests/test_cli.py`, `tests/conftest.py`, other test files found in step 2,
and `src/periodica/get.py` for step 3 only.

## Part 2: report only, change nothing
The Python builder (`python -m periodica build ions`) and the native composer (the one
`rust/periodica_core/tests/derived_regeneration.rs` checks) write different sheets for the same
registry row. Report:
- the exact key differences for one ion (`Ca2+`) and one isotope (`H-2`);
- which code adds the extra keys on the Python side, with file and line;
- whether `pip install -e .` with the Rust extension changes what `python -m periodica build`
  writes (does the CLI call the native composer when it is present?);
- which of the two the docs call canonical (`docs/`, `README.md`, `docs/claude-context.md`).

## Acceptance
- pytest, run as CI runs it (`pytest tests/ -v --tb=short -m "not slow and not gemini"`, with the
  absolute `PYTHONPATH`): the same pass count as before your change (R0 saw 1608 passed, 9
  skipped, 4 xfailed) and `git status --short` empty afterwards.
- A new test asserts that the CLI tests' derived folder is not the package's folder.
- Your report lists `git diff --name-only` for your commits.
