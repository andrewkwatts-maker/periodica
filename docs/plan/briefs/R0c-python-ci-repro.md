# R0c: reproduce the Python CI failure with the Rust extension (lane `pyrepro`)

Read `_common.md` first. This lane changes nothing in the repo. Deliverable: your final report,
which names every failing test, its cause and a proposed fix.

## Known
- GitHub Actions run 42866560822 (commit 4191f64): the three Python jobs (3.11, 3.12, 3.13, on
  Ubuntu) fail at the step "Run tests". The steps before it pass, including "Assert the Rust
  backend is live". The logs can't be read from here (no GitHub login).
- The job, from `.github/workflows/ci.yml:86-128`: clone Arithma beside the repo;
  `python -m pip install -e ".[test]"` (maturin builds the extension into `src/periodica`); check
  `periodica._HAS_RUST`; then `pytest tests/ -v --tb=short -m "not slow and not gemini"`.
- Every local pytest run so far had no extension (`src/periodica/*.pyd` is absent and ignored by
  git), so `_HAS_RUST` was False and the tests that need Rust were skipped: 1608 passed, 9
  skipped, 4 xfailed. The failures are probably among the tests that only run with Rust.
- CI clones Arithma's `main` (45a54c3). The local Arithma checkout is behind it (a842d94).
- `tests/test_cli.py` deletes and rebuilds sheets under `src/periodica/data/derived/` (lane P1 is
  fixing that). The Python builder without Rust writes about 26 more keys per ion and isotope
  sheet than the committed sheets have.

## Steps
Two other lanes are editing the checkout, so work on an export of the committed files. It is not a
clone or a worktree, and you delete the heavy parts at the end.

1. Export, with `R=H:/claude-scratch/periodica-sheets/pyrepro`:
   - `git -C H:/Projects/MathsPhysicsEcosystem/periodica-sheets archive HEAD | tar -x -C $R/repo`
   - `git -C H:/Projects/MathsPhysicsEcosystem/Arithma archive origin/main | tar -x -C $R/Arithma`
     (the workspace finds it at `../Arithma`). Don't fetch, pull or edit the Arithma repo.
   - Arithma has no `LICENSE` beside the crate, and the build may want one:
     `cp $R/Arithma/LICENSE $R/Arithma/rust/arithma_core/LICENSE` if it is missing.
2. Environment for every command: `TEMP=TMP=H:/claude-scratch/tmp`,
   `CARGO_TARGET_DIR=$R/target`, `PIP_CACHE_DIR=H:/claude-scratch/pip-cache`. Create a virtual
   environment at `$R/venv` and use only its python. Never install into the global Python.
3. In `$R/repo`: `$R/venv/Scripts/python -m pip install -e ".[test]"`, in the background. Then run
   the `_HAS_RUST` check from `ci.yml:114-125`. If it is False, find out why and stop there.
4. Run pytest as CI does, from `$R/repo`, in the background, with
   `--basetemp=$R/tmp -p no:cacheprovider`, writing the log to `$R/pytest.log`. Don't set
   `PYTHONPATH`: the editable install and `tests/conftest.py` decide the import path, as in CI.
   Check first that `python -c "import periodica; print(periodica.__file__)"` prints a path under
   `$R/repo`.
5. For each failing test: the test id, the assertion or exception, the cause in one sentence, and
   a proposed fix with file and line. Group tests that share a cause. Say for each group whether
   the test is wrong or the code is wrong.
6. If everything passes here, the failure is specific to Linux. Then look for: file names in
   `git ls-files` that differ only by case, tests that open a path whose case doesn't match the
   file, path separators, line endings, and tests that depend on the order of
   `os.listdir` or `glob`. Report the candidates with file and line.
7. Run pytest a second time in the same export and compare the failures. A different result means
   the tests depend on state that the first run changed (see the last item under Known); say
   which.
8. Delete `$R/target` and `$R/venv`. Keep the logs.

## Report (25 lines or fewer)
The pass, fail and skip counts with Rust live; the failure groups from step 5; whether the second
run differed; what you did not verify.
