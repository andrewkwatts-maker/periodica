Next step: review lane F2b and push it. Session periodica-92 launched its executor on 2026-11-11 from `docs/plan/briefs/F2b-tabor-gate-irwin.md`, on 5c70977 in `periodica-sheets`. If no agent is running and `git log origin/main..claude/sheets` is empty, the lane died: look at `git status` in `periodica-sheets` for partial work in `derive.rs`, `chains.rs`, `tests/f2b_tabor_irwin.rs` and `docs/dev/`, check the brief for a `## Checkpoint`, and relaunch.

Written 2026-11-11 by session periodica-1d, updated the same day by periodica-92. Topic: periodica datasheets and formulas for the PlayTow engine. The plan is `docs/plan/playtow-sheets.md`; read its lanes table and "Census findings" section before anything else, and no more than that.

## State
- `origin/main` and the main checkout (`H:/Projects/MathsPhysicsEcosystem/periodica`) are at the commit that holds this text. The lane checkout (`H:/Projects/MathsPhysicsEcosystem/periodica-sheets`, branch `claude/sheets`) has the F2b executor working in it.
- Landed 2026-11-11, all on main: R1b census (5f306e1), R0d follow-up (0a8bc00), S5a column export (9fb3908, d35db37), R0e grain hash (f43e075), `sha1` dependency removed (109a4c3), F2a elastic gates (8ed7d85), F2a test fix (2cd310e).
- CI: green in all four jobs on 5c70977 (run 42895675467). It was red on ff3ae8c and 2dacb57 in the Rust `Test` step: the F2a test found Steel-1018 by a directory walk, and Linux listed the `reference/alloys/` copy first, which has no Poisson's ratio. 2cd310e loads sheets by path; `briefs/_common.md` now has that rule. A local Windows pass does not show this class of failure, so watch the GitHub run after every push.
- The local Arithma checkout is at a842d94 (2026-10-17); `origin/main` there is 45a54c3 (2026-11-03), which is what CI clones. Local Rust runs build against the older one.
- PyPI still serves 2.0.2 (2026-05-17). The repo is at 2.3.0 with no tag and no release. A release needs the owner (`gh auth login`, and decision 9).

## Lane order from here
1. F2b: Tabor's rule for ductile metals only, plane-strain Irwin. Brief ready. Facts newer than the brief: F2a added `DeriveGates` (`isotropic_elastic`), `fill_derived_with(t, gates)` and `gates_for(record)` in `rust/periodica-mat/src/derive.rs`; `ingest.rs` calls the gated form.
2. S3a: ten new property ids (`briefs/S3a-new-scalar-properties.md`). It may touch Python lists of ids, so run the full pytest before pushing it.
3. R0f: Rust resolves aliases and units. No brief yet; it ties to owner decision 13.
4. S3b, S3c (need owner decision 11, the data source and licence), S1, S2.

After F2b lands, tell playtowecosystem-16: Timber-C24 and Rubber-Natural no longer have `Hardness` or `HardnessVickers`, and mortar and brick take `Hardness` from `3 * CompressiveStrength`. PlayTow's contact model refuses a sheet without `Hardness`. It was asked on 2026-11-11 whether it prefers a flagged estimate, and whether it wants a narrower row rule for the export; neither is answered.

## How a lane runs
- One executor agent, one brief, one deliverable, about 60 tool calls. At most 3 agents. Two lanes that both build `periodica-mat` run in sequence; one `periodica_core` lane may run beside one `-p periodica-mat` lane.
- Launch prompt: `subagent_type: executor`; work only in the `periodica-sheets` checkout; read `docs/plan/briefs/_common.md`, then the brief; `export CARGO_TARGET_DIR=H:/claude-scratch/periodica-sheets/target TEMP=H:/claude-scratch/tmp TMP=H:/claude-scratch/tmp`; the line "Run sweeps and builds with run_in_background and wait for the notification, or use Monitor with an until-condition, never sleep loops."; a cargo run under 10 minutes goes in the foreground with timeout 600000; no "hold open" timers; never kill by process name; H: only; hard stop at about 100 turns with a Checkpoint; commit with explicit paths, no AI attribution, no push, no new branch or worktree; `cargo fmt --check` from the repo root before committing.
- Review before pushing: `git log --format='%h %an | %s' origin/main..HEAD`; the attribution count (`git log --format='%B' origin/main..HEAD | grep -ci 'co-authored-by\|generated with'`) must be 0; read the diff. Then, from the repo root with `PYO3_PYTHON` set to the Python 3.13 exe and that folder first on `PATH`: `cargo fmt --check`; `cargo clippy -p periodica-mat --all-targets -- -D warnings`; the same for `periodica-runtime`; `cargo check --workspace --all-features`; `cargo test --workspace`; and for a core change `cargo test -p periodica_core --features python`. Run them as one background command.
- Push: `git push origin <full sha>:main`, then `git -C /h/Projects/MathsPhysicsEcosystem/periodica fetch origin --quiet` and `merge --ff-only origin/main`. Update the lanes table in the same push. Tell periodica-bd what moved (it holds no work and stays out of in-flight files) and PlayTow when something it reads changes.
- CI watcher: one background Bash loop that reads `https://api.github.com/repos/andrewkwatts-maker/periodica/actions/runs?head_sha=<full sha>` every 85 s, up to 22 times, parses with `python -I`, then lists failed steps from `/actions/runs/<id>/jobs`. `gh` is not signed in, the limit is 60 requests an hour, and job logs need a sign-in.
- Python: venv at `H:/claude-scratch/periodica-sheets/venv`. Rebuild with `venv/Scripts/python -m pip install -e . --no-deps`. Full run: `-m pytest tests/ --tb=short -q -m "not slow and not gemini" -p no:cacheprovider --basetemp=H:/claude-scratch/periodica-sheets/pytest-tmp`. Last result: 1520 passed, 0 failed, 99 skipped. Never install into the global Python.

## Open items
- Owner decisions 1 to 7 and 9 to 13 in the plan. The ones that block lanes: 11 (data source and licence; it now also covers surface energy for PlayTow's eleven sheets, measured hardness for timber and rubber, Poisson's ratio for Steel-B500B, an orthotropy block for timber, and water and air sheets), 12, 13, and 9 (the sdist cannot build alone; the release workflow is untested).
- Export rows: with `has_density` the export has 328 rows; 74 of them fail `is_material_like` (25 cells, 23 elements, 21 molecules, 3 composites, 1 material, 1 alloy). Nobody has checked that the cells and molecules are bulk matter.
- F2a: no corpus sheet was checked for `orthotropy`; that gate has a unit test only.
- R0e: not checked for NaN or huge coordinates, or for points with fewer than 3 columns.
- Steel-1018 and Aluminum-6061 exist in both `derived/` and `reference/`; one copy should go. 116 sheet names are duplicated across roots.
- Element sheets ingest emission wavelengths as `Source::Datasheet`. The gold citation is unchecked. CI clones Arithma at HEAD while the release pins 447a7c0.
- Scratch under `H:/claude-scratch/periodica-sheets/`: `pyrepro/` and `arithma-head/` are stale and can be deleted. Keep `venv/` and `target/` (3.6 GB) until S3a is pushed.
- Outside this topic, still owed: the metaphysica CI failure on 5a5d5d4 (run 40586739547), the website rebuild and deploy (https://www.metaphysicæ.com), PyPI releases, and Track S (S1 and G1 stopped mid-work; V1 complete on `claude/glow`, unmerged).

## Do not
- Do not run `git worktree add` or clone; the owner's hook bans it. Do not drop or apply the main checkout's stash (`stash@{0}`, WIP on lane/cleanup); it is someone else's.
- Do not edit the Arithma repo or the PlayTow repo. Do not touch the metaphysica main checkout's uncommitted `src/metaphysica/data/path_comparison.json`.
- periodica must not depend on Themelios or Topos; `periodica-mat` takes no new dependency for the export.
- Do not read CI logs by pulling a token out of the credential manager.
