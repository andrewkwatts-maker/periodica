Next step: launch lane S3a from `docs/plan/briefs/S3a-new-scalar-properties.md` (one executor, in `periodica-sheets`). It may touch Python lists of property ids, so run the full pytest as well as the cargo checks before pushing it.

Written 2026-11-11 by session periodica-92 (it took the topic over from periodica-1d). Topic: periodica datasheets and formulas for the PlayTow engine. The plan is `docs/plan/playtow-sheets.md`; read its lanes table and "Census findings" section before anything else, and no more than that.

## State
- `origin/main`, the main checkout (`H:/Projects/MathsPhysicsEcosystem/periodica`) and the lane checkout (`H:/Projects/MathsPhysicsEcosystem/periodica-sheets`, branch `claude/sheets`) are at the commit that adds this text. It changes only this file. No agent and no background task is running. Nothing is unpushed.
- CI: green in all four jobs on 5242039 (run 42900204282), which holds all the code below. The run for the handoff commit on top of it was not watched.
- Landed 2026-11-11, all on main: R1b census (5f306e1), R0d follow-up (0a8bc00), S5a column export (9fb3908, d35db37), R0e grain hash (f43e075), `sha1` dependency removed (109a4c3), F2a elastic gates (8ed7d85), F2a test fix (2cd310e), F2b Tabor gate and plane-strain Irwin (8a08b5a, 351d5de).
- CI was red on ff3ae8c and 2dacb57: an F2a test found Steel-1018 by a directory walk, and Linux listed the `reference/alloys/` copy first, which has no Poisson's ratio. A local Windows pass does not show this class of failure. `briefs/_common.md` now says to load a sheet by its path.
- F2b in one line: Tabor's rule runs only when `shows_ductile_metal(table)` holds. 53 sheets lost estimated values, 48 of them rightly (tissues, polymers, ceramics, concrete, timber). The lanes table has the rule and the counts.
- PyPI still serves 2.0.2 (2026-05-17). The repo is at 2.3.0 with no tag and no release. A release needs the owner (`gh auth login`, and decision 9).

## Lane order from here
1. S3a: ten new property ids. Brief ready.
2. F2c: five metal sheets state neither elongation nor resistivity, so F2b gives them no Tabor estimate (`derived/alloys` Magnesium-AZ31B and Solder-Sn63Pb37; `reference/alloys` Steel-1018, Aluminum-6061, Titanium-Ti6Al4V). No brief yet. The lanes table lists the two ways to fix it; one needs owner decision 11.
3. R0f: Rust resolves aliases and units. No brief yet; it ties to owner decision 13.
4. S3b, S3c (need owner decision 11, the data source and licence), S1, S2.

## Other sessions
- playtowecosystem-16 was told on 2026-11-11 what F2b changed in its eleven sheets: Timber-C24 and Rubber-Natural have no `Hardness` and no `HardnessVickers`; Mortar-CementLime and Brick-Clay take `Hardness` from `3 * CompressiveStrength`; the three concrete sheets lost the estimated `HardnessVickers`; the steel and aluminium sheets are unchanged. It answered the same day, and both questions are closed: (1) keep `Hardness` missing on timber and rubber, with no flagged estimate, because its contact model refuses a sheet by naming the missing input and that is the behaviour it wants; (2) no narrower row rule for the export, which it does not read yet. Its periodica checkout is held at 5c70977 until its Dynameis change SE3 lands; then it pulls and changes its own tests to expect the refusal.
- periodica-bd holds no work and stays out of in-flight files. Tell it what moved after each push.

## How a lane runs
- One executor agent, one brief, one deliverable, about 60 tool calls. At most 3 agents. Two lanes that both build `periodica-mat` run in sequence; one `periodica_core` lane may run beside one `-p periodica-mat` lane.
- Launch prompt: `subagent_type: executor`; work only in the `periodica-sheets` checkout; read `docs/plan/briefs/_common.md`, then the brief; `export CARGO_TARGET_DIR=H:/claude-scratch/periodica-sheets/target TEMP=H:/claude-scratch/tmp TMP=H:/claude-scratch/tmp`; the line "Run sweeps and builds with run_in_background and wait for the notification, or use Monitor with an until-condition, never sleep loops."; a cargo run under 10 minutes goes in the foreground with timeout 600000; no "hold open" timers; never kill by process name; H: only; hard stop at about 100 turns with a Checkpoint; commit with explicit paths, no AI attribution, no push, no new branch or worktree; `cargo fmt --check` from the repo root before committing.
- Review before pushing: `git log --format='%h %an | %s' origin/main..HEAD`; the attribution count (`git log --format='%B' origin/main..HEAD | grep -ci 'co-authored-by\|generated with'`) must be 0; read the diff. Then, from the repo root with `PYO3_PYTHON=C:/Users/Andrew/AppData/Local/Programs/Python/Python313/python.exe` and that folder first on `PATH`: `cargo fmt --check`; `cargo clippy -p periodica-mat --all-targets -- -D warnings`; the same for `periodica-runtime`; `cargo check --workspace --all-features`; `cargo test --workspace --no-fail-fast`; and for a core change `cargo test -p periodica_core --features python`. Run them as one background command that echoes a marker after each step, and count the markers: a failed `cargo fmt --check` does not stop the rest.
- Don't take an executor's count of changed sheets from `domain-census.md`. F2b's executor reported 9; the real number was 53. Diff the `props` of every sheet in `docs/dev/domain-census.json` against `git show origin/main:docs/dev/domain-census.json`: properties lost, gained, and tags changed.
- Push: `git push origin <full sha>:main`, then `git -C /h/Projects/MathsPhysicsEcosystem/periodica fetch origin --quiet` and `merge --ff-only origin/main`. Update the lanes table in the same push. Tell periodica-bd, and PlayTow when something it reads changes.
- CI watcher: one background Bash loop that reads `https://api.github.com/repos/andrewkwatts-maker/periodica/actions/runs?head_sha=<full sha>` every 85 s, 6 times (a run takes about 2 minutes), parses with `python -I`, then lists failed steps from `/actions/runs/<id>/jobs`. `gh` is not signed in, the limit is 60 requests an hour, and job logs need a sign-in. With no log, reproduce a Linux-only failure by reasoning from what the failing commit added.
- Python: venv at `H:/claude-scratch/periodica-sheets/venv`. Rebuild with `venv/Scripts/python -m pip install -e . --no-deps`. Full run: `-m pytest tests/ --tb=short -q -m "not slow and not gemini" -p no:cacheprovider --basetemp=H:/claude-scratch/periodica-sheets/pytest-tmp`. Last result (before F2a): 1520 passed, 0 failed, 99 skipped. Never install into the global Python.

## Open items
- Owner decisions 1 to 7 and 9 to 13 in the plan. The ones that block lanes: 11 (data source and licence; it now also covers surface energy for PlayTow's eleven sheets, measured hardness for timber and rubber, Poisson's ratio for Steel-B500B, an orthotropy block for timber, water and air sheets, and elongation or resistivity for the five F2c metals), 12, 13, and 9 (the sdist cannot build alone; the release workflow is untested).
- F2b: a polymer sheet with an elongation of 1 to 60 % and no resistivity still passes the ductile-metal test; no corpus sheet does today. Irwin's `Jic` stays `Estimated` even with Poisson's ratio stated; the brief did not say which tag. The three unchanged metal sheets were compared with values the executor recorded, not with a run of the old code.
- The local Arithma checkout is at a842d94 (2026-10-17); `origin/main` there is 45a54c3 (2026-11-03), which is what CI clones. Local Rust runs build against the older one. This session did not touch that repo.
- Export rows: with `has_density` the export has 328 rows; 74 of them fail `is_material_like` (25 cells, 23 elements, 21 molecules, 3 composites, 1 material, 1 alloy). Nobody has checked that the cells and molecules are bulk matter.
- F2a: no corpus sheet was checked for `orthotropy`; that gate has a unit test only.
- R0e: not checked for NaN or huge coordinates, or for points with fewer than 3 columns.
- Steel-1018 and Aluminum-6061 exist in both `derived/` and `reference/`; one copy should go. 116 sheet names are duplicated across roots.
- Element sheets ingest emission wavelengths as `Source::Datasheet`. The gold citation is unchecked. CI clones Arithma at HEAD while the release pins 447a7c0.
- Scratch under `H:/claude-scratch/periodica-sheets/`: `pyrepro/`, `arithma-head/`, `census-before.json` and the `prepush-*.log` and `f2b-review-test.log` files are stale and can be deleted. Keep `venv/` and `target/` (3.6 GB) until S3a is pushed.
- Outside this topic, still owed: the metaphysica CI failure on 5a5d5d4 (run 40586739547), the website rebuild and deploy (https://www.metaphysicæ.com), PyPI releases, and Track S (S1 and G1 stopped mid-work; V1 complete on `claude/glow`, unmerged).

## Do not
- Do not run `git worktree add` or clone; the owner's hook bans it. Do not drop or apply the main checkout's stash (`stash@{0}`, WIP on lane/cleanup); it is someone else's.
- Do not edit the Arithma repo or the PlayTow repo. Do not touch the metaphysica main checkout's uncommitted `src/metaphysica/data/path_comparison.json`.
- periodica must not depend on Themelios or Topos; `periodica-mat` takes no new dependency for the export.
- Do not read CI logs by pulling a token out of the credential manager.
