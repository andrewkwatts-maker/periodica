Next step: none until the owner or PlayTow asks for more. The owner closed this topic at lane Q1a on 2026-11-11. On a new request, start at "Lane order from here": Q1b is first and needs a brief.

Written 2026-11-11 at the close of lane Q1a. Topic: periodica datasheets and formulas for the PlayTow engine. The plan is `docs/plan/playtow-sheets.md`; read its lanes table and "Census findings" section before anything else, and no more than that.

## State
- One checkout and one branch: `H:/Projects/MathsPhysicsEcosystem/periodica` on `main`, equal to `origin/main`. The lane worktree `../periodica-sheets` and the branch `claude/sheets` are gone (decision 10). No agent and no background task is running. Nothing is unpushed. The stash (`stash@{0}`, WIP on lane/cleanup) is untouched.
- Landed: Q1a (6e4649c, 3bdaf23, 623d7ff). Every constant in `chains.rs` and `derive.rs` has a derivation or a reference. `H ~ 3 f_c` is gone. The slope correlation applies only inside its source's range (R_q 0.216 to 9.6 um) and gives an rms slope. There is one Tabor factor (3.0) and one `KGF_MM2_PA`. `nu_is_stable` is the one test of `-1 < nu < 0.5`.
- What moved in what PlayTow reads, checked in the column export and the per-sheet census diff:
  - `Hardness` is missing on 9 sheets that had it only from 3 times compressive strength: Concrete-C25, -C30, -C40, Concrete-Mix, Brick-Clay, Mortar-CementLime, `tissues/cartilage`, and `reference/ceramics` Alumina and SiliconCarbide. A cited hardness for them is data work (decision 11).
  - `HardnessVickers` estimated from UTS rose 0.952 % on 29 sheets (Steel-1018 133.33 to 134.60).
  - `SurfaceSlopeRms`: Steel_8620_CaseHardened 0.0946 to 0.1187 rad; Granite_Westerly lost it (R_q 62.7 um is outside the range).
  - No elastic value and no sound speed moved.
- Local checks on 3bdaf23: `cargo fmt --check`, clippy for `periodica-mat` and `periodica-runtime`, `cargo check --workspace --all-features` and `cargo test --workspace --no-fail-fast` all passed: 508 Rust tests, none failed, none ignored. The Python tests were not run (no Python file changed).
- CI: green in all four jobs on 623d7ff (run 42976701741). The closing commits after it change docs only.
- The owner's rule of 2026-11-11, in `briefs/_common.md` ("Real quantities only"), binds every lane: a property is a real quantity with an SI unit; a sheet value is measured or tabulated and cited; a computed value is derived in code and tagged `Derived` or `Estimated`; no clamp, floor, fudge factor or threshold without a derivation or a reference. Check each executor's diff against it, and ask the executor to list every number it added.
- PyPI still serves 2.0.2 (2026-05-17). The repo is at 2.3.0 with no tag and no release. A release needs the owner (`gh auth login`, and decision 9).

## Changed by the close
- Lanes run in the main checkout on `main` (`briefs/_common.md`; "How lanes run" in the plan). `sync.bat` commits and pushes `main`, so lane work can reach `origin` before the planner's review; review what is on `origin/main` and fix forward.
- `H:/claude-scratch/periodica-sheets/target` (4 GB) was deleted. The next build is a cold one. The scratch folder keeps its old name.
- The venv's editable install still points at the removed checkout. Before any Python run, from the main checkout: `H:/claude-scratch/periodica-sheets/venv/Scripts/python -m pip install -e . --no-deps`.
- C: had 54 MB free on 2026-11-11 (488 GB used). A command whose output passes through Claude Code's temp folder on C: fails with "unknown write failure on standard output", though the command itself has run. Redirect output to a file on H: and Read it. Nothing was deleted on C:.

## Lane order from here
1. Q1b: ids that are not material properties. No brief yet. PlayTow's answers are in the lanes table. The brief must settle what happens to stored `.pmat` files when ids are removed, because `PropertyId` is dense.
2. Q1c: uncited defaults in `record.rs` and `ingest.rs`, listed in the lanes table. No brief yet. `optics/*.rs` has not been read for constants: one scout, that folder only.
3. F2c: five metal sheets state neither elongation nor resistivity; also `DUCTILE_ELONGATION_PCT`. No brief yet; one way needs owner decision 11.
4. R0f: Rust resolves aliases and units. No brief yet; it ties to owner decision 13.
5. S3b, S3c (need owner decision 11, the data source and licence), S1, S2.

What PlayTow asked for and still lacks: the S3b values (latent heats, viscosity, sound speed, loss factors, bulk water and air sheets), the S3c spectral lines, MP-PER-2 expressions, and one uncertainty per value in the export.

## Other sessions
- PlayTow is playtowecosystem-e0. On 2026-11-11 it was sent the Q1a commits, the nine sheets that lost `Hardness` and the other changes above, then a line that CI is green and that no further lane is queued. No answer was asked for. Its earlier answers stand: keep `Hardness` missing on timber and rubber; no narrower row rule for the export; it prefers a refused sheet to an uncited number.

## How a lane runs
- One executor agent, one brief, one deliverable, about 60 tool calls. At most 3 agents. Two lanes that both build `periodica-mat` run in sequence; one `periodica_core` lane may run beside one `-p periodica-mat` lane. The owner asked for Fable 5.2 agents where the work needs it; Q1a's four agents ran on Sonnet 5.5.
- A scout brief names one file or one small folder. TokenWatch flagged a scout at 127k context on 2026-11-11 after one brief covered 18 files.
- Launch prompt: `subagent_type: executor`; work in the main checkout on `main`; read `docs/plan/briefs/_common.md`, then the brief; `export CARGO_TARGET_DIR=H:/claude-scratch/periodica-sheets/target TEMP=H:/claude-scratch/tmp TMP=H:/claude-scratch/tmp`; the line "Run sweeps and builds with run_in_background and wait for the notification, or use Monitor with an until-condition, never sleep loops."; a cargo run under 10 minutes goes in the foreground with timeout 600000; no "hold open" timers; never kill by process name; H: only; hard stop at about 100 turns with a Checkpoint; commit with explicit paths, no AI attribution, no push, no branch or worktree; `cargo fmt --check` from the repo root before committing.
- Review before pushing: `git log --format='%h %an | %s' origin/main..HEAD`; the attribution count (`git log --format='%B' origin/main..HEAD | grep -ci 'co-authored-by\|generated with'`) must be 0; read the diff. Then, from the repo root with `PYO3_PYTHON=C:/Users/Andrew/AppData/Local/Programs/Python/Python313/python.exe` and that folder first on `PATH`: `cargo fmt --check`; `cargo clippy -p periodica-mat --all-targets -- -D warnings`; the same for `periodica-runtime`; `cargo check --workspace --all-features`; `cargo test --workspace --no-fail-fast`; and for a core change `cargo test -p periodica_core --features python`. Run them as one background command that echoes a marker after each step, and count the markers: a failed `cargo fmt --check` does not stop the rest. With a warm build the sequence takes about 15 minutes.
- Check the physics, not only the tests. S3a passed every test and its brief, and 21 of its values were false; F2d's first record held false zeros; Q1a's brief predicted 3 sheets losing `Hardness` and the census showed 9. Export the columns (`cargo run -p periodica-mat --example export_columns -- <dir>`), then look at the smallest and largest values of each new or changed property against what is measured. Scripts in `H:/claude-scratch/periodica-sheets/review/`: `colcmp.py OLD NEW [Property]` compares two exports column by column (lost, gained, value changed, tag changed); `census_diff.py OLD.json NEW.json` does the same per sheet for the census; `speeds.py DIR` and `gate2.py DIR DATA_ROOT` list sound speeds and which sheets are generated. `../q1a/export/` is the export at 3bdaf23 and is the baseline for the next lane; `../f2d/export/` is the one before Q1a.
- Don't take an executor's count of changed sheets from `domain-census.md`. Diff the `props` of every sheet in `docs/dev/domain-census.json` against `git show origin/main:docs/dev/domain-census.json` with `census_diff.py`.
- Push: `git push origin main`, with the lanes table updated in the same push. Tell PlayTow when something it reads changes.
- CI watcher: one background Bash loop that reads `https://api.github.com/repos/andrewkwatts-maker/periodica/actions/runs?head_sha=<full sha>` every 85 s, 6 times (a run takes about 2 minutes), parses with `python -I`, then lists failed steps from `/actions/runs/<id>/jobs`. `gh` is not signed in, the limit is 60 requests an hour, and job logs need a sign-in. With no log, reproduce a Linux-only failure by reasoning from what the failing commit added.
- Python full run, after the reinstall above: `-m pytest tests/ --tb=short -q -m "not slow and not gemini" -p no:cacheprovider --basetemp=H:/claude-scratch/periodica-sheets/pytest-tmp`. Never install into the global Python.

## Open items
- Owner decisions 1 to 7, 9 and 11 to 15 in the plan. The ones that block lanes: 11 (data source and licence), 12, 13, 9 (the sdist cannot build alone; the release workflow is untested), 14 (the generated sheets: keep as `Estimated`, keep out of the export, or delete and rebuild from cited data). Decision 15 (the cut-off for nu near 0.5) blocks nothing; Q1a keeps 0.45.
- Q1a, not fixed: the slope range test includes its end points, where the source states the range without saying so; `HRC_TO_HV` cites no standard; `chains.rs` takes `Hardness` from 3 times yield while `derive.rs` takes HV from 3 times UTS.
- Q1a's sources: the planner read the Electronics Cooling page and the abstract of the Mott-Ioffe-Regel review on the web. Tabor's book, the 1991 ASME paper, Landau and Lifshitz and Ganneau, Constantinides and Ulm (2006) were not read; the doc comments say so.
- F2d: 177 corpus lines state E, nu and at least one of G and K, and the stated constants disagree by up to 27 % in G (Rh) and 71 % in K (HDPE); the record is `rust/periodica-mat/tests/oracles/elastic-mismatch.txt`. No sheet was changed: which constant is wrong needs a cited source per sheet (decision 11). `ingest` and `ingest_stated` are over 60 lines.
- P2, not verified: `Hardness` changed on 74 generated alloy sheets and went missing on 2; nobody checked the 74 values one by one. Granite_Westerly lost its longitudinal speed because it states a porosity of 0.9 %. The pore gate only sees `WaterContent` and `Porosity`. The 20 generated tissue sheets write `"porosity": 30.0` with no unit in the key, so `Porosity` is not ingested for them.
- F2b: a polymer sheet with an elongation of 1 to 60 % and no resistivity still passes the ductile-metal test; no corpus sheet does today. Irwin's `Jic` stays `Estimated` even with Poisson's ratio stated.
- The local Arithma checkout is at a842d94 (2026-10-17); `origin/main` there is 45a54c3 (2026-11-03), which is what CI clones. Local Rust runs build against the older one. CI clones Arithma at HEAD while the release pins 447a7c0.
- Export rows: with `has_density` the export has 328 rows; 74 of them fail `is_material_like` (25 cells, 23 elements, 21 molecules, 3 composites, 1 material, 1 alloy). Nobody has checked that the cells and molecules are bulk matter.
- F2a: no corpus sheet was checked for `orthotropy`. R0e: not checked for NaN or huge coordinates, or for points with fewer than 3 columns.
- Steel-1018 and Aluminum-6061 exist in both `derived/` and `reference/`; one copy should go. 116 sheet names are duplicated across roots. The gold citation is unchecked.
- Scratch under `H:/claude-scratch/periodica-sheets/`: keep `venv/`, `review/`, `p2/`, `f2d/`, `q1/` (`q1/constants-inventory.md` is the scout's list behind lane Q1c), `q1a/`, `pyrepro/` (the plan cites its log) and `main-checkout-backup/`. The loose `*.log` and `gha-*.txt` files are old run records.
- Outside this topic, still owed: the metaphysica CI failure on 5a5d5d4 (run 40586739547), the website rebuild and deploy (https://www.metaphysicæ.com), PyPI releases, and Track S (S1 and G1 stopped mid-work; V1 complete on `claude/glow`, unmerged).

## Do not
- Do not run `git worktree add` or clone; the owner's hook bans it. Do not drop or apply the stash (`stash@{0}`, WIP on lane/cleanup); it is someone else's.
- Do not edit the Arithma repo or the PlayTow repo. Do not touch the metaphysica main checkout's uncommitted `src/metaphysica/data/path_comparison.json`.
- periodica must not depend on Themelios or Topos; `periodica-mat` takes no new dependency for the export.
- Do not read CI logs by pulling a token out of the credential manager.
