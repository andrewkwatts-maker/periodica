# Rules for every lane brief (read before you start)

## Where to work
- Work in the main checkout, `H:/Projects/MathsPhysicsEcosystem/periodica`, on `main`. Commit there
  directly. No branches, no worktrees and no clones: a TokenWatch hook bans `git worktree add`
  (worktrees took 108 GB), and the owner's rule is one checkout on `main`.
- Older briefs name a lane branch `claude/<lane>`, a worktree `../periodica-<lane>`, or the
  checkout `../periodica-sheets` on `claude/sheets` (removed 2026-11-11). Ignore all three and
  read their paths as the main checkout.
- Another lane may be working in the same checkout. Edit only the files your brief names. Stage
  and commit only your own paths: `git add <paths>`, then `git commit -m "..." -- <paths>`. Never
  `git add -A`, `git commit -a`, `git stash`, `git reset`, `git checkout -- .` or `git clean`.
- The checkout holds a stash (`stash@{0}`, WIP on lane/cleanup) that belongs to someone else.
  Never apply, pop or drop it.
- If a build or a test fails in a file your brief doesn't name, don't fix it. Report it.
- Build into the shared `CARGO_TARGET_DIR=H:/claude-scratch/periodica-sheets/target` (the scratch
  folder keeps its old name). Set `TEMP=TMP=H:/claude-scratch/tmp`. Put other scratch under
  `H:/claude-scratch/periodica-sheets/<lane>/`. Never write to C:.
- For Python, use an absolute `PYTHONPATH=H:/Projects/MathsPhysicsEcosystem/periodica/src`.
- Git Bash rewrites `origin/main:path`. Prefix such git commands with `MSYS_NO_PATHCONV=1`.

## How to work
- Run sweeps and builds with run_in_background and wait for the notification, or use Monitor with
  an until-condition, never sleep loops.
- Never kill processes by name (python, cargo, ...). Kill only PIDs you started.
- Budget: about 60 tool calls. Hard stop at about 100 turns. If you hit it, add a `## Checkpoint`
  section to your brief (what's done, what's next, the exact commands) and stop.
- SOLID. Functions of 60 lines or fewer. Fail early. Add a test for each fix. Permissive licences
  only.
- Commits: no AI attribution and no Co-Authored-By lines. Don't push; the planner pushes.
- Final report, 15 lines or fewer: commit hashes, the test commands and their results, and open
  questions.

## Real quantities only (owner's rule, 2026-11-11)
- A property is a quantity with a standard name in physics, engineering or materials science and
  an SI unit. Its doc comment gives the definition. No tuning knobs and no engine parameters.
- A sheet value is a measured or tabulated value with its source cited in the sheet. Never type a
  value from memory, back-solve one so that a test or another value agrees, or fill a gap with a
  guess. A missing value stays missing.
- A value computed from other values is not stored in a sheet. Code derives it and tags it
  `Derived` (an exact identity) or `Estimated` (an empirical relation, named with its reference
  and the range it holds in).
- A fix changes a rule or the data for a stated physical reason. No clamp, floor, fudge factor or
  threshold without a derivation or a reference. If you add any number that is not a measured
  value or a unit conversion, list it in your report with where it comes from.

## Before each commit
- Run `cargo fmt` from the repo root; CI runs `cargo fmt --check`. Never `cargo fmt --all`: it also
  rewrites the Arithma path dependency, which is another repo.
- Run Rust tests with `--no-fail-fast`, so one failing binary doesn't hide the rest.
- A test that reads a corpus sheet names its path under `src/periodica/data`. Don't pick a sheet
  by file name from a directory walk: 116 names repeat across the data roots and `read_dir` order
  differs between Windows and Linux, so the test passes here and fails in CI.
- Report what you verified and say what you didn't. Check a claim about what a commit touches with
  `git diff --name-only`, not from memory.
- The workspace depends on `arithma_core` by path (`../Arithma/rust/arithma_core`). The sibling
  repo is at `H:/Projects/MathsPhysicsEcosystem/Arithma`, so the checkout builds. Don't edit
  `Cargo.toml` dependencies.
