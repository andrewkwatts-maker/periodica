# Rules for every lane brief (read before you start)

## Where to work
- Make your worktree beside the repo, from the integration branch:
  `git -C H:/Projects/MathsPhysicsEcosystem/periodica worktree add ../periodica-<lane> -b claude/<lane> claude/sheets`
- Put scratch and builds under `H:/claude-scratch/periodica-<lane>`. Set
  `TEMP=TMP=H:/claude-scratch/tmp` and `CARGO_TARGET_DIR=H:/claude-scratch/periodica-<lane>/target`.
  Never write to C:.
- For Python, use an absolute `PYTHONPATH=H:/Projects/MathsPhysicsEcosystem/periodica-<lane>/src`.
  Without it, subprocesses import the main checkout.
- Don't touch the main checkout `H:/Projects/MathsPhysicsEcosystem/periodica`. Its uncommitted
  `tests/deviation_report.txt` and its stash belong to someone else.
- Git Bash rewrites `origin/main:path`. Prefix such git commands with `MSYS_NO_PATHCONV=1`.

## How to work
- Run sweeps and builds with run_in_background and wait for the notification, or use Monitor with
  an until-condition, never sleep loops.
- Never kill processes by name (python, cargo, ...). Kill only PIDs you started.
- Budget: about 60 tool calls. Hard stop at about 100 turns. If you hit it, add a `## Checkpoint`
  section to your brief (what's done, what's next, the exact commands) and stop.
- SOLID. Functions of 60 lines or fewer. Fail early. Add a test for each fix. Permissive licences
  only.
- Commits: no AI attribution and no Co-Authored-By lines. Don't push; the planner merges and
  pushes.
- Final report, 15 lines or fewer: commit hashes, the test commands and their results, and open
  questions.
