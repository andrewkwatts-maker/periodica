# R0: periodica CI green (lane `ci`)

Read `_common.md` first. Deliverable: one commit on `claude/ci` after which every job in
`.github/workflows/ci.yml` and `.github/workflows/workflow.yml` would pass on GitHub.

## Known
- CI has failed on every push since 2026-11-02, most recently on run head 9890238.
- Cause 1: in the root `Cargo.toml` (about line 94), commit 15176dd added
  `arithma_core = { version = "2.0", path = "../Arithma/rust/arithma_core" }`. CI checks out
  periodica only, so the path doesn't exist there.
- Cause 2: `cargo fmt --check` fails.
- `gh` isn't logged in. Use the public API:
  `curl -s https://api.github.com/repos/andrewkwatts-maker/periodica/actions/runs?per_page=5`,
  then `.../runs/<id>/jobs` for the failing step names. Check the run annotations too.

## Steps
1. Find out what uses `arithma_core` in periodica (`git grep -n arithma`). Then pick one fix and
   say why in the commit message:
   - if nothing uses it, remove the dependency;
   - if the matching version is on crates.io, depend on that version;
   - otherwise, have CI check out the public Arithma repo at the path Cargo expects.

   periodica must not depend on Themelios or Topos. Arithma is allowed.
2. Run `cargo fmt --all`. Keep the formatting commit-only; no logic changes.
3. Run every step that both workflow files list, locally and in the same order (fmt, clippy with
   the same flags, cargo test, the Python install, pytest). Report each step with pass/fail and
   counts. If a step is Linux-only, say so.
4. Run `cargo tree -i arithma_core` (or confirm it's gone) and add a test or CI step that would
   catch a sibling-path dependency again. For example, a CI step that fails if any `path = "../`
   dependency in `Cargo.toml` points outside the repo.
5. Commit.
