# R0b: the release workflow builds with Arithma (lane `release`)

Read `_common.md` first. Deliverable: one commit on `claude/release` after which
`.github/workflows/workflow.yml` (cibuildwheel wheels, the maturin sdist, the PyPI publish on a
GitHub Release) can build. Today it fails, because `Cargo.toml` depends on `arithma_core` by the
path `../Arithma/rust/arithma_core` and the workflow checks out periodica only. `ci.yml` already
clones the public Arithma repo beside the checkout (merge 11b2a79); do the equivalent here.

## Steps
1. Wheels: cibuildwheel builds Linux wheels inside a manylinux container, where the project is at
   `/project`, so `../Arithma` is `/Arithma` there. macOS and Windows build on the host. Give every
   platform an Arithma checkout at the path Cargo expects (for example a `CIBW_BEFORE_ALL` step
   that clones to `../Arithma`; keep the existing rustup install on Linux).
2. sdist: clone Arithma beside the checkout before `maturin sdist`. Then find out whether the sdist
   carries `arithma_core` inside it (maturin vendors path dependencies) and whether a wheel builds
   from that sdist alone, with no sibling repo. Build the sdist locally under your scratch
   directory, unpack it and try `pip wheel` on it in a clean directory. Report what happens.
3. Pin what the release builds against: clone a named Arithma tag or commit, not the moving default
   branch, and say in the commit message which one and why. List the Arithma tags first.
4. You can't run GitHub Actions. Say exactly which parts you ran locally and which you couldn't.

Don't publish anything, create a tag or a release, or push.
