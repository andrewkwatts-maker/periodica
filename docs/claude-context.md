# periodica: working context

Detail behind CLAUDE.md. Read the section you need.

## Layout

```
periodica/
├── pyproject.toml
├── rust/periodica_core/      # Rust crate (cdylib + lib)
├── src/periodica/            # Python package
│   ├── __init__.py           # public surface + _HAS_RUST guard
│   ├── get.py / sample.py / export.py / folding.py / optimize.py
│   ├── _dispatch.py          # Rust/Python dispatch helper
│   └── _launcher.py          # periodica-app launcher
│   └── data/active/          # material datasheets (JSON)
└── tests/
```

Also `rust/periodica-mat` (typed SI material model, datasheet ingest) and `rust/periodica-runtime`
(crystal SDFs, mass and inertia, weakpoint fracture; no pyo3, no filesystem).

## Versioning — 4 locations must stay in sync

When bumping the package version, **all four locations must be updated together** or the PyPI release will be broken:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` under `[project]` |
| `rust/periodica_core/Cargo.toml` | `version = "X.Y.Z"` under `[package]` |
| `src/periodica/__init__.py` | `__version__ = "X.Y.Z"` |
| Git tag | `vX.Y.Z` pushed to origin |

After committing and tagging, create a **GitHub Release** from the tag to trigger the PyPI publish workflow (`release: published` event). The workflow does **not** fire on a tag push alone.

> **Important:** `pyproject.toml version` is what PyPI indexes — not the git tag. Pre-release suffixes like `2.0.0a0` are **not** matched by `>=2.0.0` in pip. Always use clean `X.Y.Z` for stable releases.

## Owner standards for Python repos

- A GUI uses KivyMD; the headless library (PyPI) and the GUI (periodica-app) live in separate repos.
- `build.bat` plus a shell equivalent; lint (ruff), type-check (mypy or pyright) and test in the build;
  at least 90% coverage before a release.
- Up-to-date How-To docs in `docs/`.
