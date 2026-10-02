#====== periodica/tests/test_rust_python_parity.py ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

"""Prove the Rust backend agrees with Python -- or that it is honestly absent.

What this file replaced
-----------------------
The previous version was 58 lines guarded by
``skipif(not _HAS_RUST)``, and CI never built the extension, so **every test in
it was skipped on every run**. The few assertions it did make were of the form
``if val is not None: assert val > 0``, which cannot fail. The suite was green
and asserted nothing while ``sample.rs`` was returning an error for every real
datasheet.

The tests below are split into two groups:

* **Always run.** Backend plumbing: mode selection, strict mode, fallback
  accounting, the version handshake. These need no compiled extension and catch
  regressions in the dispatch layer itself -- the layer whose failure was
  invisible before.
* **Run when the extension is present.** Table-driven value parity across the
  registry. These are ``skipif``-guarded of necessity, but
  :func:`test_ci_builds_the_extension` fails loudly in CI if the extension is
  missing, so the skip can no longer hide a broken backend.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

import periodica
from periodica import _dispatch
from periodica._dispatch import _HAS_RUST

needs_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="compiled extension not built in this environment"
)

# Whether the extension is *installed*, independent of this process's mode.
# `_HAS_RUST` is False under PERIODICA_BACKEND=python even when the extension
# is built, because that mode never imports it; tests that spawn a child with
# a different mode must ask the import system instead.
_EXTENSION_BUILT = importlib.util.find_spec("periodica._periodica_core") is not None


# ─────────────────────────────────────────────────────────────────────────
# Backend plumbing -- always runs
# ─────────────────────────────────────────────────────────────────────────


def test_backend_mode_defaults_to_auto():
    assert periodica.backend_mode() in _dispatch.BackendMode.ALL
    if "PERIODICA_BACKEND" not in os.environ:
        assert periodica.backend_mode() == _dispatch.BackendMode.AUTO


def test_backend_report_is_populated_before_any_call():
    # Registration used to be lazy, so the report claimed nothing was
    # accelerated until a function had already been used once.
    report = periodica.backend_report()
    assert set(report["accelerated_functions"]) >= {"sample", "data_sheet"}
    assert report["mode"] in _dispatch.BackendMode.ALL
    assert report["python_version"] == periodica.__version__
    assert isinstance(report["has_rust"], bool)


def test_assert_rust_backend_matches_has_rust():
    if _HAS_RUST:
        periodica.assert_rust_backend()
    else:
        with pytest.raises(_dispatch.RustBackendUnavailable):
            periodica.assert_rust_backend()


def test_invalid_backend_mode_is_rejected():
    out = subprocess.run(
        [sys.executable, "-c", "import periodica"],
        env={**os.environ, "PERIODICA_BACKEND": "banana"},
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "not recognised" in out.stderr


def test_python_mode_disables_the_extension_entirely():
    code = (
        "import periodica;"
        "from periodica._dispatch import _HAS_RUST;"
        "print('HAS_RUST', _HAS_RUST);"
        "print('MODE', periodica.backend_mode())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PERIODICA_BACKEND": "python"},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "HAS_RUST False" in out.stdout
    assert "MODE python" in out.stdout


def test_strict_mode_refuses_to_start_without_the_extension():
    """`PERIODICA_BACKEND=rust` must fail loudly, never degrade silently."""
    out = subprocess.run(
        [sys.executable, "-c", "import periodica"],
        env={**os.environ, "PERIODICA_BACKEND": "rust"},
        capture_output=True,
        text=True,
    )
    if _EXTENSION_BUILT:
        assert out.returncode == 0, out.stderr
    else:
        assert out.returncode != 0, "strict mode silently accepted a missing backend"
        assert "RustBackendUnavailable" in out.stderr


def test_fallback_is_counted_not_hidden():
    """A Python fallback must leave a trace.

    The old dispatcher swallowed Rust exceptions with `except Exception: pass`,
    so a completely dead backend was indistinguishable from a working one.
    """
    _dispatch.reset_fallback_counts()
    before = periodica.fallback_count()
    periodica.sample("Steel-1018", "Density_kgm3")
    after = periodica.fallback_count()
    if _HAS_RUST:
        assert after == before, "Rust is available but the call still fell back"
    else:
        assert after > before, "fallback happened but was not recorded"


def test_rust_exceptions_are_not_swallowed(monkeypatch):
    """Only FallbackRequired may become a fallback; other errors propagate.

    The mode is pinned rather than inherited from the environment: what a
    decline does depends on it (fall back in `auto`, raise in `rust`), and
    both behaviours are asserted below whatever PERIODICA_BACKEND the suite
    runs under.
    """
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        return "fell-back"

    class Exploder:
        def __call__(self, *_args):
            raise ValueError("backend bug")

    class Decliner:
        def __call__(self, *_args):
            raise _dispatch.FallbackRequired("no Rust path for this input")

    monkeypatch.setattr(_dispatch, "_HAS_RUST", True)
    monkeypatch.setattr(_dispatch, "_MODE", _dispatch.BackendMode.AUTO)

    monkeypatch.setattr(_dispatch, "_native", type("N", (), {"py_thing": Exploder()})())
    with pytest.raises(ValueError, match="backend bug"):
        _dispatch.call_rust("py_thing", fallback=boom, py_name="thing")
    assert calls["n"] == 0, "a backend bug was silently absorbed"

    # A deliberate decline, by contrast, does fall back...
    monkeypatch.setattr(_dispatch, "_native", type("N", (), {"py_thing": Decliner()})())
    assert _dispatch.call_rust("py_thing", fallback=boom, py_name="thing") == "fell-back"
    assert calls["n"] == 1

    # ...except in strict mode, where any fallback is fatal.
    monkeypatch.setattr(_dispatch, "_MODE", _dispatch.BackendMode.RUST)
    with pytest.raises(_dispatch.RustBackendUnavailable, match="fell back"):
        _dispatch.call_rust("py_thing", fallback=boom, py_name="thing")
    assert calls["n"] == 1, "strict mode ran the Python path anyway"


@needs_rust
def test_extension_version_matches_the_package():
    """CLAUDE.md requires all four version locations to move together."""
    assert periodica.backend_report()["version_mismatch"] is None


# ─────────────────────────────────────────────────────────────────────────
# CI guard
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not os.environ.get("CI"), reason="only enforced in CI")
def test_ci_builds_the_extension():
    """In CI the extension must exist, so the parity tests cannot be skipped.

    This is the test that stops the situation this file was written to fix:
    a green suite that silently skipped everything meaningful.
    """
    assert _HAS_RUST, (
        "CI ran without building the Rust extension, so every parity test "
        "below was skipped. Add `maturin develop` (features from pyproject.toml) to the job."
    )


# ─────────────────────────────────────────────────────────────────────────
# Value parity -- needs the extension
# ─────────────────────────────────────────────────────────────────────────

REL_TOL = 1e-9


def _entries(per_tier: int = 5, limit: int = 60) -> list[str]:
    """A spread of real registry names across every tier.

    `registry.by_tier[tier]` is a `_TierIndex`, not a dict -- it is not
    iterable, and its names live in `.exact`. Getting this wrong silently
    yielded three entries instead of sixty, which is why this helper asserts
    on its own output rather than trusting a bare `except: continue`.
    """
    from periodica.get import _build_registry

    registry = _build_registry()
    names: list[str] = []
    for tier, index in sorted(registry.by_tier.items()):
        exact = getattr(index, "exact", None)
        if not exact:
            continue
        # A name can appear in several tiers (Fe is both an atom and an
        # element), so dedupe while preserving order.
        for n in sorted(exact)[:per_tier]:
            if n not in names:
                names.append(n)

    # Always include the entries with the richest field models.
    for extra in ("Steel-1018", "Fe", "H2O"):
        if extra not in names:
            names.append(extra)

    assert len(names) >= 20, (
        f"only {len(names)} registry names collected from "
        f"{len(registry.by_tier)} tiers -- the registry API has changed"
    )
    return names[:limit]


def test_entries_helper_covers_many_tiers():
    """Guard the guard: a broken helper would silently shrink every parity test."""
    names = _entries()
    assert len(names) >= 20, names
    assert len(set(names)) == len(names), "duplicate names in the sample"


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        scale = max(abs(a), abs(b), 1e-300)
        return abs(a - b) / scale <= REL_TOL
    return a == b


def _resolution_divergences() -> list[str]:
    """Names where the two `Get()` implementations resolve to different entries.

    This should be **empty**. It was not: `get.rs::resolve_named` used to try
    exact stem, then casefold stem, then Symbol, then Aliases, while Python
    builds a priority-resolved index (`_PRIORITY_ALIAS_ACTIVE=5` down to
    `_PRIORITY_STEM_DERIVED=0`). The orderings were near-opposite, so the two
    disagreed on ambiguous short names -- `Get("E")` gave glutamic acid in Rust
    and the electron in Python.

    Fixed by porting Python's index into `crate::registry`. This helper is kept
    as a standing regression check: any name that starts resolving differently
    will show up here.
    """
    from periodica.get import Get

    diverged = []
    for name in _entries():
        try:
            rust = periodica.data_sheet(name)
            py = dict((Get(name).get("Properties") or {}))
            if not py:
                py = {
                    k: v
                    for k, v in Get(name).items()
                    if isinstance(v, (int, float)) and v is not None
                }
        except Exception:
            continue
        if set(rust) != set(py):
            diverged.append(name)
    return diverged


@needs_rust
def test_name_resolution_agrees_between_backends():
    """`Get()` must resolve every name identically in both backends."""
    diverged = _resolution_divergences()
    assert diverged == [], (
        "Rust and Python resolve these names to different entries: "
        f"{diverged}. `crate::registry` ports get.py's priority table; a "
        "divergence here means the two indexes have drifted apart again."
    )


@needs_rust
def test_the_e_collision_resolves_to_the_electron():
    """Regression: the specific name that exposed the priority mismatch.

    `Electron` (Symbol "E", fundamentals) and `GlutamicAcid` (symbol "E",
    amino_acids) both claim "E" at PRIORITY_SYMBOL_ACTIVE. First-writer-wins
    plus tier order gives it to the electron.
    """
    entry = periodica.Get("E")
    assert entry.get("Name") == "Electron", entry.get("Name")
    # Scoping still reaches the amino acid. Note the argument order: `Get`
    # takes the Scope first when two positional args are given, and amino-acid
    # datasheets carry no "Name" field, so assert on one they do have.
    scoped = periodica.Get(periodica.Scope.AminoAcid, "E")
    assert scoped.get("symbol") == "E", scoped.get("symbol")
    assert "can_form_disulfide" in scoped, sorted(scoped)[:8]
    assert scoped != entry, "scoped and unscoped lookups returned the same entry"


@needs_rust
def test_data_sheet_matches_between_backends():
    from periodica.sample import _data_sheet_python
    from periodica.get import Get

    # Empty in a healthy tree; retained so a resolution regression surfaces as
    # its own failure rather than corrupting this test's property comparison.
    skip = set(_resolution_divergences())
    checked = 0
    for name in _entries():
        if name in skip:
            continue
        try:
            rust = periodica.data_sheet(name)
            py = _data_sheet_python(Get(name))
        except Exception:
            continue
        assert set(rust) == set(py), f"{name}: key sets differ"
        for key in py:
            assert _close(rust[key], py[key]), f"{name}.{key}: {rust[key]} != {py[key]}"
        checked += 1
    assert checked >= 10, f"only {checked} entries were comparable"


@needs_rust
def test_sample_matches_between_backends_across_properties():
    """The real parity gate: N entries x M properties, both backends."""
    from periodica.sample import _sample_python
    from periodica.get import Get

    points = [None, (0.0, 0.0, 0.0), (1e-5, -2e-5, 3e-5), (0.01, 0.02, 0.03)]
    scales = [None, 1e-9, 1e-3]

    comparisons = 0
    for name in _entries():
        try:
            entry = Get(name)
        except Exception:
            continue
        props = list((entry.get("Properties") or {}))[:20]
        if not props:
            props = [k for k, v in entry.items() if isinstance(v, (int, float))][:20]
        for prop in props:
            for at in points:
                for scale in scales:
                    try:
                        py = _sample_python(entry, prop, at, scale)
                    except Exception:
                        continue
                    try:
                        rust = periodica.sample(name, prop, at=at, scale_m=scale)
                    except Exception as exc:  # a Rust bug, not a decline
                        pytest.fail(f"{name}.{prop} at={at} scale={scale}: {exc}")
                    assert _close(rust, py), (
                        f"{name}.{prop} at={at} scale={scale}: "
                        f"rust={rust!r} python={py!r}"
                    )
                    comparisons += 1
    assert comparisons >= 200, f"only {comparisons} comparisons ran"


@needs_rust
def test_no_silent_fallbacks_in_strict_mode():
    """With the extension present, strict mode must complete real work."""
    code = (
        "import periodica;"
        "periodica.assert_rust_backend();"
        "periodica.sample('Steel-1018','Density_kgm3');"
        "periodica.data_sheet('Fe');"
        "assert periodica.fallback_count() == 0, periodica.backend_report();"
        "print('STRICT-OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PERIODICA_BACKEND": "rust"},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "STRICT-OK" in out.stdout


@needs_rust
def test_missing_property_is_none_without_a_fallback():
    """Python returns None for a property the entry lacks; Rust must too.

    It used to raise KeyError, which the dispatcher reads as a decline: every
    such call quietly fell back in `auto` mode and raised
    RustBackendUnavailable in `rust` mode.
    """
    _dispatch.reset_fallback_counts()
    assert periodica.sample("Steel-1018", "NonExistentProperty_XYZ") is None
    assert periodica.fallback_count("sample") == 0, periodica.backend_report()


@needs_rust
def test_list_tiers_matches_between_backends():
    from periodica._dispatch import _native

    rust_tiers = sorted(_native.py_list_tiers())
    assert rust_tiers == sorted(periodica.list_tiers())
