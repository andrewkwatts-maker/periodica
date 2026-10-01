#====== periodica/src/periodica/_dispatch.py ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

# Rust impl: rust/periodica_core/src/pyfacade.rs
"""Route public API functions to the Rust backend, and prove that it happened.

Why this module was rewritten
-----------------------------
The previous implementation wrapped every Rust call in a bare
``except Exception: pass`` and fell through to Python. That is indistinguishable
from success:

- ``sample.rs`` read a schema no datasheet uses, so it raised for *every* real
  entry. Python silently did all the work.
- ``_HAS_RUST`` was ``True``, so the library reported the Rust backend as
  active while none of it ran.
- CI never built the extension, so the parity tests were skipped and the suite
  passed green while asserting nothing.

A backend you cannot prove is running is not a backend. This module now
distinguishes three things that the old code collapsed into one:

1. **The extension is absent.** Expected on a pure-Python install; falls back.
2. **Rust deliberately declined** this input (raises
   :class:`FallbackRequired`) -- for example ``backbone_path`` sampling at a 3D
   point, which needs a folded backbone. Falls back, and is counted.
3. **Rust failed.** A bug. The exception propagates instead of being swallowed.

Backend selection
-----------------
``PERIODICA_BACKEND`` selects the policy:

======== =========================================================
``auto`` Default. Use Rust when available, fall back silently.
``rust`` Require Rust. Import fails if absent; any fallback raises.
``python`` Ignore the extension entirely. Useful for parity testing.
======== =========================================================

The default stays ``auto`` deliberately: most PyPI users install the pure-Python
sdist, and making ``rust`` the default would break them. The app and the
game-engine paths opt in explicitly.
"""
from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable, Dict, Optional

__all__ = [
    "BackendMode",
    "FallbackRequired",
    "RustBackendUnavailable",
    "assert_rust_backend",
    "backend_mode",
    "backend_report",
    "declare_accelerated",
    "fallback_count",
    "rust_accelerated",
    "_HAS_RUST",
    "_native",
]


class RustBackendUnavailable(RuntimeError):
    """Raised when the Rust backend is required but is not usable."""


class FallbackRequired(Exception):
    """Raised by a dispatch wrapper when Rust declined an input on purpose.

    This is the *only* exception the dispatcher will convert into a fallback.
    Anything else is a bug in the Rust backend and must surface.
    """


class BackendMode:
    """Valid values of ``PERIODICA_BACKEND``."""

    AUTO = "auto"
    RUST = "rust"
    PYTHON = "python"
    ALL = (AUTO, RUST, PYTHON)


def _read_mode() -> str:
    raw = os.environ.get("PERIODICA_BACKEND", BackendMode.AUTO).strip().lower()
    if raw not in BackendMode.ALL:
        raise RustBackendUnavailable(
            f"PERIODICA_BACKEND={raw!r} is not recognised; "
            f"expected one of {', '.join(BackendMode.ALL)}"
        )
    return raw


_MODE: str = _read_mode()

# ── Extension import ─────────────────────────────────────────────────────

_native = None
_HAS_RUST: bool = False
_IMPORT_ERROR: Optional[BaseException] = None

if _MODE != BackendMode.PYTHON:
    try:
        import periodica._periodica_core as _native  # type: ignore[import-not-found]

        _HAS_RUST = True
    except ImportError as exc:  # pragma: no cover - depends on build config
        _IMPORT_ERROR = exc


def _check_version_handshake() -> Optional[str]:
    """Return a description of a version mismatch, or ``None`` if consistent.

    ``CLAUDE.md`` requires the version in ``pyproject.toml``, the crate's
    ``Cargo.toml``, ``__init__.py`` and the git tag to move together. A stale
    compiled extension left in the source tree is easy to miss and produces
    baffling behaviour, so check rather than assume.
    """
    if not _HAS_RUST or _native is None:
        return None
    rust_version = getattr(_native, "__version__", None)
    if rust_version is None:
        return "the extension does not report a __version__"
    from periodica import __version__ as py_version  # local import: cycle

    if str(rust_version) != str(py_version):
        return (
            f"extension version {rust_version!r} does not match the Python "
            f"package version {py_version!r}; rebuild with "
            f"`maturin develop --features python`"
        )
    return None


# `rust` mode must fail loudly and immediately rather than degrading.
if _MODE == BackendMode.RUST and not _HAS_RUST:
    raise RustBackendUnavailable(
        "PERIODICA_BACKEND=rust but the compiled extension "
        "`periodica._periodica_core` could not be imported "
        f"({_IMPORT_ERROR}). Build it with `maturin develop --features python`, "
        "or install with `pip install periodica[rust]`."
    )

# ── Fallback accounting ──────────────────────────────────────────────────

_counter_lock = threading.Lock()
_fallback_counts: Dict[str, int] = {}


def _record_fallback(fn_name: str, reason: str) -> None:
    """Note that `fn_name` ran in Python instead of Rust.

    In ``rust`` mode this is fatal: the whole point of that mode is that the
    Python paths do not execute.
    """
    if _MODE == BackendMode.RUST:
        raise RustBackendUnavailable(
            f"PERIODICA_BACKEND=rust but {fn_name!r} fell back to Python: {reason}"
        )
    with _counter_lock:
        _fallback_counts[fn_name] = _fallback_counts.get(fn_name, 0) + 1


def fallback_count(fn_name: Optional[str] = None) -> int:
    """Number of Python fallbacks recorded, in total or for one function."""
    with _counter_lock:
        if fn_name is None:
            return sum(_fallback_counts.values())
        return _fallback_counts.get(fn_name, 0)


def reset_fallback_counts() -> None:
    """Clear the fallback counters. Intended for tests."""
    with _counter_lock:
        _fallback_counts.clear()


def backend_mode() -> str:
    """The active backend mode (see :class:`BackendMode`)."""
    return _MODE


def assert_rust_backend() -> None:
    """Raise :class:`RustBackendUnavailable` unless the Rust backend is usable.

    Call this at the top of any program that depends on Rust-accelerated
    evaluation -- a game-engine host, a benchmark, the designer app -- so a
    missing or stale extension fails at startup rather than quietly costing
    100x at runtime.
    """
    if _MODE == BackendMode.PYTHON:
        raise RustBackendUnavailable(
            "PERIODICA_BACKEND=python explicitly disables the Rust backend"
        )
    if not _HAS_RUST or _native is None:
        raise RustBackendUnavailable(
            "the compiled extension `periodica._periodica_core` is not available "
            f"({_IMPORT_ERROR}). Build it with `maturin develop --features python`, "
            "or install with `pip install periodica[rust]`."
        )
    mismatch = _check_version_handshake()
    if mismatch is not None:
        raise RustBackendUnavailable(mismatch)


def backend_report() -> Dict[str, Any]:
    """Describe the backend, for diagnostics and the app's status bar."""
    from periodica import __version__ as py_version

    accelerated = sorted(_REGISTERED)
    return {
        "mode": _MODE,
        "has_rust": _HAS_RUST,
        "rust_version": getattr(_native, "__version__", None) if _HAS_RUST else None,
        "python_version": py_version,
        "version_mismatch": _check_version_handshake(),
        "import_error": None if _IMPORT_ERROR is None else str(_IMPORT_ERROR),
        "accelerated_functions": accelerated,
        "live_functions": sorted(n for n in accelerated if _is_live(n)),
        "fallback_counts": dict(_fallback_counts),
    }


# Python function name -> Rust symbol it dispatches to.
_REGISTERED: Dict[str, str] = {}


def declare_accelerated(py_name: str, rust_fn_name: str) -> None:
    """Declare that `py_name` has a Rust path, before it is ever called.

    :func:`call_rust` registers lazily, which would make
    :func:`backend_report` claim nothing is accelerated until the first call.
    Modules using :func:`call_rust` should declare their functions at import
    time so the report is truthful from the start.
    """
    _REGISTERED[py_name] = rust_fn_name


def _is_live(py_name: str) -> bool:
    """Whether `py_name` would actually dispatch to Rust right now."""
    if not _HAS_RUST or _native is None:
        return False
    return getattr(_native, _REGISTERED.get(py_name, ""), None) is not None


def rust_accelerated(rust_fn_name: str) -> Callable[[Callable], Callable]:
    """Dispatch to the named Rust function, falling back only on purpose.

    Usage::

        @rust_accelerated("py_sample")
        def sample(...):
            # Python implementation, used when Rust is unavailable or declines.
            ...

    Unlike the previous version, a Rust exception is **not** swallowed. Only
    :class:`FallbackRequired` causes a fallback; every other exception
    propagates, so a broken Rust path is visible instead of being silently
    100x slower.
    """

    def decorator(py_fn: Callable) -> Callable:
        _REGISTERED[py_fn.__name__] = rust_fn_name

        @functools.wraps(py_fn)
        def wrapper(*args, **kwargs):
            if _HAS_RUST and _native is not None:
                fn = getattr(_native, rust_fn_name, None)
                if fn is None:
                    _record_fallback(
                        py_fn.__name__,
                        f"the extension exports no {rust_fn_name!r}",
                    )
                else:
                    try:
                        return fn(*args, **kwargs)
                    except FallbackRequired as exc:
                        _record_fallback(py_fn.__name__, str(exc) or "declined by Rust")
            else:
                _record_fallback(py_fn.__name__, "extension not available")
            return py_fn(*args, **kwargs)

        return wrapper

    return decorator


def call_rust(
    rust_fn_name: str,
    *args,
    fallback: Callable[[], Any],
    py_name: Optional[str] = None,
    declines: tuple = (),
):
    """Call a Rust function directly, with the same no-swallowing contract.

    For call sites that cannot use the decorator because only *some* argument
    shapes have a Rust path -- ``sample()`` accepts either a name or a
    pre-fetched dict, and only the former is accelerated.

    `declines` lists exception types that mean "Rust legitimately cannot handle
    this input", and are treated like :class:`FallbackRequired`. Everything else
    propagates.
    """
    name = py_name or rust_fn_name
    _REGISTERED.setdefault(name, rust_fn_name)

    if not _HAS_RUST or _native is None:
        _record_fallback(name, "extension not available")
        return fallback()

    fn = getattr(_native, rust_fn_name, None)
    if fn is None:
        _record_fallback(name, f"the extension exports no {rust_fn_name!r}")
        return fallback()

    try:
        return fn(*args)
    except FallbackRequired as exc:
        _record_fallback(name, str(exc) or "declined by Rust")
        return fallback()
    except declines as exc:  # type: ignore[misc]
        _record_fallback(name, f"{type(exc).__name__}: {exc}")
        return fallback()
