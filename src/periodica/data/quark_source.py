"""
Quark data source switch — experimental vs simulated.

periodica's quark datasheets can be sourced two ways:

* ``"experimental"`` (default) — read the bundled JSON files in
  ``periodica/data/active/quarks/`` and ``periodica/data/active/antiquarks/``.
  These carry the **PDG / Particle-Data-Group experimental values**: masses
  measured at colliders, branching ratios from decay studies, the canonical
  Standard-Model parameter set every textbook references.

* ``"simulated"`` — call :func:`metaphysica.Get` to fetch the
  **G₂-manifold-derived predicted values** from the metaphysica framework
  (PyPI ``metaphysica >= 1.3.1``). Same field names + the same dict shape
  as the experimental source, plus extra ``pm_prediction`` / ``_provenance``
  blocks carrying the EML expression, the PM-derived value, the percent
  error vs PDG, and CKM couplings. Useful for cross-checking periodica's
  derivation chain against an independent theoretical prediction.

Configuration
-------------

The default is ``experimental``. Switch globally via either:

1. Environment variable::

     PERIODICA_QUARK_SOURCE=simulated python my_script.py

2. Programmatic::

     from periodica.data.quark_source import set_quark_source
     set_quark_source("simulated")

3. Per-call::

     from periodica.data.quark_source import load_quark
     d = load_quark("Up", source="simulated")

Switching to ``"simulated"`` requires the optional ``metaphysica`` package::

    pip install periodica[simulated]

If ``metaphysica`` is not installed, periodica raises a clean
:class:`ImportError` with the install hint and falls back to experimental
on all subsequent calls.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "QuarkSource",
    "EXPERIMENTAL",
    "SIMULATED",
    "get_quark_source",
    "set_quark_source",
    "load_quark",
    "list_quark_names",
    "iter_quarks",
]

# ── Source identifiers ───────────────────────────────────────────────────────

EXPERIMENTAL = "experimental"
SIMULATED = "simulated"
QuarkSource = str  # type alias — kept simple for shell users

_VALID_SOURCES = (EXPERIMENTAL, SIMULATED)
_ENV_VAR = "PERIODICA_QUARK_SOURCE"
_DEFAULT_SOURCE = EXPERIMENTAL

# Mutable session-level override; defaults to env var or EXPERIMENTAL
_session_source: Optional[str] = None


def get_quark_source() -> str:
    """Return the active quark data source.

    Resolution order:
      1. session-level override set via :func:`set_quark_source`
      2. ``PERIODICA_QUARK_SOURCE`` environment variable
      3. ``"experimental"`` (default)
    """
    if _session_source is not None:
        return _session_source
    env = os.environ.get(_ENV_VAR, "").strip().lower()
    if env in _VALID_SOURCES:
        return env
    return _DEFAULT_SOURCE


def set_quark_source(source: str) -> None:
    """Set the active quark data source for the current Python session.

    Parameters
    ----------
    source : {"experimental", "simulated"}
    """
    s = source.strip().lower()
    if s not in _VALID_SOURCES:
        raise ValueError(
            f"quark source must be one of {_VALID_SOURCES!r}, got {source!r}"
        )
    global _session_source
    _session_source = s


# ── Experimental loader (local JSON) ────────────────────────────────────────

_QUARKS_DIR = Path(__file__).parent / "active" / "quarks"
_ANTIQUARKS_DIR = Path(__file__).parent / "active" / "antiquarks"


def _strip_jsonc_comments(text: str) -> str:
    """Remove ``// line comments`` from JSONC text. Block comments not
    used in the active quark JSON files."""
    return re.sub(r"//.*?(?=\n|$)", "", text)


_EXPERIMENTAL_NAME_TO_FILE: Dict[str, Path] = {}


def _index_experimental_files() -> Dict[str, Path]:
    """Build a case-insensitive name → file index for experimental quarks.

    Indexed by:
      * canonical filename stem (``UpQuark`` → ``UpQuark.json``)
      * lowered single-word name (``up`` → ``UpQuark.json``)
      * antiquark prefix (``antiup`` → AntiQuark file in antiquarks/)
    """
    if _EXPERIMENTAL_NAME_TO_FILE:
        return _EXPERIMENTAL_NAME_TO_FILE
    for d, prefix in ((_QUARKS_DIR, ""), (_ANTIQUARKS_DIR, "anti")):
        if not d.exists():
            continue
        for f in d.glob("*Quark.json"):
            stem = f.stem                    # "UpQuark"
            short = stem.replace("Quark", "")  # "Up"
            _EXPERIMENTAL_NAME_TO_FILE[stem.lower()] = f
            _EXPERIMENTAL_NAME_TO_FILE[short.lower()] = f
            _EXPERIMENTAL_NAME_TO_FILE[(prefix + short).lower()] = f
    return _EXPERIMENTAL_NAME_TO_FILE


def _load_experimental(name: str) -> Optional[Dict[str, Any]]:
    """Load experimental (PDG) quark data from bundled JSON. Returns None if
    the name is unknown."""
    idx = _index_experimental_files()
    key = name.strip().lower().replace(" ", "").replace("-", "")
    f = idx.get(key) or idx.get(key + "quark")
    if f is None or not f.exists():
        return None
    with open(f, "r", encoding="utf-8") as h:
        text = _strip_jsonc_comments(h.read())
    data = json.loads(text)
    data.setdefault("_source", EXPERIMENTAL)
    data.setdefault("_source_file", f.name)
    return data


# ── Simulated loader (metaphysica.Get) ──────────────────────────────────────

_SIMULATED_WARN_LOGGED = False


def _load_simulated(name: str) -> Optional[Dict[str, Any]]:
    """Load simulated quark data via :func:`metaphysica.Get`.

    Raises :class:`ImportError` (with install hint) if metaphysica isn't
    installed. Returns ``None`` if the name is unknown to metaphysica.
    """
    try:
        from metaphysica import Get as _meta_get   # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Simulated quark data source requires the 'metaphysica' package. "
            "Install with:  pip install periodica[simulated]   "
            "(or: pip install metaphysica>=1.3.1)"
        ) from exc

    try:
        d = _meta_get(name)
    except KeyError:
        return None
    if not isinstance(d, dict):
        return None

    out = dict(d)
    out["_source"] = SIMULATED
    out["_source_pkg"] = "metaphysica"
    return out


# ── Dispatch ────────────────────────────────────────────────────────────────

def load_quark(name: str, *, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a single quark datasheet by name.

    Parameters
    ----------
    name : str
        Quark name. Accepted aliases (case-insensitive):

        * ``"Up"``, ``"up"``, ``"u"`` (filenames ``UpQuark.json``)
        * ``"AntiUp"``, ``"anti-up"`` (in ``antiquarks/``)
        * Full quark filenames work too: ``"UpQuark"``, ``"DownQuark"``, …

    source : {"experimental", "simulated", None}
        Override the global source for this single call. If ``None`` (default),
        :func:`get_quark_source` is consulted.

    Returns
    -------
    dict or None
        The quark datasheet (with the same Mass_MeVc2 / Charge_e / Spin_hbar /
        … field names regardless of source) or ``None`` if the name is unknown.
        The returned dict carries a ``"_source"`` field tagged with the
        source actually used.
    """
    src = (source or get_quark_source()).strip().lower()
    if src == EXPERIMENTAL:
        return _load_experimental(name)
    if src == SIMULATED:
        return _load_simulated(name)
    raise ValueError(f"unknown quark source: {src!r}")


def list_quark_names(*, source: Optional[str] = None) -> List[str]:
    """List the canonical quark names the active source can resolve."""
    src = (source or get_quark_source()).strip().lower()
    if src == EXPERIMENTAL:
        names: List[str] = []
        for d, prefix in ((_QUARKS_DIR, ""), (_ANTIQUARKS_DIR, "Anti")):
            if d.exists():
                for f in d.glob("*Quark.json"):
                    stem = f.stem
                    # Drop sample / template files
                    if stem.lower().startswith(("example", "demo", "sample")):
                        continue
                    short = stem.replace("Quark", "")
                    names.append(prefix + short)
        return sorted(set(names))
    if src == SIMULATED:
        from metaphysica import list_quarks   # type: ignore
        return list(list_quarks())
    raise ValueError(f"unknown quark source: {src!r}")


def iter_quarks(*, source: Optional[str] = None):
    """Yield ``(name, datasheet)`` pairs for every quark from the active source."""
    for n in list_quark_names(source=source):
        d = load_quark(n, source=source)
        if d is not None:
            yield n, d
