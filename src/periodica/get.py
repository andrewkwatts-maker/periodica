"""Generic composition + named-registry layer for periodica.

`Get(spec)` composes a result from fundamentals defined in JSON. `Save(name,
result, tier=...)` persists a composed result; after that, `Get(name)`
returns it. The composer is content-agnostic: it walks data files, sums
additive scalar properties, and knows nothing about specific elements or
particles.

Registry tiers
--------------
Every JSON-backed entity belongs to a *tier*:

  fundamentals  - data/active/quarks/      (quarks, leptons, gauge bosons)
  subatomic     - data/active/subatomic/   (curated hadrons)
  atoms         - data/derived/atoms/      (built by build_periodic_table)
  molecules     - data/derived/molecules/  (built by build_molecules)
  hadrons_gen   - data/derived/hadrons/    (built by build_hadrons)
  ...           - any other data/derived/<tier>/ subfolder works

Lookup
------
- `Get(spec, from_="subatomic")` resolves constituent symbols ONLY against
  the named tier. Use this in generators to avoid cross-level collisions
  (e.g. "P" must mean Proton when building atoms, not Phosphorus).
- `Get(spec)` with no `from_` searches all tiers, with priority:
    alias-active > alias-derived > Symbol-derived > Symbol-active >
    stem-derived > stem-active.
  Exact-case matches first; casefold fallback consulted second.

This means tier-1 derived entries can shadow tier-0 fundamentals by Symbol
(saving `H` for Hydrogen overrides Higgs Boson's `H` Symbol), while explicit
aliases on hand-curated fundamentals always win (alias `P` on Proton beats
Phosphorus's `P` Symbol).
"""
from __future__ import annotations

import json
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class Scope(str, Enum):
    """Friendly enum for the standard registry tiers.

    Use as the optional first positional arg to `Get()` to constrain
    constituent resolution to one tier:

        Get(Scope.SubAtomic, "P")   -> Proton (resolved in `subatomic` tier)
        Get(Scope.Atom, "P")        -> Phosphorus (resolved in `atoms` tier)

    Singular and plural forms are accepted for ergonomics. Raw tier-name
    strings ("subatomic", "atoms", ...) and lists of tiers are also accepted
    by `Get()`, so this enum is convenience, not the only way.
    """

    Fundamentals = "fundamentals"
    Fundamental = "fundamentals"
    SubAtomic = "subatomic"
    Subatomic = "subatomic"
    Atom = "atoms"
    Atoms = "atoms"
    Molecule = "molecules"
    Molecules = "molecules"
    HadronsGen = "hadrons_gen"
    Isotope = "isotopes"
    Isotopes = "isotopes"
    Ion = "ions"
    Ions = "ions"
    Alloy = "alloys"
    Alloys = "alloys"
    Polymer = "polymers"
    Polymers = "polymers"
    Ceramic = "ceramics"
    Ceramics = "ceramics"
    Composite = "composites"
    Composites = "composites"
    AminoAcid = "amino_acids"
    AminoAcids = "amino_acids"
    Protein = "proteins"
    Proteins = "proteins"

    def __str__(self) -> str:  # so Scope.Atom prints as 'atoms'
        return self.value

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_PATH = _DATA_DIR / "config" / "composition_rules.json"


def _load_config() -> dict:
    """Read composition rules from data/config/composition_rules.json.

    No fallback hardcoding: if the config is missing, registry build fails.
    """
    text = _CONFIG_PATH.read_text(encoding="utf-8")
    text = re.sub(r"//.*?(?=\n|$)", "", text)
    return json.loads(text)


def _config():
    return _load_config()


def _tier_sources() -> Tuple[Tuple[str, Path, bool], ...]:
    cfg = _config()
    out = []
    for entry in cfg.get("tier_definitions", []):
        name = entry["name"]
        is_active = bool(entry.get("is_active", False))
        src = _DATA_DIR / entry["source"]
        out.append((name, src, is_active))
    return tuple(out)


def _derived_dir() -> Path:
    return _DATA_DIR / _config().get("derived_root", "derived")


def _additive_props() -> Tuple[str, ...]:
    return tuple(_config().get("additive_properties", ()))


def _placeholder_prefixes() -> Tuple[str, ...]:
    return tuple(_config().get("placeholder_prefixes", ()))

_PRIORITY_ALIAS_ACTIVE = 5
_PRIORITY_ALIAS_DERIVED = 4
_PRIORITY_SYMBOL_DERIVED = 3
_PRIORITY_STEM_ACTIVE = 2
_PRIORITY_SYMBOL_ACTIVE = 1
_PRIORITY_STEM_DERIVED = 0


class UnknownConstituent(KeyError):
    """Spec token did not match any registry entry."""


class UnknownName(KeyError):
    """`Get(name)` could not find a saved entry."""


class UnknownTier(KeyError):
    """`Get(spec, from_=...)` referenced a tier that has no entries."""


class RegistryCollision(RuntimeError):
    """Two entries within the same tier claim the same key."""


_registry_cache: Optional[dict] = None
_registry_lock = threading.Lock()


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//.*?(?=\n|$)", "", text)
    return json.loads(text)


def _entry_keys(
    data: dict, path: Path, is_active: bool
) -> Iterable[Tuple[str, int]]:
    # Accept both PascalCase and snake_case field names so legacy data
    # (amino acids use lowercase 'symbol'/'aliases') registers correctly.
    sym = data.get("Symbol") or data.get("symbol")
    if sym:
        yield (
            str(sym),
            _PRIORITY_SYMBOL_ACTIVE if is_active else _PRIORITY_SYMBOL_DERIVED,
        )
    aliases = data.get("Aliases") or data.get("aliases")
    if isinstance(aliases, (list, tuple)):
        for a in aliases:
            if a:
                yield (
                    str(a),
                    _PRIORITY_ALIAS_ACTIVE if is_active else _PRIORITY_ALIAS_DERIVED,
                )
    yield (
        path.stem,
        _PRIORITY_STEM_ACTIVE if is_active else _PRIORITY_STEM_DERIVED,
    )


class _TierIndex:
    """Per-tier exact and casefold lookup tables."""

    __slots__ = ("exact", "casefold", "priority")

    def __init__(self) -> None:
        self.exact: dict = {}
        self.casefold: dict = {}
        self.priority: dict = {}

    def add(self, key: str, prio: int, data: dict) -> None:
        if self.priority.get(key, -1) < prio:
            self.exact[key] = data
            self.priority[key] = prio
        cf = key.casefold()
        cf_key = ("cf", cf)
        if self.priority.get(cf_key, -1) < prio:
            self.casefold[cf] = data
            self.priority[cf_key] = prio


class _Registry:
    """Container for all tier indices + a global merged index."""

    __slots__ = ("by_tier", "merged")

    def __init__(self) -> None:
        self.by_tier: dict = {}
        self.merged = _TierIndex()

    def lookup(self, sym: str, *, from_: Optional[Union[str, Sequence[str]]] = None) -> Optional[dict]:
        if from_ is None:
            entry = self.merged.exact.get(sym)
            if entry is not None:
                return entry
            return self.merged.casefold.get(sym.casefold())
        tiers = [from_] if isinstance(from_, str) else list(from_)
        for t in tiers:
            idx = self.by_tier.get(t)
            if idx is None:
                continue
            entry = idx.exact.get(sym)
            if entry is not None:
                return entry
            entry = idx.casefold.get(sym.casefold())
            if entry is not None:
                return entry
        return None


def _index_files(reg: _Registry, tier_name: str, dir_path: Path, is_active: bool) -> None:
    if not dir_path.is_dir():
        return
    tier_idx = reg.by_tier.setdefault(tier_name, _TierIndex())
    seen_in_tier: dict = {}
    for path in sorted(dir_path.glob("*.json")):
        if any(path.stem.lower().startswith(p) for p in _placeholder_prefixes()):
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        for key, prio in _entry_keys(data, path, is_active):
            existing = seen_in_tier.get(key)
            if existing is not None and existing != path:
                # Within-tier duplicate (e.g. AsparticAcid.json + Aspartic_Acid.json
                # both claiming Symbol 'D'). First sorted occurrence wins; later
                # duplicates are silently skipped for robustness.
                continue
            seen_in_tier[key] = path
            tier_idx.add(key, prio, data)
            reg.merged.add(key, prio, data)


def _build_registry() -> _Registry:
    reg = _Registry()
    for tier_name, src_dir, is_active in _tier_sources():
        _index_files(reg, tier_name, src_dir, is_active)
    derived_dir = _derived_dir()
    if derived_dir.is_dir():
        for sub in sorted(derived_dir.iterdir()):
            if not sub.is_dir():
                continue
            _index_files(reg, sub.name, sub, is_active=False)
    return reg


def _registry() -> _Registry:
    global _registry_cache
    if _registry_cache is None:
        with _registry_lock:
            if _registry_cache is None:
                _registry_cache = _build_registry()
    return _registry_cache


def reload_registry() -> None:
    """Drop the cached registry; next access re-walks the data directories."""
    global _registry_cache
    with _registry_lock:
        _registry_cache = None


def list_tiers() -> List[str]:
    """Return all tier names currently present in the registry."""
    return sorted(_registry().by_tier.keys())


_SPEC_TOKEN = re.compile(r"\s*([^\s=:,{}]+)\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*")


def _parse_spec_string(spec: str) -> dict:
    s = spec.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        return {}
    out: dict = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _SPEC_TOKEN.fullmatch(chunk)
        if not m:
            raise ValueError(
                f"Cannot parse spec fragment {chunk!r}; "
                f"expected 'Symbol=count' or 'Symbol:count'"
            )
        sym, count_s = m.group(1), m.group(2)
        n = float(count_s)
        n_int = int(n)
        out[sym] = n_int if n_int == n else n
    return out


def _is_bare_name(s: str) -> bool:
    s = s.strip()
    return bool(s) and not any(c in s for c in "{}=:,")


def _resolve(
    sym: str,
    reg: _Registry,
    from_: Optional[Union[str, Sequence[str]]],
) -> dict:
    entry = reg.lookup(sym, from_=from_)
    if entry is None:
        if from_ is not None:
            raise UnknownConstituent(
                f"{sym!r} not found in tier(s) {from_!r}"
            )
        raise UnknownConstituent(sym)
    return entry


def _compose(
    spec_map: Mapping[str, Union[int, float]],
    reg: _Registry,
    from_: Optional[Union[str, Sequence[str]]],
) -> dict:
    constituents = []
    additive = _additive_props()
    totals = {p: 0.0 for p in additive}

    for sym, count in spec_map.items():
        entry = _resolve(sym, reg, from_)
        constituents.append(
            {
                "Symbol": sym,
                "Count": count,
                "Resolved": entry.get("Name", sym),
            }
        )
        for prop in additive:
            v = entry.get(prop)
            if v is None:
                continue
            try:
                totals[prop] += float(v) * float(count)
            except (TypeError, ValueError):
                pass

    cleaned: dict = {}
    for prop, total in totals.items():
        if abs(total - round(total)) < 1e-9:
            cleaned[prop] = int(round(total))
        else:
            cleaned[prop] = total

    return {
        "Composition": dict(spec_map),
        "Constituents": constituents,
        **cleaned,
    }


def _coerce_scope(value) -> Optional[Union[str, List[str]]]:
    """Normalise a scope argument to the form `_Registry.lookup` understands.

    Accepts:
      - None (no constraint)
      - Scope enum member -> its value
      - str (raw tier name)
      - iterable of any of the above
    """
    if value is None:
        return None
    if isinstance(value, Scope):
        return value.value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            if isinstance(v, Scope):
                out.append(v.value)
            elif isinstance(v, str):
                out.append(v)
            else:
                raise TypeError(
                    f"Scope element must be Scope or str, got {type(v).__name__}"
                )
        return out
    raise TypeError(
        f"Scope must be Scope/str/None or a list of those, got {type(value).__name__}"
    )


def _looks_like_scope(value) -> bool:
    """True if the first positional arg is plausibly a scope (not a spec)."""
    if isinstance(value, Scope):
        return True
    if isinstance(value, (list, tuple, set)):
        return all(isinstance(v, (Scope, str)) for v in value) and len(value) > 0
    # A bare string is ambiguous (could be a name or a tier). We treat strings
    # as specs by default; users wanting tier-as-string constraint should pass
    # via `from_=`. The Scope enum is the unambiguous overload form.
    return False


def Get(
    scope_or_spec: Any = None,
    spec: Any = None,
    *,
    from_: Optional[Union[str, Sequence[str], Scope]] = None,
) -> dict:
    """Compose a result from fundamentals, or look up a saved name.

    Two call shapes:

      Get(spec, from_=...)
        Single positional arg = the spec. `from_` (keyword) optionally
        constrains constituent resolution to a tier or list of tiers.

      Get(scope, spec)
        Two positional args. `scope` is a `Scope` enum member (or list of
        them) used to constrain resolution; `spec` is the actual spec.
        Equivalent to `Get(spec, from_=scope)` but reads more naturally
        for one-off disambiguation:

            Get(Scope.SubAtomic, "P")   -> Proton
            Get(Scope.Atom, "P")        -> Phosphorus
            Get(Scope.Molecule, "H2O")  -> Water

    `spec` can be:
      - dict: ``{"P": 1, "N": 0, "E": 1}``
      - string spec: ``"{P=1,N=0,E=1}"`` or ``"{u:2,d:1}"``
      - bare name: ``"H"`` (must already exist in some tier)
    """
    if spec is not None:
        if not _looks_like_scope(scope_or_spec):
            raise TypeError(
                "Get: when called with two positional args, the first must be "
                "a Scope enum member (or a list/tuple of them); "
                f"got {type(scope_or_spec).__name__}. "
                f"For tier-name strings, use the keyword form `Get(spec, from_=...)`."
            )
        if from_ is not None:
            raise TypeError(
                "Get: pass tier constraint either positionally OR via `from_`, not both."
            )
        actual_spec = spec
        actual_from = _coerce_scope(scope_or_spec)
    else:
        actual_spec = scope_or_spec
        actual_from = _coerce_scope(from_)

    if actual_spec is None:
        raise TypeError("Get: spec is required")

    reg = _registry()
    if isinstance(actual_spec, Mapping):
        return _compose(dict(actual_spec), reg, actual_from)
    if isinstance(actual_spec, str):
        s = actual_spec.strip()
        if _is_bare_name(s):
            entry = reg.lookup(s, from_=actual_from)
            if entry is None:
                raise UnknownName(s)
            return dict(entry)
        return _compose(_parse_spec_string(s), reg, actual_from)
    raise TypeError(f"Get: unsupported spec type {type(actual_spec).__name__}")


def Save(
    name: str,
    composed: Mapping[str, Any],
    *,
    tier: str = "general",
    dir: Optional[Path] = None,
    allow_overwrite: bool = True,
) -> Path:
    """Persist `composed` as `data/derived/<tier>/<name>.json`.

    `tier` is the registry tier the entry will belong to. Pass `dir=` to
    override the location (e.g. for unit tests writing to tmp_path).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Save: name must be a non-empty string")
    name = name.strip()
    if not isinstance(tier, str) or not tier.strip():
        raise ValueError("Save: tier must be a non-empty string")
    tier = tier.strip()
    out_dir = Path(dir) if dir is not None else _derived_dir() / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.json"
    if path.exists() and not allow_overwrite:
        raise FileExistsError(path)

    payload = dict(composed)
    payload.setdefault("Name", name)
    # Note: we deliberately do NOT auto-default Symbol. Setting Symbol
    # registers an entry at high registry priority (symbol_derived > stem_active),
    # which would let derived entries (e.g. molecules.json's "Alanine") shadow
    # curated active entries (e.g. amino_acids/Alanine.json). Save honours an
    # explicit "Symbol" if the caller passes it (used by build_periodic_table
    # to deliberately shadow Higgs Boson "H" with Hydrogen).

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    reload_registry()
    return path
