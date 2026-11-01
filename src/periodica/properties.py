"""Canonical property names, units, and flattened data sheets.

Curated JSON in this package was written by many hands over a long time, so
one physical quantity appears under many spellings and in many units::

    active/materials/Chromium_Steel_Hot_rolled.json
        PhysicalProperties.Density_g_cm3      = 7.782
    active/elements/026_Fe.json
        density                               = 7.874
    derived/polymers/HDPE.json
        Properties.Density_kgm3               = 970
    active/biological_materials/Bone.json
        physical_properties.density_g_cm3     = 1.9

This module turns all of those into one canonical name (``density``) in one
SI unit (``kg_m3``) without hardcoding a single material or element. The
vocabulary lives in ``data/config/property_schema.json``; this file is only
the reader.

How a raw key is read
---------------------
1. The longest matching suffix from the schema's ``units`` table is stripped
   off, giving the unit (``Density_g_cm3`` -> unit ``g_cm3``).
2. The remaining stem is slugified: CamelCase and snake_case both collapse to
   lowercase underscore words (``UltimateTensileStrength`` ->
   ``ultimate_tensile_strength``).
3. The slug is looked up in the schema's alias table, giving the canonical
   name (``ultimate_tensile_strength`` -> ``tensile_strength``).
4. A key with no unit suffix takes the property's ``default_unit``, which is
   how bare ``density`` is correctly read as g/cm3 and bare ``melting_point``
   as kelvin.

Public API
----------
``parse_key(key)``        -> ``ParsedKey(canonical, unit, slug, si_unit)``
``flatten(entry)``        -> nested property groups collapsed to one dict
``DataSheet``             -> a dict that also answers alias/unit-converted keys
``sheet(entry)``          -> a ``DataSheet`` over ``flatten(entry)``
``si_sheet(entry)``       -> ``{canonical: value_in_SI}``
``to_si(value, unit)`` / ``from_si(value, unit)``
``validate_sheet(entry)`` -> range/unit sanity issues for an entry
``coverage(entry, props)``-> which canonical properties an entry actually has

Why it matters
--------------
``sample()`` and ``data_sheet()`` used to read only a flat ``Properties``
block, so every curated material and alloy sheet -- which nest their numbers
under ``PhysicalProperties`` / ``MechanicalProperties`` / ... -- sampled as
empty. Flattening here is what makes those sheets usable.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

_SCHEMA_PATH = Path(__file__).parent / "data" / "config" / "property_schema.json"

_schema_cache: Optional[dict] = None
_schema_lock = threading.Lock()

#: Deepest nesting `flatten` will descend into a property group.
MAX_GROUP_DEPTH = 3


# ── Schema loading ──────────────────────────────────────────────────────

def _strip_comments(text: str) -> str:
    return re.sub(r"//.*?(?=\n|$)", "", text)


def _load_schema() -> dict:
    raw = json.loads(_strip_comments(_SCHEMA_PATH.read_text(encoding="utf-8")))

    units: Dict[str, dict] = dict(raw.get("units") or {})
    props: Dict[str, dict] = dict(raw.get("properties") or {})

    # alias slug -> (canonical, alias_unit_override or None)
    alias_index: Dict[str, Tuple[str, Optional[str]]] = {}
    for canonical, spec in props.items():
        entries = list(spec.get("aliases") or ())
        entries.append(canonical)
        for alias in entries:
            if isinstance(alias, Mapping):
                name, unit = alias.get("name"), alias.get("unit")
            else:
                name, unit = alias, None
            if not name:
                continue
            slug = slugify(str(name))
            if not slug:
                continue
            # First definition wins, so a property never silently steals
            # another's alias when the schema is edited.
            alias_index.setdefault(slug, (canonical, unit))

    # Unit suffixes, longest first so `g_cm3` beats `g` and `MPa_sqrt_m`
    # beats `MPa`.
    unit_tokens = sorted(units, key=len, reverse=True)
    unit_tokens_ci = {tok.casefold(): tok for tok in unit_tokens}

    return {
        "version": raw.get("version"),
        "units": units,
        "properties": props,
        "alias_index": alias_index,
        "unit_tokens": tuple(unit_tokens),
        "unit_tokens_ci": unit_tokens_ci,
        "group_keys": tuple(raw.get("property_group_keys") or ()),
        "group_keys_set": frozenset(raw.get("property_group_keys") or ()),
        "excluded_keys": frozenset(raw.get("excluded_group_keys") or ()),
    }


def schema() -> dict:
    """The parsed property schema (cached)."""
    global _schema_cache
    if _schema_cache is None:
        with _schema_lock:
            if _schema_cache is None:
                _schema_cache = _load_schema()
    return _schema_cache


def reload_schema() -> None:
    """Drop the cached schema; next access re-reads the JSON."""
    global _schema_cache
    with _schema_lock:
        _schema_cache = None


def list_properties() -> List[str]:
    """Every canonical property name, sorted."""
    return sorted(schema()["properties"])


def property_spec(canonical: str) -> Optional[dict]:
    """The schema entry for a canonical property, or None."""
    return schema()["properties"].get(canonical)


# ── Key parsing ─────────────────────────────────────────────────────────

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^0-9a-z]+")


def slugify(name: str) -> str:
    """Normalise a property name to lowercase underscore words.

    ``UltimateTensileStrength`` -> ``ultimate_tensile_strength``
    ``youngs_modulus``          -> ``youngs_modulus``
    ``Poisson's Ratio``         -> ``poissons_ratio``
    """
    if not name:
        return ""
    text = _CAMEL_BOUNDARY.sub("_", str(name))
    text = text.replace("'", "").replace("’", "")
    text = _NON_WORD.sub("_", text.lower())
    return text.strip("_")


class ParsedKey(NamedTuple):
    """The meaning read out of a raw data-sheet key."""

    canonical: Optional[str]   # canonical property name, None if unrecognised
    unit: Optional[str]        # unit token the raw value is expressed in
    slug: str                  # slugified stem, with the unit removed
    si_unit: Optional[str]     # SI unit the canonical property normalises to


def _split_unit(key: str) -> Tuple[str, Optional[str]]:
    """Strip a trailing unit token off a raw key.

    Matches case-sensitively first so ``_K`` (kelvin) is not read as ``_k``,
    then falls back to a case-insensitive match for sloppier spellings.
    """
    sch = schema()
    for token in sch["unit_tokens"]:
        for sep in ("_", ""):
            suffix = sep + token
            if len(key) > len(suffix) and key.endswith(suffix):
                return key[: -len(suffix)], token
    low = key.casefold()
    for token_cf, token in sorted(
        sch["unit_tokens_ci"].items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        for sep in ("_", ""):
            suffix = sep + token_cf
            if len(low) > len(suffix) and low.endswith(suffix):
                return key[: -len(suffix)], token
    return key, None


def parse_key(key: str) -> ParsedKey:
    """Read a raw data-sheet key into canonical name + unit."""
    if not key:
        return ParsedKey(None, None, "", None)
    sch = schema()

    # Try with the unit stripped, then the whole key (a property whose name
    # ends in something unit-shaped, e.g. bare `mass`).
    stem, unit = _split_unit(str(key))
    for candidate_slug, candidate_unit in ((slugify(stem), unit),
                                           (slugify(str(key)), None)):
        hit = sch["alias_index"].get(candidate_slug)
        if hit is None:
            continue
        canonical, alias_unit = hit
        spec = sch["properties"][canonical]
        resolved = (
            candidate_unit
            or alias_unit
            or spec.get("default_unit")
            or spec.get("si_unit")
        )
        return ParsedKey(canonical, resolved, candidate_slug, spec.get("si_unit"))

    return ParsedKey(None, unit, slugify(stem), None)


def canonical_of(key: str) -> Optional[str]:
    """Canonical property name for a raw key, or None if unrecognised."""
    return parse_key(key).canonical


# ── Unit conversion ─────────────────────────────────────────────────────

_SIG_FIGS = 12


def _clean(value: float) -> float:
    """Round conversion noise away: 0.97 * 1000 -> 970.0, not 969.9999999."""
    if value == 0 or not _isfinite(value):
        return value
    rounded = float(f"%.{_SIG_FIGS}g" % value)
    nearest = round(rounded)
    if nearest != 0 and abs(rounded - nearest) <= abs(rounded) * 1e-12:
        return float(nearest)
    return rounded


def _isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def unit_spec(unit: str) -> Optional[dict]:
    """The schema entry for a unit token, or None."""
    return schema()["units"].get(unit)


def to_si(value: Any, unit: Optional[str]) -> Optional[float]:
    """Convert `value` from `unit` into the schema's SI unit for it.

    An unknown or absent unit passes the value through unchanged, so callers
    never lose data to a missing table entry.
    """
    num = _as_float(value)
    if num is None:
        return None
    spec = unit_spec(unit) if unit else None
    if spec is None:
        return num
    return _clean(num * float(spec.get("factor", 1.0)) + float(spec.get("offset", 0.0)))


def from_si(value: Any, unit: Optional[str]) -> Optional[float]:
    """Inverse of `to_si`: express an SI value in `unit`."""
    num = _as_float(value)
    if num is None:
        return None
    spec = unit_spec(unit) if unit else None
    if spec is None:
        return num
    factor = float(spec.get("factor", 1.0)) or 1.0
    return _clean((num - float(spec.get("offset", 0.0))) / factor)


def convert(value: Any, from_unit: Optional[str], to_unit: Optional[str]) -> Optional[float]:
    """Convert between two unit tokens of the same SI dimension."""
    if from_unit == to_unit:
        return _as_float(value)
    src, dst = unit_spec(from_unit or ""), unit_spec(to_unit or "")
    if src is not None and dst is not None and src.get("si") != dst.get("si"):
        raise ValueError(
            f"Cannot convert {from_unit!r} ({src.get('si')}) to "
            f"{to_unit!r} ({dst.get('si')}): different dimensions."
        )
    return from_si(to_si(value, from_unit), to_unit)


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


# ── Flattening nested property groups ───────────────────────────────────

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


def flatten(entry: Mapping[str, Any], *, max_depth: int = MAX_GROUP_DEPTH) -> Dict[str, Any]:
    """Collapse an entry's property groups into one flat dict.

    Scalars sitting at the entry root come first, then each recognised
    property group is merged in. Keys already present are never overwritten,
    so a root scalar or an explicit ``Properties`` entry outranks the same
    name appearing deeper in the tree. Group names themselves are dropped:
    ``PhysicalProperties.Density_g_cm3`` flattens to ``Density_g_cm3``.

    Groups listed under ``excluded_group_keys`` in the schema (composition,
    derivation metadata, microstructure blocks) are skipped: they are
    structure, not bulk properties.
    """
    sch = schema()
    group_keys = sch["group_keys_set"]
    excluded = sch["excluded_keys"]
    out: Dict[str, Any] = {}

    def merge(source: Mapping[str, Any], depth: int) -> None:
        for key, value in source.items():
            if key in excluded or str(key).startswith("_"):
                continue
            if _is_scalar(value):
                out.setdefault(key, value)
            elif isinstance(value, Mapping) and depth < max_depth:
                # Recognised group, or any nested dict shallow enough to be
                # a property group rather than a data structure.
                merge(value, depth + 1)

    # Pass 1: root scalars and the explicit Properties block win outright.
    for key, value in entry.items():
        if key in excluded or str(key).startswith("_"):
            continue
        if _is_scalar(value):
            out.setdefault(key, value)
    for key in ("Properties", "properties"):
        block = entry.get(key)
        if isinstance(block, Mapping):
            merge(block, 1)

    # Pass 2: named groups in schema order, then any other shallow dict.
    for key in sch["group_keys"]:
        if key in ("Properties", "properties"):
            continue
        block = entry.get(key)
        if isinstance(block, Mapping):
            merge(block, 1)
    for key, value in entry.items():
        if key in excluded or key in group_keys or str(key).startswith("_"):
            continue
        if isinstance(value, Mapping):
            merge(value, 1)

    return out


# ── DataSheet ───────────────────────────────────────────────────────────

class DataSheet(Dict[str, Any]):
    """A flattened data sheet that also answers alias and unit-converted keys.

    Iteration, ``len`` and ``keys`` see exactly the source keys, so a sheet
    stays faithful to the JSON it came from. Lookup is forgiving::

        sheet = DataSheet({"Density_g_cm3": 7.874})
        sheet["Density_g_cm3"]   # 7.874  (source key, source unit)
        sheet["Density_kgm3"]    # 7874.0 (same quantity, asked for in kg/m3)
        sheet["density"]         # 7.874  (canonical alias, default unit)
        sheet.si("density")      # 7874.0 (canonical, SI)

    Resolution never guesses: a key only matches when both spellings map to
    the same canonical property in the schema.
    """

    __slots__ = ("_by_canonical",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._by_canonical: Dict[str, Tuple[str, Optional[str], Any]] = {}
        for key, value in self.items():
            parsed = parse_key(key)
            if parsed.canonical is None:
                continue
            # First source key for a canonical property wins, matching
            # `flatten`'s precedence.
            self._by_canonical.setdefault(parsed.canonical, (key, parsed.unit, value))

    # -- alias-aware lookup --

    def resolve(self, key: str) -> Tuple[Optional[Any], Optional[str]]:
        """``(value, source_key)`` for `key`, converting units when needed.

        Returns ``(None, None)`` when the sheet has nothing for it.
        """
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key), key
        wanted = parse_key(key)
        if wanted.canonical is None:
            return None, None
        hit = self._by_canonical.get(wanted.canonical)
        if hit is None:
            return None, None
        source_key, source_unit, raw = hit
        if source_unit == wanted.unit:
            return raw, source_key
        try:
            converted = convert(raw, source_unit, wanted.unit)
        except ValueError:
            return None, None
        if converted is None:
            return raw, source_key
        return converted, source_key

    def __getitem__(self, key: str) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        value, source = self.resolve(key)
        if source is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        if dict.__contains__(self, key):
            return True
        if not isinstance(key, str):
            return False
        return self.resolve(key)[1] is not None

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        value, source = self.resolve(key)
        return default if source is None else value

    # -- canonical views --

    def canonical_keys(self) -> List[str]:
        """Canonical property names this sheet carries, sorted."""
        return sorted(self._by_canonical)

    def si(self, key: str) -> Optional[float]:
        """The value of `key` in its SI unit, or None."""
        wanted = parse_key(key)
        if wanted.canonical is None:
            num = _as_float(self.get(key))
            return num
        hit = self._by_canonical.get(wanted.canonical)
        if hit is None:
            return None
        _source_key, source_unit, raw = hit
        return to_si(raw, source_unit)

    def source_of(self, key: str) -> Optional[str]:
        """Which raw key answered `key`, or None."""
        return self.resolve(key)[1]

    def unit_of(self, key: str) -> Optional[str]:
        """The unit the sheet stores `key`'s quantity in."""
        wanted = parse_key(key)
        if wanted.canonical is None:
            return None
        hit = self._by_canonical.get(wanted.canonical)
        return hit[1] if hit else None


def sheet(entry: Mapping[str, Any]) -> DataSheet:
    """A `DataSheet` over an entry's flattened properties."""
    return DataSheet(flatten(entry))


def si_sheet(entry: Mapping[str, Any]) -> Dict[str, float]:
    """``{canonical_name: value_in_SI}`` for every recognised property.

    This is the machine-readable view: one name, one unit, no spelling
    variation. It is what the engine layer and any exporter should read.
    """
    flat = sheet(entry)
    out: Dict[str, float] = {}
    for canonical in flat.canonical_keys():
        value = flat.si(canonical)
        if value is not None and _isfinite(value):
            out[canonical] = value
    return out


# ── Validation ──────────────────────────────────────────────────────────

class Issue(NamedTuple):
    """One thing wrong (or suspicious) about a data sheet."""

    severity: str    # "error" | "warning"
    key: str         # the raw source key
    canonical: str   # canonical property, "" if unrecognised
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.key}: {self.message}"


def validate_sheet(entry: Mapping[str, Any], *, name: str = "") -> List[Issue]:
    """Range- and unit-check every recognised property in an entry.

    An ``error`` is a value that cannot be physically right (negative
    density, Poisson's ratio above 0.5, a melting point of 0 K). A
    ``warning`` is a value outside the schema's plausible range, which
    usually means the number is stored in a different unit than its key
    claims.
    """
    issues: List[Issue] = []
    flat = sheet(entry)
    props = schema()["properties"]

    for key in list(flat.keys()):
        parsed = parse_key(key)
        if parsed.canonical is None:
            continue
        raw = dict.__getitem__(flat, key)
        value = to_si(raw, parsed.unit)
        if value is None:
            if _is_scalar(raw) and isinstance(raw, str):
                continue  # descriptive string under a numeric-looking key
            continue
        if not _isfinite(value):
            issues.append(Issue("error", key, parsed.canonical,
                                f"non-finite value {raw!r}"))
            continue
        spec = props[parsed.canonical]
        rng = spec.get("range")
        if not rng or len(rng) != 2:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        if value < lo or value > hi:
            # A value that is off by a clean power of 1000 is almost always a
            # unit mislabel; call that out specifically because it is fixable.
            hint = ""
            for factor, label in ((1e-3, "1000x too large"), (1e3, "1000x too small"),
                                  (1e-6, "1e6x too large"), (1e6, "1e6x too small")):
                if lo <= value * factor <= hi:
                    hint = f" (looks {label} -- unit mislabel?)"
                    break
            severity = "error" if _impossible(parsed.canonical, value, lo, hi) else "warning"
            issues.append(Issue(
                severity, key, parsed.canonical,
                f"{value:g} {spec.get('si_unit')} outside plausible "
                f"[{lo:g}, {hi:g}]{hint}",
            ))
    if name:
        issues = [i._replace(key=f"{name}.{i.key}") for i in issues]
    return issues


#: Canonical properties whose range is a hard physical bound, not a guess.
_HARD_BOUNDS = frozenset({
    "poissons_ratio", "packing_factor", "porosity", "crystallinity",
    "water_content", "emissivity", "reflectance", "transmittance",
    "absorbance", "elongation", "reduction_of_area",
})


def _impossible(canonical: str, value: float, lo: float, hi: float) -> bool:
    """True when a range violation is physically impossible, not just odd."""
    if canonical in _HARD_BOUNDS:
        return True
    # Any positive-definite quantity going negative is an error.
    return value < 0 <= lo


def coverage(entry: Mapping[str, Any], props: Iterable[str]) -> Dict[str, bool]:
    """Which of `props` (canonical names) the entry actually provides."""
    flat = sheet(entry)
    have = set(flat.canonical_keys())
    return {p: (canonical_of(p) or p) in have for p in props}


__all__ = [
    "DataSheet",
    "Issue",
    "MAX_GROUP_DEPTH",
    "ParsedKey",
    "canonical_of",
    "convert",
    "coverage",
    "flatten",
    "from_si",
    "list_properties",
    "parse_key",
    "property_spec",
    "reload_schema",
    "schema",
    "sheet",
    "si_sheet",
    "slugify",
    "to_si",
    "unit_spec",
    "validate_sheet",
]
