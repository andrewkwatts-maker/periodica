"""3D property sampling for material entries (and any entry with a Field).

A "data sheet" is the dict of bulk properties stored under `Properties`
in a registry entry. A "field model" describes how each property varies
in space - declared in JSON under `Field` so no chemistry/physics is
hardcoded in this module.

Public API
----------
- `data_sheet(name)` -> the bulk Properties dict for an entry.
- `sample(name, prop, at=None, scale_m=None)` -> a single property value
  evaluated at a 3D point.
- `register_field_model(name, evaluator)` -> add a custom field model.

Built-in field models (declared in entries via `"Field": {"model": ...}`)
- `homogeneous` (default): same value everywhere; ignores `at`/`scale_m`.
- `mixture`: volume- or mass-weighted average of constituent entries'
  properties (rule-of-mixtures). Composition fractions are read from
  Field.weights, falling back to entry.Composition.
- `anisotropic_axial`: returns Field.axial[prop] when sampled along a
  fiber direction, Field.transverse[prop] perpendicular to it. Direction
  is inferred from `at` (the sample point's unit vector) projected onto
  Field.fiber_direction.

Heterogeneity at small scales is supported by Field.scale_dependent:

    "Field": {
      "model": "homogeneous",
      "scale_dependent": {
        "matrix":     {"YoungsModulus_GPa": 200},
        "precipitate":{"YoungsModulus_GPa": 350, "fraction": 0.05}
      }
    }

When `scale_m` is below Field.macro_scale_m the sampler picks a phase
based on the spatial point hash + fractions. When `scale_m` is None
or large, the sampler returns the bulk Properties.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from periodica.get import Get, UnknownName


Point = Tuple[float, float, float]
FieldEvaluator = Callable[[dict, str, Optional[Point], Optional[float], dict], Any]

_FIELD_MODELS: Dict[str, FieldEvaluator] = {}


def register_field_model(name: str, evaluator: FieldEvaluator) -> None:
    """Register a field-model evaluator. Evaluator signature:

        evaluator(field_dict, prop_name, at, scale_m, entry) -> value
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Field model name must be a non-empty string.")
    _FIELD_MODELS[name] = evaluator


# ── Helpers ─────────────────────────────────────────────────────────────

def _props_of(entry: dict) -> dict:
    """Bulk properties dict of an entry, falling back to root scalars."""
    p = entry.get("Properties")
    if isinstance(p, dict):
        return p
    # Promote root scalars (Mass_amu, Charge_e, ...) so atoms/molecules sample too.
    return {k: v for k, v in entry.items() if isinstance(v, (int, float)) and v is not None}


def _maybe_phase_at(point: Optional[Point], fractions: Mapping[str, float]) -> Optional[str]:
    """Pick a phase at a 3D point using a deterministic hash of the point.

    `fractions` is {phase_name: volume_fraction} summing to <= 1; the leftover
    is treated as the dominant matrix.
    """
    if point is None:
        return None
    h = hashlib.sha1(repr(tuple(round(c, 9) for c in point)).encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64  # uniform in [0, 1)
    cum = 0.0
    for name, frac in fractions.items():
        cum += float(frac)
        if u < cum:
            return name
    return None


# ── Built-in field models ───────────────────────────────────────────────

def _homogeneous(field: dict, prop: str, at, scale_m, entry) -> Any:
    # If small-scale heterogeneity is declared and scale_m is finer than
    # macro_scale_m, dispatch to a phase.
    sd = field.get("scale_dependent")
    macro = field.get("macro_scale_m")
    if sd and scale_m is not None and macro is not None and scale_m < macro:
        fractions = {k: v.get("fraction", 0.0) for k, v in sd.items() if isinstance(v, dict)}
        phase = _maybe_phase_at(at, fractions)
        if phase and phase in sd:
            phase_props = sd[phase]
            if prop in phase_props:
                return phase_props[prop]
    props = _props_of(entry)
    return props.get(prop)


def _mixture(field: dict, prop: str, at, scale_m, entry) -> Any:
    composition = entry.get("Composition") or {}
    weights = field.get("weights") or composition
    total = 0.0
    total_w = 0.0
    for sym, _count in composition.items():
        try:
            c_entry = Get(sym)
        except (UnknownName, Exception):
            continue
        v = _props_of(c_entry).get(prop)
        if v is None:
            continue
        w = float(weights.get(sym, 1.0))
        total += float(v) * w
        total_w += w
    if total_w == 0:
        return None
    return total / total_w


def _anisotropic_axial(field: dict, prop: str, at, scale_m, entry) -> Any:
    axial = field.get("axial") or {}
    transverse = field.get("transverse") or {}
    direction = field.get("fiber_direction") or [1.0, 0.0, 0.0]
    if at is None:
        # No sample point -> caller wants either axial or bulk; pick axial.
        return axial.get(prop, _props_of(entry).get(prop))
    nx, ny, nz = direction
    norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    px, py, pz = at
    pnorm = math.sqrt(px * px + py * py + pz * pz)
    if pnorm == 0:
        return axial.get(prop, _props_of(entry).get(prop))
    pxn, pyn, pzn = px / pnorm, py / pnorm, pz / pnorm
    cos = abs(nx * pxn + ny * pyn + nz * pzn)  # |cos(angle)|, 1 = parallel
    a_val = axial.get(prop)
    t_val = transverse.get(prop)
    if a_val is None and t_val is None:
        return _props_of(entry).get(prop)
    if a_val is None:
        return t_val
    if t_val is None:
        return a_val
    return float(a_val) * cos + float(t_val) * (1.0 - cos)


def _backbone_path(field: dict, prop: str, at, scale_m, entry) -> Any:
    """Sample a per-residue property at a 3D point along a protein backbone.

    Lazy-builds the backbone from `entry['residues']` phi/psi the first
    time the field is sampled (cache lives on the entry dict). When `at`
    is an integer (or 1-tuple), it's treated as a residue index. When
    `at` is a 3-vector, finds the nearest CA and returns that residue's
    property value. Property is read from the residues array entry.
    """
    residues = entry.get("residues") or entry.get("Residues") or []
    if not residues:
        return _props_of(entry).get(prop)

    # Direct integer addressing: at == residue index
    idx: Optional[int] = None
    if isinstance(at, int):
        idx = at
    elif isinstance(at, (tuple, list)) and len(at) == 1 and isinstance(at[0], (int,)):
        idx = at[0]

    if idx is None and at is not None:
        # Build/cache backbone, find nearest CA
        cached = entry.get("_BackboneCache")
        if cached is None:
            try:
                from periodica.folding import build_backbone_from_entry
                # Use the entry dict already in hand.
                from periodica.folding import build_backbone, extract_phi_psi
                seq = entry.get("sequence") or "".join(
                    r.get("residue", "X") for r in residues
                )
                phi_psi = extract_phi_psi(entry)
                if len(seq) > len(phi_psi):
                    seq = seq[:len(phi_psi)]
                else:
                    phi_psi = phi_psi[:len(seq)]
                coords = build_backbone(seq, phi_psi)
                entry["_BackboneCache"] = coords
                cached = coords
            except Exception:
                cached = None
        if cached is not None and len(cached) > 0:
            ca = cached[:, 1, :]   # (n, 3)
            px, py, pz = float(at[0]), float(at[1]), float(at[2])
            d2 = (ca[:, 0] - px) ** 2 + (ca[:, 1] - py) ** 2 + (ca[:, 2] - pz) ** 2
            idx = int(d2.argmin())

    if idx is not None and 0 <= idx < len(residues):
        residue = residues[idx]
        if prop in residue:
            return residue[prop]
        # Fall through to AA propensity lookup if requested.
        aa_letter = residue.get("residue")
        if aa_letter:
            try:
                from periodica.get import Get as _Get
                from periodica.get import Scope as _Scope
                aa = _Get(_Scope.AminoAcid, aa_letter)
                if prop in aa:
                    return aa[prop]
            except Exception:
                pass
    return _props_of(entry).get(prop)


def _microstructure_voronoi(field: dict, prop: str, at, scale_m, entry) -> Any:
    """Microstructure-aware sampling: at a 3D point, determine the grain and
    return that grain's phase property. Falls back to bulk at macro scales.

    Field declaration:
      {
        "model": "microstructure_voronoi",
        "macro_scale_m": 1e-4,
        "grain_density": 1000,
        "phases": {
          "ferrite":  { "fraction": 0.95, "YoungsModulus_GPa": 195 },
          "pearlite": { "fraction": 0.05, "YoungsModulus_GPa": 220 }
        }
      }
    """
    bulk = _props_of(entry).get(prop)
    macro = field.get("macro_scale_m")
    if at is None or scale_m is None or macro is None or scale_m >= macro:
        return bulk
    phases = field.get("phases") or {}
    if not isinstance(phases, dict) or not phases:
        return bulk
    # Hash 3D point + grain_density to a deterministic grain id, then a phase.
    g_density = float(field.get("grain_density", 1.0))
    # Quantize the point by ~1/grain_density to pick a grain "cell".
    grain_size = max(1e-12, 1.0 / max(1e-12, g_density)) ** (1.0 / 3.0)
    qx = round(float(at[0]) / grain_size)
    qy = round(float(at[1]) / grain_size)
    qz = round(float(at[2]) / grain_size)
    h = hashlib.sha1(repr((qx, qy, qz)).encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    cum = 0.0
    chosen = None
    for phase_name, phase_data in phases.items():
        if not isinstance(phase_data, dict):
            continue
        cum += float(phase_data.get("fraction", 0.0))
        if u < cum:
            chosen = phase_data
            break
    if chosen is None:
        return bulk
    return chosen.get(prop, bulk)


register_field_model("homogeneous", _homogeneous)
register_field_model("mixture", _mixture)
register_field_model("anisotropic_axial", _anisotropic_axial)
register_field_model("backbone_path", _backbone_path)
register_field_model("microstructure_voronoi", _microstructure_voronoi)


# ── Public API ──────────────────────────────────────────────────────────

def data_sheet(name_or_entry: Union[str, dict]) -> dict:
    """Return the full property dict for a registry entry.

    Accepts either a name (resolved via Get) or a pre-fetched entry dict.
    Raises UnknownName when the name doesn't resolve.
    """
    if isinstance(name_or_entry, str):
        entry = Get(name_or_entry)
    elif isinstance(name_or_entry, dict):
        entry = name_or_entry
    else:
        raise TypeError(
            f"data_sheet: expected str or dict, got {type(name_or_entry).__name__}"
        )
    return dict(_props_of(entry))


def sample(
    name_or_entry: Union[str, dict],
    prop: str,
    *,
    at: Optional[Sequence[float]] = None,
    scale_m: Optional[float] = None,
) -> Any:
    """Sample property `prop` of a registry entry at 3D point `at`.

    `at` is `(x, y, z)` in metres (or whatever convention the entry uses).
    `scale_m` is the characteristic length you're sampling over; when the
    Field declares `macro_scale_m` and scale_m is smaller, the sampler may
    pick a phase from `scale_dependent` data.

    For homogeneous materials with no `scale_dependent` block, `at` and
    `scale_m` are ignored - the bulk value is returned.
    """
    if isinstance(name_or_entry, str):
        entry = Get(name_or_entry)
    elif isinstance(name_or_entry, dict):
        entry = name_or_entry
    else:
        raise TypeError(
            f"sample: expected str or dict, got {type(name_or_entry).__name__}"
        )

    field = entry.get("Field")
    if not isinstance(field, dict):
        field = {"model": "homogeneous"}
    model = field.get("model", "homogeneous")
    evaluator = _FIELD_MODELS.get(model)
    if evaluator is None:
        raise ValueError(
            f"Unknown field model {model!r}. Registered: {sorted(_FIELD_MODELS)}"
        )
    point = tuple(float(c) for c in at) if at is not None else None
    return evaluator(field, prop, point, scale_m, entry)
