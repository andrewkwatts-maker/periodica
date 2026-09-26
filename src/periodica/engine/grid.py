"""Bulk field sampling: 2D and 3D, vectorised, strategy-selectable.

`periodica.sample.sample()` evaluates one property at one point. Asking it for
a volume means a Python call per voxel, and the existing `voxel_sample` did
exactly that with a triple loop -- a 128^3 grid is two million interpreter
round trips for arithmetic that is the same at every point.

This module evaluates a whole grid at once instead::

    field = sample_field("Steel_AISI_1045", "hardness",
                         bounds=((0, 0, 0), (0.01, 0.01, 0.01)),
                         resolution=128)            # (128, 128, 128)

    slab = sample_field("Steel_AISI_1045", "hardness",
                        bounds=((0, 0), (0.01, 0.01)),
                        resolution=512, dims=2)     # (512, 512)

Both channels (`"hardness"`, resolved through `engine.channels`) and raw sheet
properties (`"YoungsModulus_GPa"`) are accepted, so this serves engine
consumers and data consumers with one path.

2D is a first-class mode, not a 3D volume with one voxel of depth: a slice
plane and its offset are explicit (`plane="xz"`, `at=0.004`), so a cross
section costs one plane's worth of work.

Execution strategy comes from `periodica.engine.backends` -- `how="scalar"`,
`"vector"`, `"chunked"`, `"parallel"`, or left to auto. Strategy changes the
schedule, never the answer: the phase hash is shared with the scalar sampler
(see `periodica._hash`), so the same point lands in the same grain whichever
strategy runs. Two grids that differ only in `how` compare equal.

Homogeneous fields short-circuit to a constant fill, which is the common case
and makes it nearly free.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from periodica._hash import pick_by_fraction, unit_at_grains, unit_at_points
from periodica.engine import backends
from periodica.engine.backends import Plan, Strategy
from periodica.get import Get
from periodica.properties import sheet as properties_sheet
from periodica.sample import sample as scalar_sample

Bounds2 = Tuple[Sequence[float], Sequence[float]]
Bounds3 = Tuple[Sequence[float], Sequence[float]]

#: Plane orientations for 2D sampling: name -> (axis0, axis1, fixed axis).
PLANES = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}

#: Field models this module evaluates without falling back to per-point calls.
VECTORISED_MODELS = frozenset({
    "homogeneous", "mixture", "anisotropic_axial", "microstructure_voronoi",
})


# ── Value resolution ────────────────────────────────────────────────────

def _entry_of(name_or_entry: Union[str, Mapping[str, Any]]) -> Tuple[str, dict]:
    if isinstance(name_or_entry, str):
        return name_or_entry, Get(name_or_entry)
    entry = dict(name_or_entry)
    return str(entry.get("Name") or entry.get("name") or "<entry>"), entry


def _bulk_value(entry: Mapping[str, Any], what: str) -> Optional[float]:
    """The position-independent value of a channel or property, as a float.

    Tries the data sheet first so a raw property key wins by its own name,
    then the channel layer, which can derive and default.
    """
    flat = properties_sheet(entry)
    direct = flat.get(what)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return float(direct)

    from periodica.engine.channels import config as channel_config, resolve

    if what in channel_config()["channels"]:
        resolved = resolve(entry, what)
        if isinstance(resolved.value, float):
            return resolved.value
        if isinstance(resolved.value, tuple):
            raise ValueError(
                f"Channel {what!r} is a {len(resolved.value)}-component value; "
                f"sample its components separately or use sample_rgb()."
            )
    return None


def _phase_value(
    phase: Mapping[str, Any],
    entry: Mapping[str, Any],
    what: str,
    fallback: Optional[float],
) -> float:
    """A phase's value for `what`, falling back to the bulk value.

    A phase is a material in its own right, so a channel has to be resolved
    *against the phase*, not looked up by name in it. Ferrite and pearlite
    differ in Hardness_HV; a caller asking for the `hardness` channel wants
    each phase's own hardness, which means running the channel's chain over a
    view of the entry whose properties are the phase's. Looking `hardness` up
    as a key would miss, because a channel name is not a property name -- and
    silently returning the bulk value would make a two-phase steel sample
    perfectly uniform.

    The phase view keeps the parent's other fields so a formula can still
    reach, say, density, but drops Field to avoid recursing into the phase
    dispatch that got us here.
    """
    from periodica.properties import DataSheet

    direct = DataSheet(dict(phase)).get(what)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return float(direct)

    from periodica.engine.channels import config as channel_config, resolve

    if what in channel_config()["channels"]:
        view = {k: v for k, v in entry.items() if k != "Field"}
        # Phase values override the parent's: the whole point is that this
        # grain differs from the bulk. Merging the other way round let the
        # parent's bulk modulus mask the phase's and made a two-phase steel
        # sample uniform through the channel while varying through the raw key.
        view["Properties"] = {**dict(view.get("Properties") or {}), **dict(phase)}
        resolved = resolve(view, what)
        if isinstance(resolved.value, float):
            return resolved.value

    return float("nan") if fallback is None else float(fallback)


# ── Grid geometry ───────────────────────────────────────────────────────

def _axis(lo: float, hi: float, count: int) -> np.ndarray:
    """Cell-centre coordinates: `count` samples spanning [lo, hi)."""
    if count < 1:
        raise ValueError("resolution must be >= 1")
    if not (hi > lo):
        raise ValueError(f"bounds upper {hi} must exceed lower {lo}")
    step = (hi - lo) / count
    return lo + (np.arange(count, dtype=np.float64) + 0.5) * step


def _resolve_resolution(resolution: Union[int, Sequence[int]], dims: int) -> Tuple[int, ...]:
    if isinstance(resolution, int):
        return (resolution,) * dims
    res = tuple(int(r) for r in resolution)
    if len(res) != dims:
        raise ValueError(f"resolution needs {dims} values for dims={dims}, got {len(res)}")
    return res


def grid_points(
    bounds: Union[Bounds2, Bounds3],
    resolution: Union[int, Sequence[int]],
    *,
    dims: int = 3,
    plane: str = "xy",
    at: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Return ``(points, shape)`` for a regular grid.

    `points` is always ``(n, 3)`` -- a 2D grid is embedded in 3-space at `at`
    on the axis its `plane` leaves out, since every field model is defined in
    three dimensions. `shape` is the grid shape the caller asked for.
    """
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    lo, hi = bounds
    lo = [float(v) for v in lo]
    hi = [float(v) for v in hi]
    if len(lo) != dims or len(hi) != dims:
        raise ValueError(f"bounds need {dims} components per corner for dims={dims}")
    shape = _resolve_resolution(resolution, dims)

    if dims == 3:
        axes = [_axis(lo[i], hi[i], shape[i]) for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.reshape(-1) for m in mesh], axis=-1)
        return points, shape

    if plane not in PLANES:
        raise ValueError(f"plane must be one of {sorted(PLANES)}, got {plane!r}")
    a0, a1, fixed = PLANES[plane]
    axes = [_axis(lo[0], hi[0], shape[0]), _axis(lo[1], hi[1], shape[1])]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.empty((mesh[0].size, 3), dtype=np.float64)
    points[:, a0] = mesh[0].reshape(-1)
    points[:, a1] = mesh[1].reshape(-1)
    points[:, fixed] = float(at)
    return points, shape


# ── Vectorised field models ─────────────────────────────────────────────

def _vector_homogeneous(
    field: Mapping[str, Any], entry: Mapping[str, Any], what: str,
    points: np.ndarray, scale_m: Optional[float],
) -> np.ndarray:
    bulk = _bulk_value(entry, what)
    scale_dependent = field.get("scale_dependent")
    macro = field.get("macro_scale_m")

    if not (scale_dependent and scale_m is not None and macro is not None
            and scale_m < float(macro)):
        return np.full(len(points), float("nan") if bulk is None else bulk)

    names = [k for k, v in scale_dependent.items() if isinstance(v, Mapping)]
    fractions = [float(scale_dependent[k].get("fraction", 0.0)) for k in names]
    picked = pick_by_fraction(unit_at_points(points), fractions)
    out = np.full(len(points), float("nan") if bulk is None else bulk)
    for index, name in enumerate(names):
        mask = picked == index
        if mask.any():
            out[mask] = _phase_value(scale_dependent[name], entry, what, bulk)
    return out


def _vector_microstructure_voronoi(
    field: Mapping[str, Any], entry: Mapping[str, Any], what: str,
    points: np.ndarray, scale_m: Optional[float],
) -> np.ndarray:
    bulk = _bulk_value(entry, what)
    filler = float("nan") if bulk is None else bulk
    macro = field.get("macro_scale_m")
    phases = field.get("phases") or {}
    if (scale_m is None or macro is None or scale_m >= float(macro)
            or not isinstance(phases, Mapping) or not phases):
        return np.full(len(points), filler)

    grain_density = float(field.get("grain_density", 1.0))
    grain_size = max(1e-12, 1.0 / max(1e-12, grain_density)) ** (1.0 / 3.0)
    uniforms = unit_at_grains(points, grain_size)

    names = list(phases)
    fractions = [float(phases[k].get("fraction", 0.0)) if isinstance(phases[k], Mapping)
                 else 0.0 for k in names]
    picked = pick_by_fraction(uniforms, fractions)

    out = np.full(len(points), filler)
    for index, name in enumerate(names):
        mask = picked == index
        if mask.any() and isinstance(phases[name], Mapping):
            out[mask] = _phase_value(phases[name], entry, what, bulk)
    return out


def _vector_anisotropic_axial(
    field: Mapping[str, Any], entry: Mapping[str, Any], what: str,
    points: np.ndarray, scale_m: Optional[float],
) -> np.ndarray:
    from periodica.properties import DataSheet

    bulk = _bulk_value(entry, what)
    axial = DataSheet(dict(field.get("axial") or {})).get(what)
    transverse = DataSheet(dict(field.get("transverse") or {})).get(what)
    if axial is None and transverse is None:
        return np.full(len(points), float("nan") if bulk is None else bulk)
    if axial is None:
        return np.full(len(points), float(transverse))
    if transverse is None:
        return np.full(len(points), float(axial))

    direction = np.asarray(field.get("fiber_direction") or [1.0, 0.0, 0.0], dtype=np.float64)
    norm = np.linalg.norm(direction) or 1.0
    direction = direction / norm

    lengths = np.linalg.norm(points, axis=1)
    safe = np.where(lengths > 0, lengths, 1.0)
    cos = np.abs((points @ direction) / safe)
    cos = np.where(lengths > 0, cos, 1.0)   # the origin is treated as axial
    return float(axial) * cos + float(transverse) * (1.0 - cos)


def _vector_constant(
    field: Mapping[str, Any], entry: Mapping[str, Any], what: str,
    points: np.ndarray, scale_m: Optional[float],
) -> np.ndarray:
    """For models whose value does not vary with position (e.g. `mixture`)."""
    bulk = _bulk_value(entry, what)
    if bulk is None:
        value = scalar_sample(dict(entry), what, scale_m=scale_m)
        bulk = float(value) if isinstance(value, (int, float)) else None
    return np.full(len(points), float("nan") if bulk is None else bulk)


_VECTOR_MODELS = {
    "homogeneous": _vector_homogeneous,
    "mixture": _vector_constant,
    "anisotropic_axial": _vector_anisotropic_axial,
    "microstructure_voronoi": _vector_microstructure_voronoi,
}


def _model_of(entry: Mapping[str, Any]) -> str:
    field = entry.get("Field")
    if not isinstance(field, Mapping):
        return "homogeneous"
    return str(field.get("model", "homogeneous"))


def is_vectorisable(name_or_entry: Union[str, Mapping[str, Any]]) -> bool:
    """Whether this entry's field model has a vectorised implementation.

    A custom model registered through `register_field_model` does not, so
    sampling it falls back to the scalar path -- correct, just slower.
    """
    _name, entry = _entry_of(name_or_entry)
    return _model_of(entry) in VECTORISED_MODELS


# ── Public API ──────────────────────────────────────────────────────────

def sample_points(
    name_or_entry: Union[str, Mapping[str, Any]],
    what: str,
    points: np.ndarray,
    *,
    scale_m: Optional[float] = None,
    how: Optional[str | Strategy] = None,
    workers: Optional[int] = None,
    chunk_bytes: int = backends.DEFAULT_CHUNK_BYTES,
) -> np.ndarray:
    """Sample `what` at an ``(n, 3)`` array of positions.

    `what` is a channel name or a raw data-sheet property key. Unknown or
    non-numeric values come back as NaN rather than raising, so one bad
    property does not abort a whole grid.
    """
    _name, entry = _entry_of(name_or_entry)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (n, 3), got {pts.shape}")

    total = len(pts)
    model = _model_of(entry)
    vector_fn = _VECTOR_MODELS.get(model)
    strategy, _reason = backends.choose(
        total, how=how, workers=workers, vectorisable=vector_fn is not None,
    )

    field = entry.get("Field") if isinstance(entry.get("Field"), Mapping) else {}

    if vector_fn is None:
        # No vectorised implementation for this field model (a custom one
        # registered through register_field_model). Fall back to the scalar
        # sampler, which is correct, just slower.
        out = np.empty(total, dtype=np.float64)
        for index in range(total):
            value = scalar_sample(dict(entry), what,
                                  at=tuple(pts[index]), scale_m=scale_m)
            out[index] = float("nan") if not isinstance(value, (int, float)) \
                or isinstance(value, bool) else float(value)
        return out

    if strategy is Strategy.SCALAR:
        # Same implementation as `vector`, one point at a time. Routing this
        # through a separate scalar code path is how the two silently drift
        # apart: the guarantee that strategy changes only the schedule has to
        # hold by construction, not by two implementations agreeing.
        out = np.empty(total, dtype=np.float64)
        for index in range(total):
            out[index] = vector_fn(field, entry, what,
                                   pts[index:index + 1], scale_m)[0]
        return out

    if strategy is Strategy.VECTOR:
        return vector_fn(field, entry, what, pts, scale_m)

    out = np.empty(total, dtype=np.float64)
    chunk_points = max(1, chunk_bytes // 8)

    def kernel(window: slice) -> np.ndarray:
        return vector_fn(field, entry, what, pts[window], scale_m)

    if strategy is Strategy.PARALLEL:
        n_workers = workers if workers is not None else backends.worker_default()
        return backends.run_parallel(total, kernel, out=out,
                                     chunk_points=chunk_points, workers=n_workers)
    return backends.run_chunked(total, kernel, out=out, chunk_points=chunk_points)


def sample_field(
    name_or_entry: Union[str, Mapping[str, Any]],
    what: str,
    *,
    bounds: Union[Bounds2, Bounds3],
    resolution: Union[int, Sequence[int]],
    dims: int = 3,
    plane: str = "xy",
    at: float = 0.0,
    scale_m: Optional[float] = None,
    how: Optional[str | Strategy] = None,
    workers: Optional[int] = None,
) -> np.ndarray:
    """Sample `what` over a regular 2D or 3D grid.

    `scale_m` defaults to the cell size, which is the natural sampling
    resolution: it is what tells a scale-dependent field whether the caller is
    looking at grains or at bulk.

    Returns an array of the requested `resolution` shape.
    """
    points, shape = grid_points(bounds, resolution, dims=dims, plane=plane, at=at)
    if scale_m is None:
        lo, hi = bounds
        spans = [abs(float(hi[i]) - float(lo[i])) / shape[i] for i in range(dims)]
        scale_m = min(spans) if spans else None
    values = sample_points(name_or_entry, what, points,
                           scale_m=scale_m, how=how, workers=workers)
    return values.reshape(shape)


def sample_slice(
    name_or_entry: Union[str, Mapping[str, Any]],
    what: str,
    *,
    bounds: Bounds2,
    resolution: Union[int, Sequence[int]],
    plane: str = "xy",
    at: float = 0.0,
    scale_m: Optional[float] = None,
    how: Optional[str | Strategy] = None,
) -> np.ndarray:
    """A 2D cross section. `sample_field(..., dims=2)` with a clearer name."""
    return sample_field(name_or_entry, what, bounds=bounds, resolution=resolution,
                        dims=2, plane=plane, at=at, scale_m=scale_m, how=how)


def sample_rgb(
    name_or_entry: Union[str, Mapping[str, Any]],
    *,
    bounds: Union[Bounds2, Bounds3],
    resolution: Union[int, Sequence[int]],
    dims: int = 3,
    plane: str = "xy",
    at: float = 0.0,
    channel: str = "colour",
) -> np.ndarray:
    """Sample a 3-component channel, returning ``(..., 3)``.

    Colour is per-entry rather than per-point today, so this broadcasts one
    triple across the grid. It exists so a caller does not special-case
    colour, and so per-phase colour can arrive here later without the call
    sites changing.
    """
    from periodica.engine.channels import resolve

    _name, entry = _entry_of(name_or_entry)
    _points, shape = grid_points(bounds, resolution, dims=dims, plane=plane, at=at)
    value = resolve(entry, channel).value
    triple = value if isinstance(value, tuple) else (0.0, 0.0, 0.0)
    return np.broadcast_to(np.asarray(triple, dtype=np.float64),
                           (*shape, 3)).copy()


def plan_field(
    name_or_entry: Union[str, Mapping[str, Any]],
    *,
    resolution: Union[int, Sequence[int]],
    dims: int = 3,
    how: Optional[str | Strategy] = None,
    workers: Optional[int] = None,
) -> Plan:
    """What `sample_field` would do for this grid, without sampling it."""
    shape = _resolve_resolution(resolution, dims)
    total = int(math.prod(shape))
    return backends.plan(total, how=how, workers=workers,
                         vectorisable=is_vectorisable(name_or_entry))


__all__ = [
    "PLANES",
    "VECTORISED_MODELS",
    "grid_points",
    "is_vectorisable",
    "plan_field",
    "sample_field",
    "sample_points",
    "sample_rgb",
    "sample_slice",
]
