"""Deterministic spatial hashing, shared by the scalar and vectorised samplers.

Phase and grain assignment has to be a pure function of position: the same
point must land in the same grain on every call, in every process, on every
machine, with no seed to thread through and no lookup table to store. A hash
of the quantised coordinate gives exactly that.

The hash must also be the *same* function whichever sampler runs. The scalar
sampler originally hashed with SHA-1, which cannot be vectorised -- and a
vectorised sampler with a different hash would assign a different grain to the
same point depending on which execution strategy the caller happened to pick.
Same input, different output, silently. So both now call in here.

This is SplitMix64's finalising mix over the packed integer coordinate: a
well-tested avalanche function, three multiplies and three shifts, no state.
It is not cryptographic and does not need to be -- the requirement is uniform
spread and reproducibility, not unpredictability.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S1 = np.uint64(30)
_S2 = np.uint64(27)
_S3 = np.uint64(31)

# Odd constants, so each axis contributes to every output bit.
_AX = np.uint64(0x9E3779B97F4A7C15)   # golden-ratio odd constant
_AY = np.uint64(0xC2B2AE3D27D4EB4F)
_AZ = np.uint64(0x165667B19E3779F9)
_SEED = np.uint64(0x2545F4914F6CDD1D)

#: Coordinates are rounded to this many decimals before hashing, so floating
#: point noise at the last bit cannot flip a point into a neighbouring grain.
COORD_DECIMALS = 9


def _mix(z: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser. Operates elementwise on a uint64 array."""
    with np.errstate(over="ignore"):
        z = (z ^ (z >> _S1)) * _M1
        z = (z ^ (z >> _S2)) * _M2
        return z ^ (z >> _S3)


def hash_cells(
    ix: np.ndarray,
    iy: Optional[np.ndarray] = None,
    iz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Hash integer cell coordinates to uint64. Accepts 1, 2 or 3 axes."""
    acc = np.asarray(ix).astype(np.int64).astype(np.uint64) * _AX
    if iy is not None:
        with np.errstate(over="ignore"):
            acc = acc + np.asarray(iy).astype(np.int64).astype(np.uint64) * _AY
    if iz is not None:
        with np.errstate(over="ignore"):
            acc = acc + np.asarray(iz).astype(np.int64).astype(np.uint64) * _AZ
    with np.errstate(over="ignore"):
        acc = acc + _SEED
    return _mix(acc)


def unit_from_cells(
    ix: np.ndarray,
    iy: Optional[np.ndarray] = None,
    iz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Hash integer cell coordinates to a float in [0, 1)."""
    return (hash_cells(ix, iy, iz) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def unit_at_points(points: np.ndarray, *, decimals: int = COORD_DECIMALS) -> np.ndarray:
    """Hash an ``(n, 3)`` array of positions to uniforms in [0, 1).

    Positions are rounded first, then scaled by ``10**decimals`` and taken as
    integers, so two positions that differ only in float noise hash alike.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    scale = 10.0 ** decimals
    cells = np.rint(np.round(pts, decimals) * scale).astype(np.int64)
    axes = [cells[:, i] for i in range(min(3, cells.shape[1]))]
    while len(axes) < 3:
        axes.append(np.zeros_like(axes[0]))
    return unit_from_cells(*axes)


def unit_at_point(point: Sequence[float], *, decimals: int = COORD_DECIMALS) -> float:
    """Scalar form of `unit_at_points`, for the one-point path."""
    return float(unit_at_points(np.asarray(point, dtype=np.float64).reshape(1, -1),
                                decimals=decimals)[0])


def grain_cells(
    points: np.ndarray,
    grain_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise positions into grain cells of side `grain_size`."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    size = max(float(grain_size), 1e-30)
    cells = np.rint(pts / size).astype(np.int64)
    cols = [cells[:, i] if i < cells.shape[1] else np.zeros(len(cells), dtype=np.int64)
            for i in range(3)]
    return cols[0], cols[1], cols[2]


def unit_at_grains(points: np.ndarray, grain_size: float) -> np.ndarray:
    """Hash positions to a uniform per *grain*, not per point.

    Every position inside one grain cell gets the same value, which is what
    makes a grain a grain rather than per-voxel noise.
    """
    return unit_from_cells(*grain_cells(points, grain_size))


def pick_by_fraction(
    uniforms: np.ndarray,
    fractions: Sequence[float],
) -> np.ndarray:
    """Map uniforms in [0, 1) to indices by cumulative `fractions`.

    Returns -1 where the uniform falls past the last cumulative bound, which
    is how a leftover matrix phase is represented.
    """
    u = np.asarray(uniforms, dtype=np.float64)
    out = np.full(u.shape, -1, dtype=np.int32)
    cumulative = 0.0
    unassigned = np.ones(u.shape, dtype=bool)
    for index, fraction in enumerate(fractions):
        cumulative += float(fraction)
        hit = unassigned & (u < cumulative)
        out[hit] = index
        unassigned &= ~hit
        if not unassigned.any():
            break
    return out


__all__ = [
    "COORD_DECIMALS",
    "grain_cells",
    "hash_cells",
    "pick_by_fraction",
    "unit_at_grains",
    "unit_at_point",
    "unit_at_points",
    "unit_from_cells",
]
