"""Internal-coordinate geometry helpers for the curated molecule geometries (todo Q3).

``zmatrix_to_cartesian`` places atoms with the natural-extension reference
frame (NeRF) construction; ``distance``/``angle``/``dihedral`` measure them
back so every transcribed source value can be checked against the Cartesians.
Lengths in angstrom, angles in degrees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec = tuple[float, float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a: Vec) -> Vec:
    n = math.sqrt(_dot(a, a))
    return _scale(a, 1.0 / n)


def distance(p: Vec, q: Vec) -> float:
    d = _sub(p, q)
    return math.sqrt(_dot(d, d))


def angle(p: Vec, q: Vec, r: Vec) -> float:
    """Angle p-q-r at q, degrees."""
    u, v = _unit(_sub(p, q)), _unit(_sub(r, q))
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(u, v)))))


def dihedral(p: Vec, q: Vec, r: Vec, s: Vec) -> float:
    """Dihedral p-q-r-s in degrees, (-180, 180], IUPAC sign (positive = clockwise viewed along q->r)."""
    b0, b1, b2 = _sub(p, q), _unit(_sub(r, q)), _sub(s, r)
    v = _sub(b0, _scale(b1, _dot(b0, b1)))
    w = _sub(b2, _scale(b1, _dot(b2, b1)))
    return math.degrees(math.atan2(_dot(_cross(b1, v), w), _dot(v, w)))


def place(a: Vec, b: Vec, c: Vec, bond: float, ang: float, tors: float) -> Vec:
    """Position d with |d-c| = bond, angle(b, c, d) = ang, dihedral(a, b, c, d) = tors (NeRF)."""
    t, p = math.radians(ang), math.radians(tors)
    d2 = (-bond * math.cos(t), bond * math.sin(t) * math.cos(p), bond * math.sin(t) * math.sin(p))
    bc = _unit(_sub(c, b))
    n = _unit(_cross(_sub(b, a), bc))
    m = _cross(n, bc)
    return _add(c, _add(_scale(bc, d2[0]), _add(_scale(m, d2[1]), _scale(n, d2[2]))))


def zmatrix_to_cartesian(rows: Sequence[tuple[int | None, float | None, int | None, float | None, int | None, float | None]]) -> list[Vec]:
    """Rows are (i, bond, j, angle, k, dihedral) with 0-based reference atoms; unused entries None.

    Atom 0 is at the origin, atom 1 on +z, atom 2 in the xz plane (positive x).
    """
    xyz: list[Vec] = []
    for idx, (i, bond, j, ang, k, tors) in enumerate(rows):
        if idx == 0:
            xyz.append((0.0, 0.0, 0.0))
        elif idx == 1:
            assert i is not None and bond is not None
            xyz.append(_add(xyz[i], (0.0, 0.0, bond)))
        elif idx == 2 or k is None:
            assert i is not None and j is not None and bond is not None and ang is not None
            # Third atom (or any atom placed without a dihedral): use a fixed auxiliary point off the i-j axis.
            ci, cj = xyz[i], xyz[j]
            axis = _unit(_sub(cj, ci))
            helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
            aux = _add(cj, helper)
            xyz.append(place(aux, cj, ci, bond, ang, 0.0))
        else:
            assert i is not None and j is not None and bond is not None and ang is not None and tors is not None
            xyz.append(place(xyz[k], xyz[j], xyz[i], bond, ang, tors))
    return xyz


def centre(xyz: Sequence[Vec]) -> list[Vec]:
    """Translate so the geometric centroid is at the origin."""
    n = len(xyz)
    c = (sum(p[0] for p in xyz) / n, sum(p[1] for p in xyz) / n, sum(p[2] for p in xyz) / n)
    return [_sub(p, c) for p in xyz]


def kabsch_rmsd(a: Sequence[Vec], b: Sequence[Vec]) -> float:
    """RMSD between two conformations after optimal superposition (Horn quaternion method, no numpy)."""
    ca, cb = centre(a), centre(b)
    s = [[sum(p[x] * q[y] for p, q in zip(ca, cb, strict=True)) for y in range(3)] for x in range(3)]
    (sxx, sxy, sxz), (syx, syy, syz), (szx, szy, szz) = s
    k = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    # largest eigenvalue by power iteration on a shifted matrix
    shift = sum(abs(v) for row in k for v in row) + 1.0
    m = [[k[r][c] + (shift if r == c else 0.0) for c in range(4)] for r in range(4)]
    vec = [1.0, 0.1, 0.1, 0.1]
    lam = 0.0
    for _ in range(2000):
        nxt = [sum(m[r][c] * vec[c] for c in range(4)) for r in range(4)]
        norm = math.sqrt(sum(x * x for x in nxt))
        vec = [x / norm for x in nxt]
        lam = norm
    lam -= shift
    e0 = sum(_dot(p, p) for p in ca) + sum(_dot(q, q) for q in cb)
    return math.sqrt(max(0.0, (e0 - 2.0 * lam) / len(ca)))
