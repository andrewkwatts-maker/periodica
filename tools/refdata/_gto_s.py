"""Closed-form integrals over contracted s-type Gaussians (validation helper only).

Used by the refdata scripts to cross-check transcribed reference integrals and
energies (H2/STO-3G, the s-s blocks of the Crawford H2O/CH4 fixtures) against
an independent evaluation from the Basis Set Exchange exponents.  This is not
library code: the production integral engine lives in ``periodica-qm``.

All quantities are in atomic units (bohr, hartree).  Primitives are
normalised, ``N = (2a/pi)^(3/4)``; contracted functions are renormalised.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


def _dist2(a: Vec3, b: Vec3) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def boys0(t: float) -> float:
    """Boys function F0(t) = (1/2) sqrt(pi/t) erf(sqrt t), with the t -> 0 series."""
    if t < 1e-12:
        return 1.0 - t / 3.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


@dataclass(frozen=True)
class SShell:
    """A contracted s function centred at ``centre`` (bohr)."""

    centre: Vec3
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]

    def primitives(self) -> list[tuple[float, float]]:
        """(exponent, coefficient * primitive norm * contraction norm) pairs."""
        prims = [(a, d * (2.0 * a / math.pi) ** 0.75) for a, d in zip(self.exponents, self.coefficients, strict=True)]
        self_overlap = sum(
            ca * cb * (math.pi / (a + b)) ** 1.5 for a, ca in prims for b, cb in prims
        )
        scale = 1.0 / math.sqrt(self_overlap)
        return [(a, c * scale) for a, c in prims]


def _gaussian_product(a: float, ra: Vec3, b: float, rb: Vec3) -> tuple[float, Vec3, float]:
    p = a + b
    rp = tuple((a * x + b * y) / p for x, y in zip(ra, rb, strict=True))
    k = math.exp(-a * b / p * _dist2(ra, rb))
    return p, rp, k  # type: ignore[return-value]


def overlap(f: SShell, g: SShell) -> float:
    total = 0.0
    for a, ca in f.primitives():
        for b, cb in g.primitives():
            p, _, k = _gaussian_product(a, f.centre, b, g.centre)
            total += ca * cb * (math.pi / p) ** 1.5 * k
    return total


def kinetic(f: SShell, g: SShell) -> float:
    r2 = _dist2(f.centre, g.centre)
    total = 0.0
    for a, ca in f.primitives():
        for b, cb in g.primitives():
            p, _, k = _gaussian_product(a, f.centre, b, g.centre)
            mu = a * b / p
            total += ca * cb * mu * (3.0 - 2.0 * mu * r2) * (math.pi / p) ** 1.5 * k
    return total


def nuclear(f: SShell, g: SShell, charges: Sequence[tuple[float, Vec3]]) -> float:
    total = 0.0
    for a, ca in f.primitives():
        for b, cb in g.primitives():
            p, rp, k = _gaussian_product(a, f.centre, b, g.centre)
            for z, rc in charges:
                total -= z * ca * cb * 2.0 * math.pi / p * k * boys0(p * _dist2(rp, rc))
    return total


def eri(f: SShell, g: SShell, h: SShell, m: SShell) -> float:
    """Chemists' notation (fg|hm)."""
    total = 0.0
    for a, ca in f.primitives():
        for b, cb in g.primitives():
            p, rp, kab = _gaussian_product(a, f.centre, b, g.centre)
            for c, cc in h.primitives():
                for d, cd in m.primitives():
                    q, rq, kcd = _gaussian_product(c, h.centre, d, m.centre)
                    pref = 2.0 * math.pi**2.5 / (p * q * math.sqrt(p + q))
                    total += ca * cb * cc * cd * pref * kab * kcd * boys0(p * q / (p + q) * _dist2(rp, rq))
    return total
