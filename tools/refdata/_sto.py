"""Radial Slater-type-orbital algebra used to validate the RHF-STO tables (helper only).

Convention (Bunge 1993, Koga 2000): a normalised STO is

    chi_{n l}(r) = N_n(zeta) r^(n-1) exp(-zeta r) Y_lm,   N_n(zeta) = (2 zeta)^(n + 1/2) / sqrt((2n)!)

and an orbital's radial part is R(r) = sum_i c_i N_i r^(n_i - 1) exp(-zeta_i r).
All radial integrals are closed form:

    int_0^inf r^(n_i + n_j + k) exp(-(zeta_i + zeta_j) r) dr = (n_i + n_j + k)! / (zeta_i + zeta_j)^(n_i + n_j + k + 1)
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def norm(n: int, zeta: float) -> float:
    return (2.0 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def radial_moment(ns: Sequence[int], zetas: Sequence[float], coefs: Sequence[float], k: int) -> float:
    """<R| r^k |R> = int R(r)^2 r^(2 + k) dr for one orbital (k >= -2, or -3 when every n >= 2)."""
    total = 0.0
    for ni, zi, ci in zip(ns, zetas, coefs, strict=True):
        for nj, zj, cj in zip(ns, zetas, coefs, strict=True):
            p = ni + nj + k
            total += ci * cj * norm(ni, zi) * norm(nj, zj) * math.gamma(p + 1) / (zi + zj) ** (p + 1)
    return total


def overlap(ns: Sequence[int], zetas: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    """<R_a|R_b> for two orbitals of the same l expanded in the same STO basis."""
    total = 0.0
    for ni, zi, ai in zip(ns, zetas, a, strict=True):
        for nj, zj, bj in zip(ns, zetas, b, strict=True):
            p = ni + nj
            total += ai * bj * norm(ni, zi) * norm(nj, zj) * math.gamma(p + 1) / (zi + zj) ** (p + 1)
    return total


def value_at_origin(ns: Sequence[int], zetas: Sequence[float], coefs: Sequence[float]) -> float:
    """R(0): only n = 1 functions are non-zero at the nucleus."""
    return sum(c * norm(n, z) for n, z, c in zip(ns, zetas, coefs, strict=True) if n == 1)


def cusp(ns: Sequence[int], zetas: Sequence[float], coefs: Sequence[float]) -> float:
    """-R'(0)/R(0) for an s orbital (equals Z for the exact cusp)."""
    r0 = value_at_origin(ns, zetas, coefs)
    d0 = 0.0
    for n, z, c in zip(ns, zetas, coefs, strict=True):
        if n == 1:
            d0 -= c * norm(n, z) * z
        elif n == 2:
            d0 += c * norm(n, z)
    return -d0 / r0
