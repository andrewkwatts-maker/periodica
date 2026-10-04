"""H2 / STO-3G at R = 1.4 bohr: published reference values plus an independent check.

Writes ``rust/periodica-qm/tests/fixtures/h2_sto3g/reference.json`` holding

* ``published`` - the values printed in Szabo & Ostlund section 3.5
  (4 decimals; transcribed by hand), and
* ``crosscheck`` - the same quantities evaluated in closed form by this script
  (minimal-basis H2 is fully determined by symmetry, so no SCF iteration is
  needed) for both the Szabo-Ostlund 6-digit STO-3G parameters and the Basis
  Set Exchange STO-3G parameters.  These are *derived* numbers, labelled as
  such, used to show the transcription and the target tolerance are consistent.

Usage::

    python tools/refdata/crosscheck_h2_sto3g.py
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from _common import REPO_ROOT, provenance, rel, write_json
from _gto_s import SShell, eri, kinetic, nuclear, overlap

R_BOHR = 1.4

# Szabo & Ostlund section 3.5: STO-3G for zeta = 1.24 (6 significant digits).
SZABO_EXPONENTS = (3.42525, 0.623913, 0.168856)
SZABO_COEFFICIENTS = (0.154329, 0.535328, 0.444635)

# Values printed in Szabo & Ostlund section 3.5 for R = 1.4 bohr, zeta = 1.24.
PUBLISHED = {
    "S12": 0.6593,
    "T11": 0.7600,
    "T12": 0.2365,
    "V1_11": -1.2266,
    "V1_12": -0.5974,
    "V1_22": -0.6538,
    "eri_11_11": 0.7746,
    "eri_11_22": 0.5697,
    "eri_21_11": 0.4441,
    "eri_21_21": 0.2970,
    "orbital_energy_1": -0.578,
    "orbital_energy_2": 0.670,
    "electronic_energy": -1.831,
    "total_energy": -1.117,
}
PUBLISHED_TOLERANCE = {k: (5e-4 if k in ("orbital_energy_1", "orbital_energy_2", "electronic_energy", "total_energy") else 5e-5) for k in PUBLISHED}


def _bse_hydrogen() -> tuple[tuple[float, ...], tuple[float, ...]]:
    path = REPO_ROOT / "rust" / "periodica-qm" / "data" / "basis" / "sto-3g.json"
    shell = json.loads(path.read_text(encoding="utf-8"))["elements"]["1"]["electron_shells"][0]
    return tuple(float(x) for x in shell["exponents"]), tuple(float(x) for x in shell["coefficients"][0])


def evaluate(exponents: tuple[float, ...], coefficients: tuple[float, ...]) -> dict[str, float]:
    """All S&O section 3.5 quantities for minimal-basis H2 at R_BOHR."""
    a, b = (0.0, 0.0, 0.0), (0.0, 0.0, R_BOHR)
    f1, f2 = SShell(a, exponents, coefficients), SShell(b, exponents, coefficients)
    s12 = overlap(f1, f2)
    nuc_a, nuc_b = [(1.0, a)], [(1.0, b)]
    h = [[kinetic(x, y) + nuclear(x, y, nuc_a + nuc_b) for y in (f1, f2)] for x in (f1, f2)]
    basis = (f1, f2)
    g = {(i, j, k, q): eri(basis[i], basis[j], basis[k], basis[q]) for i in range(2) for j in range(2) for k in range(2) for q in range(2)}
    cg = [1.0 / math.sqrt(2.0 * (1.0 + s12))] * 2
    cu = [1.0 / math.sqrt(2.0 * (1.0 - s12)), -1.0 / math.sqrt(2.0 * (1.0 - s12))]

    def one(c: list[float], d: list[float]) -> float:
        return sum(c[i] * d[j] * h[i][j] for i in range(2) for j in range(2))

    def two(c: list[float], d: list[float], e: list[float], f: list[float]) -> float:
        return sum(c[i] * d[j] * e[k] * f[q] * v for (i, j, k, q), v in g.items())

    j_gg = two(cg, cg, cg, cg)
    e_elec = 2.0 * one(cg, cg) + j_gg
    return {
        "S12": s12,
        "T11": kinetic(f1, f1),
        "T12": kinetic(f1, f2),
        "V1_11": nuclear(f1, f1, nuc_a),
        "V1_12": nuclear(f1, f2, nuc_a),
        "V1_22": nuclear(f2, f2, nuc_a),
        "eri_11_11": g[(0, 0, 0, 0)],
        "eri_11_22": g[(0, 0, 1, 1)],
        "eri_21_11": g[(1, 0, 0, 0)],
        "eri_21_21": g[(1, 0, 1, 0)],
        "orbital_energy_1": one(cg, cg) + j_gg,
        "orbital_energy_2": one(cu, cu) + 2.0 * two(cu, cu, cg, cg) - two(cu, cg, cg, cu),
        "electronic_energy": e_elec,
        "total_energy": e_elec + 1.0 / R_BOHR,
    }


def main() -> int:
    szabo = evaluate(SZABO_EXPONENTS, SZABO_COEFFICIENTS)
    bse = evaluate(*_bse_hydrogen())
    failures = [k for k, v in PUBLISHED.items() if abs(szabo[k] - v) > PUBLISHED_TOLERANCE[k]]
    for key, value in PUBLISHED.items():
        print(f"{key:18s} published {value:+.4f}  S&O-basis {szabo[key]:+.10f}  BSE-basis {bse[key]:+.12f}")
    if failures:
        print(f"MISMATCH between published and recomputed values: {failures}")
        return 1

    payload: dict[str, Any] = {
        "_provenance": provenance(
            source=(
                "A. Szabo and N. S. Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic "
                "Structure Theory (Dover, Mineola NY, 1996; unabridged republication of the 1989 first revised "
                "edition, McGraw-Hill), ISBN 0-486-69186-1, section 3.5 'Model calculations on H2 and HeH+' "
                "(STO-3G H2 at R = 1.4 bohr; total energy quoted as p. 167 by the secondary "
                "citation Psi4 forum thread https://forum.psicode.org/t/ground-state-energy-of-hydrogen-molecule-which-number-should-it-be/610). "
                "Integral values corroborated by the McCullagh-lab notebook "
                "https://mccullaghlab.github.io/computational_chemistry/quantum_mechanics/HF_for_H2.html."
            ),
            license_or_terms=(
                "Textbook numerical results (facts) reproduced with attribution; no text or figures copied. "
                "The page/equation numbers above are from secondary citations and should be checked against the book."
            ),
            transform="hand transcription (published); tools/refdata/crosscheck_h2_sto3g.py (crosscheck)",
            units="hartree (energies, integrals); bohr (R)",
            uncertainty=(
                "Published values are rounded to the printed digits: +/-5e-5 for integrals, +/-5e-4 for energies. "
                "The crosscheck values are exact to ~1e-12 for the stated basis parameters."
            ),
            conventions=(
                "Basis functions 1 and 2 are the STO-3G 1s functions on nucleus 1 at z = 0 and nucleus 2 at z = R. "
                "V1_ij is the attraction to nucleus 1 only (Szabo-Ostlund V^1). Two-electron integrals are in "
                "chemists' notation (ij|kl). Contracted functions are renormalised (the S&O 6-digit coefficients "
                "give <1|1> = 1.0000013 before renormalisation, which shifts E_tot by ~1e-6)."
            ),
        ),
        "system": {"molecule": "H2", "basis": "STO-3G (zeta = 1.24)", "method": "RHF", "R_bohr": R_BOHR},
        "published": PUBLISHED,
        "crosscheck": {
            "note": "Derived by this script, not literature values.",
            "szabo_ostlund_basis": {"exponents": SZABO_EXPONENTS, "coefficients": SZABO_COEFFICIENTS, "values": szabo},
            "bse_sto3g_basis": {"source": "rust/periodica-qm/data/basis/sto-3g.json (Z = 1)", "values": bse},
        },
    }
    path = REPO_ROOT / "rust" / "periodica-qm" / "tests" / "fixtures" / "h2_sto3g" / "reference.json"
    write_json(path, payload)
    print(f"wrote {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
