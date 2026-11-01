"""Validate the RHF-STO atomic tables (todo Q6).

For every atom in ``rust/periodica-qm/data/sto/{bunge1993,koga2000}.json``:

* each occupied orbital is normalised, <R|R> = 1, and orbitals of equal l are orthogonal;
* the electron count sum(occupations) equals Z;
* the radial expectation values recomputed from the STO expansion reproduce the
  values tabulated by the source (Bunge 1993 prints <r>, <r^2>, <1/r>, <1/r^2>, <1/r^3>);
* the printed 'RHOat0' and 'Kato cusp' (Bunge) are reproduced from the expansion;
* (build_sto_tables.py additionally compares Bunge 1993 orbital by orbital with
  the independent Koga et al. 1999 He-Xe wave functions: energies and <r>).

Usage::

    python tools/refdata/check_sto_tables.py            # all atoms, summary
    python tools/refdata/check_sto_tables.py He Be Ne   # detailed report for some atoms

Exit status is non-zero if any gate fails.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from _common import REPO_ROOT
from _sto import cusp, norm, overlap, radial_moment, value_at_origin

STO_DIR = REPO_ROOT / "rust" / "periodica-qm" / "data" / "sto"
MOMENTS = {"r": 1, "r2": 2, "r_inv": -1, "r_inv2": -2, "r_inv3": -3}

# Gates. Coefficients are printed to 6 (Bunge) or 7 (Koga) decimals, which bounds what can be reproduced.
NORM_TOL = 2e-5
ORTHO_TOL = 2e-5
# <r>, <r^2>, <1/r> are smooth; <1/r^2>, <1/r^3> are dominated by the nuclear region, where the
# 6-decimal rounding of tiny tight-function coefficients of valence orbitals shows up (~5e-5).
MOMENT_GROUP = {"r": "regular", "r2": "regular", "r_inv": "regular", "r_inv2": "singular", "r_inv3": "singular"}
MOMENT_REL_TOL = {"regular": 2e-5, "singular": 1e-4}
MOMENT_ABS_TOL = 1e-6  # tabulated values have 6 decimals, so tiny <r^2> of inner shells carry ~4 digits
SOURCE_TOL = 2e-5  # RHOat0 / Kato cusp printed to ~7 significant digits


def _orbital_arrays(atom: dict[str, Any], l_key: str) -> tuple[list[int], list[float]]:
    basis = atom["shells"][l_key]["basis"]
    return [b["n"] for b in basis], [b["zeta"] for b in basis]


def validate(atom: dict[str, Any], z: int) -> dict[str, Any]:
    """Return per-atom metrics; raises nothing (the caller applies the gates)."""
    worst = {"norm": 0.0, "ortho": 0.0, "regular": 0.0, "singular": 0.0, "cusp_rel": 0.0}
    detail: list[dict[str, Any]] = []
    electrons = 0.0
    for l_key, shell in atom["shells"].items():
        ns, zetas = _orbital_arrays(atom, l_key)
        orbitals = shell["orbitals"]
        for i, orb in enumerate(orbitals):
            electrons += orb["occupation"]
            c = orb["coefficients"]
            norm_err = abs(radial_moment(ns, zetas, c, 0) - 1.0)
            worst["norm"] = max(worst["norm"], norm_err)
            row: dict[str, Any] = {"orbital": orb["label"], "norm_error": norm_err}
            for other in orbitals[:i]:
                worst["ortho"] = max(worst["ortho"], abs(overlap(ns, zetas, c, other["coefficients"])))
            for key, value in (orb.get("expectation") or {}).items():
                calc = radial_moment(ns, zetas, c, MOMENTS[key])
                # excess over the printed precision, expressed relative to the value
                rel = max(0.0, abs(calc - value) - MOMENT_ABS_TOL) / abs(value)
                group = MOMENT_GROUP[key]
                worst[group] = max(worst[group], rel)
                row[key] = (value, calc)
            if l_key == "s":
                k = cusp(ns, zetas, c)
                row["cusp"] = k
                worst["cusp_rel"] = max(worst["cusp_rel"], abs(k - z) / z)
            detail.append(row)
    source = _source_scalars(atom, z)
    return {"worst": worst, "electrons": electrons, "detail": detail, "source": source}


def _source_scalars(atom: dict[str, Any], z: int) -> dict[str, float]:
    """Relative deviations of the recomputed Bunge 'RHOat0' and 'Kato cusp' from the printed values."""
    if "source_RHOat0" not in atom:
        return {}
    ns, zetas = _orbital_arrays(atom, "s")
    sum_r0, rho0, drho0 = 0.0, 0.0, 0.0
    for orb in atom["shells"]["s"]["orbitals"]:
        c = orb["coefficients"]
        r0 = value_at_origin(ns, zetas, c)
        dr0 = sum(
            -ci * norm(n, zeta) * zeta if n == 1 else (ci * norm(n, zeta) if n == 2 else 0.0)
            for n, zeta, ci in zip(ns, zetas, c, strict=True)
        )
        sum_r0 += r0 * r0
        rho0 += orb["occupation"] * r0 * r0
        drho0 += orb["occupation"] * 2.0 * r0 * dr0
    kato = -drho0 / (z * rho0)
    return {
        "RHOat0": abs(sum_r0 - atom["source_RHOat0"]) / atom["source_RHOat0"],
        "kato": abs(kato - atom["source_kato_cusp"]) / atom["source_kato_cusp"],
    }


def gates_ok(z: int, metrics: dict[str, Any]) -> list[str]:
    w = metrics["worst"]
    failures = []
    if abs(metrics["electrons"] - z) > 1e-9:
        failures.append(f"electron count {metrics['electrons']} != Z")
    if w["norm"] > NORM_TOL:
        failures.append(f"normalisation error {w['norm']:.2e}")
    if w["ortho"] > ORTHO_TOL:
        failures.append(f"orthogonality error {w['ortho']:.2e}")
    for group, tol in MOMENT_REL_TOL.items():
        if w[group] > tol:
            failures.append(f"{group} <r^k> mismatch {w[group]:.2e}")
    for key, dev in metrics["source"].items():
        if dev > SOURCE_TOL:
            failures.append(f"recomputed {key} deviates by {dev:.2e}")
    return failures


def main(argv: list[str]) -> int:
    wanted = set(argv)
    failures = 0
    for name in ("bunge1993", "koga2000"):
        path = STO_DIR / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = {"norm": 0.0, "ortho": 0.0, "regular": 0.0, "singular": 0.0, "cusp_rel": 0.0}
        for z_text, atom in data["atoms"].items():
            z = int(z_text)
            metrics = validate(atom, z)
            for k in summary:
                summary[k] = max(summary[k], metrics["worst"][k])
            problems = gates_ok(z, metrics)
            if problems:
                failures += 1
                print(f"FAIL {name} {atom['symbol']}: {'; '.join(problems)}")
            if atom["symbol"] in wanted:
                print(f"\n{name} {atom['symbol']} (Z={z}) E = {atom['total_energy']}")
                for row in metrics["detail"]:
                    extras = "  ".join(
                        f"{k}={v[0]:.6f}/{v[1]:.6f}" if isinstance(v, tuple) else f"{k}={v:.6f}"
                        for k, v in row.items()
                        if k not in ("orbital", "norm_error")
                    )
                    print(f"  {row['orbital']:>3}  |<R|R>-1| = {row['norm_error']:.2e}  {extras}")
        print(
            f"{name}: {len(data['atoms'])} atoms; worst |<R|R>-1| = {summary['norm']:.2e}, "
            f"worst |<R_i|R_j>| = {summary['ortho']:.2e}, worst rel. dev. <r>,<r2>,<1/r> = {summary['regular']:.2e}, "
            f"<1/r2>,<1/r3> = {summary['singular']:.2e}, "
            f"worst cusp rel. dev. from Z = {summary['cusp_rel']:.2e}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
