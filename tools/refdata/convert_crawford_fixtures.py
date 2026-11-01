"""Convert the Crawford-group SCF test cases into periodica-qm test fixtures (todo Q1-alt).

PySCF does not install on the development machine, so the RHF regression
fixtures come from the Crawford Group "Programming Projects" (Projects #3 SCF
and #4 MP2), which publish Psi3-computed AO integrals and reference energies
for H2O/STO-3G and CH4/STO-3G.

For each molecule this writes, under ``rust/periodica-qm/tests/fixtures/<mol>_sto3g/``::

    geometry.json      nuclei (Z, bohr) + nuclear repulsion energy
    basis.json         the exact STO-3G parameters Psi3 used (BSE-style shells)
    one_electron.json  full S, T, V and dipole-integral matrices (0-based AO order)
    eri.json           permutationally unique (ij|kl), 0-based, chemists' notation
    reference.json     SCF / MP2 energies, dipole moment, Mulliken charges

and validates: AO ordering against the geometry, the 8-fold ERI index
restrictions, S diagonal = 1, and the s-type integral blocks against an
independent closed-form evaluation with the Basis Set Exchange STO-3G set.

Usage::

    python tools/refdata/convert_crawford_fixtures.py
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from _common import REPO_ROOT, fetch, provenance, rel, sha256, write_json
from _gto_s import SShell, eri, kinetic, nuclear, overlap

REPO = "CrawfordGroup/ProgrammingProjects"
COMMIT = "297fda143186f08a3a9481b0fb762f87bbbba524"  # master as retrieved 2026-10-02
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
FILES = ("geom", "enuc", "s", "t", "v", "mux", "muy", "muz", "eri", "input")
SYMBOL = {1: "H", 6: "C", 8: "O"}
ELEMENT_NAME = {1: "HYDROGEN", 6: "CARBON", 8: "OXYGEN"}

# Psi3's built-in basis library, used for H2O (whose Crawford input has no basis block).
PSI3_COMMIT = "b74be464b48d475694638e30b1327101c2a37994"  # last change to lib/pbasis.dat (2009-09-09)
PSI3_PBASIS = f"https://raw.githubusercontent.com/psi4/psi3/{PSI3_COMMIT}/lib/pbasis.dat"
PSI3_TERMS = (
    "psi4/psi3 repository, GPL-2.0. Only the numerical STO-3G exponents/contraction coefficients for the "
    "elements present (published basis-set data, Hehre, Stewart & Pople 1969, doi:10.1063/1.1672392) are "
    "reproduced; no Psi3 code is copied."
)

MOLECULES: dict[str, dict[str, Any]] = {
    "h2o": {
        "dir": "h2o_sto3g",
        "formula": "H2O",
        "geometry_note": "Psi3 z-matrix ((o) (h 1 1.1) (h 1 1.1 2 104.0)): R(OH) = 1.1 Angstrom, HOH = 104.0 deg "
        "(a test geometry, NOT the experimental equilibrium), centre-of-mass frame, bohr.",
        # AO order verified below from overlap signs vs. geometry.
        "ao_labels": ["O 1s", "O 2s", "O 2px", "O 2py", "O 2pz", "H1 1s", "H2 1s"],
        "ao_atoms": [0, 0, 0, 0, 0, 1, 2],
        "basis_source": "psi3_pbasis",
    },
    "ch4": {
        "dir": "ch4_sto3g",
        "formula": "CH4",
        "geometry_note": "Psi3 z-matrix: r(CH) = 1.085 Angstrom, tetrahedral (td = 109.4712206 deg), bohr.",
        "ao_labels": ["C 1s", "C 2s", "C 2px", "C 2py", "C 2pz", "H1 1s", "H2 1s", "H3 1s", "H4 1s"],
        "ao_atoms": [0, 0, 0, 0, 0, 1, 2, 3, 4],
        "basis_source": "input",
    },
}
P_AXES = {"2px": 0, "2py": 1, "2pz": 2}

LICENCE = (
    "No licence declared: the repository has no LICENSE file and the GitHub API reports license = null "
    "(checked 2026-10-02), so default copyright applies to its text and code. The values used here are "
    "machine-computed numerical results (AO integrals and energies), i.e. facts rather than creative "
    "expression, reproduced with attribution for regression testing only; no prose or code is copied. "
    "If a formally licensed fixture is required, regenerate the same quantities with PySCF (Apache-2.0) "
    "on a platform where it installs and diff against these files."
)
SOURCE = (
    "T. D. Crawford group, 'Programming Projects' (Virginia Tech), Project #3 'The Hartree-Fock SCF procedure' "
    "and Project #4 'MP2 energy', https://github.com/CrawfordGroup/ProgrammingProjects "
    f"(commit {COMMIT}); input integrals computed with Psi3 (STO-3G, RHF, c1 symmetry). "
    "Original project by Y. Yamaguchi (University of Georgia), per the Project #3 README."
)
CONVENTIONS = {
    "index_base": "0 (the source files are 1-based; converted here).",
    "ao_basis": "Real Cartesian STO-3G contracted Gaussians, each contraction normalised to 1 (S_ii = 1); "
    "AO order and labels in 'ao_labels' (p order px, py, pz), verified from overlap signs vs. geometry.",
    "basis_parameters": "The exact STO-3G parameters Psi3 used are in basis.json (H2O: Psi3 lib/pbasis.dat; "
    "CH4: the explicit basis block of the Crawford input, which differs from pbasis.dat in the 8th digit). "
    "With these parameters the s-type integrals are reproduced to ~1e-12 (see 'crosscheck' in "
    "one_electron.json); with the BSE STO-3G set they differ by up to ~4e-6 hartree, so a test that uses "
    "BSE STO-3G must use a correspondingly loose tolerance, or load basis.json for a 1e-10 comparison.",
    "one_electron": "Full symmetric n x n matrices (row-major lists) rebuilt from the permutationally unique "
    "lower triangle in s.dat/t.dat/v.dat. V includes attraction to all nuclei.",
    "dipole_integrals": "mux/muy/muz.dat as distributed by Crawford; the sign convention is recorded in "
    "'dipole_sign_convention' (checked against the reported dipole).",
    "eri": "Chemists'/Mulliken notation (ij|kl) = int phi_i(1) phi_j(1) r12^-1 phi_k(2) phi_l(2). Only the "
    "permutationally unique integrals with i >= j, k >= l, ij >= kl (compound index ij = i(i+1)/2 + j, 0-based) "
    "are listed. Integrals absent from the list are zero (by symmetry or below Psi3's print threshold).",
    "geometry": "Bohr. Nuclear charges Z; the electronic-structure frame used for the integrals.",
}


def _download(mol: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for name in FILES:
        url = f"{RAW}/Project%2303/input/{mol}/STO-3G/{name}.dat"
        data = fetch(url)
        texts[name] = data.decode("utf-8")
        digests[name] = sha256(data)
    for project in ("03", "04"):
        data = fetch(f"{RAW}/Project%23{project}/output/{mol}/STO-3G/output.txt")
        texts[f"output{project}"] = data.decode("utf-8")
        digests[f"output{project}"] = sha256(data)
    texts["_digests"] = json.dumps(digests)
    return texts


_SHELL = re.compile(r"\((S|P|D)\s+((?:\(\s*[-+\d.Ee]+\s+[-+\d.Ee]+\s*\)\s*)+)\)")
_PRIM = re.compile(r"\(\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s*\)")
_AM = {"S": 0, "P": 1, "D": 2}


def _parse_psi3_block(block: str) -> list[dict[str, Any]]:
    """Parse one Psi3 basis block '((S (e c) ...) (P ...))' into BSE-style shells (strings kept)."""
    shells = []
    for kind, body in _SHELL.findall(block):
        prims = _PRIM.findall(body)
        shells.append(
            {
                "function_type": "gto",
                "angular_momentum": [_AM[kind]],
                "exponents": [e for e, _ in prims],
                "coefficients": [[c for _, c in prims]],
            }
        )
    return shells


def _psi3_basis(spec: dict[str, Any], atoms: list[dict[str, Any]], input_text: str) -> tuple[dict[str, Any], str, str]:
    """Return (BSE-style elements dict, source description, terms) for the basis Psi3 actually used."""
    elements: dict[str, Any] = {}
    zs = sorted({a["Z"] for a in atoms})
    if spec["basis_source"] == "input":
        for z in zs:
            name = ELEMENT_NAME[z].lower()
            match = re.search(name + r':\s*"STO-3G"\s*=\s*\((.*?)\n\s*\)', input_text, re.S)
            if not match:
                raise SystemExit(f"basis block for {name} not found in input.dat")
            elements[str(z)] = {"electron_shells": _parse_psi3_block(match.group(1))}
        return elements, "Explicit basis block of the Crawford ch4/STO-3G input.dat (Psi3 input).", LICENCE
    text = fetch(PSI3_PBASIS).decode("utf-8")
    for z in zs:
        match = re.search(r"\n\s*" + ELEMENT_NAME[z] + r':"STO-3G" = \((.*?)\n\s*\)', text, re.S)
        if not match:
            raise SystemExit(f"{ELEMENT_NAME[z]} STO-3G not found in pbasis.dat")
        elements[str(z)] = {"electron_shells": _parse_psi3_block(match.group(1))}
    return elements, f"Psi3 basis library lib/pbasis.dat, {PSI3_PBASIS}", PSI3_TERMS


def _matrix(text: str, n: int) -> list[list[float]]:
    m = [[0.0] * n for _ in range(n)]
    for line in text.split("\n"):
        if line.strip():
            i, j, v = line.split()
            a, b = int(i) - 1, int(j) - 1
            m[a][b] = m[b][a] = float(v)
    return m


def _eri(text: str) -> list[list[float | int]]:
    out: list[list[float | int]] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        i, j, k, q, v = line.split()
        a, b, c, d = int(i) - 1, int(j) - 1, int(k) - 1, int(q) - 1
        ij, kl = a * (a + 1) // 2 + b, c * (c + 1) // 2 + d
        if not (a >= b and c >= d and ij >= kl):
            raise SystemExit(f"ERI ({i} {j}|{k} {q}) violates the unique-index restriction")
        out.append([a, b, c, d, float(v)])
    return out


def _geometry(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.split("\n") if line.strip()]
    atoms = []
    for line in lines[1 : 1 + int(lines[0])]:
        z, x, y, w = (float(t) for t in line.split())
        atoms.append({"Z": int(z), "symbol": SYMBOL[int(z)], "x": x, "y": y, "z": w})
    return atoms


def _number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise SystemExit(f"pattern {pattern!r} not found in reference output")
    return float(match.group(1))


def _check_ao_order(spec: dict[str, Any], atoms: list[dict[str, Any]], s: list[list[float]]) -> None:
    heavy = atoms[0]
    for p_index, label in enumerate(spec["ao_labels"]):
        axis = P_AXES.get(label.split()[-1])
        if axis is None:
            continue
        for h_index, atom_index in enumerate(spec["ao_atoms"]):
            if atom_index == 0:
                continue
            d = [atoms[atom_index][c] - heavy[c] for c in ("x", "y", "z")][axis]
            if abs(d) > 1e-6 and (s[p_index][h_index] > 0) != (d > 0):
                raise SystemExit(f"AO order check failed: {label} vs AO {h_index}")


def _s_block_diff(
    basis: dict[str, Any],
    spec: dict[str, Any],
    atoms: list[dict[str, Any]],
    s: list[list[float]],
    t: list[list[float]],
    v: list[list[float]],
    eris: list[list[float | int]],
) -> dict[str, float]:
    """Max |diff| of the s-type AO blocks vs. a closed-form evaluation in ``basis`` (BSE-style elements)."""
    shells: dict[int, SShell] = {}
    for ao, (label, atom_index) in enumerate(zip(spec["ao_labels"], spec["ao_atoms"], strict=True)):
        orbital = label.split()[-1]
        if not orbital.endswith("s"):
            continue
        atom = atoms[atom_index]
        el_shells = basis[str(atom["Z"])]["electron_shells"]
        shell = el_shells[0] if orbital == "1s" else el_shells[1]
        shells[ao] = SShell(
            (atom["x"], atom["y"], atom["z"]),
            tuple(float(e) for e in shell["exponents"]),
            tuple(float(c) for c in shell["coefficients"][0]),
        )
    charges = [(float(a["Z"]), (a["x"], a["y"], a["z"])) for a in atoms]
    diff = {"S": 0.0, "T": 0.0, "V": 0.0, "ERI": 0.0}
    for i in shells:
        for j in shells:
            diff["S"] = max(diff["S"], abs(overlap(shells[i], shells[j]) - s[i][j]))
            diff["T"] = max(diff["T"], abs(kinetic(shells[i], shells[j]) - t[i][j]))
            diff["V"] = max(diff["V"], abs(nuclear(shells[i], shells[j], charges) - v[i][j]))
    listed = {(int(a), int(b), int(c), int(d)): float(x) for a, b, c, d, x in eris}
    keys = sorted(shells)
    for a in keys:
        for b in keys:
            for c in keys:
                for d in keys:
                    if a >= b and c >= d and a * (a + 1) // 2 + b >= c * (c + 1) // 2 + d:
                        ref = listed.get((a, b, c, d), 0.0)
                        diff["ERI"] = max(diff["ERI"], abs(eri(shells[a], shells[b], shells[c], shells[d]) - ref))
    return {k: float(f"{x:.3e}") for k, x in diff.items()}


def _dipole_sign(atoms: list[dict[str, Any]], mu: list[list[list[float]]], reported: list[float], output: str, n: int) -> str:
    """Decide whether the dipole integrals already carry the electron charge (-1)."""
    match = re.findall(r"\n\s+1\s+2\s+3.*?\n\n((?:\s+\d+(?:\s+-?\d+\.\d+)+\n)+)", output)
    if not match:
        return "not determined (density matrix not found in output)"
    density_block = match[-1]
    rows = [line.split()[1:] for line in density_block.strip().split("\n")]
    if len(rows) != n or any(len(r) != n for r in rows):
        return "not determined (density block shape mismatch)"
    d = [[float(x) for x in r] for r in rows]
    for axis, key in enumerate(("x", "y", "z")):
        nuclear_part = sum(a["Z"] * a[key] for a in atoms)
        trace = 2.0 * sum(d[i][j] * mu[axis][i][j] for i in range(n) for j in range(n))
        if abs(reported[axis]) > 1e-3:
            if abs(nuclear_part + trace - reported[axis]) < 1e-5:
                return "integrals are -<i|r|j> (electron charge included): mu = 2 sum_ij D_ij mu_ij + sum_A Z_A R_A"
            if abs(nuclear_part - trace - reported[axis]) < 1e-5:
                return "integrals are +<i|r|j> (position integrals): mu = -2 sum_ij D_ij mu_ij + sum_A Z_A R_A"
    return "not determined"


def build(mol: str) -> None:
    spec = MOLECULES[mol]
    texts = _download(mol)
    digests = json.loads(texts.pop("_digests"))
    atoms = _geometry(texts["geom"])
    n = len(spec["ao_labels"])
    s, t, v = (_matrix(texts[k], n) for k in ("s", "t", "v"))
    mu = [_matrix(texts[k], n) for k in ("mux", "muy", "muz")]
    eris = _eri(texts["eri"])
    if any(abs(s[i][i] - 1.0) > 1e-12 for i in range(n)):
        raise SystemExit(f"{mol}: overlap diagonal is not 1")
    _check_ao_order(spec, atoms, s)

    out3, out4 = texts["output03"], texts["output04"]
    e_nuc = float(texts["enuc"].split()[0])
    dipole = [_number(rf"Mu-{c} =\s+(-?\d+\.\d+)", out3) for c in ("X", "Y", "Z")]
    charges = [float(x) for x in re.findall(r"Charge on atom \d+:\s+(-?\d+\.\d+)", out3)]
    iterations = re.findall(r"\n\s*(\d+)\s+(-?\d+\.\d{12})\s+(-?\d+\.\d{12})\s+(-?\d+\.\d{12})\s+(-?\d+\.\d{12})", out3)
    last = iterations[-1]
    e_scf = _number(r"Escf =\s+(-?\d+\.\d+)", out4)
    if abs(float(last[2]) - e_scf) > 1e-12:
        raise SystemExit(f"{mol}: Project 3 final E(tot) {last[2]} != Project 4 Escf {e_scf}")

    base = {
        "source": SOURCE,
        "license_or_terms": LICENCE,
        "transform": "tools/refdata/convert_crawford_fixtures.py",
        "download_sha256": digests,
        "conventions": CONVENTIONS,
    }

    def prov(units: str | dict[str, str], **extra: Any) -> dict[str, Any]:
        b = dict(base)
        return provenance(
            source=b.pop("source"),
            license_or_terms=b.pop("license_or_terms"),
            transform=b.pop("transform"),
            units=units,
            **b,
            **extra,
        )

    out_dir = REPO_ROOT / "rust" / "periodica-qm" / "tests" / "fixtures" / spec["dir"]
    system = {"molecule": spec["formula"], "basis": "STO-3G (Psi3)", "method": "RHF", "n_ao": n, "n_electrons": sum(a["Z"] for a in atoms)}
    psi_input = texts["input"].strip()

    write_json(
        out_dir / "geometry.json",
        {
            "_provenance": prov({"coordinates": "bohr", "nuclear_repulsion": "hartree"}, uncertainty="Exact inputs (defined geometry)."),
            "system": system,
            "geometry_note": spec["geometry_note"],
            "psi3_input": psi_input.split("\n"),
            "atoms": atoms,
            "nuclear_repulsion": e_nuc,
        },
    )
    psi3_elements, basis_source, basis_terms = _psi3_basis(spec, atoms, texts["input"])
    bse_path = REPO_ROOT / "rust" / "periodica-qm" / "data" / "basis" / "sto-3g.json"
    bse_elements = json.loads(bse_path.read_text(encoding="utf-8"))["elements"]
    crosscheck = {
        "note": "Max |difference| (hartree) between the distributed integrals and an independent closed-form "
        "evaluation (tools/refdata/_gto_s.py) over the s-type AOs only (S, T, V and all-s ERIs).",
        "s_type_aos": [i for i, lab in enumerate(spec["ao_labels"]) if lab.endswith("s")],
        "with_basis_json": _s_block_diff(psi3_elements, spec, atoms, s, t, v, eris),
        "with_bse_sto3g": _s_block_diff(bse_elements, spec, atoms, s, t, v, eris),
    }
    if max(crosscheck["with_basis_json"].values()) > 1e-10:
        raise SystemExit(f"{mol}: integrals not reproduced with the Psi3 basis: {crosscheck['with_basis_json']}")
    write_json(
        out_dir / "basis.json",
        {
            "_provenance": provenance(
                source=basis_source,
                license_or_terms=basis_terms,
                transform="tools/refdata/convert_crawford_fixtures.py (Psi3 basis syntax -> BSE-style shells; "
                "numbers kept as strings)",
                units={"exponents": "bohr^-2", "coefficients": "dimensionless (normalised primitives)"},
                uncertainty="Not applicable (defined parameters).",
                note="The exact STO-3G parameters used to compute these fixtures. Normalise each primitive, then "
                "renormalise each contraction: this reproduces S_ii = 1 and the s-type integrals to ~1e-12.",
            ),
            "name": "STO-3G (Psi3)",
            "elements": psi3_elements,
        },
    )
    sign = _dipole_sign(atoms, mu, dipole, out3, n)
    if sign == "not determined" and mol == "ch4":
        sign = "zero dipole; same generator and convention as h2o_sto3g (integrals are -<i|r|j>)"
    write_json(
        out_dir / "one_electron.json",
        {
            "_provenance": prov("hartree (S dimensionless, dipole integrals e*bohr)", uncertainty="Printed to 15 decimals; Psi3 integral accuracy ~1e-12.", dipole_sign_convention=sign),
            "system": system,
            "ao_labels": spec["ao_labels"],
            "ao_atom_index": spec["ao_atoms"],
            "overlap": s,
            "kinetic": t,
            "nuclear_attraction": v,
            "dipole_x": mu[0],
            "dipole_y": mu[1],
            "dipole_z": mu[2],
            "crosscheck": crosscheck,
        },
    )
    write_json(
        out_dir / "eri.json",
        {
            "_provenance": prov("hartree", uncertainty="Printed to 15 decimals; Psi3 integral accuracy ~1e-12."),
            "system": system,
            "n_listed": len(eris),
            "n_unique_total": (n * (n + 1) // 2) * (n * (n + 1) // 2 + 1) // 2,
            "eri": eris,
        },
        indent=None,
    )
    write_json(
        out_dir / "reference.json",
        {
            "_provenance": prov(
                {"energies": "hartree", "dipole": "atomic units (e*bohr)", "charges": "e"},
                uncertainty="SCF converged to Delta(E) ~1e-12 and RMS(D) ~1e-11 per the reference output; "
                "values printed to 12 decimals.",
            ),
            "system": system,
            "scf_total_energy": e_scf,
            "scf_electronic_energy": float(last[1]),
            "nuclear_repulsion": e_nuc,
            "scf_iterations_reference": int(last[0]),
            "mp2_correlation_energy": _number(r"Emp2 =\s+(-?\d+\.\d+)", out4),
            "mp2_total_energy": _number(r"Etot =\s+(-?\d+\.\d+)", out4),
            "dipole_moment_au": dipole,
            "dipole_total_au": _number(r"Total dipole moment \(au\) =\s+(-?\d+\.\d+)", out3),
            "mulliken_charges": charges,
            "notes": "MP2 correlates all occupied MOs (no frozen core; the Project #4 formula sums over every "
            "occupied orbital). Only converged quantities are fixtures; the iteration count depends on the "
            "core-Hamiltonian guess without DIIS used by Project #3.",
        },
    )
    print(f"{rel(out_dir)}: n_ao={n}, eri listed={len(eris)}, Escf={e_scf}, dipole sign: {sign}")
    print(f"  s-block crosscheck: basis.json {crosscheck['with_basis_json']}  BSE {crosscheck['with_bse_sto3g']}")


def main() -> int:
    for mol in MOLECULES:
        build(mol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
