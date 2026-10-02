"""Atomic Roothaan-Hartree-Fock Slater-type-orbital tables (todo Q6).

Writes ``rust/periodica-qm/data/sto/bunge1993.json`` (He-Xe, Z = 2-54) and
``rust/periodica-qm/data/sto/koga2000.json`` (Cs-Lr, Z = 55-103):

* Bunge, Barrientos & Bunge 1993 - the authors' ASCII file RHF.TABLES, announced
  for free anonymous-FTP distribution by C. F. Bunge (CCL list, 13 Sep 1993)
  and mirrored by the Computational Chemistry List.
* Koga, Kanayama, Watanabe, Imai & Thakkar 2000 - the authors' per-atom files
  (A. J. Thakkar's download page, stf/k99heavy.zip, now offline), obtained from
  the AtomDB redistribution (theochem/AtomDB, atomdb/data/slater_atom.tar.xz).

The Koga et al. 1999 cusp-constrained He-Xe functions from the same archive
are parsed only as an independent cross-check of Bunge 1993 (not stored).

Then runs the checks of ``check_sto_tables.py`` (normalisation, orthogonality,
electron count, tabulated <r^k>) and records the worst deviations.

Usage::

    python tools/refdata/build_sto_tables.py
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
from typing import Any

from _common import REPO_ROOT, SYMBOLS, fetch, provenance, rel, sha256, write_json
from _sto import radial_moment
from check_sto_tables import gates_ok, validate

OUT_DIR = REPO_ROOT / "rust" / "periodica-qm" / "data" / "sto"
BUNGE_URL = "https://server.ccl.net/cca/data/atomic-RHF-wavefunctions/tables"
BUNGE_README = "https://server.ccl.net/cca/data/atomic-RHF-wavefunctions/README"
ATOMDB_COMMIT = "9562659f51198860012f1e4edfc4792f4754d56b"
ATOMDB_TAR = f"https://raw.githubusercontent.com/theochem/AtomDB/{ATOMDB_COMMIT}/atomdb/data/slater_atom.tar.xz"

L_KEYS = "spdf"
CLOSED = {"s": 2, "p": 6, "d": 10, "f": 14}
CORES = {
    "He": "1s(2)",
    "Ne": "[He]2s(2)2p(6)",
    "Ar": "[Ne]3s(2)3p(6)",
    "Kr": "[Ar]3d(10)4s(2)4p(6)",
    "Xe": "[Kr]4d(10)5s(2)5p(6)",
    "Rn": "[Xe]4f(14)5d(10)6s(2)6p(6)",
}
SHELL_NOTATION = {"K": "1s(2)", "L": "2s(2)2p(6)", "M": "3s(2)3p(6)3d(10)"}
CONVENTIONS = {
    "sto": "chi(r) = N r^(n-1) exp(-zeta r) Y_lm(theta, phi) with N = (2 zeta)^(n+1/2) / sqrt((2n)!): coefficients "
    "multiply NORMALISED STOs (verified: every orbital integrates to 1 within the printed precision).",
    "orbital": "R_nl(r) = sum_i c_i N_i r^(n_i - 1) exp(-zeta_i r); shells[l].basis lists (n_i, zeta_i) in the "
    "source order; each orbital's coefficients follow that order.",
    "density": "Spherically averaged density rho(r) = (1/4pi) sum_orbitals occupation * R(r)^2 (open shells "
    "averaged over the configuration, as in the source RHF).",
    "units": "Atomic units: zeta in bohr^-1, energies in hartree, <r^k> in bohr^k.",
}


def _expand_configuration(config: str) -> dict[str, float]:
    """'[Ar]3d(10)4s(2)' -> {'1s': 2, ..., '3d': 10, '4s': 2} (labels lower-case)."""
    occ: dict[str, float] = {}
    rest = config
    match = re.match(r"\[(\w+)\]", rest)
    if match:
        occ.update(_expand_configuration(CORES[match.group(1).capitalize()]))
        rest = rest[match.end() :]
    for shell in re.findall(r"([KLM])\(\d+\)", rest):  # closed K/L/M shells (Koga 1999 notation)
        occ.update(_expand_configuration(SHELL_NOTATION[shell]))
    for label, count in re.findall(r"(\d[spdfSPDF])\((\d+)\)", rest):
        occ[label.lower()] = float(count)
    return occ


# --- Bunge 1993 ------------------------------------------------------------------------------
_QUANTITY = {"ORB.ENERGY": "energy", "<R>": "r", "<R**2>": "r2", "<1/R>": "r_inv", "<1/R**2>": "r_inv2", "<1/R**3>": "r_inv3"}
_QUANTITY_RE = re.compile(r"ORB\.ENERGY|<R>|<R\*\*2>|<1/R>|<1/R\*\*2>|<1/R\*\*3>")
_NUMBER_RE = re.compile(r"-?\d*\.\d+")  # the Koga files omit the leading zero ("-.1178932")


def _parse_bunge_atom(lines: list[str]) -> dict[str, Any]:
    title = re.match(r"^([A-Z]+), Z=(\d+)\s+(.*)$", lines[0])
    assert title is not None
    name, z, rest = title.group(1), int(title.group(2)), title.group(3).split()
    config = rest[0]
    term = rest[1] if len(rest) > 1 else None
    atom: dict[str, Any] = {"symbol": SYMBOLS[z], "name": name.capitalize(), "configuration": config, "term": term}
    orbitals: dict[str, dict[str, Any]] = {}
    basis: dict[str, list[dict[str, Any]]] = {}
    header_cols: list[tuple[int, str]] = []
    sign_restored: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("TOTAL ENERGY"):
            e, t, v, vr = (float(x) for x in _NUMBER_RE.findall(lines[i + 1]))
            atom.update(total_energy=e, kinetic_energy=t, potential_energy=v, virial_ratio=vr)
        elif line.startswith("RHOat0"):
            rho, kato = (float(x) for x in _NUMBER_RE.findall(line))
            atom.update(source_RHOat0=rho, source_kato_cusp=kato)
        elif re.fullmatch(r"\s+(\d[spdf]\s*)+", line):
            header_cols = [(m.end(), m.group(0)) for m in re.finditer(r"\d[spdf]", line)]
            for _, label in header_cols:
                orbitals.setdefault(label, {"label": label, "expectation": {}})
        elif _QUANTITY_RE.match(line):
            tags = [(m.start(), _QUANTITY[m.group(0)]) for m in _QUANTITY_RE.finditer(line)]
            for m in _NUMBER_RE.finditer(line):
                quantity = [q for start, q in tags if start < m.start()][-1]
                label = min(header_cols, key=lambda hc, end=m.end(): abs(hc[0] - (end - 4)))[1]
                value = float(m.group(0))
                if quantity == "energy":
                    if value > 0:  # fixed-width overflow dropped the minus sign (e.g. Xe 1s)
                        value = -value
                        sign_restored.append(label)
                    orbitals[label]["energy"] = value
                else:
                    orbitals[label]["expectation"][quantity] = value
        elif re.match(r"^\d[SPDF]\s", line):
            tokens = line.split()
            starts = [k for k, tok in enumerate(tokens) if re.fullmatch(r"\d[SPDF]", tok)]
            for a, b in zip(starts, starts[1:] + [len(tokens)], strict=True):
                n, l_key = int(tokens[a][0]), tokens[a][1].lower()
                zeta, coefs = float(tokens[a + 1]), [float(x) for x in tokens[a + 2 : b]]
                basis.setdefault(l_key, []).append({"n": n, "zeta": zeta})
                members = [lab for lab in orbitals if lab[1] == l_key]
                if len(coefs) != len(members):
                    raise SystemExit(f"Bunge Z={z}: {len(coefs)} coefficients for {len(members)} {l_key} orbitals")
                for label, c in zip(members, coefs, strict=True):
                    orbitals[label].setdefault("coefficients", []).append(c)
    occupations = _expand_configuration(config)
    atom["shells"] = {}
    for l_key in L_KEYS:
        if l_key not in basis:
            continue
        members = [orbitals[lab] for lab in orbitals if lab[1] == l_key]
        for orb in members:
            orb["occupation"] = occupations[orb["label"]]
            orb["l"] = L_KEYS.index(l_key)
        atom["shells"][l_key] = {"basis": basis[l_key], "orbitals": [_ordered(o) for o in members]}
    if set(occupations) != set(orbitals):
        raise SystemExit(f"Bunge Z={z}: configuration {sorted(occupations)} vs orbitals {sorted(orbitals)}")
    if term is None:
        atom["term_note"] = "No term symbol printed in the source file."
    if sign_restored:
        atom["note"] = f"Orbital energy sign restored for {', '.join(sign_restored)} (fixed-width overflow in the source file)."
    return atom


def _ordered(orb: dict[str, Any]) -> dict[str, Any]:
    keys = ("label", "l", "occupation", "energy", "coefficients", "expectation")
    return {k: orb[k] for k in keys if k in orb}


def parse_bunge(text: str) -> dict[int, dict[str, Any]]:
    lines = [ln.rstrip() for ln in text.replace("\r", "").split("\n")]
    lines = [ln for ln in lines if not re.fullmatch(r"\s*\d+", ln)]  # page numbers
    starts = [i for i, ln in enumerate(lines) if re.match(r"^[A-Z]+, Z=\d+", ln)]
    atoms = {}
    for a, b in zip(starts, starts[1:] + [len(lines)], strict=True):
        atom = _parse_bunge_atom(lines[a:b])
        atoms[SYMBOLS.index(atom["symbol"])] = atom
    return atoms


# --- Koga (Thakkar group file format) -----------------------------------------------------------
def parse_koga_file(text: str) -> dict[str, Any]:
    lines = [ln.rstrip() for ln in text.replace("\r", "").split("\n")]
    head = re.match(r"^\s*([A-Z]+)\s+(\S+),\s*(\w+)", lines[0])
    assert head is not None
    name, config, term = head.groups()
    atom: dict[str, Any] = {"name": name.capitalize(), "configuration": re.sub(r"\[(\w+)\]", lambda m: f"[{m.group(1).capitalize()}]", config).replace("S(", "s(").replace("P(", "p(").replace("D(", "d(").replace("F(", "f("), "term": term}
    shells: dict[str, dict[str, Any]] = {}
    current: tuple[str, list[str]] | None = None
    for line in lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "E" and "=" in line and "total_energy" not in atom:
            atom["total_energy"] = float(_NUMBER_RE.findall(line)[0])
        elif tokens[0] == "T" and "V/T" in line:
            t, v, vt = (float(x) for x in _NUMBER_RE.findall(line))
            atom.update(kinetic_energy=t, potential_energy=v, virial_ratio=vt)
        elif tokens[0] in ("S", "P", "D", "F") and all(re.fullmatch(r"\d[SPDF]", t) for t in tokens[1:]):
            l_key = tokens[0].lower()
            labels = [t.lower() for t in tokens[1:]]
            shells[l_key] = {"basis": [], "orbitals": [{"label": lab, "l": L_KEYS.index(l_key), "coefficients": []} for lab in labels]}
            current = (l_key, labels)
        elif tokens[0] == "BASIS/ORB.ENERGY" and current:
            for orb, e in zip(shells[current[0]]["orbitals"], _NUMBER_RE.findall(line), strict=True):
                orb["energy"] = float(e)
        elif tokens[0] == "CUSP" and current:
            for orb, k in zip(shells[current[0]]["orbitals"], _NUMBER_RE.findall(line), strict=True):
                orb["cusp_ratio"] = float(k)
        elif re.fullmatch(r"\d[SPDF]", tokens[0]) and current:
            l_key = tokens[0][1].lower()
            if l_key != current[0]:
                raise SystemExit(f"Koga {name}: {tokens[0]} row inside the {current[0]} block")
            shells[l_key]["basis"].append({"n": int(tokens[0][0]), "zeta": float(tokens[1])})
            coefs = [float(x) for x in tokens[2:]]
            for orb, c in zip(shells[l_key]["orbitals"], coefs, strict=True):
                orb["coefficients"].append(c)
    occupations = _expand_configuration(atom["configuration"])
    for l_key, shell in shells.items():
        kept = []
        for orb in shell["orbitals"]:
            occ = occupations.get(orb["label"])
            if occ is None:
                raise SystemExit(f"Koga {name}: orbital {orb['label']} not in configuration {atom['configuration']}")
            orb["occupation"] = occ
            kept.append(_ordered(orb))
        shell["orbitals"] = kept
    atom["shells"] = {k: shells[k] for k in L_KEYS if k in shells}
    return atom


def load_koga() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], str]:
    raw = fetch(ATOMDB_TAR)
    heavy: dict[int, dict[str, Any]] = {}
    light: dict[int, dict[str, Any]] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as tar:
        for member in tar.getmembers():
            match = re.fullmatch(r"neutral/([a-z]+)\.slater", member.name)
            if not match:
                continue
            symbol = match.group(1).capitalize()
            z = SYMBOLS.index(symbol)
            handle = tar.extractfile(member)
            assert handle is not None
            atom = {"symbol": symbol, **parse_koga_file(handle.read().decode("latin-1"))}
            (heavy if z >= 55 else light)[z] = atom
    return heavy, light, sha256(raw)


# --- validation summary -------------------------------------------------------------------
def _summarise(atoms: dict[int, dict[str, Any]]) -> dict[str, Any]:
    worst = {"norm": 0.0, "ortho": 0.0, "regular": 0.0, "singular": 0.0, "cusp_rel": 0.0}
    source = {"RHOat0": 0.0, "kato": 0.0}
    for z, atom in atoms.items():
        metrics = validate(atom, z)
        problems = gates_ok(z, metrics)
        if problems:
            raise SystemExit(f"{atom['symbol']}: {problems}")
        for k in worst:
            worst[k] = max(worst[k], metrics["worst"][k])
        for k, dev in metrics["source"].items():
            source[k] = max(source[k], dev)
    extra = (
        {
            "max_rel_dev_source_RHOat0": float(f"{source['RHOat0']:.2e}"),
            "max_rel_dev_source_kato_cusp": float(f"{source['kato']:.2e}"),
        }
        if any(source.values())
        else {}
    )
    tabulated = (
        {
            "max_rel_excess_tabulated_r_r2_rinv": float(f"{worst['regular']:.2e}"),
            "max_rel_excess_tabulated_rinv2_rinv3": float(f"{worst['singular']:.2e}"),
            "tabulated_note": "relative deviation beyond the 1e-6 printing precision of the source values",
        }
        if worst["regular"] or worst["singular"]
        else {}
    )
    return {
        "max_abs_norm_error": float(f"{worst['norm']:.2e}"),
        "max_abs_overlap_same_l": float(f"{worst['ortho']:.2e}"),
        **tabulated,
        **extra,
        "max_rel_dev_single_s_orbital_cusp_from_Z": float(f"{worst['cusp_rel']:.2e}"),
        "checked_by": "tools/refdata/check_sto_tables.py",
    }


def _crosscheck_bunge_koga1999(bunge: dict[int, dict[str, Any]], koga: dict[int, dict[str, Any]]) -> dict[str, Any]:
    worst_e, worst_r = 0.0, (0.0, "")
    for z, atom in bunge.items():
        other = koga.get(z)
        if other is None:
            continue
        worst_e = max(worst_e, abs(atom["total_energy"] - other["total_energy"]) / abs(other["total_energy"]))
        for l_key, shell in atom["shells"].items():
            ns, zs = [b["n"] for b in shell["basis"]], [b["zeta"] for b in shell["basis"]]
            o_shell = other["shells"][l_key]
            ons, ozs = [b["n"] for b in o_shell["basis"]], [b["zeta"] for b in o_shell["basis"]]
            for orb in shell["orbitals"]:
                o_orb = next(o for o in o_shell["orbitals"] if o["label"] == orb["label"])
                a = radial_moment(ns, zs, orb["coefficients"], 1)
                b = radial_moment(ons, ozs, o_orb["coefficients"], 1)
                dev = abs(a - b) / b
                if dev > worst_r[0]:
                    worst_r = (dev, f"{atom['symbol']} {orb['label']}")
    return {
        "against": "T. Koga, K. Kanayama, S. Watanabe, A. J. Thakkar, Int. J. Quantum Chem. 71, 491 (1999), "
        "doi:10.1002/(SICI)1097-461X(1999)71:6<491::AID-QUA6>3.0.CO;2-T (cusp-constrained He-Xe; independent basis "
        "optimisation), from the same AtomDB archive",
        "max_rel_dev_total_energy": float(f"{worst_e:.2e}"),
        "max_rel_dev_orbital_r": float(f"{worst_r[0]:.2e}"),
        "max_rel_dev_orbital_r_at": worst_r[1],
    }


def main() -> int:
    bunge_raw = fetch(BUNGE_URL)
    bunge = parse_bunge(bunge_raw.decode("latin-1"))
    if sorted(bunge) != list(range(2, 55)):
        raise SystemExit(f"Bunge: expected Z = 2-54, got {sorted(bunge)}")
    heavy, light, tar_digest = load_koga()
    if sorted(heavy) != list(range(55, 104)):
        raise SystemExit(f"Koga 2000: expected Z = 55-103, got {sorted(heavy)}")
    for z, atom in bunge.items():
        if "note" in atom:  # confirm a restored sign against the independent Koga 1999 value
            for shell in atom["shells"].values():
                for orb in shell["orbitals"]:
                    ref = next((o["energy"] for s in light[z]["shells"].values() for o in s["orbitals"] if o["label"] == orb["label"]), None)
                    if ref is not None and abs(orb["energy"] - ref) > 1e-3 * abs(ref):
                        raise SystemExit(f"Bunge Z={z} {orb['label']}: energy {orb['energy']} vs Koga 1999 {ref}")

    common_terms = (
        "Published scientific data (facts) reproduced with attribution for computation; no text or software copied. "
    )
    bunge_payload = {
        "_provenance": provenance(
            source=(
                "C. F. Bunge, J. A. Barrientos and A. V. Bunge, 'Roothaan-Hartree-Fock ground-state atomic wave "
                "functions: Slater-type orbital expansions and expectation values for Z = 2-54', At. Data Nucl. Data "
                "Tables 53, 113-162 (1993), doi:10.1006/adnd.1993.1003; complementary: C. F. Bunge, J. A. Barrientos, "
                "A. V. Bunge and J. A. Cogordan, Phys. Rev. A 46, 3691 (1992), doi:10.1103/PhysRevA.46.3691. "
                f"Machine-readable file RHF.TABLES as distributed by the authors, mirrored at {BUNGE_URL} "
                f"(see {BUNGE_README})."
            ),
            license_or_terms=common_terms
            + "The authors released the file for free anonymous-FTP distribution (announcement by C. F. Bunge to the "
            "CCL list, 13 Sep 1993, reproduced in the CCL README); no licence terms are stated.",
            transform="tools/refdata/build_sto_tables.py (fixed-format text parsed; column-aligned orbital mapping)",
            units=CONVENTIONS["units"],
            uncertainty=(
                "Coefficients/exponents printed to 6/4 decimals; total energies agree with the numerical HF limit to "
                "~1e-6 Eh or better per the source. Recomputed <r^k> reproduce the tabulated values within the "
                "printed precision (see 'validation')."
            ),
            conventions=CONVENTIONS,
            source_fields=(
                "source_RHOat0 = the file's 'RHOat0' as printed. Verified for all 53 atoms to equal "
                "sum over s orbitals of R_ns(0)^2 with every orbital counted once (= 4 pi rho(0) / 2 for closed "
                "shells only; for open-shell s atoms such as Li it is NOT 4 pi rho(0)/2). source_kato_cusp = the "
                "file's 'Kato cusp' = -rho'(0) / (Z rho(0)) for the occupation-weighted spherical density "
                "(exact value 2); verified for all 53 atoms."
            ),
            download_sha256=sha256(bunge_raw),
            validation=_summarise(bunge),
            crosscheck=_crosscheck_bunge_koga1999(bunge, light),
        ),
        "atoms": {str(z): bunge[z] for z in sorted(bunge)},
        "by_symbol": {bunge[z]["symbol"]: z for z in sorted(bunge)},
    }
    koga_payload = {
        "_provenance": provenance(
            source=(
                "T. Koga, K. Kanayama, T. Watanabe, T. Imai and A. J. Thakkar, 'Analytical Hartree-Fock wave "
                "functions for the atoms Cs to Lr', Theor. Chem. Acc. 104, 411-413 (2000), doi:10.1007/s002140000150. "
                "Per-atom files as distributed by the authors (A. J. Thakkar, UNB, "
                "http://www.unb.ca/chem/ajit/download.htm -> stf/k99heavy.zip; page archived 2003, file no longer "
                f"online), obtained from the AtomDB redistribution {ATOMDB_TAR} (neutral/*.slater, Z >= 55)."
            ),
            license_or_terms=common_terms
            + "The authors distributed the files freely ('available upon request from the authors or from the Web "
            "page', per the paper); no licence terms are stated. The AtomDB repository that now carries them is "
            "GPL-3.0; only the numerical wave-function data (not AtomDB code) is used. Owner decision recommended "
            "if a stricter provenance chain is required.",
            transform="tools/refdata/build_sto_tables.py (Thakkar-group text format parsed)",
            units=CONVENTIONS["units"],
            uncertainty="Coefficients printed to 7 decimals, exponents to 6; energies to 1e-9 Eh as printed.",
            conventions=CONVENTIONS,
            download_sha256=tar_digest,
            validation=_summarise(heavy),
            note="Configurations and terms are those chosen by Koga et al. (e.g. Ce [Xe]4f(1)5d(1)6s(2) 1G, U "
            "[Rn]5f(3)6d(1)7s(2) 5L).",
        ),
        "atoms": {str(z): heavy[z] for z in sorted(heavy)},
        "by_symbol": {heavy[z]["symbol"]: z for z in sorted(heavy)},
    }
    for name, payload in (("bunge1993", bunge_payload), ("koga2000", koga_payload)):
        path = OUT_DIR / f"{name}.json"
        write_json(path, payload, indent=1)
        p = payload["_provenance"]
        print(f"{rel(path)}: {len(payload['atoms'])} atoms; validation {p['validation']}; crosscheck {p.get('crosscheck')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
