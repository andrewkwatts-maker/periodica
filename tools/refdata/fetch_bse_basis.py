"""Download Gaussian basis sets (H-Kr) from the Basis Set Exchange (todo Q2).

Writes ``rust/periodica-qm/data/basis/<bse-name>.json`` in BSE's own
"complete" JSON schema (``molssi_bse_schema`` 0.1) unchanged, plus a top-level
``_provenance`` block holding the citation, licence and the full bibliographic
entries of every original basis-set paper BSE lists for these elements.

Usage::

    python tools/refdata/fetch_bse_basis.py

The BSE version is pinned (``version=1``) so the download is reproducible.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from _common import REPO_ROOT, fetch, fetch_json, provenance, rel, sha256, write_json

API = "https://www.basissetexchange.org/api"
ELEMENTS = "1-36"
VERSION = "1"

# BSE canonical names; BSE itself spells "*" as "_st_" in file names.
BASIS_SETS = ("sto-3g", "6-31g", "6-31g_st_")
DISPLAY = {"sto-3g": "STO-3G", "6-31g": "6-31G", "6-31g_st_": "6-31G*"}

BSE_CITATION = (
    "Basis Set Exchange (https://www.basissetexchange.org), MolSSI. "
    "B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus, "
    "'A New Basis Set Exchange: An Open, Up-to-date Resource for the Molecular "
    "Sciences Community', J. Chem. Inf. Model. 59, 4814-4820 (2019), "
    "doi:10.1021/acs.jcim.9b00725"
)
BSE_LICENSE = (
    "BSD-3-Clause (MolSSI Basis Set Exchange repository licence, "
    "https://github.com/MolSSI-BSE/basis_set_exchange/blob/master/LICENSE, "
    "(c) 2020 The Molecular Sciences Software Institute, Virginia Tech). "
    "BSE asks users to cite Pritchard et al. 2019 and the original basis-set "
    "papers listed under 'basis_set_references'. Exponents and coefficients are "
    "published scientific data."
)
CONVENTIONS = {
    "exponents": "Primitive Gaussian exponents alpha in bohr^-2 (atomic units), "
    "stored as strings exactly as BSE distributes them (no float round trip).",
    "coefficients": "Contraction coefficients as published, multiplying "
    "NORMALISED primitive Gaussians; the contracted function must be "
    "renormalised by the consumer. One coefficient row per angular momentum in "
    "'angular_momentum' (Pople 'sp' shells have angular_momentum [0, 1] and two rows).",
    "function_type": "'gto' = s/p shells (no spherical/cartesian distinction); "
    "'gto_cartesian' = d shells meant to be used as 6 Cartesian components "
    "(Pople 6-31G* convention, 6D); 'gto_spherical' = pure functions. Honour this "
    "per shell or energies will not match published/reference values.",
    "region": "BSE region tag (empty for these sets).",
}


def _download_basis(name: str) -> tuple[dict[str, Any], str, str]:
    url = f"{API}/basis/{name}/format/json/?version={VERSION}&elements={ELEMENTS}"
    raw = fetch(url)
    return json.loads(raw.decode("utf-8")), url, sha256(raw)


def _demojibake(value: Any) -> Any:
    """Repair UTF-8 text that BSE double-encoded via Latin-1 (e.g. 'Farkas, Ã–.' -> 'Farkas, Ö.')."""
    if isinstance(value, str) and "Ã" in value:
        try:
            return value.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    if isinstance(value, list):
        return [_demojibake(v) for v in value]
    if isinstance(value, dict):
        return {k: _demojibake(v) for k, v in value.items()}
    return value


def _download_references(name: str) -> tuple[dict[str, Any], str]:
    url = f"{API}/references/{name}/format/json/?version={VERSION}&elements={ELEMENTS}"
    blocks = fetch_json(url)
    references: dict[str, Any] = {}
    for block in blocks:
        for info in block["reference_info"]:
            for key, entry in info["reference_data"]:
                references[key] = _demojibake(entry)
    return dict(sorted(references.items())), url


def _check(name: str, basis: dict[str, Any], references: dict[str, Any]) -> None:
    elements = basis["elements"]
    expected = {str(z) for z in range(1, 37)}
    missing = expected - set(elements)
    if missing:
        raise SystemExit(f"{name}: BSE returned no data for Z = {sorted(missing, key=int)}")
    for z, data in elements.items():
        for ref in data.get("references", []):
            for key in ref["reference_keys"]:
                if key not in references:
                    raise SystemExit(f"{name}: Z={z} cites unknown reference key {key!r}")
        for shell in data["electron_shells"]:
            if len(shell["coefficients"]) != len(shell["angular_momentum"]) and len(shell["angular_momentum"]) != 1:
                raise SystemExit(f"{name}: Z={z} shell has inconsistent coefficient rows")
            for row in shell["coefficients"]:
                if len(row) != len(shell["exponents"]):
                    raise SystemExit(f"{name}: Z={z} coefficient row length != number of exponents")


def build(name: str) -> dict[str, Any]:
    basis, basis_url, digest = _download_basis(name)
    references, ref_url = _download_references(name)
    _check(name, basis, references)
    block = provenance(
        source=[BSE_CITATION, f"Original basis-set literature: see 'basis_set_references' ({len(references)} entries)."],
        license_or_terms=BSE_LICENSE,
        transform=(
            "tools/refdata/fetch_bse_basis.py (BSE JSON stored verbatim; only '_provenance' added; "
            "Latin-1 mojibake in BSE author names repaired in the reference list)"
        ),
        units={"exponents": "bohr^-2", "coefficients": "dimensionless (normalised primitives)"},
        uncertainty="Not applicable: basis-set parameters are defined quantities, not measurements.",
        basis_name=DISPLAY[name],
        bse_name=name,
        bse_version=basis.get("version"),
        bse_revision_description=basis.get("revision_description"),
        bse_revision_date=basis.get("revision_date"),
        elements=f"Z = {ELEMENTS} (H-Kr), {len(basis['elements'])} elements",
        download_urls=[basis_url, ref_url],
        download_sha256=digest,
        conventions=CONVENTIONS,
        basis_set_references=references,
    )
    return {"_provenance": block, **basis}


def main() -> int:
    out_dir = REPO_ROOT / "rust" / "periodica-qm" / "data" / "basis"
    for name in BASIS_SETS:
        payload = build(name)
        path = out_dir / f"{name}.json"
        write_json(path, payload, indent=1)
        n_shells = sum(len(e["electron_shells"]) for e in payload["elements"].values())
        print(f"{rel(path)}: {len(payload['elements'])} elements, {n_shells} shells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
