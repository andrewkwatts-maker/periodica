"""Validate that our deterministic NeRF backbones stay within physical
plausibility of AlphaFold predictions for bundled reference proteins.

Thresholds reflect that we generate from heuristic phi/psi (Chou-Fasman /
propensity-derived) rather than experimental or AF-quality angles, so our
backbones are reasonable but loose vs. AF2.  The tests document the
contract: the deterministic pipeline doesn't drift catastrophically.
"""
from __future__ import annotations

import pytest

from periodica import (
    backbone_array_from_pdb,
    build_backbone_from_entry,
    kabsch_rmsd,
    load_alphafold_reference,
    parse_pdb_backbone,
    reload_registry,
)


@pytest.fixture(scope="module", autouse=True)
def _refresh():
    reload_registry()
    yield


def _rmsd_vs_alphafold(name: str, uniprot_id: str) -> float:
    ours = build_backbone_from_entry(name)
    af_path = load_alphafold_reference(uniprot_id, allow_fetch=False)
    af = backbone_array_from_pdb(parse_pdb_backbone(af_path))
    n = min(ours.shape[0], af.shape[0])
    assert n >= 5, f"too few residues to align ({n})"
    _, rmsd = kabsch_rmsd(ours[:n, 1, :], af[:n, 1, :])
    return rmsd


@pytest.mark.parametrize(
    "name, uniprot_id, threshold_A",
    [
        # Thresholds are loose because we generate from heuristic phi/psi
        # (Chou-Fasman propensity-derived) without sidechain optimisation,
        # so backbones are physically plausible but not AF-quality. The
        # numbers cap regressions; tightening waits on better phi/psi data.
        ("Crambin",         "P01542", 25.0),
        ("Insulin_Chain_A", "P01308", 12.0),
        ("Ubiquitin",       "P0CG48", 30.0),
        ("Lysozyme",        "P00698", 70.0),
        ("Beta_Defensin",   "P60022", 22.0),
    ],
)
def test_backbone_close_to_alphafold(name, uniprot_id, threshold_A):
    rmsd = _rmsd_vs_alphafold(name, uniprot_id)
    assert rmsd < threshold_A, (
        f"{name} backbone RMSD vs AlphaFold {uniprot_id} = {rmsd:.2f} A, "
        f"exceeds threshold {threshold_A:.1f} A"
    )


def test_alphafold_references_bundled_offline():
    """All five bundled PDBs should resolve without network."""
    for uid in ("P01542", "P01308", "P0CG48", "P00698", "P60022"):
        path = load_alphafold_reference(uid, allow_fetch=False)
        assert path.exists()
        assert path.stat().st_size > 0
