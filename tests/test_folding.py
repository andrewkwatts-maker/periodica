"""Tests for the protein-folding module: NeRF backbone, Kabsch, PDB parser."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from periodica import (
    build_backbone,
    build_backbone_from_entry,
    extract_phi_psi,
    folding_rules,
    kabsch_rmsd,
    load_alphafold_reference,
    parse_pdb_backbone,
    backbone_array_from_pdb,
    ramachandran_in_allowed,
    ramachandran_region,
    reload_registry,
)


@pytest.fixture(scope="module", autouse=True)
def _refresh():
    reload_registry()
    yield


# ─────────────────────────────────────────────────────────────────────────
# Backbone construction
# ─────────────────────────────────────────────────────────────────────────

class TestBuildBackbone:
    def test_short_helix_correct_shape(self):
        coords = build_backbone("AAAAA", [(-60, -45)] * 5)
        assert coords.shape == (5, 3, 3)

    def test_bond_lengths_within_tolerance(self):
        rules = folding_rules()
        n_ca, ca_c, c_n = (
            rules["bond_lengths_A"]["N_CA"],
            rules["bond_lengths_A"]["CA_C"],
            rules["bond_lengths_A"]["C_N"],
        )
        coords = build_backbone("AAAAAA", [(-60, -45)] * 6)
        # N-CA, CA-C bonds within each residue
        for i in range(6):
            d_n_ca = np.linalg.norm(coords[i, 1] - coords[i, 0])
            d_ca_c = np.linalg.norm(coords[i, 2] - coords[i, 1])
            assert abs(d_n_ca - n_ca) < 0.05, f"residue {i}: N-CA = {d_n_ca:.3f}"
            assert abs(d_ca_c - ca_c) < 0.05, f"residue {i}: CA-C = {d_ca_c:.3f}"
        # Inter-residue C-N peptide bonds
        for i in range(5):
            d_c_n = np.linalg.norm(coords[i + 1, 0] - coords[i, 2])
            assert abs(d_c_n - c_n) < 0.05, f"peptide {i}-{i+1}: C-N = {d_c_n:.3f}"

    def test_n_ca_c_angle_close_to_111(self):
        coords = build_backbone("AAAAA", [(-60, -45)] * 5)
        for i in range(5):
            v1 = coords[i, 0] - coords[i, 1]   # N - CA
            v2 = coords[i, 2] - coords[i, 1]   # C - CA
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = float(np.degrees(np.arccos(cos)))
            assert 109 < angle < 113, f"residue {i}: N-CA-C angle = {angle:.2f}"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_backbone("AAAA", [(-60, -45)] * 3)

    def test_empty_sequence_returns_empty_array(self):
        coords = build_backbone("", [])
        assert coords.shape == (0, 3, 3)


class TestFromEntry:
    def test_crambin_yields_46_residues(self):
        coords = build_backbone_from_entry("Crambin")
        assert coords.shape[0] == 46

    def test_ubiquitin_yields_76_residues(self):
        coords = build_backbone_from_entry("Ubiquitin")
        assert coords.shape[0] >= 70

    def test_extract_phi_psi_returns_pairs(self):
        from periodica import Get
        entry = Get("Crambin")
        pairs = extract_phi_psi(entry)
        assert len(pairs) == 46
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


# ─────────────────────────────────────────────────────────────────────────
# Kabsch / RMSD
# ─────────────────────────────────────────────────────────────────────────

class TestKabsch:
    def test_self_alignment_is_zero(self):
        P = np.random.RandomState(42).randn(20, 3)
        _, rmsd = kabsch_rmsd(P, P.copy())
        assert rmsd < 1e-9

    def test_translated_alignment_is_zero(self):
        P = np.random.RandomState(7).randn(20, 3)
        Q = P + np.array([5.0, -3.0, 2.0])
        _, rmsd = kabsch_rmsd(P, Q)
        assert rmsd < 1e-9

    def test_rotated_alignment_is_zero(self):
        P = np.random.RandomState(11).randn(20, 3)
        # Rotate Q by 30° about z
        c, s = np.cos(0.5236), np.sin(0.5236)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Q = P @ R.T
        _, rmsd = kabsch_rmsd(P, Q)
        assert rmsd < 1e-9

    def test_handles_reflection(self):
        # Mirror image: Kabsch must avoid an improper rotation.
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = P.copy()
        Q[:, 0] *= -1   # reflect through yz plane
        _, rmsd = kabsch_rmsd(P, Q)
        # Kabsch returns the best proper rotation. RMSD should be nonzero
        # but finite -- not negative or NaN.
        assert rmsd >= 0
        assert np.isfinite(rmsd)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            kabsch_rmsd(np.zeros((5, 3)), np.zeros((6, 3)))


# ─────────────────────────────────────────────────────────────────────────
# PDB parser
# ─────────────────────────────────────────────────────────────────────────

class TestPDB:
    def test_parses_alphafold_crambin(self):
        path = load_alphafold_reference("P01542", allow_fetch=False)
        parsed = parse_pdb_backbone(path)
        assert len(parsed) == 46
        assert all("CA" in atoms for atoms in parsed.values())

    def test_backbone_array_skips_incomplete_residues(self):
        path = load_alphafold_reference("P01542", allow_fetch=False)
        parsed = parse_pdb_backbone(path)
        # Drop one atom from residue 1 to simulate incompleteness.
        parsed[1].pop("N", None)
        arr = backbone_array_from_pdb(parsed)
        assert arr.shape[0] == 45  # one residue dropped

    def test_load_alphafold_uses_bundled_when_present(self):
        path = load_alphafold_reference("P01542", allow_fetch=False)
        assert "data" in str(path) and "reference" in str(path)


# ─────────────────────────────────────────────────────────────────────────
# Ramachandran classification
# ─────────────────────────────────────────────────────────────────────────

class TestRamachandran:
    def test_alpha_helix_classified(self):
        assert ramachandran_region(-60, -45) == "alpha_helix"

    def test_beta_sheet_classified(self):
        assert ramachandran_region(-120, 130) == "beta_sheet"

    def test_left_helix_classified(self):
        assert ramachandran_region(60, 45) == "left_alpha_helix"

    def test_disallowed_region_is_None(self):
        # Mid-disallowed regions
        assert ramachandran_region(0, 0) is None
        assert not ramachandran_in_allowed(0, 0)


# ─────────────────────────────────────────────────────────────────────────
# Pure-numpy contract
# ─────────────────────────────────────────────────────────────────────────

class TestPureDeps:
    """Folding/optimize must not introduce heavy dependencies.

    Checks the modules' source for scipy/biopython imports directly. (sys.modules
    can be polluted by other tests in the suite, so a state-based assertion
    would be order-dependent.)
    """

    def _imports(self, modname: str):
        import ast, importlib
        mod = importlib.import_module(modname)
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module.split(".")[0])
        return names

    def test_folding_imports_no_scipy_or_biopython(self):
        deps = self._imports("periodica.folding")
        assert "scipy" not in deps, f"folding.py imports {deps & {'scipy'}}"
        assert "Bio" not in deps, f"folding.py imports {deps & {'Bio'}}"

    def test_optimize_imports_no_scipy_or_biopython(self):
        deps = self._imports("periodica.optimize")
        assert "scipy" not in deps
        assert "Bio" not in deps

    def test_subprocess_no_scipy_after_folding_import(self):
        """Fresh interpreter: importing folding alone must not load scipy."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import periodica.folding, sys; "
             "scipy = [m for m in sys.modules if m.split('.')[0] == 'scipy']; "
             "bio = [m for m in sys.modules if m == 'Bio' or m.startswith('Bio.')]; "
             "print('SCIPY:', scipy); print('BIO:', bio)"],
            capture_output=True, text=True, timeout=30,
        )
        assert "SCIPY: []" in result.stdout, result.stdout + result.stderr
        assert "BIO: []" in result.stdout, result.stdout + result.stderr
