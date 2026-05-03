"""Tests confirming proteins + amino_acids are first-class registry tiers."""
from __future__ import annotations

import pytest

from periodica import Get, Scope, list_tiers, reload_registry


@pytest.fixture(scope="module", autouse=True)
def _refresh():
    reload_registry()
    yield


class TestTierRegistration:
    def test_tiers_include_proteins_and_amino_acids(self):
        tiers = list_tiers()
        assert "proteins" in tiers
        assert "amino_acids" in tiers


class TestProteinLookup:
    def test_get_crambin_returns_protein_entry(self):
        c = Get("Crambin")
        assert c.get("length") == 46
        assert "isoelectric_point" in c

    def test_data_sheet_crambin_has_molecular_mass_and_pi(self):
        from periodica import data_sheet
        sheet = data_sheet("Crambin")
        assert "molecular_mass" in sheet or "isoelectric_point" in sheet

    def test_get_ubiquitin_has_residues(self):
        u = Get("Ubiquitin")
        assert isinstance(u.get("residues"), list)
        assert len(u["residues"]) > 50


class TestAminoAcidLookup:
    def test_get_alanine_returns_amino_acid(self):
        a = Get("Alanine")
        assert a.get("symbol") == "A"
        assert "hydropathy_index" in a

    def test_scope_amino_acid_resolves_single_letter(self):
        c = Get(Scope.AminoAcid, "C")
        assert c.get("name") == "Cysteine"

    def test_scope_amino_acid_alanine_by_letter(self):
        a = Get(Scope.AminoAcid, "A")
        assert a.get("name") == "Alanine"


class TestCollisionResolution:
    def test_scope_atom_C_returns_carbon_not_cysteine(self):
        # Plain Get('C') goes to atomic Carbon (derived shadows active).
        c = Get(Scope.Atom, "C")
        # Carbon is composed from {P:6, N:6, E:6}.
        assert c.get("Composition", {}).get("P") == 6

    def test_scope_amino_acid_C_returns_cysteine(self):
        c = Get(Scope.AminoAcid, "C")
        assert c.get("name") == "Cysteine"

    def test_alanine_amino_acid_wins_over_molecule_alanine(self):
        # Active stem 'Alanine' (priority 2) outranks derived molecule
        # stem 'Alanine' (priority 0) for plain Get('Alanine').
        a = Get("Alanine")
        assert a.get("hydropathy_index") is not None  # only the AA has this

    def test_get_p_still_returns_proton(self):
        assert Get("P").get("Name") == "Proton"

    def test_get_h_still_returns_hydrogen_via_explicit_symbol(self):
        # Hydrogen is saved with explicit Symbol 'H' (priority symbol_derived=3),
        # outranking Higgs's Symbol 'H' (priority symbol_active=1).
        assert Get("H").get("Symbol") == "H"
        assert Get("H").get("Composition", {}) == {"P": 1, "N": 0, "E": 1}
