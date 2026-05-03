"""Tests for periodica.optimize: protein folding (SA on phi/psi) + alloy search."""
from __future__ import annotations

import pytest

from periodica import (
    optimize_alloy,
    optimize_protein_folding,
    reload_registry,
)


@pytest.fixture(scope="module", autouse=True)
def _refresh():
    reload_registry()
    yield


# ─────────────────────────────────────────────────────────────────────────
# Protein folding optimizer
# ─────────────────────────────────────────────────────────────────────────

class TestOptimizeFolding:
    def test_helical_target_returns_alpha_helix_region(self):
        result = optimize_protein_folding(
            "AAAAAAAAAA", target="helical", iterations=200, seed=1
        )
        assert result["target_region"] == "alpha_helix"
        assert result["coords"].shape == (10, 3, 3)

    def test_phi_psi_shape(self):
        r = optimize_protein_folding("MQIFV", target="helical", iterations=100, seed=0)
        assert r["phi_psi"].shape == (5, 2)

    def test_seed_reproducible(self):
        r1 = optimize_protein_folding("AAAA", target="helical", iterations=200, seed=42)
        r2 = optimize_protein_folding("AAAA", target="helical", iterations=200, seed=42)
        assert (r1["phi_psi"] == r2["phi_psi"]).all()
        assert r1["final_energy"] == r2["final_energy"]

    def test_energy_history_recorded(self):
        r = optimize_protein_folding("AAAA", target="helical", iterations=100, seed=1)
        assert len(r["energy_history"]) == 101  # initial + iterations

    def test_unknown_region_rejected(self):
        with pytest.raises(ValueError):
            optimize_protein_folding("AA", target="bogus_region", iterations=10, seed=0)

    def test_empty_sequence_rejected(self):
        with pytest.raises(ValueError):
            optimize_protein_folding("", target="helical", iterations=10)

    def test_helix_propensity_in_target(self):
        # Most residues should land in the alpha_helix region for 'helical' target.
        r = optimize_protein_folding(
            "AAAAAAAAAA", target="helical", iterations=300, seed=2
        )
        from periodica.folding import folding_rules
        rules = folding_rules()
        bounds = rules["ramachandran_allowed"]["alpha_helix"]
        in_helix = 0
        for phi, psi in r["phi_psi"]:
            if (bounds["phi"][0] <= phi <= bounds["phi"][1] and
                bounds["psi"][0] <= psi <= bounds["psi"][1]):
                in_helix += 1
        assert in_helix >= 7, f"only {in_helix}/10 residues in helical region"


# ─────────────────────────────────────────────────────────────────────────
# Alloy composition optimizer
# ─────────────────────────────────────────────────────────────────────────

class TestOptimizeAlloy:
    def test_returns_top_k_or_fewer(self):
        results = optimize_alloy(
            {"YieldStrength_MPa_min": 100},
            base="Fe", candidates=200, top_k=5, seed=1,
        )
        assert len(results) <= 5

    def test_seed_reproducible(self):
        a = optimize_alloy(
            {"YieldStrength_MPa_min": 100},
            base="Fe", candidates=200, top_k=3, seed=99,
        )
        b = optimize_alloy(
            {"YieldStrength_MPa_min": 100},
            base="Fe", candidates=200, top_k=3, seed=99,
        )
        assert [r["composition"] for r in a] == [r["composition"] for r in b]

    def test_density_max_constraint_respected(self):
        results = optimize_alloy(
            {"Density_kgm3_max": 5000},
            base="Al", candidates=200, top_k=5, seed=2,
        )
        for r in results:
            assert r["estimated_properties"]["Density_kgm3"] <= 5000.0

    def test_yield_min_constraint_respected(self):
        results = optimize_alloy(
            {"YieldStrength_MPa_min": 200},
            base="Fe", candidates=200, top_k=5, seed=3,
        )
        for r in results:
            assert r["estimated_properties"]["YieldStrength_MPa"] >= 200.0

    def test_top_k_sorted_descending(self):
        results = optimize_alloy(
            {"YieldStrength_MPa_min": 100},
            base="Fe", candidates=200, top_k=5, seed=4,
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_satisfying_candidates_returns_empty(self):
        # Impossibly tight constraint
        results = optimize_alloy(
            {"YieldStrength_MPa_min": 1e9},
            base="Fe", candidates=50, top_k=5, seed=5,
        )
        assert results == []
