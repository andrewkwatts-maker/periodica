"""Tests for post-composition refinements (periodica.refine).

These pin down the accuracy the refinements buy. Additive composition of
free constituents is systematically wrong -- it omits nuclear binding energy
and reports zero for units the constituents never carried -- and these tests
hold the corrected values against measured ones.
"""
from __future__ import annotations

import pytest

from periodica import Get
from periodica.get import _config
from periodica.refine import (
    apply_refinements,
    list_refiners,
    register_refiner,
)


def _atom(Z: int, N: int) -> dict:
    return Get({"P": Z, "N": N, "E": Z}, from_=["subatomic", "fundamentals"])


#: AME2020 / CODATA measured neutral-atom masses, in amu.
MEASURED_ISOTOPE_MASS = {
    "H-1":   (1, 0, 1.00782503),
    "He-4":  (2, 2, 4.00260325),
    "C-12":  (6, 6, 12.0),
    "O-16":  (8, 8, 15.99491462),
    "Fe-56": (26, 30, 55.93493633),
    "Cu-63": (29, 34, 62.92959772),
    "Ag-107": (47, 60, 106.9050916),
    "Au-197": (79, 118, 196.96656879),
    "U-238": (92, 146, 238.05078826),
}


# ─────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────

class TestRefinerRegistry:
    def test_built_ins_are_registered(self):
        assert {"nuclear_binding", "mass_units", "cross_reference"} <= set(list_refiners())

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError):
            register_refiner("", lambda c, p: None)

    def test_register_rejects_non_callable(self):
        with pytest.raises(TypeError):
            register_refiner("bad", "not callable")  # type: ignore[arg-type]

    def test_unknown_refiner_in_rules_raises(self):
        with pytest.raises(KeyError):
            apply_refinements({}, {"refinements": [{"refiner": "no_such_refiner"}]})

    def test_malformed_refinement_entry_raises(self):
        with pytest.raises(ValueError):
            apply_refinements({}, {"refinements": ["not an object"]})

    def test_empty_rules_is_a_no_op(self):
        entry = {"Mass_amu": 1.0}
        assert apply_refinements(dict(entry), {"refinements": []}) == entry

    def test_config_declares_the_refinements(self):
        names = [r["refiner"] for r in _config()["refinements"]]
        assert names == ["nuclear_binding", "mass_units", "cross_reference"]


# ─────────────────────────────────────────────────────────────────────────
# nuclear_binding
# ─────────────────────────────────────────────────────────────────────────

class TestNuclearBinding:
    @pytest.mark.parametrize("label", sorted(MEASURED_ISOTOPE_MASS))
    def test_mass_within_0_05_percent_of_measured(self, label):
        Z, N, measured = MEASURED_ISOTOPE_MASS[label]
        got = _atom(Z, N)["Mass_amu"]
        error_pct = abs(got - measured) / measured * 100
        assert error_pct < 0.05, (
            f"{label}: composed {got:.6f} vs measured {measured:.6f} "
            f"({error_pct:.4f}% > 0.05%)"
        )

    def test_mean_error_across_the_set_is_under_0_02_percent(self):
        errors = []
        for Z, N, measured in MEASURED_ISOTOPE_MASS.values():
            got = _atom(Z, N)["Mass_amu"]
            errors.append(abs(got - measured) / measured * 100)
        mean = sum(errors) / len(errors)
        assert mean < 0.02, f"mean error {mean:.4f}% -- accuracy regressed"

    def test_binding_energy_is_reported(self):
        fe = _atom(26, 30)
        assert fe["BindingEnergy_MeV"] > 0
        # Fe-56 is near the peak of the binding curve, ~8.8 MeV per nucleon.
        assert 8.0 < fe["BindingEnergyPerNucleon_MeV"] < 9.2

    def test_nucleon_counts_are_reported(self):
        fe = _atom(26, 30)
        assert (fe["AtomicNumber_Z"], fe["NeutronNumber_N"], fe["MassNumber_A"]) == (26, 30, 56)

    def test_mass_defect_reduces_the_additive_sum(self):
        fe = _atom(26, 30)
        additive = fe["Mass_amu"] + fe["MassDefect_amu"]
        assert additive > fe["Mass_amu"]
        # The additive sum is the ~1% overestimate this refinement removes.
        assert 0.9 < (additive / 55.93493633 - 1) * 100 < 1.1

    def test_mev_and_amu_stay_consistent(self):
        fe = _atom(26, 30)
        assert fe["Mass_MeVc2"] == pytest.approx(fe["Mass_amu"] * 931.49410242, rel=1e-9)

    def test_single_proton_gets_no_binding_correction(self):
        h = Get({"P": 1, "E": 1}, from_=["subatomic", "fundamentals"])
        assert "BindingEnergy_MeV" not in h

    def test_molecule_is_untouched_by_the_nuclear_refiner(self):
        water = Get({"H": 2, "O": 1}, from_="atoms")
        assert "BindingEnergy_MeV" not in water
        assert "MassNumber_A" not in water

    def test_fractional_nucleon_counts_are_not_a_nucleus(self):
        entry = {"Composition": {"P": 1.5, "N": 1.0}, "Mass_amu": 2.5}
        apply_refinements(entry, {"refinements": [
            {"refiner": "nuclear_binding",
             "params": {"proton_symbols": ["P"], "neutron_symbols": ["N"]}},
        ]})
        assert entry["Mass_amu"] == 2.5

    def test_no_composition_is_a_no_op(self):
        entry = {"Mass_amu": 5.0}
        apply_refinements(entry, {"refinements": [
            {"refiner": "nuclear_binding",
             "params": {"proton_symbols": ["P"], "neutron_symbols": ["N"]}},
        ]})
        assert entry == {"Mass_amu": 5.0}


# ─────────────────────────────────────────────────────────────────────────
# mass_units
# ─────────────────────────────────────────────────────────────────────────

class TestMassUnits:
    def test_mass_kg_is_no_longer_zero(self):
        # The bug this fixes: every composed entry reported Mass_kg: 0,
        # because no constituent JSON carries kilograms to add up.
        fe = _atom(26, 30)
        assert fe["Mass_kg"] > 0
        assert fe["Mass_kg"] == pytest.approx(fe["Mass_amu"] * 1.66053906660e-27, rel=1e-12)

    def test_molecules_get_mass_kg_too(self):
        assert Get({"H": 2, "O": 1}, from_="atoms")["Mass_kg"] > 0

    def test_amu_derived_from_mev_when_absent(self):
        entry = {"Mass_MeVc2": 931.49410242}
        apply_refinements(entry, {"refinements": [{"refiner": "mass_units", "params": {}}]})
        assert entry["Mass_amu"] == pytest.approx(1.0, rel=1e-9)

    def test_curated_mass_kg_is_not_overwritten(self):
        entry = {"Mass_amu": 1.0, "Mass_kg": 42.0}
        apply_refinements(entry, {"refinements": [{"refiner": "mass_units", "params": {}}]})
        assert entry["Mass_kg"] == 42.0


# ─────────────────────────────────────────────────────────────────────────
# cross_reference
# ─────────────────────────────────────────────────────────────────────────

class TestCrossReference:
    def test_composed_atom_carries_measured_properties(self):
        # Before this refinement a composed atom had no measurable property
        # at all -- only composition arithmetic.
        fe = _atom(26, 30)
        props = fe["Properties"]
        assert props["density"] == 7.874
        assert props["melting_point"] == 1811
        assert props["electronegativity"] == 1.83

    def test_standard_atomic_weight_is_kept_separate_from_isotope_mass(self):
        fe = _atom(26, 30)
        # Mass_amu is Fe-56; AtomicWeight_amu is chlorine-style natural
        # abundance weighting. Conflating them is what made the old numbers
        # look wrong in two different ways at once.
        assert fe["AtomicWeight_amu"] == 55.845
        assert fe["Mass_amu"] != fe["AtomicWeight_amu"]

    def test_element_identity_is_promoted(self):
        fe = _atom(26, 30)
        assert fe["ElementName"] == "Iron"
        assert fe["ElementSymbol"] == "Fe"

    def test_provenance_records_the_join(self):
        prov = _atom(26, 30)["_provenance"]["Properties"]
        assert prov["tier"] == "elements"
        assert prov["matched_on"] == "atomic_number=26"
        assert prov["entry"] == "Iron"

    def test_all_118_elements_join(self):
        missing = []
        for Z in range(1, 119):
            atom = Get({"P": Z, "N": Z, "E": Z}, from_=["subatomic", "fundamentals"])
            if not atom.get("Properties", {}).get("melting_point") and Z <= 96:
                # Above ~96 the synthetic elements genuinely have no measured
                # melting point; below that, a miss means the join failed.
                missing.append(Z)
        assert not missing, f"element join failed for Z={missing}"

    def test_unmatched_join_value_is_a_no_op(self):
        entry = {"AtomicNumber_Z": 9999}
        apply_refinements(entry, {"refinements": [{
            "refiner": "cross_reference",
            "params": {"tier": "elements", "match_field": "AtomicNumber_Z",
                       "match_key": "atomic_number"},
        }]})
        assert "Properties" not in entry

    def test_missing_params_is_a_no_op(self):
        entry = {"AtomicNumber_Z": 26}
        apply_refinements(entry, {"refinements": [
            {"refiner": "cross_reference", "params": {}},
        ]})
        assert "Properties" not in entry

    def test_existing_property_values_are_not_clobbered(self):
        entry = {"AtomicNumber_Z": 26, "Properties": {"density": 1.0}}
        apply_refinements(entry, {"refinements": [{
            "refiner": "cross_reference",
            "params": {"tier": "elements", "match_field": "AtomicNumber_Z",
                       "match_key": "atomic_number"},
        }]})
        assert entry["Properties"]["density"] == 1.0


# ─────────────────────────────────────────────────────────────────────────
# Saved artifacts agree with live composition
# ─────────────────────────────────────────────────────────────────────────

class TestSavedDataMatchesLiveComposition:
    @pytest.mark.parametrize("symbol, Z, N", [("H", 1, 0), ("C", 6, 6), ("Fe", 26, 30)])
    def test_derived_atom_json_matches_a_fresh_compose(self, symbol, Z, N):
        saved = Get(symbol, from_="atoms")
        fresh = _atom(Z, N)
        assert saved["Mass_amu"] == pytest.approx(fresh["Mass_amu"], rel=1e-12)
        assert saved["Mass_kg"] == pytest.approx(fresh["Mass_kg"], rel=1e-12)

    def test_no_derived_atom_still_reports_zero_mass_kg(self):
        zeros = [
            s for s in ("H", "He", "C", "O", "Fe", "Cu", "Au", "U")
            if not Get(s, from_="atoms").get("Mass_kg")
        ]
        assert not zeros, f"Mass_kg still zero for {zeros}"
