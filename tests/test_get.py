"""Tests for the generic Get/Save/registry layer.

Generated values are cross-checked against `data/reference/` (real-world
ground truth from CODATA / IUPAC / PDG) at every composition level.
The library NEVER reads from `data/reference/` at runtime - these tests
are the only consumer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from periodica import Get, Save, Scope, list_tiers, reload_registry
from periodica.get import (
    UnknownConstituent,
    UnknownName,
)


REF = Path(__file__).resolve().parents[1] / "src" / "periodica" / "data" / "reference"


def _ref(folder: str, name: str) -> dict:
    return json.loads((REF / folder / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def _registry_clean():
    reload_registry()
    yield


# ─────────────────────────────────────────────────────────────────────────
# Level 1: subatomic composed from quarks/leptons
# ─────────────────────────────────────────────────────────────────────────

class TestQuarksToHadron:
    def test_proton_quark_charge_matches_reference(self):
        p = Get({"u": 2, "d": 1}, from_="fundamentals")
        ref = _ref("subatomic", "Proton")
        assert p["Charge_e"] == ref["Charge_e"]

    def test_proton_quark_baryon_number_matches_reference(self):
        p = Get({"u": 2, "d": 1}, from_="fundamentals")
        ref = _ref("subatomic", "Proton")
        assert abs(p["BaryonNumber_B"] - ref["BaryonNumber_B"]) < 1e-3

    def test_neutron_quark_charge_is_zero(self):
        n = Get({"u": 1, "d": 2}, from_="fundamentals")
        ref = _ref("subatomic", "Neutron")
        assert abs(n["Charge_e"] - ref["Charge_e"]) < 1e-3

    def test_bare_quark_mass_is_far_below_real_proton(self):
        """Bare quark composition cannot match real hadron mass: most of the
        mass comes from QCD binding energy, not constituent quark masses."""
        p = Get({"u": 2, "d": 1}, from_="fundamentals")
        ref = _ref("subatomic", "Proton")
        assert p["Mass_MeVc2"] < ref["Mass_MeVc2"]

    def test_string_spec_equivalent_to_dict(self):
        a = Get("{u=2,d=1}", from_="fundamentals")
        b = Get({"u": 2, "d": 1}, from_="fundamentals")
        assert a["Charge_e"] == b["Charge_e"]
        assert a["Mass_MeVc2"] == b["Mass_MeVc2"]


# ─────────────────────────────────────────────────────────────────────────
# Level 2: atoms composed from subatomic + fundamentals
# ─────────────────────────────────────────────────────────────────────────

class TestSubatomicToAtom:
    @pytest.mark.parametrize(
        "spec, ref_name, mass_tolerance",
        [
            ({"P": 1, "N": 0, "E": 1}, "H", 0.01),
            ({"P": 2, "N": 2, "E": 2}, "He", 0.05),
            ({"P": 6, "N": 6, "E": 6}, "C", 0.10),
            ({"P": 8, "N": 8, "E": 8}, "O", 0.15),
            ({"P": 26, "N": 30, "E": 26}, "Fe", 0.50),
        ],
    )
    def test_atom_neutral_charge(self, spec, ref_name, mass_tolerance):
        atom = Get(spec, from_=["subatomic", "fundamentals"])
        ref = _ref("atoms", ref_name)
        assert atom["Charge_e"] == ref["Charge_e"]

    @pytest.mark.parametrize(
        "spec, ref_name, mass_tolerance",
        [
            ({"P": 1, "N": 0, "E": 1}, "H", 0.01),
            ({"P": 2, "N": 2, "E": 2}, "He", 0.05),
            ({"P": 6, "N": 6, "E": 6}, "C", 0.10),
            ({"P": 8, "N": 8, "E": 8}, "O", 0.15),
            ({"P": 26, "N": 30, "E": 26}, "Fe", 1.00),
        ],
    )
    def test_atom_mass_close_to_reference(self, spec, ref_name, mass_tolerance):
        atom = Get(spec, from_=["subatomic", "fundamentals"])
        ref = _ref("atoms", ref_name)
        delta = abs(atom["Mass_amu"] - ref["Mass_amu"])
        assert delta < mass_tolerance, (
            f"{ref_name}: generated {atom['Mass_amu']:.5f} vs ref {ref['Mass_amu']}, "
            f"diff {delta:.5f} > tol {mass_tolerance}"
        )

    def test_capital_P_resolves_to_proton_not_phosphorus(self):
        """Active alias must outrank derived element symbol of same case."""
        proton = Get("P")
        assert proton["Name"] == "Proton"

    def test_capital_N_resolves_to_neutron_not_nitrogen(self):
        n = Get("N")
        assert n["Name"] == "Neutron"

    def test_capital_E_alias_resolves_to_electron(self):
        e = Get("E")
        assert e["Name"] == "Electron"


# ─────────────────────────────────────────────────────────────────────────
# Level 3: molecules composed from atoms
# ─────────────────────────────────────────────────────────────────────────

class TestAtomsToMolecule:
    @pytest.mark.parametrize(
        "spec, ref_name, mass_tolerance",
        [
            ({"H": 2, "O": 1}, "H2O", 0.20),
            ({"C": 1, "O": 2}, "CO2", 0.50),
            ({"C": 1, "H": 4}, "CH4", 0.15),
            ({"N": 1, "H": 3}, "NH3", 0.15),
            ({"Na": 1, "Cl": 1}, "NaCl", 0.50),
        ],
    )
    def test_molecule_mass_close_to_reference(self, spec, ref_name, mass_tolerance):
        mol = Get(spec, from_="atoms")
        ref = _ref("molecules", ref_name)
        delta = abs(mol["Mass_amu"] - ref["Mass_amu"])
        assert delta < mass_tolerance, (
            f"{ref_name}: generated {mol['Mass_amu']:.4f} vs ref {ref['Mass_amu']}, "
            f"diff {delta:.4f} > tol {mass_tolerance}"
        )

    def test_water_via_atoms_context(self):
        water = Get("{H=2,O=1}", from_="atoms")
        assert water["Charge_e"] == 0
        assert "H" in water["Composition"]
        assert "O" in water["Composition"]


# ─────────────────────────────────────────────────────────────────────────
# Level 4: bare-name lookups + contextual disambiguation
# ─────────────────────────────────────────────────────────────────────────

class TestBareNameLookup:
    def test_get_H_returns_hydrogen_not_higgs(self):
        """Derived 'H' (Hydrogen) shadows Higgs Boson Symbol 'H'."""
        h = Get("H")
        ref = _ref("atoms", "H")
        # The saved entry has Symbol H and atomic-mass-shaped scalars.
        assert h["Symbol"] == "H"
        assert abs(h["Mass_amu"] - ref["Mass_amu"]) < 0.01
        # Composition records 1 proton + 1 electron, not a Higgs scalar.
        assert h.get("Composition") == {"P": 1, "N": 0, "E": 1}

    def test_higgs_remains_reachable_via_alias(self):
        higgs = Get("Higgs")
        assert higgs["Name"] == "Higgs Boson"

    def test_quark_symbol_does_not_shadow_uranium(self):
        u = Get("u")
        assert "Quark" in u.get("Name", "") or u["Name"] == "Up Quark"
        U = Get("U")
        assert U["Name"] in ("U", "Uranium")

    def test_get_water_by_name(self):
        water = Get("H2O")
        ref = _ref("molecules", "H2O")
        assert abs(water["Mass_amu"] - ref["Mass_amu"]) < 0.20


# ─────────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_unknown_constituent(self):
        with pytest.raises(UnknownConstituent):
            Get("{Xxxxxxx=1}")

    def test_unknown_name(self):
        with pytest.raises(UnknownName):
            Get("Imaginary_Element_Q3")

    def test_unknown_constituent_in_constrained_tier(self):
        # Quark symbol 'u' is in fundamentals, not atoms tier.
        with pytest.raises(UnknownConstituent):
            Get({"u": 2, "d": 1}, from_="atoms")

    def test_malformed_spec_string(self):
        with pytest.raises(ValueError):
            Get("{not valid spec}")


# ─────────────────────────────────────────────────────────────────────────
# Tier-aware Save() and registry refresh
# ─────────────────────────────────────────────────────────────────────────

class TestScopeOverload:
    """Get(scope, spec) positional form using the Scope enum."""

    def test_subatomic_P_resolves_to_proton(self):
        p = Get(Scope.SubAtomic, "P")
        assert p["Name"] == "Proton"

    def test_atom_P_resolves_to_phosphorus(self):
        ph = Get(Scope.Atom, "P")
        # Phosphorus's atomic number is 15.
        assert ph["Composition"] == ph.get("Composition")  # exists
        assert ph["Symbol"] == "P"
        # AtomicNumber-equivalent: Composition has 15 protons.
        assert ph["Composition"].get("P") == 15 or ph["Composition"].get("P") == 15.0

    def test_atoms_alias_equals_atom(self):
        a = Get(Scope.Atom, "Fe")
        b = Get(Scope.Atoms, "Fe")
        assert a["Mass_amu"] == b["Mass_amu"]

    def test_molecule_lookup_by_name(self):
        w = Get(Scope.Molecule, "H2O")
        assert w["Composition"] == {"H": 2, "O": 1}

    def test_scope_with_dict_spec(self):
        # SubAtomic scope contains Proton+Neutron (alias P/N) but not Electron
        # (which lives in `fundamentals`). So E must fail to resolve here -
        # demonstrates scope is genuinely restrictive.
        with pytest.raises(UnknownConstituent):
            Get(Scope.SubAtomic, {"P": 1, "N": 0, "E": 1})

    def test_scope_with_dict_spec_proton_neutron_only(self):
        # Without E, P+N do resolve in subatomic.
        nucleus = Get(Scope.SubAtomic, {"P": 1, "N": 0})
        assert nucleus["Charge_e"] == 1

    def test_scope_string_form_via_from_keyword(self):
        # Existing keyword form still works.
        p = Get("P", from_="subatomic")
        assert p["Name"] == "Proton"

    def test_scope_list_via_keyword(self):
        # Multi-tier scope via keyword.
        h = Get({"P": 1, "N": 0, "E": 1}, from_=[Scope.SubAtomic, Scope.Fundamentals])
        assert h["Charge_e"] == 0

    def test_two_positional_args_with_string_first_arg_rejected(self):
        # Strings as first positional arg are treated as spec, not tier.
        # Passing a tier-string in the first slot when a spec is in the second
        # is a type error - users must use Scope enum or keyword form.
        with pytest.raises(TypeError):
            Get("subatomic", "P")  # type: ignore[arg-type]

    def test_double_constraint_rejected(self):
        with pytest.raises(TypeError):
            Get(Scope.Atom, "Fe", from_=Scope.SubAtomic)

    def test_scope_enum_string_value_round_trip(self):
        assert str(Scope.Atom) == "atoms"
        assert str(Scope.SubAtomic) == "subatomic"


class TestSaveRoundtrip:
    def test_save_then_get_in_tmpdir(self, tmp_path):
        composed = Get({"P": 1, "N": 0, "E": 1}, from_=["subatomic", "fundamentals"])
        Save("MyHydrogen", composed, dir=tmp_path)
        loaded = json.loads((tmp_path / "MyHydrogen.json").read_text(encoding="utf-8"))
        assert loaded["Name"] == "MyHydrogen"
        assert abs(loaded["Mass_amu"] - composed["Mass_amu"]) < 1e-9

    def test_list_tiers_includes_all_built_tiers(self):
        tiers = list_tiers()
        assert "fundamentals" in tiers
        assert "subatomic" in tiers
        assert "atoms" in tiers
        assert "molecules" in tiers
