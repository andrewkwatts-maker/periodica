"""
Tests for the BondingRulesEngine.
Verifies valence rules, bond type classification, and formula generation.
"""

import pytest
from periodica.utils.bonding_rules import BondingRulesEngine


@pytest.fixture
def rules():
    return BondingRulesEngine()


class TestValence:
    def test_hydrogen_valence(self, rules):
        assert rules.get_valence('H') == 1

    def test_carbon_valence(self, rules):
        assert rules.get_valence('C') == 4

    def test_nitrogen_valence(self, rules):
        assert rules.get_valence('N') == 3

    def test_oxygen_valence(self, rules):
        assert rules.get_valence('O') == 2

    def test_fluorine_valence(self, rules):
        assert rules.get_valence('F') == 1

    def test_chlorine_valence(self, rules):
        assert rules.get_valence('Cl') == 1

    def test_sodium_valence(self, rules):
        assert rules.get_valence('Na') == 1


class TestMaxBonds:
    def test_hydrogen_max_bonds(self, rules):
        assert rules.get_max_bonds('H') == 1

    def test_carbon_max_bonds(self, rules):
        assert rules.get_max_bonds('C') == 4

    def test_sulfur_max_bonds(self, rules):
        assert rules.get_max_bonds('S') == 6


class TestCanBond:
    def test_h_o_can_bond(self, rules):
        assert rules.can_bond('H', 'O') is True

    def test_c_h_can_bond(self, rules):
        assert rules.can_bond('C', 'H') is True

    def test_noble_gas_cannot_bond(self, rules):
        assert rules.can_bond('He', 'H') is False
        assert rules.can_bond('Ne', 'O') is False
        assert rules.can_bond('Ar', 'Cl') is False


class TestBondType:
    def test_nacl_ionic(self, rules):
        assert rules.get_bond_type('Na', 'Cl') == "Ionic"

    def test_hf_polar_covalent(self, rules):
        bond_type = rules.get_bond_type('H', 'F')
        assert bond_type in ("Polar Covalent", "Ionic")

    def test_cc_covalent(self, rules):
        assert rules.get_bond_type('C', 'C') == "Covalent"

    def test_ch_covalent_or_polar(self, rules):
        bt = rules.get_bond_type('C', 'H')
        assert bt in ("Covalent", "Polar Covalent")

    def test_ho_polar_covalent(self, rules):
        bt = rules.get_bond_type('H', 'O')
        assert bt in ("Polar Covalent", "Ionic")


class TestSatisfiesOctet:
    def test_hydrogen_one_bond(self, rules):
        assert rules.satisfies_octet('H', 1) is True

    def test_hydrogen_two_bonds(self, rules):
        assert rules.satisfies_octet('H', 2) is False

    def test_carbon_four_bonds(self, rules):
        assert rules.satisfies_octet('C', 4) is True

    def test_noble_gas_zero(self, rules):
        assert rules.satisfies_octet('He', 0) is True
        assert rules.satisfies_octet('Ne', 0) is True


class TestFormula:
    def test_water(self, rules):
        assert rules.get_formula({'H': 2, 'O': 1}) == "H2O"

    def test_methane(self, rules):
        assert rules.get_formula({'C': 1, 'H': 4}) == "CH4"

    def test_co2(self, rules):
        assert rules.get_formula({'C': 1, 'O': 2}) == "CO2"

    def test_nacl(self, rules):
        assert rules.get_formula({'Na': 1, 'Cl': 1}) == "ClNa"

    def test_single_element(self, rules):
        assert rules.get_formula({'O': 2}) == "O2"


class TestBinaryRatios:
    def test_h_o_gives_2_1(self, rules):
        ratios = rules.get_valid_binary_ratios('H', 'O')
        assert (2, 1) in ratios

    def test_na_cl_gives_1_1(self, rules):
        ratios = rules.get_valid_binary_ratios('Na', 'Cl')
        assert (1, 1) in ratios

    def test_noble_gas_gives_empty(self, rules):
        ratios = rules.get_valid_binary_ratios('He', 'O')
        assert ratios == []
