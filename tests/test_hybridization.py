"""
Tests for Molecular Hybridization, Bond Dipoles, and Huckel Resonance

Validates:
1. Hybridization from steric number
2. Bond dipole vector summation for molecular dipole moment
3. Huckel resonance energy for conjugated ring systems
"""

import pytest
from periodica.utils.predictors.molecule.vsepr_predictor import VSEPRPredictor
from periodica.utils.predictors.molecule.huckel import (
    calculate_huckel_energies,
    is_aromatic,
)
from periodica.utils.predictors.base import MoleculeInput


@pytest.fixture
def vsepr():
    return VSEPRPredictor()


def _make_input(central_element, bonded_elements, central_en=None, central_r=None):
    """Helper to create MoleculeInput for testing."""
    atoms = [{'element': central_element}]
    if central_en:
        atoms[0]['electronegativity'] = central_en
    if central_r:
        atoms[0]['covalent_radius'] = central_r

    bonds = []
    for i, elem in enumerate(bonded_elements):
        atom = {'element': elem}
        atoms.append(atom)
        bonds.append({'from': 0, 'to': i + 1, 'order': 1})

    return MoleculeInput(atoms=atoms, bonds=bonds)


class TestHybridization:
    """Test hybridization determination from steric number."""

    def test_water_sp3(self, vsepr):
        """H2O: 2 bonding + 2 lone pairs = steric 4 = sp3."""
        inp = _make_input('O', ['H', 'H'])
        result = vsepr.predict(inp)
        assert result.hybridization is not None
        central_hyb = list(result.hybridization.values())[0]
        assert central_hyb == 'sp3', f"H2O hybridization: {central_hyb}, expected sp3"

    def test_methane_sp3(self, vsepr):
        """CH4: 4 bonding + 0 lone = steric 4 = sp3."""
        inp = _make_input('C', ['H', 'H', 'H', 'H'])
        result = vsepr.predict(inp)
        central_hyb = list(result.hybridization.values())[0]
        assert central_hyb == 'sp3'

    def test_becl2_sp(self, vsepr):
        """BeCl2: 2 bonding + 0 lone = steric 2 = sp."""
        inp = _make_input('Be', ['Cl', 'Cl'])
        result = vsepr.predict(inp)
        central_hyb = list(result.hybridization.values())[0]
        assert central_hyb == 'sp', f"BeCl2 hybridization: {central_hyb}, expected sp"

    def test_bf3_sp2(self, vsepr):
        """BF3: 3 bonding + 0 lone = steric 3 = sp2."""
        inp = _make_input('B', ['F', 'F', 'F'])
        result = vsepr.predict(inp)
        central_hyb = list(result.hybridization.values())[0]
        assert central_hyb == 'sp2'


class TestDipoleMoment:
    """Test bond dipole vector summation."""

    def test_becl2_zero_dipole(self, vsepr):
        """BeCl2: linear + symmetric = zero dipole."""
        inp = _make_input('Be', ['Cl', 'Cl'])
        result = vsepr.predict(inp)
        assert result.dipole_moment == 0.0, f"BeCl2 dipole {result.dipole_moment}, expected 0"

    def test_ch4_zero_dipole(self, vsepr):
        """CH4: tetrahedral + symmetric = zero dipole."""
        inp = _make_input('C', ['H', 'H', 'H', 'H'])
        result = vsepr.predict(inp)
        assert result.dipole_moment == 0.0, f"CH4 dipole {result.dipole_moment}, expected 0"

    def test_water_nonzero_dipole(self, vsepr):
        """H2O: bent geometry = non-zero dipole (~1.85 D)."""
        inp = _make_input('O', ['H', 'H'])
        result = vsepr.predict(inp)
        assert result.dipole_moment > 0.5, f"H2O dipole {result.dipole_moment}, expected >0.5 D"

    def test_hcl_nonzero_dipole(self, vsepr):
        """HCl: polar bond = non-zero dipole."""
        inp = _make_input('H', ['Cl'])
        result = vsepr.predict(inp)
        assert result.dipole_moment > 0.0, f"HCl dipole {result.dipole_moment}, expected >0"


class TestHuckelResonance:
    """Test Huckel MO calculator for conjugated rings."""

    def test_benzene_aromatic(self):
        """Benzene (6 atoms, 6 pi electrons) should be aromatic."""
        assert is_aromatic(6, 6) is True

    def test_cyclobutadiene_antiaromatic(self):
        """Cyclobutadiene (4 atoms, 4 pi electrons) should NOT be aromatic."""
        assert is_aromatic(4, 4) is False

    def test_benzene_resonance_energy(self):
        """Benzene resonance energy should be ~-150 kJ/mol."""
        result = calculate_huckel_energies(6, 6)
        E_res = result['resonance_energy_kjmol']
        # Resonance energy should be negative (stabilizing)
        assert E_res < 0, f"Resonance energy {E_res} should be negative"
        assert abs(E_res - (-150)) < 30, f"Benzene E_res={E_res}, expected ~-150 kJ/mol"

    def test_benzene_resonance_in_beta(self):
        """Benzene should have 2*beta resonance energy."""
        result = calculate_huckel_energies(6, 6)
        # Total pi = 2*(2) + 2*(1) + 2*(1) = 8 beta
        # Localized = 3 * 2 = 6 beta
        # Resonance = 8 - 6 = 2 beta
        assert abs(result['resonance_energy_beta'] - 2.0) < 0.01

    def test_cyclopentadienyl_anion_aromatic(self):
        """Cyclopentadienyl anion (5 ring, 6 pi e-) should be aromatic."""
        assert is_aromatic(5, 6) is True

    def test_naphthalene_resonance(self):
        """Naphthalene (10 ring, 10 pi electrons) resonance should be significant."""
        result = calculate_huckel_energies(10, 10)
        assert result['resonance_energy_kjmol'] < -100, \
            f"Naphthalene resonance {result['resonance_energy_kjmol']} should be < -100 kJ/mol"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
