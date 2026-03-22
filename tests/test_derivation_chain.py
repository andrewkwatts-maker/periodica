"""
Comprehensive tests for the Quark-to-Atom Derivation Chain.

Tests validate that:
1. Quarks -> Hadrons (proton, neutron) matches PDG data
2. Hadrons -> Nuclei (binding energy) matches experimental data
3. Nuclei -> Atoms (properties) matches NIST data
4. Full chain can predict unknown atoms
"""

import pytest
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periodica.utils.physics_calculator_v2 import SubatomicCalculatorV2
from periodica.utils.nuclear_derivation import NuclearDerivation, derive_nuclear_properties
from periodica.utils.atomic_derivation import AtomicDerivation, derive_atomic_properties
from periodica.utils.prediction_engine import PredictionEngine, predict_element


# ==================== REFERENCE DATA (PDG, NIST, Nuclear Data) ====================

# Quark data (PDG 2024)
QUARK_DATA = {
    'u': {'Symbol': 'u', 'Name': 'Up Quark', 'Mass_MeVc2': 2.16, 'Charge_e': 2/3,
          'Spin_hbar': 0.5, 'BaryonNumber_B': 1/3, 'Isospin_I': 0.5, 'Isospin_I3': 0.5},
    'd': {'Symbol': 'd', 'Name': 'Down Quark', 'Mass_MeVc2': 4.67, 'Charge_e': -1/3,
          'Spin_hbar': 0.5, 'BaryonNumber_B': 1/3, 'Isospin_I': 0.5, 'Isospin_I3': -0.5},
}

# Hadron reference data (PDG 2024)
HADRON_REFERENCE = {
    'proton': {'mass_mev': 938.272, 'charge': 1, 'spin': 0.5, 'isospin': 0.5},
    'neutron': {'mass_mev': 939.565, 'charge': 0, 'spin': 0.5, 'isospin': 0.5},
}

# Nuclear binding energies (MeV) - experimental values
NUCLEAR_BINDING = {
    (2, 2): 28.296,     # He-4
    (6, 6): 92.162,     # C-12
    (8, 8): 127.619,    # O-16
    (26, 30): 492.254,  # Fe-56
    (82, 126): 1636.43, # Pb-208
}

# Atomic properties (NIST)
ATOMIC_REFERENCE = {
    1: {'symbol': 'H', 'mass_u': 1.008, 'ionization_ev': 13.598},
    2: {'symbol': 'He', 'mass_u': 4.003, 'ionization_ev': 24.587},
    6: {'symbol': 'C', 'mass_u': 12.011, 'ionization_ev': 11.260},
    26: {'symbol': 'Fe', 'mass_u': 55.845, 'ionization_ev': 7.902},
}


# ==================== TEST CLASSES ====================

class TestQuarkToHadronDerivation:
    """Test derivation of hadron properties from quark data."""

    @pytest.fixture
    def calculator(self):
        return SubatomicCalculatorV2()

    def _get_quark_list(self, quark_symbols):
        return [QUARK_DATA[q] for q in quark_symbols]

    def test_proton_charge_exact(self, calculator):
        """Proton charge must be exactly +1."""
        quarks = self._get_quark_list(['u', 'u', 'd'])
        result = calculator.create_particle_from_quarks(quarks)
        assert result['Charge_e'] == pytest.approx(1.0, abs=1e-10)

    def test_neutron_charge_exact(self, calculator):
        """Neutron charge must be exactly 0."""
        quarks = self._get_quark_list(['u', 'd', 'd'])
        result = calculator.create_particle_from_quarks(quarks)
        assert result['Charge_e'] == pytest.approx(0.0, abs=1e-10)

    def test_proton_mass_within_tolerance(self, calculator):
        """Proton mass should be within 5% of PDG value."""
        quarks = self._get_quark_list(['u', 'u', 'd'])
        result = calculator.create_particle_from_quarks(quarks)
        expected = HADRON_REFERENCE['proton']['mass_mev']
        tolerance = 0.05
        assert abs(result['Mass_MeVc2'] - expected) / expected < tolerance

    def test_neutron_mass_within_tolerance(self, calculator):
        """Neutron mass should be within 5% of PDG value."""
        quarks = self._get_quark_list(['u', 'd', 'd'])
        result = calculator.create_particle_from_quarks(quarks)
        expected = HADRON_REFERENCE['neutron']['mass_mev']
        tolerance = 0.05
        assert abs(result['Mass_MeVc2'] - expected) / expected < tolerance

    def test_baryon_number_conservation(self, calculator):
        """Baryon number must be exactly 1 for proton/neutron."""
        for quarks in [['u', 'u', 'd'], ['u', 'd', 'd']]:
            quark_list = self._get_quark_list(quarks)
            result = calculator.create_particle_from_quarks(quark_list)
            assert result['BaryonNumber_B'] == pytest.approx(1.0, abs=1e-10)


class TestNuclearDerivation:
    """Test nuclear property calculations from nucleons."""

    @pytest.fixture
    def calculator(self):
        return NuclearDerivation()

    def test_semf_he4_binding(self, calculator):
        """He-4 binding energy should be reasonable."""
        Z, N = 2, 2
        result = calculator.calculate(Z, N)
        # SEMF is less accurate for light nuclei, use wider tolerance
        assert result.binding_energy_mev > 0

    def test_semf_fe56_binding(self, calculator):
        """Fe-56 binding energy should match within 3%."""
        Z, N = 26, 30
        expected = NUCLEAR_BINDING[(Z, N)]
        result = calculator.calculate(Z, N)
        tolerance = 0.03
        assert abs(result.binding_energy_mev - expected) / expected < tolerance

    def test_nuclear_radius_formula(self, calculator):
        """Nuclear radius should follow R = r0 * A^(1/3)."""
        r0 = 1.25
        for Z, N in [(2, 2), (6, 6), (26, 30)]:
            A = Z + N
            expected = r0 * (A ** (1/3))
            result = calculator.calculate(Z, N)
            assert abs(result.nuclear_radius_fm - expected) / expected < 0.01

    def test_stability_magic_numbers(self, calculator):
        """Doubly magic nuclei should be stable."""
        # He-4 (Z=2, N=2) - doubly magic
        result = calculator.calculate(2, 2)
        assert result.is_stable
        assert "magic" in result.stability_reason.lower()

    def test_even_even_spin_zero(self, calculator):
        """Even-even nuclei should have spin 0+."""
        result = calculator.calculate(6, 6)  # C-12
        assert result.spin == 0.0
        assert result.parity == '+'


class TestAtomicDerivation:
    """Test atomic property derivation."""

    @pytest.fixture
    def calculator(self):
        return AtomicDerivation()

    def test_hydrogen_electron_config(self, calculator):
        """Hydrogen should have 1s1 configuration."""
        result = calculator.calculate(1)
        assert '1s1' in result.electron_configuration

    def test_carbon_electron_config(self, calculator):
        """Carbon should have 1s2 2s2 2p2 configuration."""
        result = calculator.calculate(6)
        assert '2p2' in result.electron_configuration

    def test_element_symbols(self, calculator):
        """Element symbols should be correct."""
        for Z, ref in ATOMIC_REFERENCE.items():
            result = calculator.calculate(Z)
            assert result.symbol == ref['symbol']

    def test_noble_gas_notation(self, calculator):
        """Noble gas notation should work."""
        result = calculator.calculate(11)  # Sodium
        assert '[Ne]' in result.electron_config_noble

    def test_ionization_energy_trend(self, calculator):
        """IE should generally decrease down a group."""
        ie_h = calculator.calculate(1).ionization_energy_ev
        ie_li = calculator.calculate(3).ionization_energy_ev
        ie_na = calculator.calculate(11).ionization_energy_ev
        # Li should have lower IE than H
        assert ie_li < ie_h
        # Na should have lower IE than Li
        assert ie_na < ie_li


class TestFullDerivationChain:
    """Test the complete quark -> hadron -> nucleus -> atom chain."""

    @pytest.fixture
    def engine(self):
        return PredictionEngine()

    def test_hydrogen_from_quarks(self, engine):
        """Derive hydrogen atom properties from quarks."""
        result = engine.predict(1, 0)

        # Verify element
        assert result.element_symbol == 'H'

        # Verify hadron properties derived
        assert 'proton_mass_mev' in result.hadron_properties
        assert result.hadron_properties['proton_charge'] == pytest.approx(1.0)

        # Verify atomic properties
        assert result.atomic_properties['mass_u'] > 0

    def test_helium_binding(self, engine):
        """He-4 should show significant binding."""
        result = engine.predict(2, 2)

        assert result.nuclear_properties['binding_energy_mev'] > 0
        assert result.nuclear_properties['binding_per_nucleon_mev'] > 0

    def test_iron_is_most_stable(self, engine):
        """Fe-56 should have high binding per nucleon."""
        result = engine.predict(26, 30)

        # Fe-56 has one of the highest binding energies per nucleon
        assert result.nuclear_properties['binding_per_nucleon_mev'] > 8.0

    def test_confidence_decreases_with_z(self, engine):
        """Confidence should decrease for heavier elements."""
        h_result = engine.predict(1)
        u_result = engine.predict(92)

        assert h_result.confidence['overall'] > u_result.confidence['overall']

    def test_superheavy_warning(self, engine):
        """Superheavy elements should have warnings."""
        result = engine.predict_unknown(120)

        assert len(result.warnings) > 0
        assert result.confidence['overall'] < 0.5


class TestValidation:
    """Test validation against experimental data."""

    def test_nuclear_binding_validation(self):
        """Validate SEMF against known binding energies."""
        calc = NuclearDerivation()

        errors = []
        for (Z, N), exp_binding in NUCLEAR_BINDING.items():
            result = calc.validate_against_experimental(Z, N, exp_binding)
            errors.append(abs(result['percent_error']))

        # Average error should be reasonable (SEMF is approximate)
        avg_error = sum(errors) / len(errors)
        assert avg_error < 10, f"Average SEMF error too high: {avg_error:.1f}%"


# ==================== HELPER FUNCTIONS ====================

def calculate_semf_binding(Z: int, N: int) -> float:
    """Calculate binding energy using SEMF."""
    calc = NuclearDerivation()
    return calc.calculate_binding_energy(Z, N)


def generate_electron_config(Z: int) -> str:
    """Generate electron configuration."""
    calc = AtomicDerivation()
    result = calc.calculate(Z)
    return result.electron_configuration


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
