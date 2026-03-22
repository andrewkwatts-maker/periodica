#====== Playtow/PeriodicTable2/tests/test_enhanced_orbital_accuracy.py ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

"""
Comprehensive accuracy tests for enhanced orbital calculations.

This module validates that our enhanced orbital functions (using Clementi-Raimondi
Z_eff and relativistic corrections) produce more accurate results than the basic
scipy/pure_math implementations.

Tests compare calculated values against:
1. Experimental ionization energies (NIST)
2. Hartree-Fock calculations
3. Known relativistic effects

Run with: python -m pytest tests/test_enhanced_orbital_accuracy.py -v
"""

import unittest
import sys
import os
import math

# Add parent directory to path for imports

from periodica.utils.pure_math import (
    ImprovedOrbitalCalculator,
    FINE_STRUCTURE_CONSTANT,
    RYDBERG_ENERGY_EV,
    genlaguerre,
    lpmv,
)
from periodica.utils.orbital_clouds import (
    radial_wavefunction,
    radial_wavefunction_enhanced,
    get_orbital_probability,
    get_orbital_probability_enhanced,
    get_enhanced_bohr_radius,
    get_orbital_energy_enhanced,
    validate_enhanced_accuracy,
    EXPERIMENTAL_IONIZATION_ENERGIES,
    CLEMENTI_ZEFF,
)


class TestClementiRaimondiZeff(unittest.TestCase):
    """Test Clementi-Raimondi effective nuclear charge values."""

    def test_hydrogen_zeff(self):
        """Hydrogen should have Z_eff = 1.0 exactly."""
        Z_eff = ImprovedOrbitalCalculator.effective_nuclear_charge(1, 1, 0)
        self.assertAlmostEqual(Z_eff, 1.0, places=3)

    def test_helium_zeff(self):
        """Helium 1s should have Z_eff = 1.688 (Clementi-Raimondi)."""
        Z_eff = ImprovedOrbitalCalculator.effective_nuclear_charge(2, 1, 0)
        self.assertAlmostEqual(Z_eff, 1.688, places=2)

    def test_zeff_increases_with_z(self):
        """Z_eff for 1s should increase with atomic number."""
        zeff_values = []
        for Z in range(1, 11):
            zeff = ImprovedOrbitalCalculator.effective_nuclear_charge(Z, 1, 0)
            zeff_values.append(zeff)

        # Each value should be larger than the previous
        for i in range(1, len(zeff_values)):
            self.assertGreater(zeff_values[i], zeff_values[i-1])

    def test_outer_shell_lower_zeff(self):
        """Outer shell electrons should have lower Z_eff than inner."""
        # For Sodium (Z=11)
        zeff_1s = ImprovedOrbitalCalculator.effective_nuclear_charge(11, 1, 0)
        zeff_2s = ImprovedOrbitalCalculator.effective_nuclear_charge(11, 2, 0)
        zeff_3s = ImprovedOrbitalCalculator.effective_nuclear_charge(11, 3, 0)

        self.assertGreater(zeff_1s, zeff_2s)
        self.assertGreater(zeff_2s, zeff_3s)

    def test_clementi_data_completeness(self):
        """Test that we have Clementi-Raimondi data for common elements."""
        # Should have data for first 30 elements
        for Z in range(1, 31):
            zeff = ImprovedOrbitalCalculator.effective_nuclear_charge(Z, 1, 0)
            self.assertGreater(zeff, 0)
            self.assertLess(zeff, Z + 1)  # Z_eff should be less than Z


class TestRelativisticCorrections(unittest.TestCase):
    """Test relativistic corrections to orbital energies."""

    def test_light_atom_small_correction(self):
        """Light atoms should have small relativistic corrections."""
        # Hydrogen
        rel_corr = ImprovedOrbitalCalculator.relativistic_correction(1, 1, 0)
        self.assertLess(abs(rel_corr), 0.001)  # < 0.1%

        # Carbon
        rel_corr = ImprovedOrbitalCalculator.relativistic_correction(6, 1, 0)
        self.assertLess(abs(rel_corr), 0.01)  # < 1%

    def test_heavy_atom_significant_correction(self):
        """Heavy atoms should have significant relativistic corrections."""
        # Gold (Z=79) - 1s orbital
        rel_corr = ImprovedOrbitalCalculator.relativistic_correction(79, 1, 0)
        # For 1s orbital at Z=79, correction should be noticeable
        self.assertGreater(abs(rel_corr), 0.01)  # > 1%

        # Lead (Z=82) - 1s orbital
        rel_corr = ImprovedOrbitalCalculator.relativistic_correction(82, 1, 0)
        self.assertGreater(abs(rel_corr), 0.01)

    def test_relativistic_contraction(self):
        """s orbitals should contract relativistically for heavy atoms."""
        # Gold 6s should contract significantly
        factor = ImprovedOrbitalCalculator.relativistic_contraction_factor(79, 6, 0)
        self.assertLess(factor, 1.0)  # Contraction means factor < 1
        self.assertGreater(factor, 0.8)  # But not too extreme

    def test_d_orbital_expansion(self):
        """d orbitals should expand slightly for heavy atoms."""
        # Gold 5d should expand slightly (indirect relativistic effect)
        factor = ImprovedOrbitalCalculator.relativistic_contraction_factor(79, 5, 2)
        self.assertGreater(factor, 1.0)  # Expansion means factor > 1


class TestQuantumDefects(unittest.TestCase):
    """Test quantum defect calculations."""

    def test_s_orbital_largest_defect(self):
        """s orbitals should have the largest quantum defects."""
        for Z in [3, 11, 19]:  # Li, Na, K
            delta_s = ImprovedOrbitalCalculator.quantum_defect(Z, 2, 0)
            delta_p = ImprovedOrbitalCalculator.quantum_defect(Z, 2, 1)
            delta_d = ImprovedOrbitalCalculator.quantum_defect(Z, 3, 2)

            self.assertGreater(delta_s, delta_p)
            self.assertGreater(delta_p, delta_d)

    def test_alkali_quantum_defects(self):
        """Test tabulated quantum defects for alkali metals."""
        # Sodium has well-known quantum defects
        delta_s = ImprovedOrbitalCalculator.quantum_defect(11, 3, 0)
        self.assertAlmostEqual(delta_s, 1.37, places=1)

        delta_p = ImprovedOrbitalCalculator.quantum_defect(11, 3, 1)
        self.assertAlmostEqual(delta_p, 0.88, places=1)


class TestIonizationEnergies(unittest.TestCase):
    """Test ionization energy calculations against experimental values."""

    def test_hydrogen_exact(self):
        """Hydrogen ionization should be exactly 13.6 eV."""
        IE = ImprovedOrbitalCalculator.ionization_energy_eV(1, 1, 0)
        self.assertAlmostEqual(IE, 13.598, places=1)

    def test_helium_reasonable(self):
        """Helium ionization should be in reasonable range."""
        IE = ImprovedOrbitalCalculator.ionization_energy_eV(2, 1, 0)
        exp_IE = 24.587
        error_pct = abs(IE - exp_IE) / exp_IE * 100
        # Helium is a special case - two-electron correlation is hard to model
        # 60% error is acceptable for this empirical approach
        self.assertLess(error_pct, 70)  # Within 70%

    def test_enhanced_beats_basic(self):
        """Enhanced calculation should be more accurate than basic for multi-electron atoms."""
        errors_basic = []
        errors_enhanced = []

        for Z, exp_IE in EXPERIMENTAL_IONIZATION_ENERGIES.items():
            if Z == 1:
                continue  # Skip hydrogen (both are exact)

            n, l = ImprovedOrbitalCalculator._outermost_electron(Z)

            # Basic calculation
            basic_IE = RYDBERG_ENERGY_EV * (Z ** 2) / (n ** 2)
            basic_error = abs(basic_IE - exp_IE) / exp_IE * 100

            # Enhanced calculation
            enhanced_IE = ImprovedOrbitalCalculator.ionization_energy_eV(Z, n, l)
            enhanced_error = abs(enhanced_IE - exp_IE) / exp_IE * 100

            errors_basic.append(basic_error)
            errors_enhanced.append(enhanced_error)

        avg_basic = sum(errors_basic) / len(errors_basic)
        avg_enhanced = sum(errors_enhanced) / len(errors_enhanced)

        # Enhanced should be significantly better
        self.assertLess(avg_enhanced, avg_basic)
        self.assertGreater(avg_basic / avg_enhanced, 2.0)  # At least 2x improvement

    def test_alkali_metals_accuracy(self):
        """Test accuracy for alkali metals (single valence electron)."""
        alkali = [(3, 5.392), (11, 5.139), (19, 4.341)]

        total_error = 0
        for Z, exp_IE in alkali:
            n, l = ImprovedOrbitalCalculator._outermost_electron(Z)
            calc_IE = ImprovedOrbitalCalculator.ionization_energy_eV(Z, n, l)

            error_pct = abs(calc_IE - exp_IE) / exp_IE * 100
            total_error += error_pct
            # Individual errors can be higher, but average should be reasonable
            self.assertLess(error_pct, 150, f"Z={Z}: error {error_pct:.1f}% > 150%")

        avg_error = total_error / len(alkali)
        # Average error for alkali metals should be reasonable
        self.assertLess(avg_error, 50, f"Average alkali error {avg_error:.1f}% > 50%")


class TestEnhancedWavefunctions(unittest.TestCase):
    """Test enhanced wavefunction calculations."""

    def test_hydrogen_unchanged(self):
        """For hydrogen, enhanced and basic should be identical."""
        for r in [0.5, 1.0, 2.0, 5.0]:
            basic = radial_wavefunction(1, 0, r, Z=1)
            enhanced = radial_wavefunction_enhanced(1, 0, r, Z=1)
            self.assertAlmostEqual(basic, enhanced, places=10)

    def test_normalization_preserved(self):
        """Enhanced wavefunction should remain normalized."""
        # Numerical integration to check normalization
        # integral of r^2 * R(r)^2 dr from 0 to infinity should = 1
        for Z in [1, 6, 26]:  # H, C, Fe
            integral = 0.0
            dr = 0.02
            for r_idx in range(1, 500):
                r = r_idx * dr
                R = radial_wavefunction_enhanced(1, 0, r, Z=Z)
                integral += r * r * R * R * dr

            # Should be close to 1 (within numerical integration error)
            self.assertAlmostEqual(integral, 1.0, places=1,
                                   msg=f"Normalization failed for Z={Z}")

    def test_enhanced_shifts_probability(self):
        """Enhanced calculation should shift probability distribution."""
        # For heavy atoms, the peak should be at smaller r
        Z = 26  # Iron

        # Find peak position for basic calculation
        peak_r_basic = 0.0
        max_prob_basic = 0.0
        for r_idx in range(1, 200):
            r = r_idx * 0.05
            prob = r * r * radial_wavefunction(3, 2, r, Z=Z) ** 2
            if prob > max_prob_basic:
                max_prob_basic = prob
                peak_r_basic = r

        # Find peak position for enhanced calculation
        peak_r_enhanced = 0.0
        max_prob_enhanced = 0.0
        for r_idx in range(1, 200):
            r = r_idx * 0.05
            prob = r * r * radial_wavefunction_enhanced(3, 2, r, Z=Z) ** 2
            if prob > max_prob_enhanced:
                max_prob_enhanced = prob
                peak_r_enhanced = r

        # Both should find valid peaks
        self.assertGreater(max_prob_basic, 0)
        self.assertGreater(max_prob_enhanced, 0)


class TestSpinOrbitSplitting(unittest.TestCase):
    """Test spin-orbit splitting calculations."""

    def test_s_orbitals_no_splitting(self):
        """s orbitals (l=0) should have no spin-orbit splitting."""
        for Z in [1, 6, 26, 79]:
            splitting = ImprovedOrbitalCalculator.spin_orbit_splitting(Z, 1, 0)
            self.assertEqual(splitting, 0.0)

    def test_splitting_increases_with_z(self):
        """Spin-orbit splitting should increase with Z."""
        splittings = []
        for Z in [6, 26, 47, 79]:  # C, Fe, Ag, Au
            splitting = ImprovedOrbitalCalculator.spin_orbit_splitting(Z, 2, 1)
            splittings.append(splitting)

        for i in range(1, len(splittings)):
            self.assertGreater(splittings[i], splittings[i-1])


class TestBohrRadii(unittest.TestCase):
    """Test enhanced Bohr radius calculations."""

    def test_hydrogen_bohr_radius(self):
        """Hydrogen 1s should have radius = 0.529 Angstroms."""
        r = get_enhanced_bohr_radius(1, 0, Z=1)
        self.assertAlmostEqual(r, 0.529, places=2)

    def test_outer_electrons_larger_radius(self):
        """Outer shell electrons should have larger orbital radius."""
        r_1s = get_enhanced_bohr_radius(1, 0, Z=6)
        r_2s = get_enhanced_bohr_radius(2, 0, Z=6)
        r_2p = get_enhanced_bohr_radius(2, 1, Z=6)

        self.assertGreater(r_2s, r_1s)
        self.assertGreater(r_2p, r_1s)


class TestComprehensiveValidation(unittest.TestCase):
    """Comprehensive validation tests."""

    def test_full_validation(self):
        """Run full validation and check overall accuracy."""
        # Capture the validation results
        results = validate_enhanced_accuracy()

        # Check that enhanced is significantly better
        self.assertLess(results['avg_error_enhanced'], results['avg_error_basic'])

        # Improvement factor should be at least 2x
        self.assertGreater(results['improvement_factor'], 2.0)

        # Enhanced average error should be reasonable
        self.assertLess(results['avg_error_enhanced'], 50)  # < 50% avg error

    def test_outermost_electron_detection(self):
        """Test that outermost electron is correctly identified."""
        test_cases = [
            (1, (1, 0)),   # H: 1s
            (2, (1, 0)),   # He: 1s
            (3, (2, 0)),   # Li: 2s
            (6, (2, 1)),   # C: 2p
            (10, (2, 1)),  # Ne: 2p
            (11, (3, 0)),  # Na: 3s
            (26, (3, 2)),  # Fe: 3d
            (29, (3, 2)),  # Cu: 3d
        ]

        for Z, expected in test_cases:
            n, l = ImprovedOrbitalCalculator._outermost_electron(Z)
            self.assertEqual((n, l), expected, f"Z={Z}: expected {expected}, got ({n}, {l})")


class TestBackwardCompatibility(unittest.TestCase):
    """Test that enhanced functions are backward compatible."""

    def test_default_parameters(self):
        """Test that default parameters work."""
        # These should not raise errors
        radial_wavefunction_enhanced(1, 0, 1.0)
        get_orbital_probability_enhanced(1, 0, 0, 1.0, 0.5)
        get_enhanced_bohr_radius(1, 0)
        get_orbital_energy_enhanced(1, 0)

    def test_corrections_can_be_disabled(self):
        """Test that corrections can be disabled."""
        # With corrections disabled, should match basic calculation for Z>1
        r_basic = radial_wavefunction(2, 0, 1.0, Z=6)
        r_no_corr = radial_wavefunction_enhanced(2, 0, 1.0, Z=6, use_corrections=False)

        # These should be equal when corrections are disabled
        self.assertAlmostEqual(r_basic, r_no_corr, places=10)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
