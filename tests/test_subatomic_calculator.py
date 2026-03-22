"""
Test script for SubatomicCalculator
Compares predictions against real particle data from JSON files.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periodica.utils.physics_calculator import SubatomicCalculator


# Real particle data for validation
# Format: (name, quarks, actual_mass_MeV, actual_charge, actual_spin, spin_aligned)
PARTICLE_DATA = [
    # Baryons - Ground state (spin 1/2)
    ("Proton", ['u', 'u', 'd'], 938.27, 1.0, 0.5, False),
    ("Neutron", ['u', 'd', 'd'], 939.57, 0.0, 0.5, False),
    ("Lambda", ['u', 'd', 's'], 1115.68, 0.0, 0.5, False),
    ("Sigma+", ['u', 'u', 's'], 1189.37, 1.0, 0.5, False),
    ("Sigma-", ['d', 'd', 's'], 1197.45, -1.0, 0.5, False),
    ("Xi0", ['u', 's', 's'], 1314.86, 0.0, 0.5, False),
    ("Xi-", ['d', 's', 's'], 1321.71, -1.0, 0.5, False),

    # Baryons - Delta resonances (spin 3/2)
    ("Delta++", ['u', 'u', 'u'], 1232.0, 2.0, 1.5, True),
    ("Delta+", ['u', 'u', 'd'], 1232.0, 1.0, 1.5, True),
    ("Delta0", ['u', 'd', 'd'], 1232.0, 0.0, 1.5, True),
    ("Delta-", ['d', 'd', 'd'], 1232.0, -1.0, 1.5, True),
    ("Omega-", ['s', 's', 's'], 1672.45, -1.0, 1.5, True),

    # Mesons - Pseudoscalar (spin 0)
    ("Pion+", ['u', 'd̅'], 139.57, 1.0, 0.0, False),
    ("Pion-", ['d', 'u̅'], 139.57, -1.0, 0.0, False),
    ("Pion0", ['u', 'u̅'], 134.98, 0.0, 0.0, False),  # Actually (uu̅-dd̅)/sqrt(2)
    ("Kaon+", ['u', 's̅'], 493.68, 1.0, 0.0, False),
    ("Kaon-", ['s', 'u̅'], 493.68, -1.0, 0.0, False),
    ("Kaon0", ['d', 's̅'], 497.61, 0.0, 0.0, False),
]


def test_charge_calculation():
    """Test charge calculation for all particles."""
    print("\n" + "="*80)
    print("CHARGE CALCULATION TEST")
    print("="*80)
    print(f"{'Particle':<12} {'Quarks':<15} {'Expected':>10} {'Calculated':>12} {'Status':>10}")
    print("-"*80)

    all_passed = True
    for name, quarks, _, expected_charge, _, _ in PARTICLE_DATA:
        calculated = SubatomicCalculator.calculate_charge(quarks)
        # Allow small floating point differences
        passed = abs(calculated - expected_charge) < 0.01
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        quark_str = ''.join(quarks)
        print(f"{name:<12} {quark_str:<15} {expected_charge:>10.2f} {calculated:>12.2f} {status:>10}")

    assert all_passed, "Not all tests passed"


def test_spin_calculation():
    """Test spin calculation for all particles."""
    print("\n" + "="*80)
    print("SPIN CALCULATION TEST")
    print("="*80)
    print(f"{'Particle':<12} {'Quarks':<15} {'Aligned':>8} {'Expected':>10} {'Calculated':>12} {'Status':>10}")
    print("-"*80)

    all_passed = True
    for name, quarks, _, _, expected_spin, spin_aligned in PARTICLE_DATA:
        calculated = SubatomicCalculator.calculate_spin(quarks, spin_aligned)
        passed = abs(calculated - expected_spin) < 0.01
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        quark_str = ''.join(quarks)
        aligned_str = "Yes" if spin_aligned else "No"
        print(f"{name:<12} {quark_str:<15} {aligned_str:>8} {expected_spin:>10.1f} {calculated:>12.1f} {status:>10}")

    assert all_passed, "Not all tests passed"


def test_mass_calculation():
    """Test mass calculation for all particles."""
    print("\n" + "="*80)
    print("MASS CALCULATION TEST")
    print("="*80)
    print(f"{'Particle':<12} {'Quarks':<15} {'Expected':>12} {'Calculated':>12} {'Error %':>10} {'Status':>10}")
    print("-"*80)

    all_passed = True
    total_error = 0
    for name, quarks, expected_mass, _, _, spin_aligned in PARTICLE_DATA:
        # Try to call with spin_aligned if supported, else fall back
        try:
            calculated = SubatomicCalculator.calculate_mass(quarks, spin_aligned=spin_aligned)
        except TypeError:
            calculated = SubatomicCalculator.calculate_mass(quarks)

        error_pct = abs(calculated - expected_mass) / expected_mass * 100
        total_error += error_pct

        # Pass if within 10% of actual mass
        passed = error_pct < 10
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        quark_str = ''.join(quarks)
        print(f"{name:<12} {quark_str:<15} {expected_mass:>12.2f} {calculated:>12.2f} {error_pct:>9.1f}% {status:>10}")

    avg_error = total_error / len(PARTICLE_DATA)
    print("-"*80)
    print(f"Average Error: {avg_error:.1f}%")

    assert all_passed, "Not all tests passed"


def test_particle_type():
    """Test particle type determination."""
    print("\n" + "="*80)
    print("PARTICLE TYPE TEST")
    print("="*80)
    print(f"{'Particle':<12} {'Quarks':<15} {'Expected':>15} {'Calculated':>15} {'Status':>10}")
    print("-"*80)

    test_cases = [
        ("Proton", ['u', 'u', 'd'], "Baryon"),
        ("Neutron", ['u', 'd', 'd'], "Baryon"),
        ("Pion+", ['u', 'd̅'], "Meson"),
        ("Kaon+", ['u', 's̅'], "Meson"),
    ]

    all_passed = True
    for name, quarks, expected_type in test_cases:
        calculated = SubatomicCalculator.determine_particle_type(quarks)
        passed = calculated == expected_type
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        quark_str = ''.join(quarks)
        print(f"{name:<12} {quark_str:<15} {expected_type:>15} {calculated:>15} {status:>10}")

    assert all_passed, "Not all tests passed"


def test_baryon_number():
    """Test baryon number calculation."""
    print("\n" + "="*80)
    print("BARYON NUMBER TEST")
    print("="*80)
    print(f"{'Particle':<12} {'Quarks':<15} {'Expected':>10} {'Calculated':>12} {'Status':>10}")
    print("-"*80)

    test_cases = [
        ("Proton", ['u', 'u', 'd'], 1),
        ("Pion+", ['u', 'd̅'], 0),
        ("Kaon+", ['u', 's̅'], 0),
        ("Lambda", ['u', 'd', 's'], 1),
    ]

    all_passed = True
    for name, quarks, expected in test_cases:
        calculated = SubatomicCalculator.calculate_baryon_number(quarks)
        passed = calculated == expected
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        quark_str = ''.join(quarks)
        print(f"{name:<12} {quark_str:<15} {expected:>10} {calculated:>12} {status:>10}")

    assert all_passed, "Not all tests passed"


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("SubatomicCalculator Validation Tests")
    print("="*80)

    results = {
        "Charge": test_charge_calculation(),
        "Spin": test_spin_calculation(),
        "Mass": test_mass_calculation(),
        "Particle Type": test_particle_type(),
        "Baryon Number": test_baryon_number(),
    }

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:<20}: {status}")

    all_passed = all(results.values())
    print("="*80)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    assert all_passed, "Not all tests passed"


if __name__ == "__main__":
    run_all_tests()
