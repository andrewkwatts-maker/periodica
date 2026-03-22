#!/usr/bin/env python3
"""
Test script to compare AtomCalculator predictions against real element data.
Identifies which formulas have the largest errors for refinement.
"""

import json
import os
import sys

# Add parent directory to path

from periodica.utils.physics_calculator import AtomCalculator


def load_element_data(symbol: str, atomic_number: int) -> dict:
    """Load element data from JSON file."""
    filename = f"{atomic_number:03d}_{symbol}.json"
    filepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "elements", filename
    )

    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_error(calculated: float, actual: float) -> tuple:
    """Calculate absolute and percentage error."""
    if actual is None or actual == 0:
        return (None, None)

    if calculated is None:
        return (None, None)

    abs_error = calculated - actual
    pct_error = (abs_error / abs(actual)) * 100

    return (abs_error, pct_error)


def get_most_abundant_isotope(element_data: dict) -> tuple:
    """Get the most abundant isotope's neutron count."""
    isotopes = element_data.get('isotopes', [])
    if not isotopes:
        # Estimate neutrons from atomic mass
        Z = element_data['atomic_number']
        A = round(element_data['atomic_mass'])
        return (A, A - Z)

    # Find most abundant isotope
    most_abundant = max(isotopes, key=lambda x: x.get('abundance', 0))
    return (most_abundant['mass_number'], most_abundant['neutrons'])


def run_comparison():
    """Run comparison for selected elements."""

    # Test elements covering different blocks and periods
    test_elements = [
        ('H', 1),    # s-block, period 1
        ('He', 2),   # s-block, period 1 (noble gas)
        ('C', 6),    # p-block, period 2
        ('N', 7),    # p-block, period 2
        ('O', 8),    # p-block, period 2
        ('Fe', 26),  # d-block, period 4
        ('Au', 79),  # d-block, period 6
        ('U', 92),   # f-block, period 7
    ]

    properties = [
        'atomic_mass',
        'ionization_energy',
        'electronegativity',
        'atomic_radius',
        'melting_point',
        'boiling_point',
        'density',
    ]

    # Store all errors for analysis
    all_errors = {prop: [] for prop in properties}

    print("=" * 120)
    print("ATOM CALCULATOR ACCURACY COMPARISON")
    print("=" * 120)
    print()

    for symbol, atomic_number in test_elements:
        print(f"\n{'='*80}")
        print(f"Element: {symbol} (Z={atomic_number})")
        print(f"{'='*80}")

        # Load actual data
        actual_data = load_element_data(symbol, atomic_number)

        # Get neutron count from most abundant isotope
        mass_number, neutrons = get_most_abundant_isotope(actual_data)

        print(f"Using isotope: A={mass_number}, N={neutrons}")
        print()

        # Calculate values using AtomCalculator
        calculated = {
            'atomic_mass': AtomCalculator.calculate_atomic_mass(atomic_number, neutrons),
            'ionization_energy': AtomCalculator.calculate_ionization_energy(atomic_number),
            'electronegativity': AtomCalculator.calculate_electronegativity(atomic_number),
            'atomic_radius': AtomCalculator.calculate_atomic_radius(atomic_number),
            'melting_point': AtomCalculator.calculate_melting_point(atomic_number, neutrons),
            'boiling_point': AtomCalculator.calculate_boiling_point(atomic_number, neutrons),
            'density': AtomCalculator.calculate_density(atomic_number, neutrons),
        }

        # Print comparison table
        print(f"{'Property':<25} {'Calculated':>15} {'Actual':>15} {'% Error':>12} {'Status':>10}")
        print("-" * 80)

        for prop in properties:
            calc_val = calculated[prop]
            actual_val = actual_data.get(prop)

            if actual_val is None:
                status = "N/A"
                pct_str = "N/A"
            else:
                _, pct_error = calculate_error(calc_val, actual_val)

                if pct_error is not None:
                    all_errors[prop].append((symbol, calc_val, actual_val, pct_error))
                    pct_str = f"{pct_error:+.1f}%"

                    if abs(pct_error) <= 10:
                        status = "GOOD"
                    elif abs(pct_error) <= 30:
                        status = "OK"
                    elif abs(pct_error) <= 50:
                        status = "POOR"
                    else:
                        status = "BAD"
                else:
                    pct_str = "N/A"
                    status = "N/A"

            print(f"{prop:<25} {calc_val:>15.4f} {str(actual_val):>15} {pct_str:>12} {status:>10}")

    # Print summary statistics
    print("\n")
    print("=" * 120)
    print("ERROR SUMMARY BY PROPERTY")
    print("=" * 120)
    print()
    print(f"{'Property':<25} {'Mean % Error':>15} {'Max % Error':>15} {'Min % Error':>15} {'Samples':>10}")
    print("-" * 90)

    for prop in properties:
        errors = all_errors[prop]
        if errors:
            pct_errors = [e[3] for e in errors if e[3] is not None]
            if pct_errors:
                mean_err = sum(abs(e) for e in pct_errors) / len(pct_errors)
                max_err = max(pct_errors, key=abs)
                min_err = min(pct_errors, key=abs)
                print(f"{prop:<25} {mean_err:>15.1f}% {max_err:>14.1f}% {min_err:>14.1f}% {len(pct_errors):>10}")

    # Print detailed error analysis
    print("\n")
    print("=" * 120)
    print("DETAILED ERROR ANALYSIS (for formula refinement)")
    print("=" * 120)

    for prop in properties:
        print(f"\n{prop.upper()}")
        print("-" * 60)
        errors = all_errors[prop]
        if errors:
            # Sort by absolute error
            sorted_errors = sorted(errors, key=lambda x: abs(x[3]) if x[3] else 0, reverse=True)
            for symbol, calc, actual, pct in sorted_errors:
                if pct is not None:
                    print(f"  {symbol:>3}: Calculated={calc:>12.4f}, Actual={actual:>12}, Error={pct:>+8.1f}%")

    return all_errors


if __name__ == "__main__":
    run_comparison()
