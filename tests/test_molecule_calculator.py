"""
Test script for MoleculeCalculator
Compares predictions against real molecule data from JSON files.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.physics_calculator import MoleculeCalculator


# Real molecule data from JSON files
REAL_MOLECULES = {
    "Water": {
        "composition": [{"Element": "H", "Count": 2}, {"Element": "O", "Count": 1}],
        "actual": {
            "MolecularMass_amu": 18.015,
            "MeltingPoint_K": 273.15,
            "BoilingPoint_K": 373.15,
            "Density_g_cm3": 1.0,
            "Geometry": "Bent",
            "BondType": "Covalent",
            "Polarity": "Polar"
        }
    },
    "Methane": {
        "composition": [{"Element": "C", "Count": 1}, {"Element": "H", "Count": 4}],
        "actual": {
            "MolecularMass_amu": 16.04,
            "MeltingPoint_K": 90.7,
            "BoilingPoint_K": 111.65,
            "Density_g_cm3": 0.000656,
            "Geometry": "Tetrahedral",
            "BondType": "Covalent",
            "Polarity": "Nonpolar"
        }
    },
    "Carbon Dioxide": {
        "composition": [{"Element": "C", "Count": 1}, {"Element": "O", "Count": 2}],
        "actual": {
            "MolecularMass_amu": 44.01,
            "MeltingPoint_K": 194.65,
            "BoilingPoint_K": 216.55,
            "Density_g_cm3": 0.001977,
            "Geometry": "Linear",
            "BondType": "Covalent",
            "Polarity": "Nonpolar"
        }
    },
    "Ammonia": {
        "composition": [{"Element": "N", "Count": 1}, {"Element": "H", "Count": 3}],
        "actual": {
            "MolecularMass_amu": 17.031,
            "MeltingPoint_K": 195.42,
            "BoilingPoint_K": 239.81,
            "Density_g_cm3": 0.000730,
            "Geometry": "Trigonal Pyramidal",
            "BondType": "Covalent",
            "Polarity": "Polar"
        }
    },
    "Ethanol": {
        "composition": [{"Element": "C", "Count": 2}, {"Element": "H", "Count": 6}, {"Element": "O", "Count": 1}],
        "actual": {
            "MolecularMass_amu": 46.07,
            "MeltingPoint_K": 159.0,
            "BoilingPoint_K": 351.5,
            "Density_g_cm3": 0.789,
            "Geometry": "Tetrahedral",
            "BondType": "Covalent",
            "Polarity": "Polar"
        }
    },
    "Sodium Chloride": {
        "composition": [{"Element": "Na", "Count": 1}, {"Element": "Cl", "Count": 1}],
        "actual": {
            "MolecularMass_amu": 58.44,
            "MeltingPoint_K": 1074.0,
            "BoilingPoint_K": 1686.0,
            "Density_g_cm3": 2.165,
            "Geometry": "Face-centered Cubic",
            "BondType": "Ionic",
            "Polarity": "Ionic"
        }
    }
}


def calculate_error(predicted, actual):
    """Calculate percentage error."""
    if actual == 0:
        return float('inf') if predicted != 0 else 0
    return abs((predicted - actual) / actual) * 100


def check_molecule(name, data):
    """Test a single molecule and return results."""
    composition = data["composition"]
    actual = data["actual"]

    # Get predictions from calculator
    predicted_mass = MoleculeCalculator.calculate_molecular_mass(composition)
    predicted_geometry = MoleculeCalculator.estimate_geometry(composition)
    predicted_bond_type = MoleculeCalculator.determine_bond_type(composition)
    predicted_polarity = MoleculeCalculator.estimate_polarity(composition, predicted_geometry)
    predicted_mp = MoleculeCalculator.estimate_melting_point(predicted_mass, predicted_polarity, predicted_bond_type, composition)
    predicted_bp = MoleculeCalculator.estimate_boiling_point(predicted_mp, predicted_polarity, composition, predicted_mass)
    predicted_state = MoleculeCalculator.determine_state(predicted_mp, predicted_bp)
    predicted_density = MoleculeCalculator.estimate_density(predicted_mass, composition, predicted_state, predicted_bond_type)

    results = {
        "name": name,
        "mass": {
            "predicted": predicted_mass,
            "actual": actual["MolecularMass_amu"],
            "error": calculate_error(predicted_mass, actual["MolecularMass_amu"])
        },
        "geometry": {
            "predicted": predicted_geometry,
            "actual": actual["Geometry"],
            "correct": predicted_geometry == actual["Geometry"]
        },
        "bond_type": {
            "predicted": predicted_bond_type,
            "actual": actual["BondType"],
            "correct": predicted_bond_type == actual["BondType"]
        },
        "polarity": {
            "predicted": predicted_polarity,
            "actual": actual["Polarity"],
            "correct": predicted_polarity == actual["Polarity"]
        },
        "melting_point": {
            "predicted": predicted_mp,
            "actual": actual["MeltingPoint_K"],
            "error": calculate_error(predicted_mp, actual["MeltingPoint_K"])
        },
        "boiling_point": {
            "predicted": predicted_bp,
            "actual": actual["BoilingPoint_K"],
            "error": calculate_error(predicted_bp, actual["BoilingPoint_K"])
        },
        "density": {
            "predicted": predicted_density,
            "actual": actual["Density_g_cm3"],
            "error": calculate_error(predicted_density, actual["Density_g_cm3"])
        }
    }

    return results


def print_results(results):
    """Print test results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"MOLECULE: {results['name']}")
    print(f"{'='*60}")

    # Mass
    mass = results['mass']
    status = "PASS" if mass['error'] < 1 else "FAIL"
    print(f"Mass:       {mass['predicted']:.3f} vs {mass['actual']:.3f} amu ({mass['error']:.1f}% error) [{status}]")

    # Geometry
    geom = results['geometry']
    status = "PASS" if geom['correct'] else "FAIL"
    print(f"Geometry:   {geom['predicted']} vs {geom['actual']} [{status}]")

    # Bond Type
    bond = results['bond_type']
    status = "PASS" if bond['correct'] else "FAIL"
    print(f"Bond Type:  {bond['predicted']} vs {bond['actual']} [{status}]")

    # Polarity
    polar = results['polarity']
    status = "PASS" if polar['correct'] else "FAIL"
    print(f"Polarity:   {polar['predicted']} vs {polar['actual']} [{status}]")

    # Melting Point
    mp = results['melting_point']
    status = "PASS" if mp['error'] < 20 else "WARN" if mp['error'] < 50 else "FAIL"
    print(f"Melting Pt: {mp['predicted']:.1f} vs {mp['actual']:.1f} K ({mp['error']:.1f}% error) [{status}]")

    # Boiling Point
    bp = results['boiling_point']
    status = "PASS" if bp['error'] < 20 else "WARN" if bp['error'] < 50 else "FAIL"
    print(f"Boiling Pt: {bp['predicted']:.1f} vs {bp['actual']:.1f} K ({bp['error']:.1f}% error) [{status}]")

    # Density
    dens = results['density']
    status = "PASS" if dens['error'] < 50 else "WARN" if dens['error'] < 100 else "FAIL"
    print(f"Density:    {dens['predicted']:.4f} vs {dens['actual']:.4f} g/cm3 ({dens['error']:.1f}% error) [{status}]")


def run_all_tests():
    """Run all molecule tests."""
    print("\n" + "="*60)
    print("MOLECULE CALCULATOR TEST SUITE")
    print("="*60)

    all_results = []

    for name, data in REAL_MOLECULES.items():
        results = check_molecule(name, data)
        all_results.append(results)
        print_results(results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    mass_errors = [r['mass']['error'] for r in all_results]
    mp_errors = [r['melting_point']['error'] for r in all_results]
    bp_errors = [r['boiling_point']['error'] for r in all_results]

    geometry_correct = sum(1 for r in all_results if r['geometry']['correct'])
    bond_type_correct = sum(1 for r in all_results if r['bond_type']['correct'])
    polarity_correct = sum(1 for r in all_results if r['polarity']['correct'])

    print(f"\nMass Accuracy:      avg error = {sum(mass_errors)/len(mass_errors):.1f}%")
    print(f"Geometry Accuracy:  {geometry_correct}/{len(all_results)} correct")
    print(f"Bond Type Accuracy: {bond_type_correct}/{len(all_results)} correct")
    print(f"Polarity Accuracy:  {polarity_correct}/{len(all_results)} correct")
    print(f"Melting Pt Accuracy: avg error = {sum(mp_errors)/len(mp_errors):.1f}%")
    print(f"Boiling Pt Accuracy: avg error = {sum(bp_errors)/len(bp_errors):.1f}%")

    return all_results


if __name__ == "__main__":
    run_all_tests()
