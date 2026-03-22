#!/usr/bin/env python3
"""
Comprehensive Quark to Hadron Test Suite
==========================================

Tests building ALL hadrons from their constituent quarks using the JSON quark data.

For EACH subatomic particle (hadron) in the JSON files:
   - Reads its Composition field to get the constituent quarks
   - Builds the hadron using `propagate_quark_to_hadron` from `utils/simulation_schema.py`
   - Compares calculated properties (charge, mass, baryon_number, spin) against JSON values
   - Tracks pass/fail for each property

Author: Periodics Test Suite
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.simulation_schema import propagate_quark_to_hadron


# =============================================================================
# Constants and Mappings
# =============================================================================

# All quark files in data/active/quarks/
QUARK_FILES = [
    "UpQuark.json",
    "DownQuark.json",
    "StrangeQuark.json",
    "CharmQuark.json",
    "BottomQuark.json",
    "TopQuark.json",
    "Electron.json",
    "Muon.json",
    "TauLepton.json",
    "Photon.json",
    "Gluon.json",
    "WPlusBoson.json",
    "ZBoson.json",
    "Higgs Boson.json",
    "ElectronNeutrino.json",
    "MuonNuetrino.json",  # Note: typo in filename
    "TauNeutrino.json",
]

# All subatomic particle files in data/active/subatomic/
SUBATOMIC_FILES = [
    "Proton.json",
    "Neutron.json",
    "PionPlus.json",
    "PionMinus.json",
    "PionZero.json",
    "KaonPlus.json",
    "KaonMinus.json",
    "KaonZero.json",
    "Lambda.json",
    "SigmaPlus.json",
    "SigmaZero.json",
    "SigmaMinus.json",
    "DeltaPlusPlus.json",
    "DeltaPlus.json",
    "DeltaZero.json",
    "DeltaMinus.json",
    "Xi_Zero.json",
    "Xi_Minus.json",
    "Omega_Minus.json",
    "Eta.json",
    "JPsi.json",
    "Upsilon.json",
]

# Mapping from constituent names to quark symbols
CONSTITUENT_TO_SYMBOL = {
    # Quarks
    "Up Quark": "u",
    "Down Quark": "d",
    "Strange Quark": "s",
    "Charm Quark": "c",
    "Bottom Quark": "b",
    "Top Quark": "t",
    # Anti-quarks
    "Anti-Up Quark": "u\u0305",
    "Anti-Down Quark": "d\u0305",
    "Anti-Strange Quark": "s\u0305",
    "Anti-Charm Quark": "c\u0305",
    "Anti-Bottom Quark": "b\u0305",
    "Anti-Top Quark": "t\u0305",
    # Alternative spellings
    "Antiup Quark": "u\u0305",
    "Antidown Quark": "d\u0305",
    "Antistrange Quark": "s\u0305",
    "Anticharm Quark": "c\u0305",
    "Antibottom Quark": "b\u0305",
    "Antitop Quark": "t\u0305",
}

# Mapping from JSON symbol fields to standard symbols
SYMBOL_TO_STANDARD = {
    "u": "u",
    "d": "d",
    "s": "s",
    "c": "c",
    "b": "b",
    "t": "t",
    "u-bar": "u\u0305",
    "d-bar": "d\u0305",
    "s-bar": "s\u0305",
    "c-bar": "c\u0305",
    "b-bar": "b\u0305",
    "t-bar": "t\u0305",
}

# Quark properties for building hadrons
QUARK_PROPERTIES = {
    "u": {"Charge_e": 2/3, "BaryonNumber_B": 1/3, "Mass_MeVc2": 2.2, "Isospin_I3": 0.5},
    "d": {"Charge_e": -1/3, "BaryonNumber_B": 1/3, "Mass_MeVc2": 4.7, "Isospin_I3": -0.5},
    "s": {"Charge_e": -1/3, "BaryonNumber_B": 1/3, "Mass_MeVc2": 95.0, "Isospin_I3": 0.0},
    "c": {"Charge_e": 2/3, "BaryonNumber_B": 1/3, "Mass_MeVc2": 1270.0, "Isospin_I3": 0.0},
    "b": {"Charge_e": -1/3, "BaryonNumber_B": 1/3, "Mass_MeVc2": 4180.0, "Isospin_I3": 0.0},
    "t": {"Charge_e": 2/3, "BaryonNumber_B": 1/3, "Mass_MeVc2": 173000.0, "Isospin_I3": 0.0},
    # Anti-quarks (inverted charge and baryon number)
    "u\u0305": {"Charge_e": -2/3, "BaryonNumber_B": -1/3, "Mass_MeVc2": 2.2, "Isospin_I3": -0.5},
    "d\u0305": {"Charge_e": 1/3, "BaryonNumber_B": -1/3, "Mass_MeVc2": 4.7, "Isospin_I3": 0.5},
    "s\u0305": {"Charge_e": 1/3, "BaryonNumber_B": -1/3, "Mass_MeVc2": 95.0, "Isospin_I3": 0.0},
    "c\u0305": {"Charge_e": -2/3, "BaryonNumber_B": -1/3, "Mass_MeVc2": 1270.0, "Isospin_I3": 0.0},
    "b\u0305": {"Charge_e": 1/3, "BaryonNumber_B": -1/3, "Mass_MeVc2": 4180.0, "Isospin_I3": 0.0},
    "t\u0305": {"Charge_e": -2/3, "BaryonNumber_B": -1/3, "Mass_MeVc2": 173000.0, "Isospin_I3": 0.0},
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PropertyTestResult:
    """Result of testing a single property."""
    property_name: str
    expected: Any
    actual: Any
    passed: bool
    error_percent: Optional[float] = None


@dataclass
class HadronTestResult:
    """Result of testing a single hadron."""
    hadron_name: str
    quark_composition: List[str]
    particle_type: str  # baryon, meson, exotic
    property_results: List[PropertyTestResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.property_results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.property_results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.property_results if not r.passed)


@dataclass
class HadronCheckSummary:
    """Overall test summary."""
    total_hadrons: int = 0
    tested_hadrons: int = 0
    skipped_hadrons: int = 0
    total_properties: int = 0
    passed_properties: int = 0
    failed_properties: int = 0
    hadron_results: List[HadronTestResult] = field(default_factory=list)
    quark_data: Dict[str, Dict] = field(default_factory=dict)

    # Per-property statistics
    charge_tests: int = 0
    charge_passed: int = 0
    mass_tests: int = 0
    mass_passed: int = 0
    baryon_tests: int = 0
    baryon_passed: int = 0
    spin_tests: int = 0
    spin_passed: int = 0


# =============================================================================
# JSON Loading Utilities
# =============================================================================

def strip_json_comments(content: str) -> str:
    """Remove JavaScript-style // comments from JSON content."""
    lines = []
    for line in content.split('\n'):
        # Find // that's not inside a string
        result = []
        in_string = False
        i = 0
        while i < len(line):
            char = line[i]

            # Handle string boundaries
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            # Check for // comment start (outside string)
            elif not in_string and char == '/' and i + 1 < len(line) and line[i+1] == '/':
                # Found comment, stop processing this line
                break
            else:
                result.append(char)
            i += 1

        lines.append(''.join(result))

    return '\n'.join(lines)


def load_json_with_comments(filepath: Path) -> Optional[Dict]:
    """Load JSON file, handling JavaScript-style // comments."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            content = strip_json_comments(content)
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error in {filepath.name}: {e}")
        return None
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None


def load_all_quarks(data_dir: Path) -> Dict[str, Dict]:
    """Load all quark data files from quarks directory."""
    quarks = {}
    quark_dir = data_dir / "quarks"

    if not quark_dir.exists():
        print(f"Warning: Quark directory not found: {quark_dir}")
        return quarks

    # Load all JSON files in the directory
    for json_file in quark_dir.glob("*.json"):
        data = load_json_with_comments(json_file)
        if data:
            name = data.get('Name', json_file.stem)
            quarks[name] = data
            # Also index by symbol
            symbol = data.get('Symbol', '')
            if symbol:
                quarks[symbol] = data

    return quarks


def load_all_subatomic(data_dir: Path) -> Dict[str, Dict]:
    """Load all subatomic particle data files."""
    particles = {}
    subatomic_dir = data_dir / "subatomic"

    if not subatomic_dir.exists():
        print(f"Warning: Subatomic directory not found: {subatomic_dir}")
        return particles

    # Load all JSON files in the directory
    for json_file in subatomic_dir.glob("*.json"):
        data = load_json_with_comments(json_file)
        if data:
            name = data.get('Name', json_file.stem)
            particles[name] = data

    return particles


# =============================================================================
# Quark Composition Parsing
# =============================================================================

def parse_composition_to_quarks(composition: List[Dict], quark_data: Dict[str, Dict]) -> List[Dict]:
    """
    Parse the Composition field from a hadron JSON and return a list of quark dictionaries
    suitable for propagate_quark_to_hadron.
    """
    quarks = []

    for item in composition:
        constituent_name = item.get('Constituent', '')
        count = item.get('Count', 1)
        symbol = item.get('Symbol', '')
        is_anti = item.get('IsAnti', False)
        charge_e = item.get('Charge_e', None)

        # Determine the quark symbol
        quark_symbol = None

        # First, try the Symbol field
        if symbol:
            if symbol in SYMBOL_TO_STANDARD:
                quark_symbol = SYMBOL_TO_STANDARD[symbol]
            elif len(symbol) == 1 and symbol in 'udscbt':
                quark_symbol = symbol
                if is_anti:
                    quark_symbol += '\u0305'

        # Fall back to constituent name
        if not quark_symbol and constituent_name:
            if constituent_name in CONSTITUENT_TO_SYMBOL:
                quark_symbol = CONSTITUENT_TO_SYMBOL[constituent_name]
            else:
                # Try partial matching
                for name, sym in CONSTITUENT_TO_SYMBOL.items():
                    if name.lower() in constituent_name.lower() or constituent_name.lower() in name.lower():
                        quark_symbol = sym
                        break

        if quark_symbol:
            # Get quark properties
            if quark_symbol in QUARK_PROPERTIES:
                props = QUARK_PROPERTIES[quark_symbol].copy()
                # Override with actual charge from JSON if available
                if charge_e is not None:
                    props['Charge_e'] = charge_e
            else:
                # Unknown quark, use what we have from JSON
                props = {
                    'Charge_e': charge_e if charge_e is not None else 0,
                    'BaryonNumber_B': 1/3 if not is_anti else -1/3,
                    'Mass_MeVc2': 100.0,
                    'Isospin_I3': 0.0,
                }

            props['Symbol'] = quark_symbol
            props['Name'] = constituent_name

            # Add this quark 'count' times
            for _ in range(count):
                quarks.append(props.copy())

    return quarks


def determine_particle_type(num_quarks: int) -> str:
    """Determine particle type based on number of quarks."""
    if num_quarks == 3:
        return "baryon"
    elif num_quarks == 2:
        return "meson"
    else:
        return "exotic"


# =============================================================================
# Property Comparison
# =============================================================================

def compare_charge(expected: float, actual: float, tolerance: float = 0.01) -> PropertyTestResult:
    """Compare charge values."""
    passed = abs(expected - actual) < tolerance
    return PropertyTestResult(
        property_name="Charge",
        expected=expected,
        actual=actual,
        passed=passed
    )


def compare_mass(expected: float, actual: float, tolerance_percent: float = 15.0) -> PropertyTestResult:
    """Compare mass values with percentage tolerance."""
    if expected == 0:
        error_percent = 0 if actual == 0 else 100
    else:
        error_percent = abs(actual - expected) / expected * 100

    passed = error_percent < tolerance_percent
    return PropertyTestResult(
        property_name="Mass",
        expected=expected,
        actual=actual,
        passed=passed,
        error_percent=error_percent
    )


def compare_baryon_number(expected: float, actual: float, tolerance: float = 0.01) -> PropertyTestResult:
    """Compare baryon number values."""
    passed = abs(expected - actual) < tolerance
    return PropertyTestResult(
        property_name="BaryonNumber",
        expected=expected,
        actual=actual,
        passed=passed
    )


def compare_spin(expected: float, actual: float, tolerance: float = 0.01) -> PropertyTestResult:
    """Compare spin values."""
    # Note: propagate_quark_to_hadron doesn't calculate spin, so we skip this comparison
    # but include it for completeness
    passed = abs(expected - actual) < tolerance
    return PropertyTestResult(
        property_name="Spin",
        expected=expected,
        actual=actual,
        passed=passed
    )


# =============================================================================
# Main Test Function
# =============================================================================

def _build_and_test_hadron(hadron_data: Dict, quark_data: Dict[str, Dict]) -> Optional[HadronTestResult]:
    """
    Test building a hadron from its constituent quarks.

    Returns None if the hadron has no composition (fundamental particle).
    """
    name = hadron_data.get('Name', 'Unknown')
    composition = hadron_data.get('Composition', [])

    if not composition:
        return None  # Fundamental particle, no quark composition

    # Parse composition to quark list
    quarks = parse_composition_to_quarks(composition, quark_data)

    if not quarks:
        print(f"  Warning: Could not parse quarks for {name}")
        return None

    # Determine particle type
    particle_type = determine_particle_type(len(quarks))

    # Check if this is a Delta baryon (spin 3/2 excited state)
    # Delta baryons have same quark content as nucleons but different spin
    expected_spin = hadron_data.get('Spin_hbar', 0)
    spin_hint = None
    if 'Delta' in name:
        spin_hint = 1.5  # Delta baryons are spin 3/2

    # Build hadron using propagate_quark_to_hadron
    calculated_hadron = propagate_quark_to_hadron(quarks, spin_hint=spin_hint)

    # Create result object
    result = HadronTestResult(
        hadron_name=name,
        quark_composition=[q.get('Symbol', '?') for q in quarks],
        particle_type=particle_type
    )

    # Compare properties

    # 1. Charge
    expected_charge = hadron_data.get('Charge_e', 0)
    actual_charge = calculated_hadron.charge_e
    result.property_results.append(compare_charge(expected_charge, actual_charge))

    # 2. Mass
    expected_mass = hadron_data.get('Mass_MeVc2', 0)
    actual_mass = calculated_hadron.mass_MeV
    result.property_results.append(compare_mass(expected_mass, actual_mass))

    # 3. Baryon Number
    expected_baryon = hadron_data.get('BaryonNumber_B', 0)
    actual_baryon = calculated_hadron.baryon_number
    result.property_results.append(compare_baryon_number(expected_baryon, actual_baryon))

    # 4. Spin - Note: propagate_quark_to_hadron doesn't calculate spin currently
    # We'll check if spin_hbar is in the calculated hadron
    expected_spin = hadron_data.get('Spin_hbar', 0)
    actual_spin = calculated_hadron.spin_hbar
    # Only include spin test if we have both values
    if expected_spin is not None:
        result.property_results.append(compare_spin(expected_spin, actual_spin))

    return result


def run_all_tests(data_dir: Path) -> HadronCheckSummary:
    """Run all hadron tests and return summary."""
    summary = HadronCheckSummary()

    print("=" * 80)
    print("COMPREHENSIVE QUARK TO HADRON TEST SUITE")
    print("=" * 80)

    # Load quark data
    print("\n--- Loading Quark Data ---")
    quark_data = load_all_quarks(data_dir)
    summary.quark_data = quark_data
    print(f"Loaded {len(quark_data)} quark entries")

    # Count actual quark files (not including leptons/bosons)
    quark_files_found = list((data_dir / "quarks").glob("*.json"))
    print(f"Found {len(quark_files_found)} files in quarks directory:")
    for qf in sorted(quark_files_found):
        print(f"  - {qf.name}")

    # Load subatomic particles
    print("\n--- Loading Subatomic Particle Data ---")
    subatomic_data = load_all_subatomic(data_dir)
    summary.total_hadrons = len(subatomic_data)
    print(f"Loaded {len(subatomic_data)} subatomic particles")

    # List all files
    subatomic_files_found = list((data_dir / "subatomic").glob("*.json"))
    print(f"Found {len(subatomic_files_found)} files in subatomic directory:")
    for sf in sorted(subatomic_files_found):
        print(f"  - {sf.name}")

    # Test each hadron
    print("\n--- Testing Hadrons ---")
    print("-" * 80)

    for name, hadron_data in sorted(subatomic_data.items()):
        print(f"\nTesting: {name}")

        result = _build_and_test_hadron(hadron_data, quark_data)

        if result is None:
            summary.skipped_hadrons += 1
            print(f"  SKIPPED: No quark composition")
            continue

        summary.tested_hadrons += 1
        summary.hadron_results.append(result)

        # Print result details
        print(f"  Quark composition: {' '.join(result.quark_composition)}")
        print(f"  Particle type: {result.particle_type}")

        for prop_result in result.property_results:
            summary.total_properties += 1
            if prop_result.passed:
                summary.passed_properties += 1
                status = "PASS"
            else:
                summary.failed_properties += 1
                status = "FAIL"

            # Update per-property statistics
            if prop_result.property_name == "Charge":
                summary.charge_tests += 1
                if prop_result.passed:
                    summary.charge_passed += 1
            elif prop_result.property_name == "Mass":
                summary.mass_tests += 1
                if prop_result.passed:
                    summary.mass_passed += 1
            elif prop_result.property_name == "BaryonNumber":
                summary.baryon_tests += 1
                if prop_result.passed:
                    summary.baryon_passed += 1
            elif prop_result.property_name == "Spin":
                summary.spin_tests += 1
                if prop_result.passed:
                    summary.spin_passed += 1

            # Format output
            if prop_result.error_percent is not None:
                print(f"  {prop_result.property_name:15s}: Expected={prop_result.expected:10.4f}, "
                      f"Actual={prop_result.actual:10.4f}, Error={prop_result.error_percent:6.2f}% [{status}]")
            else:
                print(f"  {prop_result.property_name:15s}: Expected={prop_result.expected:10.4f}, "
                      f"Actual={prop_result.actual:10.4f} [{status}]")

    return summary


def print_report(summary: HadronCheckSummary):
    """Print comprehensive test report."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 80)

    # Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"  Total hadrons found:    {summary.total_hadrons}")
    print(f"  Hadrons tested:         {summary.tested_hadrons}")
    print(f"  Hadrons skipped:        {summary.skipped_hadrons}")
    print(f"  Total properties tested: {summary.total_properties}")
    print(f"  Properties passed:      {summary.passed_properties}")
    print(f"  Properties failed:      {summary.failed_properties}")

    if summary.total_properties > 0:
        overall_pass_rate = summary.passed_properties / summary.total_properties * 100
        print(f"  Overall pass rate:      {overall_pass_rate:.1f}%")

    # Per-property statistics
    print("\n--- Pass/Fail Rate by Property Type ---")

    if summary.charge_tests > 0:
        charge_rate = summary.charge_passed / summary.charge_tests * 100
        print(f"  Charge:      {summary.charge_passed:3d}/{summary.charge_tests:3d} passed ({charge_rate:5.1f}%)")

    if summary.mass_tests > 0:
        mass_rate = summary.mass_passed / summary.mass_tests * 100
        print(f"  Mass:        {summary.mass_passed:3d}/{summary.mass_tests:3d} passed ({mass_rate:5.1f}%)")

    if summary.baryon_tests > 0:
        baryon_rate = summary.baryon_passed / summary.baryon_tests * 100
        print(f"  BaryonNumber: {summary.baryon_passed:3d}/{summary.baryon_tests:3d} passed ({baryon_rate:5.1f}%)")

    if summary.spin_tests > 0:
        spin_rate = summary.spin_passed / summary.spin_tests * 100
        print(f"  Spin:        {summary.spin_passed:3d}/{summary.spin_tests:3d} passed ({spin_rate:5.1f}%)")

    # Group results by particle type
    print("\n--- Results by Particle Type ---")
    by_type = defaultdict(list)
    for result in summary.hadron_results:
        by_type[result.particle_type].append(result)

    for ptype, results in sorted(by_type.items()):
        total = len(results)
        passed = sum(1 for r in results if r.all_passed)
        print(f"  {ptype.capitalize()}: {passed}/{total} hadrons with all properties passing")

    # Failures detail
    failures = [r for r in summary.hadron_results if not r.all_passed]
    if failures:
        print("\n--- Failed Tests Detail ---")
        for result in failures:
            print(f"\n  {result.hadron_name} ({result.particle_type})")
            print(f"    Quarks: {' '.join(result.quark_composition)}")
            for prop in result.property_results:
                if not prop.passed:
                    if prop.error_percent is not None:
                        print(f"    FAILED {prop.property_name}: Expected={prop.expected:.4f}, "
                              f"Actual={prop.actual:.4f}, Error={prop.error_percent:.2f}%")
                    else:
                        print(f"    FAILED {prop.property_name}: Expected={prop.expected:.4f}, "
                              f"Actual={prop.actual:.4f}")

    # Mass error distribution
    print("\n--- Mass Error Distribution ---")
    mass_errors = []
    for result in summary.hadron_results:
        for prop in result.property_results:
            if prop.property_name == "Mass" and prop.error_percent is not None:
                mass_errors.append((result.hadron_name, prop.error_percent))

    if mass_errors:
        mass_errors.sort(key=lambda x: x[1], reverse=True)

        # Statistics
        errors_only = [e[1] for e in mass_errors]
        avg_error = sum(errors_only) / len(errors_only)
        min_error = min(errors_only)
        max_error = max(errors_only)

        print(f"  Average mass error: {avg_error:.2f}%")
        print(f"  Minimum mass error: {min_error:.2f}%")
        print(f"  Maximum mass error: {max_error:.2f}%")

        print("\n  Top 5 highest mass errors:")
        for name, error in mass_errors[:5]:
            print(f"    {name}: {error:.2f}%")

        print("\n  Top 5 lowest mass errors:")
        for name, error in mass_errors[-5:]:
            print(f"    {name}: {error:.2f}%")

    # Summary table
    print("\n--- Hadron Test Summary Table ---")
    print("-" * 95)
    print(f"{'Hadron':<20} {'Type':<8} {'Quarks':<15} {'Charge':<8} {'Mass':<8} {'Baryon':<8} {'Spin':<8} {'Status':<8}")
    print("-" * 95)

    for result in sorted(summary.hadron_results, key=lambda r: r.hadron_name):
        quark_str = ''.join(result.quark_composition)[:13]

        # Get individual property results
        charge_status = "-"
        mass_status = "-"
        baryon_status = "-"
        spin_status = "-"

        for prop in result.property_results:
            if prop.property_name == "Charge":
                charge_status = "OK" if prop.passed else "FAIL"
            elif prop.property_name == "Mass":
                mass_status = "OK" if prop.passed else "FAIL"
            elif prop.property_name == "BaryonNumber":
                baryon_status = "OK" if prop.passed else "FAIL"
            elif prop.property_name == "Spin":
                spin_status = "OK" if prop.passed else "FAIL"

        overall = "PASS" if result.all_passed else "FAIL"

        print(f"{result.hadron_name:<20} {result.particle_type:<8} {quark_str:<15} "
              f"{charge_status:<8} {mass_status:<8} {baryon_status:<8} {spin_status:<8} {overall:<8}")

    print("-" * 95)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total_hadrons_pass = sum(1 for r in summary.hadron_results if r.all_passed)
    total_hadrons_tested = len(summary.hadron_results)

    if total_hadrons_tested > 0:
        hadron_pass_rate = total_hadrons_pass / total_hadrons_tested * 100
        print(f"\n  Hadrons with ALL properties passing: {total_hadrons_pass}/{total_hadrons_tested} ({hadron_pass_rate:.1f}%)")

    if summary.total_properties > 0:
        overall_pass_rate = summary.passed_properties / summary.total_properties * 100
        print(f"  Overall property pass rate: {summary.passed_properties}/{summary.total_properties} ({overall_pass_rate:.1f}%)")

    # Assessment
    print("\n  Assessment:")
    if summary.charge_tests > 0 and summary.charge_passed == summary.charge_tests:
        print("    [OK] Charge conservation: 100% accuracy")
    else:
        print(f"    [!!] Charge conservation: {summary.charge_passed}/{summary.charge_tests} tests passed")

    if summary.baryon_tests > 0 and summary.baryon_passed == summary.baryon_tests:
        print("    [OK] Baryon number conservation: 100% accuracy")
    else:
        print(f"    [!!] Baryon number conservation: {summary.baryon_passed}/{summary.baryon_tests} tests passed")

    if mass_errors:
        avg_mass_error = sum(e[1] for e in mass_errors) / len(mass_errors)
        if avg_mass_error < 5:
            print(f"    [OK] Mass predictions: Excellent (avg error {avg_mass_error:.1f}%)")
        elif avg_mass_error < 10:
            print(f"    [OK] Mass predictions: Good (avg error {avg_mass_error:.1f}%)")
        elif avg_mass_error < 20:
            print(f"    [..] Mass predictions: Fair (avg error {avg_mass_error:.1f}%)")
        else:
            print(f"    [!!] Mass predictions: Need improvement (avg error {avg_mass_error:.1f}%)")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    # Determine data directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data" / "active"

    print(f"Data directory: {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    # Run all tests
    summary = run_all_tests(data_dir)

    # Print comprehensive report
    print_report(summary)

    # Return exit code based on results
    # Consider it a success if charge and baryon number are 100% correct
    if summary.charge_tests > 0 and summary.charge_passed == summary.charge_tests:
        if summary.baryon_tests > 0 and summary.baryon_passed == summary.baryon_tests:
            return 0

    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
