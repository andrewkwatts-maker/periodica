#!/usr/bin/env python3
"""
Comprehensive Test Suite for All 118 Elements
==============================================

This test validates ALL 118 elements in the periodic table by:
1. Loading element JSON files from data/elements/ (001_H.json through 118_Og.json)
2. Loading subatomic particle data for proton, neutron, electron
3. Building atoms using propagate_hadrons_to_atom from simulation_schema.py
4. Comparing calculated properties against JSON reference values
5. Grouping results by block, period, and category
6. Generating comprehensive reports with pass rates and analysis

Research/Testing only - does not modify any production code.
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path

from periodica.utils.simulation_schema import propagate_hadrons_to_atom


# =============================================================================
# Configuration and Constants
# =============================================================================

TOLERANCES = {
    'atomic_mass': 0.01,          # 1% tolerance
    'ionization_energy': 0.15,    # 15% tolerance
    'electronegativity': 0.20,    # 20% tolerance
    'atomic_radius': 0.20,        # 20% tolerance
    'nuclear_binding_energy': 0.10,  # 10% tolerance for binding energy
}

# Element categories based on chemical properties
ELEMENT_CATEGORIES = {
    'alkali_metals': [3, 11, 19, 37, 55, 87],  # Li, Na, K, Rb, Cs, Fr
    'alkaline_earth_metals': [4, 12, 20, 38, 56, 88],  # Be, Mg, Ca, Sr, Ba, Ra
    'transition_metals': list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)) + list(range(104, 113)),
    'post_transition_metals': [13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116],  # Al, Ga, In, Sn, Tl, Pb, Bi, Po, Nh, Fl, Mc, Lv
    'metalloids': [5, 14, 32, 33, 51, 52],  # B, Si, Ge, As, Sb, Te
    'reactive_nonmetals': [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53],  # H, C, N, O, F, P, S, Cl, Se, Br, I
    'noble_gases': [2, 10, 18, 36, 54, 86, 118],  # He, Ne, Ar, Kr, Xe, Rn, Og
    'lanthanides': list(range(57, 72)),  # La through Lu
    'actinides': list(range(89, 104)),  # Ac through Lr
}


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class PropertyResult:
    """Result of testing a single property."""
    property_name: str
    calculated_value: Optional[float]
    reference_value: Optional[float]
    passed: bool
    error_percent: Optional[float] = None
    tolerance_used: float = 0.0
    note: str = ""


@dataclass
class ElementResult:
    """Result of testing a single element."""
    atomic_number: int
    symbol: str
    name: str
    block: str
    period: int
    group: Optional[int]
    category: str
    neutron_count: int
    mass_number: int
    property_results: List[PropertyResult] = field(default_factory=list)
    all_passed: bool = True
    error_message: str = ""


@dataclass
class GroupedResults:
    """Results grouped by various criteria."""
    by_block: Dict[str, List[ElementResult]] = field(default_factory=lambda: defaultdict(list))
    by_period: Dict[int, List[ElementResult]] = field(default_factory=lambda: defaultdict(list))
    by_category: Dict[str, List[ElementResult]] = field(default_factory=lambda: defaultdict(list))


# =============================================================================
# JSON Loading with Comment Support
# =============================================================================

def strip_json_comments(json_str: str) -> str:
    """
    Remove JavaScript-style comments from JSON string.
    Handles both // line comments and /* block comments */.
    """
    # Remove single-line comments (// ...)
    # Be careful not to remove // inside strings
    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            escape_next = True
            result.append(char)
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if not in_string:
            # Check for // comment
            if i < len(json_str) - 1 and json_str[i:i+2] == '//':
                # Skip to end of line
                while i < len(json_str) and json_str[i] != '\n':
                    i += 1
                continue

            # Check for /* comment */
            if i < len(json_str) - 1 and json_str[i:i+2] == '/*':
                # Skip to */
                i += 2
                while i < len(json_str) - 1 and json_str[i:i+2] != '*/':
                    i += 1
                i += 2  # Skip */
                continue

        result.append(char)
        i += 1

    return ''.join(result)


def load_json_with_comments(filepath: str) -> Dict:
    """Load a JSON file that may contain JavaScript-style comments."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned = strip_json_comments(content)
    return json.loads(cleaned)


# =============================================================================
# Data Loading Functions
# =============================================================================

def find_elements_directory() -> str:
    """Find the elements data directory."""
    base_dir = Path(__file__).parent.parent

    # Check possible locations
    candidates = [
        base_dir / "src" / "periodica" / "data" / "active" / "elements",
        base_dir / "data" / "active" / "elements",
        base_dir / "data" / "elements",
    ]

    for candidate in candidates:
        if candidate.exists():
            # Count JSON files
            json_files = list(candidate.glob("*.json"))
            if len(json_files) >= 118:
                return str(candidate)

    # Default to data/elements
    return str(base_dir / "data" / "elements")


def find_subatomic_directory() -> str:
    """Find the subatomic particles data directory."""
    base_dir = Path(__file__).parent.parent

    candidates = [
        base_dir / "src" / "periodica" / "data" / "active" / "subatomic",
        base_dir / "data" / "active" / "subatomic",
        base_dir / "data" / "defaults" / "subatomic",
        base_dir / "data" / "subatomic",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def load_all_elements(elements_dir: str) -> Dict[int, Dict]:
    """Load all 118 element JSON files."""
    elements = {}

    for i in range(1, 119):
        # Try to find the file with pattern XXX_Sy.json
        pattern = f"{i:03d}_*.json"
        matches = list(Path(elements_dir).glob(pattern))

        if matches:
            filepath = matches[0]
            try:
                element_data = load_json_with_comments(str(filepath))
                elements[i] = element_data
            except Exception as e:
                print(f"Error loading element {i}: {e}")
        else:
            print(f"Warning: No file found for element {i}")

    return elements


def load_subatomic_particles(subatomic_dir: str) -> Dict[str, Dict]:
    """Load proton, neutron, and electron data."""
    particles = {}

    particle_files = ['Proton.json', 'Neutron.json']

    for filename in particle_files:
        filepath = Path(subatomic_dir) / filename
        if filepath.exists():
            try:
                particles[filename.replace('.json', '').lower()] = load_json_with_comments(str(filepath))
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return particles


def get_most_abundant_isotope(element_data: Dict) -> Tuple[int, int]:
    """
    Get the most abundant isotope's mass number and neutron count.
    Returns (mass_number, neutron_count).
    """
    isotopes = element_data.get('isotopes', [])
    Z = element_data.get('atomic_number', 0)

    if not isotopes:
        # Estimate from atomic mass
        A = round(element_data.get('atomic_mass', Z * 2))
        return (A, A - Z)

    # Find most abundant isotope
    most_abundant = max(isotopes, key=lambda x: x.get('abundance', 0))
    mass_number = most_abundant.get('mass_number', Z * 2)
    neutrons = most_abundant.get('neutrons', mass_number - Z)

    return (mass_number, neutrons)


def get_element_category(atomic_number: int) -> str:
    """Determine the category of an element based on its atomic number."""
    for category, elements in ELEMENT_CATEGORIES.items():
        if atomic_number in elements:
            return category
    return "unknown"


# =============================================================================
# Property Calculation and Comparison
# =============================================================================

def calculate_percent_error(calculated: float, reference: float) -> Optional[float]:
    """Calculate percentage error between calculated and reference values."""
    if calculated is None or reference is None:
        return None

    # Special case: both are zero (exact match)
    if reference == 0 and calculated == 0:
        return 0.0

    # Cannot calculate relative error if reference is 0
    if reference == 0:
        return None

    return abs(calculated - reference) / abs(reference)


def compare_property(
    property_name: str,
    calculated: Optional[float],
    reference: Optional[float],
    tolerance: float
) -> PropertyResult:
    """Compare a calculated property against its reference value."""
    result = PropertyResult(
        property_name=property_name,
        calculated_value=calculated,
        reference_value=reference,
        passed=True,
        tolerance_used=tolerance
    )

    # Handle missing values
    if reference is None:
        result.note = "No reference value"
        return result

    if calculated is None:
        result.passed = False
        result.note = "Calculation returned None"
        return result

    # Calculate error
    error = calculate_percent_error(calculated, reference)
    result.error_percent = error

    if error is not None and error <= tolerance:
        result.passed = True
    else:
        result.passed = False
        if error is not None:
            result.note = f"Error {error*100:.1f}% exceeds tolerance {tolerance*100:.0f}%"

    return result


def compare_electron_configuration(calculated: str, reference: str) -> PropertyResult:
    """Compare electron configurations (allow for notation variations)."""
    result = PropertyResult(
        property_name="electron_configuration",
        calculated_value=None,  # Not numeric
        reference_value=None,
        passed=True
    )

    if not reference:
        result.note = "No reference configuration"
        return result

    # Normalize configurations for comparison
    # Remove spaces and convert to lowercase for comparison
    calc_norm = calculated.replace(" ", "").lower() if calculated else ""
    ref_norm = reference.replace(" ", "").lower()

    # Handle superscript conversion (common in JSON files)
    superscript_map = {'¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
                       '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁰': '0',
                       '¹⁴': '14', '¹⁰': '10'}

    for sup, num in superscript_map.items():
        ref_norm = ref_norm.replace(sup, num)
        calc_norm = calc_norm.replace(sup, num)

    if calc_norm == ref_norm:
        result.passed = True
    else:
        # Check if they're "close" (e.g., noble gas notation vs full)
        result.passed = False
        result.note = f"Config mismatch: calculated vs reference"

    return result


# =============================================================================
# Main Test Function
# =============================================================================

def _test_single_element(
    element_data: Dict,
    proton_data: Dict,
    neutron_data: Dict
) -> ElementResult:
    """
    Test a single element by building it from hadrons and comparing properties.
    """
    Z = element_data.get('atomic_number', 0)
    symbol = element_data.get('symbol', '?')
    name = element_data.get('name', 'Unknown')
    block = element_data.get('block', '?')
    period = element_data.get('period', 0)
    group = element_data.get('group')

    # Get most abundant isotope
    mass_number, neutrons = get_most_abundant_isotope(element_data)

    result = ElementResult(
        atomic_number=Z,
        symbol=symbol,
        name=name,
        block=block,
        period=period,
        group=group,
        category=get_element_category(Z),
        neutron_count=neutrons,
        mass_number=mass_number
    )

    try:
        # Build atom using propagate_hadrons_to_atom
        atom = propagate_hadrons_to_atom(
            protons=Z,
            neutrons=neutrons,
            electrons=Z,  # Neutral atom
            proton_data=proton_data,
            neutron_data=neutron_data
        )

        # Compare atomic mass
        calc_mass = atom.atomic_mass_amu
        ref_mass = element_data.get('atomic_mass')
        mass_result = compare_property('atomic_mass', calc_mass, ref_mass, TOLERANCES['atomic_mass'])
        result.property_results.append(mass_result)

        # Compare binding energy (if we have reference)
        calc_binding = atom.nuclear_binding_energy_MeV
        # Reference binding energy might not be in element data, so we estimate
        # Using semi-empirical mass formula reference values
        ref_binding = calculate_reference_binding_energy(Z, neutrons)
        binding_result = compare_property('nuclear_binding_energy', calc_binding, ref_binding,
                                         TOLERANCES['nuclear_binding_energy'])
        result.property_results.append(binding_result)

        # Note: ionization_energy, electronegativity, atomic_radius are NOT calculated
        # by propagate_hadrons_to_atom (it focuses on nuclear properties)
        # We'll note these as "not calculated by this function" but still track reference values

        # Track ionization energy (reference only - not calculated by propagate_hadrons_to_atom)
        ref_ie = element_data.get('ionization_energy')
        ie_result = PropertyResult(
            property_name='ionization_energy',
            calculated_value=None,
            reference_value=ref_ie,
            passed=True,  # Pass by default since we can't calculate it
            note="Not calculated by propagate_hadrons_to_atom (quantum mechanics required)"
        )
        result.property_results.append(ie_result)

        # Track electronegativity (reference only)
        ref_en = element_data.get('electronegativity')
        en_result = PropertyResult(
            property_name='electronegativity',
            calculated_value=None,
            reference_value=ref_en,
            passed=True,
            note="Not calculated by propagate_hadrons_to_atom (requires electron density)"
        )
        result.property_results.append(en_result)

        # Track atomic radius (reference only)
        ref_radius = element_data.get('atomic_radius')
        radius_result = PropertyResult(
            property_name='atomic_radius',
            calculated_value=None,
            reference_value=ref_radius,
            passed=True,
            note="Not calculated by propagate_hadrons_to_atom (requires orbital calculations)"
        )
        result.property_results.append(radius_result)

        # Compare nuclear radius (can be calculated)
        calc_nuclear_radius = atom.nuclear_radius_fm
        ref_nuclear_radius = 1.2 * (mass_number ** (1/3))  # Standard formula
        nuclear_radius_result = compare_property('nuclear_radius', calc_nuclear_radius,
                                                 ref_nuclear_radius, 0.05)
        result.property_results.append(nuclear_radius_result)

        # Check if any core properties failed
        core_results = [r for r in result.property_results
                       if r.property_name in ['atomic_mass', 'nuclear_binding_energy', 'nuclear_radius']]
        result.all_passed = all(r.passed for r in core_results)

    except Exception as e:
        result.all_passed = False
        result.error_message = str(e)

    return result


# Tabulated experimental data for light nuclei (A <= 9)
# The Weizsacker semi-empirical mass formula doesn't work well for these
LIGHT_NUCLEI_BINDING_ENERGY = {
    (1, 0): 0.0,      # H-1 (protium) - single proton, no binding
    (1, 1): 2.22,     # H-2 (deuterium)
    (1, 2): 8.48,     # H-3 (tritium)
    (2, 1): 7.72,     # He-3
    (2, 2): 28.30,    # He-4 (alpha particle) - doubly magic
    (3, 3): 31.99,    # Li-6
    (3, 4): 39.24,    # Li-7
    (4, 4): 56.50,    # Be-8 (unstable)
    (4, 5): 58.16,    # Be-9
}


def calculate_reference_binding_energy(Z: int, N: int) -> float:
    """
    Calculate reference binding energy.
    Uses experimental data for light nuclei, semi-empirical formula for heavier ones.
    """
    A = Z + N
    if A == 0:
        return 0.0

    # Use tabulated values for light nuclei
    if (Z, N) in LIGHT_NUCLEI_BINDING_ENERGY:
        return LIGHT_NUCLEI_BINDING_ENERGY[(Z, N)]

    # Weizsacker coefficients (standard values)
    av = 15.75  # Volume
    as_ = 17.8  # Surface
    ac = 0.711  # Coulomb
    aa = 23.7   # Asymmetry
    ap = 11.2   # Pairing

    # Volume term
    volume = av * A

    # Surface term
    surface = -as_ * (A ** (2/3))

    # Coulomb term
    coulomb = -ac * Z * (Z - 1) / (A ** (1/3)) if A > 0 else 0

    # Asymmetry term
    asymmetry = -aa * ((N - Z) ** 2) / A

    # Pairing term
    if Z % 2 == 0 and N % 2 == 0:
        pairing = ap / (A ** 0.5)
    elif Z % 2 == 1 and N % 2 == 1:
        pairing = -ap / (A ** 0.5)
    else:
        pairing = 0

    return volume + surface + coulomb + asymmetry + pairing


# =============================================================================
# Results Analysis and Reporting
# =============================================================================

def group_results(results: List[ElementResult]) -> GroupedResults:
    """Group element results by block, period, and category."""
    grouped = GroupedResults()

    for result in results:
        grouped.by_block[result.block].append(result)
        grouped.by_period[result.period].append(result)
        grouped.by_category[result.category].append(result)

    return grouped


def calculate_pass_rate(results: List[ElementResult]) -> float:
    """Calculate pass rate for a list of element results."""
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.all_passed)
    return passed / len(results) * 100


def calculate_property_pass_rate(results: List[ElementResult], property_name: str) -> Tuple[float, int]:
    """Calculate pass rate for a specific property across all elements."""
    total = 0
    passed = 0

    for result in results:
        for prop_result in result.property_results:
            if prop_result.property_name == property_name:
                if prop_result.calculated_value is not None or prop_result.reference_value is not None:
                    total += 1
                    if prop_result.passed:
                        passed += 1
                break

    return (passed / total * 100 if total > 0 else 0.0, total)


def get_average_error(results: List[ElementResult], property_name: str) -> Optional[float]:
    """Get average error for a property across all elements."""
    errors = []

    for result in results:
        for prop_result in result.property_results:
            if prop_result.property_name == property_name and prop_result.error_percent is not None:
                errors.append(prop_result.error_percent)
                break

    return sum(errors) / len(errors) * 100 if errors else None


def identify_systematic_errors(results: List[ElementResult]) -> Dict[str, List[str]]:
    """Identify systematic errors in calculations."""
    errors = {
        'mass_consistently_high': [],
        'mass_consistently_low': [],
        'binding_consistently_high': [],
        'binding_consistently_low': [],
        'period_7_issues': [],
        'lanthanide_issues': [],
        'actinide_issues': [],
    }

    for result in results:
        for prop_result in result.property_results:
            if prop_result.property_name == 'atomic_mass' and prop_result.error_percent:
                calc = prop_result.calculated_value
                ref = prop_result.reference_value
                if calc and ref:
                    if calc > ref * 1.005:  # >0.5% high
                        errors['mass_consistently_high'].append(result.symbol)
                    elif calc < ref * 0.995:  # >0.5% low
                        errors['mass_consistently_low'].append(result.symbol)

            if prop_result.property_name == 'nuclear_binding_energy' and prop_result.error_percent:
                if prop_result.error_percent > 0.05:  # >5% error
                    if result.period == 7:
                        errors['period_7_issues'].append(result.symbol)
                    if result.category == 'lanthanides':
                        errors['lanthanide_issues'].append(result.symbol)
                    if result.category == 'actinides':
                        errors['actinide_issues'].append(result.symbol)

    return errors


def generate_report(results: List[ElementResult], grouped: GroupedResults) -> str:
    """Generate a comprehensive report of the test results."""
    report = []

    # Header
    report.append("=" * 100)
    report.append("COMPREHENSIVE TEST REPORT: ALL 118 ELEMENTS")
    report.append("=" * 100)
    report.append("")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.all_passed)
    failed = total - passed

    report.append("OVERALL SUMMARY")
    report.append("-" * 50)
    report.append(f"Total Elements Tested: {total}")
    report.append(f"Elements Passed: {passed}")
    report.append(f"Elements Failed: {failed}")
    report.append(f"Overall Pass Rate: {passed/total*100:.1f}%")
    report.append("")

    # Pass rate by property
    report.append("PASS RATE BY PROPERTY")
    report.append("-" * 50)
    properties = ['atomic_mass', 'nuclear_binding_energy', 'nuclear_radius']
    for prop in properties:
        rate, count = calculate_property_pass_rate(results, prop)
        avg_error = get_average_error(results, prop)
        error_str = f", Avg Error: {avg_error:.2f}%" if avg_error else ""
        report.append(f"  {prop:<25}: {rate:>6.1f}% ({count} tested){error_str}")
    report.append("")

    # Pass rate by block
    report.append("PASS RATE BY BLOCK")
    report.append("-" * 50)
    for block in ['s', 'p', 'd', 'f']:
        if block in grouped.by_block:
            elements = grouped.by_block[block]
            rate = calculate_pass_rate(elements)
            report.append(f"  {block}-block: {rate:>6.1f}% ({len(elements)} elements)")
    report.append("")

    # Pass rate by period
    report.append("PASS RATE BY PERIOD")
    report.append("-" * 50)
    for period in range(1, 8):
        if period in grouped.by_period:
            elements = grouped.by_period[period]
            rate = calculate_pass_rate(elements)
            report.append(f"  Period {period}: {rate:>6.1f}% ({len(elements)} elements)")
    report.append("")

    # Pass rate by category
    report.append("PASS RATE BY CATEGORY")
    report.append("-" * 50)
    for category in sorted(grouped.by_category.keys()):
        elements = grouped.by_category[category]
        rate = calculate_pass_rate(elements)
        report.append(f"  {category:<25}: {rate:>6.1f}% ({len(elements)} elements)")
    report.append("")

    # Failed elements detail
    failed_elements = [r for r in results if not r.all_passed]
    if failed_elements:
        report.append("ELEMENTS WITH FAILURES")
        report.append("-" * 50)
        for elem in failed_elements:
            report.append(f"\n  {elem.symbol} ({elem.name}, Z={elem.atomic_number})")
            report.append(f"    Block: {elem.block}, Period: {elem.period}, Category: {elem.category}")
            if elem.error_message:
                report.append(f"    Error: {elem.error_message}")
            for prop in elem.property_results:
                if not prop.passed and prop.calculated_value is not None:
                    error_str = f"{prop.error_percent*100:.2f}%" if prop.error_percent is not None else "N/A"
                    report.append(f"    - {prop.property_name}: calc={prop.calculated_value:.4f}, "
                                f"ref={prop.reference_value}, error={error_str}")
                    if prop.note:
                        report.append(f"      Note: {prop.note}")
    report.append("")

    # Systematic error analysis
    report.append("SYSTEMATIC ERROR ANALYSIS")
    report.append("-" * 50)
    systematic = identify_systematic_errors(results)
    for error_type, symbols in systematic.items():
        if symbols:
            report.append(f"  {error_type.replace('_', ' ').title()}: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                report.append(f"    ... and {len(symbols) - 10} more")
    report.append("")

    # Detailed property statistics
    report.append("DETAILED PROPERTY STATISTICS")
    report.append("-" * 50)

    for prop_name in ['atomic_mass', 'nuclear_binding_energy', 'nuclear_radius']:
        errors = []
        for result in results:
            for prop in result.property_results:
                if prop.property_name == prop_name and prop.error_percent is not None:
                    errors.append((result.symbol, prop.error_percent * 100))

        if errors:
            errors_sorted = sorted(errors, key=lambda x: x[1], reverse=True)
            avg_error = sum(e[1] for e in errors) / len(errors)
            max_error = errors_sorted[0]
            min_error = errors_sorted[-1]

            report.append(f"\n  {prop_name.upper()}")
            report.append(f"    Average Error: {avg_error:.3f}%")
            report.append(f"    Max Error: {max_error[1]:.3f}% ({max_error[0]})")
            report.append(f"    Min Error: {min_error[1]:.3f}% ({min_error[0]})")

            # Top 5 worst
            report.append(f"    Worst 5: {', '.join(f'{s}({e:.2f}%)' for s, e in errors_sorted[:5])}")

    report.append("")
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)

    return "\n".join(report)


def print_progress_bar(current: int, total: int, width: int = 50):
    """Print a progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = '=' * filled + '-' * (width - filled)
    print(f"\r[{bar}] {current}/{total} ({percent*100:.1f}%)", end='', flush=True)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_118_element_tests():
    """Run comprehensive tests on all 118 elements."""
    print("=" * 100)
    print("COMPREHENSIVE TEST SUITE: ALL 118 ELEMENTS")
    print("=" * 100)
    print()

    # Find data directories
    elements_dir = find_elements_directory()
    subatomic_dir = find_subatomic_directory()

    print(f"Elements directory: {elements_dir}")
    print(f"Subatomic directory: {subatomic_dir}")
    print()

    # Load data
    print("Loading element data...")
    elements = load_all_elements(elements_dir)
    print(f"  Loaded {len(elements)} elements")

    if len(elements) != 118:
        print(f"  WARNING: Expected 118 elements, got {len(elements)}")

    print("Loading subatomic particle data...")
    particles = load_subatomic_particles(subatomic_dir)
    print(f"  Loaded particles: {list(particles.keys())}")

    proton_data = particles.get('proton', {})
    neutron_data = particles.get('neutron', {})

    print()
    print("Testing all elements...")
    print()

    # Run tests
    results: List[ElementResult] = []

    for i, (Z, element_data) in enumerate(sorted(elements.items())):
        print_progress_bar(i + 1, len(elements))
        result = _test_single_element(element_data, proton_data, neutron_data)
        results.append(result)

    print()  # New line after progress bar
    print()

    # Group results
    grouped = group_results(results)

    # Generate and print report
    report = generate_report(results, grouped)
    print(report)

    # Print individual element results table
    print("\n")
    print("INDIVIDUAL ELEMENT RESULTS")
    print("=" * 100)
    print(f"{'Z':>3} {'Sym':>4} {'Name':<15} {'Block':>5} {'Per':>3} {'Mass Err%':>10} {'Bind Err%':>10} {'Status':>8}")
    print("-" * 100)

    for result in results:
        mass_err = None
        bind_err = None

        for prop in result.property_results:
            if prop.property_name == 'atomic_mass' and prop.error_percent is not None:
                mass_err = prop.error_percent * 100
            if prop.property_name == 'nuclear_binding_energy' and prop.error_percent is not None:
                bind_err = prop.error_percent * 100

        mass_str = f"{mass_err:.3f}" if mass_err is not None else "N/A"
        bind_str = f"{bind_err:.3f}" if bind_err is not None else "N/A"
        status = "PASS" if result.all_passed else "FAIL"

        print(f"{result.atomic_number:>3} {result.symbol:>4} {result.name:<15} {result.block:>5} "
              f"{result.period:>3} {mass_str:>10} {bind_str:>10} {status:>8}")

    print()
    print("Test complete!")

    # Return summary for programmatic access
    return {
        'total': len(results),
        'passed': sum(1 for r in results if r.all_passed),
        'failed': sum(1 for r in results if not r.all_passed),
        'pass_rate': sum(1 for r in results if r.all_passed) / len(results) * 100,
        'results': results,
        'grouped': grouped,
    }


# =============================================================================
# Unit Test Class (for pytest/unittest compatibility)
# =============================================================================

import unittest


class TestAll118Elements(unittest.TestCase):
    """Unit test class for all 118 elements."""

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        cls.elements_dir = find_elements_directory()
        cls.subatomic_dir = find_subatomic_directory()
        cls.elements = load_all_elements(cls.elements_dir)
        cls.particles = load_subatomic_particles(cls.subatomic_dir)
        cls.proton_data = cls.particles.get('proton', {})
        cls.neutron_data = cls.particles.get('neutron', {})

    def test_element_count(self):
        """Test that we have all 118 elements."""
        self.assertEqual(len(self.elements), 118, "Should have 118 elements")

    def test_all_elements_load(self):
        """Test that all elements can be loaded."""
        for Z in range(1, 119):
            self.assertIn(Z, self.elements, f"Element {Z} should be loaded")

    def test_all_elements_have_required_fields(self):
        """Test that all elements have required fields."""
        required_fields = ['symbol', 'name', 'atomic_number', 'atomic_mass',
                          'block', 'period', 'isotopes']

        for Z, element in self.elements.items():
            for field in required_fields:
                self.assertIn(field, element,
                            f"Element {Z} ({element.get('symbol', '?')}) missing {field}")

    def test_atomic_mass_accuracy(self):
        """Test atomic mass calculation accuracy for all elements."""
        failed = []

        for Z, element_data in sorted(self.elements.items()):
            result = _test_single_element(element_data, self.proton_data, self.neutron_data)

            for prop in result.property_results:
                if prop.property_name == 'atomic_mass':
                    if prop.error_percent is not None and prop.error_percent > TOLERANCES['atomic_mass']:
                        failed.append((result.symbol, prop.error_percent * 100))

        # Report failures
        if failed:
            fail_msg = "\n".join(f"  {sym}: {err:.2f}% error" for sym, err in failed[:10])
            if len(failed) > 10:
                fail_msg += f"\n  ... and {len(failed) - 10} more"
            self.fail(f"Atomic mass errors exceed {TOLERANCES['atomic_mass']*100}% tolerance:\n{fail_msg}")

    def test_binding_energy_accuracy(self):
        """Test nuclear binding energy calculation accuracy."""
        failed = []

        for Z, element_data in sorted(self.elements.items()):
            result = _test_single_element(element_data, self.proton_data, self.neutron_data)

            for prop in result.property_results:
                if prop.property_name == 'nuclear_binding_energy':
                    if prop.error_percent is not None and prop.error_percent > TOLERANCES['nuclear_binding_energy']:
                        failed.append((result.symbol, prop.error_percent * 100))

        if failed:
            fail_msg = "\n".join(f"  {sym}: {err:.2f}% error" for sym, err in failed[:10])
            if len(failed) > 10:
                fail_msg += f"\n  ... and {len(failed) - 10} more"
            self.fail(f"Binding energy errors exceed {TOLERANCES['nuclear_binding_energy']*100}% tolerance:\n{fail_msg}")

    def test_blocks_distribution(self):
        """Test that elements are correctly distributed across blocks."""
        block_counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0}

        for Z, element in self.elements.items():
            block = element.get('block', '?')
            if block in block_counts:
                block_counts[block] += 1

        # Expected counts (approximately)
        # s-block: 14 (groups 1, 2)
        # p-block: 36 (groups 13-18)
        # d-block: 40 (groups 3-12)
        # f-block: 28 (lanthanides + actinides)

        self.assertGreater(block_counts['s'], 10, "Should have s-block elements")
        self.assertGreater(block_counts['p'], 30, "Should have p-block elements")
        self.assertGreater(block_counts['d'], 35, "Should have d-block elements")
        self.assertGreater(block_counts['f'], 25, "Should have f-block elements")

    def test_periods_distribution(self):
        """Test that elements are correctly distributed across periods."""
        period_counts = defaultdict(int)

        for Z, element in self.elements.items():
            period = element.get('period', 0)
            period_counts[period] += 1

        # Expected: Period 1: 2, Period 2: 8, Period 3: 8, Period 4: 18,
        #           Period 5: 18, Period 6: 32, Period 7: 32

        self.assertEqual(period_counts[1], 2, "Period 1 should have 2 elements")
        self.assertEqual(period_counts[2], 8, "Period 2 should have 8 elements")
        self.assertEqual(period_counts[3], 8, "Period 3 should have 8 elements")
        self.assertEqual(period_counts[4], 18, "Period 4 should have 18 elements")
        self.assertEqual(period_counts[5], 18, "Period 5 should have 18 elements")
        self.assertEqual(period_counts[6], 32, "Period 6 should have 32 elements")
        self.assertEqual(period_counts[7], 32, "Period 7 should have 32 elements")

    def test_light_elements_accuracy(self):
        """Test accuracy for light elements (Z=1-20)."""
        max_mass_error = 0
        max_binding_error = 0

        for Z in range(1, 21):
            if Z not in self.elements:
                continue

            result = _test_single_element(self.elements[Z], self.proton_data, self.neutron_data)

            for prop in result.property_results:
                if prop.property_name == 'atomic_mass' and prop.error_percent:
                    max_mass_error = max(max_mass_error, prop.error_percent)
                if prop.property_name == 'nuclear_binding_energy' and prop.error_percent:
                    max_binding_error = max(max_binding_error, prop.error_percent)

        self.assertLess(max_mass_error, 0.02,
                       f"Light element mass error {max_mass_error*100:.2f}% exceeds 2%")

    def test_transition_metals_accuracy(self):
        """Test accuracy for transition metals."""
        transition_metals = ELEMENT_CATEGORIES['transition_metals']
        passed = 0
        total = 0

        for Z in transition_metals:
            if Z not in self.elements:
                continue

            total += 1
            result = _test_single_element(self.elements[Z], self.proton_data, self.neutron_data)

            if result.all_passed:
                passed += 1

        pass_rate = passed / total * 100 if total > 0 else 0
        self.assertGreater(pass_rate, 90,
                          f"Transition metal pass rate {pass_rate:.1f}% below 90%")

    def test_lanthanides_and_actinides(self):
        """Test accuracy for lanthanides and actinides (f-block)."""
        f_block = ELEMENT_CATEGORIES['lanthanides'] + ELEMENT_CATEGORIES['actinides']
        passed = 0
        total = 0

        for Z in f_block:
            if Z not in self.elements:
                continue

            total += 1
            result = _test_single_element(self.elements[Z], self.proton_data, self.neutron_data)

            if result.all_passed:
                passed += 1

        pass_rate = passed / total * 100 if total > 0 else 0
        self.assertGreater(pass_rate, 85,
                          f"f-block pass rate {pass_rate:.1f}% below 85%")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test all 118 elements')
    parser.add_argument('--unittest', action='store_true',
                       help='Run as unittest (for CI integration)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    if args.unittest:
        # Run as unittest
        unittest.main(argv=[''], verbosity=2 if args.verbose else 1)
    else:
        # Run comprehensive report
        summary = run_all_118_element_tests()

        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        print(f"Total: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
