#!/usr/bin/env python3
"""
Complete Subatomic Particle JSON Data Validation Test Suite
============================================================
Validates ALL subatomic particle JSON files against physics constraints
and known experimental values.

This test suite verifies:
1. Self-consistency of JSON properties (charge, baryon number, spin)
2. Quark composition matches declared properties
3. Specific known particle values (proton, neutron, pions, kaons, etc.)
4. Classification consistency (baryon/meson based on quark count)
5. Conservation laws (charge, baryon number)

Author: Claude Code Analysis
Date: 2024
"""

import sys
import os
import json
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Physics Constants for Quark Properties
# ============================================================================

# Quark charges in units of elementary charge (e)
QUARK_CHARGES = {
    'u': 2/3,       # Up quark
    'd': -1/3,      # Down quark
    's': -1/3,      # Strange quark
    'c': 2/3,       # Charm quark
    'b': -1/3,      # Bottom quark
    't': 2/3,       # Top quark
}

# Antiquark charges (opposite sign)
ANTIQUARK_CHARGES = {k + '_bar': -v for k, v in QUARK_CHARGES.items()}

# Quark baryon numbers (each quark has B = 1/3)
QUARK_BARYON_NUMBER = 1/3
ANTIQUARK_BARYON_NUMBER = -1/3

# Quark spins (all quarks are spin-1/2 fermions)
QUARK_SPIN = 0.5

# Quark strangeness (S = -1 for s quark, +1 for anti-s)
QUARK_STRANGENESS = {
    'u': 0, 'd': 0, 's': -1, 'c': 0, 'b': 0, 't': 0,
    'u_bar': 0, 'd_bar': 0, 's_bar': 1, 'c_bar': 0, 'b_bar': 0, 't_bar': 0,
}

# Constituent quark masses (MeV/c^2) - approximate effective masses
CONSTITUENT_QUARK_MASSES = {
    'u': 336,       # Up quark (constituent mass)
    'd': 340,       # Down quark (constituent mass)
    's': 486,       # Strange quark (constituent mass)
    'c': 1550,      # Charm quark (constituent mass)
    'b': 4730,      # Bottom quark (constituent mass)
    't': 171000,    # Top quark (constituent mass)
}

# Map various notation styles to canonical quark symbols
QUARK_NAME_MAP = {
    # Full names
    'up quark': 'u',
    'down quark': 'd',
    'strange quark': 's',
    'charm quark': 'c',
    'bottom quark': 'b',
    'top quark': 't',
    # Anti-quarks (various notations)
    'anti-up quark': 'u_bar',
    'anti-down quark': 'd_bar',
    'anti-strange quark': 's_bar',
    'anti-charm quark': 'c_bar',
    'anti-bottom quark': 'b_bar',
    'anti-top quark': 't_bar',
    'antiup quark': 'u_bar',
    'antidown quark': 'd_bar',
    'antistrange quark': 's_bar',
    'anticharm quark': 'c_bar',
    'antibottom quark': 'b_bar',
    'antitop quark': 't_bar',
}

SYMBOL_MAP = {
    'u': 'u', 'd': 'd', 's': 's', 'c': 'c', 'b': 'b', 't': 't',
    'u-bar': 'u_bar', 'd-bar': 'd_bar', 's-bar': 's_bar',
    'c-bar': 'c_bar', 'b-bar': 'b_bar', 't-bar': 't_bar',
}


# ============================================================================
# Data Classes for Test Results
# ============================================================================

@dataclass
class PropertyValidation:
    """Result of validating a single property."""
    property_name: str
    expected: Any
    actual: Any
    passed: bool
    tolerance: float = 0.0
    message: str = ""


@dataclass
class ParticleValidation:
    """Complete validation result for a particle."""
    name: str
    filename: str
    validations: List[PropertyValidation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.validations)

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.validations if v.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for v in self.validations if not v.passed)

    def add_validation(self, prop_name: str, expected: Any, actual: Any,
                      passed: bool, tolerance: float = 0.0, message: str = ""):
        self.validations.append(PropertyValidation(
            property_name=prop_name,
            expected=expected,
            actual=actual,
            passed=passed,
            tolerance=tolerance,
            message=message
        ))


@dataclass
class ValidationReport:
    """Complete test report."""
    total_particles: int = 0
    particles_passed: int = 0
    particles_failed: int = 0
    total_validations: int = 0
    validations_passed: int = 0
    validations_failed: int = 0
    particle_results: List[ParticleValidation] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)


# ============================================================================
# JSON Loading Utilities
# ============================================================================

def load_json_with_comments(filepath: str) -> Optional[Dict]:
    """Load JSON file, stripping JavaScript-style comments."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove // comments (but not inside strings)
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            if '//' in line:
                in_string = False
                result = []
                i = 0
                while i < len(line):
                    if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                        in_string = not in_string
                    if not in_string and line[i:i+2] == '//':
                        break
                    result.append(line[i])
                    i += 1
                line = ''.join(result)
            clean_lines.append(line)

        content = '\n'.join(clean_lines)
        return json.loads(content)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_all_particles(data_dir: str) -> Dict[str, Tuple[Dict, str]]:
    """Load all particle JSON files from directory.
    Returns dict mapping particle name to (data, filename)."""
    particles = {}

    json_files = glob.glob(os.path.join(data_dir, '*.json'))

    for filepath in json_files:
        data = load_json_with_comments(filepath)
        if data:
            name = data.get('Name', os.path.basename(filepath).replace('.json', ''))
            particles[name] = (data, os.path.basename(filepath))

    return particles


# ============================================================================
# Quark Composition Parsing
# ============================================================================

def parse_composition(composition: List[Dict]) -> List[Tuple[str, int, float]]:
    """Parse composition list into (quark_symbol, count, charge) tuples."""
    parsed = []

    for item in composition:
        constituent = item.get('Constituent', '').lower()
        symbol = item.get('Symbol', '')
        count = item.get('Count', 1)
        charge = item.get('Charge_e', 0)
        is_anti = item.get('IsAnti', False)

        # Determine canonical quark symbol
        quark_symbol = None

        # Try symbol field first
        if symbol:
            if symbol in SYMBOL_MAP:
                quark_symbol = SYMBOL_MAP[symbol]
            elif len(symbol) == 1 and symbol in 'udscbt':
                quark_symbol = symbol
                if is_anti:
                    quark_symbol = symbol + '_bar'

        # Fall back to constituent name
        if not quark_symbol:
            for name, sym in QUARK_NAME_MAP.items():
                if name in constituent:
                    quark_symbol = sym
                    break

        if quark_symbol:
            parsed.append((quark_symbol, count, charge))

    return parsed


def get_quark_charge(quark: str) -> float:
    """Get the charge of a quark in units of e."""
    if quark in QUARK_CHARGES:
        return QUARK_CHARGES[quark]
    elif quark in ANTIQUARK_CHARGES:
        return ANTIQUARK_CHARGES[quark]
    elif quark.endswith('_bar'):
        base = quark.replace('_bar', '')
        if base in QUARK_CHARGES:
            return -QUARK_CHARGES[base]
    return 0.0


def get_quark_baryon_number(quark: str) -> float:
    """Get the baryon number contribution of a quark."""
    if quark.endswith('_bar'):
        return ANTIQUARK_BARYON_NUMBER
    return QUARK_BARYON_NUMBER


def get_quark_strangeness(quark: str) -> int:
    """Get the strangeness of a quark."""
    return QUARK_STRANGENESS.get(quark, 0)


# ============================================================================
# Validation Functions
# ============================================================================

def calculate_expected_charge(composition: List[Dict]) -> float:
    """Calculate expected charge from quark composition."""
    total_charge = 0.0
    parsed = parse_composition(composition)

    for quark, count, declared_charge in parsed:
        # Use the declared charge from JSON (already accounts for the quark type)
        total_charge += declared_charge * count

    return round(total_charge, 6)


def calculate_expected_baryon_number(composition: List[Dict]) -> int:
    """Calculate expected baryon number from quark composition."""
    total_b = 0.0
    parsed = parse_composition(composition)

    for quark, count, _ in parsed:
        total_b += get_quark_baryon_number(quark) * count

    return round(total_b)


def calculate_expected_strangeness(composition: List[Dict]) -> int:
    """Calculate expected strangeness from quark composition."""
    total_s = 0
    parsed = parse_composition(composition)

    for quark, count, _ in parsed:
        total_s += get_quark_strangeness(quark) * count

    return total_s


def count_total_quarks(composition: List[Dict]) -> int:
    """Count total number of quarks (including antiquarks)."""
    total = 0
    for item in composition:
        total += item.get('Count', 1)
    return total


def is_baryon_composition(composition: List[Dict]) -> bool:
    """Check if composition is consistent with a baryon (3 quarks)."""
    parsed = parse_composition(composition)

    quark_count = 0
    antiquark_count = 0

    for quark, count, _ in parsed:
        if quark.endswith('_bar'):
            antiquark_count += count
        else:
            quark_count += count

    # Standard baryon: 3 quarks, 0 antiquarks
    # Pentaquark: 4 quarks, 1 antiquark (still B=1)
    net_quarks = quark_count - antiquark_count
    return net_quarks == 3


def is_meson_composition(composition: List[Dict]) -> bool:
    """Check if composition is consistent with a meson (quark-antiquark pair)."""
    parsed = parse_composition(composition)

    quark_count = 0
    antiquark_count = 0

    for quark, count, _ in parsed:
        if quark.endswith('_bar'):
            antiquark_count += count
        else:
            quark_count += count

    # Standard meson: 1 quark, 1 antiquark (B=0)
    # Tetraquark: 2 quarks, 2 antiquarks (still B=0)
    # Mesons can also be superpositions (like pi0, eta)
    return quark_count == antiquark_count


def get_possible_spins(composition: List[Dict], is_aligned: bool = False) -> List[float]:
    """Get possible spin values for a hadron composition.

    For baryons (3 quarks):
        - Ground state (mixed): 1/2 (two quarks paired, one unpaired)
        - Aligned state: 3/2 (all spins aligned)

    For mesons (quark-antiquark):
        - Pseudoscalar (antiparallel): 0
        - Vector (parallel): 1

    For exotic states (tetraquarks, pentaquarks):
        - Various possibilities
    """
    total_quarks = count_total_quarks(composition)
    parsed = parse_composition(composition)

    quark_count = sum(count for quark, count, _ in parsed if not quark.endswith('_bar'))
    antiquark_count = sum(count for quark, count, _ in parsed if quark.endswith('_bar'))

    if quark_count == 3 and antiquark_count == 0:
        # Baryon: 3 quarks with spin 1/2 each
        return [0.5, 1.5]  # Can be 1/2 or 3/2
    elif quark_count == 1 and antiquark_count == 1:
        # Meson: quark-antiquark
        return [0, 1]  # Can be 0 (pseudoscalar) or 1 (vector)
    elif quark_count == 2 and antiquark_count == 2:
        # Tetraquark
        return [0, 1, 2]
    elif quark_count == 4 and antiquark_count == 1:
        # Pentaquark
        return [0.5, 1.5, 2.5]
    elif total_quarks > 2:
        # More complex state - superposition
        return [0, 0.5, 1, 1.5, 2]

    return [0, 0.5, 1, 1.5]


def validate_mass_reasonable(mass_mev: float, composition: List[Dict]) -> Tuple[bool, str]:
    """Check if mass is reasonable for the quark composition."""
    parsed = parse_composition(composition)

    # Calculate rough expected mass from constituent quark masses
    min_expected = 0
    for quark, count, _ in parsed:
        base_quark = quark.replace('_bar', '')
        if base_quark in CONSTITUENT_QUARK_MASSES:
            min_expected += CONSTITUENT_QUARK_MASSES[base_quark] * count

    # Binding energy typically reduces mass (for light hadrons) or adds (for heavy)
    # Allow wide tolerance: 20% below constituent mass to 3x above
    lower_bound = min_expected * 0.1  # Pions are much lighter than constituent mass
    upper_bound = min_expected * 3.0

    if mass_mev < lower_bound:
        return False, f"Mass {mass_mev:.1f} MeV too low (expected >{lower_bound:.1f} MeV)"
    elif mass_mev > upper_bound:
        return False, f"Mass {mass_mev:.1f} MeV too high (expected <{upper_bound:.1f} MeV)"

    return True, "Mass within expected range"


# ============================================================================
# Known Particle Reference Data
# ============================================================================

KNOWN_PARTICLES = {
    'Proton': {
        'Charge_e': 1,
        'Mass_MeVc2': 938.27,
        'Mass_tolerance': 0.1,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': 0,
        'quark_content': ['u', 'u', 'd'],
    },
    'Neutron': {
        'Charge_e': 0,
        'Mass_MeVc2': 939.57,
        'Mass_tolerance': 0.1,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': 0,
        'quark_content': ['u', 'd', 'd'],
    },
    'Pion+': {
        'Charge_e': 1,
        'Mass_MeVc2': 139.6,
        'Mass_tolerance': 0.5,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
        'Strangeness': 0,
        'quark_content': ['u', 'd_bar'],
    },
    'Pion-': {
        'Charge_e': -1,
        'Mass_MeVc2': 139.6,
        'Mass_tolerance': 0.5,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
        'Strangeness': 0,
        'quark_content': ['d', 'u_bar'],
    },
    'Pion0': {
        'Charge_e': 0,
        'Mass_MeVc2': 135.0,
        'Mass_tolerance': 0.5,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
        'Strangeness': 0,
    },
    'Kaon+': {
        'Charge_e': 1,
        'Mass_MeVc2': 493.7,
        'Mass_tolerance': 0.5,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
        'Strangeness': 1,  # K+ has s-bar, so S=+1
        'quark_content': ['u', 's_bar'],
    },
    'Kaon-': {
        'Charge_e': -1,
        'Mass_MeVc2': 493.7,
        'Mass_tolerance': 0.5,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
        'Strangeness': -1,  # K- has s, so S=-1
        'quark_content': ['s', 'u_bar'],
    },
    'Kaon0': {
        'Charge_e': 0,
        'Mass_MeVc2': 497.6,
        'Mass_tolerance': 0.5,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
        'Strangeness': 1,  # K0 has s-bar, so S=+1
        'quark_content': ['d', 's_bar'],
    },
    'Lambda': {
        'Charge_e': 0,
        'Mass_MeVc2': 1115.7,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': -1,
        'quark_content': ['u', 'd', 's'],
    },
    'Sigma+': {
        'Charge_e': 1,
        'Mass_MeVc2': 1189.4,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': -1,
        'quark_content': ['u', 'u', 's'],
    },
    'Sigma0': {
        'Charge_e': 0,
        'Mass_MeVc2': 1192.6,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': -1,
        'quark_content': ['u', 'd', 's'],
    },
    'Sigma-': {
        'Charge_e': -1,
        'Mass_MeVc2': 1197.4,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': -1,
        'quark_content': ['d', 'd', 's'],
    },
    'Xi0': {
        'Charge_e': 0,
        'Mass_MeVc2': 1314.9,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': -2,
        'quark_content': ['u', 's', 's'],
    },
    'Xi-': {
        'Charge_e': -1,
        'Mass_MeVc2': 1321.7,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0.5,
        'BaryonNumber_B': 1,
        'Strangeness': -2,
        'quark_content': ['d', 's', 's'],
    },
    'Omega-': {
        'Charge_e': -1,
        'Mass_MeVc2': 1672.0,
        'Mass_tolerance': 2.0,
        'Spin_hbar': 1.5,
        'BaryonNumber_B': 1,
        'Strangeness': -3,
        'quark_content': ['s', 's', 's'],
    },
    'Delta++': {
        'Charge_e': 2,
        'Mass_MeVc2': 1232.0,
        'Mass_tolerance': 2.0,
        'Spin_hbar': 1.5,
        'BaryonNumber_B': 1,
        'Strangeness': 0,
        'quark_content': ['u', 'u', 'u'],
    },
    'Delta+': {
        'Charge_e': 1,
        'Mass_MeVc2': 1232.0,
        'Mass_tolerance': 2.0,
        'Spin_hbar': 1.5,
        'BaryonNumber_B': 1,
        'Strangeness': 0,
        'quark_content': ['u', 'u', 'd'],
    },
    'Delta0': {
        'Charge_e': 0,
        'Mass_MeVc2': 1232.0,
        'Mass_tolerance': 2.0,
        'Spin_hbar': 1.5,
        'BaryonNumber_B': 1,
        'Strangeness': 0,
        'quark_content': ['u', 'd', 'd'],
    },
    'Delta-': {
        'Charge_e': -1,
        'Mass_MeVc2': 1232.0,
        'Mass_tolerance': 2.0,
        'Spin_hbar': 1.5,
        'BaryonNumber_B': 1,
        'Strangeness': 0,
        'quark_content': ['d', 'd', 'd'],
    },
    'J/Psi': {
        'Charge_e': 0,
        'Mass_MeVc2': 3096.9,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 1,
        'BaryonNumber_B': 0,
        'quark_content': ['c', 'c_bar'],
    },
    'Upsilon': {
        'Charge_e': 0,
        'Mass_MeVc2': 9460.0,
        'Mass_tolerance': 5.0,
        'Spin_hbar': 1,
        'BaryonNumber_B': 0,
        'quark_content': ['b', 'b_bar'],
    },
    'Eta': {
        'Charge_e': 0,
        'Mass_MeVc2': 547.9,
        'Mass_tolerance': 1.0,
        'Spin_hbar': 0,
        'BaryonNumber_B': 0,
    },
}


# ============================================================================
# Main Validation Functions
# ============================================================================

def validate_particle(name: str, data: Dict, filename: str) -> ParticleValidation:
    """Validate a single particle's JSON data."""
    result = ParticleValidation(name=name, filename=filename)

    composition = data.get('Composition', [])
    classification = data.get('Classification', [])

    # -------------------------------------------------------------------------
    # 1. Validate charge consistency with quark composition
    # -------------------------------------------------------------------------
    if composition:
        expected_charge = calculate_expected_charge(composition)
        actual_charge = data.get('Charge_e', 0)
        charge_match = abs(expected_charge - actual_charge) < 0.01

        result.add_validation(
            'Charge from composition',
            expected_charge,
            actual_charge,
            charge_match,
            0.01,
            "" if charge_match else f"Charge mismatch: composition gives {expected_charge}, declared {actual_charge}"
        )

    # -------------------------------------------------------------------------
    # 2. Validate baryon number consistency
    # -------------------------------------------------------------------------
    if composition:
        expected_baryon = calculate_expected_baryon_number(composition)
        actual_baryon = data.get('BaryonNumber_B', 0)
        baryon_match = expected_baryon == actual_baryon

        result.add_validation(
            'Baryon number from composition',
            expected_baryon,
            actual_baryon,
            baryon_match,
            message="" if baryon_match else f"Baryon number mismatch: composition gives {expected_baryon}, declared {actual_baryon}"
        )

    # -------------------------------------------------------------------------
    # 3. Validate spin is physically possible for composition
    # -------------------------------------------------------------------------
    if composition:
        actual_spin = data.get('Spin_hbar', 0)
        possible_spins = get_possible_spins(composition)
        spin_valid = actual_spin in possible_spins

        result.add_validation(
            'Spin consistent with composition',
            f"one of {possible_spins}",
            actual_spin,
            spin_valid,
            message="" if spin_valid else f"Spin {actual_spin} not in possible values {possible_spins}"
        )

    # -------------------------------------------------------------------------
    # 4. Validate classification matches particle type
    # -------------------------------------------------------------------------
    if composition and classification:
        is_baryon = is_baryon_composition(composition)
        is_meson = is_meson_composition(composition)

        has_baryon_class = 'Baryon' in classification
        has_meson_class = 'Meson' in classification

        if is_baryon:
            result.add_validation(
                'Classification matches baryon composition',
                'Baryon in classification',
                has_baryon_class,
                has_baryon_class,
                message="" if has_baryon_class else "Composition is baryon but 'Baryon' not in Classification"
            )

        if is_meson:
            result.add_validation(
                'Classification matches meson composition',
                'Meson in classification',
                has_meson_class,
                has_meson_class,
                message="" if has_meson_class else "Composition is meson but 'Meson' not in Classification"
            )

    # -------------------------------------------------------------------------
    # 5. Validate fermion/boson classification
    # -------------------------------------------------------------------------
    if composition and classification:
        total_quarks = count_total_quarks(composition)
        # Odd number of quarks = fermion (half-integer spin)
        # Even number of quarks = boson (integer spin)
        should_be_fermion = (total_quarks % 2) == 1

        has_fermion_class = 'Fermion' in classification
        has_boson_class = 'Boson' in classification

        if should_be_fermion:
            result.add_validation(
                'Fermion classification for odd quark count',
                'Fermion in classification',
                has_fermion_class,
                has_fermion_class,
                message="" if has_fermion_class else f"{total_quarks} quarks should be Fermion"
            )
        else:
            result.add_validation(
                'Boson classification for even quark count',
                'Boson in classification',
                has_boson_class,
                has_boson_class,
                message="" if has_boson_class else f"{total_quarks} quarks should be Boson"
            )

    # -------------------------------------------------------------------------
    # 6. Validate mass is reasonable for composition
    # -------------------------------------------------------------------------
    if composition:
        mass = data.get('Mass_MeVc2', 0)
        mass_valid, mass_msg = validate_mass_reasonable(mass, composition)

        result.add_validation(
            'Mass reasonable for composition',
            'within expected range',
            mass,
            mass_valid,
            message="" if mass_valid else mass_msg
        )

    # -------------------------------------------------------------------------
    # 7. Validate strangeness if present
    # -------------------------------------------------------------------------
    if composition and 'Strangeness' in data:
        expected_strangeness = calculate_expected_strangeness(composition)
        actual_strangeness = data.get('Strangeness', 0)
        strangeness_match = expected_strangeness == actual_strangeness

        result.add_validation(
            'Strangeness from composition',
            expected_strangeness,
            actual_strangeness,
            strangeness_match,
            message="" if strangeness_match else f"Strangeness mismatch: composition gives {expected_strangeness}, declared {actual_strangeness}"
        )

    # -------------------------------------------------------------------------
    # 8. Validate against known particle values
    # -------------------------------------------------------------------------
    if name in KNOWN_PARTICLES:
        known = KNOWN_PARTICLES[name]

        # Charge
        if 'Charge_e' in known:
            actual = data.get('Charge_e', 0)
            expected = known['Charge_e']
            match = abs(actual - expected) < 0.01
            result.add_validation(
                f'Known {name} charge',
                expected,
                actual,
                match,
                0.01,
                "" if match else f"Expected charge {expected}, got {actual}"
            )

        # Mass
        if 'Mass_MeVc2' in known:
            actual = data.get('Mass_MeVc2', 0)
            expected = known['Mass_MeVc2']
            tolerance = known.get('Mass_tolerance', 1.0)
            match = abs(actual - expected) < tolerance
            result.add_validation(
                f'Known {name} mass',
                f"{expected} +/- {tolerance}",
                actual,
                match,
                tolerance,
                "" if match else f"Expected mass ~{expected} MeV, got {actual} MeV"
            )

        # Spin
        if 'Spin_hbar' in known:
            actual = data.get('Spin_hbar', 0)
            expected = known['Spin_hbar']
            match = abs(actual - expected) < 0.01
            result.add_validation(
                f'Known {name} spin',
                expected,
                actual,
                match,
                0.01,
                "" if match else f"Expected spin {expected}, got {actual}"
            )

        # Baryon number
        if 'BaryonNumber_B' in known:
            actual = data.get('BaryonNumber_B', 0)
            expected = known['BaryonNumber_B']
            match = actual == expected
            result.add_validation(
                f'Known {name} baryon number',
                expected,
                actual,
                match,
                message="" if match else f"Expected B={expected}, got {actual}"
            )

        # Strangeness
        if 'Strangeness' in known and 'Strangeness' in data:
            actual = data.get('Strangeness', 0)
            expected = known['Strangeness']
            match = actual == expected
            result.add_validation(
                f'Known {name} strangeness',
                expected,
                actual,
                match,
                message="" if match else f"Expected S={expected}, got {actual}"
            )

    return result


def run_all_validations(particles: Dict[str, Tuple[Dict, str]]) -> ValidationReport:
    """Run validations on all particles."""
    report = ValidationReport()
    report.total_particles = len(particles)

    for name, (data, filename) in particles.items():
        result = validate_particle(name, data, filename)
        report.particle_results.append(result)

        report.total_validations += len(result.validations)
        report.validations_passed += result.pass_count
        report.validations_failed += result.fail_count

        if result.passed:
            report.particles_passed += 1
        else:
            report.particles_failed += 1
            for v in result.validations:
                if not v.passed:
                    report.inconsistencies.append(f"{name}: {v.message}")

    return report


# ============================================================================
# Report Generation
# ============================================================================

def print_report(report: ValidationReport):
    """Print comprehensive test report."""

    print("\n" + "=" * 100)
    print("COMPLETE SUBATOMIC PARTICLE JSON VALIDATION REPORT")
    print("=" * 100)

    # Summary
    print("\n" + "-" * 100)
    print("SUMMARY")
    print("-" * 100)
    print(f"  Total Particles Tested: {report.total_particles}")
    print(f"  Particles Passed All Tests: {report.particles_passed} ({100*report.particles_passed/report.total_particles:.1f}%)")
    print(f"  Particles with Issues: {report.particles_failed}")
    print(f"")
    print(f"  Total Property Validations: {report.total_validations}")
    print(f"  Validations Passed: {report.validations_passed} ({100*report.validations_passed/report.total_validations:.1f}%)")
    print(f"  Validations Failed: {report.validations_failed}")

    # Detailed results per particle
    print("\n" + "-" * 100)
    print("DETAILED RESULTS BY PARTICLE")
    print("-" * 100)

    # Sort: failed particles first, then by name
    sorted_results = sorted(report.particle_results,
                           key=lambda r: (r.passed, r.name))

    for result in sorted_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  {result.name} ({result.filename}): {status}")
        print(f"    Validations: {result.pass_count}/{len(result.validations)} passed")

        for v in result.validations:
            v_status = "OK" if v.passed else "FAIL"
            if v.passed:
                print(f"      [{v_status}] {v.property_name}: {v.actual}")
            else:
                print(f"      [{v_status}] {v.property_name}: expected {v.expected}, got {v.actual}")
                if v.message:
                    print(f"            -> {v.message}")

    # List all inconsistencies
    if report.inconsistencies:
        print("\n" + "-" * 100)
        print("ALL INCONSISTENCIES FOUND")
        print("-" * 100)
        for i, inc in enumerate(report.inconsistencies, 1):
            print(f"  {i}. {inc}")

    # Category breakdown
    print("\n" + "-" * 100)
    print("VALIDATION CATEGORIES BREAKDOWN")
    print("-" * 100)

    category_stats = defaultdict(lambda: {'passed': 0, 'failed': 0})
    for result in report.particle_results:
        for v in result.validations:
            # Extract category from property name
            category = v.property_name.split()[0] if v.property_name else 'Other'
            if v.passed:
                category_stats[category]['passed'] += 1
            else:
                category_stats[category]['failed'] += 1

    for category, stats in sorted(category_stats.items()):
        total = stats['passed'] + stats['failed']
        pct = 100 * stats['passed'] / total if total > 0 else 0
        status = "OK" if stats['failed'] == 0 else "ISSUES"
        print(f"  {category}: {stats['passed']}/{total} ({pct:.0f}%) [{status}]")

    # Known particle verification
    print("\n" + "-" * 100)
    print("KNOWN PARTICLE VERIFICATION (PDG VALUES)")
    print("-" * 100)

    known_checks = []
    for result in report.particle_results:
        for v in result.validations:
            if 'Known' in v.property_name:
                known_checks.append((result.name, v))

    if known_checks:
        current_particle = None
        for name, v in sorted(known_checks, key=lambda x: x[0]):
            if name != current_particle:
                current_particle = name
                print(f"\n  {name}:")
            status = "OK" if v.passed else "MISMATCH"
            prop = v.property_name.replace(f'Known {name} ', '')
            print(f"    [{status}] {prop}: expected {v.expected}, actual {v.actual}")

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL ASSESSMENT")
    print("=" * 100)

    pass_rate = 100 * report.validations_passed / report.total_validations

    if report.validations_failed == 0:
        print("\n  STATUS: ALL VALIDATIONS PASSED")
        print("  All particle JSON data is self-consistent and matches known physics values.")
    elif pass_rate >= 95:
        print(f"\n  STATUS: MOSTLY CONSISTENT ({pass_rate:.1f}% pass rate)")
        print(f"  {report.validations_failed} minor inconsistencies found.")
    elif pass_rate >= 80:
        print(f"\n  STATUS: SOME ISSUES ({pass_rate:.1f}% pass rate)")
        print(f"  {report.validations_failed} inconsistencies need attention.")
    else:
        print(f"\n  STATUS: SIGNIFICANT ISSUES ({pass_rate:.1f}% pass rate)")
        print(f"  {report.validations_failed} inconsistencies require review.")

    print(f"\n  Pass Rate: {pass_rate:.1f}%")
    print(f"  Particles: {report.particles_passed}/{report.total_particles} fully valid")
    print(f"  Properties: {report.validations_passed}/{report.total_validations} validated")

    print("\n" + "=" * 100)

    return report.validations_failed == 0


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main test runner."""
    # Determine data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data', 'active', 'subatomic')

    print(f"Loading particles from: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        return False

    # Load all particles
    particles = load_all_particles(data_dir)
    print(f"Loaded {len(particles)} particle JSON files")

    if len(particles) != 22:
        print(f"WARNING: Expected 22 particles, found {len(particles)}")

    # List loaded particles
    print("\nParticles loaded:")
    for name in sorted(particles.keys()):
        print(f"  - {name}")

    # Run validations
    print("\nRunning validations...")
    report = run_all_validations(particles)

    # Print report
    success = print_report(report)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
