#!/usr/bin/env python3
"""
Test Prediction Chain: Quarks to Alloys
=========================================

Comprehensive tests validating the entire particle prediction chain:
    Quarks -> Hadrons -> Atoms -> Molecules -> Alloys

Tests verify:
1. Property propagation through each level
2. Conservation laws (charge, baryon number, etc.)
3. Accuracy against stored JSON data
4. Edge cases (hypothetical elements, exotic particles)
5. Novel combinations not in default data

Author: Periodics Test Suite
"""

import sys
import os
import json
import math
import pytest
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.simulation_schema import (
    propagate_quark_to_hadron,
    propagate_hadrons_to_atom,
    propagate_atoms_to_molecule,
    propagate_elements_to_alloy,
    SimulationConstants,
    QuarkSimulationData,
    HadronSimulationData,
    AtomSimulationData,
    MoleculeSimulationData,
    AlloySimulationData,
)

# Try to import V2 calculators
try:
    from periodica.utils.physics_calculator_v2 import (
        SubatomicCalculatorV2,
        AtomCalculatorV2,
        MoleculeCalculatorV2,
        PhysicsConstantsV2
    )
    HAS_V2_CALCULATORS = True
except ImportError:
    HAS_V2_CALCULATORS = False
    print("Warning: V2 calculators not available, some tests will be skipped")


# =============================================================================
# Test Data Definitions - Fundamental Particles
# =============================================================================

# Quark definitions with full properties
UP_QUARK = {
    "Name": "Up Quark",
    "Symbol": "u",
    "Charge_e": 2/3,
    "Mass_MeVc2": 2.16,
    "Mass_kg": 3.85e-30,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": 1/3,
    "Isospin_I": 0.5,
    "Isospin_I3": 0.5,
    "LeptonNumber_L": 0,
    "ColorCharge": "red",
    "Generation": 1,
}

DOWN_QUARK = {
    "Name": "Down Quark",
    "Symbol": "d",
    "Charge_e": -1/3,
    "Mass_MeVc2": 4.67,
    "Mass_kg": 8.32e-30,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": 1/3,
    "Isospin_I": 0.5,
    "Isospin_I3": -0.5,
    "LeptonNumber_L": 0,
    "ColorCharge": "green",
    "Generation": 1,
}

STRANGE_QUARK = {
    "Name": "Strange Quark",
    "Symbol": "s",
    "Charge_e": -1/3,
    "Mass_MeVc2": 93.4,
    "Mass_kg": 1.66e-28,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": 1/3,
    "Isospin_I": 0,
    "Isospin_I3": 0,
    "LeptonNumber_L": 0,
    "Strangeness": -1,
    "ColorCharge": "blue",
    "Generation": 2,
}

CHARM_QUARK = {
    "Name": "Charm Quark",
    "Symbol": "c",
    "Charge_e": 2/3,
    "Mass_MeVc2": 1270,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": 1/3,
    "Isospin_I": 0,
    "Isospin_I3": 0,
    "LeptonNumber_L": 0,
    "Charm": 1,
    "Generation": 2,
}

BOTTOM_QUARK = {
    "Name": "Bottom Quark",
    "Symbol": "b",
    "Charge_e": -1/3,
    "Mass_MeVc2": 4180,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": 1/3,
    "Isospin_I": 0,
    "Isospin_I3": 0,
    "LeptonNumber_L": 0,
    "Bottomness": -1,
    "Generation": 3,
}

# Antiquark definitions
ANTI_UP_QUARK = {
    "Name": "Anti-Up Quark",
    "Symbol": "u\u0305",
    "Charge_e": -2/3,
    "Mass_MeVc2": 2.16,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": -1/3,
    "Isospin_I": 0.5,
    "Isospin_I3": -0.5,
    "LeptonNumber_L": 0,
}

ANTI_DOWN_QUARK = {
    "Name": "Anti-Down Quark",
    "Symbol": "d\u0305",
    "Charge_e": 1/3,
    "Mass_MeVc2": 4.67,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": -1/3,
    "Isospin_I": 0.5,
    "Isospin_I3": 0.5,
    "LeptonNumber_L": 0,
}

ANTI_STRANGE_QUARK = {
    "Name": "Anti-Strange Quark",
    "Symbol": "s\u0305",
    "Charge_e": 1/3,
    "Mass_MeVc2": 93.4,
    "Spin_hbar": 0.5,
    "BaryonNumber_B": -1/3,
    "Isospin_I": 0,
    "Isospin_I3": 0,
    "Strangeness": 1,
}

# Electron definition
ELECTRON = {
    "Name": "Electron",
    "Symbol": "e-",
    "Charge_e": -1,
    "Mass_MeVc2": 0.511,
    "Mass_kg": 9.109e-31,
    "Mass_amu": 0.000548579,
    "Spin_hbar": 0.5,
    "LeptonNumber_L": 1,
    "BaryonNumber_B": 0,
}


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class ChainTestResult:
    """Result from a single chain test."""
    chain_name: str
    step: str
    property_name: str
    expected: Any
    actual: Any
    passed: bool
    error_percent: Optional[float] = None
    notes: str = ""


@dataclass
class ChainTestSummary:
    """Summary of all chain tests."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    results: List[ChainTestResult] = field(default_factory=list)

    def add_result(self, result: ChainTestResult):
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1


@pytest.fixture
def summary():
    """Fixture providing a ChainTestSummary instance for tests."""
    return ChainTestSummary()


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_json_with_comments(filepath: Path) -> Optional[Dict]:
    """Load JSON file, handling JavaScript-style comments."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove // comments
            lines = []
            for line in content.split('\n'):
                idx = line.find('//')
                if idx != -1:
                    # Check if inside string
                    quote_count = line[:idx].count('"')
                    if quote_count % 2 == 0:
                        line = line[:idx]
                lines.append(line)
            content = '\n'.join(lines)
            return json.loads(content)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_all_defaults():
    """Load all default data files."""
    base_path = Path(__file__).parent.parent / "src" / "periodica" / "data" / "defaults"

    data = {
        'quarks': {},
        'subatomic': {},
        'elements': {},
        'molecules': {},
        'alloys': {},
    }

    # Load quarks
    quark_path = base_path / "quarks"
    if quark_path.exists():
        for f in quark_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['quarks'][d.get('Name', f.stem)] = d

    # Load subatomic particles
    subatomic_path = base_path / "subatomic"
    if subatomic_path.exists():
        for f in subatomic_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['subatomic'][d.get('Name', f.stem)] = d

    # Load elements
    element_path = base_path / "elements"
    if element_path.exists():
        for f in sorted(element_path.glob("*.json")):
            d = load_json_with_comments(f)
            if d and 'atomic_number' in d:
                data['elements'][d['atomic_number']] = d

    # Load molecules
    molecule_path = base_path / "molecules"
    if molecule_path.exists():
        for f in molecule_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['molecules'][d.get('Name', f.stem)] = d

    # Load alloys
    alloy_path = base_path / "alloys"
    if alloy_path.exists():
        for f in alloy_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['alloys'][d.get('Name', f.stem)] = d

    return data


def load_root_level_data():
    """Load data from root-level Quarks, SubAtomic, and Molecules folders."""
    root_path = Path(__file__).parent.parent

    data = {
        'quarks': {},
        'subatomic': {},
        'molecules': {},
    }

    # Load from Quarks folder
    quark_path = root_path / "Quarks"
    if quark_path.exists():
        for f in quark_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['quarks'][d.get('Name', f.stem)] = d

    # Load from SubAtomic folder
    subatomic_path = root_path / "SubAtomic"
    if subatomic_path.exists():
        for f in subatomic_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['subatomic'][d.get('Name', f.stem)] = d

    # Load from Molecules folder
    molecule_path = root_path / "Molecules"
    if molecule_path.exists():
        for f in molecule_path.glob("*.json"):
            d = load_json_with_comments(f)
            if d:
                data['molecules'][d.get('Name', f.stem)] = d

    return data


# =============================================================================
# Test: Quarks to Hadrons
# =============================================================================

def test_quark_to_hadron(summary: ChainTestSummary):
    """Test building hadrons from quarks."""
    print("\n" + "="*80)
    print("TEST 1: QUARKS -> HADRONS")
    print("="*80)

    # Test 1: Build Proton from uud quarks
    print("\n--- Building Proton from uud ---")
    proton = propagate_quark_to_hadron([UP_QUARK, UP_QUARK, DOWN_QUARK])

    # Expected: charge = 2/3 + 2/3 - 1/3 = 1
    expected_charge = 1.0
    actual_charge = proton.charge_e
    charge_pass = abs(actual_charge - expected_charge) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Proton", step="Quarks->Hadron", property_name="Charge",
        expected=expected_charge, actual=actual_charge, passed=charge_pass
    ))
    print(f"  Charge: Expected {expected_charge}, Got {actual_charge} {'PASS' if charge_pass else 'FAIL'}")

    # Expected: baryon number = 1/3 + 1/3 + 1/3 = 1
    expected_baryon = 1.0
    actual_baryon = proton.baryon_number
    baryon_pass = abs(actual_baryon - expected_baryon) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Proton", step="Quarks->Hadron", property_name="BaryonNumber",
        expected=expected_baryon, actual=actual_baryon, passed=baryon_pass
    ))
    print(f"  Baryon Number: Expected {expected_baryon}, Got {actual_baryon} {'PASS' if baryon_pass else 'FAIL'}")

    # Expected: quark content string = "uud"
    expected_content = "uud"
    actual_content = proton.quark_content_string
    content_pass = actual_content == expected_content
    summary.add_result(ChainTestResult(
        chain_name="Proton", step="Quarks->Hadron", property_name="QuarkContent",
        expected=expected_content, actual=actual_content, passed=content_pass
    ))
    print(f"  Quark Content: Expected '{expected_content}', Got '{actual_content}' {'PASS' if content_pass else 'FAIL'}")

    # Expected: mass ~ 938 MeV (within 10%)
    expected_mass = 938.3
    actual_mass = proton.mass_MeV
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 10
    summary.add_result(ChainTestResult(
        chain_name="Proton", step="Quarks->Hadron", property_name="Mass_MeV",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Mass: Expected ~{expected_mass} MeV, Got {actual_mass:.1f} MeV (Error: {mass_error:.1f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Test 2: Build Neutron from udd quarks
    print("\n--- Building Neutron from udd ---")
    neutron = propagate_quark_to_hadron([UP_QUARK, DOWN_QUARK, DOWN_QUARK])

    # Expected: charge = 2/3 - 1/3 - 1/3 = 0
    expected_charge = 0.0
    actual_charge = neutron.charge_e
    charge_pass = abs(actual_charge - expected_charge) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Neutron", step="Quarks->Hadron", property_name="Charge",
        expected=expected_charge, actual=actual_charge, passed=charge_pass
    ))
    print(f"  Charge: Expected {expected_charge}, Got {actual_charge} {'PASS' if charge_pass else 'FAIL'}")

    # Test 3: Build Pion+ from ud-bar (meson)
    print("\n--- Building Pion+ from u d-bar ---")
    pion_plus = propagate_quark_to_hadron([UP_QUARK, ANTI_DOWN_QUARK])

    # Expected: charge = 2/3 + 1/3 = 1
    expected_charge = 1.0
    actual_charge = pion_plus.charge_e
    charge_pass = abs(actual_charge - expected_charge) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Pion+", step="Quarks->Hadron", property_name="Charge",
        expected=expected_charge, actual=actual_charge, passed=charge_pass
    ))
    print(f"  Charge: Expected {expected_charge}, Got {actual_charge} {'PASS' if charge_pass else 'FAIL'}")

    # Expected: baryon number = 1/3 - 1/3 = 0 (meson)
    expected_baryon = 0.0
    actual_baryon = pion_plus.baryon_number
    baryon_pass = abs(actual_baryon - expected_baryon) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Pion+", step="Quarks->Hadron", property_name="BaryonNumber",
        expected=expected_baryon, actual=actual_baryon, passed=baryon_pass
    ))
    print(f"  Baryon Number: Expected {expected_baryon}, Got {actual_baryon} {'PASS' if baryon_pass else 'FAIL'}")

    # Expected: particle type = "meson"
    expected_type = "meson"
    actual_type = pion_plus.particle_type
    type_pass = actual_type == expected_type
    summary.add_result(ChainTestResult(
        chain_name="Pion+", step="Quarks->Hadron", property_name="ParticleType",
        expected=expected_type, actual=actual_type, passed=type_pass
    ))
    print(f"  Type: Expected '{expected_type}', Got '{actual_type}' {'PASS' if type_pass else 'FAIL'}")

    # Test 4: Build Lambda from uds (strange baryon)
    print("\n--- Building Lambda from uds ---")
    lambda_particle = propagate_quark_to_hadron([UP_QUARK, DOWN_QUARK, STRANGE_QUARK])

    # Expected: charge = 2/3 - 1/3 - 1/3 = 0
    expected_charge = 0.0
    actual_charge = lambda_particle.charge_e
    charge_pass = abs(actual_charge - expected_charge) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Lambda", step="Quarks->Hadron", property_name="Charge",
        expected=expected_charge, actual=actual_charge, passed=charge_pass
    ))
    print(f"  Charge: Expected {expected_charge}, Got {actual_charge} {'PASS' if charge_pass else 'FAIL'}")

    # Expected: mass ~ 1116 MeV (within 15%)
    expected_mass = 1115.7
    actual_mass = lambda_particle.mass_MeV
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 15
    summary.add_result(ChainTestResult(
        chain_name="Lambda", step="Quarks->Hadron", property_name="Mass_MeV",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Mass: Expected ~{expected_mass} MeV, Got {actual_mass:.1f} MeV (Error: {mass_error:.1f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Test 5: Build Omega- from sss (triple strange)
    print("\n--- Building Omega- from sss ---")
    omega = propagate_quark_to_hadron([STRANGE_QUARK, STRANGE_QUARK, STRANGE_QUARK])

    # Expected: charge = -1/3 - 1/3 - 1/3 = -1
    expected_charge = -1.0
    actual_charge = omega.charge_e
    charge_pass = abs(actual_charge - expected_charge) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Omega-", step="Quarks->Hadron", property_name="Charge",
        expected=expected_charge, actual=actual_charge, passed=charge_pass
    ))
    print(f"  Charge: Expected {expected_charge}, Got {actual_charge} {'PASS' if charge_pass else 'FAIL'}")


# =============================================================================
# Test: Hadrons to Atoms
# =============================================================================

def test_hadron_to_atom(summary: ChainTestSummary, proton_data=None, neutron_data=None):
    """Test building atoms from hadrons (protons/neutrons)."""
    print("\n" + "="*80)
    print("TEST 2: HADRONS -> ATOMS")
    print("="*80)

    # Test 1: Build Hydrogen (1 proton, 0 neutrons, 1 electron)
    print("\n--- Building Hydrogen (Z=1, N=0) ---")
    hydrogen = propagate_hadrons_to_atom(
        protons=1, neutrons=0, electrons=1,
        proton_data=proton_data.to_dict() if proton_data else None,
        neutron_data=neutron_data.to_dict() if neutron_data else None
    )

    # Check atomic number
    expected_z = 1
    actual_z = hydrogen.atomic_number
    z_pass = actual_z == expected_z
    summary.add_result(ChainTestResult(
        chain_name="Hydrogen", step="Hadrons->Atom", property_name="AtomicNumber",
        expected=expected_z, actual=actual_z, passed=z_pass
    ))
    print(f"  Atomic Number: Expected {expected_z}, Got {actual_z} {'PASS' if z_pass else 'FAIL'}")

    # Check atomic mass ~ 1.008 amu
    expected_mass = 1.008
    actual_mass = hydrogen.atomic_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 1
    summary.add_result(ChainTestResult(
        chain_name="Hydrogen", step="Hadrons->Atom", property_name="AtomicMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Atomic Mass: Expected ~{expected_mass} amu, Got {actual_mass:.4f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Test 2: Build Helium-4 (2 protons, 2 neutrons, 2 electrons)
    print("\n--- Building Helium-4 (Z=2, N=2) ---")
    helium = propagate_hadrons_to_atom(protons=2, neutrons=2, electrons=2)

    # Check atomic mass ~ 4.003 amu
    expected_mass = 4.003
    actual_mass = helium.atomic_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 2
    summary.add_result(ChainTestResult(
        chain_name="Helium-4", step="Hadrons->Atom", property_name="AtomicMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Atomic Mass: Expected ~{expected_mass} amu, Got {actual_mass:.4f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Check binding energy per nucleon ~ 7.07 MeV
    expected_be = 28.3  # Total BE for He-4
    actual_be = helium.nuclear_binding_energy_MeV
    be_error = abs(actual_be - expected_be) / expected_be * 100
    be_pass = be_error < 5
    summary.add_result(ChainTestResult(
        chain_name="Helium-4", step="Hadrons->Atom", property_name="BindingEnergy",
        expected=expected_be, actual=actual_be, passed=be_pass,
        error_percent=be_error
    ))
    print(f"  Binding Energy: Expected ~{expected_be} MeV, Got {actual_be:.1f} MeV (Error: {be_error:.1f}%) {'PASS' if be_pass else 'FAIL'}")

    # Test 3: Build Carbon-12 (6 protons, 6 neutrons, 6 electrons)
    print("\n--- Building Carbon-12 (Z=6, N=6) ---")
    carbon = propagate_hadrons_to_atom(protons=6, neutrons=6, electrons=6)

    # Check atomic mass ~ 12.0 amu (by definition)
    expected_mass = 12.0
    actual_mass = carbon.atomic_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 1
    summary.add_result(ChainTestResult(
        chain_name="Carbon-12", step="Hadrons->Atom", property_name="AtomicMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Atomic Mass: Expected ~{expected_mass} amu, Got {actual_mass:.4f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Check nuclear radius
    expected_radius = 1.2 * (12 ** (1/3))  # r = r0 * A^(1/3), r0 = 1.2 fm
    actual_radius = carbon.nuclear_radius_fm
    radius_error = abs(actual_radius - expected_radius) / expected_radius * 100
    radius_pass = radius_error < 5
    summary.add_result(ChainTestResult(
        chain_name="Carbon-12", step="Hadrons->Atom", property_name="NuclearRadius",
        expected=expected_radius, actual=actual_radius, passed=radius_pass,
        error_percent=radius_error
    ))
    print(f"  Nuclear Radius: Expected ~{expected_radius:.2f} fm, Got {actual_radius:.2f} fm (Error: {radius_error:.1f}%) {'PASS' if radius_pass else 'FAIL'}")

    # Test 4: Build Iron-56 (most stable nucleus)
    print("\n--- Building Iron-56 (Z=26, N=30) ---")
    iron = propagate_hadrons_to_atom(protons=26, neutrons=30, electrons=26)

    # Check binding energy per nucleon ~ 8.79 MeV (highest)
    expected_be_per_nucleon = 8.79
    actual_be_per_nucleon = iron.nuclear_binding_energy_MeV / 56
    be_error = abs(actual_be_per_nucleon - expected_be_per_nucleon) / expected_be_per_nucleon * 100
    be_pass = be_error < 5
    summary.add_result(ChainTestResult(
        chain_name="Iron-56", step="Hadrons->Atom", property_name="BE_per_nucleon",
        expected=expected_be_per_nucleon, actual=actual_be_per_nucleon, passed=be_pass,
        error_percent=be_error
    ))
    print(f"  BE/nucleon: Expected ~{expected_be_per_nucleon} MeV, Got {actual_be_per_nucleon:.2f} MeV (Error: {be_error:.1f}%) {'PASS' if be_pass else 'FAIL'}")


# =============================================================================
# Test: Atoms to Molecules
# =============================================================================

def test_atom_to_molecule(summary: ChainTestSummary, hydrogen_atom=None, carbon_atom=None):
    """Test building molecules from atoms."""
    print("\n" + "="*80)
    print("TEST 3: ATOMS -> MOLECULES")
    print("="*80)

    # Atom data for molecule building
    h_atom = {"symbol": "H", "atomic_mass": 1.008, "atomic_number": 1, "electronegativity": 2.2}
    o_atom = {"symbol": "O", "atomic_mass": 15.999, "atomic_number": 8, "electronegativity": 3.44}
    c_atom = {"symbol": "C", "atomic_mass": 12.011, "atomic_number": 6, "electronegativity": 2.55}
    n_atom = {"symbol": "N", "atomic_mass": 14.007, "atomic_number": 7, "electronegativity": 3.04}

    # Test 1: Build H2 molecule
    print("\n--- Building H2 (Hydrogen Gas) ---")
    h2_bonds = [{"atom1": 0, "atom2": 1, "type": "single", "length_pm": 74}]
    h2 = propagate_atoms_to_molecule([h_atom, h_atom], h2_bonds)

    # Check molecular mass ~ 2.016 amu
    expected_mass = 2.016
    actual_mass = h2.molecular_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 1
    summary.add_result(ChainTestResult(
        chain_name="H2", step="Atoms->Molecule", property_name="MolecularMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Molecular Mass: Expected ~{expected_mass} amu, Got {actual_mass:.3f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Check total electrons
    expected_electrons = 2
    actual_electrons = h2.total_electrons
    electrons_pass = actual_electrons == expected_electrons
    summary.add_result(ChainTestResult(
        chain_name="H2", step="Atoms->Molecule", property_name="TotalElectrons",
        expected=expected_electrons, actual=actual_electrons, passed=electrons_pass
    ))
    print(f"  Total Electrons: Expected {expected_electrons}, Got {actual_electrons} {'PASS' if electrons_pass else 'FAIL'}")

    # Check formula
    expected_formula = "H2"
    actual_formula = h2.formula
    formula_pass = actual_formula == expected_formula
    summary.add_result(ChainTestResult(
        chain_name="H2", step="Atoms->Molecule", property_name="Formula",
        expected=expected_formula, actual=actual_formula, passed=formula_pass
    ))
    print(f"  Formula: Expected '{expected_formula}', Got '{actual_formula}' {'PASS' if formula_pass else 'FAIL'}")

    # Test 2: Build H2O molecule
    print("\n--- Building H2O (Water) ---")
    h2o_bonds = [
        {"atom1": 0, "atom2": 2, "type": "single", "length_pm": 96},
        {"atom1": 1, "atom2": 2, "type": "single", "length_pm": 96}
    ]
    h2o = propagate_atoms_to_molecule([h_atom, h_atom, o_atom], h2o_bonds)

    # Check molecular mass ~ 18.015 amu
    expected_mass = 18.015
    actual_mass = h2o.molecular_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 1
    summary.add_result(ChainTestResult(
        chain_name="H2O", step="Atoms->Molecule", property_name="MolecularMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Molecular Mass: Expected ~{expected_mass} amu, Got {actual_mass:.3f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Check total electrons (2 + 8 = 10)
    expected_electrons = 10
    actual_electrons = h2o.total_electrons
    electrons_pass = actual_electrons == expected_electrons
    summary.add_result(ChainTestResult(
        chain_name="H2O", step="Atoms->Molecule", property_name="TotalElectrons",
        expected=expected_electrons, actual=actual_electrons, passed=electrons_pass
    ))
    print(f"  Total Electrons: Expected {expected_electrons}, Got {actual_electrons} {'PASS' if electrons_pass else 'FAIL'}")

    # Test 3: Build CO2 molecule
    print("\n--- Building CO2 (Carbon Dioxide) ---")
    co2_bonds = [
        {"atom1": 0, "atom2": 1, "type": "double", "length_pm": 116},
        {"atom1": 0, "atom2": 2, "type": "double", "length_pm": 116}
    ]
    co2 = propagate_atoms_to_molecule([c_atom, o_atom, o_atom], co2_bonds)

    # Check molecular mass ~ 44.01 amu
    expected_mass = 44.009
    actual_mass = co2.molecular_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 1
    summary.add_result(ChainTestResult(
        chain_name="CO2", step="Atoms->Molecule", property_name="MolecularMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Molecular Mass: Expected ~{expected_mass} amu, Got {actual_mass:.3f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Test 4: Build CH4 (Methane)
    print("\n--- Building CH4 (Methane) ---")
    ch4_bonds = [
        {"atom1": 0, "atom2": 1, "type": "single", "length_pm": 109},
        {"atom1": 0, "atom2": 2, "type": "single", "length_pm": 109},
        {"atom1": 0, "atom2": 3, "type": "single", "length_pm": 109},
        {"atom1": 0, "atom2": 4, "type": "single", "length_pm": 109},
    ]
    ch4 = propagate_atoms_to_molecule([c_atom, h_atom, h_atom, h_atom, h_atom], ch4_bonds)

    # Check molecular mass ~ 16.04 amu
    expected_mass = 16.043
    actual_mass = ch4.molecular_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 1
    summary.add_result(ChainTestResult(
        chain_name="CH4", step="Atoms->Molecule", property_name="MolecularMass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass,
        error_percent=mass_error
    ))
    print(f"  Molecular Mass: Expected ~{expected_mass} amu, Got {actual_mass:.3f} amu (Error: {mass_error:.2f}%) {'PASS' if mass_pass else 'FAIL'}")

    # Check total electrons (6 + 4 = 10)
    expected_electrons = 10
    actual_electrons = ch4.total_electrons
    electrons_pass = actual_electrons == expected_electrons
    summary.add_result(ChainTestResult(
        chain_name="CH4", step="Atoms->Molecule", property_name="TotalElectrons",
        expected=expected_electrons, actual=actual_electrons, passed=electrons_pass
    ))
    print(f"  Total Electrons: Expected {expected_electrons}, Got {actual_electrons} {'PASS' if electrons_pass else 'FAIL'}")


# =============================================================================
# Test: Elements to Alloys
# =============================================================================

def test_elements_to_alloy(summary: ChainTestSummary):
    """Test building alloys from elements."""
    print("\n" + "="*80)
    print("TEST 4: ELEMENTS -> ALLOYS")
    print("="*80)

    # Element data for alloy building
    fe_element = {"symbol": "Fe", "atomic_mass": 55.845, "density": 7.874, "melting_point": 1811}
    cr_element = {"symbol": "Cr", "atomic_mass": 51.996, "density": 7.19, "melting_point": 2180}
    ni_element = {"symbol": "Ni", "atomic_mass": 58.693, "density": 8.908, "melting_point": 1728}
    cu_element = {"symbol": "Cu", "atomic_mass": 63.546, "density": 8.96, "melting_point": 1358}
    zn_element = {"symbol": "Zn", "atomic_mass": 65.38, "density": 7.14, "melting_point": 693}
    al_element = {"symbol": "Al", "atomic_mass": 26.982, "density": 2.70, "melting_point": 933}

    # Test 1: Build Stainless Steel 304 (Fe-Cr-Ni)
    print("\n--- Building Stainless Steel 304 (Fe 72%, Cr 18%, Ni 10%) ---")
    ss304 = propagate_elements_to_alloy(
        elements=[fe_element, cr_element, ni_element],
        weight_fractions=[0.72, 0.18, 0.10],
        lattice_type="FCC"
    )

    # Check density (should be ~7.9 g/cm3)
    expected_density = 7.93
    actual_density = ss304.density_g_cm3
    density_error = abs(actual_density - expected_density) / expected_density * 100
    density_pass = density_error < 5
    summary.add_result(ChainTestResult(
        chain_name="SS304", step="Elements->Alloy", property_name="Density",
        expected=expected_density, actual=actual_density, passed=density_pass,
        error_percent=density_error
    ))
    print(f"  Density: Expected ~{expected_density} g/cm3, Got {actual_density:.2f} g/cm3 (Error: {density_error:.1f}%) {'PASS' if density_pass else 'FAIL'}")

    # Check lattice type
    expected_structure = "FCC"
    actual_structure = ss304.primary_structure
    structure_pass = actual_structure == expected_structure
    summary.add_result(ChainTestResult(
        chain_name="SS304", step="Elements->Alloy", property_name="Structure",
        expected=expected_structure, actual=actual_structure, passed=structure_pass
    ))
    print(f"  Structure: Expected '{expected_structure}', Got '{actual_structure}' {'PASS' if structure_pass else 'FAIL'}")

    # Check packing factor for FCC
    expected_packing = 0.74
    actual_packing = ss304.packing_factor
    packing_pass = abs(actual_packing - expected_packing) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="SS304", step="Elements->Alloy", property_name="PackingFactor",
        expected=expected_packing, actual=actual_packing, passed=packing_pass
    ))
    print(f"  Packing Factor: Expected {expected_packing}, Got {actual_packing} {'PASS' if packing_pass else 'FAIL'}")

    # Test 2: Build Brass (Cu-Zn)
    print("\n--- Building Brass (Cu 60%, Zn 40%) ---")
    brass = propagate_elements_to_alloy(
        elements=[cu_element, zn_element],
        weight_fractions=[0.60, 0.40],
        lattice_type="FCC"
    )

    # Check density (should be ~8.4 g/cm3)
    expected_density = 8.4
    actual_density = brass.density_g_cm3
    density_error = abs(actual_density - expected_density) / expected_density * 100
    density_pass = density_error < 10
    summary.add_result(ChainTestResult(
        chain_name="Brass", step="Elements->Alloy", property_name="Density",
        expected=expected_density, actual=actual_density, passed=density_pass,
        error_percent=density_error
    ))
    print(f"  Density: Expected ~{expected_density} g/cm3, Got {actual_density:.2f} g/cm3 (Error: {density_error:.1f}%) {'PASS' if density_pass else 'FAIL'}")

    # Test 3: Build Aluminum Alloy (Al-Cu)
    print("\n--- Building Duralumin-like (Al 95%, Cu 5%) ---")
    duralumin = propagate_elements_to_alloy(
        elements=[al_element, cu_element],
        weight_fractions=[0.95, 0.05],
        lattice_type="FCC"
    )

    # Check density (should be ~2.8 g/cm3)
    expected_density = 2.8
    actual_density = duralumin.density_g_cm3
    density_error = abs(actual_density - expected_density) / expected_density * 100
    density_pass = density_error < 10
    summary.add_result(ChainTestResult(
        chain_name="Duralumin", step="Elements->Alloy", property_name="Density",
        expected=expected_density, actual=actual_density, passed=density_pass,
        error_percent=density_error
    ))
    print(f"  Density: Expected ~{expected_density} g/cm3, Got {actual_density:.2f} g/cm3 (Error: {density_error:.1f}%) {'PASS' if density_pass else 'FAIL'}")


# =============================================================================
# Test: Edge Cases
# =============================================================================

def test_edge_cases(summary: ChainTestSummary):
    """Test edge cases and unusual combinations."""
    print("\n" + "="*80)
    print("TEST 5: EDGE CASES")
    print("="*80)

    # Test 1: Hypothetical superheavy element (Z > 118)
    print("\n--- Hypothetical Element Z=120 (Unbinilium) ---")
    ubn = propagate_hadrons_to_atom(protons=120, neutrons=180, electrons=120)

    # Check that properties are calculated
    has_mass = ubn.atomic_mass_amu > 0
    has_radius = ubn.nuclear_radius_fm > 0
    has_be = ubn.nuclear_binding_energy_MeV != 0

    summary.add_result(ChainTestResult(
        chain_name="Z=120", step="EdgeCase", property_name="HasMass",
        expected=True, actual=has_mass, passed=has_mass
    ))
    print(f"  Has valid mass: {has_mass} (Mass: {ubn.atomic_mass_amu:.2f} amu)")
    print(f"  Has valid radius: {has_radius} (Radius: {ubn.nuclear_radius_fm:.2f} fm)")
    print(f"  Has binding energy: {has_be} (BE: {ubn.nuclear_binding_energy_MeV:.1f} MeV)")

    # Check reasonableness - binding energy per nucleon should be positive but less than Fe-56
    be_per_nucleon = ubn.nuclear_binding_energy_MeV / 300
    reasonable_be = 5 < be_per_nucleon < 8.5
    summary.add_result(ChainTestResult(
        chain_name="Z=120", step="EdgeCase", property_name="ReasonableBE",
        expected="5-8.5 MeV", actual=be_per_nucleon, passed=reasonable_be
    ))
    print(f"  BE/nucleon: {be_per_nucleon:.2f} MeV (reasonable: {reasonable_be})")

    # Test 2: Unusual isotope - neutron-rich Tin (neutron drip line)
    print("\n--- Neutron-rich Sn-150 (Z=50, N=100) ---")
    sn150 = propagate_hadrons_to_atom(protons=50, neutrons=100, electrons=50)

    # This isotope should have positive but lower binding energy
    has_be = sn150.nuclear_binding_energy_MeV > 0
    summary.add_result(ChainTestResult(
        chain_name="Sn-150", step="EdgeCase", property_name="HasPositiveBE",
        expected=True, actual=has_be, passed=has_be
    ))
    print(f"  Has positive binding energy: {has_be} (BE: {sn150.nuclear_binding_energy_MeV:.1f} MeV)")

    # Test 3: Exotic hadron - tetraquark (ccud)
    print("\n--- Exotic Tetraquark (c c u d-bar) ---")
    tetraquark = propagate_quark_to_hadron([CHARM_QUARK, CHARM_QUARK, UP_QUARK, ANTI_DOWN_QUARK])

    # Check particle type
    expected_type = "exotic"
    actual_type = tetraquark.particle_type
    type_pass = actual_type == expected_type
    summary.add_result(ChainTestResult(
        chain_name="Tetraquark", step="EdgeCase", property_name="ParticleType",
        expected=expected_type, actual=actual_type, passed=type_pass
    ))
    print(f"  Type: Expected '{expected_type}', Got '{actual_type}'")

    # Check charge = 2/3 + 2/3 + 2/3 + 1/3 = 7/3 ? No, c=2/3, c=2/3, u=2/3, dbar=1/3 = 7/3
    # Actually ccud-bar: 2/3 + 2/3 + 2/3 + 1/3 = 7/3 = 2.333
    expected_charge = 2/3 + 2/3 + 2/3 + 1/3
    actual_charge = tetraquark.charge_e
    charge_pass = abs(actual_charge - expected_charge) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Tetraquark", step="EdgeCase", property_name="Charge",
        expected=expected_charge, actual=actual_charge, passed=charge_pass
    ))
    print(f"  Charge: Expected {expected_charge:.3f}, Got {actual_charge:.3f}")

    # Check baryon number = 1/3 + 1/3 + 1/3 - 1/3 = 2/3
    expected_baryon = 1/3 + 1/3 + 1/3 - 1/3
    actual_baryon = tetraquark.baryon_number
    baryon_pass = abs(actual_baryon - expected_baryon) < 0.01
    summary.add_result(ChainTestResult(
        chain_name="Tetraquark", step="EdgeCase", property_name="BaryonNumber",
        expected=expected_baryon, actual=actual_baryon, passed=baryon_pass
    ))
    print(f"  Baryon Number: Expected {expected_baryon:.3f}, Got {actual_baryon:.3f}")

    # Test 4: Ion - Sodium cation (Na+)
    print("\n--- Sodium Ion Na+ (Z=11, N=12, e=10) ---")
    na_plus = propagate_hadrons_to_atom(protons=11, neutrons=12, electrons=10)

    # Check mass is similar to Na
    expected_mass = 23.0
    actual_mass = na_plus.atomic_mass_amu
    mass_error = abs(actual_mass - expected_mass) / expected_mass * 100
    mass_pass = mass_error < 2
    summary.add_result(ChainTestResult(
        chain_name="Na+", step="EdgeCase", property_name="Mass",
        expected=expected_mass, actual=actual_mass, passed=mass_pass
    ))
    print(f"  Mass: Expected ~{expected_mass} amu, Got {actual_mass:.2f} amu")


# =============================================================================
# Test: Full Chain Integration
# =============================================================================

def test_full_chain(summary: ChainTestSummary):
    """Test complete chain from quarks to alloys."""
    print("\n" + "="*80)
    print("TEST 6: FULL CHAIN (Quarks -> Hadrons -> Atoms -> Molecules/Alloys)")
    print("="*80)

    print("\n--- Full Chain: Building Steel from Quarks ---")

    # Step 1: Build proton from quarks
    print("\n  Step 1: Quarks -> Proton")
    proton = propagate_quark_to_hadron([UP_QUARK, UP_QUARK, DOWN_QUARK])
    print(f"    Proton mass: {proton.mass_MeV:.1f} MeV, charge: {proton.charge_e}")

    # Step 2: Build neutron from quarks
    print("\n  Step 2: Quarks -> Neutron")
    neutron = propagate_quark_to_hadron([UP_QUARK, DOWN_QUARK, DOWN_QUARK])
    print(f"    Neutron mass: {neutron.mass_MeV:.1f} MeV, charge: {neutron.charge_e}")

    # Step 3: Build Iron atom from hadrons
    print("\n  Step 3: Hadrons -> Iron Atom")
    iron = propagate_hadrons_to_atom(protons=26, neutrons=30, electrons=26)
    print(f"    Iron-56 mass: {iron.atomic_mass_amu:.3f} amu, Z={iron.atomic_number}")

    # Step 4: Build Carbon atom from hadrons
    print("\n  Step 4: Hadrons -> Carbon Atom")
    carbon = propagate_hadrons_to_atom(protons=6, neutrons=6, electrons=6)
    print(f"    Carbon-12 mass: {carbon.atomic_mass_amu:.3f} amu, Z={carbon.atomic_number}")

    # Step 5: Build Carbon Steel from atoms
    print("\n  Step 5: Atoms -> Carbon Steel Alloy")
    fe_element = {"symbol": "Fe", "atomic_mass": iron.atomic_mass_amu, "density": 7.874, "melting_point": 1811}
    c_element = {"symbol": "C", "atomic_mass": carbon.atomic_mass_amu, "density": 2.26, "melting_point": 3823}

    carbon_steel = propagate_elements_to_alloy(
        elements=[fe_element, c_element],
        weight_fractions=[0.99, 0.01],
        lattice_type="BCC"
    )
    print(f"    Carbon Steel density: {carbon_steel.density_g_cm3:.2f} g/cm3")
    print(f"    Structure: {carbon_steel.primary_structure}")

    # Verify chain completeness
    chain_complete = (
        proton.charge_e == 1 and
        neutron.charge_e == 0 and
        iron.atomic_number == 26 and
        carbon.atomic_number == 6 and
        carbon_steel.density_g_cm3 > 0
    )

    summary.add_result(ChainTestResult(
        chain_name="FullChain", step="Integration", property_name="Complete",
        expected=True, actual=chain_complete, passed=chain_complete
    ))
    print(f"\n  Full chain complete: {chain_complete}")

    # Test full chain for Water molecule
    print("\n--- Full Chain: Building Water from Quarks ---")

    # Build hydrogen atom
    hydrogen = propagate_hadrons_to_atom(protons=1, neutrons=0, electrons=1)
    print(f"  Hydrogen atom: Z={hydrogen.atomic_number}, mass={hydrogen.atomic_mass_amu:.4f} amu")

    # Build oxygen atom
    oxygen = propagate_hadrons_to_atom(protons=8, neutrons=8, electrons=8)
    print(f"  Oxygen atom: Z={oxygen.atomic_number}, mass={oxygen.atomic_mass_amu:.4f} amu")

    # Build water molecule
    h_atom = {"symbol": "H", "atomic_mass": hydrogen.atomic_mass_amu, "atomic_number": 1}
    o_atom = {"symbol": "O", "atomic_mass": oxygen.atomic_mass_amu, "atomic_number": 8}
    h2o_bonds = [
        {"atom1": 0, "atom2": 2, "type": "single"},
        {"atom1": 1, "atom2": 2, "type": "single"}
    ]
    water = propagate_atoms_to_molecule([h_atom, h_atom, o_atom], h2o_bonds)

    print(f"  Water molecule: {water.formula}, mass={water.molecular_mass_amu:.3f} amu")

    # Verify water chain
    expected_water_mass = 2 * hydrogen.atomic_mass_amu + oxygen.atomic_mass_amu
    actual_water_mass = water.molecular_mass_amu
    water_mass_pass = abs(actual_water_mass - expected_water_mass) < 0.01

    summary.add_result(ChainTestResult(
        chain_name="WaterChain", step="Integration", property_name="MassConservation",
        expected=expected_water_mass, actual=actual_water_mass, passed=water_mass_pass
    ))
    print(f"  Mass conservation: Expected {expected_water_mass:.3f}, Got {actual_water_mass:.3f} {'PASS' if water_mass_pass else 'FAIL'}")


# =============================================================================
# Test: Compare with JSON Data
# =============================================================================

def test_compare_with_json(summary: ChainTestSummary):
    """Compare calculated values with stored JSON data."""
    print("\n" + "="*80)
    print("TEST 7: COMPARE WITH JSON DATA")
    print("="*80)

    # Load data from both default and root-level folders
    defaults_data = load_all_defaults()
    root_data = load_root_level_data()

    # Merge data
    subatomic_data = {**root_data.get('subatomic', {}), **defaults_data.get('subatomic', {})}
    elements_data = defaults_data.get('elements', {})
    molecules_data = {**root_data.get('molecules', {}), **defaults_data.get('molecules', {})}
    alloys_data = defaults_data.get('alloys', {})

    print(f"\nLoaded data: {len(subatomic_data)} subatomic, {len(elements_data)} elements, "
          f"{len(molecules_data)} molecules, {len(alloys_data)} alloys")

    # Test 1: Compare calculated proton with JSON proton
    print("\n--- Comparing Proton Calculation with JSON ---")
    if "Proton" in subatomic_data:
        json_proton = subatomic_data["Proton"]
        calc_proton = propagate_quark_to_hadron([UP_QUARK, UP_QUARK, DOWN_QUARK])

        json_mass = json_proton.get("Mass_MeVc2", 938.27)
        calc_mass = calc_proton.mass_MeV
        mass_error = abs(calc_mass - json_mass) / json_mass * 100
        mass_pass = mass_error < 10

        summary.add_result(ChainTestResult(
            chain_name="Proton", step="JSON Comparison", property_name="Mass",
            expected=json_mass, actual=calc_mass, passed=mass_pass,
            error_percent=mass_error
        ))
        print(f"  Mass: JSON={json_mass:.2f}, Calc={calc_mass:.2f}, Error={mass_error:.1f}% {'PASS' if mass_pass else 'FAIL'}")

        json_charge = json_proton.get("Charge_e", 1)
        calc_charge = calc_proton.charge_e
        charge_pass = abs(calc_charge - json_charge) < 0.01

        summary.add_result(ChainTestResult(
            chain_name="Proton", step="JSON Comparison", property_name="Charge",
            expected=json_charge, actual=calc_charge, passed=charge_pass
        ))
        print(f"  Charge: JSON={json_charge}, Calc={calc_charge} {'PASS' if charge_pass else 'FAIL'}")

    # Test 2: Compare calculated elements with JSON
    print("\n--- Comparing Element Calculations with JSON ---")
    test_elements = [1, 6, 26, 79]  # H, C, Fe, Au

    for z in test_elements:
        if z in elements_data:
            json_elem = elements_data[z]
            symbol = json_elem.get('symbol', f'E{z}')

            # Get most abundant isotope
            isotopes = json_elem.get('isotopes', [])
            if isotopes:
                most_abundant = max(isotopes, key=lambda x: x.get('abundance', 0))
                n = most_abundant.get('neutrons', z)
            else:
                n = z  # Approximate

            calc_elem = propagate_hadrons_to_atom(protons=z, neutrons=n, electrons=z)

            json_mass = json_elem.get('atomic_mass', 0)
            calc_mass = calc_elem.atomic_mass_amu

            if json_mass > 0:
                mass_error = abs(calc_mass - json_mass) / json_mass * 100
                mass_pass = mass_error < 2

                summary.add_result(ChainTestResult(
                    chain_name=f"{symbol} (Z={z})", step="JSON Comparison", property_name="AtomicMass",
                    expected=json_mass, actual=calc_mass, passed=mass_pass,
                    error_percent=mass_error
                ))
                print(f"  {symbol}: JSON mass={json_mass:.3f}, Calc mass={calc_mass:.3f}, Error={mass_error:.2f}% {'PASS' if mass_pass else 'FAIL'}")

    # Test 3: Compare molecules
    print("\n--- Comparing Molecule Calculations with JSON ---")
    test_molecules = ["Water", "Hydrogen", "Methane"]

    for mol_name in test_molecules:
        if mol_name in molecules_data:
            json_mol = molecules_data[mol_name]

            # Build molecule from composition
            composition = json_mol.get('Composition', [])
            if composition:
                # Get atom data for each element
                atoms = []
                for comp in composition:
                    elem_symbol = comp.get('Element', '')
                    count = comp.get('Count', 1)

                    # Find element data
                    elem_data = None
                    for z, elem in elements_data.items():
                        if elem.get('symbol') == elem_symbol:
                            elem_data = elem
                            break

                    if elem_data:
                        for _ in range(count):
                            atoms.append({
                                'symbol': elem_symbol,
                                'atomic_mass': elem_data.get('atomic_mass', 0),
                                'atomic_number': elem_data.get('atomic_number', 0)
                            })

                if atoms:
                    bonds = json_mol.get('Bonds', [])
                    calc_mol = propagate_atoms_to_molecule(atoms, bonds)

                    json_mass = json_mol.get('MolecularMass_amu', 0)
                    calc_mass = calc_mol.molecular_mass_amu

                    if json_mass > 0:
                        mass_error = abs(calc_mass - json_mass) / json_mass * 100
                        mass_pass = mass_error < 1

                        summary.add_result(ChainTestResult(
                            chain_name=mol_name, step="JSON Comparison", property_name="MolecularMass",
                            expected=json_mass, actual=calc_mass, passed=mass_pass,
                            error_percent=mass_error
                        ))
                        print(f"  {mol_name}: JSON mass={json_mass:.3f}, Calc mass={calc_mass:.3f}, Error={mass_error:.2f}% {'PASS' if mass_pass else 'FAIL'}")


# =============================================================================
# Test: Property Propagation Verification
# =============================================================================

def test_property_propagation(summary: ChainTestSummary):
    """Verify that properties are correctly propagated through the hierarchy."""
    print("\n" + "="*80)
    print("TEST 8: PROPERTY PROPAGATION VERIFICATION")
    print("="*80)

    # Test 1: Charge conservation through hadron formation
    print("\n--- Charge Conservation in Hadrons ---")

    test_cases = [
        ("Proton", [UP_QUARK, UP_QUARK, DOWN_QUARK], 1.0),
        ("Neutron", [UP_QUARK, DOWN_QUARK, DOWN_QUARK], 0.0),
        ("Delta++", [UP_QUARK, UP_QUARK, UP_QUARK], 2.0),
        ("Sigma-", [DOWN_QUARK, DOWN_QUARK, STRANGE_QUARK], -1.0),
        ("Pion+", [UP_QUARK, ANTI_DOWN_QUARK], 1.0),
        ("Kaon+", [UP_QUARK, ANTI_STRANGE_QUARK], 1.0),
    ]

    for name, quarks, expected_charge in test_cases:
        hadron = propagate_quark_to_hadron(quarks)
        input_charge = sum(q['Charge_e'] for q in quarks)
        output_charge = hadron.charge_e

        charge_conserved = abs(output_charge - input_charge) < 0.01
        summary.add_result(ChainTestResult(
            chain_name=name, step="PropagationTest", property_name="ChargeConservation",
            expected=input_charge, actual=output_charge, passed=charge_conserved
        ))
        print(f"  {name}: Input charge={input_charge:.3f}, Output charge={output_charge:.3f} {'CONSERVED' if charge_conserved else 'NOT CONSERVED'}")

    # Test 2: Baryon number conservation
    print("\n--- Baryon Number Conservation ---")

    for name, quarks, _ in test_cases:
        hadron = propagate_quark_to_hadron(quarks)
        input_baryon = sum(q['BaryonNumber_B'] for q in quarks)
        output_baryon = hadron.baryon_number

        baryon_conserved = abs(output_baryon - input_baryon) < 0.01
        summary.add_result(ChainTestResult(
            chain_name=name, step="PropagationTest", property_name="BaryonConservation",
            expected=input_baryon, actual=output_baryon, passed=baryon_conserved
        ))
        print(f"  {name}: Input B={input_baryon:.3f}, Output B={output_baryon:.3f} {'CONSERVED' if baryon_conserved else 'NOT CONSERVED'}")

    # Test 3: Mass-energy propagation in atoms
    print("\n--- Mass Propagation in Atoms ---")

    atom_cases = [
        ("H-1", 1, 0, 1.008),
        ("He-4", 2, 2, 4.003),
        ("C-12", 6, 6, 12.000),
        ("O-16", 8, 8, 15.999),
    ]

    for name, z, n, expected_mass in atom_cases:
        atom = propagate_hadrons_to_atom(protons=z, neutrons=n, electrons=z)

        # Check mass is close to expected
        mass_error = abs(atom.atomic_mass_amu - expected_mass) / expected_mass * 100
        mass_reasonable = mass_error < 2

        summary.add_result(ChainTestResult(
            chain_name=name, step="PropagationTest", property_name="MassPropagation",
            expected=expected_mass, actual=atom.atomic_mass_amu, passed=mass_reasonable,
            error_percent=mass_error
        ))
        print(f"  {name}: Expected mass={expected_mass:.3f}, Calc mass={atom.atomic_mass_amu:.3f}, Error={mass_error:.2f}%")

    # Test 4: Mass conservation in molecules
    print("\n--- Mass Conservation in Molecules ---")

    h_atom = {"symbol": "H", "atomic_mass": 1.008, "atomic_number": 1}
    o_atom = {"symbol": "O", "atomic_mass": 15.999, "atomic_number": 8}
    c_atom = {"symbol": "C", "atomic_mass": 12.011, "atomic_number": 6}

    mol_cases = [
        ("H2", [h_atom, h_atom], 2.016),
        ("H2O", [h_atom, h_atom, o_atom], 18.015),
        ("CO2", [c_atom, o_atom, o_atom], 44.009),
    ]

    for name, atoms, expected_mass in mol_cases:
        input_mass = sum(a['atomic_mass'] for a in atoms)
        molecule = propagate_atoms_to_molecule(atoms, [])
        output_mass = molecule.molecular_mass_amu

        mass_error = abs(output_mass - input_mass) / input_mass * 100
        mass_conserved = mass_error < 0.1

        summary.add_result(ChainTestResult(
            chain_name=name, step="PropagationTest", property_name="MolecularMassConservation",
            expected=input_mass, actual=output_mass, passed=mass_conserved,
            error_percent=mass_error
        ))
        print(f"  {name}: Input mass={input_mass:.3f}, Output mass={output_mass:.3f} {'CONSERVED' if mass_conserved else 'NOT CONSERVED'}")


# =============================================================================
# Main Test Runner
# =============================================================================

def print_summary(summary: ChainTestSummary):
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print(f"\nTotal Tests: {summary.total_tests}")
    print(f"Passed: {summary.passed_tests}")
    print(f"Failed: {summary.failed_tests}")
    print(f"Pass Rate: {summary.passed_tests / summary.total_tests * 100:.1f}%")

    # Group results by chain
    chains = defaultdict(list)
    for r in summary.results:
        chains[r.chain_name].append(r)

    print("\n--- Results by Chain ---")
    for chain_name, results in sorted(chains.items()):
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        status = "PASS" if passed == total else "FAIL"
        print(f"  {chain_name:20s}: {passed}/{total} passed [{status}]")

    # List all failures
    failures = [r for r in summary.results if not r.passed]
    if failures:
        print("\n--- Failed Tests ---")
        for r in failures:
            print(f"  {r.chain_name} - {r.step} - {r.property_name}:")
            print(f"    Expected: {r.expected}")
            print(f"    Actual: {r.actual}")
            if r.error_percent is not None:
                print(f"    Error: {r.error_percent:.2f}%")


def main():
    """Run all prediction chain tests."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " PREDICTION CHAIN TESTS: QUARKS TO ALLOYS ".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    summary = ChainTestSummary()

    # Run all test suites
    proton, neutron = test_quark_to_hadron(summary)
    hydrogen, carbon, iron = test_hadron_to_atom(summary, proton, neutron)
    h2, h2o = test_atom_to_molecule(summary, hydrogen, carbon)
    ss304, brass = test_elements_to_alloy(summary)
    test_edge_cases(summary)
    test_full_chain(summary)
    test_compare_with_json(summary)
    test_property_propagation(summary)

    # Print summary
    print_summary(summary)

    print("\n" + "#"*80)
    print("#" + " TEST COMPLETE ".center(78) + "#")
    print("#"*80 + "\n")

    # Return exit code based on test results
    return 0 if summary.failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
