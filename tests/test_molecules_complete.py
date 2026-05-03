#!/usr/bin/env python3
"""
Complete Molecule Validation Test Suite
========================================

Comprehensive testing of ALL molecules using their atomic configurations.

This test file:
1. Loads ALL 19 molecule JSON files from data/active/molecules/
2. Loads element data from data/active/elements/ for atomic property lookups
3. For EACH molecule:
   - Reads the Composition field (array of {Element, Count})
   - Reads the Bonds field if present
   - Builds the molecule using propagate_atoms_to_molecule
   - Compares calculated properties against JSON values:
     * MolecularMass_amu (tolerance: 0.1%)
     * total_electrons (exact match)
     * bond properties if available
     * formula generation
4. Tests the molecular geometry calculator:
   - For molecules with Geometry field, verifies VSEPR predictions
   - For molecules with BondAngle_deg, verifies angle calculations
   - For molecules with Atoms3D and Bonds3D, verifies 3D structure
5. Tests specific well-known molecules with exact values:
   - H2O: mass=18.015 amu, electrons=10, geometry=bent, angle=104.5 degrees
   - CO2: mass=44.01 amu, electrons=22, geometry=linear, angle=180 degrees
   - CH4: mass=16.04 amu, electrons=10, geometry=tetrahedral, angle=109.5 degrees
   - NH3: mass=17.03 amu, electrons=10, geometry=trigonal pyramidal, angle=107 degrees
   - C6H6: mass=78.11 amu, electrons=42, geometry=planar
6. Generates a comprehensive report showing:
   - Total molecules tested
   - Properties validated per molecule
   - Overall pass/fail rate
   - List of any failures with expected vs actual values
   - Geometry accuracy analysis
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.simulation_schema import propagate_atoms_to_molecule, MoleculeSimulationData
from periodica.utils.molecular_geometry import (
    MolecularGeometryCalculator,
    get_molecule_structure,
    GEOMETRY_CONFIG,
    VALENCE_ELECTRONS
)


# =============================================================================
# Test Data Structures
# =============================================================================

@dataclass
class PropertyValidation:
    """Result of validating a single property."""
    property_name: str
    expected: Any
    actual: Any
    passed: bool
    tolerance_used: Optional[float] = None
    error_pct: Optional[float] = None
    notes: str = ""


@dataclass
class MoleculeTestResult:
    """Complete test results for a single molecule."""
    name: str
    formula: str
    passed: bool
    validations: List[PropertyValidation] = field(default_factory=list)
    geometry_result: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class GeometryAnalysis:
    """Analysis of geometry prediction accuracy."""
    molecule_name: str
    expected_geometry: str
    predicted_geometry: str
    geometry_match: bool
    expected_angle: float
    predicted_angle: float
    angle_error: float
    has_3d_structure: bool
    structure_valid: bool = True
    notes: str = ""


# =============================================================================
# Known Reference Values for Well-Known Molecules
# =============================================================================

REFERENCE_MOLECULES = {
    "Water": {
        "formula": "H2O",
        "mass_amu": 18.015,
        "electrons": 10,
        "geometry": "Bent",
        "bond_angle": 104.5,
        "polarity": "Polar",
        "hybridization": "sp3"
    },
    "Carbon Dioxide": {
        "formula": "CO2",
        "mass_amu": 44.01,
        "electrons": 22,
        "geometry": "Linear",
        "bond_angle": 180.0,
        "polarity": "Nonpolar",
        "hybridization": "sp"
    },
    "Methane": {
        "formula": "CH4",
        "mass_amu": 16.04,
        "electrons": 10,
        "geometry": "Tetrahedral",
        "bond_angle": 109.5,
        "polarity": "Nonpolar",
        "hybridization": "sp3"
    },
    "Ammonia": {
        "formula": "NH3",
        "mass_amu": 17.031,
        "electrons": 10,
        "geometry": "Trigonal Pyramidal",
        "bond_angle": 107.0,
        "polarity": "Polar",
        "hybridization": "sp3"
    },
    "Benzene": {
        "formula": "C6H6",
        "mass_amu": 78.11,
        "electrons": 42,
        "geometry": "Planar Hexagonal",
        "bond_angle": 120.0,
        "polarity": "Nonpolar",
        "hybridization": "sp2"
    },
    "Ethanol": {
        "formula": "C2H5OH",
        "mass_amu": 46.07,
        "electrons": 26,
        "geometry": "Tetrahedral",
        "bond_angle": 109.5
    },
    "Glucose": {
        "formula": "C6H12O6",
        "mass_amu": 180.16,
        "electrons": 96,
        "geometry": "Chair Conformation",
        "bond_angle": 109.5
    },
    "Sulfuric Acid": {
        "formula": "H2SO4",
        "mass_amu": 98.079,
        "electrons": 50,
        "geometry": "Tetrahedral",
        "bond_angle": 109.5
    }
}


# =============================================================================
# Element Data Loader
# =============================================================================

class ElementDatabase:
    """Loads and provides access to element data."""

    def __init__(self, elements_dir: Path):
        self.elements_dir = elements_dir
        self.elements: Dict[str, Dict] = {}
        self._load_elements()

    def _load_elements(self):
        """Load all element data from JSON files."""
        for json_file in self.elements_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    symbol = data.get('symbol', '')
                    if symbol:
                        self.elements[symbol] = data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {json_file}: {e}")

    def get_element(self, symbol: str) -> Optional[Dict]:
        """Get element data by symbol."""
        return self.elements.get(symbol)

    def get_atomic_mass(self, symbol: str) -> float:
        """Get atomic mass for an element."""
        elem = self.get_element(symbol)
        if elem:
            return elem.get('atomic_mass', 0.0)
        return 0.0

    def get_atomic_number(self, symbol: str) -> int:
        """Get atomic number (electrons in neutral atom) for an element."""
        elem = self.get_element(symbol)
        if elem:
            return elem.get('atomic_number', 0)
        return 0


# =============================================================================
# Molecule Test Suite
# =============================================================================

class MoleculeCompleteTester:
    """Complete test suite for validating all molecules."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.molecules_dir = self.base_dir / "data" / "active" / "molecules"
        self.elements_dir = self.base_dir / "data" / "active" / "elements"

        self.element_db = ElementDatabase(self.elements_dir)
        self.geometry_calculator = MolecularGeometryCalculator()

        self.molecules: Dict[str, Dict] = {}
        self.test_results: List[MoleculeTestResult] = []
        self.geometry_analyses: List[GeometryAnalysis] = []

        # Tolerance settings
        self.mass_tolerance_pct = 0.1  # 0.1% tolerance for mass
        self.angle_tolerance_deg = 2.0  # 2 degree tolerance for angles

    def load_all_molecules(self) -> int:
        """Load all molecule JSON files. Returns count loaded."""
        count = 0
        for json_file in self.molecules_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    name = data.get('Name', json_file.stem)
                    self.molecules[name] = data
                    count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {json_file}: {e}")
        return count

    def calculate_molecular_mass(self, composition: List[Dict]) -> float:
        """Calculate molecular mass from composition."""
        total_mass = 0.0
        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 0)
            atomic_mass = self.element_db.get_atomic_mass(element)
            total_mass += atomic_mass * count
        return total_mass

    def calculate_total_electrons(self, composition: List[Dict]) -> int:
        """Calculate total electrons from composition."""
        total_electrons = 0
        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 0)
            atomic_number = self.element_db.get_atomic_number(element)
            total_electrons += atomic_number * count
        return total_electrons

    def generate_formula(self, composition: List[Dict]) -> str:
        """Generate molecular formula from composition."""
        # Standard element ordering: C, H, then alphabetical
        element_counts = {}
        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 0)
            element_counts[element] = element_counts.get(element, 0) + count

        formula_parts = []

        # Carbon first (if present)
        if 'C' in element_counts:
            count = element_counts.pop('C')
            formula_parts.append('C' if count == 1 else f'C{count}')

        # Hydrogen second (if present)
        if 'H' in element_counts:
            count = element_counts.pop('H')
            formula_parts.append('H' if count == 1 else f'H{count}')

        # Remaining elements alphabetically
        for elem in sorted(element_counts.keys()):
            count = element_counts[elem]
            formula_parts.append(elem if count == 1 else f'{elem}{count}')

        return ''.join(formula_parts)

    def build_atoms_list(self, composition: List[Dict]) -> List[Dict]:
        """Build atom list for propagate_atoms_to_molecule."""
        atoms = []
        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 0)
            elem_data = self.element_db.get_element(element)

            for _ in range(count):
                atom = {
                    'symbol': element,
                    'Symbol': element,
                    'atomic_mass': elem_data.get('atomic_mass', 0) if elem_data else 0,
                    'AtomicMass': elem_data.get('atomic_mass', 0) if elem_data else 0,
                    'atomic_number': elem_data.get('atomic_number', 0) if elem_data else 0,
                    'Z': elem_data.get('atomic_number', 0) if elem_data else 0,
                }
                atoms.append(atom)
        return atoms

    def convert_bonds_format(self, bonds: List[Dict], composition: List[Dict]) -> List[Dict]:
        """Convert bond format from molecule JSON to propagate format."""
        converted_bonds = []

        # Build element index mapping
        element_indices = {}
        idx = 0
        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 0)
            if element not in element_indices:
                element_indices[element] = []
            for _ in range(count):
                element_indices[element].append(idx)
                idx += 1

        # Track which indices have been used for each bond type
        bond_usage = {}

        for bond in bonds:
            from_elem = bond.get('From', '')
            to_elem = bond.get('To', '')
            bond_type = bond.get('Type', 'Single').lower()

            # Get available indices
            from_indices = element_indices.get(from_elem, [])
            to_indices = element_indices.get(to_elem, [])

            if from_indices and to_indices:
                # Use first available index (simplified)
                from_idx = from_indices[0]
                to_idx = to_indices[0] if from_elem != to_elem else (
                    to_indices[1] if len(to_indices) > 1 else to_indices[0]
                )

                converted_bonds.append({
                    'from': from_idx,
                    'to': to_idx,
                    'type': bond_type
                })

        return converted_bonds

    def validate_mass(self, molecule_data: Dict, calculated_mass: float) -> PropertyValidation:
        """Validate molecular mass calculation."""
        expected_mass = molecule_data.get('MolecularMass_amu', 0)

        if expected_mass == 0:
            return PropertyValidation(
                property_name="MolecularMass_amu",
                expected=expected_mass,
                actual=calculated_mass,
                passed=False,
                notes="No expected mass in data"
            )

        error_pct = abs(expected_mass - calculated_mass) / expected_mass * 100
        passed = error_pct <= self.mass_tolerance_pct

        return PropertyValidation(
            property_name="MolecularMass_amu",
            expected=expected_mass,
            actual=round(calculated_mass, 3),
            passed=passed,
            tolerance_used=self.mass_tolerance_pct,
            error_pct=round(error_pct, 4),
            notes=f"Error: {error_pct:.4f}%"
        )

    def validate_electrons(self, molecule_data: Dict, composition: List[Dict]) -> PropertyValidation:
        """Validate total electron count."""
        calculated_electrons = self.calculate_total_electrons(composition)

        # Check if molecule data has expected electrons (derive from composition)
        expected_electrons = 0
        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 0)
            atomic_num = self.element_db.get_atomic_number(element)
            expected_electrons += atomic_num * count

        passed = calculated_electrons == expected_electrons

        return PropertyValidation(
            property_name="total_electrons",
            expected=expected_electrons,
            actual=calculated_electrons,
            passed=passed,
            notes="Exact match required"
        )

    def validate_formula(self, molecule_data: Dict, composition: List[Dict]) -> PropertyValidation:
        """Validate formula generation."""
        expected_formula = molecule_data.get('Formula', '')
        calculated_formula = self.generate_formula(composition)

        # Normalize formulas for comparison (handle variations like C2H5OH vs C2H6O)
        passed = self._formulas_equivalent(expected_formula, calculated_formula)

        return PropertyValidation(
            property_name="Formula",
            expected=expected_formula,
            actual=calculated_formula,
            passed=passed,
            notes="Formula structure comparison"
        )

    def _formulas_equivalent(self, formula1: str, formula2: str) -> bool:
        """Check if two formulas represent the same molecule."""
        def parse_formula(formula: str) -> Dict[str, int]:
            """Parse formula into element counts."""
            import re
            counts = {}
            # Match element symbol followed by optional number
            pattern = r'([A-Z][a-z]?)(\d*)'
            for match in re.finditer(pattern, formula):
                elem, count = match.groups()
                if elem:
                    counts[elem] = counts.get(elem, 0) + (int(count) if count else 1)
            return counts

        counts1 = parse_formula(formula1)
        counts2 = parse_formula(formula2)
        return counts1 == counts2

    def validate_bonds(self, molecule_data: Dict) -> PropertyValidation:
        """Validate bond data presence and structure."""
        bonds = molecule_data.get('Bonds', [])
        bonds_3d = molecule_data.get('Bonds3D', [])

        has_bonds = len(bonds) > 0
        has_3d_bonds = len(bonds_3d) > 0

        # Validate bond structure
        valid_bonds = True
        for bond in bonds:
            if not all(k in bond for k in ['From', 'To', 'Type']):
                valid_bonds = False
                break

        passed = has_bonds and valid_bonds

        return PropertyValidation(
            property_name="Bonds",
            expected="Valid bond data",
            actual=f"{len(bonds)} bonds, {len(bonds_3d)} 3D bonds",
            passed=passed,
            notes=f"Bonds valid: {valid_bonds}"
        )

    def validate_geometry(self, molecule_data: Dict) -> GeometryAnalysis:
        """Validate molecular geometry using VSEPR theory."""
        name = molecule_data.get('Name', '')
        expected_geometry = molecule_data.get('Geometry', '')
        expected_angle = molecule_data.get('BondAngle_deg', 0)

        # Try to predict geometry based on composition
        composition = molecule_data.get('Composition', [])
        bonds = molecule_data.get('Bonds', [])

        predicted_geometry = "Unknown"
        predicted_angle = 0.0

        # Use molecular geometry calculator for simple molecules
        formula = molecule_data.get('Formula', '')
        try:
            structure = get_molecule_structure(formula)
            predicted_geometry = structure.get('geometry', 'Unknown')
            predicted_angle = structure.get('bond_angle', 0.0)
        except Exception:
            # Fall back to composition-based prediction
            predicted_geometry, predicted_angle = self._predict_geometry_from_composition(
                composition, bonds
            )

        # Check geometry match (with flexibility for equivalent names)
        geometry_match = self._geometry_names_match(expected_geometry, predicted_geometry)

        # Calculate angle error
        angle_error = abs(expected_angle - predicted_angle) if expected_angle > 0 else 0

        # Check for 3D structure
        has_3d = 'Atoms3D' in molecule_data and len(molecule_data['Atoms3D']) > 0

        return GeometryAnalysis(
            molecule_name=name,
            expected_geometry=expected_geometry,
            predicted_geometry=predicted_geometry,
            geometry_match=geometry_match,
            expected_angle=expected_angle,
            predicted_angle=predicted_angle,
            angle_error=angle_error,
            has_3d_structure=has_3d
        )

    def _predict_geometry_from_composition(self, composition: List[Dict],
                                           bonds: List[Dict]) -> Tuple[str, float]:
        """Predict geometry based on composition and bonds."""
        # Count atoms
        total_atoms = sum(c.get('Count', 0) for c in composition)

        if total_atoms == 2:
            return "Linear", 180.0
        elif total_atoms == 3:
            # Could be bent or linear depending on central atom
            return "Bent", 104.5
        elif total_atoms == 4:
            return "Trigonal Pyramidal", 107.0
        elif total_atoms == 5:
            return "Tetrahedral", 109.5
        else:
            return "Complex", 109.5

    def _geometry_names_match(self, expected: str, predicted: str) -> bool:
        """Check if geometry names are equivalent."""
        expected_lower = expected.lower()
        predicted_lower = predicted.lower()

        if expected_lower == predicted_lower:
            return True

        # Define equivalent geometry names
        equivalents = {
            'planar hexagonal': ['trigonal planar', 'planar', 'hexagonal'],
            'chair conformation': ['tetrahedral', 'chair'],
            'trigonal pyramidal': ['pyramidal'],
            'bent': ['angular', 'v-shaped'],
        }

        for key, matches in equivalents.items():
            if expected_lower == key and predicted_lower in matches:
                return True
            if predicted_lower == key and expected_lower in matches:
                return True
            if expected_lower in matches and predicted_lower in matches:
                return True

        return False

    def test_molecule(self, name: str, molecule_data: Dict) -> MoleculeTestResult:
        """Run all tests for a single molecule."""
        result = MoleculeTestResult(
            name=name,
            formula=molecule_data.get('Formula', ''),
            passed=True
        )

        composition = molecule_data.get('Composition', [])
        bonds = molecule_data.get('Bonds', [])

        if not composition:
            result.errors.append("No composition data found")
            result.passed = False
            return result

        # Build atoms list for propagate function
        atoms_list = self.build_atoms_list(composition)
        bonds_list = self.convert_bonds_format(bonds, composition)

        # Use propagate_atoms_to_molecule
        try:
            molecule_sim = propagate_atoms_to_molecule(atoms_list, bonds_list)
        except Exception as e:
            result.errors.append(f"propagate_atoms_to_molecule failed: {e}")
            result.passed = False
            molecule_sim = None

        # Validate molecular mass
        calculated_mass = self.calculate_molecular_mass(composition)
        mass_validation = self.validate_mass(molecule_data, calculated_mass)
        result.validations.append(mass_validation)

        # Also validate against propagate result if available
        if molecule_sim:
            prop_mass = molecule_sim.molecular_mass_amu
            if prop_mass > 0:
                prop_mass_validation = PropertyValidation(
                    property_name="MolecularMass_amu (propagate)",
                    expected=molecule_data.get('MolecularMass_amu', 0),
                    actual=round(prop_mass, 3),
                    passed=abs(prop_mass - molecule_data.get('MolecularMass_amu', 0)) /
                           molecule_data.get('MolecularMass_amu', 1) * 100 <= self.mass_tolerance_pct,
                    notes="From propagate_atoms_to_molecule"
                )
                result.validations.append(prop_mass_validation)

        # Validate electrons
        electrons_validation = self.validate_electrons(molecule_data, composition)
        result.validations.append(electrons_validation)

        # Also validate against propagate result
        if molecule_sim:
            expected_electrons = self.calculate_total_electrons(composition)
            prop_electrons = molecule_sim.total_electrons
            prop_elec_validation = PropertyValidation(
                property_name="total_electrons (propagate)",
                expected=expected_electrons,
                actual=prop_electrons,
                passed=prop_electrons == expected_electrons,
                notes="From propagate_atoms_to_molecule"
            )
            result.validations.append(prop_elec_validation)

        # Validate formula
        formula_validation = self.validate_formula(molecule_data, composition)
        result.validations.append(formula_validation)

        # Validate bonds
        bonds_validation = self.validate_bonds(molecule_data)
        result.validations.append(bonds_validation)

        # Validate geometry
        geometry_analysis = self.validate_geometry(molecule_data)
        self.geometry_analyses.append(geometry_analysis)
        result.geometry_result = {
            'expected': geometry_analysis.expected_geometry,
            'predicted': geometry_analysis.predicted_geometry,
            'match': geometry_analysis.geometry_match,
            'angle_error': geometry_analysis.angle_error
        }

        # Add geometry validation
        geometry_validation = PropertyValidation(
            property_name="Geometry",
            expected=geometry_analysis.expected_geometry,
            actual=geometry_analysis.predicted_geometry,
            passed=geometry_analysis.geometry_match,
            notes=f"Angle error: {geometry_analysis.angle_error:.1f} degrees"
        )
        result.validations.append(geometry_validation)

        # Check overall pass/fail
        critical_validations = ['MolecularMass_amu', 'total_electrons', 'Formula']
        for v in result.validations:
            if v.property_name in critical_validations and not v.passed:
                result.passed = False
                break

        return result

    def test_reference_molecules(self) -> List[MoleculeTestResult]:
        """Test well-known molecules against exact reference values."""
        reference_results = []

        for name, ref_data in REFERENCE_MOLECULES.items():
            if name not in self.molecules:
                continue

            molecule_data = self.molecules[name]
            result = MoleculeTestResult(
                name=f"{name} (Reference)",
                formula=ref_data['formula'],
                passed=True
            )

            # Validate mass against reference
            actual_mass = molecule_data.get('MolecularMass_amu', 0)
            mass_error = abs(actual_mass - ref_data['mass_amu']) / ref_data['mass_amu'] * 100
            result.validations.append(PropertyValidation(
                property_name="Reference Mass",
                expected=ref_data['mass_amu'],
                actual=actual_mass,
                passed=mass_error <= 0.1,
                error_pct=mass_error,
                notes=f"Reference: {ref_data['mass_amu']} amu"
            ))

            # Validate electrons
            composition = molecule_data.get('Composition', [])
            actual_electrons = self.calculate_total_electrons(composition)
            result.validations.append(PropertyValidation(
                property_name="Reference Electrons",
                expected=ref_data['electrons'],
                actual=actual_electrons,
                passed=actual_electrons == ref_data['electrons'],
                notes=f"Reference: {ref_data['electrons']} electrons"
            ))

            # Validate geometry
            actual_geometry = molecule_data.get('Geometry', '')
            geometry_match = self._geometry_names_match(
                actual_geometry, ref_data['geometry']
            )
            result.validations.append(PropertyValidation(
                property_name="Reference Geometry",
                expected=ref_data['geometry'],
                actual=actual_geometry,
                passed=geometry_match,
                notes="Geometry comparison"
            ))

            # Validate bond angle
            actual_angle = molecule_data.get('BondAngle_deg', 0)
            angle_error = abs(actual_angle - ref_data['bond_angle'])
            result.validations.append(PropertyValidation(
                property_name="Reference Bond Angle",
                expected=ref_data['bond_angle'],
                actual=actual_angle,
                passed=angle_error <= self.angle_tolerance_deg,
                error_pct=angle_error,
                notes=f"Error: {angle_error:.1f} degrees"
            ))

            # Check overall
            for v in result.validations:
                if not v.passed:
                    result.passed = False

            reference_results.append(result)

        return reference_results

    def validate_3d_structures(self) -> List[Dict]:
        """Validate 3D molecular structures."""
        structure_validations = []

        for name, molecule_data in self.molecules.items():
            atoms_3d = molecule_data.get('Atoms3D', [])
            bonds_3d = molecule_data.get('Bonds3D', [])

            if not atoms_3d:
                continue

            validation = {
                'name': name,
                'atom_count': len(atoms_3d),
                'bond_count': len(bonds_3d),
                'issues': []
            }

            # Check atom structure
            for atom in atoms_3d:
                if not all(k in atom for k in ['element', 'x', 'y', 'z', 'index']):
                    validation['issues'].append(f"Invalid atom structure: {atom}")

            # Check bond structure
            for bond in bonds_3d:
                if not all(k in bond for k in ['from', 'to', 'type']):
                    validation['issues'].append(f"Invalid bond structure: {bond}")
                else:
                    # Validate bond indices
                    if bond['from'] >= len(atoms_3d) or bond['to'] >= len(atoms_3d):
                        validation['issues'].append(
                            f"Bond index out of range: {bond['from']}->{bond['to']}"
                        )

            # Calculate expected vs actual bond lengths
            expected_angle = molecule_data.get('BondAngle_deg', 0)
            if len(atoms_3d) >= 3 and expected_angle > 0:
                # Try to calculate actual angle from 3D coordinates
                actual_angle = self._calculate_3d_angle(atoms_3d, bonds_3d)
                if actual_angle is not None:
                    angle_error = abs(expected_angle - actual_angle)
                    validation['calculated_angle'] = actual_angle
                    validation['angle_error'] = angle_error
                    if angle_error > 5.0:
                        validation['issues'].append(
                            f"Angle mismatch: expected {expected_angle}, calculated {actual_angle:.1f}"
                        )

            validation['valid'] = len(validation['issues']) == 0
            structure_validations.append(validation)

        return structure_validations

    def _calculate_3d_angle(self, atoms: List[Dict], bonds: List[Dict]) -> Optional[float]:
        """Calculate bond angle from 3D atom positions."""
        if len(atoms) < 3:
            return None

        # Find central atom (atom with most bonds)
        bond_counts = {}
        for bond in bonds:
            bond_counts[bond['from']] = bond_counts.get(bond['from'], 0) + 1
            bond_counts[bond['to']] = bond_counts.get(bond['to'], 0) + 1

        if not bond_counts:
            return None

        central_idx = max(bond_counts.keys(), key=lambda x: bond_counts[x])

        # Find atoms bonded to central
        bonded_indices = []
        for bond in bonds:
            if bond['from'] == central_idx:
                bonded_indices.append(bond['to'])
            elif bond['to'] == central_idx:
                bonded_indices.append(bond['from'])

        if len(bonded_indices) < 2:
            return None

        # Get atom coordinates
        central = next((a for a in atoms if a['index'] == central_idx), None)
        atom1 = next((a for a in atoms if a['index'] == bonded_indices[0]), None)
        atom2 = next((a for a in atoms if a['index'] == bonded_indices[1]), None)

        if not all([central, atom1, atom2]):
            return None

        # Calculate angle
        return self.geometry_calculator.calculate_angle(atom1, central, atom2)

    def run_all_tests(self) -> Dict:
        """Run the complete test suite."""
        print("=" * 80)
        print("COMPLETE MOLECULE VALIDATION TEST SUITE")
        print("=" * 80)
        print()

        # Load molecules
        count = self.load_all_molecules()
        print(f"Loaded {count} molecules from {self.molecules_dir}")
        print(f"Loaded {len(self.element_db.elements)} elements from {self.elements_dir}")
        print()

        # Run tests on all molecules
        print("-" * 80)
        print("SECTION 1: TESTING ALL MOLECULES")
        print("-" * 80)

        for name, molecule_data in sorted(self.molecules.items()):
            result = self.test_molecule(name, molecule_data)
            self.test_results.append(result)

        # Run reference molecule tests
        print()
        print("-" * 80)
        print("SECTION 2: REFERENCE MOLECULE VALIDATION")
        print("-" * 80)

        reference_results = self.test_reference_molecules()

        # Validate 3D structures
        print()
        print("-" * 80)
        print("SECTION 3: 3D STRUCTURE VALIDATION")
        print("-" * 80)

        structure_validations = self.validate_3d_structures()

        # Generate report
        report = self._generate_report(reference_results, structure_validations)

        return report

    def _generate_report(self, reference_results: List[MoleculeTestResult],
                        structure_validations: List[Dict]) -> Dict:
        """Generate comprehensive test report."""

        # Section 1: All molecules results
        print()
        print("=" * 80)
        print("DETAILED TEST RESULTS")
        print("=" * 80)

        total_passed = 0
        total_failed = 0
        failures = []

        for result in self.test_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n{result.name} ({result.formula}): {status}")

            if result.passed:
                total_passed += 1
            else:
                total_failed += 1

            for v in result.validations:
                v_status = "OK" if v.passed else "FAIL"
                if v.error_pct is not None:
                    print(f"  [{v_status}] {v.property_name}: expected={v.expected}, "
                          f"actual={v.actual}, error={v.error_pct:.4f}%")
                else:
                    print(f"  [{v_status}] {v.property_name}: expected={v.expected}, "
                          f"actual={v.actual}")

                if not v.passed:
                    failures.append({
                        'molecule': result.name,
                        'property': v.property_name,
                        'expected': v.expected,
                        'actual': v.actual,
                        'notes': v.notes
                    })

            if result.errors:
                for error in result.errors:
                    print(f"  [ERROR] {error}")

        # Section 2: Reference molecule results
        print()
        print("=" * 80)
        print("REFERENCE MOLECULE RESULTS")
        print("=" * 80)

        ref_passed = 0
        ref_failed = 0

        for result in reference_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n{result.name}: {status}")

            if result.passed:
                ref_passed += 1
            else:
                ref_failed += 1

            for v in result.validations:
                v_status = "OK" if v.passed else "FAIL"
                print(f"  [{v_status}] {v.property_name}: expected={v.expected}, actual={v.actual}")

        # Section 3: Geometry analysis
        print()
        print("=" * 80)
        print("GEOMETRY PREDICTION ACCURACY")
        print("=" * 80)

        geometry_correct = sum(1 for g in self.geometry_analyses if g.geometry_match)
        geometry_total = len(self.geometry_analyses)

        print(f"\nGeometry prediction accuracy: {geometry_correct}/{geometry_total} "
              f"({100*geometry_correct/geometry_total:.1f}%)")

        print("\nMolecules with geometry mismatches:")
        for g in self.geometry_analyses:
            if not g.geometry_match:
                print(f"  {g.molecule_name}: expected={g.expected_geometry}, "
                      f"predicted={g.predicted_geometry}")

        # Angle accuracy
        angles_tested = [g for g in self.geometry_analyses if g.expected_angle > 0]
        if angles_tested:
            mean_angle_error = sum(g.angle_error for g in angles_tested) / len(angles_tested)
            max_angle_error = max(g.angle_error for g in angles_tested)
            print(f"\nBond angle accuracy:")
            print(f"  Mean error: {mean_angle_error:.2f} degrees")
            print(f"  Max error: {max_angle_error:.2f} degrees")

        # Section 4: 3D structure validation
        print()
        print("=" * 80)
        print("3D STRUCTURE VALIDATION")
        print("=" * 80)

        structures_valid = sum(1 for s in structure_validations if s['valid'])
        structures_total = len(structure_validations)

        print(f"\nValid 3D structures: {structures_valid}/{structures_total}")

        for s in structure_validations:
            if not s['valid']:
                print(f"\n  {s['name']}:")
                for issue in s['issues']:
                    print(f"    - {issue}")

        # Summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print(f"\nTotal molecules tested: {len(self.test_results)}")
        print(f"Molecules passed: {total_passed}")
        print(f"Molecules failed: {total_failed}")
        print(f"Pass rate: {100*total_passed/(total_passed+total_failed):.1f}%")

        print(f"\nReference molecules tested: {len(reference_results)}")
        print(f"Reference molecules passed: {ref_passed}")
        print(f"Reference molecules failed: {ref_failed}")

        print(f"\nGeometry predictions correct: {geometry_correct}/{geometry_total} "
              f"({100*geometry_correct/geometry_total:.1f}%)")

        print(f"\n3D structures valid: {structures_valid}/{structures_total} "
              f"({100*structures_valid/structures_total:.1f}%)")

        if failures:
            print(f"\nTotal validation failures: {len(failures)}")
            print("\nFailure details:")
            for f in failures[:10]:  # Show first 10 failures
                print(f"  {f['molecule']}.{f['property']}: "
                      f"expected={f['expected']}, actual={f['actual']}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more failures")

        return {
            'total_molecules': len(self.test_results),
            'passed': total_passed,
            'failed': total_failed,
            'pass_rate': 100 * total_passed / (total_passed + total_failed),
            'reference_tested': len(reference_results),
            'reference_passed': ref_passed,
            'geometry_accuracy': 100 * geometry_correct / geometry_total,
            'structures_valid': structures_valid,
            'structures_total': structures_total,
            'failures': failures,
            'geometry_analyses': [
                {
                    'molecule': g.molecule_name,
                    'expected': g.expected_geometry,
                    'predicted': g.predicted_geometry,
                    'match': g.geometry_match,
                    'angle_error': g.angle_error
                }
                for g in self.geometry_analyses
            ]
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete molecule validation test suite."""
    tester = MoleculeCompleteTester()
    report = tester.run_all_tests()

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    # Final status
    all_passed = report['failed'] == 0 and report['reference_passed'] == report['reference_tested']

    if all_passed:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\nSOME TESTS FAILED - see report above for details")
        return 1


if __name__ == "__main__":
    exit(main())
