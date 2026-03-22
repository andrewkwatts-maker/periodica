#!/usr/bin/env python3
"""
Comprehensive Test Suite for MoleculeCalculator

This script tests the MoleculeCalculator against:
1. All 17 default molecules - comparing predicted vs actual values
2. Novel molecular combinations with unusual stoichiometry
3. Large organic molecules
4. Ionic compounds
5. Noble gas compounds

Outputs detailed error statistics and improvement recommendations.
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.physics_calculator import MoleculeCalculator


@dataclass
class MoleculeTestResult:
    """Stores test results for a single molecule."""
    name: str
    formula: str
    properties: Dict[str, Dict]  # property -> {actual, predicted, error_pct, error_abs}
    overall_error_pct: float
    geometry_match: bool
    polarity_match: bool


class MoleculeCalculatorTester:
    """Test suite for MoleculeCalculator accuracy assessment."""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "src" / "periodica" / "data" / "defaults" / "molecules"
        self.elements_dir = Path(__file__).parent.parent / "src" / "periodica" / "data" / "active" / "elements"
        self.molecules = {}
        self.results: List[MoleculeTestResult] = []
        self.novel_results: List[Dict] = []

    def load_all_molecules(self) -> Dict[str, Dict]:
        """Load all default molecule JSON files."""
        molecules = {}
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                molecules[data['Name']] = data
        return molecules

    def load_element_data(self) -> Dict[str, Dict]:
        """Load element data for atom properties."""
        elements = {}
        for json_file in self.elements_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                elements[data['symbol']] = data
        return elements

    def calculate_error(self, actual: float, predicted: float) -> Tuple[float, float]:
        """Calculate absolute and percentage error."""
        if actual == 0:
            if predicted == 0:
                return 0.0, 0.0
            return abs(predicted), 100.0
        error_abs = abs(actual - predicted)
        error_pct = (error_abs / abs(actual)) * 100
        return error_abs, error_pct

    def test_molecule(self, molecule_data: Dict) -> MoleculeTestResult:
        """Test calculator predictions against actual molecule data."""
        composition = molecule_data['Composition']
        name = molecule_data['Name']
        formula = molecule_data['Formula']

        # Get predicted values from calculator
        pred_mass = MoleculeCalculator.calculate_molecular_mass(composition)
        pred_geometry = MoleculeCalculator.estimate_geometry(composition)
        pred_bond_type = MoleculeCalculator.determine_bond_type(composition)
        pred_polarity = MoleculeCalculator.estimate_polarity(composition, pred_geometry)
        pred_mp = MoleculeCalculator.estimate_melting_point(pred_mass, pred_polarity, pred_bond_type, composition)
        pred_bp = MoleculeCalculator.estimate_boiling_point(pred_mp, pred_polarity, composition, pred_mass)
        pred_state = MoleculeCalculator.determine_state(pred_mp, pred_bp)
        pred_density = MoleculeCalculator.estimate_density(pred_mass, composition, pred_state, pred_bond_type)

        # Get actual values
        actual_mass = molecule_data.get('MolecularMass_amu', 0)
        actual_geometry = molecule_data.get('Geometry', '')
        actual_polarity = molecule_data.get('Polarity', '')
        actual_mp = molecule_data.get('MeltingPoint_K', 0)
        actual_bp = molecule_data.get('BoilingPoint_K', 0)
        actual_density = molecule_data.get('Density_g_cm3', 0)

        # Calculate errors for numeric properties
        properties = {}

        # Molecular mass
        mass_err_abs, mass_err_pct = self.calculate_error(actual_mass, pred_mass)
        properties['molecular_mass'] = {
            'actual': actual_mass,
            'predicted': pred_mass,
            'error_abs': mass_err_abs,
            'error_pct': mass_err_pct
        }

        # Melting point
        mp_err_abs, mp_err_pct = self.calculate_error(actual_mp, pred_mp)
        properties['melting_point'] = {
            'actual': actual_mp,
            'predicted': pred_mp,
            'error_abs': mp_err_abs,
            'error_pct': mp_err_pct
        }

        # Boiling point
        bp_err_abs, bp_err_pct = self.calculate_error(actual_bp, pred_bp)
        properties['boiling_point'] = {
            'actual': actual_bp,
            'predicted': pred_bp,
            'error_abs': bp_err_abs,
            'error_pct': bp_err_pct
        }

        # Density (handle gases with very low density)
        density_err_abs, density_err_pct = self.calculate_error(actual_density, pred_density)
        properties['density'] = {
            'actual': actual_density,
            'predicted': pred_density,
            'error_abs': density_err_abs,
            'error_pct': density_err_pct
        }

        # Geometry (categorical - compare strings)
        geometry_match = self._geometry_match(actual_geometry, pred_geometry)
        properties['geometry'] = {
            'actual': actual_geometry,
            'predicted': pred_geometry,
            'match': geometry_match
        }

        # Polarity (categorical)
        polarity_match = actual_polarity.lower() == pred_polarity.lower()
        properties['polarity'] = {
            'actual': actual_polarity,
            'predicted': pred_polarity,
            'match': polarity_match
        }

        # Calculate overall error (average of numeric properties)
        numeric_errors = [properties[p]['error_pct'] for p in ['molecular_mass', 'melting_point', 'boiling_point', 'density']]
        overall_error = sum(numeric_errors) / len(numeric_errors)

        return MoleculeTestResult(
            name=name,
            formula=formula,
            properties=properties,
            overall_error_pct=overall_error,
            geometry_match=geometry_match,
            polarity_match=polarity_match
        )

    def _geometry_match(self, actual: str, predicted: str) -> bool:
        """Check if geometry predictions match (with some flexibility)."""
        actual_lower = actual.lower()
        predicted_lower = predicted.lower()

        # Exact match
        if actual_lower == predicted_lower:
            return True

        # Handle special cases
        equivalents = {
            'planar hexagonal': ['trigonal planar', 'planar'],
            'chair conformation': ['tetrahedral', 'complex'],
            'face-centered cubic': ['complex', 'octahedral'],
        }

        for key, matches in equivalents.items():
            if actual_lower == key and predicted_lower in matches:
                return True
            if predicted_lower == key and actual_lower in matches:
                return True

        return False

    def test_novel_molecule(self, name: str, composition: List[Dict], description: str) -> Dict:
        """Test calculator on a novel/hypothetical molecule."""
        try:
            pred_mass = MoleculeCalculator.calculate_molecular_mass(composition)
            pred_geometry = MoleculeCalculator.estimate_geometry(composition)
            pred_bond_type = MoleculeCalculator.determine_bond_type(composition)
            pred_polarity = MoleculeCalculator.estimate_polarity(composition, pred_geometry)
            pred_mp = MoleculeCalculator.estimate_melting_point(pred_mass, pred_polarity, pred_bond_type, composition)
            pred_bp = MoleculeCalculator.estimate_boiling_point(pred_mp, pred_polarity, composition, pred_mass)
            pred_state = MoleculeCalculator.determine_state(pred_mp, pred_bp)
            pred_density = MoleculeCalculator.estimate_density(pred_mass, composition, pred_state, pred_bond_type)

            # Generate formula
            formula = MoleculeCalculator.generate_formula(composition)

            result = {
                'name': name,
                'formula': formula,
                'description': description,
                'composition': composition,
                'success': True,
                'predictions': {
                    'molecular_mass': pred_mass,
                    'geometry': pred_geometry,
                    'bond_type': pred_bond_type,
                    'polarity': pred_polarity,
                    'melting_point_K': pred_mp,
                    'boiling_point_K': pred_bp,
                    'state_STP': pred_state,
                    'density': pred_density
                },
                'physically_reasonable': self._check_physical_reasonability(pred_mp, pred_bp, pred_density, pred_mass)
            }
        except Exception as e:
            result = {
                'name': name,
                'formula': '',
                'description': description,
                'composition': composition,
                'success': False,
                'error': str(e),
                'physically_reasonable': False
            }

        return result

    def _check_physical_reasonability(self, mp: float, bp: float, density: float, mass: float) -> Dict:
        """Check if predicted values are physically reasonable."""
        issues = []

        # MP should be less than BP
        if mp > bp:
            issues.append("Melting point exceeds boiling point")

        # MP and BP should be positive
        if mp < 0:
            issues.append(f"Negative melting point ({mp} K)")
        if bp < 0:
            issues.append(f"Negative boiling point ({bp} K)")

        # Density should be reasonable
        if density < 0:
            issues.append(f"Negative density ({density})")
        if density > 25:  # Osmium is ~22.6 g/cm3
            issues.append(f"Unreasonably high density ({density})")

        # Mass should match composition
        # (This is generally accurate since it's calculated from atomic masses)

        # Temperature ranges for molecular substances
        if mass < 200:  # Small molecules
            if mp > 2000:
                issues.append(f"Melting point too high for small molecule ({mp} K)")
            if bp > 3000:
                issues.append(f"Boiling point too high for small molecule ({bp} K)")

        return {
            'is_reasonable': len(issues) == 0,
            'issues': issues
        }

    def run_all_tests(self):
        """Run complete test suite."""
        print("=" * 80)
        print("COMPREHENSIVE MOLECULECALCULATOR TEST SUITE")
        print("=" * 80)
        print()

        # Load molecules
        self.molecules = self.load_all_molecules()
        print(f"Loaded {len(self.molecules)} default molecules")
        print()

        # Test all default molecules
        print("-" * 80)
        print("SECTION 1: TESTING DEFAULT MOLECULES")
        print("-" * 80)

        for name, data in self.molecules.items():
            result = self.test_molecule(data)
            self.results.append(result)

        self._print_default_molecule_results()

        # Test novel molecules
        print()
        print("-" * 80)
        print("SECTION 2: TESTING NOVEL MOLECULAR COMBINATIONS")
        print("-" * 80)

        self._test_novel_molecules()
        self._print_novel_molecule_results()

        # Statistics and analysis
        print()
        print("-" * 80)
        print("SECTION 3: ERROR ANALYSIS AND STATISTICS")
        print("-" * 80)
        self._print_statistics()

        # Recommendations
        print()
        print("-" * 80)
        print("SECTION 4: IMPROVEMENT RECOMMENDATIONS")
        print("-" * 80)
        self._print_recommendations()

        return self.results, self.novel_results

    def _print_default_molecule_results(self):
        """Print detailed results for default molecules."""
        print()
        for result in self.results:
            print(f"\n{result.name} ({result.formula})")
            print("-" * 40)

            for prop_name, prop_data in result.properties.items():
                if prop_name in ['geometry', 'polarity']:
                    match_str = "MATCH" if prop_data['match'] else "MISMATCH"
                    print(f"  {prop_name:20s}: Actual={prop_data['actual']:20s} Predicted={prop_data['predicted']:20s} [{match_str}]")
                else:
                    print(f"  {prop_name:20s}: Actual={prop_data['actual']:12.4f} Predicted={prop_data['predicted']:12.4f} Error={prop_data['error_pct']:6.2f}%")

            print(f"  {'Overall Error':20s}: {result.overall_error_pct:.2f}%")

    def _test_novel_molecules(self):
        """Test calculator on novel molecular combinations."""

        # Category 1: Unusual stoichiometry
        unusual_stoichiometry = [
            ("Hydronium ion", [{"Element": "H", "Count": 3}, {"Element": "O", "Count": 1}], "H3O+ - Unusual H3O"),
            ("Propyne", [{"Element": "C", "Count": 3}, {"Element": "H", "Count": 4}], "C3H4 - Triple bond hydrocarbon"),
            ("Dinitrogen Pentoxide", [{"Element": "N", "Count": 2}, {"Element": "O", "Count": 5}], "N2O5 - Higher nitrogen oxide"),
            ("Hydrogen Peroxide", [{"Element": "H", "Count": 2}, {"Element": "O", "Count": 2}], "H2O2 - Peroxide"),
            ("Sulfur Trioxide", [{"Element": "S", "Count": 1}, {"Element": "O", "Count": 3}], "SO3 - Sulfur oxide"),
            ("Phosphoric Acid", [{"Element": "H", "Count": 3}, {"Element": "P", "Count": 1}, {"Element": "O", "Count": 4}], "H3PO4 - Triprotic acid"),
        ]

        # Category 2: Large organic molecules
        large_organics = [
            ("Decane", [{"Element": "C", "Count": 10}, {"Element": "H", "Count": 22}], "C10H22 - Long chain alkane"),
            ("Fructose", [{"Element": "C", "Count": 6}, {"Element": "H", "Count": 12}, {"Element": "O", "Count": 6}], "C6H12O6 - Sugar isomer of glucose"),
            ("Octane", [{"Element": "C", "Count": 8}, {"Element": "H", "Count": 18}], "C8H18 - Gasoline component"),
            ("Hexane", [{"Element": "C", "Count": 6}, {"Element": "H", "Count": 14}], "C6H14 - Common solvent"),
            ("Propanol", [{"Element": "C", "Count": 3}, {"Element": "H", "Count": 8}, {"Element": "O", "Count": 1}], "C3H7OH - Propyl alcohol"),
            ("Butanoic Acid", [{"Element": "C", "Count": 4}, {"Element": "H", "Count": 8}, {"Element": "O", "Count": 2}], "C4H8O2 - Butyric acid"),
        ]

        # Category 3: Ionic compounds
        ionic_compounds = [
            ("Calcium Carbonate", [{"Element": "Ca", "Count": 1}, {"Element": "C", "Count": 1}, {"Element": "O", "Count": 3}], "CaCO3 - Limestone/chalk"),
            ("Magnesium Sulfate", [{"Element": "Mg", "Count": 1}, {"Element": "S", "Count": 1}, {"Element": "O", "Count": 4}], "MgSO4 - Epsom salt"),
            ("Potassium Chloride", [{"Element": "K", "Count": 1}, {"Element": "Cl", "Count": 1}], "KCl - Salt substitute"),
            ("Calcium Chloride", [{"Element": "Ca", "Count": 1}, {"Element": "Cl", "Count": 2}], "CaCl2 - Desiccant"),
            ("Sodium Bicarbonate", [{"Element": "Na", "Count": 1}, {"Element": "H", "Count": 1}, {"Element": "C", "Count": 1}, {"Element": "O", "Count": 3}], "NaHCO3 - Baking soda"),
            ("Ammonium Nitrate", [{"Element": "N", "Count": 2}, {"Element": "H", "Count": 4}, {"Element": "O", "Count": 3}], "NH4NO3 - Fertilizer"),
        ]

        # Category 4: Noble gas compounds
        noble_gas_compounds = [
            ("Xenon Difluoride", [{"Element": "Xe", "Count": 1}, {"Element": "F", "Count": 2}], "XeF2 - Linear noble gas compound"),
            ("Xenon Tetrafluoride", [{"Element": "Xe", "Count": 1}, {"Element": "F", "Count": 4}], "XeF4 - Square planar compound"),
            ("Xenon Hexafluoride", [{"Element": "Xe", "Count": 1}, {"Element": "F", "Count": 6}], "XeF6 - Distorted octahedral"),
            ("Krypton Difluoride", [{"Element": "Kr", "Count": 1}, {"Element": "F", "Count": 2}], "KrF2 - Rare krypton compound"),
            ("Xenon Trioxide", [{"Element": "Xe", "Count": 1}, {"Element": "O", "Count": 3}], "XeO3 - Xenon oxide"),
        ]

        # Category 5: Complex inorganics
        complex_inorganics = [
            ("Phosphorus Pentachloride", [{"Element": "P", "Count": 1}, {"Element": "Cl", "Count": 5}], "PCl5 - Trigonal bipyramidal"),
            ("Sulfur Hexafluoride", [{"Element": "S", "Count": 1}, {"Element": "F", "Count": 6}], "SF6 - Octahedral gas"),
            ("Boron Trifluoride", [{"Element": "B", "Count": 1}, {"Element": "F", "Count": 3}], "BF3 - Trigonal planar"),
            ("Silicon Tetrachloride", [{"Element": "Si", "Count": 1}, {"Element": "Cl", "Count": 4}], "SiCl4 - Tetrahedral"),
            ("Nitrogen Trifluoride", [{"Element": "N", "Count": 1}, {"Element": "F", "Count": 3}], "NF3 - Trigonal pyramidal"),
        ]

        all_test_cases = [
            ("Unusual Stoichiometry", unusual_stoichiometry),
            ("Large Organic Molecules", large_organics),
            ("Ionic Compounds", ionic_compounds),
            ("Noble Gas Compounds", noble_gas_compounds),
            ("Complex Inorganics", complex_inorganics),
        ]

        for category_name, molecules in all_test_cases:
            print(f"\n{category_name}:")
            print("-" * 40)
            for name, composition, description in molecules:
                result = self.test_novel_molecule(name, composition, description)
                result['category'] = category_name
                self.novel_results.append(result)

    def _print_novel_molecule_results(self):
        """Print results for novel molecule tests."""
        current_category = None

        for result in self.novel_results:
            if result.get('category') != current_category:
                current_category = result.get('category')
                print(f"\n--- {current_category} ---")

            print(f"\n  {result['name']} ({result['formula']})")
            print(f"    Description: {result['description']}")

            if result['success']:
                pred = result['predictions']
                print(f"    Molecular Mass: {pred['molecular_mass']:.3f} g/mol")
                print(f"    Geometry: {pred['geometry']}")
                print(f"    Bond Type: {pred['bond_type']}")
                print(f"    Polarity: {pred['polarity']}")
                print(f"    Melting Point: {pred['melting_point_K']:.1f} K")
                print(f"    Boiling Point: {pred['boiling_point_K']:.1f} K")
                print(f"    State at STP: {pred['state_STP']}")
                print(f"    Density: {pred['density']:.6f} g/cm3")

                reasonability = result['physically_reasonable']
                if reasonability['is_reasonable']:
                    print(f"    Physical Reasonability: PASS")
                else:
                    print(f"    Physical Reasonability: FAIL")
                    for issue in reasonability['issues']:
                        print(f"      - {issue}")
            else:
                print(f"    ERROR: {result.get('error', 'Unknown error')}")

    def _print_statistics(self):
        """Print error statistics and analysis."""
        print("\n1. OVERALL ACCURACY STATISTICS")
        print("=" * 50)

        # Calculate aggregate statistics per property
        properties = ['molecular_mass', 'melting_point', 'boiling_point', 'density']

        for prop in properties:
            errors = [r.properties[prop]['error_pct'] for r in self.results]
            mean_error = sum(errors) / len(errors)
            max_error = max(errors)
            min_error = min(errors)

            # Find worst molecule for this property
            worst_idx = errors.index(max_error)
            worst_molecule = self.results[worst_idx].name

            print(f"\n  {prop.upper().replace('_', ' ')}:")
            print(f"    Mean Error: {mean_error:.2f}%")
            print(f"    Max Error:  {max_error:.2f}% ({worst_molecule})")
            print(f"    Min Error:  {min_error:.2f}%")

        # Geometry and polarity accuracy
        geometry_correct = sum(1 for r in self.results if r.geometry_match)
        polarity_correct = sum(1 for r in self.results if r.polarity_match)
        total = len(self.results)

        print(f"\n  GEOMETRY PREDICTION:")
        print(f"    Accuracy: {geometry_correct}/{total} ({100*geometry_correct/total:.1f}%)")

        print(f"\n  POLARITY PREDICTION:")
        print(f"    Accuracy: {polarity_correct}/{total} ({100*polarity_correct/total:.1f}%)")

        # Overall scores
        overall_errors = [r.overall_error_pct for r in self.results]
        print(f"\n  COMBINED NUMERIC ERROR:")
        print(f"    Mean: {sum(overall_errors)/len(overall_errors):.2f}%")
        print(f"    Max:  {max(overall_errors):.2f}%")

        print("\n2. WORST PERFORMING MOLECULES")
        print("=" * 50)

        # Sort by overall error
        sorted_results = sorted(self.results, key=lambda x: x.overall_error_pct, reverse=True)

        for i, result in enumerate(sorted_results[:5]):
            print(f"\n  #{i+1}: {result.name} ({result.formula})")
            print(f"      Overall Error: {result.overall_error_pct:.2f}%")

            # Find which property has highest error
            worst_prop = max(
                ['molecular_mass', 'melting_point', 'boiling_point', 'density'],
                key=lambda p: result.properties[p]['error_pct']
            )
            print(f"      Worst Property: {worst_prop} ({result.properties[worst_prop]['error_pct']:.2f}%)")
            print(f"        Actual: {result.properties[worst_prop]['actual']}")
            print(f"        Predicted: {result.properties[worst_prop]['predicted']}")

            if not result.geometry_match:
                print(f"      Geometry Mismatch: {result.properties['geometry']['actual']} vs {result.properties['geometry']['predicted']}")
            if not result.polarity_match:
                print(f"      Polarity Mismatch: {result.properties['polarity']['actual']} vs {result.properties['polarity']['predicted']}")

        print("\n3. NOVEL MOLECULE TEST SUMMARY")
        print("=" * 50)

        successful = sum(1 for r in self.novel_results if r['success'])
        reasonable = sum(1 for r in self.novel_results if r.get('success') and r.get('physically_reasonable', {}).get('is_reasonable', False))
        total_novel = len(self.novel_results)

        print(f"  Total Novel Molecules Tested: {total_novel}")
        print(f"  Successfully Calculated: {successful}/{total_novel} ({100*successful/total_novel:.1f}%)")
        print(f"  Physically Reasonable: {reasonable}/{total_novel} ({100*reasonable/total_novel:.1f}%)")

        # Count issues by category
        category_stats = {}
        for r in self.novel_results:
            cat = r.get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'total': 0, 'success': 0, 'reasonable': 0}
            category_stats[cat]['total'] += 1
            if r['success']:
                category_stats[cat]['success'] += 1
            if r.get('success') and r.get('physically_reasonable', {}).get('is_reasonable', False):
                category_stats[cat]['reasonable'] += 1

        print("\n  By Category:")
        for cat, stats in category_stats.items():
            print(f"    {cat}: {stats['reasonable']}/{stats['total']} reasonable")

    def _print_recommendations(self):
        """Print specific improvement recommendations."""
        print("""
1. MOLECULAR MASS CALCULATION
   Status: Generally accurate (uses atomic mass lookup)

   Recommendation: None needed - calculations are correct.

2. MELTING POINT PREDICTIONS
   Current Issues:
   - Uses simple empirical correlations
   - Does not account for crystal packing effects
   - Hydrogen bonding strength varies by molecule structure

   Recommendations:
   a) Implement group contribution methods (e.g., Joback method)
   b) Add specific corrections for:
      - Ring structures (benzene, cyclic compounds)
      - Carboxylic acids (dimer formation increases MP)
      - Symmetry effects (symmetric molecules pack better)
   c) Consider using machine learning model trained on large dataset

   Code Change Suggestion:
   ```python
   def estimate_melting_point_improved(cls, composition, molecular_mass, ...):
       # Add ring detection
       has_ring = cls._detect_ring_structure(composition)

       # Add carboxylic acid detection
       is_carboxylic = cls._is_carboxylic_acid(composition)

       # Add symmetry factor
       symmetry_factor = cls._calculate_symmetry(composition)

       base_mp = current_calculation()

       if has_ring:
           base_mp *= 1.2  # Ring structures increase MP
       if is_carboxylic:
           base_mp *= 1.3  # Dimer formation
       base_mp *= symmetry_factor

       return base_mp
   ```

3. BOILING POINT PREDICTIONS
   Current Issues:
   - BP/MP ratio is too simplistic
   - Does not properly model hydrogen bonding in liquid phase
   - Alcohols have very high BP relative to MP (ratio ~2.2)

   Recommendations:
   a) Use Clausius-Clapeyron based approach
   b) Calculate enthalpy of vaporization from molecular properties
   c) Account for:
      - Number of hydrogen bonding sites
      - Molecular surface area
      - Dispersion forces (polarizability)

   Code Change Suggestion:
   ```python
   def estimate_boiling_point_improved(cls, composition, molecular_mass, ...):
       # Calculate hydrogen bonding sites
       hb_donors = cls._count_hb_donors(composition)  # O-H, N-H
       hb_acceptors = cls._count_hb_acceptors(composition)  # O, N, F

       # Estimate enthalpy of vaporization
       delta_Hvap = 85 * molecular_mass**0.5  # Base Trouton
       delta_Hvap += hb_donors * 20  # kJ/mol per donor
       delta_Hvap += hb_acceptors * 10  # kJ/mol per acceptor

       # BP from Clausius-Clapeyron
       bp = delta_Hvap * 1000 / (8.314 * 10.5)  # Trouton constant

       return bp
   ```

4. GEOMETRY PREDICTIONS
   Current Issues:
   - VSEPR logic is incomplete for larger molecules
   - Does not handle resonance structures
   - Special geometries (aromatic, chair) not predicted

   Recommendations:
   a) Add functional group detection
   b) Implement proper VSEPR with lone pair counting
   c) Add special cases:
      - Aromatic rings -> Planar
      - Six-membered saturated rings -> Chair
      - Carboxylic acids -> Trigonal planar at COOH

   Code Change Suggestion:
   ```python
   def estimate_geometry_improved(cls, composition):
       # Detect aromatic systems
       if cls._is_aromatic(composition):
           return "Planar"

       # Detect chair conformations
       if cls._is_6_membered_ring(composition):
           return "Chair Conformation"

       # Fall back to VSEPR with proper lone pair counting
       central_atom = cls._identify_central_atom(composition)
       steric_number = cls._calculate_steric_number(central_atom, composition)
       lone_pairs = cls._count_lone_pairs(central_atom, composition)

       return VSEPR_TABLE[(steric_number, lone_pairs)]
   ```

5. POLARITY PREDICTIONS
   Current Issues:
   - Does not consider 3D geometry for dipole cancellation
   - Symmetric molecules with polar bonds not always identified

   Recommendations:
   a) Calculate molecular dipole as vector sum of bond dipoles
   b) Use 3D coordinates when available
   c) Add explicit rules for symmetric molecules

   Code Change Suggestion:
   ```python
   def estimate_polarity_improved(cls, composition, geometry):
       # Check for symmetric nonpolar cases
       symmetric_nonpolar = [
           "Linear" with 2 identical terminal atoms,
           "Tetrahedral" with 4 identical terminal atoms,
           "Octahedral" with 6 identical terminal atoms,
           "Trigonal Planar" with 3 identical terminal atoms
       ]

       if cls._is_symmetric_geometry(composition, geometry):
           return "Nonpolar"

       # Calculate net dipole
       bond_dipoles = cls._calculate_bond_dipoles(composition)
       net_dipole = cls._sum_dipole_vectors(bond_dipoles, geometry)

       if abs(net_dipole) < 0.4:
           return "Nonpolar"
       return "Polar"
   ```

6. DENSITY PREDICTIONS
   Current Issues:
   - Gas density calculation needs ideal gas law refinement
   - Liquid/solid density estimation is too empirical

   Recommendations:
   a) Use molar volume estimations (Rackett equation for liquids)
   b) For gases: use van der Waals or other equation of state
   c) For solids: use packing efficiency models

7. GENERAL IMPROVEMENTS
   a) Add a database of known molecular properties for common molecules
   b) Implement group contribution methods (Benson, Joback)
   c) Add temperature-dependent property calculations
   d) Consider using SMILES/InChI for structure representation
   e) Add uncertainty estimates for all predictions
""")


def main():
    """Run the comprehensive test suite."""
    tester = MoleculeCalculatorTester()
    results, novel_results = tester.run_all_tests()

    # Return summary statistics
    overall_errors = [r.overall_error_pct for r in results]
    geometry_accuracy = sum(1 for r in results if r.geometry_match) / len(results)
    polarity_accuracy = sum(1 for r in results if r.polarity_match) / len(results)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Mean Overall Error (default molecules): {sum(overall_errors)/len(overall_errors):.2f}%")
    print(f"Geometry Prediction Accuracy: {100*geometry_accuracy:.1f}%")
    print(f"Polarity Prediction Accuracy: {100*polarity_accuracy:.1f}%")
    print(f"Novel Molecules Successfully Processed: {sum(1 for r in novel_results if r['success'])}/{len(novel_results)}")

    return results, novel_results


if __name__ == "__main__":
    main()
