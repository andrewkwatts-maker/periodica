"""
Comprehensive AtomCalculator Testing Suite

This module tests the AtomCalculator against all 118 elements, evaluates
novel atom combinations, and provides detailed accuracy analysis with
recommendations for improvements.

Author: Comprehensive Test Suite
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Add the project root to path

from periodica.utils.physics_calculator import AtomCalculator, PhysicsConstants


@dataclass
class PropertyError:
    """Stores error information for a single property prediction."""
    element_name: str
    atomic_number: int
    property_name: str
    actual_value: float
    predicted_value: float
    absolute_error: float
    percent_error: float
    block: str
    period: int
    group: Optional[int]


@dataclass
class ElementTestResult:
    """Stores complete test results for a single element."""
    element_name: str
    symbol: str
    atomic_number: int
    block: str
    period: int
    group: Optional[int]
    mass_number: int
    neutrons: int
    property_errors: Dict[str, PropertyError] = field(default_factory=dict)
    overall_error: float = 0.0


@dataclass
class NovelAtomResult:
    """Stores results for novel atom combinations."""
    protons: int
    neutrons: int
    description: str
    predicted_properties: Dict[str, Any]
    is_valid: bool
    stability_info: Tuple[bool, Optional[str]]
    notes: List[str] = field(default_factory=list)


class AtomCalculatorComprehensiveTest:
    """Comprehensive testing class for AtomCalculator."""

    ELEMENTS_DIR = Path(__file__).parent.parent / "src" / "periodica" / "data" / "active" / "elements"
    PROPERTIES_TO_TEST = [
        'atomic_mass',
        'ionization_energy',
        'density',
        'electronegativity',
        'atomic_radius'
    ]

    def __init__(self):
        self.element_results: List[ElementTestResult] = []
        self.novel_results: List[NovelAtomResult] = []
        self.property_stats: Dict[str, Dict] = {}
        self.block_stats: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
        self.period_stats: Dict[int, Dict] = defaultdict(lambda: defaultdict(list))

    def load_all_elements(self) -> List[Dict]:
        """Load all 118 element JSON files."""
        elements = []
        for json_file in sorted(self.ELEMENTS_DIR.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                elements.append(json.load(f))
        return elements

    def get_most_abundant_isotope(self, element: Dict) -> Tuple[int, int]:
        """Get protons and neutrons for the most abundant isotope."""
        protons = element['atomic_number']
        isotopes = element.get('isotopes', [])

        if not isotopes:
            # Estimate neutrons from atomic mass
            mass = element.get('atomic_mass', protons * 2)
            neutrons = round(mass) - protons
            return protons, neutrons

        # Find most abundant isotope
        most_abundant = max(isotopes, key=lambda x: x.get('abundance', 0))
        neutrons = most_abundant.get('neutrons', round(element.get('atomic_mass', protons * 2)) - protons)
        return protons, neutrons

    def calculate_error(self, actual: float, predicted: float) -> Tuple[float, float]:
        """Calculate absolute and percent error."""
        if actual is None or predicted is None:
            return float('inf'), float('inf')

        absolute_error = abs(predicted - actual)

        if actual != 0:
            percent_error = (absolute_error / abs(actual)) * 100
        else:
            percent_error = 100 if predicted != 0 else 0

        return absolute_error, percent_error

    def test_single_element(self, element: Dict) -> ElementTestResult:
        """Test calculator predictions for a single element."""
        protons, neutrons = self.get_most_abundant_isotope(element)

        result = ElementTestResult(
            element_name=element['name'],
            symbol=element['symbol'],
            atomic_number=element['atomic_number'],
            block=element.get('block', 'unknown'),
            period=element.get('period', 0),
            group=element.get('group'),
            mass_number=protons + neutrons,
            neutrons=neutrons
        )

        # Test each property
        total_percent_error = 0
        valid_properties = 0

        for prop in self.PROPERTIES_TO_TEST:
            actual = element.get(prop)

            # Skip if actual value is missing or zero (for properties that can be zero)
            if actual is None:
                continue
            if prop in ['electronegativity'] and actual == 0:
                continue  # Noble gases have 0 electronegativity

            # Get predicted value
            if prop == 'atomic_mass':
                predicted = AtomCalculator.calculate_atomic_mass(protons, neutrons)
            elif prop == 'ionization_energy':
                predicted = AtomCalculator.calculate_ionization_energy(protons)
            elif prop == 'density':
                predicted = AtomCalculator.calculate_density(protons, neutrons)
            elif prop == 'electronegativity':
                predicted = AtomCalculator.calculate_electronegativity(protons)
            elif prop == 'atomic_radius':
                predicted = AtomCalculator.calculate_atomic_radius(protons)
            else:
                continue

            abs_err, pct_err = self.calculate_error(actual, predicted)

            error_info = PropertyError(
                element_name=result.element_name,
                atomic_number=protons,
                property_name=prop,
                actual_value=actual,
                predicted_value=predicted,
                absolute_error=abs_err,
                percent_error=pct_err,
                block=result.block,
                period=result.period,
                group=result.group
            )
            result.property_errors[prop] = error_info

            if pct_err != float('inf'):
                total_percent_error += pct_err
                valid_properties += 1

                # Store for block and period statistics
                self.block_stats[result.block][prop].append(pct_err)
                self.period_stats[result.period][prop].append(pct_err)

        if valid_properties > 0:
            result.overall_error = total_percent_error / valid_properties

        return result

    def test_all_elements(self) -> None:
        """Test calculator against all 118 elements."""
        elements = self.load_all_elements()

        print(f"\n{'='*80}")
        print("TESTING ATOMCALCULATOR AGAINST ALL {0} ELEMENTS".format(len(elements)))
        print('='*80)

        for element in elements:
            result = self.test_single_element(element)
            self.element_results.append(result)

        # Calculate property statistics
        for prop in self.PROPERTIES_TO_TEST:
            errors = [r.property_errors.get(prop) for r in self.element_results if prop in r.property_errors]
            if errors:
                pct_errors = [e.percent_error for e in errors if e.percent_error != float('inf')]
                if pct_errors:
                    self.property_stats[prop] = {
                        'mean_error': sum(pct_errors) / len(pct_errors),
                        'max_error': max(pct_errors),
                        'min_error': min(pct_errors),
                        'median_error': sorted(pct_errors)[len(pct_errors)//2],
                        'std_dev': math.sqrt(sum((x - sum(pct_errors)/len(pct_errors))**2 for x in pct_errors) / len(pct_errors)) if len(pct_errors) > 1 else 0,
                        'count': len(pct_errors)
                    }

    def test_novel_combinations(self) -> None:
        """Test novel atom combinations that don't exist in nature."""
        print(f"\n{'='*80}")
        print("TESTING NOVEL ATOM COMBINATIONS")
        print('='*80)

        novel_tests = [
            # Unusual proton/neutron ratios
            (50, 40, "Tin-90: Neutron-deficient (normal ~69 neutrons)"),
            (50, 80, "Tin-130: Neutron-rich (normal ~69 neutrons)"),
            (26, 20, "Iron-46: Highly neutron-deficient"),
            (26, 40, "Iron-66: Neutron-rich iron"),
            (82, 100, "Lead-182: Extreme neutron-deficient lead"),
            (82, 150, "Lead-232: Neutron-rich lead"),

            # Superheavy elements (beyond 118)
            (119, 180, "Element 119 (Ununennium): First period 8 element"),
            (120, 182, "Element 120 (Unbinilium): Alkaline earth metal"),
            (126, 190, "Element 126: Predicted island of stability region"),
            (130, 200, "Element 130: Far superheavy"),
            (140, 220, "Element 140: Extreme superheavy"),
            (150, 250, "Element 150: Theoretical limit test"),

            # Edge cases
            (1, 0, "Protium: Standard hydrogen"),
            (1, 2, "Tritium: Radioactive hydrogen"),
            (1, 10, "Hydrogen with 10 neutrons: Extremely unstable"),
            (2, 0, "Helium-2: Diproton"),
            (0, 1, "Single neutron: Edge case"),
        ]

        for protons, neutrons, description in novel_tests:
            result = self.test_novel_atom(protons, neutrons, description)
            self.novel_results.append(result)

    def test_novel_atom(self, protons: int, neutrons: int, description: str) -> NovelAtomResult:
        """Test a single novel atom combination."""
        result = NovelAtomResult(
            protons=protons,
            neutrons=neutrons,
            description=description,
            predicted_properties={},
            is_valid=True,
            stability_info=(False, None),
            notes=[]
        )

        try:
            # Test atomic mass calculation
            mass = AtomCalculator.calculate_atomic_mass(protons, neutrons)
            result.predicted_properties['atomic_mass'] = mass

            # Validate mass is reasonable
            theoretical_min = protons * 0.9 + neutrons * 0.9  # Rough lower bound
            theoretical_max = protons * 1.1 + neutrons * 1.1 + 0.1 * (protons + neutrons)
            if mass < theoretical_min or mass > theoretical_max:
                result.notes.append(f"Mass {mass:.4f} may be unrealistic for A={protons+neutrons}")

            # Test other properties
            if protons > 0:
                result.predicted_properties['ionization_energy'] = AtomCalculator.calculate_ionization_energy(protons)
                result.predicted_properties['electronegativity'] = AtomCalculator.calculate_electronegativity(protons)
                result.predicted_properties['atomic_radius'] = AtomCalculator.calculate_atomic_radius(protons)
                result.predicted_properties['density'] = AtomCalculator.calculate_density(protons, neutrons)
                result.predicted_properties['block_period_group'] = AtomCalculator.get_block_period_group(protons)
                result.stability_info = AtomCalculator.determine_stability(protons, neutrons)

                # Validate predictions
                if result.predicted_properties['ionization_energy'] < 0:
                    result.notes.append("Negative ionization energy - invalid")
                    result.is_valid = False
                if result.predicted_properties['density'] < 0:
                    result.notes.append("Negative density - invalid")
                    result.is_valid = False
                if result.predicted_properties['atomic_radius'] < 0:
                    result.notes.append("Negative atomic radius - invalid")
                    result.is_valid = False

                # Check for reasonable superheavy element predictions
                if protons > 118:
                    result.notes.append(f"Superheavy element Z={protons} - extrapolated predictions")
                    if result.predicted_properties['ionization_energy'] > 20 or result.predicted_properties['ionization_energy'] < 3:
                        result.notes.append("IE may be unrealistic for superheavy element")

        except Exception as e:
            result.is_valid = False
            result.notes.append(f"Exception: {str(e)}")

        return result

    def print_overall_statistics(self) -> None:
        """Print overall accuracy statistics."""
        print(f"\n{'='*80}")
        print("OVERALL ACCURACY STATISTICS")
        print('='*80)

        print("\n## Property-wise Error Statistics (Percent Error)")
        print("-" * 70)
        print(f"{'Property':<25} {'Mean %':>10} {'Median %':>10} {'Max %':>10} {'Std Dev':>10}")
        print("-" * 70)

        for prop, stats in sorted(self.property_stats.items()):
            print(f"{prop:<25} {stats['mean_error']:>10.2f} {stats['median_error']:>10.2f} "
                  f"{stats['max_error']:>10.2f} {stats['std_dev']:>10.2f}")

        # Overall mean
        all_means = [s['mean_error'] for s in self.property_stats.values()]
        if all_means:
            print("-" * 70)
            print(f"{'OVERALL MEAN':<25} {sum(all_means)/len(all_means):>10.2f}")

    def print_worst_performers(self, n: int = 10) -> None:
        """Print the worst performing elements."""
        print(f"\n{'='*80}")
        print(f"TOP {n} WORST PERFORMING ELEMENTS (by average % error)")
        print('='*80)

        sorted_results = sorted(self.element_results, key=lambda x: x.overall_error, reverse=True)

        print(f"\n{'Rank':<6} {'Element':<15} {'Z':<5} {'Block':<8} {'Period':<8} "
              f"{'Avg Error %':<12}")
        print("-" * 70)

        for i, result in enumerate(sorted_results[:n], 1):
            print(f"{i:<6} {result.element_name:<15} {result.atomic_number:<5} "
                  f"{result.block:<8} {result.period:<8} {result.overall_error:>10.2f}%")

            # Print individual property errors
            for prop, error in result.property_errors.items():
                if error.percent_error > 20:  # Only show significant errors
                    print(f"       -> {prop}: actual={error.actual_value:.4f}, "
                          f"predicted={error.predicted_value:.4f}, error={error.percent_error:.1f}%")

    def print_worst_by_property(self, n: int = 5) -> None:
        """Print worst performers for each property."""
        print(f"\n{'='*80}")
        print(f"WORST PREDICTIONS BY PROPERTY (top {n} each)")
        print('='*80)

        for prop in self.PROPERTIES_TO_TEST:
            errors = [(r, r.property_errors.get(prop))
                      for r in self.element_results if prop in r.property_errors]
            errors = [(r, e) for r, e in errors if e.percent_error != float('inf')]

            if not errors:
                continue

            errors.sort(key=lambda x: x[1].percent_error, reverse=True)

            print(f"\n## {prop.upper().replace('_', ' ')}")
            print("-" * 70)
            print(f"{'Element':<15} {'Z':<5} {'Actual':<15} {'Predicted':<15} {'Error %':<10}")
            print("-" * 70)

            for result, error in errors[:n]:
                print(f"{result.element_name:<15} {result.atomic_number:<5} "
                      f"{error.actual_value:<15.4f} {error.predicted_value:<15.4f} "
                      f"{error.percent_error:>8.2f}%")

    def print_block_period_analysis(self) -> None:
        """Print error analysis by block and period."""
        print(f"\n{'='*80}")
        print("ERROR ANALYSIS BY BLOCK")
        print('='*80)

        print(f"\n{'Block':<10}", end='')
        for prop in self.PROPERTIES_TO_TEST:
            print(f"{prop[:10]:<12}", end='')
        print()
        print("-" * 70)

        for block in ['s', 'p', 'd', 'f']:
            print(f"{block:<10}", end='')
            for prop in self.PROPERTIES_TO_TEST:
                errors = self.block_stats[block].get(prop, [])
                if errors:
                    mean = sum(errors) / len(errors)
                    print(f"{mean:>10.2f}%", end='  ')
                else:
                    print(f"{'N/A':>10}", end='  ')
            print()

        print(f"\n{'='*80}")
        print("ERROR ANALYSIS BY PERIOD")
        print('='*80)

        print(f"\n{'Period':<10}", end='')
        for prop in self.PROPERTIES_TO_TEST:
            print(f"{prop[:10]:<12}", end='')
        print()
        print("-" * 70)

        for period in range(1, 8):
            print(f"{period:<10}", end='')
            for prop in self.PROPERTIES_TO_TEST:
                errors = self.period_stats[period].get(prop, [])
                if errors:
                    mean = sum(errors) / len(errors)
                    print(f"{mean:>10.2f}%", end='  ')
                else:
                    print(f"{'N/A':>10}", end='  ')
            print()

    def print_novel_results(self) -> None:
        """Print results of novel combination tests."""
        print(f"\n{'='*80}")
        print("NOVEL ATOM COMBINATION RESULTS")
        print('='*80)

        print("\n## Unusual Isotopes")
        print("-" * 80)

        for result in self.novel_results:
            if result.protons <= 118:
                status = "VALID" if result.is_valid else "INVALID"
                stable = "Stable" if result.stability_info[0] else f"Unstable ({result.stability_info[1]})"

                print(f"\n{result.description}")
                print(f"  Z={result.protons}, N={result.neutrons}, A={result.protons + result.neutrons}")
                print(f"  Status: {status} | Stability: {stable}")

                if 'atomic_mass' in result.predicted_properties:
                    print(f"  Predicted mass: {result.predicted_properties['atomic_mass']:.4f} u")
                if 'ionization_energy' in result.predicted_properties:
                    print(f"  Predicted IE: {result.predicted_properties['ionization_energy']:.3f} eV")
                if result.notes:
                    for note in result.notes:
                        print(f"  NOTE: {note}")

        print("\n## Superheavy Elements (Z > 118)")
        print("-" * 80)

        for result in self.novel_results:
            if result.protons > 118:
                status = "VALID" if result.is_valid else "INVALID"

                print(f"\n{result.description}")
                print(f"  Z={result.protons}, N={result.neutrons}, A={result.protons + result.neutrons}")
                print(f"  Status: {status}")

                if 'block_period_group' in result.predicted_properties:
                    block, period, group = result.predicted_properties['block_period_group']
                    print(f"  Predicted position: Block {block}, Period {period}, Group {group}")
                if 'atomic_mass' in result.predicted_properties:
                    print(f"  Predicted mass: {result.predicted_properties['atomic_mass']:.4f} u")
                if 'ionization_energy' in result.predicted_properties:
                    print(f"  Predicted IE: {result.predicted_properties['ionization_energy']:.3f} eV")
                if 'electronegativity' in result.predicted_properties:
                    print(f"  Predicted EN: {result.predicted_properties['electronegativity']:.2f}")
                if 'atomic_radius' in result.predicted_properties:
                    print(f"  Predicted radius: {result.predicted_properties['atomic_radius']} pm")
                if result.notes:
                    for note in result.notes:
                        print(f"  NOTE: {note}")

    def generate_recommendations(self) -> List[str]:
        """Generate specific recommendations for improving the calculator."""
        recommendations = []

        # Analyze atomic mass errors
        mass_errors = [(r, r.property_errors.get('atomic_mass'))
                       for r in self.element_results if 'atomic_mass' in r.property_errors]
        mass_errors = [(r, e) for r, e in mass_errors if e and e.percent_error != float('inf')]

        if mass_errors:
            mean_mass_error = sum(e.percent_error for _, e in mass_errors) / len(mass_errors)
            if mean_mass_error > 1:
                recommendations.append(
                    "ATOMIC MASS: Current semi-empirical mass formula has mean error of {:.2f}%.\n"
                    "  - Consider using updated Bethe-Weizsacker coefficients from recent literature\n"
                    "  - Current coefficients: a_v={:.2f}, a_s={:.2f}, a_c={:.3f}, a_a={:.2f}, a_p={:.2f}\n"
                    "  - Recommended: Implement shell correction terms for magic numbers (2,8,20,28,50,82,126)\n"
                    "  - Add Wigner term for N=Z nuclei".format(
                        mean_mass_error,
                        PhysicsConstants.BINDING_ENERGY_VOLUME,
                        PhysicsConstants.BINDING_ENERGY_SURFACE,
                        PhysicsConstants.BINDING_ENERGY_COULOMB,
                        PhysicsConstants.BINDING_ENERGY_ASYMMETRY,
                        PhysicsConstants.BINDING_ENERGY_PAIRING
                    )
                )

        # Analyze ionization energy errors
        ie_errors = [(r, r.property_errors.get('ionization_energy'))
                     for r in self.element_results if 'ionization_energy' in r.property_errors]
        ie_errors = [(r, e) for r, e in ie_errors if e and e.percent_error != float('inf')]

        if ie_errors:
            # Check which blocks have highest IE errors
            block_ie_errors = defaultdict(list)
            for r, e in ie_errors:
                block_ie_errors[r.block].append(e.percent_error)

            worst_block = max(block_ie_errors.items(), key=lambda x: sum(x[1])/len(x[1]))
            recommendations.append(
                "IONIZATION ENERGY: Worst predictions in {} block (mean error {:.2f}%).\n"
                "  - The empirical formulas use block-based corrections that may need refinement\n"
                "  - Consider implementing more accurate Slater's rules with penetration effects\n"
                "  - For d-block: Account for d-orbital penetration and exchange energy\n"
                "  - For f-block: Include relativistic corrections for heavy elements".format(
                    worst_block[0], sum(worst_block[1])/len(worst_block[1])
                )
            )

        # Analyze density errors
        density_errors = [(r, r.property_errors.get('density'))
                         for r in self.element_results if 'density' in r.property_errors]
        density_errors = [(r, e) for r, e in density_errors if e and e.percent_error != float('inf')]

        if density_errors:
            mean_density_error = sum(e.percent_error for _, e in density_errors) / len(density_errors)
            if mean_density_error > 20:
                recommendations.append(
                    "DENSITY: Mean error of {:.2f}% is high.\n"
                    "  - Density depends on crystal structure which varies widely\n"
                    "  - Current formula uses simple periodic trends without crystal packing\n"
                    "  - Consider: Add crystal structure classification (FCC, BCC, HCP, etc.)\n"
                    "  - Use atomic radius and molar volume correlation with structure type\n"
                    "  - For gases: Use ideal gas law at STP conditions".format(mean_density_error)
                )

        # Analyze electronegativity errors
        en_errors = [(r, r.property_errors.get('electronegativity'))
                     for r in self.element_results if 'electronegativity' in r.property_errors]
        en_errors = [(r, e) for r, e in en_errors if e and e.percent_error != float('inf') and e.actual_value != 0]

        if en_errors:
            mean_en_error = sum(e.percent_error for _, e in en_errors) / len(en_errors)
            if mean_en_error > 15:
                recommendations.append(
                    "ELECTRONEGATIVITY: Mean error of {:.2f}%.\n"
                    "  - Current empirical formula is based on periodic position\n"
                    "  - Consider implementing Mulliken electronegativity: chi = (IE + EA) / 2\n"
                    "  - Or Allred-Rochow: chi = 0.359 * Z_eff / r^2 + 0.744\n"
                    "  - These connect electronegativity to calculated IE and atomic radius".format(mean_en_error)
                )

        # Analyze atomic radius errors
        radius_errors = [(r, r.property_errors.get('atomic_radius'))
                        for r in self.element_results if 'atomic_radius' in r.property_errors]
        radius_errors = [(r, e) for r, e in radius_errors if e and e.percent_error != float('inf')]

        if radius_errors:
            mean_radius_error = sum(e.percent_error for _, e in radius_errors) / len(radius_errors)
            if mean_radius_error > 15:
                recommendations.append(
                    "ATOMIC RADIUS: Mean error of {:.2f}%.\n"
                    "  - Atomic radius definition varies (covalent, van der Waals, metallic)\n"
                    "  - Current formula uses simple periodic trends\n"
                    "  - Consider: r = r_0 * n^2 / Z_eff with better Z_eff calculation\n"
                    "  - Account for lanthanide contraction effect for periods 6-7\n"
                    "  - Use different formulas for different radius types".format(mean_radius_error)
                )

        # Check for superheavy element handling
        superheavy_issues = [r for r in self.novel_results if r.protons > 118 and not r.is_valid]
        if superheavy_issues:
            recommendations.append(
                "SUPERHEAVY ELEMENTS: Some predictions failed for Z > 118.\n"
                "  - Extend _get_block, _get_period, _get_group for periods 8+\n"
                "  - Add relativistic corrections to mass formula for Z > 100\n"
                "  - Consider island of stability predictions around Z=114, 120, 126"
            )

        # Pattern analysis
        recommendations.append(
            "\nGENERAL PATTERNS OBSERVED:\n"
            "  - s-block predictions are generally accurate (simple electronic structure)\n"
            "  - d-block has variable accuracy due to complex orbital interactions\n"
            "  - f-block (lanthanides/actinides) often has highest errors\n"
            "  - Period 7 elements often have largest errors (fewer experimental data)\n"
            "  - Noble gases have special handling that works well for most properties"
        )

        return recommendations

    def print_recommendations(self) -> None:
        """Print improvement recommendations."""
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS FOR IMPROVING ATOMCALCULATOR")
        print('='*80)

        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")

    def run_full_test_suite(self) -> None:
        """Run the complete test suite."""
        print("\n" + "=" * 80)
        print("ATOMCALCULATOR COMPREHENSIVE TEST SUITE")
        print("Testing predictions against all 118 elements + novel combinations")
        print("=" * 80)

        # Test all elements
        self.test_all_elements()

        # Test novel combinations
        self.test_novel_combinations()

        # Print all reports
        self.print_overall_statistics()
        self.print_worst_performers(10)
        self.print_worst_by_property(5)
        self.print_block_period_analysis()
        self.print_novel_results()
        self.print_recommendations()

        # Summary
        print(f"\n{'='*80}")
        print("TEST SUITE SUMMARY")
        print('='*80)

        total_elements = len(self.element_results)
        valid_novel = sum(1 for r in self.novel_results if r.is_valid)
        total_novel = len(self.novel_results)

        overall_accuracy = 100 - sum(r.overall_error for r in self.element_results) / total_elements

        print(f"\nElements tested: {total_elements}")
        print(f"Novel combinations tested: {total_novel} ({valid_novel} valid predictions)")
        print(f"Overall average accuracy: {overall_accuracy:.2f}%")

        # Best and worst properties
        if self.property_stats:
            best_prop = min(self.property_stats.items(), key=lambda x: x[1]['mean_error'])
            worst_prop = max(self.property_stats.items(), key=lambda x: x[1]['mean_error'])
            print(f"Best predicted property: {best_prop[0]} (mean error {best_prop[1]['mean_error']:.2f}%)")
            print(f"Worst predicted property: {worst_prop[0]} (mean error {worst_prop[1]['mean_error']:.2f}%)")

        print(f"\n{'='*80}")
        print("END OF TEST REPORT")
        print('='*80)


def test_atomic_mass_accuracy():
    """pytest-compatible test for atomic mass predictions."""
    tester = AtomCalculatorComprehensiveTest()
    elements = tester.load_all_elements()

    errors = []
    for element in elements:
        protons, neutrons = tester.get_most_abundant_isotope(element)
        actual = element.get('atomic_mass')
        predicted = AtomCalculator.calculate_atomic_mass(protons, neutrons)

        if actual:
            _, pct_error = tester.calculate_error(actual, predicted)
            errors.append(pct_error)

    mean_error = sum(errors) / len(errors) if errors else 0
    assert mean_error < 5, f"Atomic mass mean error {mean_error:.2f}% exceeds 5% threshold"


def test_ionization_energy_accuracy():
    """pytest-compatible test for ionization energy predictions."""
    tester = AtomCalculatorComprehensiveTest()
    elements = tester.load_all_elements()

    errors = []
    for element in elements:
        protons = element['atomic_number']
        actual = element.get('ionization_energy')
        predicted = AtomCalculator.calculate_ionization_energy(protons)

        if actual:
            _, pct_error = tester.calculate_error(actual, predicted)
            errors.append(pct_error)

    mean_error = sum(errors) / len(errors) if errors else 0
    assert mean_error < 30, f"Ionization energy mean error {mean_error:.2f}% exceeds 30% threshold"


def test_novel_atoms_valid():
    """pytest-compatible test for novel atom handling."""
    # Test superheavy element
    result = AtomCalculator.create_atom_from_particles(120, 182, 120)
    assert result is not None
    assert result['atomic_number'] == 120
    assert result['atomic_mass'] > 0

    # Test unusual isotope
    result = AtomCalculator.create_atom_from_particles(50, 80, 50)
    assert result is not None
    assert result['atomic_mass'] > 0


def test_edge_cases():
    """pytest-compatible test for edge cases."""
    # Single proton
    mass = AtomCalculator.calculate_atomic_mass(1, 0)
    assert 1.0 < mass < 1.1, f"Hydrogen mass {mass} out of expected range"

    # Heavy element
    mass = AtomCalculator.calculate_atomic_mass(92, 146)
    assert 230 < mass < 245, f"Uranium mass {mass} out of expected range"


if __name__ == "__main__":
    tester = AtomCalculatorComprehensiveTest()
    tester.run_full_test_suite()
