#!/usr/bin/env python3
"""
Comprehensive Accuracy Test for Data-Driven Physics Calculators
================================================================

Tests SubatomicCalculatorV2, AtomCalculatorV2, and MoleculeCalculatorV2
against all available default data files.

Outputs detailed comparison tables, accuracy metrics, and improvement suggestions.
"""

import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.physics_calculator_v2 import (
    SubatomicCalculatorV2,
    AtomCalculatorV2,
    MoleculeCalculatorV2,
    PhysicsConstantsV2
)


# ==================== Data Loaders ====================

class DataLoader:
    """Load JSON data files from defaults directories."""

    BASE_PATH = Path(__file__).parent.parent / "src" / "periodica" / "data" / "defaults"

    @classmethod
    def load_json_file(cls, filepath: Path) -> Optional[Dict]:
        """Load a single JSON file, handling comments."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Remove JavaScript-style comments
                lines = []
                for line in content.split('\n'):
                    comment_idx = line.find('//')
                    if comment_idx != -1:
                        # Check if it's inside a string
                        quote_count = line[:comment_idx].count('"')
                        if quote_count % 2 == 0:  # Not inside string
                            line = line[:comment_idx]
                    lines.append(line)
                content = '\n'.join(lines)
                return json.loads(content)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    @classmethod
    def load_quarks(cls) -> Dict[str, Dict]:
        """Load all quark data files."""
        quarks = {}
        quark_path = cls.BASE_PATH / "quarks"
        for file in quark_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data and "Quark" in data.get("Name", ""):
                quarks[data["Name"]] = data
        return quarks

    @classmethod
    def load_antiquarks(cls) -> Dict[str, Dict]:
        """Load antiquark data files (needed for mesons)."""
        antiquarks = {}
        # First check quarks folder for anti-quark-like data
        quark_path = cls.BASE_PATH / "quarks"
        for file in quark_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                # Store all for potential reference
                name = data.get("Name", "")
                antiquarks[name] = data

        # Also check antiquarks folder
        antiquark_path = cls.BASE_PATH / "antiquarks"
        if antiquark_path.exists():
            for file in antiquark_path.glob("*.json"):
                data = cls.load_json_file(file)
                if data:
                    antiquarks[data.get("Name", file.stem)] = data
        return antiquarks

    @classmethod
    def load_subatomic_particles(cls) -> Dict[str, Dict]:
        """Load all subatomic particle data files."""
        particles = {}
        subatomic_path = cls.BASE_PATH / "subatomic"
        for file in subatomic_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                particles[data.get("Name", file.stem)] = data
        return particles

    @classmethod
    def load_special_particles(cls) -> Dict[str, Dict]:
        """Load special particles (proton, neutron, electron) from quarks folder."""
        particles = {}
        quark_path = cls.BASE_PATH / "quarks"
        for name in ["Proton", "Neutron", "Electron"]:
            file = quark_path / f"{name}.json"
            if file.exists():
                data = cls.load_json_file(file)
                if data:
                    particles[name] = data

        # Also check subatomic folder
        subatomic_path = cls.BASE_PATH / "subatomic"
        for name in ["Proton", "Neutron", "Electron"]:
            file = subatomic_path / f"{name}.json"
            if file.exists():
                data = cls.load_json_file(file)
                if data:
                    particles[name] = data
        return particles

    @classmethod
    def load_elements(cls) -> Dict[int, Dict]:
        """Load all element data files, keyed by atomic number."""
        elements = {}
        element_path = cls.BASE_PATH / "elements"
        for file in sorted(element_path.glob("*.json")):
            data = cls.load_json_file(file)
            if data and "atomic_number" in data:
                elements[data["atomic_number"]] = data
        return elements

    @classmethod
    def load_molecules(cls) -> Dict[str, Dict]:
        """Load all molecule data files."""
        molecules = {}
        molecule_path = cls.BASE_PATH / "molecules"
        for file in molecule_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                molecules[data.get("Name", file.stem)] = data
        return molecules


# ==================== Result Tracking ====================

@dataclass
class PropertyComparison:
    """Single property comparison result."""
    property_name: str
    predicted: float
    actual: float
    error_percent: float
    error_absolute: float


@dataclass
class ParticleTestResult:
    """Test result for a single particle/element/molecule."""
    name: str
    comparisons: List[PropertyComparison] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_comparison(self, prop_name: str, predicted: Any, actual: Any):
        """Add a property comparison."""
        try:
            pred_val = float(predicted) if predicted is not None else 0
            actual_val = float(actual) if actual is not None else 0

            if actual_val != 0:
                error_pct = abs((pred_val - actual_val) / actual_val) * 100
            elif pred_val != 0:
                error_pct = 100  # Predicted something when actual is 0
            else:
                error_pct = 0  # Both zero - this is correct, not an error

            error_abs = abs(pred_val - actual_val)

            self.comparisons.append(PropertyComparison(
                property_name=prop_name,
                predicted=pred_val,
                actual=actual_val,
                error_percent=error_pct,
                error_absolute=error_abs
            ))
        except (TypeError, ValueError) as e:
            self.errors.append(f"Cannot compare {prop_name}: {e}")

    def add_match_comparison(self, prop_name: str, matched: bool):
        """Add a boolean match comparison (1 for match, 0 for no match)."""
        self.comparisons.append(PropertyComparison(
            property_name=prop_name,
            predicted=1.0 if matched else 0.0,
            actual=1.0,
            error_percent=0.0 if matched else 100.0,
            error_absolute=0.0 if matched else 1.0
        ))


@dataclass
class CalculatorTestResults:
    """Aggregate results for a calculator."""
    calculator_name: str
    particle_results: List[ParticleTestResult] = field(default_factory=list)
    property_errors: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def add_result(self, result: ParticleTestResult):
        """Add a particle test result."""
        self.particle_results.append(result)
        for comp in result.comparisons:
            self.property_errors[comp.property_name].append(comp.error_percent)

    def get_mean_error(self, property_name: str) -> float:
        """Get mean error for a property."""
        errors = self.property_errors.get(property_name, [])
        return sum(errors) / len(errors) if errors else 0

    def get_worst_predictions(self, n: int = 5) -> List[Tuple[str, str, float]]:
        """Get n worst predictions (particle, property, error)."""
        all_errors = []
        for result in self.particle_results:
            for comp in result.comparisons:
                all_errors.append((result.name, comp.property_name, comp.error_percent,
                                  comp.predicted, comp.actual))

        all_errors.sort(key=lambda x: x[2], reverse=True)
        return all_errors[:n]

    def print_summary(self):
        """Print summary statistics."""
        print(f"\n{'='*70}")
        print(f" {self.calculator_name} - ACCURACY SUMMARY")
        print(f"{'='*70}")

        print(f"\nTotal items tested: {len(self.particle_results)}")

        print("\nMean Error % by Property:")
        print("-" * 50)
        for prop, errors in sorted(self.property_errors.items()):
            mean_err = self.get_mean_error(prop)
            median_err = sorted(errors)[len(errors)//2] if errors else 0
            max_err = max(errors) if errors else 0
            print(f"  {prop:30s}: Mean={mean_err:8.2f}%  Median={median_err:8.2f}%  Max={max_err:8.2f}%")

        print(f"\nTop 5 Worst Predictions:")
        print("-" * 70)
        for name, prop, error, pred, actual in self.get_worst_predictions(5):
            print(f"  {name:20s} | {prop:20s} | Error: {error:8.2f}%")
            print(f"    Predicted: {pred:.4f}, Actual: {actual:.4f}")


# ==================== Subatomic Calculator Tests ====================

class SubatomicCalculatorTester:
    """Test SubatomicCalculatorV2 against known particles."""

    # Quark composition for known particles
    PARTICLE_COMPOSITIONS = {
        "Proton": {"quarks": ["Up Quark", "Up Quark", "Down Quark"]},
        "Neutron": {"quarks": ["Up Quark", "Down Quark", "Down Quark"]},
        "Delta++": {"quarks": ["Up Quark", "Up Quark", "Up Quark"]},
        "Delta+": {"quarks": ["Up Quark", "Up Quark", "Down Quark"]},
        "DeltaPlus": {"quarks": ["Up Quark", "Up Quark", "Down Quark"]},
        "Delta0": {"quarks": ["Up Quark", "Down Quark", "Down Quark"]},
        "DeltaZero": {"quarks": ["Up Quark", "Down Quark", "Down Quark"]},
        "Delta-": {"quarks": ["Down Quark", "Down Quark", "Down Quark"]},
        "DeltaMinus": {"quarks": ["Down Quark", "Down Quark", "Down Quark"]},
        "DeltaPlusPlus": {"quarks": ["Up Quark", "Up Quark", "Up Quark"]},
        "Lambda": {"quarks": ["Up Quark", "Down Quark", "Strange Quark"]},
        "Sigma+": {"quarks": ["Up Quark", "Up Quark", "Strange Quark"]},
        "SigmaPlus": {"quarks": ["Up Quark", "Up Quark", "Strange Quark"]},
        "Sigma0": {"quarks": ["Up Quark", "Down Quark", "Strange Quark"]},
        "SigmaZero": {"quarks": ["Up Quark", "Down Quark", "Strange Quark"]},
        "Sigma-": {"quarks": ["Down Quark", "Down Quark", "Strange Quark"]},
        "SigmaMinus": {"quarks": ["Down Quark", "Down Quark", "Strange Quark"]},
        "Xi0": {"quarks": ["Up Quark", "Strange Quark", "Strange Quark"]},
        "Xi_Zero": {"quarks": ["Up Quark", "Strange Quark", "Strange Quark"]},
        "Xi-": {"quarks": ["Down Quark", "Strange Quark", "Strange Quark"]},
        "Xi_Minus": {"quarks": ["Down Quark", "Strange Quark", "Strange Quark"]},
        "Omega-": {"quarks": ["Strange Quark", "Strange Quark", "Strange Quark"]},
        "Omega_Minus": {"quarks": ["Strange Quark", "Strange Quark", "Strange Quark"]},
    }

    def __init__(self):
        self.quarks = DataLoader.load_quarks()
        self.particles = DataLoader.load_subatomic_particles()
        self.results = CalculatorTestResults("SubatomicCalculatorV2")

    def create_antiquark(self, quark_data: Dict) -> Dict:
        """Create an antiquark from quark data by flipping quantum numbers."""
        anti = quark_data.copy()
        anti["Name"] = "Anti-" + quark_data.get("Name", "Quark")
        anti["Symbol"] = quark_data.get("Symbol", "q") + "-bar"
        anti["Charge_e"] = -quark_data.get("Charge_e", 0)
        anti["BaryonNumber_B"] = -quark_data.get("BaryonNumber_B", 0)
        anti["Isospin_I3"] = -quark_data.get("Isospin_I3", 0)
        return anti

    def run_all_tests(self):
        """Run all subatomic particle tests."""
        print("\n" + "="*70)
        print(" SUBATOMIC CALCULATOR V2 TESTS")
        print("="*70)

        # Test baryons
        self._test_baryons()

        # Test mesons (pions, kaons) - more complex due to antiquarks
        self._test_mesons()

        return self.results

    def _test_baryons(self):
        """Test baryon creation from quarks."""
        print("\n--- Testing Baryons ---")

        for particle_name, actual_data in self.particles.items():
            # Check if this is a baryon we can test
            if actual_data.get("BaryonNumber_B", 0) != 1:
                continue

            composition = actual_data.get("Composition", [])
            if not composition:
                continue

            # Try to reconstruct quark list
            quark_list = []
            can_test = True

            for comp in composition:
                constituent = comp.get("Constituent", "")
                count = comp.get("Count", 0)

                # Check if it's an antiquark (skip for now - just baryons)
                if "Anti" in constituent or "anti" in constituent:
                    can_test = False
                    break

                # Find matching quark
                quark_data = None
                for qname, qdata in self.quarks.items():
                    if qname.lower().replace(" ", "") == constituent.lower().replace(" ", ""):
                        quark_data = qdata
                        break
                    if constituent.lower() in qname.lower():
                        quark_data = qdata
                        break

                if quark_data is None:
                    can_test = False
                    break

                for _ in range(count):
                    quark_list.append(quark_data)

            if not can_test or len(quark_list) != 3:
                continue

            # Create particle from quarks
            try:
                predicted = SubatomicCalculatorV2.create_particle_from_quarks(
                    quark_list, particle_name, actual_data.get("Symbol", "X")
                )

                result = ParticleTestResult(name=particle_name)

                # Compare properties
                result.add_comparison("Charge_e", predicted.get("Charge_e"), actual_data.get("Charge_e"))
                result.add_comparison("Mass_MeVc2", predicted.get("Mass_MeVc2"), actual_data.get("Mass_MeVc2"))
                result.add_comparison("Spin_hbar", predicted.get("Spin_hbar"), actual_data.get("Spin_hbar"))
                result.add_comparison("BaryonNumber_B", predicted.get("BaryonNumber_B"), actual_data.get("BaryonNumber_B"))
                result.add_comparison("Isospin_I3", predicted.get("Isospin_I3"), actual_data.get("Isospin_I3"))

                self.results.add_result(result)

                print(f"  {particle_name:20s}: Mass error = {result.comparisons[1].error_percent:6.2f}%")

            except Exception as e:
                print(f"  {particle_name:20s}: ERROR - {e}")

    def _test_mesons(self):
        """Test meson creation from quark-antiquark pairs."""
        print("\n--- Testing Mesons ---")

        for particle_name, actual_data in self.particles.items():
            # Check if this is a meson (baryon number = 0, has composition)
            if actual_data.get("BaryonNumber_B", 0) != 0:
                continue

            composition = actual_data.get("Composition", [])
            if len(composition) != 2:
                continue

            # Try to reconstruct quark-antiquark pair
            quark_list = []
            can_test = True

            for comp in composition:
                constituent = comp.get("Constituent", "")
                is_anti = comp.get("IsAnti", False) or "Anti" in constituent or "anti" in constituent

                # Find matching quark
                quark_data = None
                search_name = constituent.replace("Anti-", "").replace("anti-", "").replace("-bar", "")

                for qname, qdata in self.quarks.items():
                    if search_name.lower().replace(" ", "") in qname.lower().replace(" ", ""):
                        quark_data = qdata.copy()
                        break

                if quark_data is None:
                    can_test = False
                    break

                # Create antiquark if needed
                if is_anti:
                    quark_data = self.create_antiquark(quark_data)

                quark_list.append(quark_data)

            if not can_test or len(quark_list) != 2:
                continue

            # Create meson from quarks
            try:
                predicted = SubatomicCalculatorV2.create_particle_from_quarks(
                    quark_list, particle_name, actual_data.get("Symbol", "X")
                )

                result = ParticleTestResult(name=particle_name)

                # Compare properties
                result.add_comparison("Charge_e", predicted.get("Charge_e"), actual_data.get("Charge_e"))
                result.add_comparison("Mass_MeVc2", predicted.get("Mass_MeVc2"), actual_data.get("Mass_MeVc2"))
                result.add_comparison("Spin_hbar", predicted.get("Spin_hbar"), actual_data.get("Spin_hbar"))
                result.add_comparison("BaryonNumber_B", predicted.get("BaryonNumber_B"), actual_data.get("BaryonNumber_B"))

                self.results.add_result(result)

                print(f"  {particle_name:20s}: Mass error = {result.comparisons[1].error_percent:6.2f}%")

            except Exception as e:
                print(f"  {particle_name:20s}: ERROR - {e}")


# ==================== Atom Calculator Tests ====================

class AtomCalculatorTester:
    """Test AtomCalculatorV2 against known elements."""

    def __init__(self):
        self.particles = DataLoader.load_special_particles()
        self.elements = DataLoader.load_elements()
        self.results = CalculatorTestResults("AtomCalculatorV2")

        # Get particle data
        self.proton_data = self.particles.get("Proton")
        self.neutron_data = self.particles.get("Neutron")
        self.electron_data = self.particles.get("Electron")

        # Also try loading from quarks folder
        if not self.proton_data:
            self.proton_data = DataLoader.load_json_file(
                DataLoader.BASE_PATH / "quarks" / "Proton.json"
            )
        if not self.electron_data:
            self.electron_data = DataLoader.load_json_file(
                DataLoader.BASE_PATH / "quarks" / "Electron.json"
            )

    def run_all_tests(self):
        """Run all element tests."""
        print("\n" + "="*70)
        print(" ATOM CALCULATOR V2 TESTS")
        print("="*70)

        if not self.proton_data or not self.neutron_data or not self.electron_data:
            print("ERROR: Missing particle data (Proton, Neutron, or Electron)")
            print(f"  Proton: {self.proton_data is not None}")
            print(f"  Neutron: {self.neutron_data is not None}")
            print(f"  Electron: {self.electron_data is not None}")
            return self.results

        print(f"\nTesting {len(self.elements)} elements...")
        print("-" * 70)

        for z, actual_data in sorted(self.elements.items()):
            self._test_element(z, actual_data)

        return self.results

    def _test_element(self, z: int, actual_data: Dict):
        """Test a single element."""
        # Determine neutron count from mass number or isotopes
        mass_number = None

        # Try to get from isotopes (most abundant)
        isotopes = actual_data.get("isotopes", [])
        if isotopes:
            # Find most abundant isotope
            most_abundant = max(isotopes, key=lambda x: x.get("abundance", 0))
            mass_number = most_abundant.get("mass_number")

        # Fallback: estimate from atomic mass
        if not mass_number:
            mass_number = round(actual_data.get("atomic_mass", z * 2))

        neutron_count = mass_number - z

        try:
            predicted = AtomCalculatorV2.create_atom_from_particles(
                self.proton_data,
                self.neutron_data,
                self.electron_data,
                proton_count=z,
                neutron_count=neutron_count,
                electron_count=z,  # Neutral atom
                element_name=actual_data.get("name", f"Element {z}"),
                element_symbol=actual_data.get("symbol", f"E{z}")
            )

            result = ParticleTestResult(name=f"{actual_data.get('symbol', z)} (Z={z})")

            # Compare properties
            result.add_comparison("atomic_mass",
                                predicted.get("atomic_mass"),
                                actual_data.get("atomic_mass"))

            if actual_data.get("ionization_energy"):
                result.add_comparison("ionization_energy",
                                    predicted.get("ionization_energy"),
                                    actual_data.get("ionization_energy"))

            if actual_data.get("electronegativity"):
                result.add_comparison("electronegativity",
                                    predicted.get("electronegativity"),
                                    actual_data.get("electronegativity"))

            if actual_data.get("density") and actual_data.get("density") > 0:
                result.add_comparison("density",
                                    predicted.get("density"),
                                    actual_data.get("density"))

            if actual_data.get("melting_point") and actual_data.get("melting_point") > 0:
                result.add_comparison("melting_point",
                                    predicted.get("melting_point"),
                                    actual_data.get("melting_point"))

            if actual_data.get("boiling_point") and actual_data.get("boiling_point") > 0:
                result.add_comparison("boiling_point",
                                    predicted.get("boiling_point"),
                                    actual_data.get("boiling_point"))

            if actual_data.get("atomic_radius"):
                result.add_comparison("atomic_radius",
                                    predicted.get("atomic_radius"),
                                    actual_data.get("atomic_radius"))

            self.results.add_result(result)

            # Print progress for every 10th element
            if z % 10 == 0 or z <= 10:
                mass_err = result.comparisons[0].error_percent if result.comparisons else 0
                print(f"  {actual_data.get('symbol', z):4s} (Z={z:3d}): Mass error = {mass_err:6.2f}%")

        except Exception as e:
            print(f"  Z={z:3d}: ERROR - {e}")


# ==================== Molecule Calculator Tests ====================

class MoleculeCalculatorTester:
    """Test MoleculeCalculatorV2 against known molecules."""

    def __init__(self):
        self.elements = DataLoader.load_elements()
        self.molecules = DataLoader.load_molecules()
        self.results = CalculatorTestResults("MoleculeCalculatorV2")

        # Create element lookup by symbol
        self.element_by_symbol = {}
        for z, data in self.elements.items():
            symbol = data.get("symbol", "")
            if symbol:
                self.element_by_symbol[symbol] = data

    def run_all_tests(self):
        """Run all molecule tests."""
        print("\n" + "="*70)
        print(" MOLECULE CALCULATOR V2 TESTS")
        print("="*70)

        print(f"\nTesting {len(self.molecules)} molecules...")
        print("-" * 70)

        for name, actual_data in self.molecules.items():
            self._test_molecule(name, actual_data)

        return self.results

    def _test_molecule(self, name: str, actual_data: Dict):
        """Test a single molecule."""
        composition = actual_data.get("Composition", [])
        if not composition:
            print(f"  {name:20s}: No composition data")
            return

        # Build element list and counts
        atom_data_list = []
        counts = []
        can_test = True

        for comp in composition:
            element_symbol = comp.get("Element", "")
            count = comp.get("Count", 1)

            element_data = self.element_by_symbol.get(element_symbol)
            if not element_data:
                print(f"  {name:20s}: Missing element {element_symbol}")
                can_test = False
                break

            atom_data_list.append(element_data)
            counts.append(count)

        if not can_test:
            return

        try:
            predicted = MoleculeCalculatorV2.create_molecule_from_atoms(
                atom_data_list,
                counts,
                molecule_name=name,
                molecule_formula=actual_data.get("Formula")
            )

            result = ParticleTestResult(name=name)

            # Compare properties
            result.add_comparison("MolecularMass_amu",
                                predicted.get("MolecularMass_amu"),
                                actual_data.get("MolecularMass_amu"))

            if actual_data.get("MeltingPoint_K"):
                result.add_comparison("MeltingPoint_K",
                                    predicted.get("MeltingPoint_K"),
                                    actual_data.get("MeltingPoint_K"))

            if actual_data.get("BoilingPoint_K"):
                result.add_comparison("BoilingPoint_K",
                                    predicted.get("BoilingPoint_K"),
                                    actual_data.get("BoilingPoint_K"))

            if actual_data.get("Density_g_cm3") and actual_data.get("Density_g_cm3") > 0:
                result.add_comparison("Density_g_cm3",
                                    predicted.get("Density_g_cm3"),
                                    actual_data.get("Density_g_cm3"))

            # Compare categorical properties
            pred_geometry = predicted.get("Geometry", "Unknown")
            actual_geometry = actual_data.get("Geometry", "Unknown")
            geometry_match = pred_geometry == actual_geometry
            result.add_match_comparison("Geometry_Match", geometry_match)

            pred_polarity = predicted.get("Polarity", "Unknown")
            actual_polarity = actual_data.get("Polarity", "Unknown")
            polarity_match = pred_polarity == actual_polarity
            result.add_match_comparison("Polarity_Match", polarity_match)

            self.results.add_result(result)

            mass_err = result.comparisons[0].error_percent
            geo_ok = "OK" if geometry_match else f"WRONG ({pred_geometry} vs {actual_geometry})"
            pol_ok = "OK" if polarity_match else f"WRONG ({pred_polarity} vs {actual_polarity})"

            print(f"  {name:20s}: Mass err={mass_err:5.2f}%  Geometry={geo_ok:30s}  Polarity={pol_ok}")

        except Exception as e:
            print(f"  {name:20s}: ERROR - {e}")


# ==================== Novel Combinations Tests ====================

class NovelCombinationTester:
    """Test calculator with combinations not in default data."""

    def __init__(self):
        self.quarks = DataLoader.load_quarks()
        self.particles = DataLoader.load_special_particles()
        self.elements = DataLoader.load_elements()

        # Get particle data
        self.proton_data = self.particles.get("Proton")
        self.neutron_data = self.particles.get("Neutron")
        self.electron_data = self.particles.get("Electron")

        # Element lookup
        self.element_by_symbol = {d.get("symbol"): d for d in self.elements.values()}

    def run_all_tests(self):
        """Run novel combination tests."""
        print("\n" + "="*70)
        print(" NOVEL COMBINATIONS TESTS")
        print("="*70)

        self._test_superheavy_elements()
        self._test_exotic_hadrons()
        self._test_unusual_molecules()

    def _test_superheavy_elements(self):
        """Test superheavy element predictions."""
        print("\n--- Superheavy Elements (Z > 118) ---")

        if not all([self.proton_data, self.neutron_data, self.electron_data]):
            print("  ERROR: Missing particle data")
            return

        # Test element 120 (Unbinilium)
        z = 120
        n = 180  # Estimated neutron count
        try:
            result = AtomCalculatorV2.create_atom_from_particles(
                self.proton_data, self.neutron_data, self.electron_data,
                z, n, z, "Unbinilium", "Ubn"
            )

            print(f"\n  Element 120 (Unbinilium, Z=120, N=180):")
            print(f"    Predicted mass: {result['atomic_mass']:.2f} amu")
            print(f"    Binding energy/nucleon: {result['binding_energy_per_nucleon_MeV']:.2f} MeV")
            print(f"    Block: {result['block']}, Period: {result['period']}")
            print(f"    Electronegativity: {result['electronegativity']:.2f}")

            # Check physical reasonableness
            reasonable = True
            issues = []

            if result['atomic_mass'] < z:
                issues.append("Mass too low")
                reasonable = False
            if result['binding_energy_per_nucleon_MeV'] < 6 or result['binding_energy_per_nucleon_MeV'] > 9:
                issues.append(f"B.E./nucleon unusual: {result['binding_energy_per_nucleon_MeV']:.2f}")

            print(f"    Physically reasonable: {reasonable} {issues if issues else ''}")

        except Exception as e:
            print(f"  Element 120: ERROR - {e}")

        # Test element 130
        z = 130
        n = 200
        try:
            result = AtomCalculatorV2.create_atom_from_particles(
                self.proton_data, self.neutron_data, self.electron_data,
                z, n, z, "Element130", "E130"
            )

            print(f"\n  Element 130 (Z=130, N=200):")
            print(f"    Predicted mass: {result['atomic_mass']:.2f} amu")
            print(f"    Binding energy/nucleon: {result['binding_energy_per_nucleon_MeV']:.2f} MeV")
            print(f"    Electronegativity: {result['electronegativity']:.2f}")

        except Exception as e:
            print(f"  Element 130: ERROR - {e}")

    def _test_exotic_hadrons(self):
        """Test exotic hadron predictions (tetraquarks, pentaquarks)."""
        print("\n--- Exotic Hadrons ---")

        up = self.quarks.get("Up Quark")
        down = self.quarks.get("Down Quark")
        strange = self.quarks.get("Strange Quark")
        charm = self.quarks.get("Charm Quark")

        if not all([up, down]):
            print("  ERROR: Missing quark data")
            return

        # Create antiquarks
        def make_anti(q):
            anti = q.copy()
            anti["Name"] = "Anti-" + q["Name"]
            anti["Charge_e"] = -q["Charge_e"]
            anti["BaryonNumber_B"] = -q["BaryonNumber_B"]
            anti["Isospin_I3"] = -q.get("Isospin_I3", 0)
            return anti

        anti_up = make_anti(up)
        anti_down = make_anti(down)

        # Test tetraquark (ccbar + uubar)
        if charm:
            anti_charm = make_anti(charm)
            try:
                quarks = [charm, anti_charm, up, anti_up]
                result = SubatomicCalculatorV2.create_particle_from_quarks(
                    quarks, "X(3872)-like", "X"
                )

                print(f"\n  Tetraquark X(3872)-like (cc-bar u u-bar):")
                print(f"    Predicted mass: {result['Mass_MeVc2']:.1f} MeV/c^2")
                print(f"    Charge: {result['Charge_e']:.0f}e")
                print(f"    Baryon number: {result['BaryonNumber_B']:.0f}")

                # X(3872) has mass ~3871.7 MeV
                print(f"    (Actual X(3872) mass: ~3871.7 MeV)")

            except Exception as e:
                print(f"  Tetraquark: ERROR - {e}")

        # Test pentaquark (uudc + anti-c)
        if charm:
            try:
                quarks = [up, up, down, charm, anti_charm]
                result = SubatomicCalculatorV2.create_particle_from_quarks(
                    quarks, "Pc(4380)-like", "Pc"
                )

                print(f"\n  Pentaquark Pc(4380)-like (uudc c-bar):")
                print(f"    Predicted mass: {result['Mass_MeVc2']:.1f} MeV/c^2")
                print(f"    Charge: {result['Charge_e']:.0f}e")
                print(f"    Baryon number: {result['BaryonNumber_B']:.0f}")

                # Pc(4380) has mass ~4380 MeV
                print(f"    (Actual Pc(4380) mass: ~4380 MeV)")

            except Exception as e:
                print(f"  Pentaquark: ERROR - {e}")

    def _test_unusual_molecules(self):
        """Test unusual molecule predictions."""
        print("\n--- Unusual Molecules ---")

        # SF6 - Sulfur hexafluoride
        S = self.element_by_symbol.get("S")
        F = self.element_by_symbol.get("F")

        if S and F:
            try:
                result = MoleculeCalculatorV2.create_molecule_from_atoms(
                    [S, F], [1, 6], "Sulfur Hexafluoride", "SF6"
                )

                print(f"\n  SF6 (Sulfur Hexafluoride):")
                print(f"    Predicted mass: {result['MolecularMass_amu']:.2f} amu (actual: 146.06)")
                print(f"    Geometry: {result['Geometry']} (actual: Octahedral)")
                print(f"    Polarity: {result['Polarity']} (actual: Nonpolar)")

            except Exception as e:
                print(f"  SF6: ERROR - {e}")

        # XeF6 - Xenon hexafluoride
        Xe = self.element_by_symbol.get("Xe")

        if Xe and F:
            try:
                result = MoleculeCalculatorV2.create_molecule_from_atoms(
                    [Xe, F], [1, 6], "Xenon Hexafluoride", "XeF6"
                )

                print(f"\n  XeF6 (Xenon Hexafluoride):")
                print(f"    Predicted mass: {result['MolecularMass_amu']:.2f} amu (actual: 245.28)")
                print(f"    Geometry: {result['Geometry']} (actual: Distorted octahedral)")
                print(f"    Polarity: {result['Polarity']} (actual: Polar)")

            except Exception as e:
                print(f"  XeF6: ERROR - {e}")

        # PF5 - Phosphorus pentafluoride
        P = self.element_by_symbol.get("P")

        if P and F:
            try:
                result = MoleculeCalculatorV2.create_molecule_from_atoms(
                    [P, F], [1, 5], "Phosphorus Pentafluoride", "PF5"
                )

                print(f"\n  PF5 (Phosphorus Pentafluoride):")
                print(f"    Predicted mass: {result['MolecularMass_amu']:.2f} amu (actual: 125.97)")
                print(f"    Geometry: {result['Geometry']} (actual: Trigonal Bipyramidal)")
                print(f"    Polarity: {result['Polarity']} (actual: Nonpolar)")

            except Exception as e:
                print(f"  PF5: ERROR - {e}")


# ==================== Improvement Suggestions ====================

def generate_improvement_suggestions(
    subatomic_results: CalculatorTestResults,
    atom_results: CalculatorTestResults,
    molecule_results: CalculatorTestResults
):
    """Generate specific formula improvement suggestions based on test results."""

    print("\n" + "="*70)
    print(" IMPROVEMENT SUGGESTIONS")
    print("="*70)

    # Subatomic suggestions
    print("\n--- SubatomicCalculatorV2 Improvements ---")

    mass_errors = subatomic_results.property_errors.get("Mass_MeVc2", [])
    if mass_errors:
        mean_mass_err = sum(mass_errors) / len(mass_errors)
        print(f"\n1. Mass Prediction (Mean error: {mean_mass_err:.1f}%)")

        if mean_mass_err > 20:
            print("   - Constituent quark mass dressing is too aggressive")
            print("   - Consider: Reduce light quark dressing from 336 MeV to ~310 MeV")
            print("   - Consider: Adjust binding correction per particle type")
            print("   - Consider: Implement hyperfine splitting based on spin state")
        elif mean_mass_err > 10:
            print("   - Fine-tune hyperfine correction strength")
            print("   - Consider: Scale = 40000-60000 MeV^3 instead of 50000")
        else:
            print("   - Mass predictions are reasonably accurate")

    spin_errors = subatomic_results.property_errors.get("Spin_hbar", [])
    if spin_errors:
        mean_spin_err = sum(spin_errors) / len(spin_errors)
        if mean_spin_err > 10:
            print(f"\n2. Spin Prediction (Mean error: {mean_spin_err:.1f}%)")
            print("   - Default to ground state spin is incorrect for some particles")
            print("   - Consider: Use particle classification to determine excited states")
            print("   - Delta baryons have J=3/2, not J=1/2")

    # Atom suggestions
    print("\n--- AtomCalculatorV2 Improvements ---")

    mass_errors = atom_results.property_errors.get("atomic_mass", [])
    if mass_errors:
        mean_mass_err = sum(mass_errors) / len(mass_errors)
        print(f"\n1. Atomic Mass (Mean error: {mean_mass_err:.2f}%)")

        if mean_mass_err > 1:
            print("   - Weizs\u00e4cker formula coefficients may need tuning")
            print("   - Consider: a_v=15.5, a_s=16.8, a_c=0.72, a_a=23.0")
            print("   - Consider: Enhanced shell corrections for magic numbers")
        else:
            print("   - Atomic mass predictions are excellent")

    ie_errors = atom_results.property_errors.get("ionization_energy", [])
    if ie_errors:
        mean_ie_err = sum(ie_errors) / len(ie_errors)
        print(f"\n2. Ionization Energy (Mean error: {mean_ie_err:.1f}%)")

        if mean_ie_err > 30:
            print("   - Slater's rules approximation too simple")
            print("   - Consider: Implement full Slater screening constants")
            print("   - Consider: Add quantum defect corrections")
            print("   - Consider: Use known values for noble gases as anchors")

    en_errors = atom_results.property_errors.get("electronegativity", [])
    if en_errors:
        mean_en_err = sum(en_errors) / len(en_errors)
        print(f"\n3. Electronegativity (Mean error: {mean_en_err:.1f}%)")

        if mean_en_err > 20:
            print("   - Periodic trend approximation needs refinement")
            print("   - Consider: Mulliken definition EN = (IE + EA) / 2")
            print("   - Consider: Allred-Rochow or Sanderson scale")

    density_errors = atom_results.property_errors.get("density", [])
    if density_errors:
        mean_density_err = sum(density_errors) / len(density_errors)
        print(f"\n4. Density (Mean error: {mean_density_err:.1f}%)")

        if mean_density_err > 50:
            print("   - Simple r^3 model inadequate for density")
            print("   - Consider: Crystal structure packing factors per element type")
            print("   - Consider: Use molar volume correlations by block")

    # Molecule suggestions
    print("\n--- MoleculeCalculatorV2 Improvements ---")

    mass_errors = molecule_results.property_errors.get("MolecularMass_amu", [])
    if mass_errors:
        mean_mass_err = sum(mass_errors) / len(mass_errors)
        print(f"\n1. Molecular Mass (Mean error: {mean_mass_err:.2f}%)")
        if mean_mass_err < 0.1:
            print("   - Molecular mass is exact (sum of atomic masses)")
        else:
            print("   - Check atomic mass data consistency")

    mp_errors = molecule_results.property_errors.get("MeltingPoint_K", [])
    if mp_errors:
        mean_mp_err = sum(mp_errors) / len(mp_errors)
        print(f"\n2. Melting/Boiling Points (Mean error: {mean_mp_err:.1f}%)")

        if mean_mp_err > 30:
            print("   - Simple MW-based estimation inadequate")
            print("   - Consider: Group contribution methods (Joback-Reid)")
            print("   - Consider: H-bond detection (O-H, N-H, F-H)")
            print("   - Consider: Separate models for organics vs inorganics")

    geo_errors = molecule_results.property_errors.get("Geometry_Match", [])
    if geo_errors:
        # geo_errors contains 0 (correct) or 100 (wrong), so accuracy = 100 - mean_error
        geo_accuracy = 100 - (sum(geo_errors) / len(geo_errors))
        print(f"\n3. Geometry Prediction (Accuracy: {geo_accuracy:.1f}%)")

        if geo_accuracy < 80:
            print("   - VSEPR implementation needs refinement")
            print("   - Consider: Better lone pair detection")
            print("   - Consider: Handle expanded octets (d-orbital involvement)")
            print("   - Consider: Special cases for pi bonding effects")


# ==================== Main Execution ====================

def main():
    """Run all tests and generate report."""

    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " COMPREHENSIVE ACCURACY TEST FOR DATA-DRIVEN CALCULATORS ".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    # Run subatomic tests
    subatomic_tester = SubatomicCalculatorTester()
    subatomic_results = subatomic_tester.run_all_tests()

    # Run atom tests
    atom_tester = AtomCalculatorTester()
    atom_results = atom_tester.run_all_tests()

    # Run molecule tests
    molecule_tester = MoleculeCalculatorTester()
    molecule_results = molecule_tester.run_all_tests()

    # Run novel combination tests
    novel_tester = NovelCombinationTester()
    novel_tester.run_all_tests()

    # Print summaries
    subatomic_results.print_summary()
    atom_results.print_summary()
    molecule_results.print_summary()

    # Generate improvement suggestions
    generate_improvement_suggestions(subatomic_results, atom_results, molecule_results)

    # Final summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)

    print(f"\nSubatomic Calculator:")
    print(f"  Particles tested: {len(subatomic_results.particle_results)}")
    mass_errs = subatomic_results.property_errors.get("Mass_MeVc2", [])
    if mass_errs:
        print(f"  Mean mass error: {sum(mass_errs)/len(mass_errs):.1f}%")

    print(f"\nAtom Calculator:")
    print(f"  Elements tested: {len(atom_results.particle_results)}")
    mass_errs = atom_results.property_errors.get("atomic_mass", [])
    if mass_errs:
        print(f"  Mean mass error: {sum(mass_errs)/len(mass_errs):.2f}%")

    print(f"\nMolecule Calculator:")
    print(f"  Molecules tested: {len(molecule_results.particle_results)}")
    mass_errs = molecule_results.property_errors.get("MolecularMass_amu", [])
    if mass_errs:
        print(f"  Mean mass error: {sum(mass_errs)/len(mass_errs):.2f}%")
    geo_errs = molecule_results.property_errors.get("Geometry_Match", [])
    if geo_errs:
        # geo_errs contains 0 (correct) or 100 (wrong), so accuracy = 100 - mean_error
        geo_accuracy = 100 - (sum(geo_errs) / len(geo_errs))
        print(f"  Geometry accuracy: {geo_accuracy:.1f}%")
    pol_errs = molecule_results.property_errors.get("Polarity_Match", [])
    if pol_errs:
        pol_accuracy = 100 - (sum(pol_errs) / len(pol_errs))
        print(f"  Polarity accuracy: {pol_accuracy:.1f}%")

    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70 + "\n")

    return subatomic_results, atom_results, molecule_results


if __name__ == "__main__":
    main()
