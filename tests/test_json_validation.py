#!/usr/bin/env python3
"""
JSON Data Validation Script
===========================

Compares calculated properties from physics calculators with known JSON data.
Validates elements, molecules, and alloys against their respective calculators.

Usage:
    python tests/test_json_validation.py

Reports:
    - Element validation (atomic mass, ionization energy, electronegativity, etc.)
    - Molecule validation (molecular mass, geometry, bond angle)
    - Alloy validation (density, melting point, tensile strength)
"""

import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Add parent directory to path

from periodica.utils.physics_calculator_v2 import (
    AtomCalculatorV2,
    MoleculeCalculatorV2,
    PhysicsConstantsV2
)
from periodica.utils.physics_calculator import MoleculeCalculator
from periodica.utils.alloy_calculator import AlloyCalculator


# ==================== Constants ====================

class ValidationThresholds:
    """Tolerance thresholds for property comparisons (as percentages)."""
    ATOMIC_MASS = 0.1          # 0.1% for atomic mass
    IONIZATION_ENERGY = 5.0    # 5% for ionization energy
    ELECTRONEGATIVITY = 10.0   # 10% for electronegativity
    ATOMIC_RADIUS = 15.0       # 15% for atomic radius

    MOLECULAR_MASS = 0.1       # 0.1% for molecular mass
    BOND_ANGLE = 5.0           # 5% for bond angle

    ALLOY_DENSITY = 10.0       # 10% for alloy density
    MELTING_POINT = 10.0       # 10% for melting point
    TENSILE_STRENGTH = 20.0    # 20% for tensile strength


class ValidationStatus(Enum):
    """Validation result status."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class PropertyResult:
    """Result of a single property validation."""
    property_name: str
    json_value: Any
    calculated_value: Any
    error_percent: Optional[float]
    threshold: float
    status: ValidationStatus
    notes: str = ""


@dataclass
class EntityResult:
    """Result of validating an entity (element, molecule, or alloy)."""
    name: str
    entity_type: str
    properties: List[PropertyResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for p in self.properties if p.status == ValidationStatus.PASS)

    @property
    def warned(self) -> int:
        return sum(1 for p in self.properties if p.status == ValidationStatus.WARN)

    @property
    def failed(self) -> int:
        return sum(1 for p in self.properties if p.status == ValidationStatus.FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for p in self.properties if p.status == ValidationStatus.SKIP)


# ==================== Data Loaders ====================

class DataLoader:
    """Load JSON data files from active directories."""

    BASE_PATH = Path(__file__).parent.parent / "src" / "periodica" / "data" / "active"

    @classmethod
    def load_json_file(cls, filepath: Path) -> Optional[Dict]:
        """Load a single JSON file, handling JavaScript-style comments."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
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
        except json.JSONDecodeError as e:
            print(f"JSON parse error in {filepath}: {e}")
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    @classmethod
    def load_elements(cls) -> Dict[str, Dict]:
        """Load all element data files."""
        elements = {}
        element_path = cls.BASE_PATH / "elements"
        if not element_path.exists():
            print(f"Warning: Elements directory not found at {element_path}")
            return elements

        for file in sorted(element_path.glob("*.json")):
            data = cls.load_json_file(file)
            if data and "symbol" in data:
                elements[data["symbol"]] = data
        return elements

    @classmethod
    def load_quarks(cls) -> Dict[str, Dict]:
        """Load all quark data files."""
        quarks = {}
        quark_path = cls.BASE_PATH / "quarks"
        if not quark_path.exists():
            return quarks

        for file in quark_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                name = data.get("Name", file.stem)
                quarks[name] = data
        return quarks

    @classmethod
    def load_subatomic(cls) -> Dict[str, Dict]:
        """Load all subatomic particle data files."""
        particles = {}
        subatomic_path = cls.BASE_PATH / "subatomic"
        if not subatomic_path.exists():
            return particles

        for file in subatomic_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                name = data.get("Name", file.stem)
                particles[name] = data
        return particles

    @classmethod
    def load_molecules(cls) -> Dict[str, Dict]:
        """Load all molecule data files."""
        molecules = {}
        molecule_path = cls.BASE_PATH / "molecules"
        if not molecule_path.exists():
            return molecules

        for file in molecule_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                name = data.get("Name", file.stem)
                molecules[name] = data
        return molecules

    @classmethod
    def load_alloys(cls) -> Dict[str, Dict]:
        """Load all alloy data files."""
        alloys = {}
        alloy_path = cls.BASE_PATH / "alloys"
        if not alloy_path.exists():
            return alloys

        for file in alloy_path.glob("*.json"):
            data = cls.load_json_file(file)
            if data:
                name = data.get("Name", file.stem)
                alloys[name] = data
        return alloys


# ==================== Validation Utilities ====================

def calculate_percent_error(actual: float, expected: float) -> Optional[float]:
    """Calculate percentage error between actual and expected values."""
    if expected == 0:
        if actual == 0:
            return 0.0
        return float('inf')
    return abs((actual - expected) / expected) * 100


def get_status(error: Optional[float], threshold: float,
               warn_threshold: Optional[float] = None) -> ValidationStatus:
    """Determine validation status based on error and thresholds."""
    if error is None:
        return ValidationStatus.SKIP
    if error <= threshold:
        return ValidationStatus.PASS
    if warn_threshold and error <= warn_threshold:
        return ValidationStatus.WARN
    # Use 2x threshold as warning threshold if not specified
    if error <= threshold * 2:
        return ValidationStatus.WARN
    return ValidationStatus.FAIL


# ==================== Element Validation ====================

class ElementValidator:
    """Validate element properties against calculated values."""

    def __init__(self, subatomic_data: Dict[str, Dict]):
        """Initialize with subatomic particle data."""
        self.proton_data = subatomic_data.get("Proton", self._default_proton())
        self.neutron_data = subatomic_data.get("Neutron", self._default_neutron())
        self.electron_data = subatomic_data.get("Electron", self._default_electron())

    @staticmethod
    def _default_proton() -> Dict:
        """Default proton data if not found."""
        return {
            "Name": "Proton",
            "Mass_MeVc2": 938.27208816,
            "Mass_amu": 1.007276466621,
            "Charge_e": 1
        }

    @staticmethod
    def _default_neutron() -> Dict:
        """Default neutron data if not found."""
        return {
            "Name": "Neutron",
            "Mass_MeVc2": 939.56542052,
            "Mass_amu": 1.008664915,
            "Charge_e": 0
        }

    @staticmethod
    def _default_electron() -> Dict:
        """Default electron data if not found."""
        return {
            "Name": "Electron",
            "Mass_MeVc2": 0.51099895,
            "Mass_amu": 0.000548579909065,
            "Charge_e": -1
        }

    def validate_element(self, element_data: Dict) -> EntityResult:
        """Validate a single element's properties."""
        symbol = element_data.get("symbol", "?")
        name = element_data.get("name", symbol)
        atomic_number = element_data.get("atomic_number", 0)

        result = EntityResult(
            name=f"{name} ({symbol})",
            entity_type="Element"
        )

        if atomic_number == 0:
            result.properties.append(PropertyResult(
                property_name="atomic_number",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.SKIP,
                notes="Missing atomic number"
            ))
            return result

        # Determine neutron count from most abundant isotope
        neutrons = self._get_neutrons(element_data, atomic_number)

        # Calculate properties using AtomCalculatorV2
        try:
            calculated = AtomCalculatorV2.create_atom_from_particles(
                proton_data=self.proton_data,
                neutron_data=self.neutron_data,
                electron_data=self.electron_data,
                proton_count=atomic_number,
                neutron_count=neutrons,
                electron_count=atomic_number,
                element_name=name,
                element_symbol=symbol
            )
        except Exception as e:
            result.properties.append(PropertyResult(
                property_name="calculation",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.FAIL,
                notes=f"Calculation error: {str(e)}"
            ))
            return result

        # Validate atomic_mass
        json_mass = element_data.get("atomic_mass")
        calc_mass = calculated.get("atomic_mass")
        if json_mass is not None and calc_mass is not None:
            error = calculate_percent_error(calc_mass, json_mass)
            result.properties.append(PropertyResult(
                property_name="atomic_mass",
                json_value=json_mass,
                calculated_value=round(calc_mass, 4),
                error_percent=round(error, 4) if error else None,
                threshold=ValidationThresholds.ATOMIC_MASS,
                status=get_status(error, ValidationThresholds.ATOMIC_MASS)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="atomic_mass",
                json_value=json_mass,
                calculated_value=calc_mass,
                error_percent=None,
                threshold=ValidationThresholds.ATOMIC_MASS,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        # Validate ionization_energy
        json_ie = element_data.get("ionization_energy")
        calc_ie = calculated.get("ionization_energy")
        if json_ie is not None and calc_ie is not None:
            error = calculate_percent_error(calc_ie, json_ie)
            result.properties.append(PropertyResult(
                property_name="ionization_energy",
                json_value=json_ie,
                calculated_value=round(calc_ie, 3),
                error_percent=round(error, 2) if error else None,
                threshold=ValidationThresholds.IONIZATION_ENERGY,
                status=get_status(error, ValidationThresholds.IONIZATION_ENERGY)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="ionization_energy",
                json_value=json_ie,
                calculated_value=calc_ie,
                error_percent=None,
                threshold=ValidationThresholds.IONIZATION_ENERGY,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        # Validate electronegativity
        json_en = element_data.get("electronegativity")
        calc_en = calculated.get("electronegativity")
        if json_en is not None and calc_en is not None and json_en > 0:
            error = calculate_percent_error(calc_en, json_en)
            result.properties.append(PropertyResult(
                property_name="electronegativity",
                json_value=json_en,
                calculated_value=round(calc_en, 2),
                error_percent=round(error, 2) if error else None,
                threshold=ValidationThresholds.ELECTRONEGATIVITY,
                status=get_status(error, ValidationThresholds.ELECTRONEGATIVITY)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="electronegativity",
                json_value=json_en,
                calculated_value=calc_en,
                error_percent=None,
                threshold=ValidationThresholds.ELECTRONEGATIVITY,
                status=ValidationStatus.SKIP,
                notes="Missing or zero value"
            ))

        # Validate atomic_radius
        json_radius = element_data.get("atomic_radius")
        calc_radius = calculated.get("atomic_radius")
        if json_radius is not None and calc_radius is not None:
            error = calculate_percent_error(calc_radius, json_radius)
            result.properties.append(PropertyResult(
                property_name="atomic_radius",
                json_value=json_radius,
                calculated_value=round(calc_radius, 0),
                error_percent=round(error, 2) if error else None,
                threshold=ValidationThresholds.ATOMIC_RADIUS,
                status=get_status(error, ValidationThresholds.ATOMIC_RADIUS)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="atomic_radius",
                json_value=json_radius,
                calculated_value=calc_radius,
                error_percent=None,
                threshold=ValidationThresholds.ATOMIC_RADIUS,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        # Validate electron_configuration (exact match)
        json_config = element_data.get("electron_configuration")
        calc_config = calculated.get("electron_configuration")
        if json_config is not None and calc_config is not None:
            # Normalize configs for comparison (remove spaces, handle superscripts)
            json_norm = self._normalize_config(json_config)
            calc_norm = self._normalize_config(calc_config)
            matches = json_norm == calc_norm
            result.properties.append(PropertyResult(
                property_name="electron_configuration",
                json_value=json_config,
                calculated_value=calc_config,
                error_percent=0 if matches else 100,
                threshold=0,
                status=ValidationStatus.PASS if matches else ValidationStatus.WARN,
                notes="" if matches else "Config differs"
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="electron_configuration",
                json_value=json_config,
                calculated_value=calc_config,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        return result

    @staticmethod
    def _normalize_config(config: str) -> str:
        """Normalize electron configuration for comparison."""
        if not config:
            return ""
        # Convert superscript digits to regular digits
        superscript_map = {
            '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
            '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
            '\u2078': '8', '\u2079': '9'
        }
        for sup, reg in superscript_map.items():
            config = config.replace(sup, reg)
        # Remove spaces and convert to lowercase
        return config.replace(" ", "").lower()

    def _get_neutrons(self, element_data: Dict, atomic_number: int) -> int:
        """Get neutron count from isotope data or estimate."""
        isotopes = element_data.get("isotopes", [])

        # Find most abundant stable isotope
        best_isotope = None
        max_abundance = -1

        for iso in isotopes:
            abundance = iso.get("abundance", 0)
            if abundance > max_abundance:
                max_abundance = abundance
                best_isotope = iso

        if best_isotope and "neutrons" in best_isotope:
            return best_isotope["neutrons"]
        if best_isotope and "mass_number" in best_isotope:
            return best_isotope["mass_number"] - atomic_number

        # Estimate neutrons from atomic mass if available
        atomic_mass = element_data.get("atomic_mass", 0)
        if atomic_mass > 0:
            return round(atomic_mass) - atomic_number

        # Default estimation
        return atomic_number if atomic_number <= 20 else int(atomic_number * 1.2)


# ==================== Molecule Validation ====================

class MoleculeValidator:
    """Validate molecule properties against calculated values."""

    def __init__(self, elements_data: Dict[str, Dict]):
        """Initialize with element data for lookups."""
        self.elements = elements_data

    def validate_molecule(self, molecule_data: Dict) -> EntityResult:
        """Validate a single molecule's properties."""
        name = molecule_data.get("Name", "Unknown")
        formula = molecule_data.get("Formula", "")

        result = EntityResult(
            name=f"{name} ({formula})",
            entity_type="Molecule"
        )

        composition = molecule_data.get("Composition", [])
        if not composition:
            result.properties.append(PropertyResult(
                property_name="composition",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.SKIP,
                notes="Missing composition data"
            ))
            return result

        # Build element data list for calculator
        element_data_list = []
        counts = []
        for comp in composition:
            elem_symbol = comp.get("Element", "")
            count = comp.get("Count", 1)
            elem_data = self.elements.get(elem_symbol)
            if elem_data:
                element_data_list.append(elem_data)
                counts.append(count)
            else:
                # Use minimal element data
                element_data_list.append({
                    "symbol": elem_symbol,
                    "atomic_mass": self._estimate_mass(elem_symbol),
                    "electronegativity": 2.0
                })
                counts.append(count)

        # Validate MolecularMass_amu
        json_mass = molecule_data.get("MolecularMass_amu")
        if json_mass is not None:
            # Calculate using MoleculeCalculator
            calc_mass = MoleculeCalculator.calculate_molecular_mass(composition)
            error = calculate_percent_error(calc_mass, json_mass)
            result.properties.append(PropertyResult(
                property_name="MolecularMass_amu",
                json_value=json_mass,
                calculated_value=round(calc_mass, 3),
                error_percent=round(error, 4) if error else None,
                threshold=ValidationThresholds.MOLECULAR_MASS,
                status=get_status(error, ValidationThresholds.MOLECULAR_MASS)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="MolecularMass_amu",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=ValidationThresholds.MOLECULAR_MASS,
                status=ValidationStatus.SKIP,
                notes="Missing JSON value"
            ))

        # Validate Geometry (exact match)
        json_geometry = molecule_data.get("Geometry")
        if json_geometry is not None:
            calc_geometry = MoleculeCalculator.estimate_geometry(composition)
            matches = self._geometries_match(json_geometry, calc_geometry)
            result.properties.append(PropertyResult(
                property_name="Geometry",
                json_value=json_geometry,
                calculated_value=calc_geometry,
                error_percent=0 if matches else 100,
                threshold=0,
                status=ValidationStatus.PASS if matches else ValidationStatus.WARN,
                notes="" if matches else "Geometry differs"
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="Geometry",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.SKIP,
                notes="Missing JSON value"
            ))

        # Validate BondAngle_deg
        json_angle = molecule_data.get("BondAngle_deg")
        if json_angle is not None:
            # Estimate bond angle from geometry
            calc_angle = self._estimate_bond_angle(composition)
            if calc_angle:
                error = calculate_percent_error(calc_angle, json_angle)
                result.properties.append(PropertyResult(
                    property_name="BondAngle_deg",
                    json_value=json_angle,
                    calculated_value=round(calc_angle, 1),
                    error_percent=round(error, 2) if error else None,
                    threshold=ValidationThresholds.BOND_ANGLE,
                    status=get_status(error, ValidationThresholds.BOND_ANGLE)
                ))
            else:
                result.properties.append(PropertyResult(
                    property_name="BondAngle_deg",
                    json_value=json_angle,
                    calculated_value=None,
                    error_percent=None,
                    threshold=ValidationThresholds.BOND_ANGLE,
                    status=ValidationStatus.SKIP,
                    notes="Cannot estimate angle"
                ))
        else:
            result.properties.append(PropertyResult(
                property_name="BondAngle_deg",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=ValidationThresholds.BOND_ANGLE,
                status=ValidationStatus.SKIP,
                notes="Missing JSON value"
            ))

        return result

    @staticmethod
    def _estimate_mass(symbol: str) -> float:
        """Estimate atomic mass for unknown elements."""
        common_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'S': 32.065, 'P': 30.974, 'F': 18.998, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904, 'Na': 22.990, 'K': 39.098,
            'Ca': 40.078, 'Mg': 24.305, 'Fe': 55.845, 'Cu': 63.546,
            'Zn': 65.38, 'Al': 26.982, 'Si': 28.086
        }
        return common_masses.get(symbol, 50.0)

    @staticmethod
    def _geometries_match(json_geom: str, calc_geom: str) -> bool:
        """Check if geometries match (case-insensitive, handles synonyms)."""
        if not json_geom or not calc_geom:
            return False

        json_lower = json_geom.lower().strip()
        calc_lower = calc_geom.lower().strip()

        if json_lower == calc_lower:
            return True

        # Handle common synonyms
        synonyms = {
            'trigonal pyramidal': ['trigonal-pyramidal', 'pyramidal'],
            'trigonal planar': ['trigonal-planar', 'planar'],
            'tetrahedral': ['tetrahedron'],
            'linear': ['diatomic'],
            'bent': ['angular', 'v-shaped'],
            'octahedral': ['octahedron'],
            'square planar': ['square-planar']
        }

        for canonical, alts in synonyms.items():
            all_forms = [canonical] + alts
            if json_lower in all_forms and calc_lower in all_forms:
                return True

        return False

    def _estimate_bond_angle(self, composition: List[Dict]) -> Optional[float]:
        """Estimate bond angle from composition using VSEPR theory."""
        geometry = MoleculeCalculator.estimate_geometry(composition)

        angle_map = {
            'Linear': 180.0,
            'Bent': 104.5,
            'Trigonal Planar': 120.0,
            'Trigonal Pyramidal': 107.0,
            'Tetrahedral': 109.5,
            'Square Planar': 90.0,
            'Octahedral': 90.0,
            'Seesaw': 117.5,
            'T-shaped': 90.0
        }

        return angle_map.get(geometry)


# ==================== Alloy Validation ====================

class AlloyValidator:
    """Validate alloy properties against calculated values."""

    def __init__(self, elements_data: Dict[str, Dict]):
        """Initialize with element data for lookups."""
        self.elements = elements_data

    def validate_alloy(self, alloy_data: Dict) -> EntityResult:
        """Validate a single alloy's properties."""
        name = alloy_data.get("Name", "Unknown")
        formula = alloy_data.get("Formula", "")

        result = EntityResult(
            name=f"{name} ({formula})",
            entity_type="Alloy"
        )

        components = alloy_data.get("Components", [])
        if not components:
            result.properties.append(PropertyResult(
                property_name="components",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.SKIP,
                notes="Missing components data"
            ))
            return result

        # Build component data and weight fractions
        component_data = []
        weight_fractions = []

        for comp in components:
            elem_symbol = comp.get("Element", "")
            # Use midpoint of range as typical composition
            min_pct = comp.get("MinPercent", 0)
            max_pct = comp.get("MaxPercent", 0)
            avg_pct = (min_pct + max_pct) / 2 / 100  # Convert to fraction

            elem_data = self.elements.get(elem_symbol, {"symbol": elem_symbol})
            component_data.append(elem_data)
            weight_fractions.append(avg_pct)

        # Normalize weight fractions
        total = sum(weight_fractions)
        if total > 0:
            weight_fractions = [w / total for w in weight_fractions]
        else:
            result.properties.append(PropertyResult(
                property_name="weight_fractions",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.SKIP,
                notes="Invalid weight fractions"
            ))
            return result

        # Get lattice type
        lattice_props = alloy_data.get("LatticeProperties", {})
        lattice_type = lattice_props.get("PrimaryStructure", "FCC")

        # Calculate alloy properties
        try:
            calculated = AlloyCalculator.create_alloy_from_components(
                component_data=component_data,
                weight_fractions=weight_fractions,
                lattice_type=lattice_type,
                name=name
            )
        except Exception as e:
            result.properties.append(PropertyResult(
                property_name="calculation",
                json_value=None,
                calculated_value=None,
                error_percent=None,
                threshold=0,
                status=ValidationStatus.FAIL,
                notes=f"Calculation error: {str(e)}"
            ))
            return result

        # Validate density
        phys_props = alloy_data.get("PhysicalProperties", {})
        json_density = phys_props.get("Density_g_cm3")
        calc_density = calculated.get("PhysicalProperties", {}).get("Density_g_cm3")

        if json_density is not None and calc_density is not None:
            error = calculate_percent_error(calc_density, json_density)
            result.properties.append(PropertyResult(
                property_name="density",
                json_value=json_density,
                calculated_value=round(calc_density, 3),
                error_percent=round(error, 2) if error else None,
                threshold=ValidationThresholds.ALLOY_DENSITY,
                status=get_status(error, ValidationThresholds.ALLOY_DENSITY)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="density",
                json_value=json_density,
                calculated_value=calc_density,
                error_percent=None,
                threshold=ValidationThresholds.ALLOY_DENSITY,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        # Validate melting point
        json_mp = phys_props.get("MeltingPoint_K")
        calc_mp = calculated.get("PhysicalProperties", {}).get("MeltingPoint_K")

        if json_mp is not None and calc_mp is not None:
            error = calculate_percent_error(calc_mp, json_mp)
            result.properties.append(PropertyResult(
                property_name="melting_point",
                json_value=json_mp,
                calculated_value=round(calc_mp, 1),
                error_percent=round(error, 2) if error else None,
                threshold=ValidationThresholds.MELTING_POINT,
                status=get_status(error, ValidationThresholds.MELTING_POINT)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="melting_point",
                json_value=json_mp,
                calculated_value=calc_mp,
                error_percent=None,
                threshold=ValidationThresholds.MELTING_POINT,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        # Validate tensile strength
        mech_props = alloy_data.get("MechanicalProperties", {})
        json_ts = mech_props.get("TensileStrength_MPa")
        calc_ts = calculated.get("MechanicalProperties", {}).get("TensileStrength_MPa")

        if json_ts is not None and calc_ts is not None:
            error = calculate_percent_error(calc_ts, json_ts)
            result.properties.append(PropertyResult(
                property_name="tensile_strength",
                json_value=json_ts,
                calculated_value=round(calc_ts, 0),
                error_percent=round(error, 2) if error else None,
                threshold=ValidationThresholds.TENSILE_STRENGTH,
                status=get_status(error, ValidationThresholds.TENSILE_STRENGTH)
            ))
        else:
            result.properties.append(PropertyResult(
                property_name="tensile_strength",
                json_value=json_ts,
                calculated_value=calc_ts,
                error_percent=None,
                threshold=ValidationThresholds.TENSILE_STRENGTH,
                status=ValidationStatus.SKIP,
                notes="Missing value"
            ))

        return result


# ==================== Report Generation ====================

class ValidationReport:
    """Generate and format validation reports."""

    def __init__(self):
        self.element_results: List[EntityResult] = []
        self.molecule_results: List[EntityResult] = []
        self.alloy_results: List[EntityResult] = []
        self.discrepancies: List[Dict] = []

    def add_element_result(self, result: EntityResult):
        self.element_results.append(result)

    def add_molecule_result(self, result: EntityResult):
        self.molecule_results.append(result)

    def add_alloy_result(self, result: EntityResult):
        self.alloy_results.append(result)

    def generate_report(self) -> str:
        """Generate complete validation report."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("JSON DATA VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Element report
        if self.element_results:
            lines.extend(self._generate_section_report(
                "ELEMENT VALIDATION REPORT",
                self.element_results
            ))

        # Molecule report
        if self.molecule_results:
            lines.extend(self._generate_section_report(
                "MOLECULE VALIDATION REPORT",
                self.molecule_results
            ))

        # Alloy report
        if self.alloy_results:
            lines.extend(self._generate_section_report(
                "ALLOY VALIDATION REPORT",
                self.alloy_results
            ))

        # Discrepancy analysis
        lines.extend(self._generate_discrepancy_analysis())

        # Overall summary
        lines.extend(self._generate_overall_summary())

        return "\n".join(lines)

    def _generate_section_report(self, title: str,
                                  results: List[EntityResult]) -> List[str]:
        """Generate report for a specific entity type."""
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        lines.append("")

        # Detailed results for each entity
        for result in results:
            lines.append(f"Entity: {result.name}")
            lines.append("-" * 60)
            lines.append(f"{'Property':<25} | {'JSON':<12} | {'Calculated':<12} | {'Error':<8} | Status")
            lines.append("-" * 60)

            for prop in result.properties:
                json_val = self._format_value(prop.json_value)
                calc_val = self._format_value(prop.calculated_value)
                error_str = f"{prop.error_percent:.2f}%" if prop.error_percent is not None else "N/A"
                status_str = prop.status.value

                lines.append(
                    f"{prop.property_name:<25} | {json_val:<12} | {calc_val:<12} | "
                    f"{error_str:<8} | {status_str}"
                )

                # Track significant discrepancies
                if prop.status == ValidationStatus.FAIL:
                    self.discrepancies.append({
                        'entity': result.name,
                        'entity_type': result.entity_type,
                        'property': prop.property_name,
                        'json_value': prop.json_value,
                        'calculated_value': prop.calculated_value,
                        'error': prop.error_percent,
                        'notes': prop.notes
                    })

            lines.append("")

        # Section summary
        total = len(results)
        passed = sum(1 for r in results if r.failed == 0)
        with_warnings = sum(1 for r in results if r.warned > 0 and r.failed == 0)
        failed = sum(1 for r in results if r.failed > 0)

        lines.append(f"SECTION SUMMARY: {total} entities tested")
        lines.append(f"  - Fully passed: {passed}")
        lines.append(f"  - With warnings: {with_warnings}")
        lines.append(f"  - With failures: {failed}")
        lines.append("")

        return lines

    def _generate_discrepancy_analysis(self) -> List[str]:
        """Generate analysis of significant discrepancies."""
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("DISCREPANCY ANALYSIS")
        lines.append("=" * 80)
        lines.append("")

        if not self.discrepancies:
            lines.append("No significant discrepancies found.")
            return lines

        lines.append(f"Found {len(self.discrepancies)} significant discrepancies:")
        lines.append("")

        # Group by property type
        by_property = defaultdict(list)
        for d in self.discrepancies:
            by_property[d['property']].append(d)

        for prop, disc_list in sorted(by_property.items()):
            lines.append(f"Property: {prop}")
            lines.append("-" * 40)
            for disc in disc_list:
                lines.append(f"  - {disc['entity']}")
                lines.append(f"    JSON: {disc['json_value']}, Calculated: {disc['calculated_value']}")
                if disc['error'] is not None:
                    lines.append(f"    Error: {disc['error']:.2f}%")
                if disc['notes']:
                    lines.append(f"    Notes: {disc['notes']}")
            lines.append("")

            # Suggest action
            if len(disc_list) > 3:
                lines.append(f"  >> SUGGESTION: Multiple {prop} discrepancies may indicate")
                lines.append(f"     calculation model needs adjustment")
            else:
                lines.append(f"  >> SUGGESTION: Check if JSON data needs update for these entities")
            lines.append("")

        return lines

    def _generate_overall_summary(self) -> List[str]:
        """Generate overall validation summary."""
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("OVERALL SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Count totals
        total_entities = (len(self.element_results) +
                        len(self.molecule_results) +
                        len(self.alloy_results))

        total_properties = 0
        passed_properties = 0
        warned_properties = 0
        failed_properties = 0
        skipped_properties = 0

        for results_list in [self.element_results, self.molecule_results, self.alloy_results]:
            for result in results_list:
                for prop in result.properties:
                    total_properties += 1
                    if prop.status == ValidationStatus.PASS:
                        passed_properties += 1
                    elif prop.status == ValidationStatus.WARN:
                        warned_properties += 1
                    elif prop.status == ValidationStatus.FAIL:
                        failed_properties += 1
                    elif prop.status == ValidationStatus.SKIP:
                        skipped_properties += 1

        lines.append(f"Total entities validated: {total_entities}")
        lines.append(f"  - Elements: {len(self.element_results)}")
        lines.append(f"  - Molecules: {len(self.molecule_results)}")
        lines.append(f"  - Alloys: {len(self.alloy_results)}")
        lines.append("")
        lines.append(f"Total property comparisons: {total_properties}")
        lines.append(f"  - PASS: {passed_properties} ({100*passed_properties/max(1,total_properties):.1f}%)")
        lines.append(f"  - WARN: {warned_properties} ({100*warned_properties/max(1,total_properties):.1f}%)")
        lines.append(f"  - FAIL: {failed_properties} ({100*failed_properties/max(1,total_properties):.1f}%)")
        lines.append(f"  - SKIP: {skipped_properties} ({100*skipped_properties/max(1,total_properties):.1f}%)")
        lines.append("")

        # Pass rate (excluding skipped)
        validated = total_properties - skipped_properties
        if validated > 0:
            pass_rate = 100 * (passed_properties + warned_properties) / validated
            lines.append(f"Pass rate (PASS + WARN): {pass_rate:.1f}%")

        lines.append("")
        lines.append("=" * 80)

        return lines

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) >= 10000:
                return f"{value:.2e}"
            return f"{value:.4g}"
        if isinstance(value, str) and len(value) > 12:
            return value[:10] + ".."
        return str(value)


# ==================== Main Validation Runner ====================

def run_validation(verbose: bool = True) -> ValidationReport:
    """Run complete validation suite."""
    print("Loading JSON data files...")

    # Load all data
    elements = DataLoader.load_elements()
    quarks = DataLoader.load_quarks()
    subatomic = DataLoader.load_subatomic()
    molecules = DataLoader.load_molecules()
    alloys = DataLoader.load_alloys()

    print(f"  Loaded {len(elements)} elements")
    print(f"  Loaded {len(quarks)} quarks")
    print(f"  Loaded {len(subatomic)} subatomic particles")
    print(f"  Loaded {len(molecules)} molecules")
    print(f"  Loaded {len(alloys)} alloys")
    print("")

    # Create report
    report = ValidationReport()

    # Validate elements
    print("Validating elements...")
    element_validator = ElementValidator(subatomic)
    for symbol, elem_data in sorted(elements.items(), key=lambda x: x[1].get("atomic_number", 0)):
        result = element_validator.validate_element(elem_data)
        report.add_element_result(result)
        if verbose and result.failed > 0:
            print(f"  Warning: {result.name} has {result.failed} failed properties")

    # Validate molecules
    print("Validating molecules...")
    molecule_validator = MoleculeValidator(elements)
    for name, mol_data in sorted(molecules.items()):
        result = molecule_validator.validate_molecule(mol_data)
        report.add_molecule_result(result)
        if verbose and result.failed > 0:
            print(f"  Warning: {result.name} has {result.failed} failed properties")

    # Validate alloys
    print("Validating alloys...")
    alloy_validator = AlloyValidator(elements)
    for name, alloy_data in sorted(alloys.items()):
        result = alloy_validator.validate_alloy(alloy_data)
        report.add_alloy_result(result)
        if verbose and result.failed > 0:
            print(f"  Warning: {result.name} has {result.failed} failed properties")

    print("")
    print("Validation complete!")

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate JSON data against physics calculators"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output during validation"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file for report (default: print to stdout)"
    )
    parser.add_argument(
        "--elements-only",
        action="store_true",
        help="Only validate elements"
    )
    parser.add_argument(
        "--molecules-only",
        action="store_true",
        help="Only validate molecules"
    )
    parser.add_argument(
        "--alloys-only",
        action="store_true",
        help="Only validate alloys"
    )

    args = parser.parse_args()

    # Run validation
    report = run_validation(verbose=args.verbose)

    # Generate and output report
    report_text = report.generate_report()

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report_text)
        print(f"Report written to {args.output}")
    else:
        print(report_text)

    # Return exit code based on failures
    total_failures = len(report.discrepancies)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
