"""
Alloy Data Loader
Loads alloy data from JSON files in the alloys folder.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class AlloyDataLoader:
    """Loads alloy data from JSON files"""

    def __init__(self, alloys_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            alloys_dir: Path to directory containing alloy JSON files.
                       If None, uses default 'data/active/alloys' directory.
        """
        if alloys_dir is None:
            # Default to data/active/alloys relative to this file
            base_dir = Path(__file__).parent
            alloys_dir = base_dir / "active" / "alloys"

        self.alloys_dir = Path(alloys_dir)
        self.alloys: List[Dict] = []
        self.alloys_by_name: Dict[str, Dict] = {}

    def load_all_alloys(self) -> List[Dict]:
        """
        Load all alloy data from JSON files.

        Returns:
            List of alloy dictionaries sorted by name.
        """
        if not self.alloys_dir.exists():
            print(f"Warning: Alloys directory not found: {self.alloys_dir}")
            return []

        # Find all JSON files
        json_files = sorted(self.alloys_dir.glob("*.json"))

        if not json_files:
            print(f"Warning: No alloy JSON files found in {self.alloys_dir}")
            return []

        loaded_alloys = []

        for json_file in json_files:
            try:
                alloy_data = self._load_alloy_file(json_file)
                if alloy_data:
                    loaded_alloys.append(alloy_data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
                continue

        # Sort by name
        loaded_alloys.sort(key=lambda a: a.get('Name', ''))

        # Store in instance variables
        self.alloys = loaded_alloys
        self.alloys_by_name = {a['Name']: a for a in loaded_alloys}

        return loaded_alloys

    def _load_alloy_file(self, filepath: Path) -> Optional[Dict]:
        """
        Load a single alloy JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dictionary containing alloy data or None if invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error in {filepath.name}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Failed to read {filepath.name}: {e}")
            return None

        # Validate required fields
        required_fields = ['Name', 'Category', 'Components']

        for field in required_fields:
            if field not in data:
                print(f"Warning: Missing required field '{field}' in {filepath.name}")
                return None

        # Add derived fields for easier access
        data['name'] = data['Name']
        data['formula'] = data.get('Formula', '')
        data['category'] = data.get('Category', 'Other')
        data['subcategory'] = data.get('SubCategory', '')
        data['description'] = data.get('Description', '')

        # Physical properties with defaults
        phys = data.get('PhysicalProperties', {})
        data['density'] = phys.get('Density_g_cm3', 0)
        data['melting_point'] = phys.get('MeltingPoint_K', 0)
        data['thermal_conductivity'] = phys.get('ThermalConductivity_W_mK', 0)
        data['thermal_expansion'] = phys.get('ThermalExpansion_per_K', 0)
        data['electrical_resistivity'] = phys.get('ElectricalResistivity_Ohm_m', 0)
        data['specific_heat'] = phys.get('SpecificHeat_J_kgK', 0)
        data['youngs_modulus'] = phys.get('YoungsModulus_GPa', 0)
        data['shear_modulus'] = phys.get('ShearModulus_GPa', 0)
        data['poissons_ratio'] = phys.get('PoissonsRatio', 0)

        # Hardness values (multiple scales)
        data['hardness'] = phys.get('BrinellHardness_HB', 0)
        data['hardness_brinell'] = phys.get('BrinellHardness_HB', 0)
        data['hardness_vickers'] = phys.get('VickersHardness_HV', 0)
        data['hardness_rockwell'] = phys.get('RockwellHardness_HRC', 0)

        # Mechanical properties
        mech = data.get('MechanicalProperties', {})
        data['tensile_strength'] = mech.get('TensileStrength_MPa', 0)
        data['yield_strength'] = mech.get('YieldStrength_MPa', 0)
        data['elongation'] = mech.get('Elongation_percent', 0)
        data['reduction_of_area'] = mech.get('ReductionOfArea_percent', 0)
        data['impact_strength'] = mech.get('ImpactStrength_J', 0)
        data['fatigue_strength'] = mech.get('FatigueStrength_MPa', 0)
        data['fracture_toughness'] = mech.get('FractureToughness_MPa_sqrt_m', 0)

        # Corrosion properties
        corr = data.get('CorrosionResistance', {})
        data['pren'] = corr.get('PREN', 0) if isinstance(corr, dict) else 0
        data['pitting_potential'] = corr.get('PittingPotential_mV_SCE', 0) if isinstance(corr, dict) else 0
        # Corrosion resistance rating (numeric 0-100 scale, or derived from PREN)
        if isinstance(corr, dict):
            pren_val = corr.get('PREN', 0)
            # Convert PREN to 0-100 scale (PREN typically ranges 0-50+)
            data['corrosion_resistance'] = min(100, pren_val * 2) if pren_val > 0 else 50
        else:
            data['corrosion_resistance'] = 50  # Default moderate

        # Economic properties (estimated if not present)
        econ = data.get('EconomicProperties', {})
        data['cost_per_kg'] = econ.get('CostPerKg_USD', 0)

        # Lattice properties
        lattice = data.get('LatticeProperties', {})
        data['crystal_structure'] = lattice.get('PrimaryStructure', 'Unknown')
        lattice_params = lattice.get('LatticeParameters', {})
        data['lattice_parameter_a'] = lattice_params.get('a_pm', 0)
        data['packing_factor'] = lattice.get('AtomicPackingFactor', 0)

        # Microstructure data for visualization
        micro = data.get('Microstructure', {})
        grain = micro.get('GrainStructure', {})
        data['grain_size'] = grain.get('AverageGrainSize_um', 50)
        data['grain_seed_density'] = grain.get('VoronoiSeedDensity_per_mm2', 400)

        phase_dist = micro.get('PhaseDistribution', {})
        data['noise_type'] = phase_dist.get('NoiseType', 'Simplex')
        data['noise_scale'] = phase_dist.get('NoiseScale', 0.1)

        # Get primary element (base element from components)
        components = data.get('Components', [])
        data['primary_element'] = 'Unknown'
        for comp in components:
            if comp.get('Role', '').lower() == 'base':
                data['primary_element'] = comp.get('Element', 'Unknown')
                break

        # Color
        data['color'] = data.get('Color', '#C0C0C0')

        return data

    def get_alloy_by_name(self, name: str) -> Optional[Dict]:
        """Get alloy data by name"""
        return self.alloys_by_name.get(name)

    def get_all_alloys(self) -> List[Dict]:
        """Get all loaded alloys"""
        return self.alloys

    def get_alloy_count(self) -> int:
        """Get total number of loaded alloys"""
        return len(self.alloys)

    def get_alloys_by_category(self, category: str) -> List[Dict]:
        """Get alloys filtered by category"""
        return [a for a in self.alloys if a.get('category', '').lower() == category.lower()]

    def get_alloys_by_structure(self, structure: str) -> List[Dict]:
        """Get alloys filtered by crystal structure"""
        return [a for a in self.alloys if a.get('crystal_structure', '').upper() == structure.upper()]

    def get_alloys_by_primary_element(self, element: str) -> List[Dict]:
        """Get alloys filtered by primary element"""
        return [a for a in self.alloys if a.get('primary_element') == element]

    def get_unique_categories(self) -> List[str]:
        """Get list of unique categories"""
        return sorted(list(set(a.get('category', 'Other') for a in self.alloys)))

    def get_unique_structures(self) -> List[str]:
        """Get list of unique crystal structures"""
        return sorted(list(set(a.get('crystal_structure', 'Unknown') for a in self.alloys)))

    def get_unique_primary_elements(self) -> List[str]:
        """Get list of unique primary elements"""
        return sorted(list(set(a.get('primary_element', 'Unknown') for a in self.alloys)))

    def get_property_range(self, property_name: str) -> tuple:
        """Get min/max range for a property"""
        values = [a.get(property_name, 0) for a in self.alloys if a.get(property_name, 0) > 0]
        if not values:
            return (0, 1)
        return (min(values), max(values))


# Global loader instance for convenient access
_alloy_loader = None


def get_alloy_loader() -> AlloyDataLoader:
    """Get or create the global alloy loader instance"""
    global _alloy_loader
    if _alloy_loader is None:
        _alloy_loader = AlloyDataLoader()
        _alloy_loader.load_all_alloys()
    return _alloy_loader


def get_all_alloys() -> List[Dict]:
    """Convenience function to get all alloys"""
    return get_alloy_loader().get_all_alloys()
