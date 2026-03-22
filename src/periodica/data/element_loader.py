"""
Element Data Loader
Loads element data from JSON files on application startup.
This replaces hardcoded Python data with data-driven JSON configuration.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ElementDataLoader:
    """Loads element data from JSON files and provides access methods"""

    def __init__(self, elements_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            elements_dir: Path to directory containing element JSON files.
                         If None, uses default 'data/elements' directory.
        """
        if elements_dir is None:
            # Default to data/active/elements relative to this file
            base_dir = Path(__file__).parent
            elements_dir = base_dir / "active" / "elements"

        self.elements_dir = Path(elements_dir)
        self.elements: List[Dict] = []
        self.elements_by_symbol: Dict[str, Dict] = {}
        self.elements_by_z: Dict[int, Dict] = {}
        self._blocks: Dict[str, List[str]] = {'s': [], 'p': [], 'd': [], 'f': []}
        self._periods: Dict[int, List[str]] = {i: [] for i in range(1, 8)}
        self._groups: Dict[int, List[str]] = {i: [] for i in range(1, 19)}
        self._loaded = False

    def load_all_elements(self) -> List[Dict]:
        """
        Load all element data from JSON files.

        Returns:
            List of element dictionaries sorted by atomic number.
        """
        if not self.elements_dir.exists():
            raise FileNotFoundError(f"Elements directory not found: {self.elements_dir}")

        # Find all JSON files matching pattern: ###_XX.json (e.g., 001_H.json)
        json_files = sorted(self.elements_dir.glob("*.json"))

        if not json_files:
            raise ValueError(f"No element JSON files found in {self.elements_dir}")

        loaded_elements = []

        for json_file in json_files:
            try:
                element_data = self._load_element_file(json_file)
                loaded_elements.append(element_data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
                continue

        # Sort by atomic number
        loaded_elements.sort(key=lambda e: e['atomic_number'])

        # Store in instance variables
        self.elements = loaded_elements
        self.elements_by_symbol = {e['symbol']: e for e in loaded_elements}
        self.elements_by_z = {e['atomic_number']: e for e in loaded_elements}

        # Build indices for blocks, periods, groups
        self._build_indices()
        self._loaded = True

        return loaded_elements

    def _load_element_file(self, filepath: Path) -> Dict:
        """
        Load a single element JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dictionary containing element data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields (minimal set - some properties may be null for superheavy elements)
        required_fields = ['symbol', 'name', 'atomic_number', 'block', 'period']

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in {filepath.name}")

        return data

    def _build_indices(self):
        """Build indices for fast lookup by block, period, and group"""
        self._blocks = {'s': [], 'p': [], 'd': [], 'f': []}
        self._periods = {i: [] for i in range(1, 8)}
        self._groups = {i: [] for i in range(1, 19)}

        for element in self.elements:
            symbol = element['symbol']
            block = element.get('block')
            period = element.get('period')
            group = element.get('group')

            if block and block in self._blocks:
                self._blocks[block].append(symbol)
            if period and period in self._periods:
                self._periods[period].append(symbol)
            if group and group in self._groups:
                self._groups[group].append(symbol)

    def ensure_loaded(self):
        """Ensure data is loaded, load if not already"""
        if not self._loaded:
            self.load_all_elements()

    # ==================== Basic Accessors ====================

    def get_element_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get element data by symbol (e.g., 'H', 'C', 'Fe')"""
        self.ensure_loaded()
        return self.elements_by_symbol.get(symbol)

    def get_element_by_z(self, z: int) -> Optional[Dict]:
        """Get element data by atomic number"""
        self.ensure_loaded()
        return self.elements_by_z.get(z)

    def get_all_elements(self) -> List[Dict]:
        """Get all loaded elements"""
        self.ensure_loaded()
        return self.elements

    def get_element_count(self) -> int:
        """Get total number of loaded elements"""
        self.ensure_loaded()
        return len(self.elements)

    # ==================== Property Accessors ====================

    def get_property(self, symbol: str, property_name: str, default: Any = None) -> Any:
        """Get a specific property for an element by symbol"""
        self.ensure_loaded()
        element = self.elements_by_symbol.get(symbol)
        if element:
            return element.get(property_name, default)
        return default

    def get_property_by_z(self, z: int, property_name: str, default: Any = None) -> Any:
        """Get a specific property for an element by atomic number"""
        self.ensure_loaded()
        element = self.elements_by_z.get(z)
        if element:
            return element.get(property_name, default)
        return default

    def get_ionization_energy(self, symbol: str) -> Optional[float]:
        """Get ionization energy in eV"""
        return self.get_property(symbol, 'ionization_energy')

    def get_electronegativity(self, symbol: str) -> Optional[float]:
        """Get electronegativity (Pauling scale)"""
        return self.get_property(symbol, 'electronegativity')

    def get_atomic_radius(self, symbol: str) -> Optional[int]:
        """Get atomic radius in pm"""
        return self.get_property(symbol, 'atomic_radius')

    def get_melting_point(self, symbol: str) -> Optional[float]:
        """Get melting point in K"""
        return self.get_property(symbol, 'melting_point')

    def get_boiling_point(self, symbol: str) -> Optional[float]:
        """Get boiling point in K"""
        return self.get_property(symbol, 'boiling_point')

    def get_density(self, symbol: str) -> Optional[float]:
        """Get density in g/cm³"""
        return self.get_property(symbol, 'density')

    def get_electron_affinity(self, symbol: str) -> Optional[float]:
        """Get electron affinity in kJ/mol"""
        return self.get_property(symbol, 'electron_affinity')

    def get_emission_wavelength(self, symbol: str) -> Optional[float]:
        """Get primary emission wavelength in nm"""
        return self.get_property(symbol, 'primary_emission_wavelength')

    def get_visible_emission_wavelength(self, symbol: str) -> Optional[float]:
        """Get visible emission wavelength in nm"""
        return self.get_property(symbol, 'visible_emission_wavelength')

    def get_electron_configuration(self, symbol: str) -> Optional[str]:
        """Get electron configuration string"""
        return self.get_property(symbol, 'electron_configuration')

    def get_valence_electrons(self, symbol: str) -> Optional[int]:
        """Get number of valence electrons"""
        return self.get_property(symbol, 'valence_electrons')

    def get_isotopes(self, symbol: str) -> List[Dict]:
        """Get isotope data for an element"""
        return self.get_property(symbol, 'isotopes', [])

    # ==================== Classification Accessors ====================

    def get_block(self, symbol: str) -> Optional[str]:
        """Get orbital block (s, p, d, f) for an element"""
        return self.get_property(symbol, 'block')

    def get_period(self, symbol: str) -> Optional[int]:
        """Get period number (1-7) for an element"""
        return self.get_property(symbol, 'period')

    def get_group(self, symbol: str) -> Optional[int]:
        """Get group number (1-18) for an element"""
        return self.get_property(symbol, 'group')

    def get_atomic_number(self, symbol: str) -> Optional[int]:
        """Get atomic number for an element"""
        return self.get_property(symbol, 'atomic_number')

    def get_atomic_mass(self, symbol: str) -> Optional[float]:
        """Get atomic mass in u"""
        return self.get_property(symbol, 'atomic_mass')

    # ==================== Collection Accessors ====================

    def get_elements_by_block(self, block: str) -> List[str]:
        """Get list of element symbols in a block (s, p, d, f)"""
        self.ensure_loaded()
        return self._blocks.get(block, [])

    def get_elements_by_period(self, period: int) -> List[str]:
        """Get list of element symbols in a period (1-7)"""
        self.ensure_loaded()
        return self._periods.get(period, [])

    def get_elements_by_group(self, group: int) -> List[str]:
        """Get list of element symbols in a group (1-18)"""
        self.ensure_loaded()
        return self._groups.get(group, [])

    def get_blocks(self) -> Dict[str, List[str]]:
        """Get dictionary of blocks to element symbols"""
        self.ensure_loaded()
        return self._blocks

    def get_periods(self) -> Dict[int, List[str]]:
        """Get dictionary of periods to element symbols"""
        self.ensure_loaded()
        return self._periods

    def get_groups(self) -> Dict[int, List[str]]:
        """Get dictionary of groups to element symbols"""
        self.ensure_loaded()
        return self._groups

    # ==================== Property Dictionaries ====================

    def get_property_dict(self, property_name: str) -> Dict[str, Any]:
        """
        Get a dictionary mapping element symbols to a specific property.
        Useful for compatibility with old code that used dictionaries.
        """
        self.ensure_loaded()
        return {
            e['symbol']: e.get(property_name)
            for e in self.elements
            if e.get(property_name) is not None
        }

    def get_ionization_energies(self) -> Dict[str, float]:
        """Get dictionary of symbol -> ionization energy"""
        return self.get_property_dict('ionization_energy')

    def get_electronegativities(self) -> Dict[str, float]:
        """Get dictionary of symbol -> electronegativity"""
        return self.get_property_dict('electronegativity')

    def get_atomic_radii(self) -> Dict[str, int]:
        """Get dictionary of symbol -> atomic radius"""
        return self.get_property_dict('atomic_radius')

    def get_melting_points(self) -> Dict[str, float]:
        """Get dictionary of symbol -> melting point"""
        return self.get_property_dict('melting_point')

    def get_boiling_points(self) -> Dict[str, float]:
        """Get dictionary of symbol -> boiling point"""
        return self.get_property_dict('boiling_point')

    def get_densities(self) -> Dict[str, float]:
        """Get dictionary of symbol -> density"""
        return self.get_property_dict('density')

    def get_electron_affinities(self) -> Dict[str, float]:
        """Get dictionary of symbol -> electron affinity"""
        return self.get_property_dict('electron_affinity')

    def get_emission_wavelengths(self) -> Dict[str, float]:
        """Get dictionary of symbol -> primary emission wavelength"""
        return self.get_property_dict('primary_emission_wavelength')

    # ==================== Property Metadata ====================

    def get_property_metadata(self, property_name: str) -> Dict[str, Any]:
        """
        Get metadata for a property (units, min, max, type).
        """
        self.ensure_loaded()

        metadata = {
            'ionization_energy': {'unit': 'eV', 'type': 'float', 'description': 'First ionization energy'},
            'electronegativity': {'unit': 'Pauling', 'type': 'float', 'description': 'Pauling electronegativity'},
            'atomic_radius': {'unit': 'pm', 'type': 'int', 'description': 'Atomic radius'},
            'melting_point': {'unit': 'K', 'type': 'float', 'description': 'Melting point'},
            'boiling_point': {'unit': 'K', 'type': 'float', 'description': 'Boiling point'},
            'density': {'unit': 'g/cm³', 'type': 'float', 'description': 'Density'},
            'electron_affinity': {'unit': 'kJ/mol', 'type': 'float', 'description': 'Electron affinity'},
            'primary_emission_wavelength': {'unit': 'nm', 'type': 'float', 'description': 'Primary emission wavelength'},
            'visible_emission_wavelength': {'unit': 'nm', 'type': 'float', 'description': 'Visible emission wavelength'},
            'atomic_mass': {'unit': 'u', 'type': 'float', 'description': 'Atomic mass'},
            'valence_electrons': {'unit': '', 'type': 'int', 'description': 'Number of valence electrons'},
        }

        base_meta = metadata.get(property_name, {'unit': '', 'type': 'any', 'description': property_name})

        # Calculate min/max from actual data
        values = [e.get(property_name) for e in self.elements if e.get(property_name) is not None]
        if values:
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                base_meta['min'] = min(numeric_values)
                base_meta['max'] = max(numeric_values)

        return base_meta

    # ==================== Utility Methods ====================

    def search_elements(self, **criteria) -> List[Dict]:
        """
        Search elements by criteria.

        Args:
            **criteria: Property name and value to match
                       e.g., block='s', period=4

        Returns:
            List of matching element dictionaries
        """
        self.ensure_loaded()
        results = []
        for element in self.elements:
            match = True
            for key, value in criteria.items():
                if element.get(key) != value:
                    match = False
                    break
            if match:
                results.append(element)
        return results

    def get_symbol_by_z(self, z: int) -> Optional[str]:
        """Get element symbol by atomic number"""
        element = self.get_element_by_z(z)
        return element['symbol'] if element else None

    def get_name_by_symbol(self, symbol: str) -> Optional[str]:
        """Get element name by symbol"""
        return self.get_property(symbol, 'name')

    def get_all_symbols(self) -> List[str]:
        """Get list of all element symbols in atomic number order"""
        self.ensure_loaded()
        return [e['symbol'] for e in self.elements]


# Global loader instance for easy access
_global_loader: Optional[ElementDataLoader] = None


def get_loader() -> ElementDataLoader:
    """Get the global element data loader instance"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ElementDataLoader()
        _global_loader.load_all_elements()
    return _global_loader


def get_element(symbol: str) -> Optional[Dict]:
    """Convenience function to get element by symbol"""
    return get_loader().get_element_by_symbol(symbol)


def get_element_by_z(z: int) -> Optional[Dict]:
    """Convenience function to get element by atomic number"""
    return get_loader().get_element_by_z(z)


def get_all_elements() -> List[Dict]:
    """Convenience function to get all elements"""
    return get_loader().get_all_elements()


# Compatibility aliases for old code
def create_fallback_element_data():
    """
    Legacy function - now returns data from JSON files.
    Kept for backward compatibility.
    """
    loader = get_loader()
    return {e['symbol']: e for e in loader.get_all_elements()}
