"""
Molecule Data Loader
Loads molecule data from JSON files in the Molecules folder.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class MoleculeDataLoader:
    """Loads molecule data from JSON files"""

    def __init__(self, molecules_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            molecules_dir: Path to directory containing molecule JSON files.
                         If None, uses default 'Molecules' directory.
        """
        if molecules_dir is None:
            # Default to data/active/molecules relative to this file
            base_dir = Path(__file__).parent
            molecules_dir = base_dir / "active" / "molecules"

        self.molecules_dir = Path(molecules_dir)
        self.molecules: List[Dict] = []
        self.molecules_by_name: Dict[str, Dict] = {}
        self.molecules_by_formula: Dict[str, Dict] = {}

    def load_all_molecules(self) -> List[Dict]:
        """
        Load all molecule data from JSON files.

        Returns:
            List of molecule dictionaries sorted by name.
        """
        if not self.molecules_dir.exists():
            print(f"Warning: Molecules directory not found: {self.molecules_dir}")
            return []

        # Find all JSON files
        json_files = sorted(self.molecules_dir.glob("*.json"))

        if not json_files:
            print(f"Warning: No molecule JSON files found in {self.molecules_dir}")
            return []

        loaded_molecules = []

        for json_file in json_files:
            try:
                molecule_data = self._load_molecule_file(json_file)
                if molecule_data:
                    loaded_molecules.append(molecule_data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
                continue

        # Sort by name
        loaded_molecules.sort(key=lambda m: m.get('Name', ''))

        # Store in instance variables
        self.molecules = loaded_molecules
        self.molecules_by_name = {m['Name']: m for m in loaded_molecules}
        self.molecules_by_formula = {m['Formula']: m for m in loaded_molecules}

        return loaded_molecules

    def _load_molecule_file(self, filepath: Path) -> Optional[Dict]:
        """
        Load a single molecule JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dictionary containing molecule data or None if invalid
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
        required_fields = ['Name', 'Formula', 'MolecularMass_amu', 'BondType', 'Geometry']

        for field in required_fields:
            if field not in data:
                print(f"Warning: Missing required field '{field}' in {filepath.name}")
                return None

        # Add derived fields for easier access
        data['name'] = data['Name']
        data['formula'] = data['Formula']
        data['mass'] = data['MolecularMass_amu']
        data['bond_type'] = data['BondType']
        data['geometry'] = data['Geometry']
        data['polarity'] = data.get('Polarity', 'Unknown')
        data['category'] = data.get('Category', 'Unknown')
        data['state'] = data.get('State_STP', 'Unknown')
        data['melting_point'] = data.get('MeltingPoint_K', 0)
        data['boiling_point'] = data.get('BoilingPoint_K', 0)
        data['density'] = data.get('Density_g_cm3', 0)
        data['dipole_moment'] = data.get('DipoleMoment_D', 0)
        data['bond_angle'] = data.get('BondAngle_deg', 0)
        data['color'] = data.get('Color', '#4FC3F7')

        return data

    def get_molecule_by_name(self, name: str) -> Optional[Dict]:
        """Get molecule data by name"""
        return self.molecules_by_name.get(name)

    def get_molecule_by_formula(self, formula: str) -> Optional[Dict]:
        """Get molecule data by formula"""
        return self.molecules_by_formula.get(formula)

    def get_all_molecules(self) -> List[Dict]:
        """Get all loaded molecules"""
        return self.molecules

    def get_molecule_count(self) -> int:
        """Get total number of loaded molecules"""
        return len(self.molecules)

    def get_molecules_by_category(self, category: str) -> List[Dict]:
        """Get molecules filtered by category"""
        return [m for m in self.molecules if m.get('category') == category]

    def get_molecules_by_bond_type(self, bond_type: str) -> List[Dict]:
        """Get molecules filtered by bond type"""
        return [m for m in self.molecules if m.get('bond_type') == bond_type]

    def get_molecules_by_polarity(self, polarity: str) -> List[Dict]:
        """Get molecules filtered by polarity"""
        return [m for m in self.molecules if m.get('polarity') == polarity]

    def get_molecules_by_geometry(self, geometry: str) -> List[Dict]:
        """Get molecules filtered by geometry"""
        return [m for m in self.molecules if m.get('geometry') == geometry]

    def get_molecules_by_state(self, state: str) -> List[Dict]:
        """Get molecules filtered by state at STP"""
        return [m for m in self.molecules if m.get('state') == state]

    def get_unique_categories(self) -> List[str]:
        """Get list of unique categories"""
        return sorted(list(set(m.get('category', 'Unknown') for m in self.molecules)))

    def get_unique_geometries(self) -> List[str]:
        """Get list of unique geometries"""
        return sorted(list(set(m.get('geometry', 'Unknown') for m in self.molecules)))

    def get_unique_bond_types(self) -> List[str]:
        """Get list of unique bond types"""
        return sorted(list(set(m.get('bond_type', 'Unknown') for m in self.molecules)))

    def get_unique_polarities(self) -> List[str]:
        """Get list of unique polarities"""
        return sorted(list(set(m.get('polarity', 'Unknown') for m in self.molecules)))

    def get_unique_states(self) -> List[str]:
        """Get list of unique states"""
        return sorted(list(set(m.get('state', 'Unknown') for m in self.molecules)))


# Global loader instance for convenient access
_molecule_loader = None


def get_molecule_loader() -> MoleculeDataLoader:
    """Get or create the global molecule loader instance"""
    global _molecule_loader
    if _molecule_loader is None:
        _molecule_loader = MoleculeDataLoader()
        _molecule_loader.load_all_molecules()
    return _molecule_loader


def get_all_molecules() -> List[Dict]:
    """Convenience function to get all molecules"""
    return get_molecule_loader().get_all_molecules()
