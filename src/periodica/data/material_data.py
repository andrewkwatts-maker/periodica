"""
Material Data Module
Handles loading and management of engineering material data.
Materials represent the observable/engineering level above alloys.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from periodica.utils.logger import get_logger

logger = get_logger('data.material_data')


class MaterialDataLoader:
    """Loads material data from JSON files."""

    _instance = None
    _materials: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_materials()
        return cls._instance

    def _load_materials(self):
        """Load all material JSON files from data/active/materials/."""
        from periodica.data.validation import validate_required_fields
        materials_dir = Path(__file__).parent / "active" / "materials"
        if not materials_dir.exists():
            return

        for json_file in materials_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    missing = validate_required_fields(data, 'materials', json_file.name)
                    if missing:
                        continue
                    name = data.get('Name', json_file.stem)
                    self._materials[name] = data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load %s: %s", json_file, e)

    def get_material(self, name: str) -> Optional[Dict]:
        """Get a material by name."""
        return self._materials.get(name)

    def get_all_materials(self) -> Dict[str, Dict]:
        """Get all loaded materials."""
        return self._materials.copy()

    def get_material_names(self) -> List[str]:
        """Get list of all material names."""
        return list(self._materials.keys())

    def get_materials_by_category(self, category: str) -> List[Dict]:
        """Get all materials of a specific category."""
        return [
            m for m in self._materials.values()
            if m.get('Category', '').lower() == category.lower()
        ]

    def reload(self):
        """Force reload of all material data."""
        self._materials.clear()
        self._load_materials()


# Material categories with typical property ranges
MATERIAL_CATEGORIES = {
    'Structural Steel': {
        'youngs_modulus_GPa': (190, 210),
        'yield_strength_MPa': (250, 1200),
        'density_kg_m3': (7800, 8050),
    },
    'Stainless Steel': {
        'youngs_modulus_GPa': (190, 210),
        'yield_strength_MPa': (170, 1000),
        'density_kg_m3': (7750, 8100),
    },
    'Aluminum Alloy': {
        'youngs_modulus_GPa': (68, 73),
        'yield_strength_MPa': (30, 550),
        'density_kg_m3': (2640, 2850),
    },
    'Titanium Alloy': {
        'youngs_modulus_GPa': (100, 120),
        'yield_strength_MPa': (200, 1200),
        'density_kg_m3': (4400, 4900),
    },
    'Copper Alloy': {
        'youngs_modulus_GPa': (100, 140),
        'yield_strength_MPa': (50, 600),
        'density_kg_m3': (7400, 9000),
    },
    'Nickel Superalloy': {
        'youngs_modulus_GPa': (190, 220),
        'yield_strength_MPa': (400, 1400),
        'density_kg_m3': (7800, 8900),
    },
    'Polymer': {
        'youngs_modulus_GPa': (0.1, 4.0),
        'yield_strength_MPa': (10, 100),
        'density_kg_m3': (900, 2200),
    },
    'Ceramic': {
        'youngs_modulus_GPa': (150, 450),
        'yield_strength_MPa': (100, 1000),  # Compressive
        'density_kg_m3': (2500, 6000),
    },
    'Composite': {
        'youngs_modulus_GPa': (20, 250),
        'yield_strength_MPa': (100, 2000),
        'density_kg_m3': (1400, 2500),
    },
}


# Processing effects on properties (multipliers)
PROCESSING_EFFECTS = {
    'cold_worked': {
        'yield_strength': 1.3,
        'elongation': 0.6,
        'hardness': 1.25,
    },
    'annealed': {
        'yield_strength': 0.8,
        'elongation': 1.4,
        'hardness': 0.85,
    },
    'solution_treated': {
        'yield_strength': 0.9,
        'elongation': 1.2,
        'hardness': 0.9,
    },
    'age_hardened': {
        'yield_strength': 1.5,
        'elongation': 0.7,
        'hardness': 1.4,
    },
    'quench_tempered': {
        'yield_strength': 1.4,
        'elongation': 0.8,
        'hardness': 1.35,
    },
}


def get_material_loader() -> MaterialDataLoader:
    """Get the singleton material data loader instance."""
    return MaterialDataLoader()
