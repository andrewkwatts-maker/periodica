"""
Cascade Regeneration Engine
============================
Orchestrates full derivation chain regeneration from quarks through to
biomaterials, respecting dependency order and preserving manual edits.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from periodica.utils.derivation_metadata import DerivationSource, DerivationTracker
from periodica.utils.logger import get_logger

logger = get_logger('cascade_engine')

# Derivation order: each category depends on those before it
DERIVATION_ORDER = [
    'quarks',
    'subatomic',
    'elements',
    'molecules',
    'alloys',
    'materials',
    'amino_acids',
    'proteins',
    'nucleic_acids',
    'cell_components',
    'cells',
    'biomaterials',
]

# Directory mapping for data/active/
_CATEGORY_DIRS = {
    'quarks': 'quarks',
    'subatomic': 'subatomic',
    'elements': 'elements',
    'molecules': 'molecules',
    'alloys': 'alloys',
    'materials': 'materials',
    'amino_acids': 'amino_acids',
    'proteins': 'proteins',
    'nucleic_acids': 'nucleic_acids',
    'cell_components': 'cell_components',
    'cells': 'cells',
    'biomaterials': 'biological_materials',
}

# What each category depends on for derivation
_DEPENDENCIES = {
    'quarks': [],
    'subatomic': ['quarks'],
    'elements': ['quarks', 'subatomic'],
    'molecules': ['elements'],
    'alloys': ['elements'],
    'materials': ['alloys'],
    'amino_acids': ['elements'],
    'proteins': ['amino_acids'],
    'nucleic_acids': ['elements'],
    'cell_components': ['proteins', 'nucleic_acids'],
    'cells': ['cell_components'],
    'biomaterials': ['cells'],
}


class CascadeRegenerationEngine:
    """
    Orchestrates full cascade regeneration from quarks to biomaterials.
    Respects dependency order and can preserve manual edits.
    """

    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            data_root = str(Path(__file__).parent.parent / 'data' / 'active')
        self._data_root = Path(data_root)
        self._generators = {}

    def get_categories(self) -> List[str]:
        """Return all categories in derivation order."""
        return list(DERIVATION_ORDER)

    def get_dependencies(self, category: str) -> List[str]:
        """Return what a category depends on."""
        return list(_DEPENDENCIES.get(category, []))

    def get_downstream(self, category: str) -> List[str]:
        """Return all categories downstream of the given one."""
        if category not in DERIVATION_ORDER:
            return []
        idx = DERIVATION_ORDER.index(category)
        return DERIVATION_ORDER[idx + 1:]

    def regenerate_all(
        self,
        categories: Optional[List[str]] = None,
        preserve_manual: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """
        Regenerate all categories in dependency order.

        Args:
            categories: Specific categories to regenerate (None = all)
            preserve_manual: If True, skip items with source == MANUAL
            progress_callback: fn(percent, message)

        Returns:
            Dict of {category: count_generated}
        """
        if categories is None:
            categories = list(DERIVATION_ORDER)

        # Filter to valid categories in order
        ordered = [c for c in DERIVATION_ORDER if c in categories]
        results = {}
        total = len(ordered)

        for i, category in enumerate(ordered):
            if progress_callback:
                pct = int(i / total * 100)
                progress_callback(pct, f"Stage {i+1}/{total}: Regenerating {category}...")

            try:
                count = self._regenerate_category(
                    category, preserve_manual, progress_callback
                )
                results[category] = count
                logger.info(f"Regenerated {count} items for {category}")
            except Exception as e:
                logger.error(f"Failed to regenerate {category}: {e}")
                results[category] = 0

        if progress_callback:
            total_items = sum(results.values())
            progress_callback(100, f"Cascade complete: {total_items} items across {len(results)} categories")

        return results

    def regenerate_from(
        self,
        start_category: str,
        preserve_manual: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """
        Regenerate from a specific category downward.

        Args:
            start_category: Category to start from (inclusive)
            preserve_manual: Preserve manually edited items
            progress_callback: fn(percent, message)

        Returns:
            Dict of {category: count_generated}
        """
        if start_category not in DERIVATION_ORDER:
            raise ValueError(f"Unknown category: {start_category}")

        idx = DERIVATION_ORDER.index(start_category)
        categories = DERIVATION_ORDER[idx:]
        return self.regenerate_all(categories, preserve_manual, progress_callback)

    def clear_derived(
        self,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Remove all auto-generated/derived items (preserving manual and defaults).

        Returns:
            Dict of {category: count_removed}
        """
        if categories is None:
            categories = list(DERIVATION_ORDER)

        results = {}
        for category in categories:
            dir_name = _CATEGORY_DIRS.get(category, category)
            cat_dir = self._data_root / dir_name
            if not cat_dir.exists():
                results[category] = 0
                continue

            removed = 0
            for json_file in cat_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if self._should_remove(data):
                        json_file.unlink()
                        removed += 1
                except Exception:
                    pass

            results[category] = removed
            if removed > 0:
                logger.info(f"Cleared {removed} derived items from {category}")

        return results

    def get_category_stats(self) -> Dict[str, Dict]:
        """
        Get stats for each category: total items, derived, manual, confidence.

        Returns:
            Dict of {category: {total, derived, manual, avg_confidence}}
        """
        stats = {}
        for category in DERIVATION_ORDER:
            dir_name = _CATEGORY_DIRS.get(category, category)
            cat_dir = self._data_root / dir_name
            if not cat_dir.exists():
                stats[category] = {
                    'total': 0, 'derived': 0, 'manual': 0,
                    'default': 0, 'avg_confidence': 0.0,
                }
                continue

            total = 0
            derived = 0
            manual = 0
            default = 0
            confidence_sum = 0.0
            confidence_count = 0

            for json_file in cat_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    total += 1

                    source = DerivationTracker.get_source(data)
                    if source is None:
                        default += 1
                    elif source == DerivationSource.MANUAL:
                        manual += 1
                    elif source in (DerivationSource.QUARK_DERIVED,
                                    DerivationSource.PHYSICS_DERIVED,
                                    DerivationSource.AUTO_GENERATED):
                        derived += 1
                        conf = DerivationTracker.get_confidence(data)
                        confidence_sum += conf
                        confidence_count += 1
                    else:
                        default += 1
                except Exception:
                    total += 1

            avg_conf = confidence_sum / confidence_count if confidence_count > 0 else 0.0
            stats[category] = {
                'total': total,
                'derived': derived,
                'manual': manual,
                'default': default,
                'avg_confidence': round(avg_conf, 3),
            }

        return stats

    def check_chain_integrity(self) -> Dict[str, bool]:
        """
        Check whether each category has data populated.

        Returns:
            Dict of {category: has_data}
        """
        integrity = {}
        for category in DERIVATION_ORDER:
            dir_name = _CATEGORY_DIRS.get(category, category)
            cat_dir = self._data_root / dir_name
            if cat_dir.exists():
                count = len(list(cat_dir.glob('*.json')))
                integrity[category] = count > 0
            else:
                integrity[category] = False
        return integrity

    def _regenerate_category(
        self,
        category: str,
        preserve_manual: bool,
        progress_callback: Optional[Callable],
    ) -> int:
        """Regenerate a single category. Returns count generated."""
        if category == 'quarks':
            return 0  # Quarks are the source of truth, never regenerated

        if category == 'subatomic':
            return 0  # Subatomic particles loaded from defaults

        if category == 'elements':
            return self._regenerate_elements(progress_callback)

        if category == 'molecules':
            return self._regenerate_molecules(progress_callback)

        if category == 'alloys':
            return self._regenerate_alloys(progress_callback)

        if category == 'materials':
            return self._regenerate_materials(progress_callback)

        if category in ('amino_acids', 'proteins', 'nucleic_acids',
                        'cell_components', 'cells', 'biomaterials'):
            return self._regenerate_biological(category, progress_callback)

        return 0

    def _regenerate_elements(self, progress_callback) -> int:
        """Regenerate elements from quarks."""
        try:
            from periodica.utils.regeneration_engine import RegenerationEngine
            engine = RegenerationEngine()
            output_dir = str(self._data_root / 'elements')
            count = engine.regenerate_elements(
                progress_callback=progress_callback,
                output_dir=output_dir,
            )
            return count
        except Exception as e:
            logger.error(f"Element regeneration failed: {e}")
            return 0

    def _regenerate_molecules(self, progress_callback) -> int:
        """Regenerate molecules from elements."""
        try:
            from periodica.utils.molecule_generator import MoleculeGenerator
            gen = MoleculeGenerator()
            molecules = gen.generate_all(count_limit=200, progress_callback=progress_callback)
            output_dir = str(self._data_root / 'molecules')
            return gen.save_molecules(molecules, output_dir)
        except Exception as e:
            logger.error(f"Molecule regeneration failed: {e}")
            return 0

    def _regenerate_alloys(self, progress_callback) -> int:
        """Regenerate alloys from elements."""
        try:
            from periodica.utils.alloy_generator import AlloyGenerator
            gen = AlloyGenerator()
            alloys = gen.generate_all(count_limit=50, progress_callback=progress_callback)
            output_dir = str(self._data_root / 'alloys')
            return gen.save_alloys(alloys, output_dir)
        except Exception as e:
            logger.error(f"Alloy regeneration failed: {e}")
            return 0

    def _regenerate_materials(self, progress_callback) -> int:
        """Regenerate materials from alloys."""
        try:
            from periodica.utils.alloy_generator import AlloyGenerator
            from periodica.utils.material_generator import MaterialGenerator
            alloy_gen = AlloyGenerator()
            mat_gen = MaterialGenerator()
            alloys = alloy_gen.generate_all(count_limit=30)
            materials = mat_gen.generate_all(
                alloys, count_limit=30, progress_callback=progress_callback
            )
            output_dir = str(self._data_root / 'materials')
            return mat_gen.save_materials(materials, output_dir)
        except Exception as e:
            logger.error(f"Material regeneration failed: {e}")
            return 0

    def _regenerate_biological(self, category: str, progress_callback) -> int:
        """Regenerate a biological category."""
        try:
            from periodica.utils.biological_generator import BiologicalGenerator
            gen = BiologicalGenerator()
            items = gen.generate_category(
                category, count_limit=20, progress_callback=progress_callback
            )
            return gen.save_items(items, category)
        except Exception as e:
            logger.error(f"Biological regeneration ({category}) failed: {e}")
            return 0

    def _should_remove(self, data: Dict) -> bool:
        """Check if an item should be removed during clear_derived."""
        source = DerivationTracker.get_source(data)
        if source is None:
            return False
        return source in (
            DerivationSource.QUARK_DERIVED,
            DerivationSource.PHYSICS_DERIVED,
            DerivationSource.AUTO_GENERATED,
        )
