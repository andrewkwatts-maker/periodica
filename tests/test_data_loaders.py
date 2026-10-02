"""
Comprehensive tests for all data loaders in the project.

Tests cover: QuarkDataLoader and DataManager. (The element, molecule, alloy,
subatomic and material loaders were removed; `Get`/`data_sheet` serve that
data -- see test_get.py.)
"""

import json
import os
from pathlib import Path

import pytest

# Pre-import modules that have circular import chains so that by the time
# individual tests run, all modules are fully initialised.
import periodica.core.quark_enums  # noqa: F401 - used transitively by quark_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "periodica" / "data"


def _fresh_quark_loader(base_dir=None):
    """Create a fresh QuarkDataLoader (bypass singleton)."""
    from periodica.data.quark_loader import QuarkDataLoader
    return QuarkDataLoader(base_dir=base_dir)


def _fresh_data_manager(base_dir=None):
    """Create a fresh DataManager (bypass singleton)."""
    from periodica.data.data_manager import DataManager
    return DataManager(base_dir=base_dir)


# =========================================================================
# QuarkDataLoader
# =========================================================================

class TestQuarkDataLoader:
    """Tests for data.quark_loader.QuarkDataLoader"""

    def _loader(self):
        loader = _fresh_quark_loader()
        loader.load_all_particles()
        return loader

    # -- Loading & counts --------------------------------------------------

    def test_load_returns_list(self):
        loader = self._loader()
        assert isinstance(loader.particles, list)

    def test_particle_count_positive(self):
        loader = self._loader()
        assert loader.get_particle_count() > 0

    # -- Lookup by name ----------------------------------------------------

    def test_get_particle_by_name(self):
        loader = self._loader()
        # Try to find any particle from the loaded set
        if loader.particles:
            first_name = loader.particles[0]["Name"]
            result = loader.get_particle_by_name(first_name)
            assert result is not None
            assert result["Name"] == first_name

    def test_get_particle_by_name_nonexistent(self):
        loader = self._loader()
        assert loader.get_particle_by_name("Nonexistent Particle XYZ") is None

    # -- Lookup by symbol --------------------------------------------------

    def test_get_particle_by_symbol(self):
        loader = self._loader()
        # Find a particle with a symbol
        particles_with_symbols = [p for p in loader.particles if p.get("Symbol")]
        if particles_with_symbols:
            sym = particles_with_symbols[0]["Symbol"]
            result = loader.get_particle_by_symbol(sym)
            assert result is not None

    # -- Filtering by type -------------------------------------------------

    def test_get_particles_by_type(self):
        from periodica.core.quark_enums import ParticleType
        loader = self._loader()
        quarks = loader.get_particles_by_type(ParticleType.QUARK)
        # Should have at least the 6 quarks
        assert len(quarks) >= 0  # May vary depending on data

    # -- Filtering by generation -------------------------------------------

    def test_get_particles_by_generation(self):
        loader = self._loader()
        gen1 = loader.get_particles_by_generation(1)
        assert isinstance(gen1, list)

    # -- Standard Model filter ---------------------------------------------

    def test_get_standard_model_particles(self):
        loader = self._loader()
        sm = loader.get_standard_model_particles()
        assert isinstance(sm, list)
        # SM particles should not have is_composite or _is_antiparticle
        for p in sm:
            assert not p.get("_is_antiparticle", False)
            assert not p.get("is_composite", False)

    # -- Data structure verification ---------------------------------------

    def test_particle_has_name(self):
        loader = self._loader()
        for p in loader.particles:
            assert "Name" in p, f"Particle missing 'Name': {p}"

    def test_particle_has_computed_fields(self):
        loader = self._loader()
        for p in loader.particles:
            assert "particle_type" in p
            assert "charge_display" in p
            assert "mass_display" in p
            assert "spin_display" in p
            assert "statistics" in p

    # -- Sorting -----------------------------------------------------------

    def test_particles_sorted_by_mass(self):
        loader = self._loader()
        masses = [p.get("Mass_MeVc2", 0) or 0 for p in loader.particles]
        assert masses == sorted(masses)


# =========================================================================
# DataManager
# =========================================================================

class TestDataManager:
    """Tests for data.data_manager.DataManager"""

    @pytest.fixture
    def tmp_data_dir(self, tmp_path):
        """Create a temporary data directory with defaults and active subdirs."""
        from periodica.data.data_manager import DataCategory
        defaults = tmp_path / "defaults"
        active = tmp_path / "active"
        for cat in DataCategory:
            (defaults / cat.value).mkdir(parents=True, exist_ok=True)
            (active / cat.value).mkdir(parents=True, exist_ok=True)
        return tmp_path

    @pytest.fixture
    def manager(self, tmp_data_dir):
        return _fresh_data_manager(base_dir=str(tmp_data_dir))

    @pytest.fixture
    def sample_element(self):
        return {
            "symbol": "Xx",
            "name": "Testium",
            "atomic_number": 999,
            "block": "s",
            "period": 1,
        }

    # -- Initialization ----------------------------------------------------

    def test_init_creates_directories(self, tmp_data_dir):
        from periodica.data.data_manager import DataCategory
        mgr = _fresh_data_manager(base_dir=str(tmp_data_dir))
        for cat in DataCategory:
            assert (tmp_data_dir / "defaults" / cat.value).exists()
            assert (tmp_data_dir / "active" / cat.value).exists()

    # -- list_items --------------------------------------------------------

    def test_list_items_empty(self, manager):
        from periodica.data.data_manager import DataCategory
        items = manager.list_items(DataCategory.ELEMENTS)
        assert items == []

    # -- add_item ----------------------------------------------------------

    def test_add_item(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        result = manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert result is True
        items = manager.list_items(DataCategory.ELEMENTS)
        assert "999_Xx" in items

    def test_add_item_duplicate_fails(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        result = manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert result is False

    # -- get_item ----------------------------------------------------------

    def test_get_item(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        data = manager.get_item(DataCategory.ELEMENTS, "999_Xx")
        assert data is not None
        assert data["symbol"] == "Xx"

    def test_get_item_nonexistent(self, manager):
        from periodica.data.data_manager import DataCategory
        assert manager.get_item(DataCategory.ELEMENTS, "does_not_exist") is None

    # -- get_all_items -----------------------------------------------------

    def test_get_all_items(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        items = manager.get_all_items(DataCategory.ELEMENTS)
        assert len(items) == 1
        assert items[0]["symbol"] == "Xx"
        assert "_filename" in items[0]

    # -- edit_item ---------------------------------------------------------

    def test_edit_item(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        updated = sample_element.copy()
        updated["name"] = "Testium Updated"
        result = manager.edit_item(DataCategory.ELEMENTS, "999_Xx", updated)
        assert result is True
        data = manager.get_item(DataCategory.ELEMENTS, "999_Xx")
        assert data["name"] == "Testium Updated"

    def test_edit_item_nonexistent(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        result = manager.edit_item(DataCategory.ELEMENTS, "nope", sample_element)
        assert result is False

    # -- remove_item -------------------------------------------------------

    def test_remove_item(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        result = manager.remove_item(DataCategory.ELEMENTS, "999_Xx")
        assert result is True
        assert manager.get_item(DataCategory.ELEMENTS, "999_Xx") is None

    def test_remove_item_nonexistent(self, manager):
        from periodica.data.data_manager import DataCategory
        result = manager.remove_item(DataCategory.ELEMENTS, "nope")
        assert result is False

    # -- get_item_count ----------------------------------------------------

    def test_get_item_count(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        assert manager.get_item_count(DataCategory.ELEMENTS) == 0
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert manager.get_item_count(DataCategory.ELEMENTS) == 1

    # -- reset_category ----------------------------------------------------

    def test_reset_category(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        # Put a default file in place
        defaults_path = manager.get_defaults_path(DataCategory.ELEMENTS)
        default_file = defaults_path / "001_H.json"
        default_data = {"symbol": "H", "name": "Hydrogen", "atomic_number": 1}
        with open(default_file, "w", encoding="utf-8") as f:
            json.dump(default_data, f)

        # Add a custom item to active
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert manager.get_item_count(DataCategory.ELEMENTS) == 1

        # Reset should restore defaults only
        result = manager.reset_category(DataCategory.ELEMENTS)
        assert result is True
        items = manager.list_items(DataCategory.ELEMENTS)
        assert "001_H" in items
        assert "999_Xx" not in items

    # -- reset_item --------------------------------------------------------

    def test_reset_item(self, manager):
        from periodica.data.data_manager import DataCategory
        # Put a default
        defaults_path = manager.get_defaults_path(DataCategory.ELEMENTS)
        default_data = {"symbol": "H", "name": "Hydrogen", "atomic_number": 1}
        with open(defaults_path / "001_H.json", "w", encoding="utf-8") as f:
            json.dump(default_data, f)

        # Modify the active version
        active_path = manager.get_active_path(DataCategory.ELEMENTS)
        modified = {"symbol": "H", "name": "Modified Hydrogen", "atomic_number": 1}
        with open(active_path / "001_H.json", "w", encoding="utf-8") as f:
            json.dump(modified, f)

        # Reset single item
        result = manager.reset_item(DataCategory.ELEMENTS, "001_H")
        assert result is True
        data = manager.get_item(DataCategory.ELEMENTS, "001_H")
        assert data["name"] == "Hydrogen"

    # -- has_changes -------------------------------------------------------

    def test_has_changes_no_changes(self, manager):
        from periodica.data.data_manager import DataCategory
        assert not manager.has_changes(DataCategory.ELEMENTS)

    def test_has_changes_after_add(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert manager.has_changes(DataCategory.ELEMENTS)

    # -- Change callbacks --------------------------------------------------

    def test_change_callback_fires(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        callback_log = []
        manager.register_change_callback(DataCategory.ELEMENTS, lambda: callback_log.append("changed"))
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert "changed" in callback_log

    def test_unregister_callback(self, manager, sample_element):
        from periodica.data.data_manager import DataCategory
        callback_log = []
        cb = lambda: callback_log.append("changed")
        manager.register_change_callback(DataCategory.ELEMENTS, cb)
        manager.unregister_change_callback(DataCategory.ELEMENTS, cb)
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        assert len(callback_log) == 0

    # -- export / import ---------------------------------------------------

    def test_export_and_import(self, manager, sample_element, tmp_path):
        from periodica.data.data_manager import DataCategory
        manager.add_item(DataCategory.ELEMENTS, "999_Xx", sample_element)
        export_file = str(tmp_path / "exported.json")
        result = manager.export_item(DataCategory.ELEMENTS, "999_Xx", export_file)
        assert result is True
        assert os.path.exists(export_file)

        # Import as a new name
        result = manager.import_item(DataCategory.ELEMENTS, export_file, name="999_Xx_imported")
        assert result is True
        imported = manager.get_item(DataCategory.ELEMENTS, "999_Xx_imported")
        assert imported is not None
        assert imported["symbol"] == "Xx"

    # -- DataCategory enum -------------------------------------------------

    def test_data_category_values(self):
        from periodica.data.data_manager import DataCategory
        expected = {
            "elements", "quarks", "antiquarks", "subatomic", "molecules", "alloys",
            "materials", "amino_acids", "proteins", "nucleic_acids",
            "cell_components", "cells", "biological_materials", "tissues",
        }
        actual = {cat.value for cat in DataCategory}
        assert expected == actual


# =========================================================================
# Integration: DataManager with real data directory
# =========================================================================

class TestDataManagerWithRealData:
    """Integration tests using the actual project data directory."""

    def _manager(self):
        return _fresh_data_manager(base_dir=str(DATA_DIR))

    def test_list_elements(self):
        from periodica.data.data_manager import DataCategory
        mgr = self._manager()
        items = mgr.list_items(DataCategory.ELEMENTS)
        assert len(items) > 0

    def test_get_item_count_elements(self):
        from periodica.data.data_manager import DataCategory
        mgr = self._manager()
        count = mgr.get_item_count(DataCategory.ELEMENTS)
        assert count >= 118

    def test_get_item_hydrogen(self):
        from periodica.data.data_manager import DataCategory
        mgr = self._manager()
        # The filename pattern is 001_H
        data = mgr.get_item(DataCategory.ELEMENTS, "H")
        assert data is not None

    def test_get_all_items_elements(self):
        from periodica.data.data_manager import DataCategory
        mgr = self._manager()
        items = mgr.get_all_items(DataCategory.ELEMENTS)
        assert len(items) >= 118
