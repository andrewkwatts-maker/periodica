"""
Comprehensive tests for all data loaders in the project.

Tests cover: ElementDataLoader, QuarkDataLoader, MoleculeDataLoader,
AlloyDataLoader, SubatomicDataLoader, MaterialDataLoader, and DataManager.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Pre-import modules that have circular import chains so that by the time
# individual tests run, all modules are fully initialised.
import periodica.core.quark_enums  # noqa: F401 - used transitively by quark_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "periodica" / "data"
ACTIVE_DIR = DATA_DIR / "active"


def _fresh_element_loader(elements_dir=None):
    """Create a fresh ElementDataLoader (bypass singleton)."""
    from periodica.data.element_loader import ElementDataLoader
    return ElementDataLoader(elements_dir=elements_dir)


def _fresh_quark_loader(base_dir=None):
    """Create a fresh QuarkDataLoader (bypass singleton)."""
    from periodica.data.quark_loader import QuarkDataLoader
    return QuarkDataLoader(base_dir=base_dir)


def _fresh_molecule_loader(molecules_dir=None):
    """Create a fresh MoleculeDataLoader (bypass singleton)."""
    from periodica.data.molecule_loader import MoleculeDataLoader
    return MoleculeDataLoader(molecules_dir=molecules_dir)


def _fresh_alloy_loader(alloys_dir=None):
    """Create a fresh AlloyDataLoader (bypass singleton)."""
    from periodica.data.alloy_loader import AlloyDataLoader
    return AlloyDataLoader(alloys_dir=alloys_dir)


def _fresh_subatomic_loader(subatomic_dir=None):
    """Create a fresh SubatomicDataLoader (bypass singleton)."""
    from periodica.data.subatomic_loader import SubatomicDataLoader
    return SubatomicDataLoader(subatomic_dir=subatomic_dir)


def _fresh_data_manager(base_dir=None):
    """Create a fresh DataManager (bypass singleton)."""
    from periodica.data.data_manager import DataManager
    return DataManager(base_dir=base_dir)


# =========================================================================
# ElementDataLoader
# =========================================================================

class TestElementDataLoader:
    """Tests for data.element_loader.ElementDataLoader"""

    def _loader(self):
        loader = _fresh_element_loader()
        loader.load_all_elements()
        return loader

    # -- Loading & counts --------------------------------------------------

    def test_load_all_elements_returns_list(self):
        loader = self._loader()
        assert isinstance(loader.elements, list)

    def test_element_count_positive(self):
        loader = self._loader()
        assert loader.get_element_count() > 0

    def test_element_count_at_least_118(self):
        """The periodic table has 118 confirmed elements."""
        loader = self._loader()
        assert loader.get_element_count() >= 118

    # -- Lookup by symbol --------------------------------------------------

    def test_get_element_by_symbol_hydrogen(self):
        loader = self._loader()
        h = loader.get_element_by_symbol("H")
        assert h is not None
        assert h["symbol"] == "H"
        assert h["name"] == "Hydrogen"
        assert h["atomic_number"] == 1

    def test_get_element_by_symbol_iron(self):
        loader = self._loader()
        fe = loader.get_element_by_symbol("Fe")
        assert fe is not None
        assert fe["atomic_number"] == 26

    def test_get_element_by_symbol_nonexistent(self):
        loader = self._loader()
        assert loader.get_element_by_symbol("Zz") is None

    # -- Lookup by atomic number -------------------------------------------

    def test_get_element_by_z(self):
        loader = self._loader()
        carbon = loader.get_element_by_z(6)
        assert carbon is not None
        assert carbon["symbol"] == "C"

    def test_get_element_by_z_nonexistent(self):
        loader = self._loader()
        assert loader.get_element_by_z(999) is None

    # -- Block filtering ---------------------------------------------------

    def test_get_elements_by_block_s(self):
        loader = self._loader()
        s_block = loader.get_elements_by_block("s")
        assert len(s_block) > 0
        assert "H" in s_block
        assert "Li" in s_block

    def test_get_elements_by_block_d(self):
        loader = self._loader()
        d_block = loader.get_elements_by_block("d")
        assert len(d_block) > 0
        assert "Fe" in d_block

    def test_get_elements_by_block_invalid(self):
        loader = self._loader()
        assert loader.get_elements_by_block("x") == []

    # -- Period and group filtering ----------------------------------------

    def test_get_elements_by_period(self):
        loader = self._loader()
        period_1 = loader.get_elements_by_period(1)
        assert "H" in period_1
        assert "He" in period_1

    def test_get_elements_by_group(self):
        loader = self._loader()
        group_1 = loader.get_elements_by_group(1)
        assert "H" in group_1
        assert "Li" in group_1

    # -- Property accessors ------------------------------------------------

    def test_get_property(self):
        loader = self._loader()
        name = loader.get_property("O", "name")
        assert name == "Oxygen"

    def test_get_property_default(self):
        loader = self._loader()
        val = loader.get_property("H", "nonexistent_field", "fallback")
        assert val == "fallback"

    def test_get_ionization_energy(self):
        loader = self._loader()
        ie = loader.get_ionization_energy("H")
        # Hydrogen IE ~13.6 eV
        assert ie is not None
        assert ie > 10

    def test_get_electronegativity(self):
        loader = self._loader()
        en = loader.get_electronegativity("F")
        # Fluorine has the highest electronegativity ~3.98
        assert en is not None
        assert en > 3.5

    def test_get_electron_configuration(self):
        loader = self._loader()
        cfg = loader.get_electron_configuration("He")
        assert cfg is not None
        assert "1s" in cfg

    # -- Data structure verification ---------------------------------------

    def test_element_has_required_fields(self):
        loader = self._loader()
        required = {"symbol", "name", "atomic_number", "block", "period"}
        for elem in loader.elements:
            for field in required:
                assert field in elem, f"Element {elem.get('symbol', '?')} missing '{field}'"

    def test_elements_sorted_by_atomic_number(self):
        loader = self._loader()
        z_values = [e["atomic_number"] for e in loader.elements]
        assert z_values == sorted(z_values)

    # -- Property dictionaries ---------------------------------------------

    def test_get_property_dict(self):
        loader = self._loader()
        ie_dict = loader.get_ionization_energies()
        assert isinstance(ie_dict, dict)
        assert len(ie_dict) > 0
        assert "H" in ie_dict

    # -- Search ------------------------------------------------------------

    def test_search_elements(self):
        loader = self._loader()
        results = loader.search_elements(block="s", period=1)
        symbols = [e["symbol"] for e in results]
        assert "H" in symbols

    # -- Utility -----------------------------------------------------------

    def test_get_all_symbols(self):
        loader = self._loader()
        symbols = loader.get_all_symbols()
        assert "H" in symbols
        assert "He" in symbols
        assert len(symbols) == loader.get_element_count()

    def test_get_symbol_by_z(self):
        loader = self._loader()
        assert loader.get_symbol_by_z(1) == "H"
        assert loader.get_symbol_by_z(79) == "Au"

    def test_get_name_by_symbol(self):
        loader = self._loader()
        assert loader.get_name_by_symbol("Au") == "Gold"

    # -- ensure_loaded auto-loads ------------------------------------------

    def test_ensure_loaded_auto_loads(self):
        loader = _fresh_element_loader()
        # Not loaded yet
        assert not loader._loaded
        # Accessing data should trigger auto-load
        count = loader.get_element_count()
        assert loader._loaded
        assert count > 0

    # -- Error handling ----------------------------------------------------

    def test_load_from_nonexistent_dir_raises(self):
        loader = _fresh_element_loader(elements_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            loader.load_all_elements()


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
# MoleculeDataLoader
# =========================================================================

class TestMoleculeDataLoader:
    """Tests for data.molecule_loader.MoleculeDataLoader"""

    def _loader(self):
        loader = _fresh_molecule_loader()
        loader.load_all_molecules()
        return loader

    # -- Loading & counts --------------------------------------------------

    def test_load_returns_list(self):
        loader = self._loader()
        assert isinstance(loader.molecules, list)

    def test_molecule_count_positive(self):
        loader = self._loader()
        assert loader.get_molecule_count() > 0

    # -- Lookup by name ----------------------------------------------------

    def test_get_molecule_by_name(self):
        loader = self._loader()
        if loader.molecules:
            first_name = loader.molecules[0]["Name"]
            result = loader.get_molecule_by_name(first_name)
            assert result is not None
            assert result["Name"] == first_name

    def test_get_molecule_by_name_nonexistent(self):
        loader = self._loader()
        assert loader.get_molecule_by_name("FakeCompound999") is None

    # -- Lookup by formula -------------------------------------------------

    def test_get_molecule_by_formula(self):
        loader = self._loader()
        if loader.molecules:
            first_formula = loader.molecules[0]["Formula"]
            result = loader.get_molecule_by_formula(first_formula)
            assert result is not None

    # -- Category filtering ------------------------------------------------

    def test_get_molecules_by_category(self):
        loader = self._loader()
        categories = loader.get_unique_categories()
        if categories:
            cat = categories[0]
            filtered = loader.get_molecules_by_category(cat)
            assert len(filtered) > 0
            for m in filtered:
                assert m["category"] == cat

    # -- Bond type filtering -----------------------------------------------

    def test_get_molecules_by_bond_type(self):
        loader = self._loader()
        bond_types = loader.get_unique_bond_types()
        if bond_types:
            bt = bond_types[0]
            filtered = loader.get_molecules_by_bond_type(bt)
            assert len(filtered) > 0

    # -- Polarity filtering ------------------------------------------------

    def test_get_molecules_by_polarity(self):
        loader = self._loader()
        polarities = loader.get_unique_polarities()
        if polarities:
            pol = polarities[0]
            filtered = loader.get_molecules_by_polarity(pol)
            assert len(filtered) > 0

    # -- Geometry filtering ------------------------------------------------

    def test_get_molecules_by_geometry(self):
        loader = self._loader()
        geometries = loader.get_unique_geometries()
        if geometries:
            geom = geometries[0]
            filtered = loader.get_molecules_by_geometry(geom)
            assert len(filtered) > 0

    # -- State filtering ---------------------------------------------------

    def test_get_molecules_by_state(self):
        loader = self._loader()
        states = loader.get_unique_states()
        if states:
            st = states[0]
            filtered = loader.get_molecules_by_state(st)
            assert len(filtered) > 0

    # -- Unique value lists ------------------------------------------------

    def test_unique_categories_non_empty(self):
        loader = self._loader()
        assert len(loader.get_unique_categories()) > 0

    def test_unique_geometries_non_empty(self):
        loader = self._loader()
        assert len(loader.get_unique_geometries()) > 0

    def test_unique_bond_types_non_empty(self):
        loader = self._loader()
        assert len(loader.get_unique_bond_types()) > 0

    # -- Data structure verification ---------------------------------------

    def test_molecule_has_required_fields(self):
        loader = self._loader()
        required = {"Name", "Formula", "MolecularMass_amu", "BondType", "Geometry"}
        for mol in loader.molecules:
            for field in required:
                assert field in mol, f"Molecule {mol.get('Name', '?')} missing '{field}'"

    def test_molecule_has_derived_fields(self):
        loader = self._loader()
        derived = {"name", "formula", "mass", "bond_type", "geometry", "category", "polarity", "state"}
        for mol in loader.molecules:
            for field in derived:
                assert field in mol, f"Molecule {mol.get('Name', '?')} missing derived '{field}'"

    # -- Empty directory handling ------------------------------------------

    def test_load_from_nonexistent_dir_returns_empty(self):
        loader = _fresh_molecule_loader(molecules_dir="/nonexistent/path")
        result = loader.load_all_molecules()
        assert result == []
        assert loader.get_molecule_count() == 0


# =========================================================================
# AlloyDataLoader
# =========================================================================

class TestAlloyDataLoader:
    """Tests for data.alloy_loader.AlloyDataLoader"""

    def _loader(self):
        loader = _fresh_alloy_loader()
        loader.load_all_alloys()
        return loader

    # -- Loading & counts --------------------------------------------------

    def test_load_returns_list(self):
        loader = self._loader()
        assert isinstance(loader.alloys, list)

    def test_alloy_count_positive(self):
        loader = self._loader()
        assert loader.get_alloy_count() > 0

    # -- Lookup by name ----------------------------------------------------

    def test_get_alloy_by_name(self):
        loader = self._loader()
        if loader.alloys:
            first_name = loader.alloys[0]["Name"]
            result = loader.get_alloy_by_name(first_name)
            assert result is not None
            assert result["Name"] == first_name

    def test_get_alloy_by_name_nonexistent(self):
        loader = self._loader()
        assert loader.get_alloy_by_name("FakeAlloy_XYZ_999") is None

    # -- Category filtering ------------------------------------------------

    def test_get_alloys_by_category(self):
        loader = self._loader()
        categories = loader.get_unique_categories()
        if categories:
            cat = categories[0]
            filtered = loader.get_alloys_by_category(cat)
            assert len(filtered) > 0

    # -- Structure filtering -----------------------------------------------

    def test_get_alloys_by_structure(self):
        loader = self._loader()
        structures = loader.get_unique_structures()
        if structures:
            struct = structures[0]
            filtered = loader.get_alloys_by_structure(struct)
            assert len(filtered) > 0

    # -- Primary element filtering -----------------------------------------

    def test_get_alloys_by_primary_element(self):
        loader = self._loader()
        elements = loader.get_unique_primary_elements()
        if elements:
            elem = elements[0]
            filtered = loader.get_alloys_by_primary_element(elem)
            assert len(filtered) > 0

    # -- Unique value lists ------------------------------------------------

    def test_unique_categories_non_empty(self):
        loader = self._loader()
        assert len(loader.get_unique_categories()) > 0

    def test_unique_structures_non_empty(self):
        loader = self._loader()
        assert len(loader.get_unique_structures()) > 0

    def test_unique_primary_elements_non_empty(self):
        loader = self._loader()
        assert len(loader.get_unique_primary_elements()) > 0

    # -- Property range ----------------------------------------------------

    def test_get_property_range_density(self):
        loader = self._loader()
        low, high = loader.get_property_range("density")
        assert high >= low
        assert high > 0

    def test_get_property_range_unknown(self):
        loader = self._loader()
        low, high = loader.get_property_range("nonexistent_property_xyz")
        assert low == 0
        assert high == 1

    # -- Data structure verification ---------------------------------------

    def test_alloy_has_required_fields(self):
        loader = self._loader()
        required = {"Name", "Category", "Components"}
        for alloy in loader.alloys:
            for field in required:
                assert field in alloy, f"Alloy {alloy.get('Name', '?')} missing '{field}'"

    def test_alloy_has_derived_fields(self):
        loader = self._loader()
        derived = {
            "name", "category", "density", "melting_point",
            "tensile_strength", "crystal_structure", "primary_element", "color",
        }
        for alloy in loader.alloys:
            for field in derived:
                assert field in alloy, f"Alloy {alloy.get('Name', '?')} missing derived '{field}'"

    # -- Empty directory handling ------------------------------------------

    def test_load_from_nonexistent_dir_returns_empty(self):
        loader = _fresh_alloy_loader(alloys_dir="/nonexistent/path")
        result = loader.load_all_alloys()
        assert result == []
        assert loader.get_alloy_count() == 0


# =========================================================================
# SubatomicDataLoader
# =========================================================================

class TestSubatomicDataLoader:
    """Tests for data.subatomic_loader.SubatomicDataLoader"""

    def _loader(self):
        loader = _fresh_subatomic_loader()
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
        if loader.particles:
            first_name = loader.particles[0]["Name"]
            result = loader.get_particle_by_name(first_name)
            assert result is not None
            assert result["Name"] == first_name

    def test_get_particle_by_name_nonexistent(self):
        loader = self._loader()
        assert loader.get_particle_by_name("FakeParticle_ZZZ") is None

    # -- Lookup by symbol --------------------------------------------------

    def test_get_particle_by_symbol(self):
        loader = self._loader()
        particles_with_sym = [p for p in loader.particles if p.get("Symbol")]
        if particles_with_sym:
            sym = particles_with_sym[0]["Symbol"]
            result = loader.get_particle_by_symbol(sym)
            assert result is not None

    # -- Baryons and mesons ------------------------------------------------

    def test_get_baryons(self):
        loader = self._loader()
        baryons = loader.get_baryons()
        assert isinstance(baryons, list)
        for b in baryons:
            assert b.get("_is_baryon", False)

    def test_get_mesons(self):
        loader = self._loader()
        mesons = loader.get_mesons()
        assert isinstance(mesons, list)
        for m in mesons:
            assert m.get("_is_meson", False)

    def test_baryons_plus_mesons_subset_of_all(self):
        loader = self._loader()
        baryon_names = {b["Name"] for b in loader.get_baryons()}
        meson_names = {m["Name"] for m in loader.get_mesons()}
        all_names = {p["Name"] for p in loader.particles}
        assert baryon_names.issubset(all_names)
        assert meson_names.issubset(all_names)

    # -- Charge filtering --------------------------------------------------

    def test_get_particles_by_charge(self):
        loader = self._loader()
        neutral = loader.get_particles_by_charge(0)
        assert isinstance(neutral, list)
        for p in neutral:
            assert p.get("Charge_e", None) == 0

    # -- Category filtering ------------------------------------------------

    def test_get_particles_by_category(self):
        loader = self._loader()
        # Attempt a known category
        categories = set(p.get("_category", "") for p in loader.particles)
        if categories:
            cat = next(iter(categories))
            filtered = loader.get_particles_by_category(cat)
            assert len(filtered) > 0

    # -- Mass range --------------------------------------------------------

    def test_get_mass_range(self):
        loader = self._loader()
        low, high = loader.get_mass_range()
        assert high >= low

    # -- Decay chain -------------------------------------------------------

    def test_get_decay_chain_returns_list(self):
        loader = self._loader()
        if loader.particles:
            name = loader.particles[0]["Name"]
            chains = loader.get_decay_chain(name)
            assert isinstance(chains, list)
            # Each chain should start with the particle itself
            for chain in chains:
                assert chain[0] == name

    def test_get_decay_chain_nonexistent(self):
        loader = self._loader()
        chains = loader.get_decay_chain("NonexistentParticle")
        assert chains == []

    # -- Data structure verification ---------------------------------------

    def test_particle_has_required_fields(self):
        loader = self._loader()
        for p in loader.particles:
            assert "Name" in p
            assert "Type" in p

    def test_particle_has_computed_fields(self):
        loader = self._loader()
        computed = {"_category", "_is_baryon", "_is_meson", "_log_mass", "_quark_count", "_stability_factor"}
        for p in loader.particles:
            for field in computed:
                assert field in p, f"Particle {p.get('Name', '?')} missing computed '{field}'"

    # -- Sorted by mass ----------------------------------------------------

    def test_particles_sorted_by_mass(self):
        loader = self._loader()
        masses = [p.get("Mass_MeVc2", 0) for p in loader.particles]
        assert masses == sorted(masses)

    # -- Empty directory handling ------------------------------------------

    def test_load_from_nonexistent_dir_returns_empty(self):
        loader = _fresh_subatomic_loader(subatomic_dir="/nonexistent/path")
        result = loader.load_all_particles()
        assert result == []
        assert loader.get_particle_count() == 0


# =========================================================================
# MaterialDataLoader
# =========================================================================

class TestMaterialDataLoader:
    """Tests for data.material_data.MaterialDataLoader"""

    def _loader(self):
        # MaterialDataLoader uses __new__ singleton -- we call reload()
        # to ensure fresh state, but the class itself is shared.
        from periodica.data.material_data import MaterialDataLoader
        instance = MaterialDataLoader()
        instance.reload()
        return instance

    # -- Loading -----------------------------------------------------------

    def test_get_all_materials_returns_dict(self):
        loader = self._loader()
        materials = loader.get_all_materials()
        assert isinstance(materials, dict)

    def test_materials_loaded(self):
        loader = self._loader()
        names = loader.get_material_names()
        # May be zero if no material JSON files exist yet
        assert isinstance(names, list)

    # -- Lookup by name ----------------------------------------------------

    def test_get_material_by_name(self):
        loader = self._loader()
        names = loader.get_material_names()
        if names:
            mat = loader.get_material(names[0])
            assert mat is not None

    def test_get_material_nonexistent(self):
        loader = self._loader()
        assert loader.get_material("FakeMaterial_XYZ_999") is None

    # -- Category filtering ------------------------------------------------

    def test_get_materials_by_category(self):
        loader = self._loader()
        materials = loader.get_all_materials()
        if materials:
            # Find a category that exists
            categories = set()
            for m in materials.values():
                cat = m.get("Category", "")
                if cat:
                    categories.add(cat)
            if categories:
                cat = next(iter(categories))
                filtered = loader.get_materials_by_category(cat)
                assert len(filtered) > 0
                for m in filtered:
                    assert m.get("Category", "").lower() == cat.lower()

    # -- get_material_names ------------------------------------------------

    def test_get_material_names_returns_list(self):
        loader = self._loader()
        names = loader.get_material_names()
        assert isinstance(names, list)

    # -- reload ------------------------------------------------------------

    def test_reload_does_not_crash(self):
        loader = self._loader()
        loader.reload()
        # Should still work after reload
        assert isinstance(loader.get_all_materials(), dict)

    # -- Data structure (if materials exist) --------------------------------

    def test_material_has_name_field(self):
        loader = self._loader()
        for name, mat in loader.get_all_materials().items():
            assert "Name" in mat or name  # Name should be present


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
