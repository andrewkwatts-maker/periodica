"""
Tests for the CascadeRegenerationEngine.
Verifies dependency ordering, selective regeneration,
and chain integrity checking.
"""

import pytest
from periodica.utils.cascade_engine import (
    CascadeRegenerationEngine,
    DERIVATION_ORDER,
)


@pytest.fixture
def engine():
    return CascadeRegenerationEngine()


class TestDerivationOrder:
    def test_quarks_first(self):
        assert DERIVATION_ORDER[0] == 'quarks'

    def test_biomaterials_last(self):
        assert DERIVATION_ORDER[-1] == 'biomaterials'

    def test_elements_before_molecules(self):
        assert DERIVATION_ORDER.index('elements') < DERIVATION_ORDER.index('molecules')

    def test_elements_before_alloys(self):
        assert DERIVATION_ORDER.index('elements') < DERIVATION_ORDER.index('alloys')

    def test_alloys_before_materials(self):
        assert DERIVATION_ORDER.index('alloys') < DERIVATION_ORDER.index('materials')

    def test_amino_acids_before_proteins(self):
        assert DERIVATION_ORDER.index('amino_acids') < DERIVATION_ORDER.index('proteins')

    def test_cells_before_biomaterials(self):
        assert DERIVATION_ORDER.index('cells') < DERIVATION_ORDER.index('biomaterials')

    def test_all_12_categories(self):
        assert len(DERIVATION_ORDER) == 12


class TestDependencies:
    def test_quarks_no_dependencies(self, engine):
        assert engine.get_dependencies('quarks') == []

    def test_elements_depend_on_quarks(self, engine):
        deps = engine.get_dependencies('elements')
        assert 'quarks' in deps

    def test_molecules_depend_on_elements(self, engine):
        deps = engine.get_dependencies('molecules')
        assert 'elements' in deps

    def test_proteins_depend_on_amino_acids(self, engine):
        deps = engine.get_dependencies('proteins')
        assert 'amino_acids' in deps

    def test_biomaterials_depend_on_cells(self, engine):
        deps = engine.get_dependencies('biomaterials')
        assert 'cells' in deps


class TestDownstream:
    def test_quarks_downstream_all(self, engine):
        downstream = engine.get_downstream('quarks')
        assert len(downstream) == 11  # everything except quarks
        assert 'biomaterials' in downstream

    def test_elements_downstream(self, engine):
        downstream = engine.get_downstream('elements')
        assert 'molecules' in downstream
        assert 'alloys' in downstream
        assert 'quarks' not in downstream
        assert 'elements' not in downstream

    def test_biomaterials_no_downstream(self, engine):
        downstream = engine.get_downstream('biomaterials')
        assert downstream == []

    def test_unknown_category_empty(self, engine):
        assert engine.get_downstream('nonexistent') == []


class TestGetCategories:
    def test_returns_all(self, engine):
        cats = engine.get_categories()
        assert len(cats) == 12
        assert cats[0] == 'quarks'
        assert cats[-1] == 'biomaterials'


class TestChainIntegrity:
    def test_returns_dict(self, engine):
        integrity = engine.check_chain_integrity()
        assert isinstance(integrity, dict)
        assert 'quarks' in integrity
        assert 'elements' in integrity
        assert 'biomaterials' in integrity

    def test_quarks_have_data(self, engine):
        integrity = engine.check_chain_integrity()
        assert integrity['quarks'] is True

    def test_elements_have_data(self, engine):
        integrity = engine.check_chain_integrity()
        assert integrity['elements'] is True


class TestCategoryStats:
    def test_returns_stats(self, engine):
        stats = engine.get_category_stats()
        assert isinstance(stats, dict)
        assert 'quarks' in stats
        assert 'elements' in stats

    def test_stats_have_required_keys(self, engine):
        stats = engine.get_category_stats()
        for cat, data in stats.items():
            assert 'total' in data
            assert 'derived' in data
            assert 'manual' in data
            assert 'avg_confidence' in data

    def test_elements_have_items(self, engine):
        stats = engine.get_category_stats()
        assert stats['elements']['total'] > 0


class TestRegenerateFrom:
    def test_invalid_category_raises(self, engine):
        with pytest.raises(ValueError):
            engine.regenerate_from('nonexistent')

    def test_regenerate_from_returns_dict(self, engine):
        # Regenerate just amino acids (lightweight)
        results = engine.regenerate_from('amino_acids')
        assert isinstance(results, dict)
        assert 'amino_acids' in results

    def test_regenerate_from_molecules(self, engine):
        # Test that regenerating from molecules includes downstream
        results = engine.regenerate_from('molecules')
        assert 'molecules' in results


class TestProgressCallback:
    def test_callback_called(self, engine):
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        # Just regenerate amino acids to keep it fast
        engine.regenerate_all(
            categories=['amino_acids'],
            progress_callback=cb,
        )
        assert len(calls) > 0
        assert calls[-1][0] == 100
