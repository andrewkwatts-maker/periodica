"""
Tests for the BiologicalGenerator.
Verifies auto-generation of amino acids, proteins, nucleic acids,
cell components, cells, and biomaterials.
"""

import pytest
from periodica.utils.biological_generator import BiologicalGenerator


@pytest.fixture
def gen():
    return BiologicalGenerator()


class TestAminoAcids:
    def test_generates_standard_aa(self, gen):
        aas = gen.generate_standard_amino_acids(count_limit=20)
        assert len(aas) == 20

    def test_all_have_name(self, gen):
        aas = gen.generate_standard_amino_acids(count_limit=5)
        for aa in aas:
            assert 'name' in aa
            assert len(aa['name']) > 0

    def test_all_have_symbol(self, gen):
        aas = gen.generate_standard_amino_acids(count_limit=5)
        for aa in aas:
            assert 'symbol' in aa
            assert len(aa['symbol']) == 1

    def test_all_have_derivation(self, gen):
        aas = gen.generate_standard_amino_acids(count_limit=5)
        for aa in aas:
            assert '_derivation' in aa
            assert aa['_derivation']['source'] == 'auto_generated'

    def test_progress_callback(self, gen):
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        gen.generate_standard_amino_acids(count_limit=5, progress_callback=cb)
        assert len(calls) > 0
        assert calls[-1][0] == 100

    def test_isoelectric_point_present(self, gen):
        aas = gen.generate_standard_amino_acids(count_limit=5)
        for aa in aas:
            assert 'isoelectric_point' in aa
            assert 0 < aa['isoelectric_point'] < 14


class TestProteins:
    def test_generates_proteins(self, gen):
        prots = gen.generate_proteins(count_limit=3)
        assert len(prots) > 0
        assert len(prots) <= 3

    def test_all_have_name(self, gen):
        prots = gen.generate_proteins(count_limit=3)
        for p in prots:
            assert 'name' in p

    def test_all_have_sequence(self, gen):
        prots = gen.generate_proteins(count_limit=3)
        for p in prots:
            assert 'sequence' in p
            assert len(p['sequence']) > 10

    def test_all_have_mass(self, gen):
        prots = gen.generate_proteins(count_limit=3)
        for p in prots:
            assert 'molecular_mass' in p
            assert p['molecular_mass'] > 0

    def test_all_have_derivation(self, gen):
        prots = gen.generate_proteins(count_limit=2)
        for p in prots:
            assert '_derivation' in p
            assert p['_derivation']['source'] == 'auto_generated'


class TestNucleicAcids:
    def test_generates_nucleic_acids(self, gen):
        nas = gen.generate_nucleic_acids(count_limit=3)
        assert len(nas) > 0
        assert len(nas) <= 3

    def test_all_have_name(self, gen):
        nas = gen.generate_nucleic_acids(count_limit=3)
        for na in nas:
            assert 'name' in na

    def test_all_have_sequence(self, gen):
        nas = gen.generate_nucleic_acids(count_limit=3)
        for na in nas:
            assert 'sequence' in na
            assert len(na['sequence']) > 5

    def test_all_have_derivation(self, gen):
        nas = gen.generate_nucleic_acids(count_limit=2)
        for na in nas:
            assert '_derivation' in na


class TestCellComponents:
    def test_generates_components(self, gen):
        comps = gen.generate_cell_components(count_limit=3)
        assert len(comps) > 0
        assert len(comps) <= 3

    def test_all_have_name(self, gen):
        comps = gen.generate_cell_components(count_limit=3)
        for c in comps:
            assert 'name' in c

    def test_all_have_type(self, gen):
        comps = gen.generate_cell_components(count_limit=3)
        for c in comps:
            assert 'component_type' in c

    def test_all_have_derivation(self, gen):
        comps = gen.generate_cell_components(count_limit=2)
        for c in comps:
            assert '_derivation' in c


class TestCells:
    def test_generates_cells(self, gen):
        cells = gen.generate_cells(count_limit=3)
        assert len(cells) > 0
        assert len(cells) <= 3

    def test_all_have_name(self, gen):
        cells = gen.generate_cells(count_limit=3)
        for cell in cells:
            assert 'name' in cell

    def test_all_have_derivation(self, gen):
        cells = gen.generate_cells(count_limit=2)
        for cell in cells:
            assert '_derivation' in cell

    def test_progress_callback(self, gen):
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        gen.generate_cells(count_limit=2, progress_callback=cb)
        assert len(calls) > 0
        assert calls[-1][0] == 100


class TestBiomaterials:
    def test_generates_biomaterials(self, gen):
        mats = gen.generate_biomaterials(count_limit=3)
        assert len(mats) > 0
        assert len(mats) <= 3

    def test_all_have_name(self, gen):
        mats = gen.generate_biomaterials(count_limit=3)
        for m in mats:
            assert 'name' in m

    def test_all_have_derivation(self, gen):
        mats = gen.generate_biomaterials(count_limit=2)
        for m in mats:
            assert '_derivation' in m


class TestCategoryInterface:
    def test_amino_acids_category(self, gen):
        items = gen.generate_category('amino_acids', count_limit=5)
        assert len(items) > 0

    def test_proteins_category(self, gen):
        items = gen.generate_category('proteins', count_limit=3)
        assert len(items) > 0

    def test_nucleic_acids_category(self, gen):
        items = gen.generate_category('nucleic_acids', count_limit=3)
        assert len(items) > 0

    def test_invalid_category_raises(self, gen):
        with pytest.raises(ValueError):
            gen.generate_category('invalid_category')

    def test_all_categories(self, gen):
        categories = ['amino_acids', 'proteins', 'nucleic_acids',
                      'cell_components', 'cells', 'biomaterials']
        for cat in categories:
            items = gen.generate_category(cat, count_limit=2)
            assert len(items) > 0, f"No items generated for {cat}"
