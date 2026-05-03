"""
Tests for the MoleculeGenerator.
Verifies that generated molecules have valid structure and properties.
"""

import pytest
from periodica.utils.molecule_generator import MoleculeGenerator


@pytest.fixture
def generator():
    return MoleculeGenerator()


class TestGenerateAll:
    def test_generates_molecules(self, generator):
        molecules = generator.generate_all(count_limit=20)
        assert len(molecules) > 0
        assert len(molecules) <= 20

    def test_no_duplicate_formulas(self, generator):
        molecules = generator.generate_all(count_limit=50)
        formulas = [m['Formula'] for m in molecules]
        assert len(formulas) == len(set(formulas))

    def test_all_have_required_fields(self, generator):
        molecules = generator.generate_all(count_limit=20)
        required = ['Name', 'Formula', 'MolecularMass_amu', 'BondType',
                     'Geometry', 'Category', 'Composition']
        for mol in molecules:
            for field in required:
                assert field in mol, f"Missing {field} in {mol.get('Name', '?')}"

    def test_all_have_derivation_metadata(self, generator):
        molecules = generator.generate_all(count_limit=10)
        for mol in molecules:
            assert '_derivation' in mol
            assert mol['_derivation']['source'] == 'auto_generated'

    def test_progress_callback(self, generator):
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        generator.generate_all(count_limit=10, progress_callback=cb)
        assert len(calls) > 0
        assert calls[-1][0] == 100


class TestHydrides:
    def test_water_generated(self, generator):
        mols = generator.generate_all(count_limit=100)
        names = [m['Name'] for m in mols]
        assert 'Water' in names

    def test_ammonia_generated(self, generator):
        mols = generator.generate_all(count_limit=100)
        names = [m['Name'] for m in mols]
        assert 'Ammonia' in names

    def test_methane_generated(self, generator):
        mols = generator.generate_all(count_limit=100)
        names = [m['Name'] for m in mols]
        assert 'Methane' in names

    def test_hydrogen_generated(self, generator):
        mols = generator.generate_all(count_limit=100)
        names = [m['Name'] for m in mols]
        assert 'Hydrogen' in names


class TestOxides:
    def test_oxygen_generated(self, generator):
        mols = generator.generate_all(count_limit=100)
        names = [m['Name'] for m in mols]
        assert 'Oxygen' in names

    def test_ozone_generated(self, generator):
        mols = generator.generate_all(count_limit=100)
        names = [m['Name'] for m in mols]
        assert 'Ozone' in names


class TestOrganics:
    def test_ethane_generated(self, generator):
        mols = generator.generate_all(count_limit=200)
        names = [m['Name'] for m in mols]
        assert 'Ethane' in names

    def test_ethanol_generated(self, generator):
        mols = generator.generate_all(count_limit=200, max_atoms=10)
        names = [m['Name'] for m in mols]
        assert 'Ethanol' in names


class TestMoleculeProperties:
    def test_water_mass(self, generator):
        mols = generator.generate_all(count_limit=100)
        water = next(m for m in mols if m['Name'] == 'Water')
        assert abs(water['MolecularMass_amu'] - 18.015) < 0.1

    def test_water_geometry(self, generator):
        mols = generator.generate_all(count_limit=100)
        water = next(m for m in mols if m['Name'] == 'Water')
        assert water['Geometry'] == 'Bent'

    def test_methane_mass(self, generator):
        mols = generator.generate_all(count_limit=100)
        methane = next(m for m in mols if m['Name'] == 'Methane')
        assert abs(methane['MolecularMass_amu'] - 16.043) < 0.1

    def test_positive_masses(self, generator):
        mols = generator.generate_all(count_limit=30)
        for mol in mols:
            assert mol['MolecularMass_amu'] > 0

    def test_valid_bond_types(self, generator):
        mols = generator.generate_all(count_limit=30)
        valid_types = {'Covalent', 'Polar Covalent', 'Ionic'}
        for mol in mols:
            assert mol['BondType'] in valid_types

    def test_nacl_ionic(self, generator):
        mols = generator.generate_all(count_limit=100)
        nacl = next((m for m in mols if m['Name'] == 'Sodium Chloride'), None)
        if nacl:
            assert nacl['BondType'] == 'Ionic'


class TestCountLimit:
    def test_respects_limit_small(self, generator):
        mols = generator.generate_all(count_limit=5)
        assert len(mols) <= 5

    def test_respects_limit_large(self, generator):
        mols = generator.generate_all(count_limit=200)
        assert len(mols) <= 200
