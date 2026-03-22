"""
Unit tests for biological derivation chain and component factory.
Tests the complete hierarchy from atoms to complex biomaterials.
"""

import pytest
import json
from pathlib import Path


class TestAtomicComposition:
    """Tests for AtomicComposition class."""

    @pytest.fixture
    def composition_module(self):
        from periodica.utils.biological_derivation_chain import AtomicComposition
        return AtomicComposition

    def test_empty_composition(self, composition_module):
        """Empty composition should have empty formula."""
        comp = composition_module({})
        assert comp.get_molecular_formula() == ""

    def test_single_element(self, composition_module):
        """Single element should format correctly."""
        comp = composition_module({'C': 1})
        assert comp.get_molecular_formula() == "C"

    def test_multiple_elements_standard_order(self, composition_module):
        """Elements should follow CHNOPS order."""
        comp = composition_module({'O': 2, 'H': 4, 'C': 2})
        formula = comp.get_molecular_formula()
        assert formula.startswith('C')
        assert 'H' in formula
        assert 'O' in formula

    def test_molecular_mass_calculation(self, composition_module):
        """Molecular mass should be calculated correctly."""
        # Water: H2O = 18.015 Da
        water = composition_module({'H': 2, 'O': 1})
        masses = {'H': 1.008, 'O': 15.999}
        mass = water.get_molecular_mass(masses)
        assert abs(mass - 18.015) < 0.01

    def test_composition_addition(self, composition_module):
        """Compositions should add correctly."""
        comp1 = composition_module({'C': 2, 'H': 4})
        comp2 = composition_module({'C': 1, 'O': 1})
        result = comp1 + comp2
        assert result.elements['C'] == 3
        assert result.elements['H'] == 4
        assert result.elements['O'] == 1

    def test_composition_subtraction(self, composition_module):
        """Compositions should subtract correctly."""
        comp1 = composition_module({'C': 3, 'H': 6, 'O': 2})
        water = composition_module({'H': 2, 'O': 1})
        result = comp1 - water
        assert result.elements['C'] == 3
        assert result.elements['H'] == 4
        assert result.elements['O'] == 1


class TestBiologicalDerivationChain:
    """Tests for BiologicalDerivationChain class."""

    @pytest.fixture
    def chain(self):
        from periodica.utils.biological_derivation_chain import BiologicalDerivationChain
        return BiologicalDerivationChain()

    def test_amino_acid_composition_alanine(self, chain):
        """Alanine composition should be C3H7NO2."""
        comp = chain._get_standard_aa_composition('A')
        assert comp.elements.get('C') == 3
        assert comp.elements.get('H') == 7
        assert comp.elements.get('N') == 1
        assert comp.elements.get('O') == 2

    def test_amino_acid_composition_cysteine(self, chain):
        """Cysteine should contain sulfur."""
        comp = chain._get_standard_aa_composition('C')
        assert comp.elements.get('S') == 1

    def test_amino_acid_composition_methionine(self, chain):
        """Methionine should contain sulfur."""
        comp = chain._get_standard_aa_composition('M')
        assert comp.elements.get('S') == 1

    def test_build_protein_single_residue(self, chain):
        """Single residue protein should have correct mass."""
        protein = chain.build_protein_from_sequence('A', 'Single Ala')
        assert protein.name == 'Single Ala'
        assert protein.properties['length'] == 1
        assert protein.properties['molecular_mass'] > 0

    def test_build_protein_dipeptide(self, chain):
        """Dipeptide should lose one water molecule."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        protein = chain.build_protein_from_sequence('AA', 'Ala-Ala')
        assert protein.level == BiologicalLevel.PROTEIN
        assert protein.properties['length'] == 2

        # Mass should be less than 2x single residue due to water loss
        single_mass = chain.build_protein_from_sequence('A', 'Ala').properties['molecular_mass']
        dipeptide_mass = protein.properties['molecular_mass']
        assert dipeptide_mass < 2 * single_mass

    def test_build_protein_properties(self, chain):
        """Protein should have all expected properties."""
        protein = chain.build_protein_from_sequence('MVLSPADKTNVK', 'Test Protein')
        props = protein.properties

        assert 'sequence' in props
        assert 'length' in props
        assert 'molecular_mass' in props
        assert 'molecular_formula' in props

    def test_build_nucleic_acid_dna(self, chain):
        """DNA should be built correctly."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        dna = chain.build_nucleic_acid_from_sequence('ATGC', 'Test DNA', is_rna=False)
        assert dna.level == BiologicalLevel.NUCLEIC_ACID
        assert dna.properties['type'] == 'DNA'
        assert dna.properties['length'] == 4
        assert dna.properties['gc_content'] == 50.0

    def test_build_nucleic_acid_rna(self, chain):
        """RNA should use U instead of T."""
        rna = chain.build_nucleic_acid_from_sequence('AUGC', 'Test RNA', is_rna=True)
        assert rna.properties['type'] == 'RNA'
        assert 'U' in rna.properties['sequence']
        assert 'T' not in rna.properties['sequence']

    def test_build_cell_component(self, chain):
        """Cell component should be built from proteins."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        protein1 = chain.build_protein_from_sequence('MVLSPADKTNVK', 'Protein 1')
        protein2 = chain.build_protein_from_sequence('ACDEFGHIKLMN', 'Protein 2')

        component = chain.build_cell_component(
            'Test Complex',
            [protein1, protein2],
            copy_number=100,
            component_type='ribosome'
        )

        assert component.level == BiologicalLevel.CELL_COMPONENT
        assert component.properties['protein_count'] == 2
        assert component.properties['copy_number'] == 100
        assert component.properties['component_type'] == 'ribosome'
        assert len(component.source_components) == 2

    def test_build_cell(self, chain):
        """Cell should be built from components."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        protein = chain.build_protein_from_sequence('MVLSPADKTNVK', 'Protein')
        component = chain.build_cell_component('Component', [protein])

        cell = chain.build_cell(
            'Test Cell',
            {component: 1000},
            diameter_um=10.0,
            cell_type='epithelial'
        )

        assert cell.level == BiologicalLevel.CELL
        assert cell.properties['diameter_um'] == 10.0
        assert cell.properties['cell_type'] == 'epithelial'
        assert cell.properties['component_count'] == 1000

    def test_build_biomaterial(self, chain):
        """Biomaterial should be built from ECM composition."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        material = chain.build_biomaterial(
            'Test Material',
            ecm_composition={'collagen_i': 0.3, 'water': 0.7},
            porosity=0.1
        )

        assert material.level == BiologicalLevel.BIOMATERIAL
        assert material.properties['porosity'] == 10.0
        assert 'youngs_modulus_MPa' in material.properties
        assert 'density_g_cm3' in material.properties


class TestBiologicalMixingSystem:
    """Tests for BiologicalMixingSystem class."""

    @pytest.fixture
    def mixer(self):
        from periodica.utils.biological_derivation_chain import (
            BiologicalDerivationChain, BiologicalMixingSystem
        )
        chain = BiologicalDerivationChain()
        return BiologicalMixingSystem(chain)

    @pytest.fixture
    def sample_cell(self, mixer):
        """Create a sample cell for testing."""
        protein = mixer.chain.build_protein_from_sequence('MVLSPADKTNVK', 'Protein')
        component = mixer.chain.build_cell_component('Component', [protein])
        return mixer.chain.build_cell('Sample Cell', {component: 100}, diameter_um=10.0)

    def test_create_tissue(self, mixer, sample_cell):
        """Tissue should be created from cells and ECM."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        tissue = mixer.create_tissue(
            'Test Tissue',
            cells={sample_cell: 0.1},
            ecm={'collagen_i': 0.6, 'water': 0.3},
            vascularization=0.05
        )

        assert tissue.level == BiologicalLevel.TISSUE
        assert tissue.properties['total_cell_fraction'] == 0.1
        assert tissue.properties['vascularization'] == 0.05
        assert 'estimated_modulus_MPa' in tissue.properties

    def test_create_organ_model(self, mixer, sample_cell):
        """Organ model should be created from tissues."""
        tissue1 = mixer.create_tissue('Tissue 1', {sample_cell: 0.1}, {'collagen_i': 0.9})
        tissue2 = mixer.create_tissue('Tissue 2', {sample_cell: 0.2}, {'elastin': 0.8, 'water': 0.2})

        organ = mixer.create_organ_model(
            'Test Organ',
            tissues={tissue1: 0.6, tissue2: 0.4},
            organ_type='test'
        )

        assert 'tissue_composition' in organ.properties
        assert organ.properties['organ_type'] == 'test'


class TestBiologicalComponentFactory:
    """Tests for BiologicalComponentFactory class."""

    @pytest.fixture
    def factory(self):
        from periodica.utils.biological_component_factory import BiologicalComponentFactory
        return BiologicalComponentFactory()

    def test_create_protein_component(self, factory):
        """Factory should create protein with calculated properties."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        protein = factory.create_component(
            BiologicalLevel.PROTEIN,
            {'name': 'Test Protein', 'sequence': 'MVLSPADKTNVK'}
        )

        assert protein.name == 'Test Protein'
        assert 'molecular_mass' in protein.properties
        assert 'isoelectric_point' in protein.properties
        assert 'secondary_structure' in protein.properties

    def test_create_nucleic_acid_component(self, factory):
        """Factory should create nucleic acid with calculated properties."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        na = factory.create_component(
            BiologicalLevel.NUCLEIC_ACID,
            {'name': 'Test DNA', 'sequence': 'ATGCGATCGA', 'type': 'DNA'}
        )

        assert na.name == 'Test DNA'
        assert 'gc_content' in na.properties
        assert 'melting_temperature' in na.properties

    def test_create_cell_component(self, factory):
        """Factory should create cell with calculated properties."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        cell = factory.create_component(
            BiologicalLevel.CELL,
            {'name': 'Test Cell', 'type': 'epithelial', 'diameter_um': 15.0}
        )

        assert cell.name == 'Test Cell'
        assert 'volume_fl' in cell.properties
        assert 'metabolic_rate_fW' in cell.properties

    def test_create_biomaterial_component(self, factory):
        """Factory should create biomaterial with calculated properties."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        material = factory.create_component(
            BiologicalLevel.BIOMATERIAL,
            {
                'name': 'Test Material',
                'ecm_composition': {'collagen_i': 0.3, 'elastin': 0.1, 'water': 0.6}
            }
        )

        assert material.name == 'Test Material'
        assert 'youngs_modulus_MPa' in material.properties
        assert 'density_g_cm3' in material.properties

    def test_create_from_lower_level_protein(self, factory):
        """Should create protein from sequence."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        protein = factory.create_from_lower_level(
            BiologicalLevel.PROTEIN,
            [],
            'Built Protein',
            sequence='ACDEFGHIKLMNPQRSTVWY'
        )

        assert protein.properties['length'] == 20

    def test_create_from_lower_level_nucleic_acid(self, factory):
        """Should create nucleic acid from sequence."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        dna = factory.create_from_lower_level(
            BiologicalLevel.NUCLEIC_ACID,
            [],
            'Built DNA',
            sequence='ATGCGATCGA',
            is_rna=False
        )

        assert dna.properties['type'] == 'DNA'
        assert dna.properties['length'] == 10

    def test_create_complex_material(self, factory):
        """Should create complex material from mixed components."""
        material = factory.create_complex_material(
            'Complex Material',
            ecm_composition={'collagen_i': 0.4, 'elastin': 0.1, 'water': 0.5},
            porosity=0.15
        )

        assert 'youngs_modulus_MPa' in material.properties
        assert material.properties['porosity'] == 15.0

    def test_missing_required_fields_raises(self, factory):
        """Missing required fields should raise ValueError."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        with pytest.raises(ValueError, match="Missing required fields"):
            factory.create_component(
                BiologicalLevel.PROTEIN,
                {'name': 'Incomplete'}  # Missing 'sequence'
            )

    def test_list_available_components(self, factory):
        """Should list available components from data directory."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        # This depends on actual data files existing
        proteins = factory.list_available_components(BiologicalLevel.PROTEIN)
        assert isinstance(proteins, list)


class TestComponentSerialization:
    """Tests for component JSON serialization."""

    @pytest.fixture
    def chain(self):
        from periodica.utils.biological_derivation_chain import BiologicalDerivationChain
        return BiologicalDerivationChain()

    def test_component_to_dict(self, chain):
        """Component should serialize to dict."""
        protein = chain.build_protein_from_sequence('MVLSPADKTNVK', 'Test Protein')
        data = protein.to_dict()

        assert 'name' in data
        assert 'level' in data
        assert 'composition' in data
        assert 'properties' in data
        assert 'molecular_formula' in data

    def test_component_roundtrip(self, chain):
        """Component should survive JSON roundtrip."""
        from periodica.utils.biological_derivation_chain import BiologicalComponent, BiologicalLevel

        protein = chain.build_protein_from_sequence('MVLSPADKTNVK', 'Test Protein')
        data = protein.to_dict()

        # Serialize and deserialize
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)

        # Reconstruct
        loaded = BiologicalComponent.from_dict(loaded_data, BiologicalLevel.PROTEIN)
        assert loaded.name == protein.name


class TestDerivationChainIntegration:
    """Integration tests for the complete derivation chain."""

    @pytest.fixture
    def factory(self):
        from periodica.utils.biological_component_factory import BiologicalComponentFactory
        return BiologicalComponentFactory()

    def test_full_chain_atoms_to_biomaterial(self, factory):
        """Test complete chain from atoms to biomaterial."""
        from periodica.utils.biological_derivation_chain import BiologicalLevel

        # Build a protein
        protein = factory.create_from_lower_level(
            BiologicalLevel.PROTEIN,
            [],
            'Chain Protein',
            sequence='MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH'
        )

        # Build a cell component from the protein
        component = factory.derivation_chain.build_cell_component(
            'Chain Component',
            [protein],
            copy_number=10000
        )

        # Build a cell from the component
        cell = factory.derivation_chain.build_cell(
            'Chain Cell',
            {component: 1},
            diameter_um=10.0
        )

        # Create a tissue from the cell
        tissue = factory.mixing_system.create_tissue(
            'Chain Tissue',
            cells={cell: 0.1},
            ecm={'collagen_i': 0.5, 'water': 0.4}
        )

        # Verify the chain
        assert protein.level == BiologicalLevel.PROTEIN
        assert component.level == BiologicalLevel.CELL_COMPONENT
        assert cell.level == BiologicalLevel.CELL
        assert tissue.level == BiologicalLevel.TISSUE

        # Verify properties were calculated at each level
        assert protein.properties.get('molecular_mass', 0) > 0
        assert component.properties.get('protein_count') == 1
        assert cell.properties.get('volume_fl', 0) > 0
        assert tissue.properties.get('estimated_modulus_MPa', 0) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
