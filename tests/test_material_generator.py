"""
Tests for the MaterialGenerator.
Verifies material generation from alloy data.
"""

import pytest
from periodica.utils.alloy_generator import AlloyGenerator
from periodica.utils.material_generator import MaterialGenerator


@pytest.fixture
def alloy_gen():
    return AlloyGenerator()


@pytest.fixture
def mat_gen():
    return MaterialGenerator()


@pytest.fixture
def sample_alloy(alloy_gen):
    steels = alloy_gen.generate_steel_variants(count=1)
    return steels[0]


class TestGenerateFromAlloy:
    def test_generates_material(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert mat is not None
        assert 'Name' in mat

    def test_has_microstructure(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert 'Microstructure' in mat
        assert 'GrainStructure' in mat['Microstructure']
        assert 'Defects' in mat['Microstructure']
        assert 'Inclusions' in mat['Microstructure']

    def test_has_grain_structure_fields(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        gs = mat['Microstructure']['GrainStructure']
        required = ['AverageGrainSize_um', 'GrainSizeDistribution',
                     'ASTMGrainSizeNumber', 'VoronoiSeedDensity_per_mm2']
        for field in required:
            assert field in gs, f"Missing {field} in GrainStructure"

    def test_has_phase_composition(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert 'PhaseComposition' in mat
        assert 'Phases' in mat['PhaseComposition']
        assert len(mat['PhaseComposition']['Phases']) > 0

    def test_has_corrosion_data(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert 'CorrosionResistance' in mat
        assert 'PREN' in mat['CorrosionResistance']

    def test_has_derivation_metadata(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert '_derivation' in mat
        assert mat['_derivation']['source'] == 'auto_generated'
        assert 'alloy' in mat['_derivation']['derivation_chain']

    def test_preserves_physical_properties(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert 'PhysicalProperties' in mat
        assert mat['PhysicalProperties'] == sample_alloy['PhysicalProperties']

    def test_preserves_mechanical_properties(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert 'MechanicalProperties' in mat

    def test_source_alloy_tracked(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        assert 'SourceAlloy' in mat
        assert mat['SourceAlloy'] == sample_alloy['Name']


class TestProcessingVariants:
    def test_hot_rolled(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy, processing='Hot rolled')
        gs = mat['Microstructure']['GrainStructure']
        assert 20 <= gs['AverageGrainSize_um'] <= 80

    def test_cold_rolled(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy, processing='Cold rolled')
        gs = mat['Microstructure']['GrainStructure']
        assert gs['AverageGrainSize_um'] < 80
        assert gs['GrainAspectRatio'] > 1.0  # elongated grains

    def test_as_cast_coarser(self, mat_gen, sample_alloy):
        cast = mat_gen.generate_from_alloy(sample_alloy, processing='As-cast')
        rolled = mat_gen.generate_from_alloy(sample_alloy, processing='Hot rolled')
        cast_gs = cast['Microstructure']['GrainStructure']['AverageGrainSize_um']
        rolled_gs = rolled['Microstructure']['GrainStructure']['AverageGrainSize_um']
        assert cast_gs > rolled_gs

    def test_cold_rolled_has_texture(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy, processing='Cold rolled')
        orient = mat['CrystallographicOrientation']
        assert orient['PreferredOrientation'] is True
        assert orient['TextureType'] == 'Fiber'


class TestDefectLevels:
    def test_low_defects(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy, defect_level='low')
        defects = mat['Microstructure']['Defects']
        assert defects['DislocationDensity_per_m2'] <= 1e11

    def test_high_defects(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy, defect_level='high')
        defects = mat['Microstructure']['Defects']
        assert defects['DislocationDensity_per_m2'] >= 1e13

    def test_high_more_dislocations_than_low(self, mat_gen, sample_alloy):
        low = mat_gen.generate_from_alloy(sample_alloy, defect_level='low')
        high = mat_gen.generate_from_alloy(sample_alloy, defect_level='high')
        assert (high['Microstructure']['Defects']['DislocationDensity_per_m2'] >
                low['Microstructure']['Defects']['DislocationDensity_per_m2'])


class TestGenerateAll:
    def test_generates_from_alloy_list(self, mat_gen, alloy_gen):
        alloys = alloy_gen.generate_steel_variants(count=3)
        materials = mat_gen.generate_all(alloys, count_limit=10)
        assert len(materials) > 0
        assert len(materials) <= 10

    def test_progress_callback(self, mat_gen, alloy_gen):
        alloys = alloy_gen.generate_steel_variants(count=2)
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        mat_gen.generate_all(alloys, count_limit=5, progress_callback=cb)
        assert len(calls) > 0
        assert calls[-1][0] == 100

    def test_respects_count_limit(self, mat_gen, alloy_gen):
        alloys = alloy_gen.generate_all(count_limit=20)
        materials = mat_gen.generate_all(alloys, count_limit=5)
        assert len(materials) <= 5


class TestPhaseEstimation:
    def test_phases_have_required_fields(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        for phase in mat['PhaseComposition']['Phases']:
            assert 'Name' in phase
            assert 'Structure' in phase
            assert 'VolumePercent' in phase

    def test_phase_volumes_sum_100(self, mat_gen, sample_alloy):
        mat = mat_gen.generate_from_alloy(sample_alloy)
        total = sum(p['VolumePercent'] for p in mat['PhaseComposition']['Phases'])
        assert total == 100
