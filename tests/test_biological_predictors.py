"""
Unit tests for biological predictor modules.
Tests edge cases, validation, and scientific accuracy.
"""

import pytest
import math
from pathlib import Path


class TestBiomaterialPredictor:
    """Tests for BiomaterialPredictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor
        return BiomaterialPredictor()

    def test_empty_composition_raises_error(self, predictor):
        """Empty composition should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            predictor.calculate_voigt_modulus({})

    def test_negative_fraction_raises_error(self, predictor):
        """Negative volume fractions should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            predictor.calculate_voigt_modulus({'collagen_i': -0.5})

    def test_voigt_single_component(self, predictor):
        """Single component Voigt modulus equals component modulus."""
        result = predictor.calculate_voigt_modulus({'collagen_i': 1.0})
        assert result == predictor.ECM_MODULI['collagen_i']

    def test_reuss_handles_zero_modulus(self, predictor):
        """Reuss calculation should handle zero-modulus components (water)."""
        result = predictor.calculate_reuss_modulus({
            'collagen_i': 0.5,
            'water': 0.5
        })
        # Water is skipped, so result should be based on collagen only
        assert result > 0

    def test_porosity_out_of_range(self, predictor):
        """Porosity outside 0-1 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            predictor.calculate_porosity_effect(100, 1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            predictor.calculate_porosity_effect(100, -0.1)

    def test_porosity_full_reduces_to_zero(self, predictor):
        """100% porosity should give zero modulus."""
        result = predictor.calculate_porosity_effect(100, 1.0)
        assert result == 0.0

    def test_different_porosity_models(self, predictor):
        """Different porosity models should give different results."""
        E_dense = 100
        porosity = 0.3

        results = {}
        for model in ['gibson_ashby', 'exponential', 'linear']:
            predictor.set_porosity_model(model)
            results[model] = predictor.calculate_porosity_effect(E_dense, porosity)

        # Gibson-Ashby (n=2) gives lowest value
        # Linear (n=1) gives highest value
        assert results['gibson_ashby'] < results['exponential'] < results['linear']

    def test_voigt_greater_than_reuss(self, predictor):
        """Voigt bound should always be >= Reuss bound."""
        composition = {
            'collagen_i': 0.4,
            'elastin': 0.3,
            'proteoglycans': 0.3
        }
        voigt = predictor.calculate_voigt_modulus(composition)
        reuss = predictor.calculate_reuss_modulus(composition)
        assert voigt >= reuss


class TestCellPredictor:
    """Tests for CellPredictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.cell_predictor import CellPredictor
        return CellPredictor()

    def test_negative_mass_raises_error(self, predictor):
        """Negative cell mass should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            predictor.calculate_metabolic_rate(-100)

    def test_zero_mass_raises_error(self, predictor):
        """Zero cell mass should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            predictor.calculate_metabolic_rate(0)

    def test_metabolic_rate_scaling(self, predictor):
        """Metabolic rate should scale with mass^0.75."""
        rate1 = predictor.calculate_metabolic_rate(100)   # 100 pg
        rate2 = predictor.calculate_metabolic_rate(1000)  # 1000 pg

        # 10x mass should give ~5.6x metabolic rate (10^0.75)
        ratio = rate2 / rate1
        expected_ratio = 10 ** predictor.ALPHA
        assert abs(ratio - expected_ratio) < 0.01

    def test_temperature_correction(self, predictor):
        """Higher temperature should increase metabolic rate."""
        rate_37 = predictor.calculate_metabolic_rate(100, temperature_C=37)
        rate_20 = predictor.calculate_metabolic_rate(100, temperature_C=20)

        assert rate_37 > rate_20

    def test_invalid_alpha_raises_error(self, predictor):
        """Invalid scaling exponent should raise ValueError."""
        with pytest.raises(ValueError, match="ALPHA must be between"):
            predictor.set_alpha(1.5)  # Too high

        with pytest.raises(ValueError, match="ALPHA must be between"):
            predictor.set_alpha(0.2)  # Too low

    def test_cell_volume_calculation(self, predictor):
        """Cell volume should match spherical formula."""
        diameter = 10  # μm
        volume = predictor.calculate_cell_volume(diameter)

        expected = (4/3) * math.pi * (diameter/2) ** 3
        assert abs(volume - expected) < 0.01

    def test_doubling_time_positive(self, predictor):
        """Doubling time should be positive for valid inputs."""
        metabolic_rate = predictor.calculate_metabolic_rate(100)
        doubling = predictor.calculate_doubling_time(metabolic_rate, 100)
        assert doubling > 0


class TestNucleicAcidPredictor:
    """Tests for NucleicAcidPredictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.nucleic_acid_predictor import NucleicAcidPredictor
        return NucleicAcidPredictor()

    def test_empty_sequence_raises_error(self, predictor):
        """Empty sequence should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            predictor.calculate_gc_content("")

    def test_invalid_nucleotide_raises_error(self, predictor):
        """Invalid nucleotides should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid nucleotide"):
            predictor._validate_sequence("ATGXYZ")

    def test_gc_content_all_gc(self, predictor):
        """Sequence of all G/C should have 100% GC content."""
        result = predictor.calculate_gc_content("GCGCGC")
        assert result == 100.0

    def test_gc_content_all_at(self, predictor):
        """Sequence of all A/T should have 0% GC content."""
        result = predictor.calculate_gc_content("ATATAT")
        assert result == 0.0

    def test_gc_content_mixed(self, predictor):
        """Mixed sequence should have correct GC content."""
        result = predictor.calculate_gc_content("ATGC")  # 50% GC
        assert result == 50.0

    def test_complement_dna(self, predictor):
        """DNA complement should follow Watson-Crick rules."""
        complement = predictor.get_complement("ATGC", is_rna=False)
        assert complement == "TACG"

    def test_complement_rna(self, predictor):
        """RNA complement should use U instead of T."""
        complement = predictor.get_complement("AUGC", is_rna=True)
        assert complement == "UACG"

    def test_reverse_complement(self, predictor):
        """Reverse complement should be reversed."""
        rev_comp = predictor.get_reverse_complement("ATGC", is_rna=False)
        assert rev_comp == "GCAT"

    def test_transcription(self, predictor):
        """Transcription should convert T to U."""
        rna = predictor.transcribe("ATGC")
        assert rna == "AUGC"
        assert 'T' not in rna

    def test_reverse_transcription(self, predictor):
        """Reverse transcription should convert U to T."""
        dna = predictor.reverse_transcribe("AUGC")
        assert dna == "ATGC"
        assert 'U' not in dna

    def test_tm_increases_with_gc(self, predictor):
        """Higher GC content should increase Tm."""
        tm_low_gc = predictor.calculate_tm_nearest_neighbor("AAAAAA")
        tm_high_gc = predictor.calculate_tm_nearest_neighbor("GGGGGG")

        assert tm_high_gc > tm_low_gc

    def test_short_sequence_tm(self, predictor):
        """Very short sequences should return reasonable Tm."""
        tm = predictor.calculate_tm_nearest_neighbor("A")
        assert tm == 0.0  # Too short


class TestProteinPredictor:
    """Tests for ProteinPredictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor
        return ProteinPredictor()

    def test_molecular_mass_single_residue(self, predictor):
        """Single residue mass should match free amino acid mass."""
        # For a single amino acid, no peptide bonds form, so mass equals free AA mass
        mass = predictor.calculate_molecular_mass("A")
        expected = predictor.get_residue_mass('A')  # Free AA mass (89.094 for Alanine)
        assert abs(mass - expected) < 0.1

    def test_molecular_mass_increases_with_length(self, predictor):
        """Longer sequences should have higher mass."""
        mass_short = predictor.calculate_molecular_mass("AAA")
        mass_long = predictor.calculate_molecular_mass("AAAAAAAAA")

        assert mass_long > mass_short

    def test_gravy_hydrophobic_positive(self, predictor):
        """Hydrophobic sequences should have positive GRAVY."""
        # Isoleucine has highest hydropathy (4.5)
        gravy = predictor.calculate_gravy("IIIIII")
        assert gravy > 0

    def test_gravy_hydrophilic_negative(self, predictor):
        """Hydrophilic sequences should have negative GRAVY."""
        # Arginine has lowest hydropathy (-4.5)
        gravy = predictor.calculate_gravy("RRRRRR")
        assert gravy < 0

    def test_isoelectric_point_acidic(self, predictor):
        """Acidic sequences should have low pI."""
        # Aspartic acid rich sequence
        pI = predictor.calculate_isoelectric_point("DDDDD")
        assert pI < 5.0

    def test_isoelectric_point_basic(self, predictor):
        """Basic sequences should have high pI."""
        # Lysine rich sequence
        pI = predictor.calculate_isoelectric_point("KKKKK")
        assert pI > 9.0

    def test_charge_at_pH_acidic(self, predictor):
        """At low pH, proteins should be positively charged."""
        charge = predictor.calculate_charge_at_pH("AEDK", 2.0)
        assert charge > 0

    def test_charge_at_pH_basic(self, predictor):
        """At high pH, proteins should be negatively charged."""
        charge = predictor.calculate_charge_at_pH("AEDK", 12.0)
        assert charge < 0

    def test_instability_index(self, predictor):
        """Instability index should be calculated."""
        ii = predictor.calculate_instability_index("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
        assert isinstance(ii, float)

    def test_aliphatic_index(self, predictor):
        """Aliphatic index should increase with aliphatic residues."""
        ai_low = predictor.calculate_aliphatic_index("RRRRRR")  # No aliphatic
        ai_high = predictor.calculate_aliphatic_index("IIILLL")  # All aliphatic

        assert ai_high > ai_low

    def test_disulfide_bond_prediction(self, predictor):
        """Cysteines far apart should form disulfide bonds."""
        # C at positions 1 and 20 (19 residues apart)
        bonds = predictor.predict_disulfide_bonds("C" + "A" * 18 + "C")
        assert len(bonds) == 1
        assert bonds[0] == (1, 20)

    def test_disulfide_no_close_cysteines(self, predictor):
        """Close cysteines should not form bonds."""
        # Cysteines too close together
        bonds = predictor.predict_disulfide_bonds("CCAAA")
        assert len(bonds) == 0


class TestAminoAcidPredictor:
    """Tests for AminoAcidPredictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.amino_acid_predictor import (
            AminoAcidPredictor, AminoAcidInput
        )
        return AminoAcidPredictor()

    def test_invalid_symbol_raises_error(self, predictor):
        """Invalid amino acid symbol should fail validation."""
        from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidInput

        input_data = AminoAcidInput(symbol="X")
        valid, msg = predictor.validate(input_data)
        assert not valid

    def test_invalid_ph_raises_error(self, predictor):
        """pH outside 0-14 should fail validation."""
        from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidInput

        input_data = AminoAcidInput(symbol="A", pH=15.0)
        valid, msg = predictor.validate(input_data)
        assert not valid

    def test_neutral_residue_charge_at_neutral_ph(self, predictor):
        """Neutral residue at pH 7 should have ~0 net charge."""
        from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidInput

        result = predictor.predict_from_symbol("A", pH=7.0)
        # Should have small net charge (from termini)
        assert abs(result.charge) < 0.5

    def test_acidic_residue_charge(self, predictor):
        """Acidic residue should be negative at pH 7."""
        result = predictor.predict_from_symbol("D", pH=7.0)
        assert result.charge < 0

    def test_basic_residue_charge(self, predictor):
        """Basic residue should be positive at pH 7."""
        result = predictor.predict_from_symbol("K", pH=7.0)
        assert result.charge > 0

    def test_sequence_charge_calculation(self, predictor):
        """Sequence charge should sum individual contributions."""
        charge = predictor.calculate_sequence_charge("AAAA", pH=7.0)
        assert isinstance(charge, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
