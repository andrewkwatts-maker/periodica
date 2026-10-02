"""
Unit tests for biological predictor modules.
Tests edge cases, validation, and scientific accuracy.
"""

import pytest
import math
from pathlib import Path


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
