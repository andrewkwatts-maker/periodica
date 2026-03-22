"""
Tests for Improved Protein Secondary Structure Prediction

Validates:
1. Polyalanine → helix (Ala has high helix propensity ~1.42)
2. Polyvaline → sheet (Val has high sheet propensity ~1.70)
3. Proline breaks helix
4. Glycine breaks sheet
5. Window size is 8 (expanded from 6)
6. Nucleation rules work correctly
7. Hydrophobicity profile calculation
"""

import pytest
from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor


@pytest.fixture
def predictor():
    return ProteinPredictor()


class TestSecondaryStructurePrediction:
    """Test improved Chou-Fasman secondary structure prediction."""

    def test_window_size_is_8(self, predictor):
        """Window size should be expanded to 8."""
        assert predictor.window_size == 8

    def test_polyalanine_helix(self, predictor):
        """Polyalanine should be predominantly helix (Ala P(α) ≈ 1.42)."""
        sequence = 'A' * 20
        results = predictor.predict_secondary_structure(sequence)
        summary = predictor.get_structure_summary(results)
        assert summary['helix_percent'] > 40, \
            f"Polyalanine helix={summary['helix_percent']}%, expected >40%"

    def test_polyvaline_sheet(self, predictor):
        """Polyvaline should be predominantly sheet (Val P(β) ≈ 1.70)."""
        sequence = 'V' * 20
        results = predictor.predict_secondary_structure(sequence)
        summary = predictor.get_structure_summary(results)
        assert summary['sheet_percent'] > 40, \
            f"Polyvaline sheet={summary['sheet_percent']}%, expected >40%"

    def test_proline_breaks_helix(self, predictor):
        """Proline inserted in helix-forming sequence should break the helix."""
        # Pure alanine helix
        helix_seq = 'A' * 20
        helix_results = predictor.predict_secondary_structure(helix_seq)

        # Insert proline in the middle
        broken_seq = 'A' * 9 + 'P' + 'A' * 10
        broken_results = predictor.predict_secondary_structure(broken_seq)

        # The proline residue itself should NOT be helix
        pro_residue = broken_results[9]
        assert pro_residue['structure'] != 'H', \
            f"Proline at position 10 should not be helix, got {pro_residue['structure']}"

    def test_glycine_breaks_sheet(self, predictor):
        """Glycine inserted in sheet-forming sequence should disrupt sheet."""
        # Pure valine sheet
        sheet_seq = 'V' * 20
        sheet_results = predictor.predict_secondary_structure(sheet_seq)

        # Insert glycine in the middle
        broken_seq = 'V' * 9 + 'G' + 'V' * 10
        broken_results = predictor.predict_secondary_structure(broken_seq)

        # The glycine residue itself should NOT be sheet
        gly_residue = broken_results[9]
        assert gly_residue['structure'] != 'E', \
            f"Glycine at position 10 should not be sheet, got {gly_residue['structure']}"

    def test_mixed_sequence_has_variety(self, predictor):
        """A mixed sequence should produce multiple structure types."""
        # Mix of helix-formers, sheet-formers, and turn-formers
        sequence = 'AAALAAAKA' + 'VVIVVIVV' + 'GPNG' + 'AALAAK'
        results = predictor.predict_secondary_structure(sequence)
        summary = predictor.get_structure_summary(results)

        structures_present = sum([
            summary['helix_percent'] > 0,
            summary['sheet_percent'] > 0,
            (summary['turn_percent'] + summary['coil_percent']) > 0,
        ])
        assert structures_present >= 2, \
            f"Mixed sequence should have >=2 structure types, got {summary}"

    def test_helix_nucleation(self, predictor):
        """Helix nucleation requires >=4 of 6 residues with P(α)>1.0."""
        nucleation = predictor._find_helix_nucleation('AAAAAA')
        # Alanine has high helix propensity, should nucleate
        assert any(nucleation), "Pure alanine should nucleate helix"

    def test_sheet_nucleation(self, predictor):
        """Sheet nucleation requires >=3 of 5 residues with P(β)>1.0."""
        nucleation = predictor._find_sheet_nucleation('VVVVV')
        # Valine has high sheet propensity, should nucleate
        assert any(nucleation), "Pure valine should nucleate sheet"

    def test_structure_summary_adds_to_100(self, predictor):
        """Structure percentages should sum to 100%."""
        sequence = 'ACDEFGHIKLMNPQRSTVWY'
        results = predictor.predict_secondary_structure(sequence)
        summary = predictor.get_structure_summary(results)
        total = (summary['helix_percent'] + summary['sheet_percent'] +
                 summary['turn_percent'] + summary['coil_percent'])
        assert abs(total - 100.0) < 0.5, f"Structure percentages sum to {total}, expected 100"

    def test_results_have_phi_psi(self, predictor):
        """Each residue result should include phi/psi angles."""
        results = predictor.predict_secondary_structure('ACDEF')
        for res in results:
            assert 'phi' in res and 'psi' in res
            assert -180 <= res['phi'] <= 180
            assert -180 <= res['psi'] <= 180


class TestHydrophobicBurial:
    """Test hydrophobicity profile calculations."""

    def test_hydropathy_profile_length(self, predictor):
        """Hydropathy profile should match sequence length."""
        sequence = 'ACDEFGHIK'
        profile = predictor.calculate_hydropathy_profile(sequence)
        assert len(profile) == len(sequence)

    def test_hydrophobic_residues_positive(self, predictor):
        """Hydrophobic residues (Ile, Val, Leu) should have positive hydropathy."""
        # Isoleucine has highest hydropathy (4.5 Kyte-Doolittle)
        hydropathy_I = predictor.get_hydropathy('I')
        assert hydropathy_I > 0, f"Ile hydropathy={hydropathy_I}, expected positive"

    def test_hydrophilic_residues_negative(self, predictor):
        """Hydrophilic residues (Arg, Lys, Asp) should have negative hydropathy."""
        hydropathy_R = predictor.get_hydropathy('R')
        assert hydropathy_R < 0, f"Arg hydropathy={hydropathy_R}, expected negative"

    def test_gravy_hydrophobic_sequence(self, predictor):
        """A hydrophobic sequence should have positive GRAVY."""
        gravy = predictor.calculate_gravy('IIILLVVV')
        assert gravy > 0, f"Hydrophobic GRAVY={gravy}, expected positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
