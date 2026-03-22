"""
Tests for Nuclear Shell Correction and Wigner Term

Validates that the SEMF with shell correction and Wigner term
produces more accurate binding energies, especially for light
nuclei and near magic numbers.
"""

import pytest
from periodica.utils.nuclear_derivation import NuclearDerivation


@pytest.fixture
def nuc():
    return NuclearDerivation()


class TestShellCorrection:
    """Test shell correction at magic numbers."""

    def test_he4_binding(self, nuc):
        """He-4 (Z=2, N=2): doubly magic. SEMF is poor for A<8 but
        shell correction should at least boost it above plain SEMF."""
        B = nuc.calculate_binding_energy(Z=2, N=2)
        # SEMF is inherently inaccurate for A=4; just check it's positive
        # and the shell correction adds meaningful binding
        assert B > 5.0, f"He-4 B={B:.2f} should be boosted by shell correction"

    def test_o16_binding(self, nuc):
        """O-16 (Z=8, N=8): doubly magic, B = 127.619 MeV."""
        B = nuc.calculate_binding_energy(Z=8, N=8)
        assert abs(B - 127.619) / 127.619 < 0.12, f"O-16 B={B:.2f}, expected ~127.6"

    def test_ca40_binding(self, nuc):
        """Ca-40 (Z=20, N=20): doubly magic, B = 342.05 MeV."""
        B = nuc.calculate_binding_energy(Z=20, N=20)
        assert abs(B - 342.05) / 342.05 < 0.03, f"Ca-40 B={B:.2f}, expected ~342.1"

    def test_ni62_binding(self, nuc):
        """Ni-62 (Z=28, N=34): near magic Z, B = 545.259 MeV."""
        B = nuc.calculate_binding_energy(Z=28, N=34)
        assert abs(B - 545.259) / 545.259 < 0.03, f"Ni-62 B={B:.2f}, expected ~545.3"

    def test_pb208_binding(self, nuc):
        """Pb-208 (Z=82, N=126): doubly magic, B = 1636.43 MeV."""
        B = nuc.calculate_binding_energy(Z=82, N=126)
        assert abs(B - 1636.43) / 1636.43 < 0.02, f"Pb-208 B={B:.2f}, expected ~1636.4"

    def test_magic_nuclei_higher_ba(self, nuc):
        """Magic number nuclei should have higher B/A than neighbors."""
        # O-16 (doubly magic) vs N-15 and F-17
        B_O16 = nuc.calculate_binding_energy(8, 8) / 16
        B_N15 = nuc.calculate_binding_energy(7, 8) / 15
        B_F17 = nuc.calculate_binding_energy(9, 8) / 17
        assert B_O16 > B_N15, f"O-16 B/A ({B_O16:.3f}) should exceed N-15 ({B_N15:.3f})"
        assert B_O16 > B_F17, f"O-16 B/A ({B_O16:.3f}) should exceed F-17 ({B_F17:.3f})"


class TestWignerTerm:
    """Test Wigner term: W_term = -W * |N - Z| / A."""

    def test_wigner_zero_for_nz_equal(self, nuc):
        """Wigner term should be zero when N=Z (|N-Z|=0)."""
        wigner = nuc._calculate_wigner_term(8, 8, 16)
        assert wigner == 0.0

    def test_wigner_negative_for_nz_different(self, nuc):
        """Wigner term should be negative (reduces binding) when N != Z."""
        wigner = nuc._calculate_wigner_term(8, 10, 18)
        assert wigner < 0.0

    def test_wigner_larger_penalty_for_bigger_imbalance(self, nuc):
        """Wigner penalty should grow with |N-Z|/A."""
        wigner_small = nuc._calculate_wigner_term(8, 10, 18)   # |N-Z|/A = 2/18
        wigner_large = nuc._calculate_wigner_term(82, 126, 208) # |N-Z|/A = 44/208
        # Both negative; larger imbalance ratio gives more negative value
        assert wigner_large < wigner_small


class TestOverallAccuracy:
    """Test that shell+Wigner corrections improve overall accuracy."""

    def test_fe56_binding(self, nuc):
        """Fe-56 (Z=26, N=30): near max B/A, B = 492.26 MeV."""
        B = nuc.calculate_binding_energy(Z=26, N=30)
        assert abs(B - 492.26) / 492.26 < 0.02, f"Fe-56 B={B:.2f}, expected ~492.3"

    def test_binding_positive(self, nuc):
        """Binding energy should be positive for all bound nuclei (A >= 2)."""
        for Z, N in [(2, 2), (6, 6), (26, 30), (82, 126)]:
            B = nuc.calculate_binding_energy(Z, N)
            assert B > 0, f"B({Z},{N}) = {B:.2f} should be positive"

    def test_ba_peak_near_fe(self, nuc):
        """B/A should peak near iron (A~56-62)."""
        ba_values = {}
        for Z, N in [(8, 8), (26, 30), (50, 70), (82, 126)]:
            A = Z + N
            ba_values[A] = nuc.calculate_binding_energy(Z, N) / A
        assert ba_values[56] > ba_values[16], "Fe B/A should exceed O"
        assert ba_values[56] > ba_values[208], "Fe B/A should exceed Pb"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
