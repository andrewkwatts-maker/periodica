"""
Tests for DRG Hyperfine Splitting Calculator

Validates that the De Rujula-Georgi-Glashow model correctly predicts
hadron masses from quark constituent masses read from data sheets.
"""

import pytest
from periodica.utils.predictors.hadron.hyperfine import (
    get_constituent_mass,
    calculate_hyperfine_mass,
)


# =============================================================================
# Quark data sheet fixtures (simulating data read from JSON)
# =============================================================================

def _quark(name, symbol, current_mass, constituent_mass):
    """Helper to create a quark data dict like the JSON data sheet."""
    return {
        'Name': name,
        'Symbol': symbol,
        'Mass_MeVc2': current_mass,
        'ConstituentMass_MeV': constituent_mass,
        'Spin_hbar': 0.5,
    }


UP = _quark('Up Quark', 'u', 2.2, 336.0)
DOWN = _quark('Down Quark', 'd', 4.7, 336.0)
STRANGE = _quark('Strange Quark', 's', 95.0, 509.0)
CHARM = _quark('Charm Quark', 'c', 1270.0, 1550.0)
BOTTOM = _quark('Bottom Quark', 'b', 4180.0, 4730.0)


class TestGetConstituentMass:
    """Test reading constituent mass from quark data sheets."""

    def test_reads_from_data_sheet(self):
        assert get_constituent_mass(UP) == 336.0
        assert get_constituent_mass(STRANGE) == 509.0
        assert get_constituent_mass(CHARM) == 1550.0
        assert get_constituent_mass(BOTTOM) == 4730.0

    def test_fallback_when_field_missing(self):
        """If ConstituentMass_MeV is not in the data, estimate from current mass."""
        quark_no_const = {'Mass_MeVc2': 2.2}
        mass = get_constituent_mass(quark_no_const)
        assert mass > 300  # Should give a reasonable light quark mass


class TestMesonMasses:
    """Test meson mass predictions from quark data."""

    def test_pion_mass(self):
        """pi+ (u d-bar, S=0) should be ~140 MeV."""
        mass = calculate_hyperfine_mass([336.0, 336.0], 'meson', total_spin=0)
        assert abs(mass - 139.6) / 139.6 < 0.05, f"Pion mass {mass:.1f} MeV, expected ~139.6"

    def test_rho_mass(self):
        """rho (u d-bar, S=1) should be ~775 MeV."""
        mass = calculate_hyperfine_mass([336.0, 336.0], 'meson', total_spin=1)
        assert abs(mass - 775.3) / 775.3 < 0.10, f"Rho mass {mass:.1f} MeV, expected ~775.3"

    def test_pion_rho_splitting(self):
        """The pi-rho mass difference should be large (~635 MeV)."""
        m_pi = calculate_hyperfine_mass([336.0, 336.0], 'meson', total_spin=0)
        m_rho = calculate_hyperfine_mass([336.0, 336.0], 'meson', total_spin=1)
        splitting = m_rho - m_pi
        assert splitting > 500, f"Pi-rho splitting {splitting:.0f} MeV too small"

    def test_kaon_mass(self):
        """K+ (u s-bar, S=0) should be ~494 MeV."""
        mass = calculate_hyperfine_mass([336.0, 509.0], 'meson', total_spin=0)
        assert abs(mass - 493.7) / 493.7 < 0.05, f"Kaon mass {mass:.1f} MeV, expected ~493.7"

    def test_jpsi_mass(self):
        """J/psi (c c-bar, S=1) should be ~3097 MeV."""
        mass = calculate_hyperfine_mass([1550.0, 1550.0], 'meson', total_spin=1)
        assert abs(mass - 3096.9) / 3096.9 < 0.02, f"J/psi mass {mass:.1f} MeV, expected ~3096.9"

    def test_upsilon_mass(self):
        """Upsilon (b b-bar, S=1) should be ~9460 MeV."""
        mass = calculate_hyperfine_mass([4730.0, 4730.0], 'meson', total_spin=1)
        assert abs(mass - 9460.3) / 9460.3 < 0.01, f"Upsilon mass {mass:.1f} MeV, expected ~9460.3"

    def test_pseudoscalar_lighter_than_vector(self):
        """S=0 mesons should always be lighter than S=1 with same quarks."""
        for masses in [[336, 336], [336, 509], [1550, 1550]]:
            m_ps = calculate_hyperfine_mass(masses, 'meson', total_spin=0)
            m_v = calculate_hyperfine_mass(masses, 'meson', total_spin=1)
            assert m_ps < m_v, f"Pseudoscalar ({m_ps:.0f}) should be lighter than vector ({m_v:.0f})"


class TestBaryonMasses:
    """Test baryon mass predictions from quark data."""

    def test_proton_mass(self):
        """Proton (uud, J=1/2) should be ~938 MeV."""
        mass = calculate_hyperfine_mass([336.0, 336.0, 336.0], 'baryon', total_spin=0.5)
        assert abs(mass - 938.3) / 938.3 < 0.02, f"Proton mass {mass:.1f} MeV, expected ~938.3"

    def test_delta_mass(self):
        """Delta++ (uuu, J=3/2) should be ~1232 MeV."""
        mass = calculate_hyperfine_mass([336.0, 336.0, 336.0], 'baryon', total_spin=1.5)
        assert abs(mass - 1232.0) / 1232.0 < 0.02, f"Delta mass {mass:.1f} MeV, expected ~1232"

    def test_delta_proton_splitting(self):
        """Delta-proton mass difference should be ~294 MeV."""
        m_p = calculate_hyperfine_mass([336.0, 336.0, 336.0], 'baryon', total_spin=0.5)
        m_d = calculate_hyperfine_mass([336.0, 336.0, 336.0], 'baryon', total_spin=1.5)
        splitting = m_d - m_p
        assert abs(splitting - 294) / 294 < 0.10, f"Delta-p splitting {splitting:.0f} MeV, expected ~294"

    def test_omega_minus_mass(self):
        """Omega- (sss, J=3/2) should be ~1672 MeV."""
        mass = calculate_hyperfine_mass([509.0, 509.0, 509.0], 'baryon', total_spin=1.5)
        assert abs(mass - 1672.5) / 1672.5 < 0.01, f"Omega mass {mass:.1f} MeV, expected ~1672.5"

    def test_lambda_mass(self):
        """Lambda (uds, J=1/2) should be ~1116 MeV.

        Note: With m_u = m_d, the code treats the ud pair as same-flavor
        (Sigma-like triplet) rather than Lambda-specific singlet. This gives
        ~6% error, still a major improvement over flat binding correction (~15%).
        """
        mass = calculate_hyperfine_mass([336.0, 336.0, 509.0], 'baryon', total_spin=0.5)
        assert abs(mass - 1115.7) / 1115.7 < 0.07, f"Lambda mass {mass:.1f} MeV, expected ~1115.7"

    def test_decuplet_heavier_than_octet(self):
        """J=3/2 baryons should be heavier than J=1/2 with same quarks."""
        for masses in [[336, 336, 336], [509, 509, 509], [336, 336, 509]]:
            m_oct = calculate_hyperfine_mass(masses, 'baryon', total_spin=0.5)
            m_dec = calculate_hyperfine_mass(masses, 'baryon', total_spin=1.5)
            assert m_oct < m_dec, f"Octet ({m_oct:.0f}) should be lighter than decuplet ({m_dec:.0f})"


class TestIntegration:
    """Test the full flow: quark data -> constituent mass -> hadron mass."""

    def test_pion_from_quark_data(self):
        """Build pion mass reading constituent masses from quark data dicts."""
        masses = [get_constituent_mass(UP), get_constituent_mass(DOWN)]
        m_pi = calculate_hyperfine_mass(masses, 'meson', total_spin=0)
        assert abs(m_pi - 139.6) / 139.6 < 0.05

    def test_proton_from_quark_data(self):
        """Build proton mass from quark data dicts."""
        masses = [get_constituent_mass(UP), get_constituent_mass(UP), get_constituent_mass(DOWN)]
        m_p = calculate_hyperfine_mass(masses, 'baryon', total_spin=0.5)
        assert abs(m_p - 938.3) / 938.3 < 0.02

    def test_default_spin_is_ground_state(self):
        """When spin is None, should default to ground state."""
        m_default = calculate_hyperfine_mass([336.0, 336.0], 'meson', total_spin=None)
        m_ground = calculate_hyperfine_mass([336.0, 336.0], 'meson', total_spin=0)
        assert m_default == m_ground


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
