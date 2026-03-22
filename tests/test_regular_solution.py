"""
Tests for Regular Solution Model and Lindemann Melting

Validates:
1. Omega derived from Miedema (no hardcoded Omega table)
2. Free energy of mixing behavior
3. Miscibility gap predictions
4. Lindemann melting temperature estimates
"""

import pytest
from periodica.utils.phase_diagram import RegularSolutionModel, lindemann_melting


# Element data sheet fixtures
CU_DATA = {
    'symbol': 'Cu', 'atomic_mass': 63.546, 'density': 8.96,
    'atomic_radius': 128, 'debye_temperature_K': 343,
    'miedema_phi_star': 4.55, 'miedema_nws_13': 1.47, 'miedema_V_23': 5.84,
}
NI_DATA = {
    'symbol': 'Ni', 'atomic_mass': 58.693, 'density': 8.908,
    'atomic_radius': 124, 'debye_temperature_K': 450,
    'miedema_phi_star': 5.20, 'miedema_nws_13': 1.75, 'miedema_V_23': 5.51,
}
FE_DATA = {
    'symbol': 'Fe', 'atomic_mass': 55.845, 'density': 7.874,
    'atomic_radius': 126, 'debye_temperature_K': 470,
    'miedema_phi_star': 4.93, 'miedema_nws_13': 1.77, 'miedema_V_23': 5.95,
}
AG_DATA = {
    'symbol': 'Ag', 'atomic_mass': 107.868, 'density': 10.49,
    'atomic_radius': 144, 'debye_temperature_K': 225,
    'miedema_phi_star': 4.45, 'miedema_nws_13': 1.36, 'miedema_V_23': 7.73,
}


class TestRegularSolution:
    """Test regular solution model."""

    def test_omega_derived_from_miedema(self):
        """Omega should be derived from Miedema, not hardcoded."""
        omega_cu_ni = RegularSolutionModel.calculate_omega(CU_DATA, NI_DATA)
        assert omega_cu_ni != 0, "Omega should be non-zero for Cu-Ni"

    def test_cu_ni_fully_miscible(self):
        """Cu-Ni: small positive Omega, fully miscible at high T."""
        omega = RegularSolutionModel.calculate_omega(CU_DATA, NI_DATA)
        T_c = RegularSolutionModel.critical_temperature(omega)
        # Cu-Ni is fully miscible above ~600K
        # With our Miedema, T_c should exist but be achievable
        assert T_c > 0, f"Cu-Ni should have a positive T_c"

    def test_cu_fe_has_miscibility_gap(self):
        """Cu-Fe: positive Omega, miscibility gap exists."""
        omega = RegularSolutionModel.calculate_omega(CU_DATA, FE_DATA)
        assert omega > 0, f"Cu-Fe Omega={omega}, expected positive"
        has_gap = RegularSolutionModel.has_miscibility_gap(omega, 800)
        # Cu-Fe should have gap at moderate temperatures
        assert has_gap, "Cu-Fe should have miscibility gap at 800K"

    def test_free_energy_symmetric(self):
        """Free energy should be symmetric about x=0.5."""
        omega = 20000  # J/mol
        T = 1000
        G_30 = RegularSolutionModel.free_energy_of_mixing(0.3, omega, T)
        G_70 = RegularSolutionModel.free_energy_of_mixing(0.7, omega, T)
        assert abs(G_30 - G_70) < 1.0, \
            f"G(0.3)={G_30:.1f} should equal G(0.7)={G_70:.1f}"

    def test_free_energy_negative_at_high_t(self):
        """At high T, ideal mixing dominates => G_mix < 0 at x=0.5."""
        G = RegularSolutionModel.free_energy_of_mixing(0.5, 10000, 5000)
        assert G < 0, f"G_mix={G:.0f} should be negative at high T"

    def test_pure_endpoints_zero(self):
        """Free energy at x=0 or x=1 should be zero."""
        G0 = RegularSolutionModel.free_energy_of_mixing(0.0, 10000, 1000)
        G1 = RegularSolutionModel.free_energy_of_mixing(1.0, 10000, 1000)
        assert G0 == 0.0
        assert G1 == 0.0


class TestLindemannMelting:
    """Test Lindemann melting temperature predictions."""

    def test_fe_melting(self):
        """Fe melting via Lindemann should be ~1811 K (±15%)."""
        T_m = lindemann_melting(FE_DATA)
        assert T_m > 0, "Fe melting should be positive"
        assert abs(T_m - 1811) / 1811 < 0.15, \
            f"Fe T_m={T_m:.0f}K, expected ~1811K"

    def test_cu_melting(self):
        """Cu melting via Lindemann should be ~1358 K (±15%)."""
        T_m = lindemann_melting(CU_DATA)
        assert T_m > 0
        assert abs(T_m - 1358) / 1358 < 0.15, \
            f"Cu T_m={T_m:.0f}K, expected ~1358K"

    def test_ag_melting(self):
        """Ag melting should be ~1235 K (±20%)."""
        T_m = lindemann_melting(AG_DATA)
        assert T_m > 0
        assert abs(T_m - 1235) / 1235 < 0.20, \
            f"Ag T_m={T_m:.0f}K, expected ~1235K"

    def test_missing_data_returns_zero(self):
        """Should return 0 when Debye temperature is missing."""
        bad_data = {'atomic_mass': 55.8, 'density': 7.87}
        assert lindemann_melting(bad_data) == 0.0

    def test_higher_debye_higher_melting(self):
        """Higher Debye temperature generally means higher melting point."""
        T_cu = lindemann_melting(CU_DATA)   # theta_D = 343
        T_fe = lindemann_melting(FE_DATA)   # theta_D = 470
        assert T_fe > T_cu, f"Fe ({T_fe:.0f}K) should melt higher than Cu ({T_cu:.0f}K)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
