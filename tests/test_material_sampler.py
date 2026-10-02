"""
Tests for Material Spatial Sampling System
==========================================

Tests the equation-of-state, temperature-dependence and optical-property
transforms. (The Fourier / wavelet / stochastic field evaluators and the
MaterialSampler that combined them were removed; spatial sampling is
`periodica.sample`, see test_alloy_microstructure.py.)
"""

import pytest
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBirchMurnaghanEOS:
    """Test Birch-Murnaghan equation of state."""

    def test_reference_conditions(self):
        """At reference pressure, density equals reference density."""
        from periodica.utils.transforms.thermo_pressure import (
            BirchMurnaghanEOS, BirchMurnaghanConfig
        )

        config = BirchMurnaghanConfig(
            K0_GPa=160.0,
            K0_prime=4.0,
            reference_density_kg_m3=7850.0
        )
        eos = BirchMurnaghanEOS(config)

        # At ~0 pressure, density should be reference
        density = eos.density_at_pressure(1e5)  # ~1 bar
        assert density == pytest.approx(7850.0, rel=0.01)

    def test_compression_at_high_pressure(self):
        """Density increases with pressure."""
        from periodica.utils.transforms.thermo_pressure import (
            BirchMurnaghanEOS, BirchMurnaghanConfig
        )

        config = BirchMurnaghanConfig(
            K0_GPa=160.0,
            K0_prime=4.0,
            reference_density_kg_m3=7850.0
        )
        eos = BirchMurnaghanEOS(config)

        # At 10 GPa, density should increase
        density_10gpa = eos.density_at_pressure(10e9)
        assert density_10gpa > 7850.0
        assert density_10gpa < 10000.0  # Sanity check

    def test_bulk_modulus_at_pressure(self):
        """Bulk modulus increases linearly with pressure (approximately)."""
        from periodica.utils.transforms.thermo_pressure import (
            BirchMurnaghanEOS, BirchMurnaghanConfig
        )

        config = BirchMurnaghanConfig(K0_GPa=160.0, K0_prime=4.5)
        eos = BirchMurnaghanEOS(config)

        K_0 = eos.bulk_modulus_at_pressure(0)
        K_1gpa = eos.bulk_modulus_at_pressure(1e9)

        assert K_0 == pytest.approx(160e9)
        # K(P) ~ K0 + K0' * P
        expected_K = 160e9 + 4.5 * 1e9
        assert K_1gpa == pytest.approx(expected_K, rel=0.01)


class TestTemperatureDependence:
    """Test temperature dependence models."""

    def test_constant_model(self):
        """Constant model returns reference value at all temperatures."""
        from periodica.utils.transforms.thermo_pressure import (
            TemperatureDependence, TemperatureDependenceConfig, TemperatureModel
        )

        config = TemperatureDependenceConfig(
            property_name='Test',
            model=TemperatureModel.CONSTANT,
            reference_value=100.0
        )
        dep = TemperatureDependence(config)

        assert dep.evaluate(293) == pytest.approx(100.0)
        assert dep.evaluate(500) == pytest.approx(100.0)
        assert dep.evaluate(1000) == pytest.approx(100.0)

    def test_linear_model(self):
        """Linear model applies temperature coefficient."""
        from periodica.utils.transforms.thermo_pressure import (
            TemperatureDependence, TemperatureDependenceConfig, TemperatureModel
        )

        config = TemperatureDependenceConfig(
            property_name='YoungsModulus_GPa',
            model=TemperatureModel.LINEAR,
            reference_T_K=293.15,
            reference_value=200.0,
            coefficients={'alpha': -0.0002}  # -0.02% per K
        )
        dep = TemperatureDependence(config)

        # At reference T
        v_ref = dep.evaluate(293.15)
        assert v_ref == pytest.approx(200.0)

        # At 100K higher
        v_393 = dep.evaluate(393.15)
        # Expected: 200 * (1 + (-0.0002) * 100) = 200 * 0.98 = 196
        assert v_393 == pytest.approx(196.0, rel=0.01)

    def test_polynomial_model(self):
        """Polynomial model evaluates correctly."""
        from periodica.utils.transforms.thermo_pressure import (
            TemperatureDependence, TemperatureDependenceConfig, TemperatureModel
        )

        config = TemperatureDependenceConfig(
            property_name='Test',
            model=TemperatureModel.POLYNOMIAL,
            reference_T_K=300,
            reference_value=100.0,
            coefficients={'a0': 100.0, 'a1': -0.1, 'a2': 0.0001}
        )
        dep = TemperatureDependence(config)

        # At T=400K, dT=100
        # P = 100 - 0.1*100 + 0.0001*10000 = 100 - 10 + 1 = 91
        v_400 = dep.evaluate(400)
        assert v_400 == pytest.approx(91.0)


class TestOpticalProperties:
    """Test optical property calculations."""

    def test_fresnel_reflectance_normal(self):
        """Normal incidence reflectance calculation."""
        from periodica.utils.transforms.optical_properties import (
            OpticalMaterial, SpectralProperty, OpticalModel
        )

        # Glass with n=1.5
        n_property = SpectralProperty(
            model=OpticalModel.CONSTANT,
            constant_value=1.5
        )
        glass = OpticalMaterial(
            name='Glass',
            refractive_index=n_property
        )

        # R = |n-1|^2 / |n+1|^2 = (0.5)^2 / (2.5)^2 = 0.04
        R = glass.reflectance_normal(550)
        assert R == pytest.approx(0.04, rel=0.01)

    def test_rgb_conversion(self):
        """RGB color conversion produces valid values."""
        from periodica.utils.transforms.optical_properties import (
            OpticalMaterial, SpectralProperty, OpticalModel
        )

        material = OpticalMaterial(name='Test')
        rgb = material.to_rgb()

        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)

    def test_blackbody_emission(self):
        """Blackbody emission color at high temperature."""
        from periodica.utils.transforms.optical_properties import OpticalMaterial

        material = OpticalMaterial(name='Test', emissivity=1.0)

        # At 1500K (orange-yellow hot metal)
        rgb = material.blackbody_emission_rgb(1500)
        assert len(rgb) == 3
        # Should be warm color (R > G > B)
        assert rgb[0] >= rgb[1] >= rgb[2]


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
