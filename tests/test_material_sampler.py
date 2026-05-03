"""
Tests for Material Spatial Sampling System
==========================================

Tests the Fourier, wavelet, stochastic field evaluators and
the main MaterialSampler class.
"""

import pytest
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFourierFieldEvaluator:
    """Test Fourier field evaluation."""

    def test_constant_field(self):
        """Constant field (no Fourier terms) returns base value."""
        from periodica.utils.transforms.fourier_field import (
            FourierFieldEvaluator, FourierFieldConfig
        )

        config = FourierFieldConfig(
            property_name='Test',
            base_value=100.0,
            domain_size_m=(0.1, 0.1, 0.1),
            coefficients=[]
        )
        evaluator = FourierFieldEvaluator(config)

        value = evaluator.evaluate(0.05, 0.05, 0.05)
        assert value == pytest.approx(100.0)

    def test_single_mode(self):
        """Single Fourier mode produces cosine variation."""
        from periodica.utils.transforms.fourier_field import (
            FourierFieldEvaluator, FourierFieldConfig, FourierCoefficient3D
        )

        config = FourierFieldConfig(
            property_name='Test',
            base_value=0.0,
            domain_size_m=(1.0, 1.0, 1.0),
            coefficients=[
                FourierCoefficient3D(n=1, m=0, l=0, amplitude=10.0, phase=0.0)
            ]
        )
        evaluator = FourierFieldEvaluator(config)

        # At x=0, cos(0) = 1, so value should be 10
        assert evaluator.evaluate(0, 0, 0) == pytest.approx(10.0)

        # At x=0.5, cos(pi) = -1, so value should be -10
        assert evaluator.evaluate(0.5, 0, 0) == pytest.approx(-10.0)

    def test_periodic_boundary(self):
        """Values wrap at periodic boundary."""
        from periodica.utils.transforms.fourier_field import (
            FourierFieldEvaluator, FourierFieldConfig, FourierCoefficient3D
        )

        config = FourierFieldConfig(
            property_name='Test',
            base_value=0.0,
            domain_size_m=(1.0, 1.0, 1.0),
            coefficients=[
                FourierCoefficient3D(n=1, m=0, l=0, amplitude=5.0, phase=0.0)
            ],
            boundary_condition='periodic'
        )
        evaluator = FourierFieldEvaluator(config)

        # x=0 and x=1 should give same value (periodic)
        v0 = evaluator.evaluate(0, 0, 0)
        v1 = evaluator.evaluate(1.0, 0, 0)
        assert v0 == pytest.approx(v1, abs=1e-10)

    def test_layered_composite_factory(self):
        """Factory function creates valid layered composite field."""
        from periodica.utils.transforms.fourier_field import create_layered_composite_field

        config = create_layered_composite_field(
            property_name='YoungsModulus_GPa',
            layer_values=[200.0, 70.0],  # Steel and aluminum layers
            layer_thicknesses_m=[0.001, 0.001],
            num_harmonics=5
        )

        assert config.property_name == 'YoungsModulus_GPa'
        # Mean should be average
        expected_mean = (200.0 + 70.0) / 2
        assert config.base_value == pytest.approx(expected_mean)


class TestStochasticFieldEvaluator:
    """Test stochastic (Karhunen-Loeve) field evaluation."""

    def test_mean_value(self):
        """Field mean should be approximately config mean over many samples."""
        from periodica.utils.transforms.stochastic_field import (
            StochasticFieldEvaluator, StochasticFieldConfig
        )

        config = StochasticFieldConfig(
            property_name='Test',
            mean=100.0,
            std_dev=5.0,
            correlation_length_m=(0.01, 0.01, 0.01),
            num_terms=20,
            domain_size_m=(0.1, 0.1, 0.1)
        )
        evaluator = StochasticFieldEvaluator(config)

        # Sample statistics
        stats = evaluator.sample_statistics(num_samples=500)

        # Mean should be close to configured mean
        assert stats['mean'] == pytest.approx(100.0, rel=0.15)

    def test_covariance_exponential(self):
        """Exponential covariance decays correctly with distance."""
        from periodica.utils.transforms.stochastic_field import (
            StochasticFieldEvaluator, StochasticFieldConfig, CovarianceModel
        )

        config = StochasticFieldConfig(
            property_name='Test',
            mean=0.0,
            std_dev=1.0,
            correlation_length_m=(0.01, 0.01, 0.01),
            covariance_model=CovarianceModel.EXPONENTIAL
        )
        evaluator = StochasticFieldEvaluator(config)

        # Covariance at zero distance should be sigma^2
        cov_0 = evaluator.covariance((0, 0, 0), (0, 0, 0))
        assert cov_0 == pytest.approx(1.0)

        # Covariance at one correlation length should be sigma^2 * exp(-1)
        cov_L = evaluator.covariance((0, 0, 0), (0.01, 0, 0))
        assert cov_L == pytest.approx(math.exp(-1), rel=0.01)

    def test_granite_heterogeneity_factory(self):
        """Factory creates valid granite heterogeneity field."""
        from periodica.utils.transforms.stochastic_field import create_granite_heterogeneity_field

        config = create_granite_heterogeneity_field(
            property_name='YoungsModulus_GPa',
            quartz_fraction=0.33,
            feldspar_fraction=0.60,
            mica_fraction=0.07,
            grain_size_mm=2.0,
            seed=42
        )

        # Mean should be rule-of-mixtures weighted average
        expected_mean = 0.33 * 95 + 0.60 * 70 + 0.07 * 35
        assert config.mean == pytest.approx(expected_mean, rel=0.01)


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


class TestMaterialSampler:
    """Test the main MaterialSampler class."""

    @pytest.fixture
    def simple_material(self):
        """Create a simple material definition for testing."""
        return {
            'Name': 'Test Steel',
            'Category': 'Structural Steel',
            'ElasticProperties': {
                'YoungsModulus_GPa': 205,
                'ShearModulus_GPa': 80,
                'BulkModulus_GPa': 160,
                'PoissonsRatio': 0.29
            },
            'StrengthProperties': {
                'YieldStrength_MPa': 310,
                'UltimateTensileStrength_MPa': 565
            },
            'Hardness': {
                'Vickers_HV': 170
            },
            'ThermalProperties': {
                'ThermalConductivity_W_mK': 50,
                'SpecificHeat_J_kgK': 500,
                'ThermalExpansion_per_K': 12e-6,
                'MeltingPoint_K': 1773
            },
            'PhysicalProperties': {
                'Density_kg_m3': 7850
            },
            'ElectricalProperties': {
                'ElectricalResistivity_Ohm_m': 1.7e-7
            }
        }

    def test_basic_sampling(self, simple_material):
        """Basic sampling returns expected values."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        sampler = MaterialSampler(simple_material)

        props = sampler.sample(x=0.05, y=0.05, z=0.05)

        assert props.youngs_modulus_GPa == pytest.approx(205, rel=0.01)
        assert props.density_kg_m3 == pytest.approx(7850, rel=0.01)
        assert props.poissons_ratio == pytest.approx(0.29)

    def test_sample_property(self, simple_material):
        """Sample single property."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        sampler = MaterialSampler(simple_material)

        E = sampler.sample_property('YoungsModulus_GPa', 0, 0, 0)
        assert E == pytest.approx(205)

    def test_property_names(self, simple_material):
        """Get list of available properties."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        sampler = MaterialSampler(simple_material)
        names = sampler.get_property_names()

        assert 'YoungsModulus_GPa' in names
        assert 'Density_kg_m3' in names

    def test_material_with_spatial_fields(self):
        """Material with Fourier spatial field."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        material = {
            'Name': 'Test Material',
            'Category': 'Test',
            'ElasticProperties': {'YoungsModulus_GPa': 100},
            'PhysicalProperties': {'Density_kg_m3': 5000},
            'SpatialFields': {
                'FourierFields': [
                    {
                        'property': 'Density_kg_m3',
                        'base_value': 5000,
                        'domain_size_m': [1.0, 1.0, 1.0],
                        'coefficients': [
                            {'n': 1, 'm': 0, 'l': 0, 'amplitude': 100, 'phase': 0}
                        ]
                    }
                ]
            }
        }

        sampler = MaterialSampler(material)

        # At x=0, cos(0)=1, variation should add 100
        rho_0 = sampler.sample_property('Density_kg_m3', 0, 0, 0)
        assert rho_0 == pytest.approx(5100, rel=0.01)

        # At x=0.5, cos(pi)=-1, variation should subtract 100
        rho_half = sampler.sample_property('Density_kg_m3', 0.5, 0, 0)
        assert rho_half == pytest.approx(4900, rel=0.01)


class TestGraniteMaterial:
    """Test granite material loading and sampling."""

    @pytest.fixture
    def granite_material(self):
        """Load Westerly granite material."""
        import json
        from pathlib import Path

        granite_path = Path(__file__).parent.parent / 'src' / 'periodica' / 'data' / 'active' / 'materials' / 'Granite_Westerly.json'
        if not granite_path.exists():
            pytest.skip("Granite material file not found")

        with open(granite_path, 'r') as f:
            return json.load(f)

    def test_load_granite(self, granite_material):
        """Granite material loads correctly."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        sampler = MaterialSampler(granite_material)

        assert sampler.name == 'Westerly Granite'
        assert 'YoungsModulus_GPa' in sampler.get_property_names()

    def test_granite_spatial_variation(self, granite_material):
        """Granite shows spatial variation in properties."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        sampler = MaterialSampler(granite_material)

        # Sample at multiple points
        values = []
        for i in range(5):
            v = sampler.sample_property('YoungsModulus_GPa',
                                        0.01 * i, 0.01 * i, 0.005 * i)
            values.append(v)

        # With stochastic field, values should vary
        assert len(set([round(v, 1) for v in values])) > 1, \
            "Granite should show spatial variation"

    def test_granite_temperature_dependence(self, granite_material):
        """Granite properties vary with temperature."""
        from periodica.utils.transforms.material_sampler import MaterialSampler

        sampler = MaterialSampler(granite_material)

        # Room temperature
        E_293 = sampler.sample_property('YoungsModulus_GPa', 0.05, 0.05, 0.025,
                                        T=293.15)

        # Above quartz transition
        E_900 = sampler.sample_property('YoungsModulus_GPa', 0.05, 0.05, 0.025,
                                        T=900)

        # E should decrease significantly at high temperature
        assert E_900 < E_293 * 0.7, \
            f"Granite E should decrease at high T: E_293={E_293}, E_900={E_900}"


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
