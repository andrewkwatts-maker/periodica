"""
Material Sampler - Unified Spatial Property Sampling
====================================================

Orchestrates all transform modules to sample material properties at
arbitrary X,Y,Z locations with temperature and pressure dependencies.

This is the main interface for sampling materials in simulations.

Usage:
    from periodica.utils.transforms import MaterialSampler

    # Load material from JSON
    with open('data/active/materials/Granite_Westerly.json') as f:
        material_json = json.load(f)

    sampler = MaterialSampler(material_json)

    # Sample at a point
    props = sampler.sample(
        x=0.025, y=0.015, z=0.010,  # Position in meters
        T=573.15,  # Temperature in Kelvin
        P=1e8      # Pressure in Pascals
    )

    print(props['YoungsModulus_GPa'])
    print(props['Density_kg_m3'])
    print(props['Color_RGB'])
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

from .fourier_field import FourierFieldEvaluator, FourierFieldConfig
from .wavelet_field import WaveletFieldEvaluator, WaveletConfig, WaveletCoefficients3D
from .stochastic_field import StochasticFieldEvaluator, StochasticFieldConfig
from .thermo_pressure import (
    ThermoPressureSampler,
    TemperatureDependenceConfig,
    TemperatureDependence,
    BirchMurnaghanEOS,
    BirchMurnaghanConfig
)
from .optical_properties import OpticalMaterial, SpectralProperty, OpticalModel


@dataclass
class SampledProperties:
    """
    Container for sampled material properties at a point.

    All mechanical, thermal, electrical, and optical properties
    evaluated at specific (x, y, z, T, P) conditions.
    """
    # Location
    x: float
    y: float
    z: float
    temperature_K: float
    pressure_Pa: float

    # Mechanical properties
    youngs_modulus_GPa: float = 0.0
    shear_modulus_GPa: float = 0.0
    bulk_modulus_GPa: float = 0.0
    poissons_ratio: float = 0.3
    yield_strength_MPa: float = 0.0
    ultimate_strength_MPa: float = 0.0
    hardness_HV: float = 0.0

    # Thermal properties
    density_kg_m3: float = 0.0
    thermal_conductivity_W_mK: float = 0.0
    specific_heat_J_kgK: float = 0.0
    thermal_expansion_per_K: float = 0.0

    # Electrical properties
    electrical_resistivity_Ohm_m: float = 0.0

    # Optical properties
    color_RGB: Tuple[int, int, int] = (128, 128, 128)
    reflectance_550nm: float = 0.5
    emissivity: float = 0.9

    # Phase/material info
    phase: str = "matrix"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position_m': [self.x, self.y, self.z],
            'temperature_K': self.temperature_K,
            'pressure_Pa': self.pressure_Pa,
            'YoungsModulus_GPa': self.youngs_modulus_GPa,
            'ShearModulus_GPa': self.shear_modulus_GPa,
            'BulkModulus_GPa': self.bulk_modulus_GPa,
            'PoissonsRatio': self.poissons_ratio,
            'YieldStrength_MPa': self.yield_strength_MPa,
            'UltimateStrength_MPa': self.ultimate_strength_MPa,
            'Hardness_HV': self.hardness_HV,
            'Density_kg_m3': self.density_kg_m3,
            'ThermalConductivity_W_mK': self.thermal_conductivity_W_mK,
            'SpecificHeat_J_kgK': self.specific_heat_J_kgK,
            'ThermalExpansion_per_K': self.thermal_expansion_per_K,
            'ElectricalResistivity_Ohm_m': self.electrical_resistivity_Ohm_m,
            'Color_RGB': list(self.color_RGB),
            'Reflectance_550nm': self.reflectance_550nm,
            'Emissivity': self.emissivity,
            'Phase': self.phase,
            'Confidence': self.confidence
        }


class MaterialSampler:
    """
    Unified sampler for spatially-varying material properties.

    Combines:
    - Base scalar properties (constant values)
    - Fourier fields (periodic variations)
    - Wavelet fields (multi-scale variations)
    - Stochastic fields (random variability)
    - Temperature dependencies
    - Pressure dependencies
    - Optical/spectral properties

    Properties are combined as:
        P(x,y,z,T,P) = P_base(T,P) + P_fourier(x,y,z) + P_wavelet(x,y,z) + P_random(x,y,z)
    """

    def __init__(self, material_json: Dict[str, Any]):
        """
        Initialize sampler from material JSON specification.

        Args:
            material_json: Material definition dictionary
        """
        self.material_json = material_json
        self.name = material_json.get('Name', 'Unknown')

        # Initialize property stores
        self._base_properties: Dict[str, float] = {}
        self._fourier_fields: Dict[str, FourierFieldEvaluator] = {}
        self._wavelet_fields: Dict[str, WaveletFieldEvaluator] = {}
        self._stochastic_fields: Dict[str, StochasticFieldEvaluator] = {}
        self._thermo_pressure = ThermoPressureSampler()
        self._optical: Optional[OpticalMaterial] = None

        # Parse material JSON
        self._parse_base_properties()
        self._parse_spatial_fields()
        self._parse_temperature_dependencies()
        self._parse_pressure_dependencies()
        self._parse_optical_properties()

    def _parse_base_properties(self):
        """Extract base scalar properties from JSON."""
        # Elastic properties
        elastic = self.material_json.get('ElasticProperties', {})
        self._base_properties['YoungsModulus_GPa'] = elastic.get('YoungsModulus_GPa', 200)
        self._base_properties['ShearModulus_GPa'] = elastic.get('ShearModulus_GPa', 80)
        self._base_properties['BulkModulus_GPa'] = elastic.get('BulkModulus_GPa', 160)
        self._base_properties['PoissonsRatio'] = elastic.get('PoissonsRatio', 0.3)

        # Strength properties
        strength = self.material_json.get('StrengthProperties', {})
        self._base_properties['YieldStrength_MPa'] = strength.get('YieldStrength_MPa', 250)
        self._base_properties['UltimateTensileStrength_MPa'] = strength.get(
            'UltimateTensileStrength_MPa', 400)

        # Hardness
        hardness = self.material_json.get('Hardness', {})
        self._base_properties['Hardness_HV'] = hardness.get('Vickers_HV', 150)

        # Thermal properties
        thermal = self.material_json.get('ThermalProperties', {})
        self._base_properties['ThermalConductivity_W_mK'] = thermal.get(
            'ThermalConductivity_W_mK', 50)
        self._base_properties['SpecificHeat_J_kgK'] = thermal.get('SpecificHeat_J_kgK', 500)
        self._base_properties['ThermalExpansion_per_K'] = thermal.get(
            'ThermalExpansion_per_K', 12e-6)
        self._base_properties['MeltingPoint_K'] = thermal.get('MeltingPoint_K', 1800)

        # Physical properties
        physical = self.material_json.get('PhysicalProperties', {})
        self._base_properties['Density_kg_m3'] = physical.get('Density_kg_m3', 7850)

        # Electrical properties
        electrical = self.material_json.get('ElectricalProperties', {})
        self._base_properties['ElectricalResistivity_Ohm_m'] = electrical.get(
            'ElectricalResistivity_Ohm_m', 1e-7)

    def _parse_spatial_fields(self):
        """Parse Fourier, wavelet, and stochastic field definitions."""
        spatial = self.material_json.get('SpatialFields', {})

        # Fourier fields
        for field_def in spatial.get('FourierFields', []):
            config = FourierFieldConfig.from_dict(field_def)
            self._fourier_fields[config.property_name] = FourierFieldEvaluator(config)

        # Wavelet fields
        for field_def in spatial.get('WaveletFields', []):
            config = WaveletConfig.from_dict(field_def.get('config', field_def))
            coeffs = None
            if 'coefficients' in field_def:
                coeffs = WaveletCoefficients3D.from_dict(field_def['coefficients'])
            self._wavelet_fields[config.property_name] = WaveletFieldEvaluator(config, coeffs)

        # Stochastic fields
        for field_def in spatial.get('StochasticFields', []):
            config = StochasticFieldConfig.from_dict(field_def)
            self._stochastic_fields[config.property_name] = StochasticFieldEvaluator(config)

    def _parse_temperature_dependencies(self):
        """Parse temperature dependence definitions."""
        temp_deps = self.material_json.get('TemperatureDependencies', [])

        for dep_def in temp_deps:
            config = TemperatureDependenceConfig.from_dict(dep_def)
            self._thermo_pressure.add_temperature_dependence(config.property_name, config)

    def _parse_pressure_dependencies(self):
        """Parse pressure dependence (EOS) definitions."""
        pressure_deps = self.material_json.get('PressureDependencies', [])

        for dep_def in pressure_deps:
            prop_name = dep_def.get('property', 'Density_kg_m3')
            eos_config = BirchMurnaghanConfig(
                K0_GPa=dep_def.get('K0_GPa', 160),
                K0_prime=dep_def.get('K0_prime', 4.0),
                reference_density_kg_m3=self._base_properties.get('Density_kg_m3', 7850)
            )
            eos = BirchMurnaghanEOS(eos_config)
            self._thermo_pressure.add_pressure_dependence(
                prop_name, eos,
                self._base_properties.get(prop_name)
            )

    def _parse_optical_properties(self):
        """Parse optical property definitions."""
        optical_def = self.material_json.get('OpticalProperties', {})

        if optical_def:
            # Create spectral property for refractive index
            n_def = optical_def.get('refractive_index', {})
            model = OpticalModel(n_def.get('model', 'constant'))
            n_property = SpectralProperty(
                model=model,
                constant_value=n_def.get('value', 1.5)
            )

            self._optical = OpticalMaterial(
                name=self.name,
                refractive_index=n_property,
                emissivity=optical_def.get('emissivity', 0.9),
                surface_roughness_um=optical_def.get('surface_roughness_um', 1.0)
            )
        else:
            # Create default optical properties based on material type
            self._optical = self._create_default_optical()

    def _create_default_optical(self) -> OpticalMaterial:
        """
        Create default optical properties when none specified in JSON.

        Uses standardized defaults - all material-specific optical properties
        should be defined in the material JSON, not hardcoded here.
        """
        # Standard default optical properties for any material
        # Materials should define their own OpticalProperties in JSON
        return OpticalMaterial(
            name=self.name,
            emissivity=0.85,  # Reasonable default for most materials
            surface_roughness_um=10.0  # Moderate roughness
        )

    def sample(
        self,
        x: float,
        y: float,
        z: float,
        T: float = 293.15,
        P: float = 101325.0
    ) -> SampledProperties:
        """
        Sample all material properties at given location and conditions.

        Args:
            x, y, z: Position in meters
            T: Temperature in Kelvin (default: 20C)
            P: Pressure in Pascals (default: 1 atm)

        Returns:
            SampledProperties with all evaluated properties
        """
        result = SampledProperties(
            x=x, y=y, z=z,
            temperature_K=T,
            pressure_Pa=P
        )

        # Sample each property
        result.youngs_modulus_GPa = self.sample_property('YoungsModulus_GPa', x, y, z, T, P)
        result.shear_modulus_GPa = self.sample_property('ShearModulus_GPa', x, y, z, T, P)
        result.bulk_modulus_GPa = self.sample_property('BulkModulus_GPa', x, y, z, T, P)
        result.poissons_ratio = self._base_properties.get('PoissonsRatio', 0.3)
        result.yield_strength_MPa = self.sample_property('YieldStrength_MPa', x, y, z, T, P)
        result.ultimate_strength_MPa = self.sample_property(
            'UltimateTensileStrength_MPa', x, y, z, T, P)
        result.hardness_HV = self.sample_property('Hardness_HV', x, y, z, T, P)

        result.density_kg_m3 = self.sample_property('Density_kg_m3', x, y, z, T, P)
        result.thermal_conductivity_W_mK = self.sample_property(
            'ThermalConductivity_W_mK', x, y, z, T, P)
        result.specific_heat_J_kgK = self.sample_property('SpecificHeat_J_kgK', x, y, z, T, P)
        result.thermal_expansion_per_K = self.sample_property(
            'ThermalExpansion_per_K', x, y, z, T, P)

        result.electrical_resistivity_Ohm_m = self.sample_property(
            'ElectricalResistivity_Ohm_m', x, y, z, T, P)

        # Optical properties
        if self._optical:
            result.color_RGB = self._optical.to_rgb()
            result.reflectance_550nm = self._optical.reflectance_normal(550)
            result.emissivity = self._optical.emissivity

            # Temperature-dependent color (thermal emission)
            if T > 700:
                result.color_RGB = self._optical.blackbody_emission_rgb(T)

        return result

    def sample_property(
        self,
        property_name: str,
        x: float,
        y: float,
        z: float,
        T: float = 293.15,
        P: float = 101325.0
    ) -> float:
        """
        Sample a single property at given location and conditions.

        Combines:
        P(x,y,z,T,P) = P_base(T,P) + P_fourier(x,y,z) + P_wavelet(x,y,z) + P_random(x,y,z)

        Note: Spatial fields (Fourier, wavelet) include their own base values,
        so we use their output directly rather than adding to base_properties.

        Args:
            property_name: Name of property to sample
            x, y, z: Position in meters
            T: Temperature in Kelvin
            P: Pressure in Pascals

        Returns:
            Property value at specified conditions
        """
        # Check if property has a spatial field (Fourier overrides base)
        has_fourier = property_name in self._fourier_fields
        has_wavelet = property_name in self._wavelet_fields
        has_stochastic = property_name in self._stochastic_fields

        # Start with base value (will be overridden if spatial field exists)
        value = self._base_properties.get(property_name, 0.0)

        # Apply temperature/pressure dependence to base value
        if property_name in self._thermo_pressure._temp_deps:
            value = self._thermo_pressure.sample(property_name, T, P)
        elif property_name in self._thermo_pressure._pressure_eos:
            value = self._thermo_pressure.sample(property_name, T, P)

        # For Fourier fields, use the field's value directly (includes base_value)
        if has_fourier:
            # Fourier evaluate() returns base_value + sum(coefficients)
            # So this replaces the base value, not adds to it
            value = self._fourier_fields[property_name].evaluate(x, y, z)

        # Add wavelet field contribution
        if property_name in self._wavelet_fields:
            wavelet_contrib = self._wavelet_fields[property_name].evaluate(x, y, z)
            value += wavelet_contrib

        # Add stochastic field contribution
        if property_name in self._stochastic_fields:
            # Stochastic field returns absolute value (already includes mean)
            stochastic_value = self._stochastic_fields[property_name].evaluate(x, y, z)
            # Use stochastic variation as perturbation
            base_mean = self._stochastic_fields[property_name].config.mean
            value += (stochastic_value - base_mean)

        return value

    def get_property_names(self) -> List[str]:
        """Get list of all available property names."""
        return list(self._base_properties.keys())

    def get_spatial_fields(self) -> Dict[str, str]:
        """Get mapping of property names to their spatial field types."""
        result = {}
        for name in self._fourier_fields:
            result[name] = 'fourier'
        for name in self._wavelet_fields:
            result[name] = 'wavelet'
        for name in self._stochastic_fields:
            result[name] = 'stochastic'
        return result

    def to_enhanced_json(self) -> Dict[str, Any]:
        """
        Export material with all transform specifications to JSON.

        Returns complete material definition including spatial fields,
        temperature dependencies, and optical properties.
        """
        result = dict(self.material_json)

        # Add spatial fields
        result['SpatialFields'] = {
            'FourierFields': [
                f.config.to_dict() for f in self._fourier_fields.values()
            ],
            'WaveletFields': [
                {
                    'config': f.config.to_dict(),
                    'coefficients': f.coefficients.to_dict()
                }
                for f in self._wavelet_fields.values()
            ],
            'StochasticFields': [
                f.config.to_dict() for f in self._stochastic_fields.values()
            ]
        }

        # Add T-P dependencies
        result['ThermoPressure'] = self._thermo_pressure.to_dict()

        # Add optical properties
        if self._optical:
            result['OpticalProperties'] = self._optical.to_dict()

        return result

    def save_enhanced_json(self, filepath: str):
        """Save enhanced material definition to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_enhanced_json(), f, indent=2)

    @classmethod
    def from_json_file(cls, filepath: str) -> 'MaterialSampler':
        """Create MaterialSampler from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data)
