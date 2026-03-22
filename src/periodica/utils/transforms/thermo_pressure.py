"""
Temperature and Pressure Dependence Module
==========================================

Models for temperature and pressure dependent material properties.

Physical Models Implemented:
    - Linear/Polynomial temperature dependence
    - Arrhenius (thermally activated) processes
    - Birch-Murnaghan equation of state (pressure-volume)
    - Murnaghan equation of state
    - Combined T-P dependencies

References:
    - Birch-Murnaghan EOS: https://en.wikipedia.org/wiki/Birch–Murnaghan_equation_of_state
    - Thermodynamic EOS across materials (npj Computational Materials)
    - ASM Metals Handbook temperature-dependent data
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
import json


# Physical constants
R_GAS = 8.314462  # J/(mol*K)
BOLTZMANN = 1.380649e-23  # J/K


class TemperatureModel(Enum):
    """Available temperature dependence models."""
    CONSTANT = "constant"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    ARRHENIUS = "arrhenius"
    TABULATED = "tabulated"


class PressureModel(Enum):
    """Available pressure dependence models."""
    CONSTANT = "constant"
    LINEAR = "linear"
    BIRCH_MURNAGHAN_2 = "birch_murnaghan_2nd"
    BIRCH_MURNAGHAN_3 = "birch_murnaghan_3rd"
    MURNAGHAN = "murnaghan"
    TABULATED = "tabulated"


@dataclass
class TemperatureDependenceConfig:
    """
    Configuration for temperature-dependent property.

    Attributes:
        property_name: Name of the property
        model: Temperature dependence model type
        reference_T_K: Reference temperature in Kelvin
        valid_range_K: Valid temperature range [T_min, T_max]
        coefficients: Model-specific coefficients
        activation_energy_J_mol: For Arrhenius model (J/mol)
        pre_exponential: For Arrhenius model
        tabulated_data: For tabulated model [(T, value), ...]
    """
    property_name: str
    model: TemperatureModel = TemperatureModel.CONSTANT
    reference_T_K: float = 293.15  # 20C
    reference_value: float = 1.0
    valid_range_K: Tuple[float, float] = (200.0, 1500.0)
    coefficients: Dict[str, float] = field(default_factory=dict)
    activation_energy_J_mol: float = 0.0
    pre_exponential: float = 1.0
    tabulated_data: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'property': self.property_name,
            'model': self.model.value,
            'reference_T_K': self.reference_T_K,
            'reference_value': self.reference_value,
            'valid_range_K': list(self.valid_range_K),
            'coefficients': self.coefficients,
            'activation_energy_J_mol': self.activation_energy_J_mol,
            'pre_exponential': self.pre_exponential,
            'tabulated_data': self.tabulated_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemperatureDependenceConfig':
        return cls(
            property_name=data.get('property', ''),
            model=TemperatureModel(data.get('model', 'constant')),
            reference_T_K=data.get('reference_T_K', 293.15),
            reference_value=data.get('reference_value', 1.0),
            valid_range_K=tuple(data.get('valid_range_K', [200.0, 1500.0])),
            coefficients=data.get('coefficients', {}),
            activation_energy_J_mol=data.get('activation_energy_J_mol', 0.0),
            pre_exponential=data.get('pre_exponential', 1.0),
            tabulated_data=data.get('tabulated_data', [])
        )


class TemperatureDependence:
    """
    Evaluates temperature-dependent material properties.

    Models:
        - CONSTANT: P(T) = P_ref
        - LINEAR: P(T) = P_ref * (1 + alpha * (T - T_ref))
        - POLYNOMIAL: P(T) = a0 + a1*dT + a2*dT^2 + a3*dT^3 + ...
        - ARRHENIUS: P(T) = A * exp(-Q / (R*T))
        - TABULATED: Linear interpolation of (T, P) data
    """

    def __init__(self, config: TemperatureDependenceConfig):
        if isinstance(config, dict):
            config = TemperatureDependenceConfig.from_dict(config)
        self.config = config

        # Sort tabulated data
        if self.config.tabulated_data:
            self.config.tabulated_data.sort(key=lambda x: x[0])

    def evaluate(self, T: float) -> float:
        """
        Evaluate property at temperature T.

        Args:
            T: Temperature in Kelvin

        Returns:
            Property value at temperature T
        """
        # Clamp to valid range
        T_min, T_max = self.config.valid_range_K
        T = max(T_min, min(T_max, T))

        if self.config.model == TemperatureModel.CONSTANT:
            return self.config.reference_value

        elif self.config.model == TemperatureModel.LINEAR:
            alpha = self.config.coefficients.get('alpha', 0.0)
            dT = T - self.config.reference_T_K
            return self.config.reference_value * (1 + alpha * dT)

        elif self.config.model == TemperatureModel.POLYNOMIAL:
            dT = T - self.config.reference_T_K
            result = self.config.coefficients.get('a0', self.config.reference_value)
            for i in range(1, 6):
                coef_name = f'a{i}'
                if coef_name in self.config.coefficients:
                    result += self.config.coefficients[coef_name] * (dT ** i)
            return result

        elif self.config.model == TemperatureModel.ARRHENIUS:
            Q = self.config.activation_energy_J_mol
            A = self.config.pre_exponential
            if T <= 0:
                return 0.0
            return A * math.exp(-Q / (R_GAS * T))

        elif self.config.model == TemperatureModel.TABULATED:
            return self._interpolate_tabulated(T)

        return self.config.reference_value

    def _interpolate_tabulated(self, T: float) -> float:
        """Linear interpolation of tabulated data."""
        data = self.config.tabulated_data
        if not data:
            return self.config.reference_value

        if T <= data[0][0]:
            return data[0][1]
        if T >= data[-1][0]:
            return data[-1][1]

        for i in range(len(data) - 1):
            T1, V1 = data[i]
            T2, V2 = data[i + 1]
            if T1 <= T <= T2:
                t = (T - T1) / (T2 - T1)
                return V1 + t * (V2 - V1)

        return self.config.reference_value

    @property
    def property_name(self) -> str:
        return self.config.property_name


@dataclass
class BirchMurnaghanConfig:
    """
    Configuration for Birch-Murnaghan equation of state.

    3rd-order Birch-Murnaghan:
        P = (3/2) * K_0 * [(V_0/V)^(7/3) - (V_0/V)^(5/3)] *
            [1 + (3/4)*(K_0' - 4)*((V_0/V)^(2/3) - 1)]

    Attributes:
        K0_GPa: Bulk modulus at zero pressure (GPa)
        K0_prime: Pressure derivative of bulk modulus (dimensionless)
        V0_m3_mol: Reference molar volume (m^3/mol)
        reference_pressure_Pa: Reference pressure (Pa)
        reference_density_kg_m3: Reference density (kg/m3)
    """
    K0_GPa: float = 160.0
    K0_prime: float = 4.0
    V0_m3_mol: float = 7.09e-6  # Iron
    reference_pressure_Pa: float = 101325.0
    reference_density_kg_m3: float = 7850.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'K0_GPa': self.K0_GPa,
            'K0_prime': self.K0_prime,
            'V0_m3_mol': self.V0_m3_mol,
            'reference_pressure_Pa': self.reference_pressure_Pa,
            'reference_density_kg_m3': self.reference_density_kg_m3
        }


class BirchMurnaghanEOS:
    """
    Birch-Murnaghan equation of state for pressure-volume relations.

    Used to calculate:
    - Volume/density at given pressure
    - Pressure at given volume/density
    - Bulk modulus at pressure

    Example:
        config = BirchMurnaghanConfig(K0_GPa=160, K0_prime=4.5)
        eos = BirchMurnaghanEOS(config)
        density = eos.density_at_pressure(1e9)  # 1 GPa
    """

    def __init__(self, config: BirchMurnaghanConfig):
        if isinstance(config, dict):
            config = BirchMurnaghanConfig(**config)
        self.config = config
        self.K0 = config.K0_GPa * 1e9  # Convert to Pa
        self.K0_prime = config.K0_prime
        self.V0 = config.V0_m3_mol
        self.rho0 = config.reference_density_kg_m3

    def pressure_from_compression(self, eta: float) -> float:
        """
        Calculate pressure from compression ratio.

        Args:
            eta: Compression ratio (V0/V)^(1/3)

        Returns:
            Pressure in Pascals
        """
        if eta <= 0:
            return 0.0

        eta2 = eta * eta
        f = 0.5 * (eta2 - 1)

        # 3rd order BM
        P = 3 * self.K0 * f * ((1 + 2 * f) ** 2.5) * (
            1 + 1.5 * (self.K0_prime - 4) * f
        )

        return P

    def pressure_from_volume(self, V: float) -> float:
        """
        Calculate pressure given volume.

        Args:
            V: Volume in m^3/mol

        Returns:
            Pressure in Pascals
        """
        if V <= 0:
            return 1e15  # Very high pressure for zero volume
        eta = (self.V0 / V) ** (1 / 3)
        return self.pressure_from_compression(eta)

    def volume_at_pressure(self, P: float, tolerance: float = 1e-6,
                           max_iterations: int = 100) -> float:
        """
        Calculate volume at given pressure (Newton-Raphson iteration).

        Args:
            P: Pressure in Pascals
            tolerance: Relative tolerance for convergence
            max_iterations: Maximum iterations

        Returns:
            Volume in m^3/mol
        """
        if P <= self.config.reference_pressure_Pa:
            return self.V0

        # Initial guess using linear approximation
        V = self.V0 * (1 - P / (3 * self.K0))
        V = max(V, self.V0 * 0.5)  # Don't compress more than 50%

        for _ in range(max_iterations):
            P_calc = self.pressure_from_volume(V)
            if abs(P_calc - P) / P < tolerance:
                break

            # Numerical derivative
            dV = V * 1e-6
            dP_dV = (self.pressure_from_volume(V + dV) -
                     self.pressure_from_volume(V - dV)) / (2 * dV)

            if abs(dP_dV) < 1e-30:
                break

            # Newton step
            V_new = V - (P_calc - P) / dP_dV
            V_new = max(V_new, self.V0 * 0.3)  # Prevent negative volume
            V = V_new

        return V

    def density_at_pressure(self, P: float) -> float:
        """
        Calculate density at given pressure.

        Args:
            P: Pressure in Pascals

        Returns:
            Density in kg/m^3
        """
        V = self.volume_at_pressure(P)
        # rho = rho0 * V0/V
        return self.rho0 * (self.V0 / V)

    def bulk_modulus_at_pressure(self, P: float) -> float:
        """
        Calculate bulk modulus at given pressure.

        Uses K(P) = K0 + K0' * P (linear approximation)
        """
        return self.K0 + self.K0_prime * P


class ThermoPressureSampler:
    """
    Combined temperature and pressure sampler for material properties.

    Combines T-dependence and P-dependence models to calculate
    property values at arbitrary (T, P) conditions.

    Example:
        sampler = ThermoPressureSampler()
        sampler.add_temperature_dependence('YoungsModulus_GPa', temp_config)
        sampler.add_pressure_dependence('Density_kg_m3', eos)

        E = sampler.sample('YoungsModulus_GPa', T=573.15, P=1e8)
        rho = sampler.sample('Density_kg_m3', T=573.15, P=1e8)
    """

    def __init__(self):
        self._temp_deps: Dict[str, TemperatureDependence] = {}
        self._pressure_eos: Dict[str, BirchMurnaghanEOS] = {}
        self._base_values: Dict[str, float] = {}

    def add_temperature_dependence(
        self,
        property_name: str,
        config: TemperatureDependenceConfig
    ):
        """Add temperature dependence for a property."""
        self._temp_deps[property_name] = TemperatureDependence(config)
        self._base_values[property_name] = config.reference_value

    def add_pressure_dependence(
        self,
        property_name: str,
        eos: BirchMurnaghanEOS,
        reference_value: float = None
    ):
        """Add pressure dependence using Birch-Murnaghan EOS."""
        self._pressure_eos[property_name] = eos
        if reference_value is not None:
            self._base_values[property_name] = reference_value

    def sample(
        self,
        property_name: str,
        T: float = 293.15,
        P: float = 101325.0
    ) -> float:
        """
        Sample property at given temperature and pressure.

        Args:
            property_name: Name of property to sample
            T: Temperature in Kelvin
            P: Pressure in Pascals

        Returns:
            Property value at (T, P)
        """
        base_value = self._base_values.get(property_name, 1.0)

        # Apply temperature dependence
        if property_name in self._temp_deps:
            value = self._temp_deps[property_name].evaluate(T)
        else:
            value = base_value

        # Apply pressure dependence (multiplicative for most properties)
        if property_name in self._pressure_eos:
            eos = self._pressure_eos[property_name]
            if 'Density' in property_name or 'density' in property_name:
                # For density, directly use EOS
                value = eos.density_at_pressure(P)
            else:
                # For other properties, scale by compression
                V_P = eos.volume_at_pressure(P)
                compression_ratio = eos.V0 / V_P
                # Property scales with compression (simplified)
                value *= compression_ratio ** 0.3

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            'base_values': self._base_values,
            'temperature_dependencies': {},
            'pressure_dependencies': {}
        }

        for name, dep in self._temp_deps.items():
            result['temperature_dependencies'][name] = dep.config.to_dict()

        for name, eos in self._pressure_eos.items():
            result['pressure_dependencies'][name] = eos.config.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThermoPressureSampler':
        """Create from dictionary."""
        sampler = cls()
        sampler._base_values = data.get('base_values', {})

        for name, config_dict in data.get('temperature_dependencies', {}).items():
            config = TemperatureDependenceConfig.from_dict(config_dict)
            sampler._temp_deps[name] = TemperatureDependence(config)

        for name, eos_dict in data.get('pressure_dependencies', {}).items():
            config = BirchMurnaghanConfig(**eos_dict)
            sampler._pressure_eos[name] = BirchMurnaghanEOS(config)

        return sampler


# Common temperature coefficients from literature
STEEL_TEMP_COEFFICIENTS = {
    'YoungsModulus_GPa': {
        'model': 'polynomial',
        'reference_T_K': 293.15,
        'reference_value': 205.0,
        'coefficients': {
            'a0': 205.0,
            'a1': -0.025,  # ~-0.025 GPa/K (typical steel)
            'a2': -1.5e-5
        },
        'valid_range_K': [200, 1200]
    },
    'YieldStrength_MPa': {
        'model': 'polynomial',
        'reference_T_K': 293.15,
        'reference_value': 310.0,
        'coefficients': {
            'a0': 310.0,
            'a1': -0.15,  # ~-0.15 MPa/K
            'a2': -5e-4
        },
        'valid_range_K': [200, 1000]
    },
    'ThermalConductivity_W_mK': {
        'model': 'polynomial',
        'reference_T_K': 293.15,
        'reference_value': 50.0,
        'coefficients': {
            'a0': 50.0,
            'a1': -0.02
        },
        'valid_range_K': [200, 1200]
    }
}

ALUMINUM_TEMP_COEFFICIENTS = {
    'YoungsModulus_GPa': {
        'model': 'polynomial',
        'reference_T_K': 293.15,
        'reference_value': 70.0,
        'coefficients': {
            'a0': 70.0,
            'a1': -0.015,
            'a2': -8e-6
        },
        'valid_range_K': [200, 600]
    }
}

GRANITE_TEMP_COEFFICIENTS = {
    'YoungsModulus_GPa': {
        'model': 'tabulated',
        'reference_T_K': 293.15,
        'reference_value': 50.0,
        'tabulated_data': [
            (293, 50.0),
            (473, 48.0),
            (573, 42.0),  # Pre-transition
            (673, 35.0),  # Quartz alpha-beta transition
            (873, 25.0),
            (1073, 15.0)
        ],
        'valid_range_K': [200, 1100]
    }
}
