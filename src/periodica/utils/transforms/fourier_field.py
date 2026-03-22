"""
Fourier Field Evaluator
=======================

Evaluates Fourier series representations of spatially varying material properties.
Used for periodic structures like grain periodicity, laminated composites, etc.

Mathematical Basis:
    P(x,y,z) = P_0 + sum_{n,m,l} A_{nml} * cos(2*pi*(n*x/Lx + m*y/Ly + l*z/Lz) + phi_{nml})

References:
    - Two-scale FE-FFT computational modeling (ScienceDirect)
    - FFT-based schemes for mechanical response of materials
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json


@dataclass
class FourierCoefficient3D:
    """
    Single Fourier coefficient for 3D spatial field.

    Represents: A * cos(2*pi*(n*x/Lx + m*y/Ly + l*z/Lz) + phase)

    Attributes:
        n: Wavenumber in x-direction (integer)
        m: Wavenumber in y-direction (integer)
        l: Wavenumber in z-direction (integer)
        amplitude: Coefficient amplitude (same units as property)
        phase: Phase offset in radians
    """
    n: int = 0
    m: int = 0
    l: int = 0
    amplitude: float = 0.0
    phase: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'n': self.n,
            'm': self.m,
            'l': self.l,
            'amplitude': self.amplitude,
            'phase': self.phase
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FourierCoefficient3D':
        """Create from dictionary."""
        return cls(
            n=data.get('n', 0),
            m=data.get('m', 0),
            l=data.get('l', 0),
            amplitude=data.get('amplitude', 0.0),
            phase=data.get('phase', 0.0)
        )


@dataclass
class FourierFieldConfig:
    """
    Configuration for a Fourier-represented spatial field.

    Attributes:
        property_name: Name of the property being represented
        base_value: DC component (mean value)
        domain_size_m: Size of periodic domain [Lx, Ly, Lz] in meters
        coefficients: List of Fourier coefficients
        boundary_condition: 'periodic' or 'reflective'
    """
    property_name: str
    base_value: float = 0.0
    domain_size_m: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    coefficients: List[FourierCoefficient3D] = field(default_factory=list)
    boundary_condition: str = 'periodic'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'property': self.property_name,
            'base_value': self.base_value,
            'domain_size_m': list(self.domain_size_m),
            'coefficients': [c.to_dict() for c in self.coefficients],
            'boundary_condition': self.boundary_condition
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FourierFieldConfig':
        """Create from dictionary."""
        coeffs = [FourierCoefficient3D.from_dict(c)
                  for c in data.get('coefficients', [])]
        domain = data.get('domain_size_m', [0.1, 0.1, 0.1])
        return cls(
            property_name=data.get('property', ''),
            base_value=data.get('base_value', 0.0),
            domain_size_m=tuple(domain),
            coefficients=coeffs,
            boundary_condition=data.get('boundary_condition', 'periodic')
        )


class FourierFieldEvaluator:
    """
    Evaluates Fourier series representation of spatial property fields.

    The field is represented as:
        P(x,y,z) = base_value + sum(A_i * cos(k_i . r + phi_i))

    where k_i is the wave vector and r = (x, y, z).

    Example:
        config = FourierFieldConfig(
            property_name='YoungsModulus_GPa',
            base_value=205.0,
            domain_size_m=(0.01, 0.01, 0.01),
            coefficients=[
                FourierCoefficient3D(n=1, m=0, l=0, amplitude=5.0, phase=0.0),
                FourierCoefficient3D(n=0, m=1, l=0, amplitude=3.0, phase=1.57),
            ]
        )
        evaluator = FourierFieldEvaluator(config)
        value = evaluator.evaluate(0.005, 0.003, 0.001)
    """

    def __init__(self, config: FourierFieldConfig):
        """
        Initialize evaluator with field configuration.

        Args:
            config: FourierFieldConfig or dictionary specification
        """
        if isinstance(config, dict):
            config = FourierFieldConfig.from_dict(config)
        self.config = config
        self._precompute_wavenumbers()

    def _precompute_wavenumbers(self):
        """Precompute 2*pi/L factors for efficiency."""
        Lx, Ly, Lz = self.config.domain_size_m
        self._kx_factor = 2 * math.pi / Lx if Lx > 0 else 0
        self._ky_factor = 2 * math.pi / Ly if Ly > 0 else 0
        self._kz_factor = 2 * math.pi / Lz if Lz > 0 else 0

    def evaluate(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the field at point (x, y, z).

        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            z: Z-coordinate in meters

        Returns:
            Property value at the given point
        """
        # Handle boundary conditions
        if self.config.boundary_condition == 'periodic':
            Lx, Ly, Lz = self.config.domain_size_m
            x = x % Lx if Lx > 0 else x
            y = y % Ly if Ly > 0 else y
            z = z % Lz if Lz > 0 else z
        elif self.config.boundary_condition == 'reflective':
            # Reflect at boundaries
            x = self._reflect_coordinate(x, self.config.domain_size_m[0])
            y = self._reflect_coordinate(y, self.config.domain_size_m[1])
            z = self._reflect_coordinate(z, self.config.domain_size_m[2])

        # Start with base value (DC component)
        result = self.config.base_value

        # Sum Fourier components
        for coef in self.config.coefficients:
            phase_arg = (
                coef.n * self._kx_factor * x +
                coef.m * self._ky_factor * y +
                coef.l * self._kz_factor * z +
                coef.phase
            )
            result += coef.amplitude * math.cos(phase_arg)

        return result

    def _reflect_coordinate(self, coord: float, length: float) -> float:
        """Apply reflective boundary condition."""
        if length <= 0:
            return coord
        # Number of full periods
        periods = int(coord / length)
        local = coord - periods * length
        # Reflect odd periods
        if periods % 2 == 1:
            local = length - local
        return local

    def evaluate_gradient(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Evaluate the spatial gradient of the field.

        Returns:
            Tuple of (dP/dx, dP/dy, dP/dz)
        """
        grad_x = 0.0
        grad_y = 0.0
        grad_z = 0.0

        for coef in self.config.coefficients:
            phase_arg = (
                coef.n * self._kx_factor * x +
                coef.m * self._ky_factor * y +
                coef.l * self._kz_factor * z +
                coef.phase
            )
            sin_term = -coef.amplitude * math.sin(phase_arg)
            grad_x += sin_term * coef.n * self._kx_factor
            grad_y += sin_term * coef.m * self._ky_factor
            grad_z += sin_term * coef.l * self._kz_factor

        return (grad_x, grad_y, grad_z)

    def get_power_spectrum(self) -> Dict[Tuple[int, int, int], float]:
        """
        Get power spectrum (amplitude squared) for each mode.

        Returns:
            Dictionary mapping (n, m, l) to amplitude^2
        """
        spectrum = {}
        for coef in self.config.coefficients:
            key = (coef.n, coef.m, coef.l)
            spectrum[key] = coef.amplitude ** 2
        return spectrum

    @property
    def property_name(self) -> str:
        """Get the property name this field represents."""
        return self.config.property_name

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.config.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'FourierFieldEvaluator':
        """Create evaluator from JSON string."""
        data = json.loads(json_str)
        return cls(FourierFieldConfig.from_dict(data))


def create_layered_composite_field(
    property_name: str,
    layer_values: List[float],
    layer_thicknesses_m: List[float],
    num_harmonics: int = 10
) -> FourierFieldConfig:
    """
    Create Fourier representation of a layered composite.

    Creates a field that varies in z-direction with periodic layers.

    Args:
        property_name: Name of property
        layer_values: Property values for each layer type
        layer_thicknesses_m: Thickness of each layer
        num_harmonics: Number of Fourier harmonics to use

    Returns:
        FourierFieldConfig for the layered structure
    """
    if len(layer_values) != len(layer_thicknesses_m):
        raise ValueError("layer_values and layer_thicknesses must have same length")

    total_thickness = sum(layer_thicknesses_m)

    # Calculate mean (DC component)
    weighted_sum = sum(v * t for v, t in zip(layer_values, layer_thicknesses_m))
    mean_value = weighted_sum / total_thickness

    # For a simple two-layer system, use square wave Fourier series
    coefficients = []
    if len(layer_values) == 2:
        delta = (layer_values[0] - layer_values[1]) / 2
        duty_cycle = layer_thicknesses_m[0] / total_thickness

        for n in range(1, num_harmonics + 1):
            # Square wave Fourier coefficient
            amplitude = (2 * delta / (n * math.pi)) * math.sin(n * math.pi * duty_cycle)
            if abs(amplitude) > 1e-10:  # Skip negligible terms
                coefficients.append(FourierCoefficient3D(
                    n=0, m=0, l=n,
                    amplitude=amplitude,
                    phase=0.0
                ))

    return FourierFieldConfig(
        property_name=property_name,
        base_value=mean_value,
        domain_size_m=(0.1, 0.1, total_thickness),
        coefficients=coefficients,
        boundary_condition='periodic'
    )


def create_grain_structure_field(
    property_name: str,
    base_value: float,
    variation_amplitude: float,
    grain_size_m: float,
    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> FourierFieldConfig:
    """
    Create Fourier representation of grain-to-grain property variation.

    Simulates polycrystalline material with periodic grain structure.

    Args:
        property_name: Name of property
        base_value: Mean property value
        variation_amplitude: Peak-to-peak variation (fraction of base)
        grain_size_m: Average grain size in meters
        anisotropy: Scaling factors for x, y, z directions

    Returns:
        FourierFieldConfig for grain structure
    """
    ax, ay, az = anisotropy
    domain = (grain_size_m * ax * 4, grain_size_m * ay * 4, grain_size_m * az * 4)

    amplitude = base_value * variation_amplitude / 2

    # Create coefficients for dominant grain modes
    coefficients = [
        # Primary modes
        FourierCoefficient3D(n=1, m=0, l=0, amplitude=amplitude * 0.5, phase=0.0),
        FourierCoefficient3D(n=0, m=1, l=0, amplitude=amplitude * 0.5, phase=0.0),
        FourierCoefficient3D(n=0, m=0, l=1, amplitude=amplitude * 0.5, phase=0.0),
        # Cross terms for texture
        FourierCoefficient3D(n=1, m=1, l=0, amplitude=amplitude * 0.25, phase=0.78),
        FourierCoefficient3D(n=1, m=0, l=1, amplitude=amplitude * 0.25, phase=1.57),
        FourierCoefficient3D(n=0, m=1, l=1, amplitude=amplitude * 0.25, phase=2.35),
    ]

    return FourierFieldConfig(
        property_name=property_name,
        base_value=base_value,
        domain_size_m=domain,
        coefficients=coefficients,
        boundary_condition='periodic'
    )
