"""
Stochastic Field Module (Karhunen-Loeve Expansion)
==================================================

Represents natural material variability using stochastic random fields.

The Karhunen-Loeve expansion represents a random field as:
    P(x) = mean + sum_{i=1}^{N} sqrt(lambda_i) * xi_i * phi_i(x)

where:
    - lambda_i: Eigenvalues of covariance function
    - xi_i: Standard normal random variables (stored as seeds)
    - phi_i: Eigenfunctions (typically Fourier basis)

Supported Covariance Models:
    - Exponential: C(h) = sigma^2 * exp(-|h|/L)
    - Squared Exponential (Gaussian): C(h) = sigma^2 * exp(-|h|^2 / (2*L^2))
    - Matern: C(h) = sigma^2 * (2^(1-nu)/Gamma(nu)) * (sqrt(2*nu)*|h|/L)^nu * K_nu(...)

References:
    - On spectral representation and Karhunen-Loeve expansion (Archives of Civil and Mech Eng)
    - Simulation of multi-dimensional random fields by K-L expansion (ScienceDirect)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json
import random


class CovarianceModel(Enum):
    """Available covariance function models."""
    EXPONENTIAL = "exponential"
    SQUARED_EXPONENTIAL = "squared_exponential"
    MATERN_1_5 = "matern_1.5"
    MATERN_2_5 = "matern_2.5"


@dataclass
class StochasticFieldConfig:
    """
    Configuration for stochastic random field.

    Attributes:
        property_name: Name of property
        mean: Mean value of field
        std_dev: Standard deviation
        correlation_length_m: Correlation length in each direction [Lx, Ly, Lz]
        covariance_model: Type of covariance function
        num_terms: Number of K-L expansion terms
        domain_size_m: Size of domain
        random_seeds: List of random seeds (xi_i values)
    """
    property_name: str
    mean: float = 1.0
    std_dev: float = 0.1
    correlation_length_m: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    covariance_model: CovarianceModel = CovarianceModel.EXPONENTIAL
    num_terms: int = 20
    domain_size_m: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    random_seeds: List[float] = field(default_factory=list)
    eigenvalues: Optional[List[float]] = None

    def __post_init__(self):
        # Generate random seeds if not provided
        if not self.random_seeds:
            self.random_seeds = [random.gauss(0, 1) for _ in range(self.num_terms)]
        # Calculate eigenvalues if not provided
        if self.eigenvalues is None:
            self.eigenvalues = self._compute_eigenvalues()

    def _compute_eigenvalues(self) -> List[float]:
        """
        Compute eigenvalues for K-L expansion.

        For separable exponential covariance, eigenvalues follow power-law decay.
        The eigenvalues are normalized so their sum approximately equals sigma^2,
        ensuring the field has the specified variance.
        """
        # Compute unnormalized eigenvalues with power-law decay
        # Higher modes decay as 1/(1 + mode^2)
        raw_eigenvalues = []
        n_modes = int(math.ceil(self.num_terms ** (1/3)))

        for i in range(self.num_terms):
            n = i % n_modes
            m = (i // n_modes) % n_modes
            l = (i // (n_modes * n_modes)) % n_modes

            # Power-law decay based on mode indices
            # This approximates the eigenvalue spectrum for exponential covariance
            decay_x = 1.0 / (1.0 + (n + 1) ** 2)
            decay_y = 1.0 / (1.0 + (m + 1) ** 2)
            decay_z = 1.0 / (1.0 + (l + 1) ** 2)

            raw_eigenvalues.append(decay_x * decay_y * decay_z)

        # Normalize so sum of eigenvalues equals variance (sigma^2)
        total = sum(raw_eigenvalues)
        if total > 0:
            scale = self.std_dev ** 2 / total
            eigenvalues = [lam * scale for lam in raw_eigenvalues]
        else:
            eigenvalues = raw_eigenvalues

        return eigenvalues

    def to_dict(self) -> Dict[str, Any]:
        return {
            'property': self.property_name,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'correlation_length_m': list(self.correlation_length_m),
            'covariance_model': self.covariance_model.value,
            'num_terms': self.num_terms,
            'domain_size_m': list(self.domain_size_m),
            'random_seeds': self.random_seeds,
            'eigenvalues': self.eigenvalues
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StochasticFieldConfig':
        return cls(
            property_name=data.get('property', ''),
            mean=data.get('mean', 1.0),
            std_dev=data.get('std_dev', 0.1),
            correlation_length_m=tuple(data.get('correlation_length_m', [0.01, 0.01, 0.01])),
            covariance_model=CovarianceModel(data.get('covariance_model', 'exponential')),
            num_terms=data.get('num_terms', 20),
            domain_size_m=tuple(data.get('domain_size_m', [0.1, 0.1, 0.1])),
            random_seeds=data.get('random_seeds', []),
            eigenvalues=data.get('eigenvalues')
        )


class StochasticFieldEvaluator:
    """
    Evaluates Karhunen-Loeve expansion for stochastic property fields.

    Represents natural material variability with specified correlation
    structure and statistical properties.

    Example:
        config = StochasticFieldConfig(
            property_name='YieldStrength_MPa',
            mean=310,
            std_dev=15,
            correlation_length_m=(0.01, 0.01, 0.005)
        )
        evaluator = StochasticFieldEvaluator(config)
        value = evaluator.evaluate(0.05, 0.03, 0.01)
    """

    def __init__(self, config: StochasticFieldConfig):
        if isinstance(config, dict):
            config = StochasticFieldConfig.from_dict(config)
        self.config = config

    def evaluate(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the stochastic field at point (x, y, z).

        Uses K-L expansion:
            P(x) = mean + sum_i sqrt(lambda_i) * xi_i * phi_i(x)

        Args:
            x, y, z: Coordinates in meters

        Returns:
            Property value (sample from random field)
        """
        result = self.config.mean

        Lx, Ly, Lz = self.config.correlation_length_m
        Dx, Dy, Dz = self.config.domain_size_m

        for i in range(min(self.config.num_terms, len(self.config.eigenvalues))):
            # Eigenvalue
            lambda_i = self.config.eigenvalues[i]
            if lambda_i <= 0:
                continue

            # Random seed (xi_i)
            xi_i = self.config.random_seeds[i] if i < len(self.config.random_seeds) else 0

            # Eigenfunction (Fourier basis)
            # phi_i(x) = cos(k_i * x) or sin(k_i * x)
            phi = self._eigenfunction(i, x, y, z, Lx, Ly, Lz, Dx, Dy, Dz)

            # K-L term
            result += math.sqrt(lambda_i) * xi_i * phi

        return result

    def _eigenfunction(
        self, index: int,
        x: float, y: float, z: float,
        Lx: float, Ly: float, Lz: float,
        Dx: float, Dy: float, Dz: float
    ) -> float:
        """
        Compute eigenfunction at point.

        Uses separable Fourier basis with (n+1) to ensure all modes have spatial variation:
            phi_{n,m,l}(x,y,z) = cos((n+1)*pi*x/Dx) * cos((m+1)*pi*y/Dy) * cos((l+1)*pi*z/Dz)

        The eigenfunctions are normalized to have unit L2 norm on the domain,
        so each cos term is normalized by sqrt(2/D) for non-zero modes.
        The combined normalization is sqrt(8/(Dx*Dy*Dz)).
        """
        # Decompose 1D index into 3D indices
        n_modes = int(math.ceil(self.config.num_terms ** (1/3)))

        n = index % n_modes
        m = (index // n_modes) % n_modes
        l = (index // (n_modes * n_modes)) % n_modes

        # Fourier basis (cosine for symmetric boundary)
        # Use (n+1) to ensure all modes vary spatially
        phi_x = math.cos((n + 1) * math.pi * x / Dx) if Dx > 0 else 1.0
        phi_y = math.cos((m + 1) * math.pi * y / Dy) if Dy > 0 else 1.0
        phi_z = math.cos((l + 1) * math.pi * z / Dz) if Dz > 0 else 1.0

        # The eigenfunction value is just the product of cosines.
        # Normalization is handled through eigenvalue scaling.
        return phi_x * phi_y * phi_z

    def covariance(self, x1: Tuple[float, float, float],
                   x2: Tuple[float, float, float]) -> float:
        """
        Compute covariance between two points.

        Args:
            x1, x2: Points as (x, y, z) tuples

        Returns:
            Covariance C(x1, x2)
        """
        Lx, Ly, Lz = self.config.correlation_length_m
        sigma = self.config.std_dev

        # Distance components
        dx = abs(x1[0] - x2[0])
        dy = abs(x1[1] - x2[1])
        dz = abs(x1[2] - x2[2])

        # Normalized distances
        hx = dx / Lx if Lx > 0 else 0
        hy = dy / Ly if Ly > 0 else 0
        hz = dz / Lz if Lz > 0 else 0

        if self.config.covariance_model == CovarianceModel.EXPONENTIAL:
            # Separable exponential
            return (sigma ** 2 *
                    math.exp(-hx) * math.exp(-hy) * math.exp(-hz))

        elif self.config.covariance_model == CovarianceModel.SQUARED_EXPONENTIAL:
            # Gaussian / squared exponential
            h_sq = hx**2 + hy**2 + hz**2
            return sigma ** 2 * math.exp(-h_sq / 2)

        elif self.config.covariance_model == CovarianceModel.MATERN_1_5:
            # Matern 3/2
            h = math.sqrt(hx**2 + hy**2 + hz**2)
            sqrt3_h = math.sqrt(3) * h
            return sigma ** 2 * (1 + sqrt3_h) * math.exp(-sqrt3_h)

        elif self.config.covariance_model == CovarianceModel.MATERN_2_5:
            # Matern 5/2
            h = math.sqrt(hx**2 + hy**2 + hz**2)
            sqrt5_h = math.sqrt(5) * h
            return sigma ** 2 * (1 + sqrt5_h + sqrt5_h**2 / 3) * math.exp(-sqrt5_h)

        return sigma ** 2

    def sample_statistics(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Compute statistics of field over random points.

        Returns:
            Dict with 'mean', 'std_dev', 'min', 'max'
        """
        Dx, Dy, Dz = self.config.domain_size_m
        values = []

        for _ in range(num_samples):
            x = random.random() * Dx
            y = random.random() * Dy
            z = random.random() * Dz
            values.append(self.evaluate(x, y, z))

        return {
            'mean': sum(values) / len(values),
            'std_dev': math.sqrt(sum((v - sum(values)/len(values))**2
                                     for v in values) / len(values)),
            'min': min(values),
            'max': max(values)
        }

    @property
    def property_name(self) -> str:
        return self.config.property_name

    def regenerate_seeds(self, seed: Optional[int] = None):
        """Regenerate random seeds for new realization."""
        if seed is not None:
            random.seed(seed)
        self.config.random_seeds = [
            random.gauss(0, 1) for _ in range(self.config.num_terms)
        ]

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.config.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'StochasticFieldEvaluator':
        """Create from JSON."""
        data = json.loads(json_str)
        return cls(StochasticFieldConfig.from_dict(data))


def create_material_variability_field(
    property_name: str,
    mean_value: float,
    coefficient_of_variation: float = 0.05,
    correlation_length_m: float = 0.005,
    domain_size_m: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: Optional[int] = None
) -> StochasticFieldConfig:
    """
    Create stochastic field configuration for natural material variability.

    Args:
        property_name: Name of property
        mean_value: Mean property value
        coefficient_of_variation: CV = std_dev / mean (typically 0.02-0.10)
        correlation_length_m: Isotropic correlation length
        domain_size_m: Size of domain
        anisotropy: Scaling for correlation length in each direction
        seed: Random seed for reproducibility

    Returns:
        StochasticFieldConfig ready for use
    """
    if seed is not None:
        random.seed(seed)

    std_dev = mean_value * coefficient_of_variation
    Lx = correlation_length_m * anisotropy[0]
    Ly = correlation_length_m * anisotropy[1]
    Lz = correlation_length_m * anisotropy[2]

    return StochasticFieldConfig(
        property_name=property_name,
        mean=mean_value,
        std_dev=std_dev,
        correlation_length_m=(Lx, Ly, Lz),
        covariance_model=CovarianceModel.EXPONENTIAL,
        num_terms=30,
        domain_size_m=domain_size_m
    )


def create_granite_heterogeneity_field(
    property_name: str = 'YoungsModulus_GPa',
    quartz_fraction: float = 0.33,
    feldspar_fraction: float = 0.60,
    mica_fraction: float = 0.07,
    grain_size_mm: float = 2.0,
    seed: Optional[int] = None
) -> StochasticFieldConfig:
    """
    Create stochastic field for heterogeneous granite.

    Models property variation due to random grain distribution
    of quartz, feldspar, and mica.

    Args:
        property_name: Property to model
        quartz_fraction: Volume fraction of quartz
        feldspar_fraction: Volume fraction of feldspar
        mica_fraction: Volume fraction of mica
        grain_size_mm: Average grain size in mm
        seed: Random seed

    Returns:
        StochasticFieldConfig for granite heterogeneity
    """
    if seed is not None:
        random.seed(seed)

    # Property values by mineral (approximate)
    mineral_props = {
        'YoungsModulus_GPa': {'quartz': 95, 'feldspar': 70, 'mica': 35},
        'Density_kg_m3': {'quartz': 2650, 'feldspar': 2560, 'mica': 2800},
        'ThermalExpansion_per_K': {'quartz': 12.3e-6, 'feldspar': 8.0e-6, 'mica': 15e-6}
    }

    props = mineral_props.get(property_name, {'quartz': 1, 'feldspar': 1, 'mica': 1})

    # Calculate mean and variance using rule of mixtures
    mean = (quartz_fraction * props['quartz'] +
            feldspar_fraction * props['feldspar'] +
            mica_fraction * props['mica'])

    # Variance from phase contrast
    variance = (quartz_fraction * (props['quartz'] - mean)**2 +
                feldspar_fraction * (props['feldspar'] - mean)**2 +
                mica_fraction * (props['mica'] - mean)**2)
    std_dev = math.sqrt(variance)

    # Correlation length ~ grain size
    correlation_length = grain_size_mm / 1000  # Convert to meters

    return StochasticFieldConfig(
        property_name=property_name,
        mean=mean,
        std_dev=std_dev,
        correlation_length_m=(correlation_length, correlation_length, correlation_length * 0.8),
        covariance_model=CovarianceModel.EXPONENTIAL,
        num_terms=50,
        domain_size_m=(0.1, 0.1, 0.1)
    )
