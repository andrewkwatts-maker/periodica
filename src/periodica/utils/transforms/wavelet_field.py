"""
Wavelet Field Evaluator
=======================

Evaluates wavelet-based representations of spatially varying material properties.
Used for multi-scale variations like inclusions, defects, grain boundaries.

Wavelet transforms capture features at multiple scales simultaneously,
making them ideal for heterogeneous materials like granite, concrete, composites.

Supported Wavelet Families:
    - Haar: Simple, discontinuous (good for sharp boundaries)
    - Daubechies (db2-db20): Smooth, compact support (gradual variations)
    - Coiflet: Symmetric (oscillatory features)
    - Symlet: Near-symmetric Daubechies (general purpose)

References:
    - Wavelet-based reduced order models for microstructure (Springer)
    - Upscaling permeability by multiscale wavelet transformations
    - PyWavelets documentation (pywavelets.readthedocs.io)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import json

# Try to import PyWavelets, provide fallback
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class WaveletFamily(Enum):
    """Supported wavelet families."""
    HAAR = "haar"
    DB2 = "db2"
    DB4 = "db4"
    DB8 = "db8"
    COIF1 = "coif1"
    COIF3 = "coif3"
    SYM4 = "sym4"
    SYM8 = "sym8"
    MEXICAN_HAT = "mexh"  # Continuous wavelet


@dataclass
class WaveletConfig:
    """
    Configuration for wavelet-represented spatial field.

    Attributes:
        property_name: Name of the property being represented
        wavelet_family: Wavelet type to use
        decomposition_level: Number of decomposition levels (scales)
        domain_size_m: Size of domain [Lx, Ly, Lz] in meters
        grid_resolution: Number of grid points in each direction
    """
    property_name: str
    wavelet_family: WaveletFamily = WaveletFamily.DB4
    decomposition_level: int = 3
    domain_size_m: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    grid_resolution: Tuple[int, int, int] = (32, 32, 32)
    base_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'property': self.property_name,
            'wavelet_family': self.wavelet_family.value,
            'decomposition_level': self.decomposition_level,
            'domain_size_m': list(self.domain_size_m),
            'grid_resolution': list(self.grid_resolution),
            'base_value': self.base_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaveletConfig':
        return cls(
            property_name=data.get('property', ''),
            wavelet_family=WaveletFamily(data.get('wavelet_family', 'db4')),
            decomposition_level=data.get('decomposition_level', 3),
            domain_size_m=tuple(data.get('domain_size_m', [0.05, 0.05, 0.05])),
            grid_resolution=tuple(data.get('grid_resolution', [32, 32, 32])),
            base_value=data.get('base_value', 0.0)
        )


@dataclass
class WaveletCoefficients3D:
    """
    3D wavelet decomposition coefficients.

    For 3D discrete wavelet transform (DWT), each level produces 8 sub-bands:
    - LLL (approximation)
    - LLH, LHL, LHH, HLL, HLH, HHL, HHH (detail coefficients)

    L = Low frequency, H = High frequency

    Attributes:
        approximation: Coarse-scale approximation coefficients (3D array as nested list)
        detail_coefficients: Dict mapping level to dict of sub-band coefficients
    """
    approximation: List[List[List[float]]] = field(default_factory=list)
    detail_coefficients: Dict[int, Dict[str, List[List[List[float]]]]] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'approximation': self.approximation,
            'detail_levels': [
                {
                    'level': level,
                    **{k: v for k, v in coeffs.items()}
                }
                for level, coeffs in sorted(self.detail_coefficients.items())
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaveletCoefficients3D':
        detail_coeffs = {}
        for level_data in data.get('detail_levels', []):
            level = level_data.get('level', 0)
            detail_coeffs[level] = {
                k: v for k, v in level_data.items() if k != 'level'
            }
        return cls(
            approximation=data.get('approximation', []),
            detail_coefficients=detail_coeffs
        )


class WaveletFieldEvaluator:
    """
    Evaluates wavelet decomposition for spatial property fields.

    Reconstructs property values at arbitrary points by:
    1. Storing wavelet coefficients in JSON
    2. Reconstructing field on regular grid
    3. Interpolating to query point

    For efficiency, the full reconstruction is cached and only
    recalculated when coefficients change.

    Example:
        config = WaveletConfig(
            property_name='Density_kg_m3',
            wavelet_family=WaveletFamily.DB4,
            decomposition_level=3,
            domain_size_m=(0.01, 0.01, 0.01)
        )
        evaluator = WaveletFieldEvaluator(config, coefficients)
        value = evaluator.evaluate(0.005, 0.003, 0.001)
    """

    # Sub-band labels for 3D DWT
    SUBBANDS_3D = ['LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']

    def __init__(
        self,
        config: Union[WaveletConfig, Dict[str, Any]],
        coefficients: Optional[Union[WaveletCoefficients3D, Dict[str, Any]]] = None
    ):
        if isinstance(config, dict):
            config = WaveletConfig.from_dict(config)
        self.config = config

        if coefficients is None:
            self.coefficients = WaveletCoefficients3D()
        elif isinstance(coefficients, dict):
            self.coefficients = WaveletCoefficients3D.from_dict(coefficients)
        else:
            self.coefficients = coefficients

        # Cached reconstruction
        self._reconstructed_grid: Optional[List[List[List[float]]]] = None
        self._grid_dirty = True

    def evaluate(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the field at point (x, y, z).

        Uses trilinear interpolation on reconstructed grid.

        Args:
            x, y, z: Coordinates in meters

        Returns:
            Property value at the given point
        """
        # Ensure grid is reconstructed
        if self._grid_dirty or self._reconstructed_grid is None:
            self._reconstruct_grid()

        if self._reconstructed_grid is None:
            return self.config.base_value

        # Map coordinates to grid indices
        Lx, Ly, Lz = self.config.domain_size_m
        Nx, Ny, Nz = self.config.grid_resolution

        # Handle periodic boundary
        x = x % Lx if Lx > 0 else 0
        y = y % Ly if Ly > 0 else 0
        z = z % Lz if Lz > 0 else 0

        # Fractional grid indices
        fx = (x / Lx) * (Nx - 1)
        fy = (y / Ly) * (Ny - 1)
        fz = (z / Lz) * (Nz - 1)

        # Integer indices for interpolation
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)

        # Fractional parts
        dx = fx - ix
        dy = fy - iy
        dz = fz - iz

        # Clamp to valid range
        ix = max(0, min(ix, Nx - 2))
        iy = max(0, min(iy, Ny - 2))
        iz = max(0, min(iz, Nz - 2))

        # Trilinear interpolation
        grid = self._reconstructed_grid
        try:
            c000 = grid[ix][iy][iz]
            c001 = grid[ix][iy][iz + 1]
            c010 = grid[ix][iy + 1][iz]
            c011 = grid[ix][iy + 1][iz + 1]
            c100 = grid[ix + 1][iy][iz]
            c101 = grid[ix + 1][iy][iz + 1]
            c110 = grid[ix + 1][iy + 1][iz]
            c111 = grid[ix + 1][iy + 1][iz + 1]

            # Interpolate
            c00 = c000 * (1 - dz) + c001 * dz
            c01 = c010 * (1 - dz) + c011 * dz
            c10 = c100 * (1 - dz) + c101 * dz
            c11 = c110 * (1 - dz) + c111 * dz

            c0 = c00 * (1 - dy) + c01 * dy
            c1 = c10 * (1 - dy) + c11 * dy

            return c0 * (1 - dx) + c1 * dx
        except (IndexError, TypeError):
            return self.config.base_value

    def _reconstruct_grid(self):
        """Reconstruct the field from wavelet coefficients."""
        if not HAS_PYWT:
            # Fallback: return constant grid
            Nx, Ny, Nz = self.config.grid_resolution
            self._reconstructed_grid = [
                [[self.config.base_value for _ in range(Nz)]
                 for _ in range(Ny)]
                for _ in range(Nx)
            ]
            self._grid_dirty = False
            return

        if not self.coefficients.approximation:
            # No coefficients set
            Nx, Ny, Nz = self.config.grid_resolution
            self._reconstructed_grid = [
                [[self.config.base_value for _ in range(Nz)]
                 for _ in range(Ny)]
                for _ in range(Nx)
            ]
            self._grid_dirty = False
            return

        # Convert to numpy for pywt
        import numpy as np

        wavelet = pywt.Wavelet(self.config.wavelet_family.value)

        # Build coefficient structure for pywt.waverec n
        # pywt uses [cAn, (cDn), (cDn-1), ..., (cD1)] format
        coeffs_list = [np.array(self.coefficients.approximation)]

        for level in range(self.config.decomposition_level, 0, -1):
            level_coeffs = self.coefficients.detail_coefficients.get(level, {})
            detail_tuple = {}
            for subband in self.SUBBANDS_3D:
                if subband in level_coeffs:
                    detail_tuple[subband.lower()] = np.array(level_coeffs[subband])
                else:
                    # Create zero array of appropriate size
                    approx_shape = np.array(self.coefficients.approximation).shape
                    # Detail at level l has size approx_size * 2^(L-l)
                    scale_factor = 2 ** (self.config.decomposition_level - level)
                    detail_shape = tuple(int(s * scale_factor) for s in approx_shape)
                    detail_tuple[subband.lower()] = np.zeros(detail_shape)
            coeffs_list.append(detail_tuple)

        try:
            # Perform inverse wavelet transform
            reconstructed = pywt.waverecn(coeffs_list, wavelet)
            # Resize to target grid if needed
            Nx, Ny, Nz = self.config.grid_resolution
            # Simple nearest-neighbor resize
            rx, ry, rz = reconstructed.shape
            result = [
                [[float(reconstructed[
                    int(i * rx / Nx),
                    int(j * ry / Ny),
                    int(k * rz / Nz)
                ]) for k in range(Nz)]
                 for j in range(Ny)]
                for i in range(Nx)
            ]
            # Add base value
            self._reconstructed_grid = [
                [[self.config.base_value + v for v in row]
                 for row in plane]
                for plane in result
            ]
        except Exception:
            # Fallback on any error
            self._reconstructed_grid = [
                [[self.config.base_value for _ in range(Nz)]
                 for _ in range(Ny)]
                for _ in range(Nx)
            ]

        self._grid_dirty = False

    def set_coefficients(self, coefficients: WaveletCoefficients3D):
        """Update wavelet coefficients and mark grid for reconstruction."""
        self.coefficients = coefficients
        self._grid_dirty = True

    @property
    def property_name(self) -> str:
        return self.config.property_name

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            'config': self.config.to_dict(),
            'coefficients': self.coefficients.to_dict()
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'WaveletFieldEvaluator':
        """Create from JSON string."""
        data = json.loads(json_str)
        config = WaveletConfig.from_dict(data.get('config', {}))
        coeffs = WaveletCoefficients3D.from_dict(data.get('coefficients', {}))
        return cls(config, coeffs)


def create_inclusion_field(
    property_name: str,
    base_value: float,
    inclusion_value: float,
    volume_fraction: float,
    characteristic_size_m: float,
    domain_size_m: Tuple[float, float, float] = (0.01, 0.01, 0.01),
    seed: int = 42
) -> Tuple[WaveletConfig, WaveletCoefficients3D]:
    """
    Create wavelet representation of material with random inclusions.

    Args:
        property_name: Name of property (e.g., 'Density_kg_m3')
        base_value: Property value of matrix
        inclusion_value: Property value of inclusions
        volume_fraction: Volume fraction of inclusions (0-1)
        characteristic_size_m: Typical inclusion size
        domain_size_m: Size of domain
        seed: Random seed for reproducibility

    Returns:
        Tuple of (WaveletConfig, WaveletCoefficients3D)
    """
    import random
    random.seed(seed)

    # Grid resolution based on characteristic size
    grid_res = max(16, int(domain_size_m[0] / characteristic_size_m * 4))
    grid_res = min(grid_res, 64)  # Cap for memory

    config = WaveletConfig(
        property_name=property_name,
        wavelet_family=WaveletFamily.DB4,
        decomposition_level=3,
        domain_size_m=domain_size_m,
        grid_resolution=(grid_res, grid_res, grid_res),
        base_value=base_value
    )

    # Generate random inclusion field
    delta = inclusion_value - base_value
    num_cells = grid_res ** 3

    # Approximation coefficients (coarsest level)
    approx_size = grid_res // (2 ** config.decomposition_level)
    approx_size = max(2, approx_size)

    approx = [
        [[delta * volume_fraction * random.gauss(1.0, 0.3)
          for _ in range(approx_size)]
         for _ in range(approx_size)]
        for _ in range(approx_size)
    ]

    # Detail coefficients (add randomness at each scale)
    detail_coeffs = {}
    for level in range(1, config.decomposition_level + 1):
        size = approx_size * (2 ** (config.decomposition_level - level + 1))
        size = min(size, grid_res)

        level_details = {}
        for subband in WaveletFieldEvaluator.SUBBANDS_3D:
            # Higher frequency = smaller amplitude variation
            scale = 0.5 ** level
            level_details[subband] = [
                [[delta * volume_fraction * scale * random.gauss(0.0, 0.2)
                  for _ in range(size)]
                 for _ in range(size)]
                for _ in range(size)
            ]
        detail_coeffs[level] = level_details

    coefficients = WaveletCoefficients3D(
        approximation=approx,
        detail_coefficients=detail_coeffs
    )

    return config, coefficients


def create_gradient_field(
    property_name: str,
    surface_value: float,
    core_value: float,
    transition_depth_m: float,
    direction: str = 'z',
    domain_size_m: Tuple[float, float, float] = (0.01, 0.01, 0.01)
) -> Tuple[WaveletConfig, WaveletCoefficients3D]:
    """
    Create wavelet representation of gradient material (like case hardened steel).

    Args:
        property_name: Name of property
        surface_value: Property value at surface
        core_value: Property value in core
        transition_depth_m: Depth of transition zone
        direction: 'x', 'y', or 'z' for gradient direction
        domain_size_m: Size of domain

    Returns:
        Tuple of (WaveletConfig, WaveletCoefficients3D)
    """
    grid_res = 32
    config = WaveletConfig(
        property_name=property_name,
        wavelet_family=WaveletFamily.DB4,
        decomposition_level=3,
        domain_size_m=domain_size_m,
        grid_resolution=(grid_res, grid_res, grid_res),
        base_value=core_value
    )

    # Create gradient in specified direction using error function profile
    approx_size = grid_res // (2 ** config.decomposition_level)
    delta = surface_value - core_value

    # Direction index
    dir_idx = {'x': 0, 'y': 1, 'z': 2}.get(direction, 2)

    def error_function_profile(depth_fraction: float) -> float:
        """Approximate error function for case depth profile."""
        x = (depth_fraction * domain_size_m[dir_idx] - transition_depth_m) / (transition_depth_m * 0.5)
        # Sigmoid approximation of erf
        return delta * 0.5 * (1 - math.tanh(x))

    approx = [[[0.0 for _ in range(approx_size)]
               for _ in range(approx_size)]
              for _ in range(approx_size)]

    for i in range(approx_size):
        for j in range(approx_size):
            for k in range(approx_size):
                if dir_idx == 0:
                    depth_frac = i / (approx_size - 1)
                elif dir_idx == 1:
                    depth_frac = j / (approx_size - 1)
                else:
                    depth_frac = k / (approx_size - 1)
                approx[i][j][k] = error_function_profile(depth_frac)

    coefficients = WaveletCoefficients3D(
        approximation=approx,
        detail_coefficients={}  # Smooth gradient, no detail
    )

    return config, coefficients
