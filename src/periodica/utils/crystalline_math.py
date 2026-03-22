"""
Crystalline Mathematics Module
Provides mathematical functions for modeling crystalline structures, grain boundaries,
phase distributions, and lattice properties in alloys.

This module includes:
- Noise functions (Perlin, Simplex, Worley, fBm) for phase distribution
- Voronoi mathematics for grain structure simulation
- Lattice functions for crystal unit cell calculations
- Visualization functions for structure rendering
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib


# ==================== Constants ====================

class CrystalStructure(Enum):
    """Common crystal structure types"""
    FCC = "FCC"  # Face-Centered Cubic
    BCC = "BCC"  # Body-Centered Cubic
    HCP = "HCP"  # Hexagonal Close-Packed
    BCT = "BCT"  # Body-Centered Tetragonal
    ORTHORHOMBIC = "Orthorhombic"
    MONOCLINIC = "Monoclinic"
    TRICLINIC = "Triclinic"
    HEXAGONAL = "Hexagonal"
    TETRAGONAL = "Tetragonal"
    CUBIC = "Cubic"


# Atomic packing factors for different structures
PACKING_FACTORS = {
    CrystalStructure.FCC: 0.74,
    CrystalStructure.BCC: 0.68,
    CrystalStructure.HCP: 0.74,
    CrystalStructure.BCT: 0.70,
}

# Coordination numbers for different structures
COORDINATION_NUMBERS = {
    CrystalStructure.FCC: 12,
    CrystalStructure.BCC: 8,
    CrystalStructure.HCP: 12,
    CrystalStructure.BCT: 8,
}


# ==================== Vector Math Utilities ====================

@dataclass
class Vec2:
    """2D Vector class"""
    x: float
    y: float

    def __add__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x * scalar, self.y * scalar)

    def dot(self, other: 'Vec2') -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> 'Vec2':
        length = self.length()
        if length == 0:
            return Vec2(0, 0)
        return Vec2(self.x / length, self.y / length)


@dataclass
class Vec3:
    """3D Vector class"""
    x: float
    y: float
    z: float

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'Vec3':
        length = self.length()
        if length == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / length, self.y / length, self.z / length)


# ==================== Noise Functions ====================

class PerlinNoise:
    """
    Perlin noise generator for smooth, continuous noise patterns.
    Used for modeling phase distribution variations in alloys.
    """

    def __init__(self, seed: int = 0):
        """Initialize with a seed for reproducible results."""
        self.seed = seed
        self.permutation = self._generate_permutation()

    def _generate_permutation(self) -> List[int]:
        """Generate permutation table for noise generation."""
        random.seed(self.seed)
        perm = list(range(256))
        random.shuffle(perm)
        return perm + perm  # Duplicate for wrapping

    def _fade(self, t: float) -> float:
        """Fade function for smooth interpolation: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)

    def _grad2d(self, hash_val: int, x: float, y: float) -> float:
        """Calculate gradient for 2D noise."""
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y

    def _grad3d(self, hash_val: int, x: float, y: float, z: float) -> float:
        """Calculate gradient for 3D noise."""
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise2d(self, x: float, y: float) -> float:
        """
        Generate 2D Perlin noise at coordinates (x, y).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Find unit square containing point
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255

        # Relative position within unit square
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        # Compute fade curves
        u = self._fade(xf)
        v = self._fade(yf)

        # Hash coordinates of square corners
        p = self.permutation
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        # Blend results from 4 corners
        x1 = self._lerp(self._grad2d(aa, xf, yf), self._grad2d(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad2d(ab, xf, yf - 1), self._grad2d(bb, xf - 1, yf - 1), u)

        return self._lerp(x1, x2, v)

    def noise3d(self, x: float, y: float, z: float) -> float:
        """
        Generate 3D Perlin noise at coordinates (x, y, z).

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Find unit cube containing point
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        zi = int(math.floor(z)) & 255

        # Relative position within unit cube
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        zf = z - math.floor(z)

        # Compute fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        # Hash coordinates of cube corners
        p = self.permutation
        aaa = p[p[p[xi] + yi] + zi]
        aba = p[p[p[xi] + yi + 1] + zi]
        aab = p[p[p[xi] + yi] + zi + 1]
        abb = p[p[p[xi] + yi + 1] + zi + 1]
        baa = p[p[p[xi + 1] + yi] + zi]
        bba = p[p[p[xi + 1] + yi + 1] + zi]
        bab = p[p[p[xi + 1] + yi] + zi + 1]
        bbb = p[p[p[xi + 1] + yi + 1] + zi + 1]

        # Blend results from 8 corners
        x1 = self._lerp(
            self._grad3d(aaa, xf, yf, zf),
            self._grad3d(baa, xf - 1, yf, zf),
            u
        )
        x2 = self._lerp(
            self._grad3d(aba, xf, yf - 1, zf),
            self._grad3d(bba, xf - 1, yf - 1, zf),
            u
        )
        y1 = self._lerp(x1, x2, v)

        x1 = self._lerp(
            self._grad3d(aab, xf, yf, zf - 1),
            self._grad3d(bab, xf - 1, yf, zf - 1),
            u
        )
        x2 = self._lerp(
            self._grad3d(abb, xf, yf - 1, zf - 1),
            self._grad3d(bbb, xf - 1, yf - 1, zf - 1),
            u
        )
        y2 = self._lerp(x1, x2, v)

        return self._lerp(y1, y2, w)


class SimplexNoise:
    """
    Simplex noise generator - more efficient than Perlin for higher dimensions.
    Produces smoother, less grid-aligned noise patterns.
    """

    # Simplex skewing factors
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0

    # Gradient vectors for 2D
    GRAD2 = [
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (1, 0), (-1, 0), (0, 1), (0, -1)
    ]

    # Gradient vectors for 3D
    GRAD3 = [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)
    ]

    def __init__(self, seed: int = 0):
        """Initialize with a seed for reproducible results."""
        self.seed = seed
        self.permutation = self._generate_permutation()

    def _generate_permutation(self) -> List[int]:
        """Generate permutation table."""
        random.seed(self.seed)
        perm = list(range(256))
        random.shuffle(perm)
        return perm + perm

    def _dot2(self, g: Tuple[int, int], x: float, y: float) -> float:
        """2D dot product with gradient."""
        return g[0] * x + g[1] * y

    def _dot3(self, g: Tuple[int, int, int], x: float, y: float, z: float) -> float:
        """3D dot product with gradient."""
        return g[0] * x + g[1] * y + g[2] * z

    def noise2d(self, x: float, y: float) -> float:
        """
        Generate 2D simplex noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Skew input space to determine simplex cell
        s = (x + y) * self.F2
        i = int(math.floor(x + s))
        j = int(math.floor(y + s))

        # Unskew back to (x, y) space
        t = (i + j) * self.G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0

        # Determine which simplex we're in
        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        # Offsets for corners
        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2

        # Hash coordinates
        ii = i & 255
        jj = j & 255
        p = self.permutation

        gi0 = p[ii + p[jj]] % 8
        gi1 = p[ii + i1 + p[jj + j1]] % 8
        gi2 = p[ii + 1 + p[jj + 1]] % 8

        # Calculate contributions from three corners
        n0, n1, n2 = 0.0, 0.0, 0.0

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * self._dot2(self.GRAD2[gi0], x0, y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * self._dot2(self.GRAD2[gi1], x1, y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * self._dot2(self.GRAD2[gi2], x2, y2)

        # Scale to [-1, 1]
        return 70.0 * (n0 + n1 + n2)

    def noise3d(self, x: float, y: float, z: float) -> float:
        """
        Generate 3D simplex noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Skew input space
        s = (x + y + z) * self.F3
        i = int(math.floor(x + s))
        j = int(math.floor(y + s))
        k = int(math.floor(z + s))

        # Unskew back
        t = (i + j + k) * self.G3
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0

        # Determine which simplex
        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
            elif x0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

        # Offsets for corners
        x1 = x0 - i1 + self.G3
        y1 = y0 - j1 + self.G3
        z1 = z0 - k1 + self.G3
        x2 = x0 - i2 + 2.0 * self.G3
        y2 = y0 - j2 + 2.0 * self.G3
        z2 = z0 - k2 + 2.0 * self.G3
        x3 = x0 - 1.0 + 3.0 * self.G3
        y3 = y0 - 1.0 + 3.0 * self.G3
        z3 = z0 - 1.0 + 3.0 * self.G3

        # Hash coordinates
        ii = i & 255
        jj = j & 255
        kk = k & 255
        p = self.permutation

        gi0 = p[ii + p[jj + p[kk]]] % 12
        gi1 = p[ii + i1 + p[jj + j1 + p[kk + k1]]] % 12
        gi2 = p[ii + i2 + p[jj + j2 + p[kk + k2]]] % 12
        gi3 = p[ii + 1 + p[jj + 1 + p[kk + 1]]] % 12

        # Calculate contributions from four corners
        n0 = n1 = n2 = n3 = 0.0

        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * self._dot3(self.GRAD3[gi0], x0, y0, z0)

        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * self._dot3(self.GRAD3[gi1], x1, y1, z1)

        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * self._dot3(self.GRAD3[gi2], x2, y2, z2)

        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        if t3 >= 0:
            t3 *= t3
            n3 = t3 * t3 * self._dot3(self.GRAD3[gi3], x3, y3, z3)

        return 32.0 * (n0 + n1 + n2 + n3)


class WorleyNoise:
    """
    Worley (Cellular) noise generator.
    Creates patterns based on distance to randomly distributed feature points.
    Excellent for modeling grain structures and cellular microstructures.
    """

    def __init__(self, seed: int = 0, point_density: float = 1.0):
        """
        Initialize Worley noise generator.

        Args:
            seed: Random seed for reproducibility
            point_density: Average number of points per unit cell
        """
        self.seed = seed
        self.point_density = point_density
        self._cache: Dict[Tuple[int, int], List[Vec2]] = {}
        self._cache_3d: Dict[Tuple[int, int, int], List[Vec3]] = {}

    def _hash(self, *args) -> int:
        """Generate hash from coordinates."""
        data = str(args) + str(self.seed)
        return int(hashlib.md5(data.encode()).hexdigest(), 16)

    def _get_cell_points_2d(self, cell_x: int, cell_y: int) -> List[Vec2]:
        """Get or generate feature points for a 2D cell."""
        key = (cell_x, cell_y)
        if key not in self._cache:
            random.seed(self._hash(cell_x, cell_y))
            num_points = max(1, int(random.gauss(self.point_density, 0.5)))
            points = []
            for _ in range(num_points):
                points.append(Vec2(
                    cell_x + random.random(),
                    cell_y + random.random()
                ))
            self._cache[key] = points
        return self._cache[key]

    def _get_cell_points_3d(self, cell_x: int, cell_y: int, cell_z: int) -> List[Vec3]:
        """Get or generate feature points for a 3D cell."""
        key = (cell_x, cell_y, cell_z)
        if key not in self._cache_3d:
            random.seed(self._hash(cell_x, cell_y, cell_z))
            num_points = max(1, int(random.gauss(self.point_density, 0.5)))
            points = []
            for _ in range(num_points):
                points.append(Vec3(
                    cell_x + random.random(),
                    cell_y + random.random(),
                    cell_z + random.random()
                ))
            self._cache_3d[key] = points
        return self._cache_3d[key]

    def noise2d(self, x: float, y: float, distance_func: str = "euclidean") -> Tuple[float, int]:
        """
        Generate 2D Worley noise.

        Args:
            x: X coordinate
            y: Y coordinate
            distance_func: Distance function ("euclidean", "manhattan", "chebyshev")

        Returns:
            Tuple of (distance to nearest point, cell ID)
        """
        cell_x = int(math.floor(x))
        cell_y = int(math.floor(y))
        point = Vec2(x, y)

        min_dist = float('inf')
        nearest_cell_id = 0

        # Check 3x3 neighborhood
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx, cy = cell_x + dx, cell_y + dy
                for fp in self._get_cell_points_2d(cx, cy):
                    if distance_func == "euclidean":
                        dist = (point - fp).length()
                    elif distance_func == "manhattan":
                        diff = point - fp
                        dist = abs(diff.x) + abs(diff.y)
                    elif distance_func == "chebyshev":
                        diff = point - fp
                        dist = max(abs(diff.x), abs(diff.y))
                    else:
                        dist = (point - fp).length()

                    if dist < min_dist:
                        min_dist = dist
                        nearest_cell_id = self._hash(cx, cy)

        return min_dist, nearest_cell_id

    def noise3d(self, x: float, y: float, z: float, distance_func: str = "euclidean") -> Tuple[float, int]:
        """
        Generate 3D Worley noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            distance_func: Distance function

        Returns:
            Tuple of (distance to nearest point, cell ID)
        """
        cell_x = int(math.floor(x))
        cell_y = int(math.floor(y))
        cell_z = int(math.floor(z))
        point = Vec3(x, y, z)

        min_dist = float('inf')
        nearest_cell_id = 0

        # Check 3x3x3 neighborhood
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cx, cy, cz = cell_x + dx, cell_y + dy, cell_z + dz
                    for fp in self._get_cell_points_3d(cx, cy, cz):
                        if distance_func == "euclidean":
                            dist = (point - fp).length()
                        elif distance_func == "manhattan":
                            diff = point - fp
                            dist = abs(diff.x) + abs(diff.y) + abs(diff.z)
                        elif distance_func == "chebyshev":
                            diff = point - fp
                            dist = max(abs(diff.x), abs(diff.y), abs(diff.z))
                        else:
                            dist = (point - fp).length()

                        if dist < min_dist:
                            min_dist = dist
                            nearest_cell_id = self._hash(cx, cy, cz)

        return min_dist, nearest_cell_id


class FractalBrownianMotion:
    """
    Fractal Brownian Motion (fBm) - layered noise for natural-looking patterns.
    Combines multiple octaves of noise at different frequencies and amplitudes.
    """

    def __init__(self, noise_generator: Any, octaves: int = 4,
                 persistence: float = 0.5, lacunarity: float = 2.0):
        """
        Initialize fBm generator.

        Args:
            noise_generator: Base noise generator (Perlin or Simplex)
            octaves: Number of noise layers
            persistence: Amplitude multiplier per octave
            lacunarity: Frequency multiplier per octave
        """
        self.noise = noise_generator
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

    def noise2d(self, x: float, y: float) -> float:
        """
        Generate 2D fBm noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range approximately [-1, 1]
        """
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(self.octaves):
            total += self.noise.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return total / max_value

    def noise3d(self, x: float, y: float, z: float) -> float:
        """
        Generate 3D fBm noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range approximately [-1, 1]
        """
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(self.octaves):
            total += self.noise.noise3d(
                x * frequency, y * frequency, z * frequency
            ) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return total / max_value


# ==================== Voronoi Mathematics ====================

@dataclass
class GrainCenter:
    """Represents a grain center point in a microstructure."""
    position: Vec3
    grain_id: int
    orientation: Tuple[float, float, float]  # Euler angles (phi1, phi, phi2)
    phase_id: int = 0


class VoronoiTessellation:
    """
    Voronoi tessellation for modeling grain structure in polycrystalline materials.
    """

    def __init__(self, seed: int = 0):
        """Initialize the tessellation generator."""
        self.seed = seed
        self.grain_centers: List[GrainCenter] = []

    def generate_grain_centers_2d(self, width: float, height: float,
                                   num_grains: int,
                                   distribution: str = "random") -> List[GrainCenter]:
        """
        Generate grain centers for 2D tessellation.

        Args:
            width: Domain width
            height: Domain height
            num_grains: Number of grains to generate
            distribution: Point distribution ("random", "poisson", "regular")

        Returns:
            List of GrainCenter objects
        """
        random.seed(self.seed)
        self.grain_centers = []

        if distribution == "random":
            for i in range(num_grains):
                pos = Vec3(
                    random.uniform(0, width),
                    random.uniform(0, height),
                    0
                )
                orientation = (
                    random.uniform(0, 360),
                    random.uniform(0, 90),
                    random.uniform(0, 90)
                )
                self.grain_centers.append(GrainCenter(
                    position=pos,
                    grain_id=i,
                    orientation=orientation,
                    phase_id=0
                ))

        elif distribution == "poisson":
            # Poisson disk sampling for more uniform distribution
            min_dist = math.sqrt((width * height) / (num_grains * math.pi))
            self.grain_centers = self._poisson_disk_sampling_2d(
                width, height, min_dist, num_grains
            )

        elif distribution == "regular":
            # Regular grid with perturbation
            cols = int(math.sqrt(num_grains * width / height))
            rows = int(num_grains / cols)
            dx = width / cols
            dy = height / rows
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if i >= num_grains:
                        break
                    # Add small random perturbation
                    px = (col + 0.5) * dx + random.gauss(0, dx * 0.2)
                    py = (row + 0.5) * dy + random.gauss(0, dy * 0.2)
                    pos = Vec3(
                        max(0, min(width, px)),
                        max(0, min(height, py)),
                        0
                    )
                    orientation = (
                        random.uniform(0, 360),
                        random.uniform(0, 90),
                        random.uniform(0, 90)
                    )
                    self.grain_centers.append(GrainCenter(
                        position=pos,
                        grain_id=i,
                        orientation=orientation,
                        phase_id=0
                    ))
                    i += 1

        return self.grain_centers

    def _poisson_disk_sampling_2d(self, width: float, height: float,
                                   min_dist: float, max_points: int) -> List[GrainCenter]:
        """Poisson disk sampling for uniform point distribution."""
        cell_size = min_dist / math.sqrt(2)
        grid_width = int(math.ceil(width / cell_size))
        grid_height = int(math.ceil(height / cell_size))
        grid: Dict[Tuple[int, int], GrainCenter] = {}

        def grid_coords(p: Vec3) -> Tuple[int, int]:
            return (int(p.x / cell_size), int(p.y / cell_size))

        def is_valid(p: Vec3) -> bool:
            if p.x < 0 or p.x >= width or p.y < 0 or p.y >= height:
                return False
            gx, gy = grid_coords(p)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        if (grid[key].position - p).length() < min_dist:
                            return False
            return True

        # Start with a random point
        first_pos = Vec3(
            random.uniform(0, width),
            random.uniform(0, height),
            0
        )
        first_grain = GrainCenter(
            position=first_pos,
            grain_id=0,
            orientation=(random.uniform(0, 360), random.uniform(0, 90), random.uniform(0, 90)),
            phase_id=0
        )
        grid[grid_coords(first_pos)] = first_grain
        active = [first_grain]
        grains = [first_grain]

        while active and len(grains) < max_points:
            idx = random.randint(0, len(active) - 1)
            center = active[idx]
            found = False

            for _ in range(30):  # Try 30 candidates
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(min_dist, 2 * min_dist)
                new_pos = Vec3(
                    center.position.x + radius * math.cos(angle),
                    center.position.y + radius * math.sin(angle),
                    0
                )

                if is_valid(new_pos):
                    new_grain = GrainCenter(
                        position=new_pos,
                        grain_id=len(grains),
                        orientation=(random.uniform(0, 360), random.uniform(0, 90), random.uniform(0, 90)),
                        phase_id=0
                    )
                    grid[grid_coords(new_pos)] = new_grain
                    active.append(new_grain)
                    grains.append(new_grain)
                    found = True
                    break

            if not found:
                active.pop(idx)

        return grains

    def generate_grain_centers_3d(self, width: float, height: float, depth: float,
                                   num_grains: int) -> List[GrainCenter]:
        """
        Generate grain centers for 3D tessellation.

        Args:
            width: Domain width
            height: Domain height
            depth: Domain depth
            num_grains: Number of grains to generate

        Returns:
            List of GrainCenter objects
        """
        random.seed(self.seed)
        self.grain_centers = []

        for i in range(num_grains):
            pos = Vec3(
                random.uniform(0, width),
                random.uniform(0, height),
                random.uniform(0, depth)
            )
            orientation = (
                random.uniform(0, 360),
                random.uniform(0, 90),
                random.uniform(0, 90)
            )
            self.grain_centers.append(GrainCenter(
                position=pos,
                grain_id=i,
                orientation=orientation,
                phase_id=0
            ))

        return self.grain_centers

    def find_nearest_grain(self, point: Vec3) -> Tuple[GrainCenter, float]:
        """
        Find the nearest grain center to a given point.

        Args:
            point: Query point

        Returns:
            Tuple of (nearest GrainCenter, distance)
        """
        min_dist = float('inf')
        nearest = None

        for grain in self.grain_centers:
            dist = (grain.position - point).length()
            if dist < min_dist:
                min_dist = dist
                nearest = grain

        return nearest, min_dist

    def is_on_boundary(self, point: Vec3, tolerance: float = 0.01) -> bool:
        """
        Check if a point is on a grain boundary.

        Args:
            point: Query point
            tolerance: Distance tolerance for boundary detection

        Returns:
            True if point is on a grain boundary
        """
        distances = []
        for grain in self.grain_centers:
            dist = (grain.position - point).length()
            distances.append((dist, grain.grain_id))

        distances.sort(key=lambda x: x[0])

        if len(distances) >= 2:
            # Point is on boundary if nearly equidistant from two grains
            return abs(distances[0][0] - distances[1][0]) < tolerance

        return False

    def get_grain_size_distribution(self, sample_resolution: int = 100) -> List[float]:
        """
        Estimate grain size distribution by sampling.

        Args:
            sample_resolution: Number of samples per dimension

        Returns:
            List of estimated grain areas
        """
        if not self.grain_centers:
            return []

        # Determine domain bounds
        min_x = min(g.position.x for g in self.grain_centers)
        max_x = max(g.position.x for g in self.grain_centers)
        min_y = min(g.position.y for g in self.grain_centers)
        max_y = max(g.position.y for g in self.grain_centers)

        width = max_x - min_x
        height = max_y - min_y

        # Count samples per grain
        grain_counts: Dict[int, int] = {g.grain_id: 0 for g in self.grain_centers}
        total_samples = sample_resolution * sample_resolution

        dx = width / sample_resolution
        dy = height / sample_resolution

        for i in range(sample_resolution):
            for j in range(sample_resolution):
                x = min_x + (i + 0.5) * dx
                y = min_y + (j + 0.5) * dy
                point = Vec3(x, y, 0)
                nearest, _ = self.find_nearest_grain(point)
                if nearest:
                    grain_counts[nearest.grain_id] += 1

        # Convert counts to areas
        total_area = width * height
        grain_areas = [
            (count / total_samples) * total_area
            for count in grain_counts.values()
        ]

        return grain_areas


# ==================== Lattice Functions ====================

@dataclass
class LatticeParameters:
    """Crystal lattice parameters."""
    a: float  # pm
    b: float  # pm
    c: float  # pm
    alpha: float  # degrees
    beta: float  # degrees
    gamma: float  # degrees


@dataclass
class AtomPosition:
    """Atom position within a unit cell."""
    element: str
    fractional_coords: Tuple[float, float, float]
    occupancy: float = 1.0


class UnitCell:
    """
    Represents a crystal unit cell with lattice parameters and atomic positions.
    """

    def __init__(self, structure: CrystalStructure, lattice_params: LatticeParameters):
        """
        Initialize a unit cell.

        Args:
            structure: Crystal structure type
            lattice_params: Lattice parameters
        """
        self.structure = structure
        self.params = lattice_params
        self.atoms: List[AtomPosition] = []
        self._setup_basis_vectors()

    def _setup_basis_vectors(self):
        """Calculate basis vectors from lattice parameters."""
        a, b, c = self.params.a, self.params.b, self.params.c
        alpha = math.radians(self.params.alpha)
        beta = math.radians(self.params.beta)
        gamma = math.radians(self.params.gamma)

        # Calculate basis vectors in Cartesian coordinates
        self.a_vec = Vec3(a, 0, 0)
        self.b_vec = Vec3(b * math.cos(gamma), b * math.sin(gamma), 0)

        cx = c * math.cos(beta)
        cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
        cz = math.sqrt(c * c - cx * cx - cy * cy)
        self.c_vec = Vec3(cx, cy, cz)

        # Calculate unit cell volume
        self.volume = abs(self.a_vec.dot(self.b_vec.cross(self.c_vec)))

    def add_atom(self, element: str, frac_x: float, frac_y: float, frac_z: float,
                 occupancy: float = 1.0):
        """Add an atom to the unit cell at fractional coordinates."""
        self.atoms.append(AtomPosition(
            element=element,
            fractional_coords=(frac_x, frac_y, frac_z),
            occupancy=occupancy
        ))

    def fractional_to_cartesian(self, frac_coords: Tuple[float, float, float]) -> Vec3:
        """Convert fractional coordinates to Cartesian coordinates."""
        fx, fy, fz = frac_coords
        return (self.a_vec * fx) + (self.b_vec * fy) + (self.c_vec * fz)

    def cartesian_to_fractional(self, cart_pos: Vec3) -> Tuple[float, float, float]:
        """Convert Cartesian coordinates to fractional coordinates."""
        # Solve the linear system
        # This is a simplified version - proper implementation would use matrix inversion
        # For orthogonal systems (90 degree angles)
        if (abs(self.params.alpha - 90) < 0.01 and
            abs(self.params.beta - 90) < 0.01 and
            abs(self.params.gamma - 90) < 0.01):
            return (
                cart_pos.x / self.params.a,
                cart_pos.y / self.params.b,
                cart_pos.z / self.params.c
            )
        # For non-orthogonal systems, use proper matrix inversion
        # Simplified approximation:
        return (
            cart_pos.x / self.a_vec.length(),
            cart_pos.y / self.b_vec.length(),
            cart_pos.z / self.c_vec.length()
        )

    def get_atom_positions_cartesian(self) -> List[Tuple[str, Vec3]]:
        """Get all atom positions in Cartesian coordinates."""
        positions = []
        for atom in self.atoms:
            cart = self.fractional_to_cartesian(atom.fractional_coords)
            positions.append((atom.element, cart))
        return positions

    def apply_periodic_boundary(self, position: Vec3) -> Vec3:
        """
        Apply periodic boundary conditions to a position.

        Args:
            position: Position in Cartesian coordinates

        Returns:
            Position wrapped to unit cell
        """
        frac = self.cartesian_to_fractional(position)
        wrapped_frac = (
            frac[0] % 1.0,
            frac[1] % 1.0,
            frac[2] % 1.0
        )
        return self.fractional_to_cartesian(wrapped_frac)


def create_fcc_unit_cell(element: str, lattice_constant: float) -> UnitCell:
    """
    Create an FCC unit cell.

    Args:
        element: Element symbol
        lattice_constant: Lattice constant a in pm

    Returns:
        UnitCell object with FCC atom positions
    """
    params = LatticeParameters(
        a=lattice_constant, b=lattice_constant, c=lattice_constant,
        alpha=90, beta=90, gamma=90
    )
    cell = UnitCell(CrystalStructure.FCC, params)

    # FCC atom positions (fractional)
    fcc_positions = [
        (0.0, 0.0, 0.0),  # Corner
        (0.5, 0.5, 0.0),  # Face center xy
        (0.5, 0.0, 0.5),  # Face center xz
        (0.0, 0.5, 0.5),  # Face center yz
    ]

    for pos in fcc_positions:
        cell.add_atom(element, *pos)

    return cell


def create_bcc_unit_cell(element: str, lattice_constant: float) -> UnitCell:
    """
    Create a BCC unit cell.

    Args:
        element: Element symbol
        lattice_constant: Lattice constant a in pm

    Returns:
        UnitCell object with BCC atom positions
    """
    params = LatticeParameters(
        a=lattice_constant, b=lattice_constant, c=lattice_constant,
        alpha=90, beta=90, gamma=90
    )
    cell = UnitCell(CrystalStructure.BCC, params)

    # BCC atom positions (fractional)
    bcc_positions = [
        (0.0, 0.0, 0.0),  # Corner
        (0.5, 0.5, 0.5),  # Body center
    ]

    for pos in bcc_positions:
        cell.add_atom(element, *pos)

    return cell


def create_hcp_unit_cell(element: str, a: float, c: float) -> UnitCell:
    """
    Create an HCP unit cell.

    Args:
        element: Element symbol
        a: Lattice constant a in pm
        c: Lattice constant c in pm

    Returns:
        UnitCell object with HCP atom positions
    """
    params = LatticeParameters(
        a=a, b=a, c=c,
        alpha=90, beta=90, gamma=120
    )
    cell = UnitCell(CrystalStructure.HCP, params)

    # HCP atom positions (fractional)
    hcp_positions = [
        (0.0, 0.0, 0.0),
        (1/3, 2/3, 0.5),
    ]

    for pos in hcp_positions:
        cell.add_atom(element, *pos)

    return cell


class LatticeDefect:
    """Base class for lattice defects."""

    def __init__(self, defect_type: str, position: Vec3):
        self.defect_type = defect_type
        self.position = position


class PointDefect(LatticeDefect):
    """Point defect (vacancy, interstitial, substitutional)."""

    def __init__(self, defect_subtype: str, position: Vec3, element: Optional[str] = None):
        super().__init__("point", position)
        self.subtype = defect_subtype  # "vacancy", "interstitial", "substitutional"
        self.element = element  # For substitutional or interstitial atoms


class Dislocation(LatticeDefect):
    """Line defect (dislocation)."""

    def __init__(self, position: Vec3, burgers_vector: Vec3, line_direction: Vec3,
                 dislocation_type: str = "edge"):
        super().__init__("dislocation", position)
        self.burgers_vector = burgers_vector
        self.line_direction = line_direction
        self.type = dislocation_type  # "edge", "screw", "mixed"


def generate_vacancy_distribution(unit_cell: UnitCell, concentration: float,
                                   supercell_size: Tuple[int, int, int],
                                   seed: int = 0) -> List[PointDefect]:
    """
    Generate random vacancy distribution in a supercell.

    Args:
        unit_cell: Base unit cell
        concentration: Vacancy concentration (0-1)
        supercell_size: Number of unit cells in each direction
        seed: Random seed

    Returns:
        List of vacancy defects
    """
    random.seed(seed)
    vacancies = []

    nx, ny, nz = supercell_size
    total_sites = len(unit_cell.atoms) * nx * ny * nz
    num_vacancies = int(total_sites * concentration)

    # Generate random vacancy positions
    for _ in range(num_vacancies):
        # Random unit cell
        cx = random.randint(0, nx - 1)
        cy = random.randint(0, ny - 1)
        cz = random.randint(0, nz - 1)

        # Random atom in unit cell
        atom_idx = random.randint(0, len(unit_cell.atoms) - 1)
        atom = unit_cell.atoms[atom_idx]

        # Calculate position
        frac = atom.fractional_coords
        total_frac = (
            (cx + frac[0]) / nx,
            (cy + frac[1]) / ny,
            (cz + frac[2]) / nz
        )
        position = unit_cell.fractional_to_cartesian(total_frac)

        vacancies.append(PointDefect("vacancy", position))

    return vacancies


# ==================== Visualization Functions ====================

@dataclass
class SlicePlane:
    """Defines a slice plane through 3D structure."""
    origin: Vec3
    normal: Vec3
    width: float
    height: float


class MicrostructureRenderer:
    """
    Renderer for microstructure visualization.
    """

    def __init__(self, voronoi: Optional[VoronoiTessellation] = None):
        """Initialize the renderer."""
        self.voronoi = voronoi
        self.color_map: Dict[int, Tuple[int, int, int]] = {}

    def _grain_id_to_color(self, grain_id: int) -> Tuple[int, int, int]:
        """Convert grain ID to RGB color."""
        if grain_id not in self.color_map:
            # Generate a color based on grain ID using golden ratio
            hue = (grain_id * 0.618033988749895) % 1.0
            # HSV to RGB conversion
            h = hue * 6.0
            c = 0.8  # Saturation
            x = c * (1 - abs(h % 2 - 1))
            m = 0.2  # Value offset

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            self.color_map[grain_id] = (
                int((r + m) * 255),
                int((g + m) * 255),
                int((b + m) * 255)
            )

        return self.color_map[grain_id]

    def render_2d_slice(self, width: int, height: int,
                         grain_boundary_color: Tuple[int, int, int] = (0, 0, 0),
                         boundary_width: float = 0.02) -> List[List[Tuple[int, int, int]]]:
        """
        Render a 2D slice of the microstructure.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            grain_boundary_color: RGB color for grain boundaries
            boundary_width: Width of grain boundaries in normalized units

        Returns:
            2D array of RGB tuples
        """
        if not self.voronoi or not self.voronoi.grain_centers:
            return [[(128, 128, 128)] * width for _ in range(height)]

        # Determine domain bounds
        min_x = min(g.position.x for g in self.voronoi.grain_centers)
        max_x = max(g.position.x for g in self.voronoi.grain_centers)
        min_y = min(g.position.y for g in self.voronoi.grain_centers)
        max_y = max(g.position.y for g in self.voronoi.grain_centers)

        domain_width = max_x - min_x
        domain_height = max_y - min_y

        # Add margin
        margin = max(domain_width, domain_height) * 0.1
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        domain_width = max_x - min_x
        domain_height = max_y - min_y

        image = []

        for py in range(height):
            row = []
            for px in range(width):
                # Convert pixel to domain coordinates
                x = min_x + (px / width) * domain_width
                y = min_y + (py / height) * domain_height
                point = Vec3(x, y, 0)

                # Find nearest grain and check for boundary
                nearest, dist = self.voronoi.find_nearest_grain(point)

                if self.voronoi.is_on_boundary(point, boundary_width * max(domain_width, domain_height)):
                    row.append(grain_boundary_color)
                else:
                    row.append(self._grain_id_to_color(nearest.grain_id))

            image.append(row)

        return image

    def render_ipf_map(self, width: int, height: int,
                        projection_direction: Vec3 = Vec3(0, 0, 1)) -> List[List[Tuple[int, int, int]]]:
        """
        Render Inverse Pole Figure (IPF) colored map.
        Colors grains based on their crystallographic orientation relative to a projection direction.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            projection_direction: Projection direction for IPF coloring

        Returns:
            2D array of RGB tuples
        """
        if not self.voronoi or not self.voronoi.grain_centers:
            return [[(128, 128, 128)] * width for _ in range(height)]

        # Determine domain bounds
        min_x = min(g.position.x for g in self.voronoi.grain_centers)
        max_x = max(g.position.x for g in self.voronoi.grain_centers)
        min_y = min(g.position.y for g in self.voronoi.grain_centers)
        max_y = max(g.position.y for g in self.voronoi.grain_centers)

        domain_width = max_x - min_x
        domain_height = max_y - min_y

        # Add margin
        margin = max(domain_width, domain_height) * 0.1
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        domain_width = max_x - min_x
        domain_height = max_y - min_y

        image = []

        for py in range(height):
            row = []
            for px in range(width):
                x = min_x + (px / width) * domain_width
                y = min_y + (py / height) * domain_height
                point = Vec3(x, y, 0)

                nearest, _ = self.voronoi.find_nearest_grain(point)

                # Convert orientation to IPF color
                # Simplified: use Euler angles directly for coloring
                phi1, phi, phi2 = nearest.orientation
                r = int((phi1 / 360) * 255)
                g = int((phi / 90) * 255)
                b = int((phi2 / 90) * 255)
                row.append((r, g, b))

            image.append(row)

        return image

    def render_phase_map(self, width: int, height: int,
                          phase_colors: Dict[int, Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        """
        Render a phase distribution map.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            phase_colors: Mapping of phase IDs to RGB colors

        Returns:
            2D array of RGB tuples
        """
        if not self.voronoi or not self.voronoi.grain_centers:
            return [[(128, 128, 128)] * width for _ in range(height)]

        # Determine domain bounds
        min_x = min(g.position.x for g in self.voronoi.grain_centers)
        max_x = max(g.position.x for g in self.voronoi.grain_centers)
        min_y = min(g.position.y for g in self.voronoi.grain_centers)
        max_y = max(g.position.y for g in self.voronoi.grain_centers)

        domain_width = max_x - min_x
        domain_height = max_y - min_y

        margin = max(domain_width, domain_height) * 0.1
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        domain_width = max_x - min_x
        domain_height = max_y - min_y

        image = []

        for py in range(height):
            row = []
            for px in range(width):
                x = min_x + (px / width) * domain_width
                y = min_y + (py / height) * domain_height
                point = Vec3(x, y, 0)

                nearest, _ = self.voronoi.find_nearest_grain(point)
                color = phase_colors.get(nearest.phase_id, (128, 128, 128))
                row.append(color)

            image.append(row)

        return image


def generate_noise_phase_map(width: int, height: int,
                              noise_type: str = "perlin",
                              scale: float = 0.1,
                              octaves: int = 4,
                              persistence: float = 0.5,
                              threshold: float = 0.5,
                              seed: int = 0) -> List[List[float]]:
    """
    Generate a phase distribution map using noise functions.

    Args:
        width: Map width in pixels
        height: Map height in pixels
        noise_type: Type of noise ("perlin", "simplex", "worley", "fbm")
        scale: Noise scale factor
        octaves: Number of octaves for fBm
        persistence: Persistence for fBm
        threshold: Threshold for phase determination
        seed: Random seed

    Returns:
        2D array of noise values (0-1)
    """
    # Initialize noise generator
    if noise_type == "perlin":
        noise = PerlinNoise(seed)
        get_noise = lambda x, y: (noise.noise2d(x * scale, y * scale) + 1) / 2
    elif noise_type == "simplex":
        noise = SimplexNoise(seed)
        get_noise = lambda x, y: (noise.noise2d(x * scale, y * scale) + 1) / 2
    elif noise_type == "worley":
        noise = WorleyNoise(seed)
        get_noise = lambda x, y: 1 - min(1, noise.noise2d(x * scale, y * scale)[0])
    elif noise_type == "fbm":
        base_noise = PerlinNoise(seed)
        noise = FractalBrownianMotion(base_noise, octaves, persistence)
        get_noise = lambda x, y: (noise.noise2d(x * scale, y * scale) + 1) / 2
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    result = []
    for y in range(height):
        row = []
        for x in range(width):
            value = get_noise(x, y)
            row.append(value)
        result.append(row)

    return result


# ==================== Miller Indices Utilities ====================

def miller_to_direction(h: int, k: int, l: int) -> Vec3:
    """
    Convert Miller indices to a direction vector.

    Args:
        h, k, l: Miller indices

    Returns:
        Normalized direction vector
    """
    vec = Vec3(float(h), float(k), float(l))
    return vec.normalize()


def direction_to_miller(direction: Vec3, tolerance: float = 0.01) -> Tuple[int, int, int]:
    """
    Approximate a direction vector with Miller indices.

    Args:
        direction: Direction vector
        tolerance: Tolerance for rounding

    Returns:
        Tuple of Miller indices (h, k, l)
    """
    # Normalize and find smallest integer representation
    d = direction.normalize()

    # Find the smallest component to scale by
    components = [abs(d.x), abs(d.y), abs(d.z)]
    min_val = min(c for c in components if c > tolerance) if any(c > tolerance for c in components) else 1

    # Scale to get integer-like values
    h = d.x / min_val
    k = d.y / min_val
    l = d.z / min_val

    # Round to nearest integers
    max_index = 10
    for scale in range(1, max_index + 1):
        hi, ki, li = round(h * scale), round(k * scale), round(l * scale)
        if (abs(hi - h * scale) < tolerance and
            abs(ki - k * scale) < tolerance and
            abs(li - l * scale) < tolerance):
            # Find GCD and simplify
            from math import gcd
            from functools import reduce
            g = reduce(gcd, [abs(hi), abs(ki), abs(li)]) if any([hi, ki, li]) else 1
            return (hi // g, ki // g, li // g)

    return (round(h), round(k), round(l))


def calculate_interplanar_spacing(h: int, k: int, l: int,
                                   lattice_params: LatticeParameters) -> float:
    """
    Calculate interplanar spacing d_hkl for cubic crystals.

    Args:
        h, k, l: Miller indices
        lattice_params: Lattice parameters

    Returns:
        Interplanar spacing in same units as lattice parameter
    """
    # For cubic crystals
    if (abs(lattice_params.alpha - 90) < 0.01 and
        abs(lattice_params.beta - 90) < 0.01 and
        abs(lattice_params.gamma - 90) < 0.01 and
        abs(lattice_params.a - lattice_params.b) < 0.01 and
        abs(lattice_params.b - lattice_params.c) < 0.01):

        a = lattice_params.a
        d = a / math.sqrt(h*h + k*k + l*l) if (h*h + k*k + l*l) > 0 else float('inf')
        return d

    # For general case (orthorhombic)
    a, b, c = lattice_params.a, lattice_params.b, lattice_params.c
    denom = math.sqrt((h/a)**2 + (k/b)**2 + (l/c)**2)
    return 1 / denom if denom > 0 else float('inf')


# ==================== Convenience Functions ====================

def create_microstructure(grain_density: float,
                           domain_size: Tuple[float, float],
                           seed: int = 0) -> VoronoiTessellation:
    """
    Create a complete microstructure with Voronoi grains.

    Args:
        grain_density: Grains per unit area
        domain_size: (width, height) of domain
        seed: Random seed

    Returns:
        VoronoiTessellation object with generated grains
    """
    width, height = domain_size
    num_grains = int(grain_density * width * height)

    voronoi = VoronoiTessellation(seed)
    voronoi.generate_grain_centers_2d(width, height, num_grains, "poisson")

    return voronoi


def assign_phases_to_grains(voronoi: VoronoiTessellation,
                             phase_fractions: Dict[int, float],
                             noise_influence: float = 0.0,
                             seed: int = 0) -> None:
    """
    Assign phases to grains based on target fractions.

    Args:
        voronoi: VoronoiTessellation with grain centers
        phase_fractions: Dictionary of phase_id -> target fraction
        noise_influence: Amount of spatial noise to add (0-1)
        seed: Random seed
    """
    random.seed(seed)

    # Normalize fractions
    total = sum(phase_fractions.values())
    normalized = {k: v / total for k, v in phase_fractions.items()}

    # Sort grains randomly
    grains = list(voronoi.grain_centers)
    random.shuffle(grains)

    # Assign phases based on fractions
    phase_ids = list(normalized.keys())
    cumulative = []
    cum = 0
    for pid in phase_ids:
        cum += normalized[pid]
        cumulative.append(cum)

    for i, grain in enumerate(grains):
        t = i / len(grains)

        # Add noise influence
        if noise_influence > 0:
            noise = PerlinNoise(seed)
            pos = grain.position
            noise_val = (noise.noise2d(pos.x * 0.1, pos.y * 0.1) + 1) / 2
            t = t * (1 - noise_influence) + noise_val * noise_influence

        # Find phase
        for j, cum_frac in enumerate(cumulative):
            if t <= cum_frac:
                grain.phase_id = phase_ids[j]
                break
        else:
            grain.phase_id = phase_ids[-1]
