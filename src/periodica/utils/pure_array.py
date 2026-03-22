"""
Pure Python array and vector utilities.

Replaces numpy operations for SDF rendering with zero external dependencies.
Provides basic math wrappers, a 3D vector class, and nucleon position generation
for nuclear visualization in the Periodics application.
"""
import math
import random
from typing import List, Tuple, Optional

# =============================================================================
# Constants
# =============================================================================

pi = math.pi

# =============================================================================
# Math Function Wrappers
# =============================================================================

def sqrt(x: float) -> float:
    """
    Compute the square root of x.

    Args:
        x: Non-negative number to compute square root of.

    Returns:
        Square root of x.

    Raises:
        ValueError: If x is negative.
    """
    return math.sqrt(x)


def cos(x: float) -> float:
    """
    Compute the cosine of x (in radians).

    Args:
        x: Angle in radians.

    Returns:
        Cosine of x.
    """
    return math.cos(x)


def sin(x: float) -> float:
    """
    Compute the sine of x (in radians).

    Args:
        x: Angle in radians.

    Returns:
        Sine of x.
    """
    return math.sin(x)


def acos(x: float) -> float:
    """
    Compute the arc cosine of x.

    Args:
        x: Value in range [-1, 1].

    Returns:
        Arc cosine in radians [0, pi].
    """
    return math.acos(x)


def atan2(y: float, x: float) -> float:
    """
    Compute the arc tangent of y/x, using signs to determine quadrant.

    Args:
        y: Y coordinate.
        x: X coordinate.

    Returns:
        Arc tangent in radians [-pi, pi].
    """
    return math.atan2(y, x)


# =============================================================================
# Random Number Utilities
# =============================================================================

def random_uniform(low: float = 0.0, high: float = 1.0) -> float:
    """
    Generate a random float uniformly distributed in [low, high).

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).

    Returns:
        Random float in [low, high).
    """
    return random.uniform(low, high)


def random_seed(seed: int) -> None:
    """
    Set the random seed for reproducible random number generation.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)


# =============================================================================
# Vec3 - 3D Vector Class
# =============================================================================

class Vec3:
    """
    Simple 3D vector class for nucleon positioning and SDF rendering.

    Uses __slots__ for memory efficiency when creating many instances.
    Supports basic vector operations: addition, scalar multiplication,
    rotations, and normalization.

    Attributes:
        x: X component of the vector.
        y: Y component of the vector.
        z: Z component of the vector.
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Initialize a 3D vector.

        Args:
            x: X component (default 0).
            y: Y component (default 0).
            z: Z component (default 0).
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        """Return string representation of vector."""
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Vec3."""
        if not isinstance(other, Vec3):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def __add__(self, other: 'Vec3') -> 'Vec3':
        """
        Add two vectors.

        Args:
            other: Vector to add.

        Returns:
            New Vec3 representing the sum.
        """
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        """
        Subtract two vectors.

        Args:
            other: Vector to subtract.

        Returns:
            New Vec3 representing the difference.
        """
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        """
        Multiply vector by a scalar.

        Args:
            scalar: Scalar value to multiply by.

        Returns:
            New Vec3 scaled by the scalar.
        """
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vec3':
        """
        Right multiply (scalar * vector).

        Args:
            scalar: Scalar value to multiply by.

        Returns:
            New Vec3 scaled by the scalar.
        """
        return self.__mul__(scalar)

    def __neg__(self) -> 'Vec3':
        """
        Negate the vector.

        Returns:
            New Vec3 with negated components.
        """
        return Vec3(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> 'Vec3':
        """
        Divide vector by a scalar.

        Args:
            scalar: Scalar value to divide by.

        Returns:
            New Vec3 divided by the scalar.
        """
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'Vec3') -> float:
        """
        Compute dot product with another vector.

        Args:
            other: Vector to compute dot product with.

        Returns:
            Dot product (scalar).
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        """
        Compute cross product with another vector.

        Args:
            other: Vector to compute cross product with.

        Returns:
            New Vec3 representing the cross product.
        """
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        """
        Compute the length (magnitude) of the vector.

        Returns:
            Euclidean length of the vector.
        """
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """
        Compute the squared length of the vector.

        More efficient than length() when only comparing magnitudes.

        Returns:
            Squared Euclidean length of the vector.
        """
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> 'Vec3':
        """
        Return a unit vector in the same direction.

        Returns:
            New Vec3 with length 1, or zero vector if length is 0.
        """
        length = self.length()
        if length == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / length, self.y / length, self.z / length)

    def rotate_x(self, angle: float) -> 'Vec3':
        """
        Rotate the vector around the X axis.

        Args:
            angle: Rotation angle in radians.

        Returns:
            New Vec3 rotated around the X axis.
        """
        c, s = cos(angle), sin(angle)
        return Vec3(
            self.x,
            self.y * c - self.z * s,
            self.y * s + self.z * c
        )

    def rotate_y(self, angle: float) -> 'Vec3':
        """
        Rotate the vector around the Y axis.

        Args:
            angle: Rotation angle in radians.

        Returns:
            New Vec3 rotated around the Y axis.
        """
        c, s = cos(angle), sin(angle)
        return Vec3(
            self.x * c + self.z * s,
            self.y,
            -self.x * s + self.z * c
        )

    def rotate_z(self, angle: float) -> 'Vec3':
        """
        Rotate the vector around the Z axis.

        Args:
            angle: Rotation angle in radians.

        Returns:
            New Vec3 rotated around the Z axis.
        """
        c, s = cos(angle), sin(angle)
        return Vec3(
            self.x * c - self.y * s,
            self.x * s + self.y * c,
            self.z
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        """
        Convert vector to tuple.

        Returns:
            Tuple (x, y, z).
        """
        return (self.x, self.y, self.z)

    @staticmethod
    def from_spherical(r: float, theta: float, phi: float) -> 'Vec3':
        """
        Create a vector from spherical coordinates.

        Args:
            r: Radial distance from origin.
            theta: Polar angle from positive z-axis [0, pi].
            phi: Azimuthal angle from positive x-axis [0, 2*pi].

        Returns:
            New Vec3 in Cartesian coordinates.
        """
        sin_theta = sin(theta)
        return Vec3(
            r * sin_theta * cos(phi),
            r * sin_theta * sin(phi),
            r * cos(theta)
        )

    @staticmethod
    def zero() -> 'Vec3':
        """Return a zero vector."""
        return Vec3(0, 0, 0)

    @staticmethod
    def unit_x() -> 'Vec3':
        """Return a unit vector along the X axis."""
        return Vec3(1, 0, 0)

    @staticmethod
    def unit_y() -> 'Vec3':
        """Return a unit vector along the Y axis."""
        return Vec3(0, 1, 0)

    @staticmethod
    def unit_z() -> 'Vec3':
        """Return a unit vector along the Z axis."""
        return Vec3(0, 0, 1)


# =============================================================================
# Nucleon Position Generation
# =============================================================================

def generate_nucleon_positions(
    protons: int,
    neutrons: int,
    nuclear_radius: float,
    seed: Optional[int] = None
) -> List[Tuple[float, float, float, bool]]:
    """
    Generate positions for protons and neutrons in a nucleus.

    Uses a liquid drop model approximation with random but deterministic
    placement. Nucleons are distributed uniformly within a sphere using
    spherical coordinates with proper volume weighting.

    The algorithm ensures:
    - Uniform distribution throughout the nuclear volume (not just surface)
    - Deterministic results when a seed is provided
    - Protons and neutrons are interleaved for realistic distribution

    Args:
        protons: Number of protons (atomic number Z).
        neutrons: Number of neutrons (N = A - Z).
        nuclear_radius: Radius of the nucleus in arbitrary units.
        seed: Optional random seed for reproducibility.

    Returns:
        List of tuples (x, y, z, is_proton) where:
            - x, y, z: Cartesian coordinates of the nucleon
            - is_proton: True for proton, False for neutron

    Example:
        >>> positions = generate_nucleon_positions(6, 6, 2.5, seed=42)
        >>> len(positions)
        12
        >>> positions[0]  # (x, y, z, is_proton)
        (0.123, -0.456, 0.789, True)
    """
    if seed is not None:
        random_seed(seed)

    total_nucleons = protons + neutrons
    if total_nucleons == 0:
        return []

    positions: List[Tuple[float, float, float, bool]] = []

    # Create a shuffled list of nucleon types for uniform mixing
    nucleon_types = [True] * protons + [False] * neutrons
    random.shuffle(nucleon_types)

    for is_proton in nucleon_types:
        # Generate uniform distribution within sphere
        # Using rejection sampling would be inefficient, so we use
        # proper spherical coordinate transformation

        # Radius: r = R * cbrt(uniform(0,1)) for uniform volume distribution
        u = random_uniform(0, 1)
        r = nuclear_radius * (u ** (1.0 / 3.0))

        # Polar angle theta: cos(theta) uniformly distributed in [-1, 1]
        cos_theta = random_uniform(-1, 1)
        theta = acos(cos_theta)

        # Azimuthal angle phi: uniformly distributed in [0, 2*pi)
        phi = random_uniform(0, 2 * pi)

        # Convert to Cartesian coordinates
        sin_theta = sin(theta)
        x = r * sin_theta * cos(phi)
        y = r * sin_theta * sin(phi)
        z = r * cos_theta

        positions.append((x, y, z, is_proton))

    return positions


def generate_shell_positions(
    protons: int,
    neutrons: int,
    nuclear_radius: float,
    shell_count: int = 3,
    seed: Optional[int] = None
) -> List[Tuple[float, float, float, bool]]:
    """
    Generate nucleon positions using a shell model distribution.

    Places nucleons in concentric shells for a more structured
    nuclear visualization, similar to electron shells in atoms.

    Args:
        protons: Number of protons.
        neutrons: Number of neutrons.
        nuclear_radius: Outer radius of the nucleus.
        shell_count: Number of concentric shells (default 3).
        seed: Optional random seed for reproducibility.

    Returns:
        List of tuples (x, y, z, is_proton).
    """
    if seed is not None:
        random_seed(seed)

    total_nucleons = protons + neutrons
    if total_nucleons == 0:
        return []

    positions: List[Tuple[float, float, float, bool]] = []

    # Create shuffled nucleon types
    nucleon_types = [True] * protons + [False] * neutrons
    random.shuffle(nucleon_types)

    # Distribute nucleons across shells (inner shells are smaller)
    # Shell capacities increase with radius squared
    shell_radii = [nuclear_radius * (i + 1) / shell_count for i in range(shell_count)]
    shell_capacities = [(i + 1) ** 2 for i in range(shell_count)]
    total_capacity = sum(shell_capacities)

    # Assign nucleons to shells proportionally
    nucleon_index = 0
    for shell_idx, (shell_r, capacity) in enumerate(zip(shell_radii, shell_capacities)):
        # Calculate how many nucleons go in this shell
        if shell_idx == shell_count - 1:
            # Last shell gets remaining nucleons
            shell_nucleons = total_nucleons - nucleon_index
        else:
            shell_nucleons = int(total_nucleons * capacity / total_capacity)

        # Place nucleons on this shell surface with some radial variation
        for _ in range(shell_nucleons):
            if nucleon_index >= total_nucleons:
                break

            is_proton = nucleon_types[nucleon_index]

            # Add some radial jitter (5% of shell radius)
            r = shell_r * (1 + random_uniform(-0.05, 0.05))

            # Uniform distribution on sphere surface
            cos_theta = random_uniform(-1, 1)
            theta = acos(cos_theta)
            phi = random_uniform(0, 2 * pi)

            sin_theta = sin(theta)
            x = r * sin_theta * cos(phi)
            y = r * sin_theta * sin(phi)
            z = r * cos_theta

            positions.append((x, y, z, is_proton))
            nucleon_index += 1

    return positions


# =============================================================================
# Utility Functions
# =============================================================================

def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.

    Args:
        a: Start value.
        b: End value.
        t: Interpolation factor [0, 1].

    Returns:
        Interpolated value a + t * (b - a).
    """
    return a + t * (b - a)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a range.

    Args:
        value: Value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """
    Smooth Hermite interpolation between 0 and 1.

    Args:
        edge0: Lower edge of transition.
        edge1: Upper edge of transition.
        x: Input value.

    Returns:
        0 if x <= edge0, 1 if x >= edge1, smooth transition between.
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """
    Compute Euclidean distance between two 3D points.

    Args:
        p1: First point (x, y, z).
        p2: Second point (x, y, z).

    Returns:
        Distance between points.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


# =============================================================================
# 3D Rotation Matrix Functions
# =============================================================================

def rotation_matrix_x(angle: float) -> List[List[float]]:
    """
    Create a 3x3 rotation matrix for rotation around the X axis.

    The rotation is counter-clockwise when looking from positive X towards origin.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as list of lists [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]].

    Example:
        >>> R = rotation_matrix_x(math.pi / 2)  # 90 degrees
        >>> # Rotates Y axis to Z axis
    """
    c = cos(angle)
    s = sin(angle)
    return [
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ]


def rotation_matrix_y(angle: float) -> List[List[float]]:
    """
    Create a 3x3 rotation matrix for rotation around the Y axis.

    The rotation is counter-clockwise when looking from positive Y towards origin.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as list of lists.
    """
    c = cos(angle)
    s = sin(angle)
    return [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ]


def rotation_matrix_z(angle: float) -> List[List[float]]:
    """
    Create a 3x3 rotation matrix for rotation around the Z axis.

    The rotation is counter-clockwise when looking from positive Z towards origin.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as list of lists.
    """
    c = cos(angle)
    s = sin(angle)
    return [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ]


def rotation_matrix_axis_angle(axis: Tuple[float, float, float], angle: float) -> List[List[float]]:
    """
    Create a 3x3 rotation matrix for rotation around an arbitrary axis (Rodrigues' formula).

    Args:
        axis: Unit vector (x, y, z) defining the rotation axis.
               Will be normalized if not already unit length.
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as list of lists.

    Notes:
        Uses Rodrigues' rotation formula:
        R = I + sin(θ)K + (1-cos(θ))K²

        where K is the skew-symmetric cross-product matrix of the axis.
    """
    # Normalize the axis
    ax, ay, az = axis
    length = sqrt(ax * ax + ay * ay + az * az)
    if length < 1e-10:
        # Return identity for zero-length axis
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    ax /= length
    ay /= length
    az /= length

    c = cos(angle)
    s = sin(angle)
    t = 1.0 - c

    # Rodrigues' rotation formula
    return [
        [c + ax * ax * t, ax * ay * t - az * s, ax * az * t + ay * s],
        [ay * ax * t + az * s, c + ay * ay * t, ay * az * t - ax * s],
        [az * ax * t - ay * s, az * ay * t + ax * s, c + az * az * t]
    ]


def rotation_matrix_euler(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """
    Create a 3x3 rotation matrix from Euler angles (ZYX convention).

    Applies rotations in order: roll (X), pitch (Y), yaw (Z).
    This is the common aerospace convention (extrinsic rotations).

    Args:
        roll: Rotation around X axis in radians.
        pitch: Rotation around Y axis in radians.
        yaw: Rotation around Z axis in radians.

    Returns:
        3x3 rotation matrix as list of lists.
    """
    # Compute individual rotation matrices
    Rx = rotation_matrix_x(roll)
    Ry = rotation_matrix_y(pitch)
    Rz = rotation_matrix_z(yaw)

    # Combine: R = Rz * Ry * Rx (applied right to left)
    temp = matrix_multiply_3x3(Rz, Ry)
    return matrix_multiply_3x3(temp, Rx)


def matrix_multiply_3x3(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Multiply two 3x3 matrices.

    Args:
        A: First 3x3 matrix.
        B: Second 3x3 matrix.

    Returns:
        Result of A * B as 3x3 matrix.
    """
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_vector_multiply_3x3(M: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Multiply a 3x3 matrix by a 3D vector.

    Args:
        M: 3x3 matrix.
        v: 3D vector as tuple (x, y, z).

    Returns:
        Result as tuple (x', y', z').
    """
    return (
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]
    )


def apply_rotation_matrix(M: List[List[float]], vec: 'Vec3') -> 'Vec3':
    """
    Apply a 3x3 rotation matrix to a Vec3.

    Args:
        M: 3x3 rotation matrix.
        vec: Vec3 to transform.

    Returns:
        New rotated Vec3.
    """
    x, y, z = vec.x, vec.y, vec.z
    return Vec3(
        M[0][0] * x + M[0][1] * y + M[0][2] * z,
        M[1][0] * x + M[1][1] * y + M[1][2] * z,
        M[2][0] * x + M[2][1] * y + M[2][2] * z
    )


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    # Basic math tests
    print("Testing math functions...")
    assert abs(sqrt(4) - 2.0) < 1e-10
    assert abs(sin(pi / 2) - 1.0) < 1e-10
    assert abs(cos(0) - 1.0) < 1e-10
    print("  Math functions: OK")

    # Vec3 tests
    print("Testing Vec3...")
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)

    # Addition
    v_sum = v1 + v2
    assert v_sum.x == 5 and v_sum.y == 7 and v_sum.z == 9

    # Scalar multiplication
    v_scaled = v1 * 2
    assert v_scaled.x == 2 and v_scaled.y == 4 and v_scaled.z == 6

    # Length
    v_unit = Vec3(1, 0, 0)
    assert abs(v_unit.length() - 1.0) < 1e-10

    # Normalization
    v3 = Vec3(3, 0, 0)
    v3_norm = v3.normalized()
    assert abs(v3_norm.length() - 1.0) < 1e-10

    # Rotation (90 degrees around Y should swap x and z)
    v_x = Vec3(1, 0, 0)
    v_rotated = v_x.rotate_y(pi / 2)
    assert abs(v_rotated.x) < 1e-10  # Should be ~0
    assert abs(v_rotated.z + 1) < 1e-10  # Should be ~-1

    print("  Vec3: OK")

    # Nucleon generation tests
    print("Testing nucleon generation...")

    # Test reproducibility with seed
    pos1 = generate_nucleon_positions(6, 6, 2.5, seed=42)
    pos2 = generate_nucleon_positions(6, 6, 2.5, seed=42)
    assert pos1 == pos2, "Seeded generation should be reproducible"

    # Test correct count
    assert len(pos1) == 12, "Should have 12 nucleons"

    # Test correct proton/neutron count
    proton_count = sum(1 for p in pos1 if p[3])
    neutron_count = sum(1 for p in pos1 if not p[3])
    assert proton_count == 6, "Should have 6 protons"
    assert neutron_count == 6, "Should have 6 neutrons"

    # Test all positions are within radius
    for x, y, z, _ in pos1:
        r = sqrt(x*x + y*y + z*z)
        assert r <= 2.5, f"Position {(x,y,z)} outside nuclear radius"

    print("  Nucleon generation: OK")

    # Utility function tests
    print("Testing utility functions...")
    assert lerp(0, 10, 0.5) == 5
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    assert abs(smoothstep(0, 1, 0.5) - 0.5) < 0.1
    assert abs(distance((0, 0, 0), (3, 4, 0)) - 5.0) < 1e-10
    print("  Utility functions: OK")

    print("\nAll tests passed!")
