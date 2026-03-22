#====== packages/periodica/src/periodica/utils/sdf_core.py ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

"""
SDF (Signed Distance Field) computation core -- pure math, zero Qt dependencies.

Provides:
- SDF primitives: sphere, union, smooth union
- Smoothstep alpha conversion
- Nucleon position generation (numpy and pure-python backends)
- Orbital lobe geometry (3D rotation, projection, depth)
- Nucleus geometry helpers (radius scaling from liquid-drop model)

All functions return plain numbers, tuples, or lists.
No QImage, QPainter, QColor, QPixmap, or any PySide6 import.
"""

import math

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

USE_NUMPY = True

try:
    if USE_NUMPY:
        import numpy as np
        _NUMPY_AVAILABLE = True
    else:
        _NUMPY_AVAILABLE = False
except ImportError:
    _NUMPY_AVAILABLE = False

if not _NUMPY_AVAILABLE:
    from periodica.utils.pure_array import (
        Vec3, pi, sqrt, cos, sin,
        random_seed, random_uniform,
        generate_nucleon_positions,
    )


def set_backend(use_numpy: bool):
    """
    Switch between numpy and pure Python backends.

    Args:
        use_numpy: True to use numpy backend, False for pure Python.
    """
    global USE_NUMPY, _NUMPY_AVAILABLE
    USE_NUMPY = use_numpy

    if use_numpy:
        try:
            import numpy as np  # noqa: F811
            _NUMPY_AVAILABLE = True
        except ImportError:
            _NUMPY_AVAILABLE = False
    else:
        _NUMPY_AVAILABLE = False


def get_backend() -> str:
    """
    Return current backend name.

    Returns:
        ``"numpy"`` if using numpy backend, ``"pure_python"`` otherwise.
    """
    return "numpy" if _NUMPY_AVAILABLE and USE_NUMPY else "pure_python"


# ---------------------------------------------------------------------------
# SDF primitives
# ---------------------------------------------------------------------------

def sdf_sphere(x, y, cx, cy, radius):
    """
    Signed distance to a circle/sphere cross-section.

    Negative inside, positive outside.

    Args:
        x, y: Query point coordinates.
        cx, cy: Circle center coordinates.
        radius: Circle radius.

    Returns:
        Signed distance (float).
    """
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius


def sdf_to_alpha(distance, softness=2.0):
    """
    Convert an SDF distance to an alpha value via smoothstep.

    Uses hermite interpolation for anti-aliased edges.

    Args:
        distance: Signed distance value.
        softness: Controls the width of the falloff band.

    Returns:
        Alpha in [0, 1].
    """
    t = max(0.0, min(1.0, 0.5 - distance / softness))
    return t * t * (3.0 - 2.0 * t)  # smoothstep


def sdf_union(d1, d2):
    """Union of two SDF shapes (minimum distance)."""
    return min(d1, d2)


def sdf_smooth_union(d1, d2, k=0.5):
    """
    Smooth union of two SDF shapes for organic blending.

    Args:
        d1, d2: Signed distances to two shapes.
        k: Blending factor (higher = sharper transition).

    Returns:
        Blended signed distance (float).
    """
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


# ---------------------------------------------------------------------------
# Nucleus geometry
# ---------------------------------------------------------------------------

def nucleus_geometry(protons, neutrons, base_radius):
    """
    Compute nuclear and nucleon visual radii using the liquid-drop model.

    R = r0 * A^(1/3) where r0 ~ 1.25 fm

    Args:
        protons: Number of protons.
        neutrons: Number of neutrons.
        base_radius: Base visual radius (pixels) for scaling.

    Returns:
        (nuclear_radius, nucleon_radius) tuple, or (0, 0) if A == 0.
    """
    A = protons + neutrons
    if A == 0:
        return 0.0, 0.0

    nuclear_radius = base_radius * (A ** (1 / 3)) / 6.0
    nucleon_radius = max(2, nuclear_radius / max(1, (A ** (1 / 3))) * 0.8)
    return nuclear_radius, nucleon_radius


def nucleus_radius_fm(protons, neutrons, r0_fm=1.25):
    """
    Real nuclear radius in femtometers.

    Args:
        protons: Number of protons.
        neutrons: Number of neutrons.
        r0_fm: Nucleon radius constant (default 1.25 fm).

    Returns:
        Radius in femtometers (float).
    """
    A = protons + neutrons
    if A == 0:
        return 0.0
    return r0_fm * (A ** (1 / 3))


# ---------------------------------------------------------------------------
# Nucleon position generation
# ---------------------------------------------------------------------------

def generate_nucleons_numpy(protons, neutrons, nuclear_radius,
                            rotation_x, rotation_y):
    """
    Generate nucleon positions using the numpy backend.

    Positions are deterministic for a given (protons, neutrons) pair.

    Args:
        protons: Number of protons.
        neutrons: Number of neutrons.
        nuclear_radius: Visual radius of the nucleus (pixels).
        rotation_x, rotation_y: 3-D rotation angles (radians).

    Returns:
        List of ``(dx2, dy2, dz3, is_proton)`` tuples sorted by depth
        (back-to-front).
    """
    import numpy as np  # noqa: F811

    A = protons + neutrons
    np.random.seed(protons * 1000 + neutrons)

    cos_rx, sin_rx = np.cos(rotation_x), np.sin(rotation_x)
    cos_ry, sin_ry = np.cos(rotation_y), np.sin(rotation_y)

    nucleon_data = []

    for i in range(A):
        is_proton = i < protons

        if A == 1:
            dx, dy, dz = 0, 0, 0
        else:
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            r = nuclear_radius * 0.7 * np.random.uniform(0.3, 1.0)

            dx = r * sin_theta * np.cos(phi)
            dy = r * sin_theta * np.sin(phi)
            dz = r * cos_theta

        # Rotate around X axis
        dy2 = dy * cos_rx - dz * sin_rx
        dz2 = dy * sin_rx + dz * cos_rx

        # Rotate around Y axis
        dx2 = dx * cos_ry + dz2 * sin_ry
        dz3 = -dx * sin_ry + dz2 * cos_ry

        nucleon_data.append((dx2, dy2, dz3, is_proton))

    return nucleon_data


def generate_nucleons_pure(protons, neutrons, nuclear_radius,
                           rotation_x, rotation_y):
    """
    Generate nucleon positions using the pure-Python backend.

    Positions are deterministic for a given (protons, neutrons) pair.

    Args:
        protons: Number of protons.
        neutrons: Number of neutrons.
        nuclear_radius: Visual radius of the nucleus (pixels).
        rotation_x, rotation_y: 3-D rotation angles (radians).

    Returns:
        List of ``(dx2, dy2, dz3, is_proton)`` tuples sorted by depth
        (back-to-front).
    """
    from periodica.utils.pure_array import (
        pi, sqrt, cos, sin,
        random_seed, random_uniform,
    )

    A = protons + neutrons
    random_seed(protons * 1000 + neutrons)

    cos_rx, sin_rx = cos(rotation_x), sin(rotation_x)
    cos_ry, sin_ry = cos(rotation_y), sin(rotation_y)

    nucleon_data = []

    for i in range(A):
        is_proton = i < protons

        if A == 1:
            dx, dy, dz = 0, 0, 0
        else:
            phi = random_uniform(0, 2 * pi)
            cos_theta = random_uniform(-1, 1)
            sin_theta = sqrt(1 - cos_theta ** 2)
            r = nuclear_radius * 0.7 * random_uniform(0.3, 1.0)

            dx = r * sin_theta * cos(phi)
            dy = r * sin_theta * sin(phi)
            dz = r * cos_theta

        # Rotate around X axis
        dy2 = dy * cos_rx - dz * sin_rx
        dz2 = dy * sin_rx + dz * cos_rx

        # Rotate around Y axis
        dx2 = dx * cos_ry + dz2 * sin_ry
        dz3 = -dx * sin_ry + dz2 * cos_ry

        nucleon_data.append((dx2, dy2, dz3, is_proton))

    return nucleon_data


def generate_nucleons(protons, neutrons, nuclear_radius,
                      rotation_x=0, rotation_y=0):
    """
    Generate nucleon positions, auto-selecting the active backend.

    Args:
        protons: Number of protons.
        neutrons: Number of neutrons.
        nuclear_radius: Visual radius of the nucleus (pixels).
        rotation_x, rotation_y: 3-D rotation angles (radians).

    Returns:
        List of ``(dx2, dy2, dz3, is_proton)`` tuples.
    """
    if _NUMPY_AVAILABLE and USE_NUMPY:
        return generate_nucleons_numpy(
            protons, neutrons, nuclear_radius, rotation_x, rotation_y)
    return generate_nucleons_pure(
        protons, neutrons, nuclear_radius, rotation_x, rotation_y)


# ---------------------------------------------------------------------------
# Nucleon depth projection helpers
# ---------------------------------------------------------------------------

def project_nucleon(dx, dy, dz, cx, cy, nuclear_radius):
    """
    Project a 3-D nucleon offset to 2-D screen coordinates with depth scaling.

    Args:
        dx, dy, dz: 3-D offset from nucleus center.
        cx, cy: Screen-space center of the nucleus.
        nuclear_radius: Visual nuclear radius (pixels).

    Returns:
        ``(px, py, depth_alpha)`` -- screen position and opacity factor.
    """
    depth_scale = 1.0 / (1.0 + dz / (nuclear_radius * 3 + 1))
    px = cx + dx * depth_scale
    py = cy + dy * depth_scale

    depth_alpha = 0.4 + 0.6 * (1 + dz / (nuclear_radius + 1)) / 2
    depth_alpha = max(0.3, min(1.0, depth_alpha))

    return px, py, depth_alpha


# ---------------------------------------------------------------------------
# Orbital lobe geometry (p-orbitals)
# ---------------------------------------------------------------------------

def p_orbital_lobes(m, shell_radius, rotation_x, rotation_y):
    """
    Compute the two lobe geometries for a p-orbital.

    Args:
        m: Magnetic quantum number (-1, 0, or 1).
        shell_radius: Electron-shell radius (pixels).
        rotation_x, rotation_y: 3-D rotation angles (radians).

    Returns:
        List of two dicts, each containing:
        ``{ 'cx': float, 'cy_offset': float, 'radius': float,
            'depth_alpha': float, 'positive_phase': bool }``
        (cx/cy offsets are relative to the nucleus center).
    """
    max_extent = shell_radius * 2.2

    cos_rx, sin_rx = math.cos(rotation_x), math.sin(rotation_x)
    cos_ry, sin_ry = math.cos(rotation_y), math.sin(rotation_y)

    # Determine lobe axis based on m
    if m == 0:      # pz -- along z axis (appears vertical)
        lobe_axis = (0, 1, 0)
    elif m == -1:   # px
        lobe_axis = (1, 0, 0)
    else:           # m == 1, py (into screen)
        lobe_axis = (0, 0, 1)

    lobes = []
    for lobe_sign in [1, -1]:
        lobe_offset = max_extent * 0.4

        lobe_x = lobe_axis[0] * lobe_offset * lobe_sign
        lobe_y = lobe_axis[1] * lobe_offset * lobe_sign
        lobe_z = lobe_axis[2] * lobe_offset * lobe_sign

        # Rotate around X
        ly2 = lobe_y * cos_rx - lobe_z * sin_rx
        lz2 = lobe_y * sin_rx + lobe_z * cos_rx

        # Rotate around Y
        lx2 = lobe_x * cos_ry + lz2 * sin_ry
        lz3 = -lobe_x * sin_ry + lz2 * cos_ry

        # Depth-based scaling
        depth_factor = 1.0 / (1.0 + lz3 / (max_extent * 2))
        depth_factor = max(0.5, min(1.5, depth_factor))

        lobe_radius = max_extent * 0.5 * depth_factor

        depth_alpha = 0.4 + 0.6 * (1 + lz3 / max_extent) / 2
        depth_alpha = max(0.3, min(1.0, depth_alpha))

        lobes.append({
            'cx_offset': lx2,
            'cy_offset': ly2,
            'radius': lobe_radius,
            'radius_y_ratio': 0.7,   # ellipse aspect ratio
            'depth_alpha': depth_alpha,
            'positive_phase': lobe_sign > 0,
        })

    return lobes


# ---------------------------------------------------------------------------
# Angular-orbital sampling grid (d, f orbitals)
# ---------------------------------------------------------------------------

def angular_orbital_grid(n, l, m, shell_radius,
                         cos_rx, sin_rx, cos_ry, sin_ry,
                         Z=1, grid_size=25, prob_threshold=0.03):
    """
    Sample an angular (d/f) orbital on a 2-D grid and return visible blobs.

    Each blob is a dict ready for the Qt renderer to paint as a radial
    gradient, but contains no Qt objects itself.

    Args:
        n, l, m: Quantum numbers.
        shell_radius: Shell radius (pixels).
        cos_rx, sin_rx: Pre-computed rotation-X trig values.
        cos_ry, sin_ry: Pre-computed rotation-Y trig values.
        Z: Atomic number (effective nuclear charge).
        grid_size: Number of samples per axis.
        prob_threshold: Minimum probability to emit a blob.

    Returns:
        List of dicts:
        ``{ 'px': float, 'py_offset': float, 'size': float, 'prob': float }``
        (px/py_offset are relative to the nucleus center).
    """
    from periodica.utils.orbital_clouds import get_orbital_probability

    max_extent = shell_radius * 2.0
    blob_size = max_extent / grid_size * 1.5

    blobs = []

    for i in range(grid_size):
        for j in range(grid_size):
            nx = (i - grid_size / 2) / (grid_size / 2)
            ny = (j - grid_size / 2) / (grid_size / 2)

            r_norm = math.sqrt(nx * nx + ny * ny)
            if r_norm > 1.0 or r_norm < 0.05:
                continue

            r = r_norm * max_extent

            theta = math.acos(ny / r_norm) if r_norm > 0 else 0
            phi = math.atan2(nx, 0.1)

            r_bohr = r / (shell_radius / n) if shell_radius > 0 else r
            prob = get_orbital_probability(n, l, m, r_bohr, theta, phi, Z)
            prob = min(1.0, prob * 20)

            if prob < prob_threshold:
                continue

            # 3-D position
            x_3d = nx * max_extent
            y_3d = ny * max_extent
            z_3d = 0  # flat slice

            # Apply rotation
            y_rot = y_3d * cos_rx - z_3d * sin_rx
            z_rot = y_3d * sin_rx + z_3d * cos_rx

            x_rot = x_3d * cos_ry + z_rot * sin_ry

            blobs.append({
                'px_offset': x_rot,
                'py_offset': y_rot,
                'size': blob_size,
                'prob': prob,
            })

    return blobs


# ---------------------------------------------------------------------------
# s-orbital radial sampling
# ---------------------------------------------------------------------------

def s_orbital_rings(n, shell_radius, cos_rx, cos_ry,
                    Z=1, animation_offset=0.0, resolution=35):
    """
    Sample an s-orbital radially and return concentric ring descriptors.

    Args:
        n: Principal quantum number.
        shell_radius: Shell radius (pixels).
        cos_rx, cos_ry: Pre-computed rotation trig values (for ellipse).
        Z: Atomic number.
        animation_offset: Pulsing animation phase offset.
        resolution: Number of radial samples.

    Returns:
        List of dicts (outer-to-inner order):
        ``{ 'r': float, 'scale_x': float, 'scale_y': float,
            'prob': float }``
    """
    from periodica.utils.orbital_clouds import get_orbital_probability

    max_extent = shell_radius * 2.0
    rings = []

    for i in range(resolution - 1, -1, -1):
        t = (i + 1) / resolution
        r = t * max_extent

        r_bohr = r / (shell_radius / n) if shell_radius > 0 else r
        prob = get_orbital_probability(n, 0, 0, r_bohr, 0, 0, Z)
        prob = min(1.0, prob * 8.0)

        prob_animated = prob * (1.0 + animation_offset * math.sin(t * math.pi * 2))
        prob_animated = max(0.0, min(1.0, prob_animated))

        if prob_animated < 0.01:
            continue

        scale_x = max(0.4, abs(cos_ry))
        scale_y = max(0.4, abs(cos_rx))

        rings.append({
            'r': r,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'prob': prob_animated,
        })

    return rings


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

_orbital_cache = {}
_cache_max_size = 1000


def clear_cache():
    """Clear the orbital probability cache."""
    _orbital_cache.clear()
