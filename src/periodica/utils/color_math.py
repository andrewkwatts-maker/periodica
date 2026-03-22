"""
Pure-math color and physics utilities for the periodic table.

All functions in this module are free of Qt / PySide6 dependencies.
Colors are returned as plain (r, g, b) or (r, g, b, a) tuples with
integer channel values in 0-255.

Physical unit conversions (eV <-> wavelength, eV <-> frequency) and
emission-spectrum calculation live here as well.
"""

import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C = 299_792_458  # m/s  (speed of light in vacuum)

# Planck constant in two convenient unit systems
_H_eVs = 4.135_667_696e-15   # eV * s
_H_Js  = 6.626_070_15e-34    # J * s
_E_CHARGE = 1.602_176_634e-19  # J / eV


# ===================================================================
#  Unit conversions
# ===================================================================

def ev_to_frequency(ev: float) -> float:
    """Convert electron-volts to PetaHertz (PHz).

    Parameters
    ----------
    ev : float
        Energy in electron-volts.

    Returns
    -------
    float
        Frequency in PHz (10^15 Hz).
    """
    return ev / _H_eVs / 1e15


def ev_to_wavelength(ev: float) -> float:
    """Convert electron-volts to wavelength in nanometres.

    Parameters
    ----------
    ev : float
        Energy in electron-volts.  Must be > 0.

    Returns
    -------
    float
        Wavelength in nm.
    """
    return (_H_Js * C) / (ev * _E_CHARGE) * 1e9


# ===================================================================
#  Emission-spectrum calculation
# ===================================================================

# Module-level cache: {(z, ie, max_n): spectrum_lines}
_spectrum_cache: Dict[Tuple, List[Tuple[float, float]]] = {}


def calculate_emission_spectrum(
    z: int,
    ionization_energy_ev: float,
    max_n: int = 20,
) -> List[Tuple[float, float]]:
    """Calculate approximate emission spectrum lines for an element.

    Uses a simplified Rydberg-like formula adjusted for the element's
    actual first ionization energy.  Results are cached per unique
    ``(z, ionization_energy_ev, max_n)`` key.

    Parameters
    ----------
    z : int
        Atomic number.
    ionization_energy_ev : float
        First ionization energy in eV.
    max_n : int, optional
        Maximum principal quantum number to include (default 20).
        Higher values produce more spectral lines at the cost of speed.

    Returns
    -------
    list of (wavelength_nm, relative_intensity)
        Visible / near-visible lines sorted by wavelength, with
        intensities normalised to [0, 1].
    """
    cache_key = (z, ionization_energy_ev, max_n)
    if cache_key in _spectrum_cache:
        return _spectrum_cache[cache_key]

    scale_factor = ionization_energy_ev / 13.6
    lines: List[Tuple[float, float]] = []

    for n_upper in range(2, max_n + 1):
        for n_lower in range(1, n_upper):
            E_upper = -13.6 * scale_factor / (n_upper ** 2)
            E_lower = -13.6 * scale_factor / (n_lower ** 2)
            delta_E = E_upper - E_lower  # positive (upper is less negative)

            if delta_E <= 0:
                continue

            wavelength = ev_to_wavelength(delta_E)

            # Keep lines in extended visible range (200 -- 1000 nm)
            if 200 <= wavelength <= 1000:
                delta_n = n_upper - n_lower

                intensity = 1.0 / (delta_n * n_lower)
                intensity *= 1.0 / n_upper ** 0.5

                # Series-specific weighting
                if n_lower == 1:
                    intensity *= 2.0   # Lyman (mostly UV)
                elif n_lower == 2:
                    intensity *= 1.5   # Balmer (visible)
                elif n_lower == 3:
                    intensity *= 0.8   # Paschen (mostly IR)
                else:
                    intensity *= 0.5

                lines.append((wavelength, intensity))

    # Normalise intensities to [0, 1] and sort by wavelength
    if lines:
        max_intensity = max(i for _, i in lines)
        lines = [(wl, i / max_intensity) for wl, i in lines]
        lines.sort(key=lambda x: x[0])

    _spectrum_cache[cache_key] = lines
    return lines


# ===================================================================
#  Wavelength -> RGB
# ===================================================================

def wavelength_to_rgb(
    wavelength_nm: float,
    range_min: float = 380,
    range_max: float = 780,
    fade: float = 0.0,
) -> Tuple[int, int, int, int]:
    """Map a wavelength value to a visible-spectrum RGBA colour.

    The input wavelength is first normalised into ``[range_min, range_max]``
    and then mapped onto the 380--780 nm visible spectrum.  Values near
    the edges fade toward white (short wavelengths) or black (long
    wavelengths).

    Parameters
    ----------
    wavelength_nm : float
        Wavelength to convert.
    range_min : float
        Value that maps to the violet end (380 nm).
    range_max : float
        Value that maps to the red end (780 nm).
    fade : float
        Fade toward transparent.  0.0 = fully opaque, 1.0 = fully
        transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
        Each channel in 0-255.
    """
    if range_min >= range_max:
        range_max = range_min + 1

    normalized = (wavelength_nm - range_min) / (range_max - range_min)
    w = 380 + normalized * (780 - 380)
    w = max(380.0, min(780.0, w))

    # Out of range fallback (should not trigger after clamping)
    if w < 380 or w > 780:
        alpha = int(255 * (1.0 - fade))
        return (120, 120, 150, alpha)

    # Rainbow piecewise linear mapping
    if 380 <= w < 440:
        r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif 510 <= w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    # Edge intensity fall-off
    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / 40
    elif 645 < w <= 780:
        factor = 0.3 + 0.7 * (780 - w) / 135
    else:
        factor = 1.0

    r, g, b = r * factor, g * factor, b * factor

    # White blend for short wavelengths, black blend for long
    if normalized < 0.2:
        wf = 1.0 - (normalized / 0.2)
        r = r * (1 - wf) + wf
        g = g * (1 - wf) + wf
        b = b * (1 - wf) + wf
    elif normalized > 0.8:
        bf = (normalized - 0.8) / 0.2
        r *= 1 - bf
        g *= 1 - bf
        b *= 1 - bf

    alpha = int(255 * (1.0 - fade))
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


# ===================================================================
#  Property-based colour gradients
# ===================================================================

def get_block_color(block: str) -> Tuple[int, int, int, int]:
    """Return a base RGBA colour for the given orbital block.

    Parameters
    ----------
    block : str
        One of ``'s'``, ``'p'``, ``'d'``, ``'f'``.

    Returns
    -------
    (r, g, b, a) : tuple of int
        Fully opaque colour (a = 255).
    """
    colors: Dict[str, Tuple[int, int, int, int]] = {
        's': (255,  80, 100, 255),
        'p': ( 80, 150, 255, 255),
        'd': (255, 200,  80, 255),
        'f': (120, 255, 150, 255),
    }
    return colors.get(block, (200, 200, 200, 255))


def get_ie_color(ie: float, fade: float = 0.0) -> Tuple[int, int, int, int]:
    """Colour gradient for ionization energy (cool -> warm).

    Parameters
    ----------
    ie : float
        Ionization energy in eV.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    normalized = max(0.0, min(1.0, (ie - 3.5) / (25.0 - 3.5)))

    if normalized < 0.2:
        t = normalized / 0.2
        r, g, b = int(100 * (1 - t)), int(100 * (1 - t) + 200 * t), 255
    elif normalized < 0.4:
        t = (normalized - 0.2) / 0.2
        r, g, b = 0, int(200 * (1 - t) + 255 * t), int(255 * (1 - t) + 100 * t)
    elif normalized < 0.6:
        t = (normalized - 0.4) / 0.2
        r, g, b = int(255 * t), 255, int(100 * (1 - t) + 50 * t)
    elif normalized < 0.8:
        t = (normalized - 0.6) / 0.2
        r, g, b = 255, int(255 * (1 - t) + 150 * t), int(50 * (1 - t))
    else:
        t = (normalized - 0.8) / 0.2
        r, g, b = 255, int(150 * (1 - t) + 50 * t), 0

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)


def get_electroneg_color(
    electroneg: float, fade: float = 0.0
) -> Tuple[int, int, int, int]:
    """Colour gradient for electronegativity.

    Parameters
    ----------
    electroneg : float
        Pauling electronegativity value.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    if electroneg == 0:
        r, g, b = 100, 100, 100
    else:
        normalized = electroneg / 4.0
        r = int(100 + 155 * normalized)
        g = int(150 - 50 * normalized)
        b = int(255 - 155 * normalized)

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)


def get_melting_color(
    melting: float, fade: float = 0.0
) -> Tuple[int, int, int, int]:
    """Colour gradient for melting point.

    Parameters
    ----------
    melting : float
        Melting point in Kelvin.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    normalized = min(melting / 4000.0, 1.0)

    if normalized < 0.33:
        t = normalized / 0.33
        r, g, b = int(100 + 55 * t), int(100 + 100 * t), 255
    elif normalized < 0.67:
        t = (normalized - 0.33) / 0.34
        r, g, b = int(155 + 100 * t), int(200 - 50 * t), int(255 - 155 * t)
    else:
        t = (normalized - 0.67) / 0.33
        r, g, b = 255, int(150 + 50 * t), int(100 - 100 * t)

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)


def get_radius_color(
    radius: float, fade: float = 0.0
) -> Tuple[int, int, int, int]:
    """Colour gradient for atomic radius.

    Parameters
    ----------
    radius : float
        Atomic radius in pm.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    normalized = max(0.0, min(1.0, (radius - 30) / 320))

    if normalized < 0.5:
        t = normalized / 0.5
        r, g, b = int(150 + 105 * t), int(100 + 100 * t), 255
    else:
        t = (normalized - 0.5) / 0.5
        r, g, b = 255, int(200 - 50 * t), int(255 - 155 * t)

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)


def get_density_color(
    density: float, fade: float = 0.0
) -> Tuple[int, int, int, int]:
    """Colour gradient for density (log scale).

    Parameters
    ----------
    density : float
        Density in g/cm^3.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    log_density = math.log10(max(density, 0.0001))
    normalized = max(0.0, min(1.0, (log_density + 4) / 5.3))

    if normalized < 0.33:
        t = normalized / 0.33
        r, g, b = int(50 + 100 * t), int(50 + 150 * t), 255
    elif normalized < 0.67:
        t = (normalized - 0.33) / 0.34
        r, g, b = int(150 + 105 * t), int(200 - 50 * t), int(255 - 100 * t)
    else:
        t = (normalized - 0.67) / 0.33
        r, g, b = 255, int(150 + 50 * t), int(155 - 155 * t)

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)


def get_electron_affinity_color(
    affinity: float, fade: float = 0.0
) -> Tuple[int, int, int, int]:
    """Colour gradient for electron affinity.

    Parameters
    ----------
    affinity : float
        Electron affinity in kJ/mol.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    normalized = max(0.0, min(1.0, (affinity + 10) / 360))

    if normalized < 0.5:
        t = normalized / 0.5
        r, g, b = int(100 + 100 * t), int(100 + 100 * t), 255
    else:
        t = (normalized - 0.5) / 0.5
        r, g, b = int(200 + 55 * t), int(200 - 100 * t), int(255 - 155 * t)

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)


def get_boiling_color(
    boiling: float, fade: float = 0.0
) -> Tuple[int, int, int, int]:
    """Colour gradient for boiling point.

    Parameters
    ----------
    boiling : float
        Boiling point in Kelvin.
    fade : float
        0.0 = opaque, 1.0 = transparent.

    Returns
    -------
    (r, g, b, a) : tuple of int
    """
    normalized = min(boiling / 4000.0, 1.0)

    if normalized < 0.33:
        t = normalized / 0.33
        r, g, b = int(100 + 55 * t), int(150 + 50 * t), 255
    elif normalized < 0.67:
        t = (normalized - 0.33) / 0.34
        r, g, b = int(155 + 100 * t), int(200 - 50 * t), int(255 - 155 * t)
    else:
        t = (normalized - 0.67) / 0.33
        r, g, b = 255, int(150 + 50 * t), int(100 - 100 * t)

    a = int(255 * (1.0 - fade))
    return (r, g, b, a)
