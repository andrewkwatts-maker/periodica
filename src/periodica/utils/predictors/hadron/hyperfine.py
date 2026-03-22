"""
De Rujula-Georgi-Glashow (DRG) Hyperfine Splitting Calculator

Calculates hadron masses from quark compositions using the color-magnetic
hyperfine interaction. All quark properties (constituent masses) are read
from quark data sheets passed in at runtime.

Formula:
    M_hadron = sum(m_i) + M0 + A * sum_{i<j} <sigma_i . sigma_j> / (m_i * m_j)

Where:
    - m_i = constituent quark mass (from quark data sheet ConstituentMass_MeV)
    - M0 = sector-dependent confinement offset (model constant)
    - A = hyperfine coupling constant (model constant, sector-dependent)
    - <sigma_i . sigma_j> = Pauli spin matrix expectation values (spin algebra)

Model constants fitted from:
    - Meson sector: pi-rho mass splitting
    - Baryon sector: proton-Delta mass splitting
"""

from typing import List, Optional


# =============================================================================
# Model Constants (universal, not per-particle)
# =============================================================================

# Confinement offset (MeV) — accounts for different confining potential
# geometry in mesons (linear) vs baryons (Y-string)
M0_MESON = -55.8
M0_BARYON = 77.0

# Hyperfine coupling constants (MeV^3)
# Fitted from known mass splittings:
#   A_MESON from pi(140) - rho(775) splitting
#   A_BARYON from proton(938) - Delta(1232) splitting
A_MESON = 17_922_240.0
A_BARYON = 5_531_904.0

# Mass threshold for considering two quarks as "same flavor" (MeV)
_SAME_FLAVOR_THRESHOLD = 5.0


def get_constituent_mass(quark_data: dict) -> float:
    """
    Read constituent (dressed) quark mass from quark data sheet.

    The constituent mass includes QCD dressing effects and is much larger
    than the current (bare) mass for light quarks. This value must come
    from the quark's JSON data sheet (ConstituentMass_MeV field).

    Args:
        quark_data: Dictionary from quark JSON data sheet

    Returns:
        Constituent mass in MeV/c^2
    """
    const_mass = quark_data.get('ConstituentMass_MeV')
    if const_mass is not None:
        return float(const_mass)

    # Fallback: estimate from current mass if ConstituentMass_MeV not in data
    current_mass = float(quark_data.get('Mass_MeVc2', quark_data.get('mass', 0)))
    if current_mass < 10:
        return 336.0  # Light quark default
    elif current_mass < 200:
        return 509.0  # Strange-like
    elif current_mass < 2000:
        return current_mass + 280  # Charm-like dressing
    else:
        return current_mass + 550  # Bottom-like dressing


def calculate_hyperfine_mass(
    constituent_masses: List[float],
    hadron_type: str,
    total_spin: Optional[float] = None
) -> float:
    """
    Calculate hadron mass using the DRG hyperfine formula.

    M = sum(m_i) + M0 + A * sum_{i<j} <sigma_i . sigma_j> / (m_i * m_j)

    Args:
        constituent_masses: List of constituent quark masses in MeV
            (read from quark data sheets via get_constituent_mass)
        hadron_type: "meson" or "baryon"
        total_spin: Total spin quantum number of the hadron.
            Mesons: 0 (pseudoscalar) or 1 (vector)
            Baryons: 0.5 (octet) or 1.5 (decuplet)
            If None, defaults to ground state (S=0 for mesons, J=1/2 for baryons)

    Returns:
        Predicted hadron mass in MeV/c^2
    """
    n = len(constituent_masses)

    if hadron_type == "meson" and n == 2:
        return _meson_mass(constituent_masses, total_spin)
    elif hadron_type == "baryon" and n == 3:
        return _baryon_mass(constituent_masses, total_spin)
    else:
        # Exotic hadrons (pentaquarks, tetraquarks): sum only, no correction
        return sum(constituent_masses)


def _meson_mass(masses: List[float], spin: Optional[float]) -> float:
    """
    Calculate meson mass with hyperfine splitting.

    For mesons (q q-bar), there is one quark pair:
        S=0 (pseudoscalar): <sigma_1 . sigma_2> = -3
        S=1 (vector):       <sigma_1 . sigma_2> = +1
    """
    m1, m2 = masses
    base = m1 + m2 + M0_MESON

    # Pauli spin matrix expectation value
    if spin is not None and spin >= 0.9:
        sigma_dot_sigma = 1.0   # Vector meson (S=1)
    else:
        sigma_dot_sigma = -3.0  # Pseudoscalar meson (S=0, ground state)

    hyperfine = A_MESON * sigma_dot_sigma / (m1 * m2)
    return base + hyperfine


def _baryon_mass(masses: List[float], spin: Optional[float]) -> float:
    """
    Calculate baryon mass with hyperfine splitting.

    For baryons (qqq), there are three quark pairs. The total spin
    expectation sum_{i<j} <sigma_i . sigma_j> depends on J:
        J=1/2 (octet):    sum = -3
        J=3/2 (decuplet): sum = +3

    For J=1/2 with mixed flavors, the pair-by-pair distribution matters
    when quarks have different masses. For two identical quarks (qqQ):
        <sigma_q . sigma_q> = +1  (identical pair in spin triplet)
        <sigma_q . sigma_Q> = -2  (each mixed pair)
    For all-different flavors (q1 q2 q3):
        <sigma_i . sigma_j> = -1  (equal sharing among pairs)
    """
    m1, m2, m3 = masses
    base = m1 + m2 + m3 + M0_BARYON

    if spin is not None and spin >= 1.4:
        # J=3/2 decuplet: all pairs aligned, <sigma_i . sigma_j> = +1
        hyperfine = A_BARYON * (
            1.0 / (m1 * m2) + 1.0 / (m1 * m3) + 1.0 / (m2 * m3)
        )
    else:
        # J=1/2 octet: need pair-by-pair for different masses
        ms = sorted(masses)

        if abs(ms[0] - ms[1]) < _SAME_FLAVOR_THRESHOLD:
            # Two lighter quarks are same flavor (e.g., uud, uus)
            m_same = ms[0]
            m_diff = ms[2]
            hyperfine = A_BARYON * (
                1.0 / (m_same * m_same)
                - 2.0 / (m_same * m_diff)
                - 2.0 / (m_same * m_diff)
            )
        elif abs(ms[1] - ms[2]) < _SAME_FLAVOR_THRESHOLD:
            # Two heavier quarks are same flavor (e.g., uss, dss)
            m_same = ms[1]
            m_diff = ms[0]
            hyperfine = A_BARYON * (
                1.0 / (m_same * m_same)
                - 2.0 / (m_same * m_diff)
                - 2.0 / (m_same * m_diff)
            )
        else:
            # All different flavors (e.g., uds for Lambda)
            # Equal sharing: each pair gets <sigma_i . sigma_j> = -1
            hyperfine = A_BARYON * (
                -1.0 / (ms[0] * ms[1])
                - 1.0 / (ms[0] * ms[2])
                - 1.0 / (ms[1] * ms[2])
            )

    return base + hyperfine
