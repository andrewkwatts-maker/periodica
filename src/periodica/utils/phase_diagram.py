"""
Binary Phase Diagram Approximation
====================================
Uses Hume-Rothery rules to predict solid solubility and
eutectic behavior for binary alloy systems.
"""

import math
from typing import Dict, Optional, Tuple

from periodica.utils.logger import get_logger

logger = get_logger('phase_diagram')

# Atomic radii (pm) for Hume-Rothery size factor check
_ATOMIC_RADII = {
    'Li': 152, 'Be': 112, 'Na': 186, 'Mg': 160, 'Al': 143, 'Si': 117,
    'K': 227, 'Ca': 197, 'Sc': 162, 'Ti': 147, 'V': 134, 'Cr': 128,
    'Mn': 127, 'Fe': 126, 'Co': 125, 'Ni': 124, 'Cu': 128, 'Zn': 134,
    'Ga': 135, 'Ge': 122, 'As': 119, 'Se': 120, 'Rb': 248, 'Sr': 215,
    'Y': 180, 'Zr': 160, 'Nb': 146, 'Mo': 139, 'Ru': 134, 'Rh': 134,
    'Pd': 137, 'Ag': 144, 'Cd': 151, 'In': 167, 'Sn': 140, 'Sb': 140,
    'Te': 140, 'Cs': 265, 'Ba': 222, 'La': 187, 'Hf': 159, 'Ta': 146,
    'W': 139, 'Re': 137, 'Os': 135, 'Ir': 136, 'Pt': 139, 'Au': 144,
    'Hg': 151, 'Tl': 170, 'Pb': 175, 'Bi': 156, 'Th': 179, 'U': 156,
}

# Crystal structures at room temperature
_CRYSTAL_STRUCTURES = {
    'Li': 'BCC', 'Be': 'HCP', 'Na': 'BCC', 'Mg': 'HCP', 'Al': 'FCC',
    'Si': 'Diamond', 'K': 'BCC', 'Ca': 'FCC', 'Sc': 'HCP', 'Ti': 'HCP',
    'V': 'BCC', 'Cr': 'BCC', 'Mn': 'BCC', 'Fe': 'BCC', 'Co': 'HCP',
    'Ni': 'FCC', 'Cu': 'FCC', 'Zn': 'HCP', 'Ga': 'Orthorhombic',
    'Ge': 'Diamond', 'Sr': 'FCC', 'Y': 'HCP', 'Zr': 'HCP', 'Nb': 'BCC',
    'Mo': 'BCC', 'Ru': 'HCP', 'Rh': 'FCC', 'Pd': 'FCC', 'Ag': 'FCC',
    'Cd': 'HCP', 'In': 'Tetragonal', 'Sn': 'Tetragonal', 'Sb': 'Rhombohedral',
    'Ta': 'BCC', 'W': 'BCC', 'Re': 'HCP', 'Os': 'HCP', 'Ir': 'FCC',
    'Pt': 'FCC', 'Au': 'FCC', 'Pb': 'FCC', 'Bi': 'Rhombohedral',
}

# Hume-Rothery effective valence (metallic bonding valence, not total electrons)
# For transition metals, use the s-electron count (typically 1-2) since
# d-electrons participate in metallic bonding collectively.
_VALENCE_ELECTRONS = {
    'Li': 1, 'Be': 2, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4,
    'K': 1, 'Ca': 2, 'Ti': 2, 'V': 2, 'Cr': 1, 'Mn': 2,
    'Fe': 2, 'Co': 2, 'Ni': 2, 'Cu': 1, 'Zn': 2, 'Ag': 1,
    'Au': 1, 'Pt': 1, 'Pd': 0, 'Mo': 1, 'W': 2, 'Nb': 1,
    'Ta': 2, 'Sn': 4, 'Pb': 4,
}

# Set of transition metals (d-block) — valence rule is relaxed for TM-TM pairs
_TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
}

# Melting points (K)
_MELTING_POINTS = {
    'Li': 454, 'Be': 1560, 'Na': 371, 'Mg': 923, 'Al': 933, 'Si': 1687,
    'K': 337, 'Ca': 1115, 'Ti': 1941, 'V': 2183, 'Cr': 2180, 'Mn': 1519,
    'Fe': 1811, 'Co': 1768, 'Ni': 1728, 'Cu': 1358, 'Zn': 693, 'Ga': 303,
    'Ge': 1211, 'Zr': 2128, 'Nb': 2750, 'Mo': 2896, 'Ru': 2607, 'Rh': 2237,
    'Pd': 1828, 'Ag': 1235, 'Cd': 594, 'In': 430, 'Sn': 505, 'Sb': 904,
    'Hf': 2506, 'Ta': 3290, 'W': 3695, 'Re': 3459, 'Os': 3306, 'Ir': 2719,
    'Pt': 2041, 'Au': 1337, 'Pb': 601, 'Bi': 545,
}


class BinaryPhaseDiagram:
    """Approximate binary phase diagram predictions using Hume-Rothery rules."""

    @staticmethod
    def get_size_factor(elem1: str, elem2: str) -> float:
        """Calculate size factor: |r1 - r2| / r_larger * 100."""
        r1 = _ATOMIC_RADII.get(elem1, 140)
        r2 = _ATOMIC_RADII.get(elem2, 140)
        return abs(r1 - r2) / max(r1, r2) * 100

    @staticmethod
    def same_crystal_structure(elem1: str, elem2: str) -> bool:
        """Check if two elements share the same crystal structure."""
        cs1 = _CRYSTAL_STRUCTURES.get(elem1, 'Unknown')
        cs2 = _CRYSTAL_STRUCTURES.get(elem2, 'Unknown')
        return cs1 == cs2 and cs1 != 'Unknown'

    @staticmethod
    def is_solid_solution(elem1: str, elem2: str) -> bool:
        """
        Predict if two elements form a complete solid solution
        using Hume-Rothery rules.

        Rules:
        1. Size factor < 15%
        2. Same crystal structure
        3. Similar electronegativity
        4. Similar valence
        """
        # Rule 1: Size factor
        size_factor = BinaryPhaseDiagram.get_size_factor(elem1, elem2)
        if size_factor > 15:
            return False

        # Rule 2: Crystal structure
        if not BinaryPhaseDiagram.same_crystal_structure(elem1, elem2):
            return False

        # Rule 3: Electronegativity (allow up to 0.4, relaxed to 0.7 for TM pairs)
        from periodica.utils.bonding_rules import BondingRulesEngine
        en1 = BondingRulesEngine.get_electronegativity(elem1)
        en2 = BondingRulesEngine.get_electronegativity(elem2)
        both_tm = elem1 in _TRANSITION_METALS and elem2 in _TRANSITION_METALS
        en_threshold = 0.7 if both_tm else 0.4
        if abs(en1 - en2) > en_threshold:
            return False

        # Rule 4: Valence (should be same or close)
        # Relaxed for transition metal pairs where d-electrons blur valence
        v1 = _VALENCE_ELECTRONS.get(elem1, 0)
        v2 = _VALENCE_ELECTRONS.get(elem2, 0)
        both_tm = elem1 in _TRANSITION_METALS and elem2 in _TRANSITION_METALS
        valence_threshold = 2 if both_tm else 1
        if v1 > 0 and v2 > 0 and abs(v1 - v2) > valence_threshold:
            return False

        return True

    @staticmethod
    def predict_eutectic_temperature(elem1: str, elem2: str) -> float:
        """
        Estimate eutectic temperature for a binary system.

        Uses a simple model: eutectic ≈ lower_melting_point * depression_factor.
        """
        mp1 = _MELTING_POINTS.get(elem1, 1500)
        mp2 = _MELTING_POINTS.get(elem2, 1500)
        lower_mp = min(mp1, mp2)

        # Depression factor based on size mismatch
        size_factor = BinaryPhaseDiagram.get_size_factor(elem1, elem2)
        depression = max(0.6, 1.0 - size_factor / 100)

        return lower_mp * depression

    @staticmethod
    def predict_max_solubility(solute: str, solvent: str) -> float:
        """
        Predict maximum solid solubility (atom%) of solute in solvent.

        Returns a value between 0 and 100.
        """
        if BinaryPhaseDiagram.is_solid_solution(solute, solvent):
            return 100.0

        size_factor = BinaryPhaseDiagram.get_size_factor(solute, solvent)

        # Size factor strongly controls solubility
        if size_factor > 30:
            return 0.5
        elif size_factor > 15:
            return max(1.0, 30 - size_factor)
        else:
            return max(5.0, 50 - size_factor * 2)


class RegularSolutionModel:
    """
    Regular solution model for binary alloy thermodynamics.

    Derives the interaction parameter Omega from Miedema formation enthalpy
    (computed from element data sheets), then calculates free energy of mixing.

    ΔG_mix = nRT(x_A ln x_A + x_B ln x_B) + Ω × x_A × x_B
    Ω = 4 × ΔH_mix(equiatomic)  — derived from Miedema, not hardcoded

    Critical temperature for miscibility gap: T_c = Ω / (2R)
    """

    R = 8.314  # J/(mol·K) — universal gas constant

    @staticmethod
    def calculate_omega(element_data_a: dict, element_data_b: dict) -> float:
        """
        Derive the regular solution interaction parameter Omega from
        Miedema formation enthalpy at equiatomic composition.

        Ω = 4 × ΔH_mix(x=0.5) in J/mol

        Args:
            element_data_a: Element A data dict (from element JSON data sheet)
            element_data_b: Element B data dict (from element JSON data sheet)

        Returns:
            Omega in J/mol, or 0 if Miedema params unavailable
        """
        from periodica.utils.predictors.alloy.rule_of_mixtures import RuleOfMixturesPredictor
        predictor = RuleOfMixturesPredictor()
        dH = predictor.calculate_formation_enthalpy(
            element_data_a, element_data_b, 0.5, 0.5
        )
        if dH is None:
            return 0.0
        # dH is in kJ/mol, Omega = 4 * dH * 1000 (convert to J/mol)
        return 4.0 * dH * 1000.0

    @classmethod
    def free_energy_of_mixing(cls, x_a: float, omega: float, T: float) -> float:
        """
        Calculate Gibbs free energy of mixing per mole.

        ΔG_mix = RT(x_A ln x_A + x_B ln x_B) + Ω × x_A × x_B

        Args:
            x_a: Mole fraction of component A (0 < x_a < 1)
            omega: Interaction parameter in J/mol
            T: Temperature in Kelvin

        Returns:
            Free energy of mixing in J/mol
        """
        x_b = 1.0 - x_a
        if x_a <= 0 or x_a >= 1:
            return 0.0

        ideal = cls.R * T * (x_a * math.log(x_a) + x_b * math.log(x_b))
        excess = omega * x_a * x_b
        return ideal + excess

    @classmethod
    def critical_temperature(cls, omega: float) -> float:
        """
        Calculate critical temperature for miscibility gap.

        T_c = Ω / (2R)
        A miscibility gap exists when Ω > 2RT (i.e., T < T_c).

        Args:
            omega: Interaction parameter in J/mol

        Returns:
            Critical temperature in Kelvin (0 if Ω ≤ 0)
        """
        if omega <= 0:
            return 0.0
        return omega / (2.0 * cls.R)

    @classmethod
    def has_miscibility_gap(cls, omega: float, T: float) -> bool:
        """Check if a miscibility gap exists at temperature T."""
        return omega > 0 and T < cls.critical_temperature(omega)


def lindemann_melting(element_data: dict) -> float:
    """
    Estimate melting temperature using Lindemann criterion.

    T_m ≈ C × θ_D² × M / (f_L² × V_m)

    Simplified form (Gilvarry 1956):
        T_m ≈ 0.032 × θ_D × sqrt(M / V_m)

    where:
        θ_D = Debye temperature (from element data sheet)
        M = atomic mass (from element data sheet)
        V_m = molar volume = M / density (from element data sheet)
        f_L = Lindemann parameter per crystal structure (model constant):
            FCC: 0.125, BCC: 0.15, HCP: 0.135

    Args:
        element_data: Element data dict (from element JSON data sheet)

    Returns:
        Estimated melting temperature in Kelvin, or 0 if data insufficient
    """
    theta_D = element_data.get('debye_temperature_K')
    M = element_data.get('atomic_mass')
    density = element_data.get('density')

    if any(v is None for v in [theta_D, M, density]) or density <= 0:
        return 0.0

    # Molar volume in cm³/mol
    V_m = float(M) / float(density)

    # Lindemann estimate with empirically fitted constant
    # C_L = 1.38 fitted from Fe(1811K), Cu(1358K), Ni(1728K)
    C_L = 1.38
    T_m = C_L * float(theta_D) * math.sqrt(float(M) / V_m)
    return round(T_m, 0)
