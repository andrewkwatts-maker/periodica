"""
Bonding Rules Engine
=====================
Encodes valence/octet rules, electronegativity-based bond classification,
and common bonding patterns for combinatorial molecule generation.
"""

from typing import Dict, List, Optional, Tuple

from periodica.utils.logger import get_logger

logger = get_logger('bonding_rules')


# Standard valence electrons by group for main-group elements
_GROUP_VALENCE = {
    1: 1,   # H, Li, Na, K...
    2: 2,   # Be, Mg, Ca...
    13: 3,  # B, Al, Ga...
    14: 4,  # C, Si, Ge...
    15: 5,  # N, P, As...
    16: 6,  # O, S, Se...
    17: 7,  # F, Cl, Br...
    18: 8,  # Noble gases (0 for bonding purposes)
}

# Common oxidation states / max bonds for elements (override table)
_MAX_BONDS = {
    'H': 1, 'He': 0,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Ne': 0,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 0,
    'K': 1, 'Ca': 2, 'Br': 1, 'I': 1,
    'Fe': 3, 'Cu': 2, 'Zn': 2, 'Ag': 1, 'Au': 3, 'Pt': 4,
    'Ti': 4, 'Cr': 6, 'Mn': 7, 'Co': 3, 'Ni': 2,
}

# Typical bonding valence (not the same as max bonds; this is common stable bonds)
_TYPICAL_BONDS = {
    'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
    'S': 2, 'P': 3, 'Si': 4, 'B': 3, 'Li': 1, 'Na': 1, 'K': 1,
    'Ca': 2, 'Mg': 2, 'Al': 3, 'Be': 2,
}

# Electronegativity values (Pauling scale) for bond type classification
_ELECTRONEGATIVITIES = {
    'H': 2.20, 'He': 0, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0,
    'Rb': 0.82, 'Sr': 0.95, 'Ag': 1.93, 'I': 2.66, 'Cs': 0.79, 'Ba': 0.89,
    'Au': 2.54, 'Pt': 2.28, 'Pb': 2.33,
}


class BondingRulesEngine:
    """Encodes chemical bonding rules for molecule generation."""

    @staticmethod
    def get_valence(symbol: str) -> int:
        """Get the typical bonding valence for an element."""
        return _TYPICAL_BONDS.get(symbol, 2)

    @staticmethod
    def get_max_bonds(symbol: str) -> int:
        """Get the maximum number of bonds an element can form."""
        return _MAX_BONDS.get(symbol, 2)

    @staticmethod
    def get_electronegativity(symbol: str) -> float:
        """Get Pauling electronegativity."""
        return _ELECTRONEGATIVITIES.get(symbol, 1.5)

    @staticmethod
    def can_bond(elem1: str, elem2: str) -> bool:
        """Check if two elements can form a bond."""
        # Noble gases don't bond (simplified)
        if elem1 in ('He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'):
            return False
        if elem2 in ('He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'):
            return False

        # Both must have nonzero bonding capacity
        v1 = BondingRulesEngine.get_valence(elem1)
        v2 = BondingRulesEngine.get_valence(elem2)
        return v1 > 0 and v2 > 0

    @staticmethod
    def get_bond_type(elem1: str, elem2: str) -> str:
        """
        Classify bond type based on electronegativity difference.

        Returns: 'Ionic', 'Polar Covalent', or 'Covalent'
        """
        en1 = BondingRulesEngine.get_electronegativity(elem1)
        en2 = BondingRulesEngine.get_electronegativity(elem2)
        delta = abs(en1 - en2)

        if delta > 1.7:
            return "Ionic"
        elif delta > 0.4:
            return "Polar Covalent"
        else:
            return "Covalent"

    @staticmethod
    def satisfies_octet(symbol: str, bond_count: int) -> bool:
        """Check if an atom satisfies the octet rule with the given bond count."""
        max_b = BondingRulesEngine.get_max_bonds(symbol)
        typical = BondingRulesEngine.get_valence(symbol)

        if symbol == 'H':
            return bond_count == 1  # Duet rule
        if symbol in ('He', 'Ne', 'Ar'):
            return bond_count == 0

        return bond_count <= max_b and bond_count >= 1

    @staticmethod
    def get_formula(composition: Dict[str, int]) -> str:
        """
        Generate molecular formula using Hill system:
        C first, H second, then alphabetical.
        """
        parts = []

        # Hill system ordering
        if 'C' in composition:
            count = composition['C']
            parts.append(f"C{count}" if count > 1 else "C")
            if 'H' in composition:
                count = composition['H']
                parts.append(f"H{count}" if count > 1 else "H")

        for symbol in sorted(composition.keys()):
            if symbol in ('C', 'H') and 'C' in composition:
                continue
            count = composition[symbol]
            parts.append(f"{symbol}{count}" if count > 1 else symbol)

        return ''.join(parts)

    @staticmethod
    def get_valid_binary_ratios(elem1: str, elem2: str) -> List[Tuple[int, int]]:
        """
        Get valid stoichiometric ratios for a binary compound.

        Returns list of (count_elem1, count_elem2) tuples.
        """
        if not BondingRulesEngine.can_bond(elem1, elem2):
            return []

        v1 = BondingRulesEngine.get_valence(elem1)
        v2 = BondingRulesEngine.get_valence(elem2)

        if v1 == 0 or v2 == 0:
            return []

        ratios = []
        # Simple cross-valence ratio: X_v2 Y_v1
        # e.g., H(1) + O(2) → H2O (2:1)
        import math
        gcd = math.gcd(v1, v2)
        n1 = v2 // gcd
        n2 = v1 // gcd

        if n1 + n2 <= 8:  # Keep molecules small
            ratios.append((n1, n2))

        # Also try 1:1 if both are metals or same valence
        if v1 == v2 and (1, 1) not in ratios:
            ratios.append((1, 1))

        return ratios
