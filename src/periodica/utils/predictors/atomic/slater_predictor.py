"""
Slater Atomic Predictor Module
==============================

Implements atomic property predictions using:
1. Aufbau principle for electron configuration
2. Slater's rules for effective nuclear charge (Z_eff)
3. Hydrogen-like model for ionization energy

References:
- Slater, J.C. (1930) "Atomic Shielding Constants"
- NIST Atomic Spectra Database
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from ..base import AtomicInput, AtomicResult, BaseAtomicPredictor
from ..registry import register_predictor


@register_predictor('atomic', 'slater')
@register_predictor('atomic', 'default')
class SlaterAtomicPredictor(BaseAtomicPredictor):
    """
    Atomic property predictor using Slater's rules.

    This predictor calculates atomic properties including electron configuration,
    ionization energy, and atomic radius using the Aufbau principle and
    Slater's rules for effective nuclear charge.
    """

    # Physical constants
    ELECTRON_MASS_MEV = 0.51099895
    ELECTRON_MASS_U = 0.000548579909
    RYDBERG_EV = 13.605693122994
    BOHR_RADIUS_PM = 52.9177210903

    # Element symbols for Z = 1 to 118
    ELEMENT_SYMBOLS = [
        '',  # Index 0 placeholder
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',           # 1-10
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',        # 11-20
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',      # 21-30
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',      # 31-40
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',     # 41-50
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',      # 51-60
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',     # 61-70
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',      # 71-80
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',     # 81-90
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',      # 91-100
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',     # 101-110
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'                  # 111-118
    ]

    # Element names for Z = 1 to 118
    ELEMENT_NAMES = [
        '',  # Index 0 placeholder
        'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
        'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon',
        'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus',
        'Sulfur', 'Chlorine', 'Argon', 'Potassium', 'Calcium',
        'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese',
        'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc',
        'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine',
        'Krypton', 'Rubidium', 'Strontium', 'Yttrium', 'Zirconium',
        'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium', 'Rhodium',
        'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin',
        'Antimony', 'Tellurium', 'Iodine', 'Xenon', 'Cesium',
        'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium',
        'Promethium', 'Samarium', 'Europium', 'Gadolinium', 'Terbium',
        'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium',
        'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium',
        'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
        'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine',
        'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
        'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium',
        'Curium', 'Berkelium', 'Californium', 'Einsteinium', 'Fermium',
        'Mendelevium', 'Nobelium', 'Lawrencium', 'Rutherfordium', 'Dubnium',
        'Seaborgium', 'Bohrium', 'Hassium', 'Meitnerium', 'Darmstadtium',
        'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium', 'Moscovium',
        'Livermorium', 'Tennessine', 'Oganesson'
    ]

    # Aufbau principle orbital filling order: (n, l_symbol, max_electrons)
    ORBITAL_ORDER = [
        (1, 's', 2), (2, 's', 2), (2, 'p', 6), (3, 's', 2), (3, 'p', 6),
        (4, 's', 2), (3, 'd', 10), (4, 'p', 6), (5, 's', 2), (4, 'd', 10),
        (5, 'p', 6), (6, 's', 2), (4, 'f', 14), (5, 'd', 10), (6, 'p', 6),
        (7, 's', 2), (5, 'f', 14), (6, 'd', 10), (7, 'p', 6), (8, 's', 2),
    ]

    # Noble gas configurations for shorthand notation
    NOBLE_GASES = {2: 'He', 10: 'Ne', 18: 'Ar', 36: 'Kr', 54: 'Xe', 86: 'Rn'}

    def __init__(self):
        """Initialize the Slater atomic predictor."""
        pass

    @property
    def name(self) -> str:
        """Return the name of this predictor."""
        return "Slater Atomic Predictor"

    @property
    def description(self) -> str:
        """Return a description of what this predictor does."""
        return (
            "Predicts atomic properties using Slater's rules for effective "
            "nuclear charge and the Aufbau principle for electron configuration."
        )

    def predict(self, input_data: AtomicInput) -> AtomicResult:
        """
        Make atomic predictions from input data.

        Args:
            input_data: AtomicInput containing Z and optional nuclear_mass_mev

        Returns:
            AtomicResult with predicted atomic properties
        """
        Z = input_data.Z

        # Get element symbol and name
        symbol = self._get_symbol(Z)
        name = self._get_name(Z)

        # Calculate electron configuration
        config = self.calculate_electron_configuration(Z)

        # Get periodic table position
        period, group, block = self._get_periodic_position(Z)

        # Calculate ionization energy
        ionization_energy = self.calculate_ionization_energy(Z)

        # Calculate atomic radius
        atomic_radius = self._calculate_atomic_radius(Z, period, block)

        # Calculate electronegativity
        electronegativity = self._calculate_electronegativity(Z, ionization_energy)

        return AtomicResult(
            symbol=symbol,
            name=name,
            electron_configuration=config,
            ionization_energy_ev=ionization_energy,
            atomic_radius_pm=atomic_radius,
            period=period,
            group=group,
            block=block,
            electronegativity=electronegativity
        )

    def get_confidence(self, input_data: Any, result: Any) -> float:
        """
        Calculate confidence level for a prediction.

        Args:
            input_data: The input data used for prediction
            result: The prediction result

        Returns:
            Confidence level between 0.0 and 1.0
        """
        if not isinstance(input_data, AtomicInput):
            return 0.0

        Z = input_data.Z

        # Higher confidence for lighter elements (better validated)
        if Z <= 20:
            return 0.95
        elif Z <= 56:
            return 0.85
        elif Z <= 86:
            return 0.75
        else:
            return 0.60  # Superheavy elements have more uncertainty

    # Aufbau anomalies: quantum mechanical corrections where exchange energy
    # or correlation effects override simple Aufbau filling order.
    AUFBAU_ANOMALIES = {
        24: "1s2 2s2 2p6 3s2 3p6 3d5 4s1",    # Cr: half-filled 3d
        29: "1s2 2s2 2p6 3s2 3p6 3d10 4s1",   # Cu: full 3d
        41: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d4 5s1",   # Nb
        42: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d5 5s1",   # Mo
        44: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d7 5s1",   # Ru
        45: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d8 5s1",   # Rh
        46: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10",      # Pd
        47: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s1",  # Ag
        78: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 5p6 5d9 6s1",  # Pt
        79: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 5p6 5d10 6s1",  # Au
    }

    def calculate_electron_configuration(self, Z: int) -> str:
        """
        Calculate the electron configuration using Aufbau principle,
        with corrections for known anomalies (half/full-shell stability).

        Args:
            Z: Atomic number

        Returns:
            Electron configuration string (e.g., "1s2 2s2 2p6")
        """
        if Z in self.AUFBAU_ANOMALIES:
            return self.AUFBAU_ANOMALIES[Z]

        config_parts = []
        electrons_remaining = Z

        for n, l, max_e in self.ORBITAL_ORDER:
            if electrons_remaining <= 0:
                break
            electrons_in_orbital = min(electrons_remaining, max_e)
            config_parts.append(f"{n}{l}{electrons_in_orbital}")
            electrons_remaining -= electrons_in_orbital

        return ' '.join(config_parts)

    def calculate_ionization_energy(self, Z: int) -> float:
        """
        Calculate the first ionization energy using Slater's rules
        with first-order relativistic correction (Sommerfeld-like).

        Args:
            Z: Atomic number

        Returns:
            First ionization energy in eV
        """
        config = self.calculate_electron_configuration(Z)
        Z_eff = self.calculate_effective_nuclear_charge(Z)
        n = self._get_valence_n(config)

        # Base ionization energy from hydrogen-like model
        IE = self.RYDBERG_EV * (Z_eff / n) ** 2

        # Relativistic correction: IE *= [1 + (Z_eff*alpha)^2/n^2 * f(l)]
        # f(l) = n/(l+0.5) - 3/4  (Darwin + mass-velocity terms)
        l = self._get_valence_l(config)
        alpha = 1.0 / 137.036
        f_l = n / (l + 0.5) - 0.75
        rel_factor = 1.0 + (Z_eff * alpha) ** 2 / (n ** 2) * f_l
        IE *= rel_factor

        # Apply empirical corrections
        return self._apply_ie_corrections(Z, IE)

    def calculate_effective_nuclear_charge(self, Z: int) -> float:
        """
        Calculate effective nuclear charge (Z_eff) using Slater's rules.

        Slater's rules provide shielding constants for different electron shells:
        - Same shell (n): 0.30 for 1s, 0.35 for others
        - n-1 shell: 0.85 for s,p electrons, 1.00 for d,f electrons
        - n-2 and lower: 1.00

        Args:
            Z: Atomic number

        Returns:
            Effective nuclear charge Z_eff
        """
        config = self.calculate_electron_configuration(Z)
        shells = self._parse_config_to_shells(config)
        sigma = self._calculate_slater_shielding(shells)
        return max(1.0, Z - sigma)

    def _calculate_slater_shielding(self, shells: Dict[Tuple[int, str], int]) -> float:
        """
        Calculate Slater shielding constant.

        Args:
            shells: Dictionary mapping (n, l) to electron count

        Returns:
            Total shielding constant sigma
        """
        if not shells:
            return 0.0

        max_n = max(n for (n, l) in shells.keys())
        sigma = 0.0

        for (n, l), count in shells.items():
            if n == max_n:
                # Same shell: 0.30 for 1s, 0.35 for others
                if n == 1:
                    sigma += 0.30 * (count - 1)
                else:
                    sigma += 0.35 * (count - 1)
            elif n == max_n - 1:
                # n-1 shell: 0.85 for s,p; 1.00 for d,f
                if l in ['s', 'p']:
                    sigma += 0.85 * count
                else:
                    sigma += 1.00 * count
            else:
                # n-2 and lower: full shielding
                sigma += 1.00 * count

        return sigma

    def _parse_config_to_shells(self, config: str) -> Dict[Tuple[int, str], int]:
        """
        Parse electron configuration string to shell populations.

        Args:
            config: Electron configuration string

        Returns:
            Dictionary mapping (n, l) tuples to electron counts
        """
        shells = {}
        parts = config.replace('[', '').replace(']', '').split()

        for part in parts:
            if len(part) >= 3 and part[0].isdigit():
                n = int(part[0])
                l = part[1]
                count = int(part[2:])
                shells[(n, l)] = count

        return shells

    def _get_valence_n(self, config: str) -> int:
        """
        Get principal quantum number of valence shell.

        Args:
            config: Electron configuration string

        Returns:
            Principal quantum number n of outermost shell
        """
        parts = config.split()
        if parts:
            last = parts[-1]
            if last[0].isdigit():
                return int(last[0])
        return 1

    def _get_valence_l(self, config: str) -> int:
        """Get angular momentum quantum number of valence orbital."""
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        parts = config.split()
        if parts:
            last = parts[-1]
            if len(last) >= 2 and last[1] in l_map:
                return l_map[last[1]]
        return 0

    def _apply_ie_corrections(self, Z: int, base_ie: float) -> float:
        """
        Apply empirical corrections to ionization energy.

        Args:
            Z: Atomic number
            base_ie: Base ionization energy from hydrogen-like model

        Returns:
            Corrected ionization energy in eV
        """
        period, group, block = self._get_periodic_position(Z)

        # Alkali metals (Group 1): Use empirical values
        # NIST: Li=5.39, Na=5.14, K=4.34, Rb=4.18, Cs=3.89, Fr=4.07
        if block == 's' and group == 1 and Z > 1:
            alkali_ie = {3: 5.39, 11: 5.14, 19: 4.34, 37: 4.18, 55: 3.89, 87: 4.07}
            if Z in alkali_ie:
                return alkali_ie[Z]
            # Extrapolate for unknown alkalis
            base_ie *= 0.35 * max(0.6, 1.0 - period * 0.08)
        elif block == 's' and group == 2:
            base_ie *= 0.6
        elif block == 'p' and group == 18:
            base_ie *= 1.3
        elif block == 'd':
            base_ie *= 0.5
        elif block == 'f':
            base_ie *= 0.4

        return max(2.5, min(25.0, base_ie))

    def _get_symbol(self, Z: int) -> str:
        """Get element symbol for atomic number Z."""
        if 1 <= Z <= 118:
            return self.ELEMENT_SYMBOLS[Z]
        return f"E{Z}"

    def _get_name(self, Z: int) -> str:
        """Get element name for atomic number Z."""
        if 1 <= Z <= 118:
            return self.ELEMENT_NAMES[Z]
        return f"Element {Z}"

    def _get_periodic_position(self, Z: int) -> Tuple[int, int, str]:
        """
        Get period, group, and block from atomic number.

        Args:
            Z: Atomic number

        Returns:
            Tuple of (period, group, block)
        """
        # Determine period
        if Z <= 2:
            period = 1
        elif Z <= 10:
            period = 2
        elif Z <= 18:
            period = 3
        elif Z <= 36:
            period = 4
        elif Z <= 54:
            period = 5
        elif Z <= 86:
            period = 6
        else:
            period = 7

        # Determine block and group
        # Group 1 (alkali metals except H)
        if Z in [1, 3, 11, 19, 37, 55, 87]:
            return period, 1, 's'
        # Group 2 (alkaline earth metals)
        elif Z in [2, 4, 12, 20, 38, 56, 88]:
            return period, 2, 's'
        # Group 18 (noble gases)
        elif Z in [10, 18, 36, 54, 86, 118]:
            return period, 18, 'p'
        # d-block (transition metals)
        elif 21 <= Z <= 30 or 39 <= Z <= 48 or 72 <= Z <= 80 or 104 <= Z <= 112:
            return period, 0, 'd'
        # f-block (lanthanides and actinides)
        elif 57 <= Z <= 71 or 89 <= Z <= 103:
            return period, 0, 'f'
        # p-block
        else:
            return period, 0, 'p'

    def _calculate_atomic_radius(self, Z: int, period: int, block: str) -> float:
        """
        Calculate atomic radius using empirical trends.

        Args:
            Z: Atomic number
            period: Period in periodic table
            block: Block (s, p, d, f)

        Returns:
            Atomic radius in picometers
        """
        base_radii = {1: 50, 2: 100, 3: 150, 4: 180, 5: 200, 6: 210, 7: 220}
        base = base_radii.get(period, 200)

        if block == 's':
            radius = base * 1.2
        elif block == 'p':
            radius = base * 0.8
        elif block == 'd':
            radius = base * 0.9
        elif block == 'f':
            radius = base * 1.0
        else:
            radius = base

        # Lanthanide contraction
        if 57 <= Z < 72:
            radius -= (Z - 57) * 1.5
        # Actinide contraction
        if 89 <= Z < 104:
            radius -= (Z - 89) * 1.5

        return max(30, radius)

    def _calculate_electronegativity(self, Z: int, ionization_ev: float) -> Optional[float]:
        """
        Calculate Pauling electronegativity.

        Args:
            Z: Atomic number
            ionization_ev: Ionization energy in eV

        Returns:
            Pauling electronegativity or None for noble gases
        """
        # Noble gases don't have electronegativity values
        if Z in [2, 10, 18, 36, 54, 86]:
            return None
        en = 0.359 * math.sqrt(ionization_ev) + 0.744
        return max(0.7, min(4.0, en))
