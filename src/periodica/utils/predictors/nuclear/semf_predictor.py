"""
SEMF Nuclear Predictor Module
=============================

Implements the Semi-Empirical Mass Formula (SEMF) for predicting nuclear properties.

The SEMF, also known as the Bethe-Weizsacker formula, models the nuclear binding
energy as a sum of five terms:
1. Volume term: proportional to A (bulk nuclear matter)
2. Surface term: proportional to A^(2/3) (surface tension correction)
3. Coulomb term: electrostatic repulsion between protons
4. Asymmetry term: penalty for neutron-proton imbalance
5. Pairing term: accounts for spin pairing effects

References:
- C.F. von Weizsacker, Z. Phys. 96, 431 (1935)
- H.A. Bethe and R.F. Bacher, Rev. Mod. Phys. 8, 82 (1936)
- Nuclear Physics in a Nutshell (Bertulani)
"""

import math
from typing import Any, Tuple

from ..base import BaseNuclearPredictor, NuclearInput, NuclearResult
from ..registry import register_predictor


@register_predictor('nuclear', 'semf')
@register_predictor('nuclear', 'default')
class SEMFNuclearPredictor(BaseNuclearPredictor):
    """
    Nuclear predictor using the Semi-Empirical Mass Formula (SEMF).

    This predictor calculates nuclear binding energy, radius, mass, and stability
    using empirical formulas calibrated to experimental nuclear data.

    Attributes:
        VOLUME: Volume term coefficient (MeV)
        SURFACE: Surface term coefficient (MeV)
        COULOMB: Coulomb term coefficient (MeV)
        ASYMMETRY: Asymmetry term coefficient (MeV)
        PAIRING: Pairing term coefficient (MeV)
        R0: Nuclear radius constant (fm)
        MAGIC_NUMBERS: Set of magic numbers for shell closures
    """

    # SEMF coefficients (MeV) - standard Bethe-Weizsacker values
    VOLUME = 15.56
    SURFACE = 17.23
    COULOMB = 0.7
    ASYMMETRY = 23.285
    PAIRING = 12.0

    # Nuclear constants
    R0 = 1.25  # fm, nuclear radius constant
    PROTON_MASS_MEV = 938.272
    NEUTRON_MASS_MEV = 939.565

    # Magic numbers (closed shells) - correspond to complete nuclear shells
    MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

    @property
    def name(self) -> str:
        """Return the name of this predictor."""
        return "SEMF Nuclear Predictor"

    @property
    def description(self) -> str:
        """Return a description of what this predictor does."""
        return (
            "Predicts nuclear properties using the Semi-Empirical Mass Formula "
            "(SEMF/Bethe-Weizsacker formula). Calculates binding energy, nuclear "
            "radius, and stability based on proton and neutron counts."
        )

    def calculate_binding_energy(self, Z: int, N: int) -> float:
        """
        Calculate binding energy using Semi-Empirical Mass Formula.

        B = a_v*A - a_s*A^(2/3) - a_c*Z(Z-1)/A^(1/3) - a_a*(N-Z)^2/A + delta

        Args:
            Z: Atomic number (proton count)
            N: Neutron number

        Returns:
            Binding energy in MeV (always non-negative)
        """
        A = Z + N
        if A == 0:
            return 0.0

        # Volume term - proportional to number of nucleons
        volume = self.VOLUME * A

        # Surface term - correction for nucleons at the surface
        surface = self.SURFACE * (A ** (2/3))

        # Coulomb term - electrostatic repulsion between protons
        coulomb = self.COULOMB * Z * (Z - 1) / (A ** (1/3)) if A > 0 else 0

        # Asymmetry term - penalty for N != Z
        asymmetry = self.ASYMMETRY * ((N - Z) ** 2) / A

        # Pairing term - depends on even/odd nucleon counts
        delta = self._calculate_pairing(Z, N, A)

        # Shell correction at magic numbers
        shell = self._calculate_shell_correction(Z, N)

        # Wigner term for light nuclei with N ≈ Z
        wigner = self._calculate_wigner_term(Z, N, A)

        binding = volume - surface - coulomb - asymmetry + delta + shell + wigner
        return max(0.0, binding)

    def _calculate_pairing(self, Z: int, N: int, A: int) -> float:
        """
        Calculate pairing term based on even/odd nucleon numbers.

        Args:
            Z: Proton number
            N: Neutron number
            A: Mass number

        Returns:
            Pairing correction in MeV:
            - Positive for even-even nuclei (extra stability)
            - Negative for odd-odd nuclei (less stability)
            - Zero for odd-even nuclei
        """
        if A == 0:
            return 0.0

        delta_value = self.PAIRING / (A ** 0.5)

        if Z % 2 == 0 and N % 2 == 0:
            return delta_value  # Even-even: extra stability
        elif Z % 2 == 1 and N % 2 == 1:
            return -delta_value  # Odd-odd: less stable
        else:
            return 0.0  # Odd-even: no correction

    def _calculate_shell_correction(self, Z: int, N: int) -> float:
        """
        Calculate shell correction energy at magic numbers.

        Nuclei near closed shells have extra binding modeled as Gaussian
        peaks centered on each magic number.

        Model constants: c = 2.5 MeV, w = 2.5
        """
        c = 2.5
        w = 2.5
        correction = 0.0
        for M in self.MAGIC_NUMBERS:
            correction += c * math.exp(-((N - M) ** 2) / (2 * w ** 2))
            correction += c * math.exp(-((Z - M) ** 2) / (2 * w ** 2))
        return correction

    def _calculate_wigner_term(self, Z: int, N: int, A: int) -> float:
        """
        Calculate Wigner term (asymmetry-like correction).

        Standard formulation: W_term = -W * |N - Z| / A
        This penalizes neutron-proton imbalance, primarily significant
        for light nuclei. For N=Z nuclei, the term vanishes.

        Model constant: W = 30 MeV
        """
        W = 30.0
        if A == 0:
            return 0.0
        return -W * abs(N - Z) / A

    def calculate_radius(self, A: int) -> float:
        """
        Calculate nuclear radius using R = R0 * A^(1/3).

        This empirical formula reflects the approximately constant nuclear
        density, where the radius scales with the cube root of the mass number.

        Args:
            A: Mass number (total nucleons)

        Returns:
            Nuclear radius in femtometers (fm)
        """
        if A <= 0:
            return 0.0
        return self.R0 * (A ** (1/3))

    def calculate_nuclear_mass(self, Z: int, N: int, binding: float) -> float:
        """
        Calculate nuclear mass from constituent masses minus binding energy.

        M_nucleus = Z*m_p + N*m_n - B/c^2

        Args:
            Z: Proton number
            N: Neutron number
            binding: Binding energy in MeV

        Returns:
            Nuclear mass in MeV/c^2
        """
        return Z * self.PROTON_MASS_MEV + N * self.NEUTRON_MASS_MEV - binding

    def check_stability(self, Z: int, N: int) -> Tuple[bool, str]:
        """
        Check if nucleus is stable based on multiple criteria.

        Stability is determined by:
        1. Magic numbers (closed shells)
        2. N/Z ratio compared to beta stability valley
        3. Alpha instability for heavy nuclei

        Args:
            Z: Atomic number (proton count)
            N: Neutron number

        Returns:
            Tuple of (is_stable, stability_reason)
        """
        A = Z + N

        # Check for magic numbers (closed shells provide extra stability)
        proton_magic = Z in self.MAGIC_NUMBERS
        neutron_magic = N in self.MAGIC_NUMBERS

        if proton_magic and neutron_magic:
            return True, "Doubly magic nucleus"

        # Check N/Z ratio for beta stability
        # The optimal ratio increases with A due to Coulomb repulsion
        if A <= 40:
            optimal_ratio = 1.0
            tolerance = 0.15
        elif A <= 100:
            optimal_ratio = 1.0 + 0.01 * (A - 40)
            tolerance = 0.12
        else:
            optimal_ratio = 1.0 + 0.01 * 60 + 0.015 * (A - 100)
            tolerance = 0.10

        actual_ratio = N / Z if Z > 0 else float('inf')

        if abs(actual_ratio - optimal_ratio) > tolerance * optimal_ratio:
            if actual_ratio > optimal_ratio:
                return False, "Beta- unstable (neutron-rich)"
            else:
                return False, "Beta+ unstable (proton-rich)"

        # Check for alpha instability in heavy nuclei
        if A > 150 and Z > 82:
            return False, "Alpha unstable (superheavy)"

        if proton_magic or neutron_magic:
            return True, "Magic number stability"

        return True, "Within stability valley"

    def predict(self, input_data: NuclearInput) -> NuclearResult:
        """
        Make nuclear predictions from input data.

        Calculates all nuclear properties for the given Z and N values,
        including binding energy, radius, mass, and stability.

        Args:
            input_data: NuclearInput with Z (protons) and N (neutrons)

        Returns:
            NuclearResult with all calculated nuclear properties
        """
        Z = input_data.Z
        N = input_data.N
        A = input_data.A

        # Calculate binding energy
        binding = self.calculate_binding_energy(Z, N)
        binding_per_nucleon = binding / A if A > 0 else 0.0

        # Calculate nuclear radius
        radius = self.calculate_radius(A)

        # Calculate nuclear mass
        nuclear_mass = self.calculate_nuclear_mass(Z, N, binding)

        # Check stability
        is_stable, stability_reason = self.check_stability(Z, N)

        # Identify magic numbers present
        magic_present = []
        if Z in self.MAGIC_NUMBERS:
            magic_present.append(Z)
        if N in self.MAGIC_NUMBERS:
            magic_present.append(N)

        return NuclearResult(
            binding_energy_mev=binding,
            binding_per_nucleon_mev=binding_per_nucleon,
            nuclear_radius_fm=radius,
            nuclear_mass_mev=nuclear_mass,
            is_stable=is_stable,
            stability_reason=stability_reason,
            magic_numbers=magic_present if magic_present else None
        )

    def get_confidence(self, input_data: Any, result: Any) -> float:
        """
        Calculate confidence level for a prediction.

        Confidence is based on how well the SEMF typically performs
        for nuclei in different mass regions.

        Args:
            input_data: The NuclearInput used for prediction
            result: The NuclearResult from prediction

        Returns:
            Confidence level between 0.0 and 1.0
        """
        if not isinstance(input_data, NuclearInput):
            return 0.0

        A = input_data.A
        Z = input_data.Z

        # SEMF works best for medium to heavy nuclei
        # Light nuclei have significant shell effects not captured by SEMF
        if A < 20:
            base_confidence = 0.5  # Lower confidence for light nuclei
        elif A < 60:
            base_confidence = 0.85  # Good confidence for medium nuclei
        elif A < 200:
            base_confidence = 0.95  # Best confidence for heavy nuclei
        else:
            base_confidence = 0.7  # Lower confidence for superheavy

        # Magic number nuclei have higher shell effects
        if Z in self.MAGIC_NUMBERS or input_data.N in self.MAGIC_NUMBERS:
            base_confidence *= 0.9  # Slight reduction due to shell effects

        return min(1.0, base_confidence)
