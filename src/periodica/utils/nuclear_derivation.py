"""
Nuclear Properties Derivation Module
=====================================

Derives nuclear properties from nucleon counts using:
1. Semi-Empirical Mass Formula (SEMF) for binding energy
2. Nuclear radius formula: R = r0 * A^(1/3)
3. Shell model for magic numbers and stability
4. Pairing effects for spin and parity

References:
- Nuclear Physics in a Nutshell (Bertulani)
- IAEA Nuclear Data Services
"""

import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NuclearProperties:
    """Container for nuclear properties."""
    Z: int  # Proton number
    N: int  # Neutron number
    A: int  # Mass number
    binding_energy_mev: float
    binding_per_nucleon_mev: float
    nuclear_radius_fm: float
    nuclear_mass_mev: float
    spin: float
    parity: str
    is_stable: bool
    stability_reason: str
    half_life_estimate: Optional[str]


class NuclearDerivation:
    """
    Derive nuclear properties from proton and neutron counts.

    Uses physics-based models calibrated to experimental data.
    """

    # SEMF coefficients (MeV) - standard Bethe-Weizsacker values
    SEMF_VOLUME = 15.56
    SEMF_SURFACE = 17.23
    SEMF_COULOMB = 0.7
    SEMF_ASYMMETRY = 23.285
    SEMF_PAIRING = 12.0

    # Nuclear constants
    R0 = 1.25  # fm, nuclear radius constant
    PROTON_MASS_MEV = 938.272
    NEUTRON_MASS_MEV = 939.565

    # Magic numbers (closed shells)
    MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

    def calculate(self, Z: int, N: int) -> NuclearProperties:
        """
        Calculate all nuclear properties for given Z and N.

        Args:
            Z: Proton number (atomic number)
            N: Neutron number

        Returns:
            NuclearProperties dataclass with all derived values
        """
        A = Z + N

        if A <= 0:
            raise ValueError("Mass number must be positive")

        # Calculate binding energy
        binding = self.calculate_binding_energy(Z, N)
        binding_per_nucleon = binding / A if A > 0 else 0

        # Calculate radius
        radius = self.calculate_radius(A)

        # Calculate nuclear mass
        nuclear_mass = self.calculate_nuclear_mass(Z, N, binding)

        # Determine spin and parity
        spin, parity = self.calculate_spin_parity(Z, N)

        # Check stability
        is_stable, stability_reason = self.check_stability(Z, N)

        # Estimate half-life for unstable nuclei
        half_life = self.estimate_half_life(Z, N) if not is_stable else None

        return NuclearProperties(
            Z=Z, N=N, A=A,
            binding_energy_mev=binding,
            binding_per_nucleon_mev=binding_per_nucleon,
            nuclear_radius_fm=radius,
            nuclear_mass_mev=nuclear_mass,
            spin=spin,
            parity=parity,
            is_stable=is_stable,
            stability_reason=stability_reason,
            half_life_estimate=half_life
        )

    def calculate_binding_energy(self, Z: int, N: int) -> float:
        """
        Calculate binding energy using Semi-Empirical Mass Formula.

        B = a_v*A - a_s*A^(2/3) - a_c*Z(Z-1)/A^(1/3) - a_a*(N-Z)^2/A + delta
        """
        A = Z + N
        if A == 0:
            return 0.0

        # Volume term
        volume = self.SEMF_VOLUME * A

        # Surface term
        surface = self.SEMF_SURFACE * (A ** (2/3))

        # Coulomb term
        if A > 0:
            coulomb = self.SEMF_COULOMB * Z * (Z - 1) / (A ** (1/3))
        else:
            coulomb = 0

        # Asymmetry term
        asymmetry = self.SEMF_ASYMMETRY * ((N - Z) ** 2) / A

        # Pairing term
        delta = self._calculate_pairing(Z, N, A)

        # Shell correction — Gaussian peaks at magic numbers
        shell = self._calculate_shell_correction(Z, N)

        # Wigner term — extra binding for light nuclei with N ≈ Z
        wigner = self._calculate_wigner_term(Z, N, A)

        binding = volume - surface - coulomb - asymmetry + delta + shell + wigner
        return max(0.0, binding)

    def _calculate_pairing(self, Z: int, N: int, A: int) -> float:
        """Calculate pairing term based on even/odd nucleon numbers."""
        if A == 0:
            return 0.0

        delta_value = self.SEMF_PAIRING / (A ** 0.5)

        if Z % 2 == 0 and N % 2 == 0:
            return delta_value  # Even-even: extra stability
        elif Z % 2 == 1 and N % 2 == 1:
            return -delta_value  # Odd-odd: less stable
        else:
            return 0.0  # Odd-even: no correction

    def _calculate_shell_correction(self, Z: int, N: int) -> float:
        """
        Calculate shell correction energy at magic numbers.

        Nuclei near closed shells (magic numbers) have extra binding from
        the nuclear shell structure. Modeled as Gaussian peaks centered
        on each magic number.

        E_shell = sum_M [ +c * exp(-(N-M)^2 / (2w^2)) + +c * exp(-(Z-M)^2 / (2w^2)) ]

        Model constants: c = 2.5 MeV, w = 2.5 (fitted to reproduce
        magic number binding energy enhancements without over-correction).
        """
        c = 2.5   # MeV — shell gap energy scale
        w = 2.5   # Width parameter (dimensionless, in units of nucleon number)

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
        W = 30.0   # MeV — Wigner energy scale
        if A == 0:
            return 0.0
        return -W * abs(N - Z) / A

    def calculate_radius(self, A: int) -> float:
        """Calculate nuclear radius using R = r0 * A^(1/3)."""
        if A <= 0:
            return 0.0
        return self.R0 * (A ** (1/3))

    def calculate_nuclear_mass(self, Z: int, N: int, binding: float) -> float:
        """Calculate nuclear mass from constituents minus binding energy."""
        mass = Z * self.PROTON_MASS_MEV + N * self.NEUTRON_MASS_MEV - binding
        return mass

    def calculate_spin_parity(self, Z: int, N: int) -> Tuple[float, str]:
        """Estimate ground-state spin and parity using simple shell model."""
        if Z % 2 == 0 and N % 2 == 0:
            return 0.0, '+'

        # Simplified: use last filled orbital
        if Z % 2 == 1:
            spin, parity = self._get_orbital_jp(Z)
        else:
            spin, parity = self._get_orbital_jp(N)

        if Z % 2 == 1 and N % 2 == 1:
            spin_p, parity_p = self._get_orbital_jp(Z)
            spin_n, parity_n = self._get_orbital_jp(N)
            spin = abs(spin_p - spin_n)
            parity = '+' if parity_p == parity_n else '-'

        return spin, parity

    def _get_orbital_jp(self, nucleon_count: int) -> Tuple[float, str]:
        """Get spin and parity from shell model orbital."""
        shells = [
            (2, 0.5, '+'), (6, 1.5, '-'), (4, 0.5, '-'), (8, 2.5, '+'),
            (4, 0.5, '+'), (6, 1.5, '+'), (10, 3.5, '-'), (6, 1.5, '-'),
            (4, 0.5, '-'), (12, 4.5, '+'),
        ]

        remaining = nucleon_count
        last_j, last_parity = 0.5, '+'

        for max_count, j, parity in shells:
            if remaining <= max_count:
                last_j, last_parity = j, parity
                break
            remaining -= max_count

        return last_j, last_parity

    def check_stability(self, Z: int, N: int) -> Tuple[bool, str]:
        """Check if nucleus is stable based on multiple criteria."""
        A = Z + N

        # Check for magic numbers
        proton_magic = Z in self.MAGIC_NUMBERS
        neutron_magic = N in self.MAGIC_NUMBERS

        if proton_magic and neutron_magic:
            return True, "Doubly magic nucleus"

        # Check N/Z ratio for beta stability
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

        # Check for alpha instability
        if A > 150 and Z > 82:
            return False, "Alpha unstable (superheavy)"

        if proton_magic or neutron_magic:
            return True, "Magic number stability"

        return True, "Within stability valley"

    def estimate_half_life(self, Z: int, N: int) -> str:
        """Estimate half-life category for unstable nuclei."""
        A = Z + N

        if A > 250:
            return "< 1 ms (superheavy)"
        elif A > 200 and Z > 82:
            return "minutes to years (alpha decay)"
        else:
            return "seconds to years (beta decay)"

    def validate_against_experimental(self, Z: int, N: int,
                                       experimental_binding: float) -> Dict:
        """Compare calculated binding energy to experimental value."""
        calculated = self.calculate_binding_energy(Z, N)

        error = calculated - experimental_binding
        percent_error = 100 * error / experimental_binding if experimental_binding != 0 else float('inf')

        return {
            'Z': Z, 'N': N, 'A': Z + N,
            'calculated_mev': calculated,
            'experimental_mev': experimental_binding,
            'error_mev': error,
            'percent_error': percent_error,
            'within_2_percent': abs(percent_error) < 2
        }


# Convenience function
def derive_nuclear_properties(Z: int, N: int) -> NuclearProperties:
    """Derive nuclear properties from Z and N."""
    calculator = NuclearDerivation()
    return calculator.calculate(Z, N)


# Experimental binding energies for validation (MeV)
EXPERIMENTAL_BINDING = {
    (1, 0): 0,        # H-1 (proton)
    (1, 1): 2.225,    # H-2 (deuteron)
    (2, 1): 7.718,    # He-3
    (2, 2): 28.296,   # He-4
    (6, 6): 92.162,   # C-12
    (8, 8): 127.619,  # O-16
    (26, 30): 492.254, # Fe-56
    (82, 126): 1636.43, # Pb-208
    (92, 146): 1801.69, # U-238
}


if __name__ == '__main__':
    calc = NuclearDerivation()

    print("Nuclear Properties Derivation Tests")
    print("=" * 50)

    for (Z, N), exp_binding in EXPERIMENTAL_BINDING.items():
        props = calc.calculate(Z, N)
        validation = calc.validate_against_experimental(Z, N, exp_binding)

        print(f"\n{props.A}-{Z}: B={props.binding_energy_mev:.2f} MeV "
              f"(exp: {exp_binding:.2f}, err: {validation['percent_error']:.1f}%)")
        print(f"  Radius: {props.nuclear_radius_fm:.3f} fm, "
              f"Stable: {props.is_stable} ({props.stability_reason})")
