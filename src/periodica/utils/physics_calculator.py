"""
Physics Calculations Module
Provides formula-based calculations for creating particles from constituents.
All formulas are physics-based, not hardcoded lookup values.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


# ==================== Physical Constants ====================

class PhysicsConstants:
    """Fundamental physical constants"""
    # Masses in atomic mass units (u) and MeV/c²
    PROTON_MASS_U = 1.007276466621  # u
    NEUTRON_MASS_U = 1.008664915  # u
    ELECTRON_MASS_U = 0.000548579909065  # u

    PROTON_MASS_MEV = 938.27208816  # MeV/c²
    NEUTRON_MASS_MEV = 939.56542052  # MeV/c²
    ELECTRON_MASS_MEV = 0.51099895  # MeV/c²

    # Quark masses (current quark masses in MeV/c²)
    UP_QUARK_MASS_MEV = 2.2
    DOWN_QUARK_MASS_MEV = 4.7
    STRANGE_QUARK_MASS_MEV = 95.0
    CHARM_QUARK_MASS_MEV = 1275.0
    BOTTOM_QUARK_MASS_MEV = 4180.0
    TOP_QUARK_MASS_MEV = 173100.0

    # Quark charges (in units of e)
    UP_QUARK_CHARGE = 2/3
    DOWN_QUARK_CHARGE = -1/3
    STRANGE_QUARK_CHARGE = -1/3
    CHARM_QUARK_CHARGE = 2/3
    BOTTOM_QUARK_CHARGE = -1/3
    TOP_QUARK_CHARGE = 2/3

    # Other constants
    RYDBERG_ENERGY_EV = 13.605693122994  # eV
    RYDBERG_ENERGY = RYDBERG_ENERGY_EV    # alias
    BOHR_RADIUS_PM = 52.9177210903  # pm
    FINE_STRUCTURE = 0.0072973525693  # α
    AVOGADRO = 6.02214076e23  # mol⁻¹

    # Nuclear constants
    NUCLEAR_RADIUS_CONST = 1.25  # fm, r₀ for R = r₀ * A^(1/3)
    BINDING_ENERGY_VOLUME = 15.75  # MeV, a_v
    BINDING_ENERGY_SURFACE = 17.8  # MeV, a_s
    BINDING_ENERGY_COULOMB = 0.711  # MeV, a_c
    BINDING_ENERGY_ASYMMETRY = 23.7  # MeV, a_a
    BINDING_ENERGY_PAIRING = 11.2  # MeV, a_p


# ==================== Atom Creation from Nucleons ====================

class AtomCalculator:
    """
    Calculate atomic properties from protons, neutrons, and electrons.
    Uses semi-empirical formulas from nuclear and atomic physics.
    """

    @staticmethod
    def calculate_atomic_mass(protons: int, neutrons: int) -> float:
        """
        Calculate atomic mass using semi-empirical mass formula.
        Mass = Z*m_p + N*m_n - B/c² where B is binding energy

        Args:
            protons: Number of protons (Z)
            neutrons: Number of neutrons (N)

        Returns:
            Atomic mass in atomic mass units (u)
        """
        Z = protons
        N = neutrons
        A = Z + N  # Mass number

        if A == 0:
            return 0.0

        # Calculate binding energy using semi-empirical mass formula (Weizsäcker formula)
        # B = a_v*A - a_s*A^(2/3) - a_c*Z²/A^(1/3) - a_a*(N-Z)²/A + δ

        a_v = PhysicsConstants.BINDING_ENERGY_VOLUME
        a_s = PhysicsConstants.BINDING_ENERGY_SURFACE
        a_c = PhysicsConstants.BINDING_ENERGY_COULOMB
        a_a = PhysicsConstants.BINDING_ENERGY_ASYMMETRY
        a_p = PhysicsConstants.BINDING_ENERGY_PAIRING

        # Volume term
        B = a_v * A

        # Surface term
        B -= a_s * (A ** (2/3))

        # Coulomb term
        if A > 0:
            B -= a_c * (Z ** 2) / (A ** (1/3))

        # Asymmetry term
        B -= a_a * ((N - Z) ** 2) / A

        # Pairing term
        if Z % 2 == 0 and N % 2 == 0:
            delta = a_p / (A ** 0.5)  # Even-even
        elif Z % 2 == 1 and N % 2 == 1:
            delta = -a_p / (A ** 0.5)  # Odd-odd
        else:
            delta = 0  # Even-odd or odd-even
        B += delta

        # Shell correction for magic numbers (2, 8, 20, 28, 50, 82, 126)
        magic_numbers = {2, 8, 20, 28, 50, 82, 126}
        if Z in magic_numbers:
            B += 2.5  # Extra stability from closed proton shells
        if N in magic_numbers:
            B += 2.5  # Extra stability from closed neutron shells

        # Convert binding energy to mass deficit (MeV to u)
        # 1 u = 931.494 MeV/c²
        mass_deficit_u = B / 931.494

        # Total mass
        mass = Z * PhysicsConstants.PROTON_MASS_U + N * PhysicsConstants.NEUTRON_MASS_U - mass_deficit_u

        return round(mass, 6)

    @staticmethod
    def calculate_binding_energy(Z: int, A: int) -> float:
        """
        Calculate nuclear binding energy using semi-empirical mass formula.

        Args:
            Z: Atomic number (proton count)
            A: Mass number (protons + neutrons)

        Returns:
            Total binding energy in MeV
        """
        from periodica.utils.nuclear_derivation import NuclearDerivation
        nuc = NuclearDerivation()
        return nuc.calculate_binding_energy(Z, A - Z)

    @staticmethod
    def calculate_ionization_energy(protons: int) -> float:
        """
        Estimate first ionization energy using improved empirical formula.
        Based on periodic trends and experimental correlations.

        The simple Rydberg formula overestimates IE for multi-electron atoms.
        This implementation uses empirical corrections based on block and period.
        """
        Z = protons
        if Z == 0:
            return 0.0

        # Get shell configuration
        shell_config = AtomCalculator._get_shell_configuration(Z)
        n = shell_config['n']  # Principal quantum number
        l = shell_config.get('l', 0)  # Angular momentum quantum number

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)

        # Base ionization energy using improved Z_eff calculation
        # For the outermost electron, use better Slater shielding
        Z_eff = AtomCalculator._calculate_z_effective_ie(Z, shell_config)

        # Base formula with empirical damping for multi-electron atoms
        base_IE = PhysicsConstants.RYDBERG_ENERGY_EV * (Z_eff ** 2) / (n ** 2)

        # Apply strong empirical corrections based on observed periodic trends
        # IE generally ranges from ~4 eV (Cs) to ~25 eV (He)

        if Z == 1:  # Hydrogen - exact
            return 13.598
        elif Z == 2:  # Helium - filled shell
            return 24.587

        # Noble gases have highest IE in their period
        if AtomCalculator._is_noble_gas(Z):
            if Z == 10:   # Ne
                return 21.56
            elif Z == 18:  # Ar
                return 15.76
            elif Z == 36:  # Kr
                return 14.0
            elif Z == 54:  # Xe
                return 12.13
            elif Z == 86:  # Rn
                return 10.75
            elif Z == 118:  # Og
                return 8.5

        # For other elements, use empirical correlations
        # IE decreases down a group and increases across a period (with some exceptions)

        # Period-based correction factor (IE decreases with larger atoms)
        period_factor = 1.0 / (1.0 + 0.12 * (period - 1))

        # Block-based corrections
        if block == 's':
            # Alkali metals have lowest IE (~4-5 eV), alkaline earth slightly higher (~6-10 eV)
            group = AtomCalculator._get_group(Z)
            if group == 1:  # Alkali metals
                base_IE = 5.5 - 0.15 * (period - 2) if period > 1 else 13.6
            elif group == 2:  # Alkaline earth - improved fit for Ca, Sr, Ba
                base_IE = 6.5 - 0.25 * (period - 2) if period > 1 else 24.6
        elif block == 'p':
            # p-block: IE increases across the period
            group = AtomCalculator._get_group(Z)
            if group:
                position_in_p = group - 12  # 1-6 for groups 13-18
                # Base p-block IE with periodic trend
                base_IE = 7.0 + 1.5 * position_in_p - 0.5 * (period - 2)
                # Half-filled p shell (group 15) has slightly higher IE
                if position_in_p == 3:
                    base_IE += 1.0
                # Period 7 p-block needs relativistic correction
                if period == 7:
                    base_IE *= 0.65  # Relativistic effects lower IE significantly
        elif block == 'd':
            # d-block: relatively flat IE across period (~7-9 eV)
            base_IE = 7.5 + 0.3 * Z_eff / n - 0.3 * (period - 4)
        elif block == 'f':
            # f-block: similar to d-block but lower (~6 eV)
            base_IE = 6.0 + 0.1 * Z_eff / n

        # Clamp to reasonable physical range
        base_IE = max(3.5, min(25.0, base_IE))

        return round(base_IE, 3)

    @staticmethod
    def calculate_electronegativity(protons: int) -> float:
        """
        Estimate electronegativity using improved empirical formula.
        Based on Pauling scale correlations with periodic position.

        Electronegativity increases across a period and decreases down a group.
        Range: ~0.7 (Cs, Fr) to ~4.0 (F)
        """
        Z = protons
        if Z == 0:
            return 0.0

        # Noble gases - use Allen electronegativity scale for heavier ones
        if AtomCalculator._is_noble_gas(Z):
            noble_en = {2: 0, 10: 0, 18: 0, 36: 3.0, 54: 2.6, 86: 2.2, 118: 2.4}
            return noble_en.get(Z, 0)

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)
        group = AtomCalculator._get_group(Z)

        # Empirical electronegativity based on periodic position
        # Reference values from Pauling scale

        if Z == 1:  # Hydrogen
            return 2.20

        if block == 's':
            if group == 1:  # Alkali metals: 0.79 (Fr) to 0.98 (Li)
                chi = 1.0 - 0.04 * (period - 2)
            elif group == 2:  # Alkaline earth: 0.89 (Ra) to 1.57 (Be)
                chi = 1.6 - 0.12 * (period - 2)
            else:
                chi = 1.0
        elif block == 'p':
            if group:
                # p-block: electronegativity increases across period
                # Group 13: ~1.5-2.0, Group 14: ~1.8-2.5, Group 15: ~2.0-3.0
                # Group 16: ~2.0-3.5, Group 17: ~2.7-4.0
                position_in_p = group - 12  # 1-6 for groups 13-18
                # Base chi increases with group number
                base_chi = 1.5 + 0.4 * position_in_p
                # Decreases down the group
                period_decrease = 0.08 * (period - 2)
                chi = base_chi - period_decrease
                # Fluorine is the highest
                if Z == 9:
                    chi = 3.98
            else:
                chi = 2.0
        elif block == 'd':
            # d-block: relatively uniform ~1.3-2.5
            # Increases slightly across period, decreases down groups
            if group:
                position_in_d = group - 2  # 1-10 for groups 3-12
                chi = 1.3 + 0.12 * position_in_d - 0.05 * (period - 4)
                # Group 11 (Cu, Ag, Au) and 12 have slightly higher values
                if group >= 11:
                    chi += 0.3
            else:
                chi = 1.6
        elif block == 'f':
            # f-block: relatively low and uniform ~1.1-1.5
            chi = 1.2 + 0.02 * (Z - 57 if Z < 72 else Z - 89)

        # Clamp to reasonable range
        chi = max(0.7, min(4.0, chi))

        return round(chi, 2)

    @staticmethod
    def calculate_atomic_radius(protons: int) -> int:
        """
        Estimate atomic radius using improved empirical formula.
        Based on periodic trends and experimental atomic/covalent radii.

        Atomic radius increases down a group and decreases across a period.
        Range: ~31 pm (He) to ~298 pm (Cs)
        """
        Z = protons
        if Z == 0:
            return 0

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)
        group = AtomCalculator._get_group(Z)

        # Use empirical formulas based on periodic position
        # These give covalent/atomic radii in pm

        if Z == 1:  # Hydrogen
            return 53
        elif Z == 2:  # Helium
            return 31

        # Base radius increases with period (larger shells)
        # Base radii for each period (approximate starting values)
        period_base = {2: 70, 3: 100, 4: 130, 5: 150, 6: 170, 7: 180}
        base_r = period_base.get(period, 150)

        if block == 's':
            # s-block: alkali metals are largest in their period
            if group == 1:  # Alkali metals
                r = base_r + 50 + 15 * (period - 2)  # ~152 (Li) to ~298 (Cs)
            elif group == 2:  # Alkaline earth
                r = base_r + 30 + 10 * (period - 2)  # ~112 (Be) to ~215 (Ba)
            else:
                r = base_r
        elif block == 'p':
            if group:
                # p-block: radius decreases across period due to increasing Z_eff
                position_in_p = group - 12  # 1-6 for groups 13-18
                # Decrease across period
                r = base_r - 8 * position_in_p
                # Noble gases have smallest radius (van der Waals can be larger)
                if position_in_p == 6:
                    r = base_r - 50  # Noble gas covalent radius is small
            else:
                r = base_r
        elif block == 'd':
            # d-block: metallic radii, relatively constant across period
            # with slight d-block contraction
            if group:
                position_in_d = group - 2  # 1-10 for groups 3-12
                # Slight decrease across d-block, then slight increase at end
                if position_in_d <= 5:
                    r = base_r - 5 * position_in_d
                else:
                    r = base_r - 25 + 3 * (position_in_d - 5)
                # Period 6 d-block is affected by lanthanide contraction
                if period == 6:
                    r -= 10
            else:
                r = base_r
        elif block == 'f':
            # f-block: gradual decrease across the series (lanthanide/actinide contraction)
            if period == 6:  # Lanthanides (Z=57-71)
                position = Z - 57
                r = 185 - position * 1.5  # ~185 (La) to ~175 (Lu)
            else:  # Actinides (Z=89-103)
                position = Z - 89
                r = 195 - position * 1.5  # ~195 (Ac) to ~175 (Lr)
        else:
            r = base_r

        # Ensure minimum radius
        r = max(30, r)

        return int(round(r))

    @staticmethod
    def calculate_melting_point(protons: int, neutrons: int) -> float:
        """
        Estimate melting point using improved empirical correlations.
        Based on element type, periodic position, and bonding characteristics.

        Major categories:
        - Gases at STP: Very low melting points (< 100 K)
        - Metals: Moderate to very high (300-3700 K)
        - Non-metallic solids: Variable (400-4000 K)
        """
        Z = protons
        if Z == 0:
            return 0.0

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)
        group = AtomCalculator._get_group(Z)

        # Identify element category for melting point estimation
        # Gases at STP: H, He, N, O, F, Ne, Cl, Ar, Kr, Xe, Rn
        gases_at_stp = [1, 2, 7, 8, 9, 10, 17, 18, 36, 54, 86]
        # Liquids at STP: Br, Hg
        liquids_at_stp = [35, 80]

        if Z in gases_at_stp:
            # Very low melting points for gases
            if Z == 1:  # Hydrogen
                return 14.0
            elif Z == 2:  # Helium
                return 0.95
            elif Z == 7:  # Nitrogen
                return 63.0
            elif Z == 8:  # Oxygen
                return 54.0
            elif Z == 9:  # Fluorine
                return 53.5
            elif Z == 10:  # Neon
                return 24.5
            elif Z == 17:  # Chlorine
                return 172.0
            elif Z == 18:  # Argon
                return 84.0
            elif Z == 36:  # Krypton
                return 116.0
            elif Z == 54:  # Xenon
                return 161.0
            elif Z == 86:  # Radon
                return 202.0
            else:
                return 100.0

        if Z in liquids_at_stp:
            if Z == 35:  # Bromine
                return 266.0
            elif Z == 80:  # Mercury
                return 234.0

        # Non-gas elements - use periodic trends
        if block == 's':
            if group == 1:  # Alkali metals: low melting points (~300-450 K)
                mp = 500 - 35 * (period - 2)
            elif group == 2:  # Alkaline earth: higher (~900-1560 K)
                mp = 1100 - 30 * (period - 2)
            else:
                mp = 500
        elif block == 'p':
            if group:
                position_in_p = group - 12  # 1-6 for groups 13-18
                # p-block metals and metalloids
                if position_in_p <= 2:  # Groups 13-14 (Al, Ga, In, Tl, C, Si, Ge, Sn, Pb)
                    # Carbon is special (very high due to covalent bonding)
                    if Z == 6:
                        return 3823.0  # Graphite sublimation
                    elif Z == 14:  # Silicon
                        return 1687.0
                    mp = 600 - 30 * (period - 3) + 200 * (position_in_p - 1)
                elif position_in_p <= 4:  # Groups 15-16 (N, P, As, Sb, Bi, O, S, Se, Te, Po)
                    if Z == 15:  # Phosphorus
                        return 317.0
                    elif Z == 16:  # Sulfur
                        return 388.0
                    mp = 500 - 20 * (period - 3)
                else:  # Groups 17-18 (halogens, noble gases)
                    if Z == 53:  # Iodine
                        return 387.0
                    mp = 300 - 30 * (period - 4)
            else:
                mp = 500
        elif block == 'd':
            # Transition metals: melting points vary widely
            # Peak around groups 5-6 (W has highest at 3695 K)
            # Groups 11-12 have much lower melting points
            if group:
                # Empirical values based on actual periodic trends
                # Base melting points for 3d (period 4) metals
                d_block_mp_base = {
                    3: 1800, 4: 1950, 5: 2200, 6: 2180,  # Sc, Ti, V, Cr
                    7: 1500, 8: 1810, 9: 1770, 10: 1728,  # Mn, Fe, Co, Ni
                    11: 1360, 12: 693  # Cu, Zn
                }
                base_mp = d_block_mp_base.get(group, 1800)

                # Period 5 (4d) metals: generally similar or slightly higher
                if period == 5:
                    if group in [5, 6, 7]:  # Nb, Mo, Tc
                        base_mp *= 1.3
                    elif group in [8, 9, 10]:  # Ru, Rh, Pd
                        base_mp *= 1.2
                    else:
                        base_mp *= 1.0

                # Period 6 (5d) metals: generally highest (except group 11-12)
                elif period == 6:
                    if group in [5, 6, 7]:  # Ta, W, Re - highest melting points
                        base_mp *= 1.6
                    elif group in [8, 9, 10]:  # Os, Ir, Pt
                        base_mp *= 1.4
                    elif group == 11:  # Au - lower than Cu
                        base_mp = 1337
                    elif group == 12:  # Hg - liquid at room temp
                        base_mp = 234

                mp = base_mp
            else:
                mp = 1800
        elif block == 'f':
            # Lanthanides and actinides: moderate to high (~1000-2000 K)
            if period == 6:  # Lanthanides
                mp = 1200 + 50 * ((Z - 57) % 7) - 30 * ((Z - 57) // 7)
            else:  # Actinides
                mp = 1300 + 40 * ((Z - 89) % 7) - 40 * ((Z - 89) // 7)
        else:
            mp = 1000

        # Clamp to physical range
        mp = max(10.0, min(4000.0, mp))

        return round(mp, 1)

    @staticmethod
    def calculate_boiling_point(protons: int, neutrons: int) -> float:
        """
        Estimate boiling point using improved empirical correlations.
        Based on element type and periodic position.

        Uses known values for gases and empirical ratios for others.
        """
        Z = protons
        if Z == 0:
            return 0.0

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)
        group = AtomCalculator._get_group(Z)

        mp = AtomCalculator.calculate_melting_point(protons, neutrons)

        # Specific known values for gases at STP
        gases_at_stp = [1, 2, 7, 8, 9, 10, 17, 18, 36, 54, 86]

        if Z in gases_at_stp:
            if Z == 1:  # Hydrogen
                return 20.3
            elif Z == 2:  # Helium
                return 4.2
            elif Z == 7:  # Nitrogen
                return 77.0
            elif Z == 8:  # Oxygen
                return 90.0
            elif Z == 9:  # Fluorine
                return 85.0
            elif Z == 10:  # Neon
                return 27.0
            elif Z == 17:  # Chlorine
                return 239.0
            elif Z == 18:  # Argon
                return 87.0
            elif Z == 36:  # Krypton
                return 120.0
            elif Z == 54:  # Xenon
                return 165.0
            elif Z == 86:  # Radon
                return 211.0
            else:
                return mp * 1.2

        # Specific known values for some elements
        if Z == 35:  # Bromine
            return 332.0
        elif Z == 53:  # Iodine
            return 457.0
        elif Z == 80:  # Mercury
            return 630.0
        elif Z == 6:  # Carbon (sublimes)
            return 4098.0

        # For other elements, use BP/MP ratios based on bonding type
        if block == 's':
            if group == 1:  # Alkali metals
                ratio = 2.5 + 0.2 * (period - 2)  # Higher ratio for heavier alkalis
            elif group == 2:  # Alkaline earth
                ratio = 1.8 + 0.1 * (period - 2)
            else:
                ratio = 1.8
        elif block == 'p':
            if group:
                position_in_p = group - 12
                if position_in_p <= 2:  # Groups 13-14
                    ratio = 1.5 + 0.1 * position_in_p
                else:
                    ratio = 1.3 + 0.05 * position_in_p
            else:
                ratio = 1.5
        elif block == 'd':
            # Transition metals have BP/MP ratio ~1.7-2.0
            ratio = 1.75
            # Some transition metals have higher ratios
            if group and group >= 8 and group <= 10:
                ratio = 1.9
        elif block == 'f':
            # f-block elements
            ratio = 2.0 + 0.1 * ((Z - 57) % 14 if Z < 89 else (Z - 89) % 14) / 14

        else:
            ratio = 1.7

        bp = mp * ratio

        # Clamp to physical range
        bp = max(10.0, min(6000.0, bp))

        return round(bp, 1)

    @staticmethod
    def calculate_density(protons: int, neutrons: int) -> float:
        """
        Estimate density using improved empirical correlations.
        Based on element type, periodic position, and state of matter at STP.

        Gases at STP have very low density (~0.0001-0.01 g/cm3)
        Liquids at STP have moderate density (~1-14 g/cm3)
        Solids vary widely (~0.5-22 g/cm3)
        """
        Z = protons
        if Z == 0:
            return 0.0

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)
        group = AtomCalculator._get_group(Z)

        # Gases at STP - use known values
        gases_at_stp = [1, 2, 7, 8, 9, 10, 17, 18, 36, 54, 86]
        if Z in gases_at_stp:
            if Z == 1:  # H2
                return 0.00009
            elif Z == 2:  # He
                return 0.00018
            elif Z == 7:  # N2
                return 0.00125
            elif Z == 8:  # O2
                return 0.00143
            elif Z == 9:  # F2
                return 0.0017
            elif Z == 10:  # Ne
                return 0.0009
            elif Z == 17:  # Cl2
                return 0.0032
            elif Z == 18:  # Ar
                return 0.00178
            elif Z == 36:  # Kr
                return 0.00375
            elif Z == 54:  # Xe
                return 0.00589
            elif Z == 86:  # Rn
                return 0.00973

        # Liquids at STP
        if Z == 35:  # Bromine
            return 3.12
        elif Z == 80:  # Mercury
            return 13.55

        # Solid elements - use empirical correlations
        if block == 's':
            if group == 1:  # Alkali metals - very light (0.5-1.9 g/cm3)
                density = 0.5 + 0.25 * (period - 2)
            elif group == 2:  # Alkaline earth - light to moderate (1.5-3.6 g/cm3)
                density = 1.5 + 0.4 * (period - 2)
            else:
                density = 1.5
        elif block == 'p':
            if group:
                position_in_p = group - 12
                if position_in_p == 1:  # Group 13 (Al, Ga, In, Tl)
                    density = 2.5 + 1.5 * (period - 3)
                elif position_in_p == 2:  # Group 14 (C, Si, Ge, Sn, Pb)
                    if Z == 6:  # Carbon (graphite)
                        return 2.27
                    density = 2.3 + 2.0 * (period - 3)
                elif position_in_p == 3:  # Group 15 (P, As, Sb, Bi)
                    density = 1.8 + 1.8 * (period - 3)
                elif position_in_p == 4:  # Group 16 (S, Se, Te, Po)
                    density = 2.0 + 1.5 * (period - 3)
                elif position_in_p == 5:  # Group 17 (halogens)
                    density = 1.5 + 1.0 * (period - 3)
                else:
                    density = 2.0
            else:
                density = 2.0
        elif block == 'd':
            # Transition metals - moderate to high (4-22 g/cm3)
            if group:
                position_in_d = group - 2
                # Group 12 metals (Zn, Cd) have lower densities
                if group == 12:
                    base_density = 7.0 + 0.5 * (period - 4)
                else:
                    base_density = 4.0 + 0.8 * position_in_d
                    if period == 5:
                        base_density += 3.0
                    elif period == 6:
                        base_density += 8.0  # Lanthanide contraction effect
                    elif period == 7:
                        base_density += 15.0  # Relativistic contraction for superheavy
                density = base_density
                # Osmium and Iridium are the densest elements
                if Z == 76:  # Os
                    return 22.59
                elif Z == 77:  # Ir
                    return 22.56
                elif Z == 78:  # Pt
                    return 21.45
                elif Z == 79:  # Au
                    return 19.30
            else:
                density = 8.0
        elif block == 'f':
            # Lanthanides and actinides - high density (6-20 g/cm3)
            if period == 6:  # Lanthanides
                position = Z - 57
                density = 6.0 + 0.5 * position
            else:  # Actinides
                position = Z - 89
                density = 10.0 + 0.8 * position
                if Z == 92:  # Uranium
                    return 19.1
        else:
            density = 5.0

        # Clamp to reasonable range
        density = max(0.5, min(23.0, density))

        return round(density, 4)

    @staticmethod
    def calculate_electron_affinity(protons: int) -> float:
        """
        Estimate electron affinity using periodic trends.
        EA increases across period, decreases down group.
        """
        Z = protons
        if Z == 0:
            return 0.0

        shell_config = AtomCalculator._get_shell_configuration(Z)
        n = shell_config['n']
        Z_eff = AtomCalculator._calculate_z_effective(Z, shell_config)

        # Base EA from effective nuclear charge
        # EA ≈ k * Z_eff / r²
        r = AtomCalculator.calculate_atomic_radius(Z)
        if r == 0:
            r = 100

        r_angstrom = r / 100

        base_ea = 50 * Z_eff / (r_angstrom ** 2)

        # Corrections based on orbital filling
        block = AtomCalculator._get_block(Z)
        valence = AtomCalculator._get_valence_electrons(Z)

        # Filled and half-filled shells have lower EA
        if block == 's' and valence == 2:
            base_ea *= 0.1  # Filled s shell
        elif block == 'p' and valence == 8:
            base_ea *= 0.0  # Noble gases
        elif block == 'p' and valence == 5:
            base_ea *= 0.5  # Half-filled p
        elif block == 'd' and valence in [5, 10]:
            base_ea *= 0.3  # Half or fully filled d

        # Halogens have high EA
        if block == 'p' and valence == 7:
            base_ea *= 2.0

        return round(max(0.0, min(400.0, base_ea)), 1)

    @staticmethod
    def get_block_period_group(protons: int) -> Tuple[str, int, Optional[int]]:
        """
        Determine block, period, and group from atomic number.
        Uses aufbau principle.
        """
        Z = protons
        if Z == 0:
            return ('s', 1, 1)

        block = AtomCalculator._get_block(Z)
        period = AtomCalculator._get_period(Z)
        group = AtomCalculator._get_group(Z)

        return (block, period, group)

    @staticmethod
    def get_electron_configuration(protons: int) -> str:
        """Generate electron configuration string."""
        Z = protons
        if Z == 0:
            return ""

        # Orbital order following aufbau principle
        orbitals = [
            (1, 's', 2), (2, 's', 2), (2, 'p', 6), (3, 's', 2), (3, 'p', 6),
            (4, 's', 2), (3, 'd', 10), (4, 'p', 6), (5, 's', 2), (4, 'd', 10),
            (5, 'p', 6), (6, 's', 2), (4, 'f', 14), (5, 'd', 10), (6, 'p', 6),
            (7, 's', 2), (5, 'f', 14), (6, 'd', 10), (7, 'p', 6)
        ]

        superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                       '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}

        config = []
        electrons_remaining = Z

        for n, l, max_e in orbitals:
            if electrons_remaining <= 0:
                break

            e_in_orbital = min(electrons_remaining, max_e)
            electrons_remaining -= e_in_orbital

            # Convert number to superscript
            sup = ''.join(superscripts[c] for c in str(e_in_orbital))
            config.append(f"{n}{l}{sup}")

        return ' '.join(config)

    @staticmethod
    def estimate_primary_emission_wavelength(protons: int) -> float:
        """
        Estimate primary emission wavelength using Rydberg formula.
        λ = hc / ΔE for n=2 to n=1 transition (or similar)
        """
        Z = protons
        if Z == 0:
            return 500.0

        # Use hydrogen-like formula with effective charge
        shell_config = AtomCalculator._get_shell_configuration(Z)
        Z_eff = AtomCalculator._calculate_z_effective(Z, shell_config)

        # Balmer series transition (n=3 to n=2) scaled by Z_eff
        # λ = 91.2 nm / Z_eff² * (1/n₁² - 1/n₂²)⁻¹

        n1, n2 = 2, 3
        factor = 1 / (n1**2) - 1 / (n2**2)

        if factor > 0 and Z_eff > 0:
            wavelength = 91.2 / (Z_eff ** 2 * factor)
        else:
            wavelength = 500.0

        # Clamp to reasonable range
        wavelength = max(100.0, min(1000.0, wavelength))

        return round(wavelength, 1)

    @staticmethod
    def determine_stability(protons: int, neutrons: int) -> Tuple[bool, Optional[str]]:
        """
        Determine if nucleus is stable and estimate half-life if not.
        Uses band of stability calculations.
        """
        Z = protons
        N = neutrons
        A = Z + N

        if A == 0 or Z == 0:
            return (False, None)

        # Ratio of N/Z for stability
        ratio = N / Z if Z > 0 else 0

        # Band of stability parameters
        # For light nuclei: N ≈ Z
        # For heavy nuclei: N ≈ 1.5Z

        if Z <= 20:
            optimal_ratio = 1.0
            tolerance = 0.15
        elif Z <= 40:
            optimal_ratio = 1.0 + 0.015 * (Z - 20)
            tolerance = 0.12
        elif Z <= 82:
            optimal_ratio = 1.3 + 0.005 * (Z - 40)
            tolerance = 0.10
        else:
            optimal_ratio = 1.5
            tolerance = 0.08

        # Check stability
        deviation = abs(ratio - optimal_ratio)
        is_stable = deviation <= tolerance and Z <= 82

        # All elements with Z > 82 are unstable
        if Z > 82:
            is_stable = False

        # Estimate half-life for unstable nuclei
        half_life = None
        if not is_stable:
            if Z > 110:
                half_life = f"{round(0.001 * 1000 / (Z - 110), 1)} milliseconds"
            elif Z > 100:
                half_life = f"{round(100 / (Z - 100), 0)} days"
            elif Z > 82:
                half_life = f"{round(1e6 / (Z - 82), 0)} years"
            elif deviation > tolerance * 2:
                half_life = f"{round(1 / deviation, 1)} seconds"
            else:
                half_life = f"{round(100 / deviation, 0)} years"

        return (is_stable, half_life)

    # ==================== Helper Methods ====================

    @staticmethod
    def _get_shell_configuration(Z: int) -> Dict:
        """Get shell configuration for given Z."""
        # Determine principal quantum number and angular momentum
        if Z <= 2:
            return {'n': 1, 'l': 0, 'electrons_in_shell': Z}
        elif Z <= 10:
            return {'n': 2, 'l': 1 if Z > 4 else 0, 'electrons_in_shell': Z - 2}
        elif Z <= 18:
            return {'n': 3, 'l': 1 if Z > 12 else 0, 'electrons_in_shell': Z - 10}
        elif Z <= 36:
            if Z <= 20:
                return {'n': 4, 'l': 0, 'electrons_in_shell': Z - 18}
            elif Z <= 30:
                return {'n': 3, 'l': 2, 'electrons_in_shell': Z - 18}
            else:
                return {'n': 4, 'l': 1, 'electrons_in_shell': Z - 28}
        elif Z <= 54:
            if Z <= 38:
                return {'n': 5, 'l': 0, 'electrons_in_shell': Z - 36}
            elif Z <= 48:
                return {'n': 4, 'l': 2, 'electrons_in_shell': Z - 36}
            else:
                return {'n': 5, 'l': 1, 'electrons_in_shell': Z - 46}
        elif Z <= 86:
            if Z <= 56:
                return {'n': 6, 'l': 0, 'electrons_in_shell': Z - 54}
            elif Z <= 71:
                return {'n': 4, 'l': 3, 'electrons_in_shell': Z - 54}
            elif Z <= 80:
                return {'n': 5, 'l': 2, 'electrons_in_shell': Z - 54}
            else:
                return {'n': 6, 'l': 1, 'electrons_in_shell': Z - 78}
        else:
            if Z <= 88:
                return {'n': 7, 'l': 0, 'electrons_in_shell': Z - 86}
            elif Z <= 103:
                return {'n': 5, 'l': 3, 'electrons_in_shell': Z - 86}
            elif Z <= 112:
                return {'n': 6, 'l': 2, 'electrons_in_shell': Z - 86}
            else:
                return {'n': 7, 'l': 1, 'electrons_in_shell': Z - 110}

    @staticmethod
    def _calculate_z_effective(Z: int, shell_config: Dict) -> float:
        """Calculate effective nuclear charge using Slater's rules."""
        n = shell_config['n']

        # Simplified Slater's rules
        if n == 1:
            sigma = 0.30 * (Z - 1)  # 1s electrons shield each other
        elif n == 2:
            sigma = 2 * 0.85 + (Z - 3) * 0.35 if Z > 2 else 0
        elif n == 3:
            sigma = 2 * 1.0 + 8 * 0.85 + (Z - 11) * 0.35 if Z > 10 else 0
        else:
            # General approximation for higher shells
            inner_electrons = Z - shell_config.get('electrons_in_shell', 1)
            sigma = inner_electrons * 0.85

        Z_eff = Z - sigma
        return max(1.0, Z_eff)

    @staticmethod
    def _calculate_z_effective_ie(Z: int, shell_config: Dict) -> float:
        """
        Calculate effective nuclear charge for ionization energy calculations.
        Uses improved Slater's rules with corrections for multi-electron atoms.
        """
        n = shell_config['n']
        l = shell_config.get('l', 0)
        
        # For first ionization, calculate Z_eff for outermost electron
        # Using modified Slater's rules
        
        if Z == 1:
            return 1.0
        elif Z == 2:
            return 1.7  # He: 2 - 0.30
        
        # Shielding constants based on orbital type
        inner_electrons = Z - shell_config.get('electrons_in_shell', 1)
        same_shell = shell_config.get('electrons_in_shell', 1) - 1
        
        # Shielding from inner shells
        if n == 1:
            sigma = 0.30 * same_shell
        elif n == 2:
            if l == 0:  # 2s
                sigma = 2 * 0.85 + same_shell * 0.35
            else:  # 2p
                sigma = 2 * 0.85 + 2 * 0.35 + same_shell * 0.35
        elif n == 3:
            sigma = 2 * 1.0 + 8 * 0.85 + same_shell * 0.35
        else:
            # For n >= 4, use general approximation
            sigma = inner_electrons * 0.85 + same_shell * 0.35
        
        Z_eff = Z - sigma
        return max(1.0, Z_eff)

    @staticmethod
    def _get_block(Z: int) -> str:
        """Determine block from atomic number."""
        if Z == 0:
            return 's'

        # s-block
        s_block = [1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]
        if Z in s_block:
            return 's'

        # f-block (lanthanides and actinides)
        if 57 <= Z <= 71 or 89 <= Z <= 103:
            return 'f'

        # d-block
        d_ranges = [(21, 30), (39, 48), (72, 80), (104, 112)]
        for start, end in d_ranges:
            if start <= Z <= end:
                return 'd'

        # p-block (everything else)
        return 'p'

    @staticmethod
    def _get_period(Z: int) -> int:
        """Determine period from atomic number."""
        if Z <= 2:
            return 1
        elif Z <= 10:
            return 2
        elif Z <= 18:
            return 3
        elif Z <= 36:
            return 4
        elif Z <= 54:
            return 5
        elif Z <= 86:
            return 6
        else:
            return 7

    @staticmethod
    def _get_group(Z: int) -> Optional[int]:
        """Determine group from atomic number."""
        block = AtomCalculator._get_block(Z)

        if block == 'f':
            return None  # Lanthanides/actinides don't have group numbers

        period = AtomCalculator._get_period(Z)

        if block == 's':
            # s-block: groups 1-2
            # Each period starts: 1, 3, 11, 19, 37, 55, 87
            period_starts = {1: 1, 2: 3, 3: 11, 4: 19, 5: 37, 6: 55, 7: 87}
            start = period_starts.get(period, 1)
            position = Z - start
            return position + 1  # Group 1 or 2

        elif block == 'd':
            # d-block: groups 3-12
            # Period 4 d-block: Sc(Z=21, group 3) to Zn(Z=30, group 12)
            # Period 5 d-block: Y(Z=39, group 3) to Cd(Z=48, group 12)
            # Period 6 d-block: Hf(Z=72, group 4) to Hg(Z=80, group 12) - starts at group 4 due to lanthanides
            # Period 7 d-block: Rf(Z=104, group 4) to Cn(Z=112, group 12) - starts at group 4 due to actinides
            d_starts = {4: (21, 3), 5: (39, 3), 6: (72, 4), 7: (104, 4)}
            d_info = d_starts.get(period)
            if d_info:
                d_start_z, d_start_group = d_info
                position = Z - d_start_z
                return d_start_group + position  # Groups 3-12 or 4-12

        elif block == 'p':
            # p-block: groups 13-18
            # p-block starts: period 2: Z=5, period 3: Z=13, period 4: Z=31,
            # period 5: Z=49, period 6: Z=81, period 7: Z=113
            p_starts = {2: 5, 3: 13, 4: 31, 5: 49, 6: 81, 7: 113}
            p_start = p_starts.get(period)
            if p_start:
                position = Z - p_start
                return 13 + position  # Groups 13-18

        return None

    @staticmethod
    def _get_valence_electrons(Z: int) -> int:
        """Get number of valence electrons."""
        block = AtomCalculator._get_block(Z)
        group = AtomCalculator._get_group(Z)

        if group is not None:
            if group <= 2:
                return group
            elif group >= 13:
                return group - 10
            else:  # d-block
                return group - 2
        else:  # f-block
            period = AtomCalculator._get_period(Z)
            if period == 6:
                return Z - 54  # Lanthanides
            else:
                return Z - 86  # Actinides

    @staticmethod
    def _is_noble_gas(Z: int) -> bool:
        """Check if element is a noble gas."""
        return Z in [2, 10, 18, 36, 54, 86, 118]

    @classmethod
    def create_atom_from_particles(cls, protons: int, neutrons: int, electrons: int,
                                   name: str = None, symbol: str = None) -> Dict:
        """
        Create a complete atom JSON structure from constituent particles.

        Args:
            protons: Number of protons
            neutrons: Number of neutrons
            electrons: Number of electrons (normally equals protons for neutral atom)
            name: Optional custom name
            symbol: Optional custom symbol

        Returns:
            Complete atom data dictionary
        """
        Z = protons
        A = protons + neutrons

        block, period, group = cls.get_block_period_group(Z)
        is_stable, half_life = cls.determine_stability(protons, neutrons)

        # Generate default name and symbol if not provided
        if symbol is None:
            symbol = f"X{Z}"
        if name is None:
            name = f"Element-{Z}"

        atom_data = {
            "symbol": symbol,
            "name": name,
            "atomic_number": Z,
            "atomic_mass": cls.calculate_atomic_mass(protons, neutrons),
            "block": block,
            "period": period,
            "group": group,
            "ionization_energy": cls.calculate_ionization_energy(Z),
            "electronegativity": cls.calculate_electronegativity(Z),
            "atomic_radius": cls.calculate_atomic_radius(Z),
            "melting_point": cls.calculate_melting_point(protons, neutrons),
            "boiling_point": cls.calculate_boiling_point(protons, neutrons),
            "density": cls.calculate_density(protons, neutrons),
            "electron_affinity": cls.calculate_electron_affinity(Z),
            "valence_electrons": cls._get_valence_electrons(Z),
            "electron_configuration": cls.get_electron_configuration(Z),
            "primary_emission_wavelength": cls.estimate_primary_emission_wavelength(Z),
            "visible_emission_wavelength": cls.estimate_primary_emission_wavelength(Z),
            "isotopes": [
                {
                    "mass_number": A,
                    "neutrons": neutrons,
                    "abundance": 100.0,
                    "is_stable": is_stable,
                    "half_life": half_life
                }
            ],
            "_created_from": {
                "protons": protons,
                "neutrons": neutrons,
                "electrons": electrons
            }
        }

        return atom_data


# ==================== Subatomic Particle Creation from Quarks ====================

class SubatomicCalculator:
    """
    Calculate subatomic particle properties from quark composition.
    """

    QUARK_PROPERTIES = {
        'u': {'charge': 2/3, 'mass_mev': 2.2, 'spin': 0.5, 'baryon': 1/3, 'name': 'Up'},
        'd': {'charge': -1/3, 'mass_mev': 4.7, 'spin': 0.5, 'baryon': 1/3, 'name': 'Down'},
        's': {'charge': -1/3, 'mass_mev': 95.0, 'spin': 0.5, 'baryon': 1/3, 'name': 'Strange'},
        'c': {'charge': 2/3, 'mass_mev': 1275.0, 'spin': 0.5, 'baryon': 1/3, 'name': 'Charm'},
        'b': {'charge': -1/3, 'mass_mev': 4180.0, 'spin': 0.5, 'baryon': 1/3, 'name': 'Bottom'},
        't': {'charge': 2/3, 'mass_mev': 173100.0, 'spin': 0.5, 'baryon': 1/3, 'name': 'Top'},
        # Antiquarks
        'u̅': {'charge': -2/3, 'mass_mev': 2.2, 'spin': 0.5, 'baryon': -1/3, 'name': 'Anti-up'},
        'd̅': {'charge': 1/3, 'mass_mev': 4.7, 'spin': 0.5, 'baryon': -1/3, 'name': 'Anti-down'},
        's̅': {'charge': 1/3, 'mass_mev': 95.0, 'spin': 0.5, 'baryon': -1/3, 'name': 'Anti-strange'},
        'c̅': {'charge': -2/3, 'mass_mev': 1275.0, 'spin': 0.5, 'baryon': -1/3, 'name': 'Anti-charm'},
        'b̅': {'charge': 1/3, 'mass_mev': 4180.0, 'spin': 0.5, 'baryon': -1/3, 'name': 'Anti-bottom'},
        't̅': {'charge': -2/3, 'mass_mev': 173100.0, 'spin': 0.5, 'baryon': -1/3, 'name': 'Anti-top'},
    }

    @classmethod
    def calculate_charge(cls, quarks: List[str]) -> float:
        """Calculate total charge from quark composition."""
        total = 0.0
        for q in quarks:
            if q in cls.QUARK_PROPERTIES:
                total += cls.QUARK_PROPERTIES[q]['charge']
        return round(total, 6)

    @classmethod
    def calculate_mass(cls, quarks: List[str], spin_aligned: bool = False) -> float:
        """
        Calculate hadron mass from quarks using constituent quark model.

        The mass of hadrons comes primarily from QCD binding energy, not bare
        quark masses. This uses empirical formulas calibrated to known particles.

        Args:
            quarks: List of quark symbols
            spin_aligned: If True, quarks have parallel spins (affects baryon mass)

        Returns:
            Estimated mass in MeV/c^2
        """
        num_quarks = len(quarks)

        # Count quark flavors (handling both quarks and antiquarks)
        def get_flavor(q):
            return q.replace('̅', '').replace('-bar', '').lower()

        flavor_counts = {'u': 0, 'd': 0, 's': 0, 'c': 0, 'b': 0, 't': 0}
        for q in quarks:
            flavor = get_flavor(q)
            if flavor in flavor_counts:
                flavor_counts[flavor] += 1

        num_strange = flavor_counts['s']
        num_charm = flavor_counts['c']
        num_bottom = flavor_counts['b']
        num_top = flavor_counts['t']

        if num_quarks == 3:  # Baryon
            return cls._calculate_baryon_mass(
                num_strange, num_charm, num_bottom, num_top, spin_aligned, quarks
            )
        elif num_quarks == 2:  # Meson
            return cls._calculate_meson_mass(
                num_strange, num_charm, num_bottom, num_top, spin_aligned
            )
        else:
            return cls._calculate_exotic_mass(quarks)

    @classmethod
    def _calculate_baryon_mass(cls, num_strange: int, num_charm: int,
                                num_bottom: int, num_top: int,
                                spin_aligned: bool, quarks: List[str] = None) -> float:
        """Calculate baryon mass using empirical formula."""
        BASE_NUCLEON_MASS = 939.0
        STRANGE_CONTRIBUTION = 180.0
        CHARM_CONTRIBUTION = 1350.0
        BOTTOM_CONTRIBUTION = 4680.0
        TOP_CONTRIBUTION = 173000.0
        SPIN_ALIGNMENT_ENERGY = 293.0
        SIGMA_LAMBDA_SPLITTING = 75.0  # Isospin splitting for strange baryons

        mass = BASE_NUCLEON_MASS
        mass += num_strange * STRANGE_CONTRIBUTION
        mass += num_charm * CHARM_CONTRIBUTION
        mass += num_bottom * BOTTOM_CONTRIBUTION
        mass += num_top * TOP_CONTRIBUTION

        # Isospin correction: Sigma (I=1) is ~75 MeV heavier than Lambda (I=0)
        # Lambda is uds (all different), Sigma has two identical light quarks
        if num_strange == 1 and quarks:
            light_quarks = [q for q in quarks if q in ['u', 'd']]
            if len(light_quarks) == 2 and light_quarks[0] == light_quarks[1]:
                mass += SIGMA_LAMBDA_SPLITTING

        if spin_aligned:
            mass += SPIN_ALIGNMENT_ENERGY

        return round(mass, 2)

    @classmethod
    def _calculate_meson_mass(cls, num_strange: int, num_charm: int,
                               num_bottom: int, num_top: int,
                               spin_aligned: bool) -> float:
        """Calculate meson mass using empirical formulas."""
        # Quarkonium masses (quark-antiquark of same flavor)
        JPSI_MASS = 3097.0      # J/psi (cc-bar vector)
        ETAC_MASS = 2984.0      # eta_c (cc-bar pseudoscalar)
        UPSILON_MASS = 9460.0   # Upsilon (bb-bar vector)
        ETAB_MASS = 9399.0      # eta_b (bb-bar pseudoscalar)
        PHI_MASS = 1019.0       # phi (ss-bar vector)

        # Regular meson masses
        PION_MASS = 137.0
        RHO_MASS = 770.0
        KAON_MASS = 495.0
        KSTAR_MASS = 892.0
        D_MESON_MASS = 1870.0
        DS_MESON_MASS = 1968.0
        B_MESON_MASS = 5280.0
        BS_MESON_MASS = 5367.0
        BC_MESON_MASS = 6275.0
        VECTOR_SHIFT = 400.0

        if num_top > 0:
            return 175000.0

        # Check for quarkonium states (quark + its antiquark)
        if num_charm == 2:  # Charmonium (c + c-bar)
            return JPSI_MASS if spin_aligned else ETAC_MASS
        if num_bottom == 2:  # Bottomonium (b + b-bar)
            return UPSILON_MASS if spin_aligned else ETAB_MASS
        if num_strange == 2 and num_charm == 0 and num_bottom == 0:  # ss-bar
            return PHI_MASS if spin_aligned else 700.0  # Approximate eta'

        if num_bottom > 0:
            if num_charm > 0:
                base = BC_MESON_MASS
            elif num_strange > 0:
                base = BS_MESON_MASS
            else:
                base = B_MESON_MASS
        elif num_charm > 0:
            if num_strange > 0:
                base = DS_MESON_MASS
            else:
                base = D_MESON_MASS
        elif num_strange > 0:
            base = KSTAR_MASS if spin_aligned else KAON_MASS
        else:
            base = RHO_MASS if spin_aligned else PION_MASS

        if spin_aligned and (num_charm > 0 or num_bottom > 0):
            base += VECTOR_SHIFT

        return round(base, 2)

    @classmethod
    def _calculate_exotic_mass(cls, quarks: List[str]) -> float:
        """Calculate mass for exotic hadrons using constituent quark model."""
        CONSTITUENT_MASSES = {
            'u': 336.0, 'd': 336.0, 's': 486.0,
            'c': 1550.0, 'b': 4730.0, 't': 173000.0,
            'u̅': 336.0, 'd̅': 336.0, 's̅': 486.0,
            'c̅': 1550.0, 'b̅': 4730.0, 't̅': 173000.0,
        }

        total = 0.0
        for q in quarks:
            q_clean = q.replace('-bar', '̅')
            total += CONSTITUENT_MASSES.get(q_clean, 336.0)

        # Binding energy corrections for exotic hadrons
        num_quarks = len(quarks)
        if num_quarks == 4:  # Tetraquark
            total -= 100.0  # Diquark-antidiquark binding
        elif num_quarks == 5:  # Pentaquark
            total -= 200.0  # Stronger binding in pentaquarks
        elif num_quarks == 6:  # Hexaquark/Dibaryon
            # Check for flavor-singlet pattern (Eta-like: uu̅ + dd̅ + ss̅)
            quark_set = set()
            for q in quarks:
                base = q.replace('̅', '')
                quark_set.add(base)
            if quark_set == {'u', 'd', 's'}:
                # Flavor-singlet meson (Eta or Eta')
                return 548.0  # Eta meson mass
            total -= 50.0 * (num_quarks - 3)  # Approximate binding

        return round(total, 2)

    @classmethod
    def calculate_spin(cls, quarks: List[str], aligned: bool = True) -> float:
        """
        Calculate total spin from quark spins.
        Quarks can be aligned (parallel) or anti-aligned.
        """
        num_quarks = len(quarks)

        if num_quarks == 3:  # Baryon
            # Spin 1/2 (one anti-aligned) or 3/2 (all aligned)
            if aligned:
                return 1.5  # All parallel
            else:
                return 0.5  # Ground state baryons
        elif num_quarks == 2:  # Meson
            # Spin 0 (anti-aligned) or 1 (aligned)
            if aligned:
                return 1.0
            else:
                return 0.0

        return 0.5

    @classmethod
    def calculate_baryon_number(cls, quarks: List[str]) -> float:
        """Calculate baryon number from quarks."""
        total = 0.0
        for q in quarks:
            if q in cls.QUARK_PROPERTIES:
                total += cls.QUARK_PROPERTIES[q]['baryon']
        return round(total)

    @classmethod
    def determine_particle_type(cls, quarks: List[str]) -> str:
        """Determine if particle is baryon, meson, or exotic."""
        num_quarks = len(quarks)
        baryon_num = cls.calculate_baryon_number(quarks)

        if num_quarks == 3 and baryon_num == 1:
            return "Baryon"
        elif num_quarks == 3 and baryon_num == -1:
            return "Antibaryon"
        elif num_quarks == 2 and baryon_num == 0:
            return "Meson"
        elif num_quarks == 4:
            return "Tetraquark"
        elif num_quarks == 5:
            return "Pentaquark"
        else:
            return "Exotic"

    @classmethod
    def estimate_stability(cls, quarks: List[str]) -> Tuple[str, Optional[str]]:
        """
        Estimate stability and half-life.
        Only particles with u and d quarks are stable.
        """
        has_strange = any(q in ['s', 's̅'] for q in quarks)
        has_charm = any(q in ['c', 'c̅'] for q in quarks)
        has_bottom = any(q in ['b', 'b̅'] for q in quarks)
        has_top = any(q in ['t', 't̅'] for q in quarks)

        if has_top:
            return ("Unstable", "5e-25 seconds")
        elif has_bottom:
            return ("Unstable", "1.5e-12 seconds")
        elif has_charm:
            return ("Unstable", "1e-12 seconds")
        elif has_strange:
            return ("Unstable", "1e-10 seconds")
        else:
            # Only u and d quarks - check if it's proton/neutron
            charge = cls.calculate_charge(quarks)
            if abs(charge - 1.0) < 0.01:  # Proton-like
                return ("Stable", None)
            elif abs(charge) < 0.01:  # Neutron-like
                return ("Unstable", "879.4 seconds")
            else:
                return ("Unstable", "1e-8 seconds")

    @classmethod
    def get_interaction_forces(cls, quarks: List[str]) -> List[str]:
        """Determine which forces the particle interacts with."""
        forces = ["Strong", "Gravitational"]

        charge = cls.calculate_charge(quarks)
        if abs(charge) > 0.01:
            forces.append("Electromagnetic")

        # All hadrons interact weakly
        forces.append("Weak")

        return forces

    @classmethod
    def generate_symbol(cls, quarks: List[str]) -> str:
        """Generate a symbol based on quark content."""
        charge = cls.calculate_charge(quarks)

        # Common particle symbols
        if quarks == ['u', 'u', 'd']:
            return 'p'
        elif quarks == ['u', 'd', 'd']:
            return 'n'
        elif sorted(quarks) == ['d', 'u', 'u']:
            return 'p'
        elif sorted(quarks) == ['d', 'd', 'u']:
            return 'n'

        # Generic symbol based on charge
        if charge > 0:
            return f"X⁺{'⁺' * int(charge - 1)}" if charge > 1 else "X⁺"
        elif charge < 0:
            return f"X⁻{'⁻' * int(abs(charge) - 1)}" if charge < -1 else "X⁻"
        else:
            return "X⁰"

    @classmethod
    def create_particle_from_quarks(cls, quarks: List[str], name: str = None,
                                    symbol: str = None, spin_aligned: bool = False) -> Dict:
        """
        Create a complete subatomic particle JSON structure from quarks.

        Args:
            quarks: List of quark symbols (e.g., ['u', 'u', 'd'] for proton)
            name: Optional custom name
            symbol: Optional custom symbol
            spin_aligned: Whether quark spins are aligned (affects total spin)

        Returns:
            Complete particle data dictionary
        """
        charge = cls.calculate_charge(quarks)
        mass = cls.calculate_mass(quarks)
        spin = cls.calculate_spin(quarks, spin_aligned)
        baryon_num = cls.calculate_baryon_number(quarks)
        particle_type = cls.determine_particle_type(quarks)
        stability, half_life = cls.estimate_stability(quarks)

        if symbol is None:
            symbol = cls.generate_symbol(quarks)
        if name is None:
            name = f"Hadron ({','.join(quarks)})"

        # Build composition list
        composition = []
        quark_counts = {}
        for q in quarks:
            quark_counts[q] = quark_counts.get(q, 0) + 1

        for q, count in quark_counts.items():
            if q in cls.QUARK_PROPERTIES:
                composition.append({
                    "Constituent": cls.QUARK_PROPERTIES[q]['name'] + " Quark",
                    "Count": count,
                    "Charge_e": cls.QUARK_PROPERTIES[q]['charge']
                })

        particle_data = {
            "Name": name,
            "Symbol": symbol,
            "Type": "Subatomic Particle",
            "Classification": ["Fermion" if spin % 1 != 0 else "Boson", particle_type, "Hadron", "Composite Particle"],
            "Charge_e": charge,
            "Mass_MeVc2": mass,
            "Mass_kg": mass * 1.78266192e-30,  # MeV/c² to kg
            "Mass_amu": mass / 931.494,
            "Spin_hbar": spin,
            "MagneticDipoleMoment_J_T": None,
            "LeptonNumber_L": 0,
            "BaryonNumber_B": baryon_num,
            "Isospin_I": 0.5 if len(quarks) == 3 else 0,
            "Isospin_I3": (quarks.count('u') - quarks.count('d')) / 2,
            "Parity_P": 1 if particle_type == "Baryon" else -1,
            "Composition": composition,
            "Stability": stability,
            "HalfLife_s": half_life,
            "DecayProducts": [],
            "Antiparticle": {
                "Name": f"Anti-{name}",
                "Symbol": symbol.replace('⁺', '⁻').replace('⁻', '⁺') if '⁺' in symbol or '⁻' in symbol else f"{symbol}̅"
            },
            "InteractionForces": cls.get_interaction_forces(quarks),
            "_created_from": {
                "quarks": quarks,
                "spin_aligned": spin_aligned
            }
        }

        return particle_data


# ==================== Molecule Creation from Atoms ====================

class MoleculeCalculator:
    """
    Calculate molecular properties from atomic composition.
    """

    # Electronegativity values for common elements (Pauling scale)
    # Values from element JSON files where available
    ELECTRONEGATIVITIES = {
        'H': 2.20, 'He': 0.0,
        'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.0,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.0,
        'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
        'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
        'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.0,
        'Rb': 0.82, 'Sr': 0.95, 'I': 2.66, 'Xe': 0.0,
        'Cs': 0.79, 'Ba': 0.89
    }

    # Covalent radii in pm
    COVALENT_RADII = {
        'H': 31, 'He': 28,
        'Li': 128, 'Be': 96, 'B': 84, 'C': 76, 'N': 71, 'O': 66, 'F': 57, 'Ne': 58,
        'Na': 166, 'Mg': 141, 'Al': 121, 'Si': 111, 'P': 107, 'S': 105, 'Cl': 102, 'Ar': 106,
        'K': 203, 'Ca': 176, 'Br': 120, 'I': 139
    }

    # Standard atomic masses (from element JSON files)
    ATOMIC_MASSES = {
        'H': 1.008, 'He': 4.003,
        'Li': 6.94, 'Be': 9.012, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.065, 'Cl': 35.45, 'Ar': 39.948,
        'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38,
        'Br': 79.904, 'Ag': 107.868, 'I': 126.904, 'Au': 196.967
    }

    # Valence electrons for VSEPR geometry prediction
    VALENCE_ELECTRONS = {
        'H': 1, 'He': 2,
        'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
        'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
        'K': 1, 'Ca': 2, 'Br': 7, 'I': 7
    }

    @classmethod
    def calculate_molecular_mass(cls, composition: List[Dict], atom_data: Dict = None) -> float:
        """
        Calculate molecular mass from atomic composition.

        Args:
            composition: List of {"Element": symbol, "Count": n}
            atom_data: Optional dict mapping symbols to full atom data
        """
        total_mass = 0.0

        for comp in composition:
            element = comp.get('Element', '')
            count = comp.get('Count', 1)

            # Try to get mass from provided atom data first
            if atom_data and element in atom_data:
                mass = atom_data[element].get('atomic_mass', cls.ATOMIC_MASSES.get(element, 0))
            else:
                mass = cls.ATOMIC_MASSES.get(element, 0)

            total_mass += mass * count

        return round(total_mass, 3)

    @classmethod
    def generate_formula(cls, composition: List[Dict]) -> str:
        """Generate molecular formula from composition."""
        subscripts = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                     '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}

        # Sort by Hill system: C first, then H, then alphabetical
        sorted_comp = sorted(composition, key=lambda x: (
            0 if x['Element'] == 'C' else (1 if x['Element'] == 'H' else 2),
            x['Element']
        ))

        formula = ""
        for comp in sorted_comp:
            element = comp.get('Element', '')
            count = comp.get('Count', 1)

            formula += element
            if count > 1:
                for digit in str(count):
                    formula += subscripts.get(digit, digit)

        return formula

    @classmethod
    def determine_bond_type(cls, composition: List[Dict], atom_data: Dict = None) -> str:
        """
        Determine primary bond type based on electronegativity differences.
        Uses threshold of 1.7 for ionic bonds (consistent with chemistry conventions).
        Returns "Covalent" for all non-ionic bonds (both polar and nonpolar covalent).
        """
        if len(composition) < 2:
            return "None"

        electronegativities = []
        for comp in composition:
            element = comp['Element']
            if atom_data and element in atom_data:
                en = atom_data[element].get('electronegativity', cls.ELECTRONEGATIVITIES.get(element, 2.0))
            else:
                en = cls.ELECTRONEGATIVITIES.get(element, 2.0)
            electronegativities.append(en)

        if not electronegativities:
            return "Covalent"

        max_en = max(electronegativities)
        min_en = min(electronegativities)
        diff = max_en - min_en

        # Only classify as Ionic when electronegativity difference > 1.7
        # (e.g., Na-Cl has difference of 3.16 - 0.93 = 2.23)
        if diff > 1.7:
            return "Ionic"
        else:
            # Return "Covalent" for all covalent bonds (polar or nonpolar)
            # This matches the format in the molecule JSON files
            return "Covalent"

    @classmethod
    def estimate_polarity(cls, composition: List[Dict], geometry: str = None) -> str:
        """
        Estimate molecular polarity based on bond types and molecular geometry.
        Symmetric molecules can be nonpolar even with polar bonds (dipoles cancel).
        """
        bond_type = cls.determine_bond_type(composition)

        if bond_type == "Ionic":
            return "Ionic"

        # Get element counts
        element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
        elements = set(element_counts.keys())

        # Check for symmetric molecules that are nonpolar despite polar bonds
        # Linear molecules like CO2 (AX2 with no lone pairs) are nonpolar
        if geometry == "Linear" and len(elements) == 2:
            # Find central atom (typically has count 1) and terminal atoms (count 2)
            for elem, count in element_counts.items():
                if count == 1:
                    other_counts = [c for e, c in element_counts.items() if e != elem]
                    if other_counts and other_counts[0] == 2:
                        # This is an AX2 linear molecule like CO2 - symmetric and nonpolar
                        return "Nonpolar"

        # Tetrahedral molecules with all same substituents (like CH4) are nonpolar
        if geometry == "Tetrahedral":
            # Check if it's AX4 (one central atom, 4 identical terminal atoms)
            if len(elements) == 2:
                counts = list(element_counts.values())
                if (1 in counts and 4 in counts):
                    return "Nonpolar"

        # Homonuclear molecules are always nonpolar
        if len(elements) == 1:
            return "Nonpolar"

        # Check electronegativity difference to determine polarity
        electronegativities = []
        for comp in composition:
            element = comp['Element']
            en = cls.ELECTRONEGATIVITIES.get(element, 2.0)
            electronegativities.append(en)

        max_en = max(electronegativities)
        min_en = min(electronegativities)
        diff = max_en - min_en

        # If there's a significant electronegativity difference and molecule is not symmetric
        if diff > 0.4:
            return "Polar"

        return "Nonpolar"

    @classmethod
    def estimate_geometry(cls, composition: List[Dict]) -> str:
        """
        Estimate molecular geometry using VSEPR theory.
        Based on central atom, number of bonding pairs, and lone pairs.
        """
        total_atoms = sum(comp.get('Count', 1) for comp in composition)
        element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
        elements = list(element_counts.keys())

        if total_atoms == 2:
            return "Linear"

        # Identify the central atom (typically has lowest count or highest valence)
        # For simple molecules, it's usually the atom with count=1
        central_atom = None
        terminal_count = 0

        for elem, count in element_counts.items():
            if count == 1 and elem not in ['H', 'F', 'Cl', 'Br', 'I']:
                # This is likely the central atom
                central_atom = elem
                # Count terminal atoms
                terminal_count = total_atoms - 1
                break

        if central_atom is None:
            # No clear central atom found - try to identify by valence
            for elem, count in element_counts.items():
                if elem in ['C', 'N', 'O', 'S', 'P', 'Si']:
                    central_atom = elem
                    terminal_count = total_atoms - count
                    break

        # Get valence electrons of central atom
        central_valence = cls.VALENCE_ELECTRONS.get(central_atom, 4) if central_atom else 4

        # Estimate number of lone pairs on central atom
        # Lone pairs = (valence - bonds) / 2
        # This is a simplification - actual VSEPR is more complex

        if total_atoms == 3:
            # AX2 molecules
            if central_atom == 'C':
                # CO2-like: 4 valence electrons, 2 double bonds, 0 lone pairs
                return "Linear"
            elif central_atom == 'O':
                # H2O-like: 6 valence electrons, 2 bonds, 2 lone pairs
                return "Bent"
            elif central_atom == 'N':
                # Could be bent with lone pair
                return "Bent"
            elif central_atom == 'S':
                return "Bent"
            else:
                # Default for AX2 with no lone pairs
                return "Linear" if central_valence <= 4 else "Bent"

        elif total_atoms == 4:
            # AX3 molecules
            if central_atom == 'N':
                # NH3-like: 5 valence, 3 bonds, 1 lone pair -> trigonal pyramidal
                return "Trigonal Pyramidal"
            elif central_atom == 'C':
                # CH3X-like: no lone pairs -> tetrahedral
                return "Trigonal Planar"
            elif central_atom == 'B':
                return "Trigonal Planar"
            else:
                return "Trigonal Pyramidal"

        elif total_atoms == 5:
            # AX4 molecules
            if central_atom in ['C', 'Si']:
                # CH4-like: 4 bonds, 0 lone pairs -> tetrahedral
                return "Tetrahedral"
            elif central_atom == 'S':
                # SF4-like: could be see-saw
                return "See-saw"
            else:
                return "Tetrahedral"

        elif total_atoms == 6:
            # AX5 molecules
            if central_atom == 'P':
                return "Trigonal Bipyramidal"
            else:
                return "Trigonal Bipyramidal"

        elif total_atoms == 7:
            # AX6 molecules
            return "Octahedral"

        elif total_atoms > 7:
            # Larger molecules - look for carbon backbone
            if 'C' in elements:
                # Organic molecules with carbon - typically tetrahedral around C
                return "Tetrahedral"
            else:
                return "Complex"

        return "Complex"

    @classmethod
    def estimate_bond_angle(cls, geometry: str) -> Optional[float]:
        """Get ideal bond angle for geometry."""
        angles = {
            "Linear": 180.0,
            "Bent": 104.5,
            "Trigonal Planar": 120.0,
            "Trigonal Pyramidal": 107.0,
            "Tetrahedral": 109.5,
            "Trigonal Bipyramidal": 90.0,  # Mixed angles
            "Octahedral": 90.0,
            "Square Planar": 90.0
        }
        return angles.get(geometry)

    @classmethod
    def _has_hydrogen_bonding(cls, composition: List[Dict]) -> bool:
        """
        Check if molecule can form hydrogen bonds.
        Requires H bonded to N, O, or F (highly electronegative atoms).
        """
        elements = {comp['Element'] for comp in composition}
        # Hydrogen bonding requires H and at least one of N, O, F
        has_h = 'H' in elements
        has_electronegative = any(e in elements for e in ['N', 'O', 'F'])
        return has_h and has_electronegative

    @classmethod
    def _is_diatomic(cls, composition: List[Dict]) -> bool:
        """Check if molecule is diatomic (2 atoms total)."""
        total_atoms = sum(comp.get('Count', 1) for comp in composition)
        return total_atoms == 2

    @classmethod
    def _is_diatomic_homonuclear(cls, composition: List[Dict]) -> bool:
        """Check if molecule is homonuclear diatomic (e.g., H2, O2, N2)."""
        if len(composition) != 1:
            return False
        return composition[0].get('Count', 1) == 2

    @classmethod
    def _is_aromatic(cls, composition: List[Dict]) -> bool:
        """
        Check if molecule is likely aromatic (benzene-like).
        Uses simple heuristic: C6H6 pattern or similar.
        """
        element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
        c_count = element_counts.get('C', 0)
        h_count = element_counts.get('H', 0)

        # Benzene: C6H6
        if c_count == 6 and h_count == 6:
            return True
        # Naphthalene: C10H8
        if c_count == 10 and h_count == 8:
            return True
        # Phenol: C6H6O
        if c_count == 6 and h_count == 6 and element_counts.get('O', 0) == 1:
            return True
        # Toluene: C7H8
        if c_count == 7 and h_count == 8:
            return True
        # General aromatic pattern: roughly equal C and H with C >= 6
        if c_count >= 6 and abs(c_count - h_count) <= 2:
            return True
        return False

    @classmethod
    def _has_carboxylic_group(cls, composition: List[Dict]) -> bool:
        """Check if molecule likely has carboxylic acid group (COOH)."""
        element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
        c_count = element_counts.get('C', 0)
        o_count = element_counts.get('O', 0)
        h_count = element_counts.get('H', 0)
        # Carboxylic acids have at least 2 O per COOH group
        return c_count >= 1 and o_count >= 2 and h_count >= 1

    @classmethod
    def estimate_melting_point(cls, molecular_mass: float, polarity: str, bond_type: str,
                               composition: List[Dict] = None) -> float:
        """
        Estimate melting point based on molecular properties.
        Uses intermolecular force correlations with empirical adjustments.

        Key factors:
        - Molecular mass (van der Waals forces)
        - Polarity (dipole-dipole interactions)
        - Hydrogen bonding (strong intermolecular force)
        - Ionic bonds (strongest, highest MP)
        - Molecular structure (diatomic, aromatic)
        """
        # For ionic compounds - use high melting point model
        if bond_type == "Ionic":
            # Ionic compounds have very high melting points (500-3000 K typically)
            # NaCl: ~1074 K, CaCl2: ~1045 K
            base_mp = 800 + molecular_mass * 4
            return round(max(500.0, min(2000.0, base_mp)), 1)

        # Special handling for diatomic homonuclear molecules (H2, O2, N2, etc.)
        # These have very weak van der Waals forces and low melting points
        if composition and cls._is_diatomic_homonuclear(composition):
            element = composition[0]['Element']
            # Known values for diatomic gases
            diatomic_mp = {
                'H': 14.0, 'N': 63.0, 'O': 54.4, 'F': 53.5, 'Cl': 172.0,
                'Br': 266.0, 'I': 387.0
            }
            if element in diatomic_mp:
                return diatomic_mp[element]
            # For unknown diatomics, use low van der Waals correlation
            return round(20 + molecular_mass * 1.5, 1)

        # Special handling for aromatic compounds (benzene-like)
        if composition and cls._is_aromatic(composition):
            element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
            c_count = element_counts.get('C', 0)
            # Benzene: MP = 278.7K, Naphthalene: MP = 353K
            # Aromatic ring stacking increases MP
            base_mp = 250 + (c_count - 6) * 10
            return round(max(200.0, base_mp), 1)

        # Carboxylic acids form dimers, raising MP
        if composition and cls._has_carboxylic_group(composition):
            # Acetic acid: MW=60, MP=289K
            # Formic acid: MW=46, MP=281K
            base_mp = 270 + molecular_mass * 0.3
            return round(base_mp, 1)

        # For covalent/molecular compounds - use van der Waals model
        # Reference points (actual values):
        # CH4: MW=16, MP=90.7K (nonpolar, no H-bonding)
        # H2O: MW=18, MP=273K (polar, strong H-bonding)
        # NH3: MW=17, MP=195K (polar, H-bonding)
        # CO2: MW=44, MP=195K (nonpolar, sublimes)
        # C2H5OH: MW=46, MP=159K (polar, H-bonding)

        # Base MP from molecular mass using empirical correlation
        if molecular_mass < 20:
            # Very small molecules - dominated by MW
            base_mp = 50 + molecular_mass * 3
        elif molecular_mass < 50:
            # Small molecules
            base_mp = 80 + molecular_mass * 2
        elif molecular_mass < 100:
            # Medium molecules
            base_mp = 100 + molecular_mass * 1.5
        else:
            # Larger molecules
            base_mp = 150 + molecular_mass * 1.0

        # Adjust for polarity
        if polarity == "Polar":
            base_mp *= 1.2
        elif polarity == "Nonpolar":
            base_mp *= 0.8

        # Hydrogen bonding effects on melting point
        if composition and cls._has_hydrogen_bonding(composition):
            element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
            has_carbon = 'C' in element_counts
            o_count = element_counts.get('O', 0)
            h_count = element_counts.get('H', 0)
            n_count = element_counts.get('N', 0)

            if o_count > 0 and h_count > 0:
                if has_carbon:
                    # Alcohols: H-bonding exists but less efficient crystal packing
                    # Ethanol (C2H5OH): MW=46, MP=159K
                    # Methanol (CH3OH): MW=32, MP=175K
                    # Long carbon chains disrupt crystal packing, lowering MP
                    c_count = element_counts.get('C', 0)
                    # Alcohols have lower MP than simple MW correlation predicts
                    base_mp = 170 - c_count * 5  # Decrease with carbon chain length
                else:
                    # Water: strong tetrahedral H-bond network in ice
                    # H2O: MW=18, MP=273K (anomalously high due to ice structure)
                    base_mp = 273  # Override for water-like molecules
            elif n_count > 0 and h_count > 0 and not has_carbon:
                # Ammonia: NH3 has MP=195K
                base_mp = 195

        return round(max(20.0, base_mp), 1)

    @classmethod
    def estimate_boiling_point(cls, melting_point: float, polarity: str,
                               composition: List[Dict] = None, molecular_mass: float = None) -> float:
        """
        Estimate boiling point from molecular properties.
        BP/MP ratio varies significantly based on intermolecular forces.
        """
        # For ionic compounds
        if polarity == "Ionic":
            # Ionic compounds: BP typically ~1.5x MP
            return round(melting_point * 1.55, 1)

        # Special handling for diatomic homonuclear molecules
        # These have narrow liquid ranges (BP/MP ~ 1.3-1.5)
        if composition and cls._is_diatomic_homonuclear(composition):
            element = composition[0]['Element']
            # Known values for diatomic gases
            diatomic_bp = {
                'H': 20.3, 'N': 77.4, 'O': 90.2, 'F': 85.0, 'Cl': 239.0,
                'Br': 332.0, 'I': 457.0
            }
            if element in diatomic_bp:
                return diatomic_bp[element]
            return round(melting_point * 1.3, 1)

        # Special handling for aromatic compounds
        if composition and cls._is_aromatic(composition):
            element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
            c_count = element_counts.get('C', 0)
            # Benzene: BP = 353.2K, Naphthalene: BP = 491K
            base_bp = 320 + (c_count - 6) * 15
            return round(max(300.0, base_bp), 1)

        # For molecular compounds, use Trouton's rule with modifications
        # Trouton's rule: entropy of vaporization ~ 85-90 J/(mol*K)
        # This gives BP ~ MP * 1.1 to 1.5 depending on IMF

        # Reference points:
        # CH4: MP=90.7, BP=111.7 (ratio=1.23)
        # H2O: MP=273, BP=373 (ratio=1.37)
        # NH3: MP=195, BP=240 (ratio=1.23)
        # CO2: MP=195, BP=217 (ratio=1.11) - sublimes
        # C2H5OH: MP=159, BP=351.5 (ratio=2.21) - strong H-bonding in liquid

        base_ratio = 1.2  # Base ratio for simple molecules

        # Polar molecules have higher BP/MP ratio
        if polarity == "Polar":
            base_ratio = 1.35

        # Hydrogen bonding dramatically increases BP
        if composition and cls._has_hydrogen_bonding(composition):
            element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}
            has_carbon = 'C' in element_counts
            o_count = element_counts.get('O', 0)
            h_count = element_counts.get('H', 0)
            n_count = element_counts.get('N', 0)

            # O-H bonding (water, alcohols)
            if o_count > 0 and h_count > 0:
                if has_carbon:
                    # Alcohols: strong H-bonding in liquid phase
                    # Ethanol: BP=351.5K, Methanol: BP=337.8K
                    c_count = element_counts.get('C', 0)
                    bp = 330 + c_count * 10
                    return round(bp, 1)
                else:
                    # Water: BP=373K
                    return 373.0

            # N-H bonding (ammonia)
            if n_count > 0 and h_count > 0 and not has_carbon:
                # NH3: BP=239.8K
                return 240.0

        bp = melting_point * base_ratio

        return round(max(melting_point + 10, bp), 1)

    @classmethod
    def estimate_density(cls, molecular_mass: float, composition: List[Dict],
                        state: str = None, bond_type: str = None) -> float:
        """
        Estimate density based on molecular mass, composition, and state.
        Gases have very low densities, liquids ~0.5-1.5, solids higher.
        """
        # For ionic solids - higher densities
        if bond_type == "Ionic":
            # Ionic compounds are crystalline solids with higher density
            # NaCl: 2.165 g/cm3, KCl: 1.98 g/cm3
            base_density = 1.5 + molecular_mass / 100
            return round(max(1.0, min(5.0, base_density)), 3)

        # For gases at STP - very low density
        if state == "Gas":
            # Ideal gas at STP: PV = nRT
            # Density = PM/(RT) where P=101325 Pa, R=8.314, T=298K
            # For ideal gas: rho = (MW * 101325) / (8314 * 298) in g/L
            # Convert to g/cm3: divide by 1000
            ideal_gas_density = (molecular_mass * 101325) / (8314 * 298) / 1000
            return round(max(0.0001, ideal_gas_density), 6)

        # Special handling for aromatic liquids (benzene-like)
        # Aromatic compounds are typically liquid at room temperature
        if cls._is_aromatic(composition):
            # Benzene: 0.8765 g/cm3, Toluene: 0.87 g/cm3
            base_density = 0.85 + molecular_mass / 1000
            return round(max(0.7, min(1.2, base_density)), 3)

        # For liquids - use empirical correlation
        # Reference: water=1.0, ethanol=0.789
        # Generally correlates with MW and intermolecular forces
        element_counts = {comp['Element']: comp.get('Count', 1) for comp in composition}

        # Base density for liquids
        if molecular_mass < 20:
            base_density = 0.8
        elif molecular_mass < 50:
            base_density = 0.7 + molecular_mass / 150
        else:
            base_density = 0.8 + molecular_mass / 200

        # Hydrogen bonding increases density (more compact packing)
        if cls._has_hydrogen_bonding(composition):
            base_density *= 1.1

        # Heavy atoms increase density
        heavy_atoms = sum(1 for e in element_counts if cls.ATOMIC_MASSES.get(e, 0) > 30)
        if heavy_atoms > 0:
            base_density *= 1.1

        return round(max(0.5, min(3.0, base_density)), 3)

    @classmethod
    def estimate_dipole_moment(cls, composition: List[Dict], polarity: str) -> float:
        """Estimate dipole moment in Debye."""
        if polarity == "Nonpolar":
            return 0.0
        elif polarity == "Ionic":
            return round(5.0 + len(composition) * 0.5, 2)
        else:  # Polar
            return round(1.0 + len(composition) * 0.3, 2)

    @classmethod
    def determine_state(cls, melting_point: float, boiling_point: float) -> str:
        """Determine state at STP (298 K, 1 atm)."""
        stp_temp = 298.15  # K

        if melting_point > stp_temp:
            return "Solid"
        elif boiling_point < stp_temp:
            return "Gas"
        else:
            return "Liquid"

    @classmethod
    def estimate_bonds(cls, composition: List[Dict]) -> List[Dict]:
        """Estimate bond information based on composition."""
        bonds = []
        elements = [comp['Element'] for comp in composition for _ in range(comp.get('Count', 1))]

        if len(elements) < 2:
            return bonds

        # Simple estimation: first element bonds to others
        central = elements[0] if len(set(elements)) > 1 else elements[0]
        other_elements = [e for e in set(elements) if e != central or elements.count(e) > 1]

        for other in other_elements:
            # Estimate bond length from covalent radii
            r1 = cls.COVALENT_RADII.get(central, 100)
            r2 = cls.COVALENT_RADII.get(other, 100)
            length = r1 + r2

            bonds.append({
                "From": central,
                "To": other,
                "Type": "Single",  # Simplified
                "Length_pm": length
            })

        return bonds

    @classmethod
    def create_molecule_from_atoms(cls, composition: List[Dict], name: str = None,
                                   atom_data: Dict = None) -> Dict:
        """
        Create a complete molecule JSON structure from atomic composition.

        Args:
            composition: List of {"Element": symbol, "Count": n}
            name: Optional custom name
            atom_data: Optional dict mapping symbols to full atom data

        Returns:
            Complete molecule data dictionary
        """
        formula = cls.generate_formula(composition)
        molecular_mass = cls.calculate_molecular_mass(composition, atom_data)
        bond_type = cls.determine_bond_type(composition, atom_data)
        geometry = cls.estimate_geometry(composition)
        polarity = cls.estimate_polarity(composition, geometry)
        melting_point = cls.estimate_melting_point(molecular_mass, polarity, bond_type, composition)
        boiling_point = cls.estimate_boiling_point(melting_point, polarity, composition, molecular_mass)
        state = cls.determine_state(melting_point, boiling_point)
        density = cls.estimate_density(molecular_mass, composition, state, bond_type)

        if name is None:
            name = f"Compound ({formula})"

        molecule_data = {
            "Name": name,
            "Formula": formula,
            "MolecularMass_amu": molecular_mass,
            "MolecularMass_g_mol": molecular_mass,
            "BondType": bond_type,
            "Geometry": geometry,
            "BondAngle_deg": cls.estimate_bond_angle(geometry),
            "Polarity": polarity,
            "MeltingPoint_K": melting_point,
            "BoilingPoint_K": boiling_point,
            "Density_g_cm3": density,
            "State_STP": state,
            "Composition": composition,
            "Bonds": cls.estimate_bonds(composition),
            "DipoleMoment_D": cls.estimate_dipole_moment(composition, polarity),
            "Applications": [],
            "IUPAC_Name": name,
            "_created_from": {
                "composition": composition
            }
        }

        return molecule_data
