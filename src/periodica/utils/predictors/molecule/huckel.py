"""
Huckel Molecular Orbital Calculator for Conjugated Pi Systems

Calculates resonance (delocalization) energy for cyclic conjugated systems
using the Huckel approximation. No per-atom lookup data needed — only the
ring size and number of pi electrons.

Formula:
    E_pi = 2 * sum_{k=0}^{n_occ-1} [alpha + 2*beta*cos(2*pi*k/n)]
    E_loc = n_electrons * (alpha + beta)  (localized reference)
    E_res = E_pi - E_loc  (resonance stabilization)

Where:
    beta ≈ -75 kJ/mol (C-C pi overlap integral, model constant)
    alpha is the Coulomb integral (cancels in E_res)
    n = ring size (number of atoms in conjugated ring)
    n_occ = number of occupied MOs = n_electrons / 2
"""

import math
from typing import Optional


# Model constant: resonance integral for C-C pi overlap
# This is a universal physical parameter, not per-element
BETA_KJMOL = -75.0  # kJ/mol


def calculate_huckel_energies(ring_size: int, n_pi_electrons: Optional[int] = None):
    """
    Calculate Huckel MO energies for a cyclic conjugated system.

    Args:
        ring_size: Number of atoms in the conjugated ring
        n_pi_electrons: Number of pi electrons (default: ring_size for
            annulenes, i.e., one pi electron per atom)

    Returns:
        Dict with:
            - orbital_energies: list of MO energies in units of beta
            - total_pi_energy: total pi energy in units of beta
            - localized_energy: energy of n_pi/2 isolated double bonds
            - resonance_energy_beta: delocalization energy in units of beta
            - resonance_energy_kjmol: delocalization energy in kJ/mol
    """
    if ring_size < 3:
        return {
            'orbital_energies': [],
            'total_pi_energy': 0.0,
            'localized_energy': 0.0,
            'resonance_energy_beta': 0.0,
            'resonance_energy_kjmol': 0.0,
        }

    if n_pi_electrons is None:
        n_pi_electrons = ring_size  # Default: one pi electron per atom

    # Huckel MO energies for cyclic system: E_k = alpha + 2*beta*cos(2*pi*k/n)
    # In units of beta (alpha cancels in resonance energy):
    # e_k = 2*cos(2*pi*k/n)
    orbital_energies = []
    for k in range(ring_size):
        e_k = 2.0 * math.cos(2.0 * math.pi * k / ring_size)
        orbital_energies.append(e_k)

    # Sort by energy (most bonding first, i.e., largest positive value)
    orbital_energies.sort(reverse=True)

    # Fill electrons into orbitals (2 per orbital)
    total_pi = 0.0
    electrons_remaining = n_pi_electrons
    for e_k in orbital_energies:
        if electrons_remaining <= 0:
            break
        occupancy = min(2, electrons_remaining)
        total_pi += occupancy * e_k
        electrons_remaining -= occupancy

    # Localized reference: each double bond contributes 2*(alpha + beta)
    # In beta units: each localized pi bond = 2*1.0 = 2.0 per electron pair
    n_double_bonds = n_pi_electrons // 2
    localized_energy = n_double_bonds * 2.0  # 2 beta per double bond

    # Resonance energy
    resonance_beta = total_pi - localized_energy
    resonance_kjmol = resonance_beta * BETA_KJMOL

    return {
        'orbital_energies': orbital_energies,
        'total_pi_energy': total_pi,
        'localized_energy': localized_energy,
        'resonance_energy_beta': round(resonance_beta, 4),
        'resonance_energy_kjmol': round(resonance_kjmol, 1),
    }


def is_aromatic(ring_size: int, n_pi_electrons: Optional[int] = None) -> bool:
    """
    Check if a ring system satisfies Huckel's rule for aromaticity.

    Huckel's rule: A planar cyclic conjugated system is aromatic if
    it has (4n + 2) pi electrons, where n is a non-negative integer.

    Args:
        ring_size: Number of atoms in the ring
        n_pi_electrons: Number of pi electrons (default: ring_size)

    Returns:
        True if the system is aromatic
    """
    if n_pi_electrons is None:
        n_pi_electrons = ring_size
    return n_pi_electrons > 0 and (n_pi_electrons - 2) % 4 == 0
