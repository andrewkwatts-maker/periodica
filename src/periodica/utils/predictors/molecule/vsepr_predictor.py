"""
VSEPR (Valence Shell Electron Pair Repulsion) Predictor Module

Predicts molecular geometry and properties based on VSEPR theory.
Uses the number of bonding pairs and lone pairs around a central atom
to determine molecular shape and bond angles.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseMoleculePredictor, MoleculeInput, MoleculeResult
from ..registry import register_predictor


# VSEPR Geometry Lookup Table
# Key: (bonding_pairs, lone_pairs)
# Value: (geometry_name, bond_angle_degrees)
VSEPR_GEOMETRY = {
    # Linear geometries
    (2, 0): ('Linear', 180.0),

    # Trigonal planar geometries
    (3, 0): ('Trigonal Planar', 120.0),
    (2, 1): ('Bent', 117.0),

    # Tetrahedral geometries
    (4, 0): ('Tetrahedral', 109.5),
    (3, 1): ('Trigonal Pyramidal', 107.0),
    (2, 2): ('Bent', 104.5),

    # Trigonal bipyramidal geometries
    (5, 0): ('Trigonal Bipyramidal', 90.0),  # Axial: 180, Equatorial: 120
    (4, 1): ('Seesaw', 117.0),
    (3, 2): ('T-Shaped', 90.0),
    (2, 3): ('Linear', 180.0),

    # Octahedral geometries
    (6, 0): ('Octahedral', 90.0),
    (5, 1): ('Square Pyramidal', 90.0),
    (4, 2): ('Square Planar', 90.0),
}

# Atomic masses in amu for common elements
ATOMIC_MASSES = {
    'H': 1.008,
    'He': 4.003,
    'Li': 6.941,
    'Be': 9.012,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.086,
    'P': 30.974,
    'S': 32.065,
    'Cl': 35.453,
    'Ar': 39.948,
    'K': 39.098,
    'Ca': 40.078,
    'Br': 79.904,
    'Kr': 83.798,
    'I': 126.904,
    'Xe': 131.293,
}

# Valence electrons for common elements
VALENCE_ELECTRONS = {
    'H': 1, 'He': 2,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 1, 'Ca': 2, 'Br': 7, 'Kr': 8,
    'I': 7, 'Xe': 8,
}


@register_predictor('molecule', 'vsepr')
@register_predictor('molecule', 'default')
class VSEPRPredictor(BaseMoleculePredictor):
    """
    Predictor for molecular geometry using VSEPR theory.

    Uses the Valence Shell Electron Pair Repulsion theory to predict
    molecular geometry based on the number of bonding pairs and lone
    pairs around the central atom.
    """

    @property
    def name(self) -> str:
        """Return the name of this predictor."""
        return "VSEPR Predictor"

    @property
    def description(self) -> str:
        """Return a description of what this predictor does."""
        return (
            "Predicts molecular geometry using VSEPR (Valence Shell Electron "
            "Pair Repulsion) theory based on bonding pairs and lone pairs."
        )

    def determine_geometry(self, bonding_pairs: int, lone_pairs: int) -> Tuple[str, float]:
        """
        Determine molecular geometry based on VSEPR theory.

        Args:
            bonding_pairs: Number of bonding pairs around the central atom
            lone_pairs: Number of lone pairs on the central atom

        Returns:
            Tuple of (geometry_name, bond_angle_degrees)
        """
        key = (bonding_pairs, lone_pairs)

        if key in VSEPR_GEOMETRY:
            return VSEPR_GEOMETRY[key]

        # Default to unknown geometry with tetrahedral-like angle
        return ('Unknown', 109.5)

    def calculate_molecular_mass(self, atoms: List[Dict[str, Any]]) -> float:
        """
        Calculate the molecular mass from a list of atoms.

        Args:
            atoms: List of atom dictionaries, each containing at least
                   an 'element' key with the element symbol

        Returns:
            Molecular mass in atomic mass units (amu)
        """
        total_mass = 0.0

        for atom in atoms:
            element = atom.get('element', '')
            # Use default mass of 12 (carbon) if element not found
            mass = ATOMIC_MASSES.get(element, 12.0)
            total_mass += mass

        return total_mass

    def _get_central_atom(self, atoms: List[Dict], bonds: List[Dict]) -> Optional[int]:
        """
        Identify the central atom (atom with most bonds).

        Args:
            atoms: List of atom dictionaries
            bonds: List of bond dictionaries

        Returns:
            Index of the central atom, or None if no atoms
        """
        if not atoms:
            return None

        # Count bonds for each atom
        bond_counts = {i: 0 for i in range(len(atoms))}

        for bond in bonds:
            from_idx = bond.get('from', -1)
            to_idx = bond.get('to', -1)

            if from_idx >= 0 and from_idx < len(atoms):
                bond_counts[from_idx] += 1
            if to_idx >= 0 and to_idx < len(atoms):
                bond_counts[to_idx] += 1

        # Return atom with most bonds
        if bond_counts:
            return max(bond_counts, key=bond_counts.get)
        return 0

    def _count_bonding_pairs(self, central_idx: int, atoms: List[Dict],
                             bonds: List[Dict]) -> int:
        """
        Count bonding pairs around the central atom.

        Args:
            central_idx: Index of the central atom
            atoms: List of atom dictionaries
            bonds: List of bond dictionaries

        Returns:
            Number of bonding pairs (atoms bonded to central)
        """
        bonding_partners = set()

        for bond in bonds:
            from_idx = bond.get('from', -1)
            to_idx = bond.get('to', -1)

            if from_idx == central_idx:
                bonding_partners.add(to_idx)
            elif to_idx == central_idx:
                bonding_partners.add(from_idx)

        return len(bonding_partners)

    def _calculate_lone_pairs(self, element: str, bonding_electrons: int) -> int:
        """
        Calculate lone pairs on an atom.

        Args:
            element: Element symbol
            bonding_electrons: Number of electrons used in bonding

        Returns:
            Number of lone pairs
        """
        valence = VALENCE_ELECTRONS.get(element, 4)
        remaining = valence - bonding_electrons
        return max(0, remaining // 2)

    def _calculate_bonding_electrons(self, central_idx: int,
                                     bonds: List[Dict]) -> int:
        """
        Calculate total bonding electrons for the central atom.

        Args:
            central_idx: Index of the central atom
            bonds: List of bond dictionaries

        Returns:
            Number of electrons used in bonding
        """
        bond_multiplicity = {'single': 1, 'double': 2, 'triple': 3, 'aromatic': 1.5}
        bonding_electrons = 0

        for bond in bonds:
            from_idx = bond.get('from', -1)
            to_idx = bond.get('to', -1)

            if from_idx == central_idx or to_idx == central_idx:
                bond_type = bond.get('type', 'single')
                bonding_electrons += bond_multiplicity.get(bond_type, 1)

        return int(bonding_electrons)

    def _calculate_dipole_moment(self, atoms: List[Dict], bonds: List[Dict],
                                  geometry: str) -> float:
        """
        Calculate dipole moment using bond dipole vector summation.

        Each bond dipole is: mu_bond = delta_EN * d_bond (in Debye approx)
        where delta_EN is the electronegativity difference and d_bond is the
        bond length estimated from covalent radii (read from atom data when
        available). Vectors are placed at VSEPR angles and summed geometrically.

        Conversion: 1 Debye ≈ 3.336e-30 C·m
        Empirical: mu(D) ≈ delta_EN * d_bond(Å) * scaling_factor

        Args:
            atoms: List of atom dictionaries (from element data sheets)
            bonds: List of bond dictionaries
            geometry: The molecular geometry name

        Returns:
            Dipole moment in Debye
        """
        import math

        if len(atoms) <= 1:
            return 0.0

        # Default electronegativity and covalent radius tables
        # Used only when atom data doesn't provide these values
        _DEFAULT_EN = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96,
            'I': 2.66, 'B': 2.04, 'Li': 0.98, 'Na': 0.93, 'K': 0.82,
        }
        _DEFAULT_COVALENT_R = {
            'H': 31, 'C': 77, 'N': 75, 'O': 73, 'F': 71, 'Cl': 99,
            'Br': 114, 'I': 133, 'S': 102, 'P': 107, 'Si': 117, 'B': 82,
        }

        def _get_en(atom_data):
            """Read electronegativity from atom data, fallback to default."""
            en = atom_data.get('electronegativity', atom_data.get('Electronegativity'))
            if en is not None:
                return float(en)
            return _DEFAULT_EN.get(atom_data.get('element', ''), 2.5)

        def _get_cov_r(atom_data):
            """Read covalent radius from atom data, fallback to default."""
            r = atom_data.get('covalent_radius', atom_data.get('CovalentRadius_pm'))
            if r is not None:
                return float(r)
            return _DEFAULT_COVALENT_R.get(atom_data.get('element', ''), 77)

        # Get central atom info
        elements = [atom.get('element', '') for atom in atoms]
        central_element = elements[0] if elements else ''
        central_en = _get_en(atoms[0]) if atoms else 2.5
        central_r = _get_cov_r(atoms[0]) if atoms else 77

        # Check symmetric case first (optimization)
        bonded_elements = elements[1:] if len(elements) > 1 else []
        symmetric_geometries = {'Linear', 'Trigonal Planar', 'Tetrahedral',
                                'Trigonal Bipyramidal', 'Octahedral', 'Square Planar'}
        if geometry in symmetric_geometries and len(set(bonded_elements)) <= 1:
            return 0.0

        # VSEPR unit vectors for each geometry (2D/3D simplified to magnitudes
        # and angles for vector summation)
        # Returns list of (angle_degrees) for bonds around central atom
        geometry_angles = {
            'Linear': [0, 180],
            'Bent': [0, 104.5],
            'Trigonal Planar': [0, 120, 240],
            'Trigonal Pyramidal': [0, 107, 214],
            'Tetrahedral': [0, 109.5, 219, 328.5],
            'Seesaw': [0, 90, 180, 270],
            'T-Shaped': [0, 90, 180],
            'Square Planar': [0, 90, 180, 270],
            'Square Pyramidal': [0, 90, 180, 270, 0],  # 5th bond is axial
            'Octahedral': [0, 90, 180, 270, 0, 180],
        }
        angles = geometry_angles.get(geometry, [i * 360 / max(1, len(bonded_elements))
                                                 for i in range(len(bonded_elements))])

        # Calculate bond dipole vectors and sum
        mu_x, mu_y = 0.0, 0.0
        for i, bonded_atom in enumerate(atoms[1:]):
            bonded_en = _get_en(bonded_atom)
            bonded_r = _get_cov_r(bonded_atom)

            # Bond length from sum of covalent radii (pm -> Angstrom)
            d_bond = (central_r + bonded_r) / 100.0  # pm to Angstrom

            # Bond dipole magnitude: mu = delta_EN * d_bond
            # (empirical scaling: 1 EN unit × 1 Å ≈ 1 D for most bonds)
            delta_en = bonded_en - central_en
            mu_bond = delta_en * d_bond

            # Project onto 2D (sufficient for most geometries)
            angle_rad = math.radians(angles[i % len(angles)])
            mu_x += mu_bond * math.cos(angle_rad)
            mu_y += mu_bond * math.sin(angle_rad)

        total_dipole = math.sqrt(mu_x ** 2 + mu_y ** 2)
        return round(total_dipole, 2)

    def predict(self, input_data: MoleculeInput) -> MoleculeResult:
        """
        Make molecular predictions from input data.

        Args:
            input_data: MoleculeInput containing atoms and bonds

        Returns:
            MoleculeResult with geometry, bond angles, dipole moment, and mass
        """
        atoms = input_data.atoms
        bonds = input_data.bonds

        # Find central atom
        central_idx = self._get_central_atom(atoms, bonds)

        if central_idx is None or not atoms:
            return MoleculeResult(
                geometry='Unknown',
                bond_angles=[],
                dipole_moment=0.0,
                molecular_mass_amu=0.0
            )

        # Get central atom element
        central_element = atoms[central_idx].get('element', 'C')

        # Count bonding pairs and calculate lone pairs
        bonding_pairs = self._count_bonding_pairs(central_idx, atoms, bonds)
        bonding_electrons = self._calculate_bonding_electrons(central_idx, bonds)
        lone_pairs = self._calculate_lone_pairs(central_element, bonding_electrons)

        # Determine geometry
        geometry, bond_angle = self.determine_geometry(bonding_pairs, lone_pairs)

        # Calculate molecular mass
        molecular_mass = self.calculate_molecular_mass(atoms)

        # Estimate dipole moment
        dipole_moment = self._calculate_dipole_moment(atoms, bonds, geometry)

        # Determine hybridization for each atom
        hybridization_map = {}
        steric_number = bonding_pairs + lone_pairs
        hybridization_lookup = {2: 'sp', 3: 'sp2', 4: 'sp3', 5: 'sp3d', 6: 'sp3d2'}
        hybridization_map[central_idx] = hybridization_lookup.get(steric_number, 'sp3')

        return MoleculeResult(
            geometry=geometry,
            bond_angles=[bond_angle] if bond_angle else [],
            dipole_moment=dipole_moment,
            molecular_mass_amu=round(molecular_mass, 3),
            hybridization=hybridization_map
        )

    def get_confidence(self, input_data: MoleculeInput, result: MoleculeResult) -> float:
        """
        Calculate confidence level for a prediction.

        VSEPR predictions are highly reliable for main group elements
        with simple bonding patterns.

        Args:
            input_data: The input data used for prediction
            result: The prediction result

        Returns:
            Confidence level between 0.0 and 1.0
        """
        if result.geometry == 'Unknown':
            return 0.3

        # High confidence for common geometries
        high_confidence_geometries = {
            'Linear', 'Trigonal Planar', 'Tetrahedral',
            'Trigonal Pyramidal', 'Bent', 'Octahedral'
        }

        if result.geometry in high_confidence_geometries:
            return 0.95

        # Medium confidence for less common geometries
        return 0.80

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """
        Validate molecule input data.

        Args:
            input_data: The input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Use parent validation first
        is_valid, error_msg = super().validate(input_data)
        if not is_valid:
            return is_valid, error_msg

        # Additional VSEPR-specific validation
        atoms = input_data.atoms

        # Check that atoms have element symbols
        for i, atom in enumerate(atoms):
            if 'element' not in atom:
                return False, f"Atom at index {i} missing 'element' key"

        return True, ""
