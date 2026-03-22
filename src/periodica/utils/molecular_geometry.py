"""
Molecular Geometry Calculator

Calculates 3D atom positions for molecules based on VSEPR
(Valence Shell Electron Pair Repulsion) theory and standard bond lengths.

Pure Python implementation with no external dependencies.
"""

import math
from typing import Dict, List, Optional, Tuple


# Standard bond lengths in Angstroms
BOND_LENGTHS = {
    # Carbon bonds
    ('C', 'H', 'single'): 1.09,
    ('C', 'C', 'single'): 1.54,
    ('C', 'C', 'double'): 1.34,
    ('C', 'C', 'aromatic'): 1.40,  # Benzene C-C: experimental 1.395Å
    ('C', 'C', 'triple'): 1.20,
    ('C', 'O', 'single'): 1.43,
    ('C', 'O', 'double'): 1.16,  # CO2 experimental: 1.16Å
    ('C', 'N', 'single'): 1.47,
    ('C', 'N', 'double'): 1.27,
    ('C', 'N', 'triple'): 1.16,
    ('C', 'F', 'single'): 1.35,
    ('C', 'Cl', 'single'): 1.77,
    ('C', 'Br', 'single'): 1.94,
    ('C', 'I', 'single'): 2.14,
    ('C', 'S', 'single'): 1.82,
    ('C', 'S', 'double'): 1.60,

    # Oxygen bonds
    ('O', 'H', 'single'): 0.96,
    ('O', 'O', 'single'): 1.48,
    ('O', 'O', 'double'): 1.21,

    # Nitrogen bonds
    ('N', 'H', 'single'): 1.01,
    ('N', 'N', 'single'): 1.45,
    ('N', 'N', 'double'): 1.25,
    ('N', 'N', 'triple'): 1.10,
    ('N', 'O', 'single'): 1.40,
    ('N', 'O', 'double'): 1.21,

    # Sulfur bonds
    ('S', 'H', 'single'): 1.34,
    ('S', 'O', 'single'): 1.43,
    ('S', 'O', 'double'): 1.43,
    ('S', 'F', 'single'): 1.56,
    ('S', 'S', 'single'): 2.05,

    # Phosphorus bonds
    ('P', 'H', 'single'): 1.42,
    ('P', 'O', 'single'): 1.63,
    ('P', 'O', 'double'): 1.48,
    ('P', 'F', 'single'): 1.54,
    ('P', 'Cl', 'single'): 2.04,

    # Halogen bonds
    ('F', 'F', 'single'): 1.42,
    ('Cl', 'Cl', 'single'): 1.99,
    ('Br', 'Br', 'single'): 2.28,
    ('I', 'I', 'single'): 2.67,
    ('H', 'F', 'single'): 0.92,
    ('H', 'Cl', 'single'): 1.27,
    ('H', 'Br', 'single'): 1.41,
    ('H', 'I', 'single'): 1.61,

    # Boron bonds
    ('B', 'H', 'single'): 1.19,
    ('B', 'F', 'single'): 1.30,
    ('B', 'O', 'single'): 1.36,

    # Silicon bonds
    ('Si', 'H', 'single'): 1.48,
    ('Si', 'C', 'single'): 1.87,
    ('Si', 'O', 'single'): 1.63,
    ('Si', 'F', 'single'): 1.60,
}

# Valence electrons for common elements
VALENCE_ELECTRONS = {
    'H': 1, 'He': 2,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 1, 'Ca': 2, 'Br': 7, 'Kr': 8,
    'I': 7, 'Xe': 8,
}

# Electronegativity values (Pauling scale)
ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96,
    'I': 2.66, 'B': 2.04, 'Be': 1.57, 'Li': 0.98, 'Na': 0.93,
    'K': 0.82, 'Mg': 1.31, 'Ca': 1.00, 'Al': 1.61,
}

# Geometry configurations based on steric number and lone pairs
# VSEPR theory: steric number = bonding pairs + lone pairs
GEOMETRY_CONFIG = {
    # (steric_number, lone_pairs): (geometry_name, bond_angle, coordinate_generator)
    # Steric number 2
    (2, 0): ('Linear', 180.0, 'linear'),  # BeCl2, CO2

    # Steric number 3
    (3, 0): ('Trigonal Planar', 120.0, 'trigonal_planar'),  # BF3, BH3
    (3, 1): ('Bent', 117.0, 'bent'),  # SO2, NO2- (sp2 bent)

    # Steric number 4
    (4, 0): ('Tetrahedral', 109.5, 'tetrahedral'),  # CH4, CCl4
    (4, 1): ('Trigonal Pyramidal', 107.0, 'trigonal_pyramidal'),  # NH3, PH3
    (4, 2): ('Bent', 104.5, 'bent'),  # H2O, H2S

    # Steric number 5
    (5, 0): ('Trigonal Bipyramidal', 90.0, 'trigonal_bipyramidal'),  # PCl5
    (5, 1): ('Seesaw', 117.0, 'seesaw'),  # SF4
    (5, 2): ('T-Shaped', 90.0, 't_shaped'),  # ClF3
    (5, 3): ('Linear', 180.0, 'linear'),  # XeF2

    # Steric number 6
    (6, 0): ('Octahedral', 90.0, 'octahedral'),  # SF6
    (6, 1): ('Square Pyramidal', 90.0, 'square_pyramidal'),  # BrF5
    (6, 2): ('Square Planar', 90.0, 'square_planar'),  # XeF4
}

# Hybridization based on steric number
HYBRIDIZATION = {
    2: 'sp',
    3: 'sp2',
    4: 'sp3',
    5: 'sp3d',
    6: 'sp3d2',
}


class MolecularGeometryCalculator:
    """
    Calculator for molecular geometry using VSEPR theory.

    Computes 3D positions of atoms in molecules based on:
    - Number of bonding pairs
    - Number of lone pairs
    - Standard bond lengths
    - VSEPR geometry rules
    """

    def __init__(self):
        """Initialize the calculator with default parameters."""
        self.bond_lengths = BOND_LENGTHS.copy()
        self.valence_electrons = VALENCE_ELECTRONS.copy()

    def get_bond_length(self, element1: str, element2: str,
                        bond_type: str = 'single') -> float:
        """
        Get the standard bond length between two elements.

        Args:
            element1: First element symbol
            element2: Second element symbol
            bond_type: Type of bond ('single', 'double', 'triple')

        Returns:
            Bond length in Angstroms
        """
        # Try both orderings
        key1 = (element1, element2, bond_type)
        key2 = (element2, element1, bond_type)

        if key1 in self.bond_lengths:
            return self.bond_lengths[key1]
        if key2 in self.bond_lengths:
            return self.bond_lengths[key2]

        # Estimate from covalent radii if not found
        return self._estimate_bond_length(element1, element2, bond_type)

    def _estimate_bond_length(self, element1: str, element2: str,
                              bond_type: str) -> float:
        """Estimate bond length from covalent radii."""
        # Approximate covalent radii in Angstroms
        covalent_radii = {
            'H': 0.31, 'C': 0.77, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20,
            'I': 1.39, 'B': 0.84, 'Be': 0.96, 'Li': 1.28, 'Na': 1.66,
            'K': 2.03, 'Mg': 1.41, 'Ca': 1.76, 'Al': 1.21,
        }

        r1 = covalent_radii.get(element1, 1.0)
        r2 = covalent_radii.get(element2, 1.0)

        # Adjust for bond order
        multiplier = {'single': 1.0, 'double': 0.87, 'triple': 0.78}
        return (r1 + r2) * multiplier.get(bond_type, 1.0)

    def get_valence_electrons(self, element: str) -> int:
        """Get the number of valence electrons for an element."""
        return self.valence_electrons.get(element, 4)

    def calculate_lone_pairs(self, element: str, bonding_electrons: int) -> int:
        """
        Calculate the number of lone pairs on an atom.

        Args:
            element: Element symbol
            bonding_electrons: Number of electrons used in bonds

        Returns:
            Number of lone pairs
        """
        valence = self.get_valence_electrons(element)
        remaining = valence - bonding_electrons
        return max(0, remaining // 2)

    def determine_hybridization(self, bonding_pairs: int, lone_pairs: int) -> str:
        """
        Determine the hybridization of a central atom.

        Args:
            bonding_pairs: Number of bonding pairs
            lone_pairs: Number of lone pairs

        Returns:
            Hybridization type (sp, sp2, sp3, sp3d, sp3d2)
        """
        steric_number = bonding_pairs + lone_pairs
        return HYBRIDIZATION.get(steric_number, 'sp3')

    def determine_geometry(self, bonding_pairs: int, lone_pairs: int) -> Tuple[str, float]:
        """
        Determine molecular geometry based on VSEPR theory.

        Args:
            bonding_pairs: Number of bonding pairs around central atom
            lone_pairs: Number of lone pairs on central atom

        Returns:
            Tuple of (geometry_name, bond_angle)
        """
        steric_number = bonding_pairs + lone_pairs
        key = (steric_number, lone_pairs)

        if key in GEOMETRY_CONFIG:
            name, angle, _ = GEOMETRY_CONFIG[key]
            return name, angle

        # Default to tetrahedral-like
        return 'Unknown', 109.5

    def _generate_linear_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for linear geometry (180 degrees)."""
        positions = []
        if len(bond_lengths) >= 1:
            positions.append((bond_lengths[0], 0.0, 0.0))
        if len(bond_lengths) >= 2:
            positions.append((-bond_lengths[1], 0.0, 0.0))
        return positions

    def _generate_bent_positions(self, bond_lengths: List[float],
                                  angle_deg: float) -> List[Tuple[float, float, float]]:
        """Generate positions for bent geometry."""
        angle_rad = math.radians(angle_deg)
        positions = []

        if len(bond_lengths) >= 1:
            positions.append((bond_lengths[0], 0.0, 0.0))
        if len(bond_lengths) >= 2:
            x = bond_lengths[1] * math.cos(angle_rad)
            y = bond_lengths[1] * math.sin(angle_rad)
            positions.append((x, y, 0.0))

        return positions

    def _generate_trigonal_planar_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for trigonal planar geometry (120 degrees)."""
        positions = []
        angle = 120.0

        for i, length in enumerate(bond_lengths[:3]):
            angle_rad = math.radians(i * angle)
            x = length * math.cos(angle_rad)
            y = length * math.sin(angle_rad)
            positions.append((x, y, 0.0))

        return positions

    def _generate_trigonal_pyramidal_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for trigonal pyramidal geometry (107 degrees)."""
        positions = []
        bond_angle = 107.0

        # Calculate positions using spherical coordinates
        # The angle from the vertical axis
        theta = math.radians(180 - bond_angle)

        for i, length in enumerate(bond_lengths[:3]):
            phi = math.radians(i * 120)
            x = length * math.sin(theta) * math.cos(phi)
            y = length * math.sin(theta) * math.sin(phi)
            z = -length * math.cos(theta)
            positions.append((x, y, z))

        return positions

    def _generate_tetrahedral_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for tetrahedral geometry (109.5 degrees)."""
        positions = []

        # Tetrahedral angle
        angle = math.radians(109.47)

        # First atom along +z
        if len(bond_lengths) >= 1:
            positions.append((0.0, 0.0, bond_lengths[0]))

        # Remaining atoms equally spaced around z-axis, angled down
        for i, length in enumerate(bond_lengths[1:4]):
            phi = math.radians(i * 120)
            z = -length * math.cos(math.pi - angle)
            r = length * math.sin(math.pi - angle)
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            positions.append((x, y, z))

        return positions

    def _generate_trigonal_bipyramidal_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for trigonal bipyramidal geometry."""
        positions = []

        # Axial positions (top and bottom)
        if len(bond_lengths) >= 1:
            positions.append((0.0, 0.0, bond_lengths[0]))
        if len(bond_lengths) >= 2:
            positions.append((0.0, 0.0, -bond_lengths[1]))

        # Equatorial positions (120 degrees apart in xy plane)
        for i, length in enumerate(bond_lengths[2:5]):
            angle_rad = math.radians(i * 120)
            x = length * math.cos(angle_rad)
            y = length * math.sin(angle_rad)
            positions.append((x, y, 0.0))

        return positions

    def _generate_octahedral_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for octahedral geometry (90 degrees)."""
        positions = []

        # Six directions: +x, -x, +y, -y, +z, -z
        directions = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        for i, length in enumerate(bond_lengths[:6]):
            dx, dy, dz = directions[i]
            positions.append((length * dx, length * dy, length * dz))

        return positions

    def _generate_square_planar_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for square planar geometry."""
        positions = []

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for i, length in enumerate(bond_lengths[:4]):
            dx, dy = directions[i]
            positions.append((length * dx, length * dy, 0.0))

        return positions

    def _generate_square_pyramidal_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for square pyramidal geometry."""
        positions = []

        # Apical position
        if len(bond_lengths) >= 1:
            positions.append((0.0, 0.0, bond_lengths[0]))

        # Basal positions (square in xy plane)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for i, length in enumerate(bond_lengths[1:5]):
            dx, dy = directions[i]
            positions.append((length * dx, length * dy, 0.0))

        return positions

    def _generate_seesaw_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for seesaw geometry."""
        positions = []

        # Axial positions
        if len(bond_lengths) >= 1:
            positions.append((0.0, 0.0, bond_lengths[0]))
        if len(bond_lengths) >= 2:
            positions.append((0.0, 0.0, -bond_lengths[1]))

        # Equatorial positions (about 117 degrees apart)
        angle = math.radians(117 / 2)
        if len(bond_lengths) >= 3:
            positions.append((bond_lengths[2] * math.cos(angle),
                            bond_lengths[2] * math.sin(angle), 0.0))
        if len(bond_lengths) >= 4:
            positions.append((bond_lengths[3] * math.cos(angle),
                            -bond_lengths[3] * math.sin(angle), 0.0))

        return positions

    def _generate_t_shaped_positions(self, bond_lengths: List[float]) -> List[Tuple[float, float, float]]:
        """Generate positions for T-shaped geometry."""
        positions = []

        # Axial positions
        if len(bond_lengths) >= 1:
            positions.append((0.0, 0.0, bond_lengths[0]))
        if len(bond_lengths) >= 2:
            positions.append((0.0, 0.0, -bond_lengths[1]))

        # Single equatorial position
        if len(bond_lengths) >= 3:
            positions.append((bond_lengths[2], 0.0, 0.0))

        return positions

    def generate_positions(self, geometry_type: str, bond_lengths: List[float],
                           bond_angle: float = None) -> List[Tuple[float, float, float]]:
        """
        Generate 3D positions based on geometry type.

        Args:
            geometry_type: Name of the geometry
            bond_lengths: List of bond lengths for each bonded atom
            bond_angle: Override bond angle (optional)

        Returns:
            List of (x, y, z) positions
        """
        geometry_type_lower = geometry_type.lower().replace(' ', '_').replace('-', '_')

        generators = {
            'linear': self._generate_linear_positions,
            'bent': lambda bl: self._generate_bent_positions(bl, bond_angle or 104.5),
            'bent_sp2': lambda bl: self._generate_bent_positions(bl, 120.0),
            'bent_sp3': lambda bl: self._generate_bent_positions(bl, 104.5),
            'trigonal_planar': self._generate_trigonal_planar_positions,
            'trigonal_pyramidal': self._generate_trigonal_pyramidal_positions,
            'tetrahedral': self._generate_tetrahedral_positions,
            'trigonal_bipyramidal': self._generate_trigonal_bipyramidal_positions,
            'octahedral': self._generate_octahedral_positions,
            'square_planar': self._generate_square_planar_positions,
            'square_pyramidal': self._generate_square_pyramidal_positions,
            'seesaw': self._generate_seesaw_positions,
            't_shaped': self._generate_t_shaped_positions,
        }

        generator = generators.get(geometry_type_lower)
        if generator:
            return generator(bond_lengths)

        # Default to tetrahedral
        return self._generate_tetrahedral_positions(bond_lengths)

    def calculate_structure(self, composition: List[Dict],
                           bonds: List[Dict]) -> Dict:
        """
        Calculate 3D molecular structure.

        Args:
            composition: List of atom dictionaries with 'element' key
                        e.g., [{'element': 'C'}, {'element': 'H'}, ...]
            bonds: List of bond dictionaries with 'from', 'to', 'type' keys
                   e.g., [{'from': 0, 'to': 1, 'type': 'single'}, ...]

        Returns:
            Dictionary with 'atoms', 'bonds', 'geometry', 'bond_angle' keys
        """
        if not composition:
            return {'atoms': [], 'bonds': [], 'geometry': 'None', 'bond_angle': 0.0}

        # Find central atom (atom with most bonds, or first atom if tied)
        bond_counts = {i: 0 for i in range(len(composition))}
        for bond in bonds:
            bond_counts[bond['from']] = bond_counts.get(bond['from'], 0) + 1
            bond_counts[bond['to']] = bond_counts.get(bond['to'], 0) + 1

        central_idx = max(bond_counts, key=bond_counts.get) if bond_counts else 0
        central_element = composition[central_idx]['element']

        # Calculate bonding electrons for central atom
        bonding_electrons = 0
        bond_multiplicity = {'single': 1, 'double': 2, 'triple': 3}

        for bond in bonds:
            if bond['from'] == central_idx or bond['to'] == central_idx:
                bonding_electrons += bond_multiplicity.get(bond.get('type', 'single'), 1)

        # Get bonding pairs (number of atoms bonded to central)
        bonded_indices = []
        for bond in bonds:
            if bond['from'] == central_idx:
                bonded_indices.append(bond['to'])
            elif bond['to'] == central_idx:
                bonded_indices.append(bond['from'])

        bonding_pairs = len(bonded_indices)
        lone_pairs = self.calculate_lone_pairs(central_element, bonding_electrons)

        # Determine geometry
        geometry_name, bond_angle = self.determine_geometry(bonding_pairs, lone_pairs)
        hybridization = self.determine_hybridization(bonding_pairs, lone_pairs)

        # Calculate bond lengths for each bonded atom
        bond_lengths = []
        bond_info = []

        for bonded_idx in bonded_indices:
            bonded_element = composition[bonded_idx]['element']

            # Find bond type
            bond_type = 'single'
            for bond in bonds:
                if (bond['from'] == central_idx and bond['to'] == bonded_idx) or \
                   (bond['to'] == central_idx and bond['from'] == bonded_idx):
                    bond_type = bond.get('type', 'single')
                    break

            length = self.get_bond_length(central_element, bonded_element, bond_type)
            bond_lengths.append(length)
            bond_info.append({
                'idx': bonded_idx,
                'element': bonded_element,
                'length': length,
                'type': bond_type
            })

        # Generate positions
        positions = self.generate_positions(geometry_name, bond_lengths, bond_angle)

        # Build atoms list
        atoms = []

        # Central atom at origin
        atoms.append({
            'element': central_element,
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'index': central_idx
        })

        # Bonded atoms at calculated positions
        for i, (bonded_idx, pos) in enumerate(zip(bonded_indices, positions)):
            atoms.append({
                'element': composition[bonded_idx]['element'],
                'x': round(pos[0], 4),
                'y': round(pos[1], 4),
                'z': round(pos[2], 4),
                'index': bonded_idx
            })

        # Add any atoms not yet placed (in larger molecules)
        placed_indices = {central_idx} | set(bonded_indices)
        for i, atom in enumerate(composition):
            if i not in placed_indices:
                # Place at a default position (will need refinement for complex molecules)
                atoms.append({
                    'element': atom['element'],
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'index': i
                })

        # Build bonds list with lengths
        result_bonds = []
        for bond in bonds:
            from_elem = composition[bond['from']]['element']
            to_elem = composition[bond['to']]['element']
            bond_type = bond.get('type', 'single')
            length = self.get_bond_length(from_elem, to_elem, bond_type)

            result_bonds.append({
                'from': bond['from'],
                'to': bond['to'],
                'type': bond_type,
                'length': round(length, 4)
            })

        return {
            'atoms': atoms,
            'bonds': result_bonds,
            'geometry': geometry_name,
            'bond_angle': bond_angle,
            'hybridization': hybridization,
            'lone_pairs': lone_pairs,
            'steric_number': bonding_pairs + lone_pairs
        }

    def calculate_distance(self, atom1: Dict, atom2: Dict) -> float:
        """Calculate distance between two atoms."""
        dx = atom2['x'] - atom1['x']
        dy = atom2['y'] - atom1['y']
        dz = atom2['z'] - atom1['z']
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def calculate_angle(self, atom1: Dict, central: Dict, atom2: Dict) -> float:
        """Calculate angle between three atoms (in degrees)."""
        # Vectors from central atom to other atoms
        v1 = (atom1['x'] - central['x'], atom1['y'] - central['y'], atom1['z'] - central['z'])
        v2 = (atom2['x'] - central['x'], atom2['y'] - central['y'], atom2['z'] - central['z'])

        # Dot product
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

        # Magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Clamp to avoid numerical errors
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))


# Pre-defined molecular structures for common molecules
COMMON_MOLECULES = {
    'H2O': {
        'composition': [{'element': 'O'}, {'element': 'H'}, {'element': 'H'}],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'}
        ]
    },
    'CO2': {
        'composition': [{'element': 'C'}, {'element': 'O'}, {'element': 'O'}],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'double'},
            {'from': 0, 'to': 2, 'type': 'double'}
        ]
    },
    'CH4': {
        'composition': [
            {'element': 'C'}, {'element': 'H'}, {'element': 'H'},
            {'element': 'H'}, {'element': 'H'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'},
            {'from': 0, 'to': 4, 'type': 'single'}
        ]
    },
    'NH3': {
        'composition': [
            {'element': 'N'}, {'element': 'H'}, {'element': 'H'}, {'element': 'H'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'}
        ]
    },
    'BH3': {
        'composition': [
            {'element': 'B'}, {'element': 'H'}, {'element': 'H'}, {'element': 'H'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'}
        ]
    },
    'SF6': {
        'composition': [
            {'element': 'S'}, {'element': 'F'}, {'element': 'F'},
            {'element': 'F'}, {'element': 'F'}, {'element': 'F'}, {'element': 'F'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'},
            {'from': 0, 'to': 4, 'type': 'single'},
            {'from': 0, 'to': 5, 'type': 'single'},
            {'from': 0, 'to': 6, 'type': 'single'}
        ]
    },
    'C2H4': {  # Ethylene
        'composition': [
            {'element': 'C'}, {'element': 'C'},
            {'element': 'H'}, {'element': 'H'}, {'element': 'H'}, {'element': 'H'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'double'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'},
            {'from': 1, 'to': 4, 'type': 'single'},
            {'from': 1, 'to': 5, 'type': 'single'}
        ]
    },
    'C2H2': {  # Acetylene
        'composition': [
            {'element': 'C'}, {'element': 'C'},
            {'element': 'H'}, {'element': 'H'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'triple'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 1, 'to': 3, 'type': 'single'}
        ]
    },
    'PCl5': {
        'composition': [
            {'element': 'P'}, {'element': 'Cl'}, {'element': 'Cl'},
            {'element': 'Cl'}, {'element': 'Cl'}, {'element': 'Cl'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'},
            {'from': 0, 'to': 4, 'type': 'single'},
            {'from': 0, 'to': 5, 'type': 'single'}
        ]
    },
    'H2S': {
        'composition': [{'element': 'S'}, {'element': 'H'}, {'element': 'H'}],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'}
        ]
    },
    'HCN': {
        'composition': [{'element': 'H'}, {'element': 'C'}, {'element': 'N'}],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 1, 'to': 2, 'type': 'triple'}
        ]
    },
    'HCHO': {  # Formaldehyde
        'composition': [
            {'element': 'C'}, {'element': 'H'}, {'element': 'H'}, {'element': 'O'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'double'}
        ]
    },
    'PH3': {
        'composition': [
            {'element': 'P'}, {'element': 'H'}, {'element': 'H'}, {'element': 'H'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'}
        ]
    },
    'CCl4': {
        'composition': [
            {'element': 'C'}, {'element': 'Cl'}, {'element': 'Cl'},
            {'element': 'Cl'}, {'element': 'Cl'}
        ],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'},
            {'from': 0, 'to': 4, 'type': 'single'}
        ]
    },
    'SO2': {
        'composition': [{'element': 'S'}, {'element': 'O'}, {'element': 'O'}],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'double'},
            {'from': 0, 'to': 2, 'type': 'double'}
        ]
    },
    'NO2': {
        'composition': [{'element': 'N'}, {'element': 'O'}, {'element': 'O'}],
        'bonds': [
            {'from': 0, 'to': 1, 'type': 'double'},
            {'from': 0, 'to': 2, 'type': 'single'}
        ]
    },
}


def get_molecule_structure(formula: str) -> Dict:
    """
    Get the 3D structure for a common molecule by formula.

    Args:
        formula: Molecular formula (e.g., 'H2O', 'CH4', 'CO2')

    Returns:
        Dictionary with calculated 3D structure
    """
    calculator = MolecularGeometryCalculator()

    # Check if it's a known molecule
    if formula in COMMON_MOLECULES:
        mol_data = COMMON_MOLECULES[formula]
        return calculator.calculate_structure(mol_data['composition'], mol_data['bonds'])

    # Try to parse and build simple molecules
    structure = _parse_simple_formula(formula)
    if structure:
        return calculator.calculate_structure(structure['composition'], structure['bonds'])

    return {
        'atoms': [],
        'bonds': [],
        'geometry': 'Unknown',
        'bond_angle': 0.0,
        'error': f'Unknown molecule: {formula}'
    }


def _parse_simple_formula(formula: str) -> Optional[Dict]:
    """
    Parse a simple molecular formula and attempt to create structure.
    Only handles basic cases like HCl, NaCl, etc.
    """
    import re

    # Simple two-element compounds
    pattern = r'([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)'
    match = re.match(pattern, formula)

    if match:
        elem1, count1, elem2, count2 = match.groups()
        count1 = int(count1) if count1 else 1
        count2 = int(count2) if count2 else 1

        # Build composition
        composition = []
        for _ in range(count1):
            composition.append({'element': elem1})
        for _ in range(count2):
            composition.append({'element': elem2})

        # Simple bonding assumption: first element bonds to second
        bonds = []
        for i in range(count2):
            bonds.append({'from': 0, 'to': count1 + i, 'type': 'single'})

        return {'composition': composition, 'bonds': bonds}

    return None


def get_geometry_info(bonding_pairs: int, lone_pairs: int) -> Dict:
    """
    Get geometry information for given electron arrangement.

    Args:
        bonding_pairs: Number of bonding pairs
        lone_pairs: Number of lone pairs

    Returns:
        Dictionary with geometry details
    """
    calculator = MolecularGeometryCalculator()
    geometry, angle = calculator.determine_geometry(bonding_pairs, lone_pairs)
    hybridization = calculator.determine_hybridization(bonding_pairs, lone_pairs)

    return {
        'geometry': geometry,
        'bond_angle': angle,
        'hybridization': hybridization,
        'steric_number': bonding_pairs + lone_pairs,
        'bonding_pairs': bonding_pairs,
        'lone_pairs': lone_pairs
    }


def calculate_molecular_properties(structure: Dict) -> Dict:
    """
    Calculate additional molecular properties from a structure.

    Args:
        structure: Structure dictionary from calculate_structure()

    Returns:
        Dictionary with molecular properties
    """
    if not structure.get('atoms'):
        return {}

    calculator = MolecularGeometryCalculator()

    # Calculate center of mass (assuming all atoms have equal weight for simplicity)
    total_mass = len(structure['atoms'])
    com_x = sum(a['x'] for a in structure['atoms']) / total_mass
    com_y = sum(a['y'] for a in structure['atoms']) / total_mass
    com_z = sum(a['z'] for a in structure['atoms']) / total_mass

    # Calculate molecular extent (bounding box)
    if structure['atoms']:
        x_coords = [a['x'] for a in structure['atoms']]
        y_coords = [a['y'] for a in structure['atoms']]
        z_coords = [a['z'] for a in structure['atoms']]

        extent = {
            'x': max(x_coords) - min(x_coords),
            'y': max(y_coords) - min(y_coords),
            'z': max(z_coords) - min(z_coords)
        }
    else:
        extent = {'x': 0, 'y': 0, 'z': 0}

    # Calculate actual bond angles
    bond_angles = []
    atoms = structure['atoms']
    bonds = structure['bonds']

    # Find central atom
    central = next((a for a in atoms if a['x'] == 0 and a['y'] == 0 and a['z'] == 0), None)

    if central:
        # Get atoms bonded to central
        bonded_atoms = [a for a in atoms if a['index'] != central['index']]

        # Calculate angles between pairs of bonded atoms
        for i in range(len(bonded_atoms)):
            for j in range(i + 1, len(bonded_atoms)):
                angle = calculator.calculate_angle(bonded_atoms[i], central, bonded_atoms[j])
                bond_angles.append({
                    'atoms': [bonded_atoms[i]['index'], central['index'], bonded_atoms[j]['index']],
                    'angle': round(angle, 2)
                })

    return {
        'center_of_mass': {'x': round(com_x, 4), 'y': round(com_y, 4), 'z': round(com_z, 4)},
        'molecular_extent': extent,
        'bond_angles': bond_angles,
        'atom_count': len(structure['atoms']),
        'bond_count': len(structure['bonds'])
    }


# Example usage and testing
if __name__ == '__main__':
    # Test with water molecule
    print("=== Water (H2O) ===")
    water = get_molecule_structure('H2O')
    print(f"Geometry: {water['geometry']}")
    print(f"Bond angle: {water['bond_angle']}°")
    print(f"Hybridization: {water['hybridization']}")
    print(f"Lone pairs: {water['lone_pairs']}")
    print("Atoms:")
    for atom in water['atoms']:
        print(f"  {atom['element']}: ({atom['x']}, {atom['y']}, {atom['z']})")

    print("\n=== Methane (CH4) ===")
    methane = get_molecule_structure('CH4')
    print(f"Geometry: {methane['geometry']}")
    print(f"Bond angle: {methane['bond_angle']}°")
    print(f"Hybridization: {methane['hybridization']}")

    print("\n=== Carbon Dioxide (CO2) ===")
    co2 = get_molecule_structure('CO2')
    print(f"Geometry: {co2['geometry']}")
    print(f"Bond angle: {co2['bond_angle']}°")

    print("\n=== Ammonia (NH3) ===")
    ammonia = get_molecule_structure('NH3')
    print(f"Geometry: {ammonia['geometry']}")
    print(f"Bond angle: {ammonia['bond_angle']}°")
    print(f"Lone pairs: {ammonia['lone_pairs']}")

    print("\n=== Sulfur Hexafluoride (SF6) ===")
    sf6 = get_molecule_structure('SF6')
    print(f"Geometry: {sf6['geometry']}")
    print(f"Bond angle: {sf6['bond_angle']}°")

    print("\n=== Borane (BH3) ===")
    bh3 = get_molecule_structure('BH3')
    print(f"Geometry: {bh3['geometry']}")
    print(f"Bond angle: {bh3['bond_angle']}°")

    # Test properties calculation
    print("\n=== Molecular Properties for CH4 ===")
    props = calculate_molecular_properties(methane)
    print(f"Center of mass: {props['center_of_mass']}")
    print(f"Bond angles: {props['bond_angles']}")
