"""
Enums for molecule visualization properties and encodings.
Centralizes all molecule-related property checking to prevent typos and improve maintainability.
"""

from enum import Enum, auto


class MoleculeLayoutMode(Enum):
    """Layout modes for molecule visualization"""
    GRID = "grid"
    MASS_ORDER = "mass_order"
    POLARITY = "polarity"
    BOND_TYPE = "bond_type"
    GEOMETRY = "geometry"
    PHASE_DIAGRAM = "phase_diagram"
    DIPOLE = "dipole"
    DENSITY = "density"
    BOND_COMPLEXITY = "bond_complexity"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID  # Default to grid

    @classmethod
    def get_display_name(cls, mode):
        """Get UI display name for a layout mode"""
        if isinstance(mode, str):
            mode = cls.from_string(mode)

        display_names = {
            cls.GRID: "Grid View",
            cls.MASS_ORDER: "Mass Order",
            cls.POLARITY: "By Polarity",
            cls.BOND_TYPE: "By Bond Type",
            cls.GEOMETRY: "By Geometry",
            cls.PHASE_DIAGRAM: "Phase Diagram",
            cls.DIPOLE: "Dipole-Polarity",
            cls.DENSITY: "Density-Mass",
            cls.BOND_COMPLEXITY: "Bond Complexity"
        }
        return display_names.get(mode, "Unknown")


class MoleculeProperty(Enum):
    """Molecule properties that can be visualized or used for ordering"""
    # Basic properties
    NAME = "name"
    FORMULA = "formula"
    MOLECULAR_MASS = "molecular_mass"

    # Physical properties
    MELTING_POINT = "melting_point"
    BOILING_POINT = "boiling_point"
    DENSITY = "density"
    DIPOLE_MOMENT = "dipole_moment"
    VAPOR_PRESSURE = "vapor_pressure"
    SOLUBILITY = "solubility"

    # Chemical properties
    BOND_TYPE = "bond_type"
    GEOMETRY = "geometry"
    POLARITY = "polarity"
    BOND_ANGLE = "bond_angle"
    ELECTRONEGATIVITY_DIFF = "electronegativity_diff"

    # Structural properties (derived)
    NUM_ATOMS = "num_atoms"
    NUM_BONDS = "num_bonds"
    BOND_LENGTH_AVG = "bond_length_avg"

    # State
    STATE_STP = "state_stp"

    # Category
    CATEGORY = "category"

    # Special value
    NONE = "none"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum, return NONE if not found"""
        if value is None:
            return cls.NONE
        for member in cls:
            if member.value == value:
                return member
        return cls.NONE

    @classmethod
    def get_display_name(cls, prop):
        """Get UI display name for a property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)

        display_names = {
            cls.NAME: "Name",
            cls.FORMULA: "Formula",
            cls.MOLECULAR_MASS: "Molecular Mass",
            cls.MELTING_POINT: "Melting Point",
            cls.BOILING_POINT: "Boiling Point",
            cls.DENSITY: "Density",
            cls.DIPOLE_MOMENT: "Dipole Moment",
            cls.VAPOR_PRESSURE: "Vapor Pressure",
            cls.SOLUBILITY: "Solubility",
            cls.BOND_TYPE: "Bond Type",
            cls.GEOMETRY: "Geometry",
            cls.POLARITY: "Polarity",
            cls.BOND_ANGLE: "Bond Angle",
            cls.ELECTRONEGATIVITY_DIFF: "Electronegativity Diff",
            cls.NUM_ATOMS: "Number of Atoms",
            cls.NUM_BONDS: "Number of Bonds",
            cls.BOND_LENGTH_AVG: "Avg Bond Length",
            cls.STATE_STP: "State at STP",
            cls.CATEGORY: "Category",
            cls.NONE: "None"
        }
        return display_names.get(prop, "Unknown")

    @classmethod
    def get_color_properties(cls):
        """Get ordered list of properties suitable for color encoding"""
        return [
            cls.MOLECULAR_MASS,
            cls.MELTING_POINT,
            cls.BOILING_POINT,
            cls.DENSITY,
            cls.DIPOLE_MOMENT,
            cls.VAPOR_PRESSURE,
            cls.SOLUBILITY,
            cls.BOND_ANGLE,
            cls.ELECTRONEGATIVITY_DIFF,
            cls.NUM_ATOMS,
            cls.NUM_BONDS,
            cls.BOND_LENGTH_AVG,
            cls.BOND_TYPE,
            cls.GEOMETRY,
            cls.POLARITY,
            cls.CATEGORY,
            cls.NONE
        ]

    @classmethod
    def get_size_properties(cls):
        """Get ordered list of properties suitable for size encoding"""
        return [
            cls.MOLECULAR_MASS,
            cls.DENSITY,
            cls.DIPOLE_MOMENT,
            cls.BOND_ANGLE,
            cls.NUM_ATOMS,
            cls.NUM_BONDS,
            cls.BOND_LENGTH_AVG,
            cls.VAPOR_PRESSURE,
            cls.SOLUBILITY,
            cls.ELECTRONEGATIVITY_DIFF,
            cls.NONE
        ]

    @classmethod
    def get_numeric_properties(cls):
        """Get ordered list of all numeric properties for visual encoding"""
        return [
            cls.MOLECULAR_MASS,
            cls.DENSITY,
            cls.MELTING_POINT,
            cls.BOILING_POINT,
            cls.BOND_ANGLE,
            cls.DIPOLE_MOMENT,
            cls.VAPOR_PRESSURE,
            cls.SOLUBILITY,
            cls.NUM_ATOMS,
            cls.NUM_BONDS,
            cls.ELECTRONEGATIVITY_DIFF,
            cls.BOND_LENGTH_AVG,
        ]

    @classmethod
    def get_glow_properties(cls):
        """Get ordered list of properties suitable for glow/highlight encoding"""
        return [
            cls.DIPOLE_MOMENT,
            cls.ELECTRONEGATIVITY_DIFF,
            cls.MOLECULAR_MASS,
            cls.DENSITY,
            cls.MELTING_POINT,
            cls.BOILING_POINT,
            cls.VAPOR_PRESSURE,
            cls.NONE
        ]

    @classmethod
    def get_border_properties(cls):
        """Get ordered list of properties suitable for border intensity encoding"""
        return [
            cls.BOILING_POINT,
            cls.MELTING_POINT,
            cls.DENSITY,
            cls.DIPOLE_MOMENT,
            cls.MOLECULAR_MASS,
            cls.NUM_BONDS,
            cls.ELECTRONEGATIVITY_DIFF,
            cls.NONE
        ]


class BondType(Enum):
    """Types of chemical bonds"""
    SINGLE = "Single"
    DOUBLE = "Double"
    TRIPLE = "Triple"
    AROMATIC = "Aromatic"
    IONIC = "Ionic"
    RESONANCE = "Resonance"
    HYDROGEN = "Hydrogen"
    METALLIC = "Metallic"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.SINGLE  # Default to single

    @classmethod
    def get_color(cls, bond_type):
        """Get color for a bond type"""
        if isinstance(bond_type, str):
            bond_type = cls.from_string(bond_type)

        colors = {
            cls.SINGLE: "#4CAF50",      # Green
            cls.DOUBLE: "#2196F3",      # Blue
            cls.TRIPLE: "#9C27B0",      # Purple
            cls.AROMATIC: "#E91E63",    # Pink
            cls.IONIC: "#FF9800",       # Orange
            cls.RESONANCE: "#00BCD4",   # Cyan
            cls.HYDROGEN: "#FFEB3B",    # Yellow
            cls.METALLIC: "#607D8B"     # Blue Grey
        }
        return colors.get(bond_type, "#9E9E9E")


class MolecularGeometry(Enum):
    """Molecular geometries (VSEPR theory)"""
    LINEAR = "Linear"
    BENT = "Bent"
    TRIGONAL_PLANAR = "Trigonal Planar"
    TRIGONAL_PYRAMIDAL = "Trigonal Pyramidal"
    TETRAHEDRAL = "Tetrahedral"
    TRIGONAL_BIPYRAMIDAL = "Trigonal Bipyramidal"
    OCTAHEDRAL = "Octahedral"
    SQUARE_PLANAR = "Square Planar"
    SQUARE_PYRAMIDAL = "Square Pyramidal"
    SEESAW = "Seesaw"
    T_SHAPED = "T-Shaped"
    PLANAR_HEXAGONAL = "Planar Hexagonal"
    CHAIR_CONFORMATION = "Chair Conformation"
    FACE_CENTERED_CUBIC = "Face-centered Cubic"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.LINEAR  # Default to linear

    @classmethod
    def get_color(cls, geometry):
        """Get color for a geometry type"""
        if isinstance(geometry, str):
            geometry = cls.from_string(geometry)

        colors = {
            cls.LINEAR: "#4FC3F7",           # Light Blue
            cls.BENT: "#81C784",              # Light Green
            cls.TRIGONAL_PLANAR: "#FFB74D",  # Orange
            cls.TRIGONAL_PYRAMIDAL: "#BA68C8", # Purple
            cls.TETRAHEDRAL: "#F06292",       # Pink
            cls.TRIGONAL_BIPYRAMIDAL: "#64B5F6", # Blue
            cls.OCTAHEDRAL: "#4DB6AC",        # Teal
            cls.SQUARE_PLANAR: "#A1887F",     # Brown
            cls.SQUARE_PYRAMIDAL: "#90A4AE",  # Blue Grey
            cls.SEESAW: "#DCE775",            # Lime
            cls.T_SHAPED: "#4DD0E1",          # Cyan
            cls.PLANAR_HEXAGONAL: "#CE93D8",  # Light Purple
            cls.CHAIR_CONFORMATION: "#FFCC80", # Light Orange
            cls.FACE_CENTERED_CUBIC: "#ECEFF1" # Grey
        }
        return colors.get(geometry, "#9E9E9E")


class MoleculePolarity(Enum):
    """Molecule polarity types"""
    POLAR = "Polar"
    NONPOLAR = "Nonpolar"
    IONIC = "Ionic"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.NONPOLAR  # Default to nonpolar

    @classmethod
    def get_color(cls, polarity):
        """Get color for polarity type"""
        if isinstance(polarity, str):
            polarity = cls.from_string(polarity)

        colors = {
            cls.POLAR: "#2196F3",      # Blue
            cls.NONPOLAR: "#4CAF50",   # Green
            cls.IONIC: "#FF9800"       # Orange
        }
        return colors.get(polarity, "#9E9E9E")


class MoleculeCategory(Enum):
    """Molecule category types"""
    ORGANIC = "Organic"
    INORGANIC = "Inorganic"
    IONIC = "Ionic"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.INORGANIC  # Default to inorganic

    @classmethod
    def get_color(cls, category):
        """Get color for category type"""
        if isinstance(category, str):
            category = cls.from_string(category)

        colors = {
            cls.ORGANIC: "#8BC34A",     # Light Green
            cls.INORGANIC: "#03A9F4",   # Light Blue
            cls.IONIC: "#FFC107"        # Amber
        }
        return colors.get(category, "#9E9E9E")


class MoleculeState(Enum):
    """Physical state at STP"""
    SOLID = "Solid"
    LIQUID = "Liquid"
    GAS = "Gas"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.GAS  # Default to gas

    @classmethod
    def get_color(cls, state):
        """Get color for state"""
        if isinstance(state, str):
            state = cls.from_string(state)

        colors = {
            cls.SOLID: "#795548",    # Brown
            cls.LIQUID: "#2196F3",   # Blue
            cls.GAS: "#E0E0E0"       # Light Grey
        }
        return colors.get(state, "#9E9E9E")


# Element colors for atom visualization
ELEMENT_COLORS = {
    "H": "#FFFFFF",   # White
    "C": "#909090",   # Grey
    "N": "#3050F8",   # Blue
    "O": "#FF0D0D",   # Red
    "S": "#FFFF30",   # Yellow
    "P": "#FF8000",   # Orange
    "Cl": "#1FF01F",  # Green
    "Br": "#A62929",  # Dark Red
    "F": "#90E050",   # Light Green
    "I": "#940094",   # Purple
    "Na": "#AB5CF2",  # Violet
    "K": "#8F40D4",   # Purple
    "Ca": "#3DFF00",  # Green
    "Fe": "#E06633",  # Orange
    "Mg": "#8AFF00",  # Green Yellow
    "default": "#FF1493"  # Deep Pink for unknown elements
}


def get_element_color(symbol):
    """Get color for an element symbol"""
    return ELEMENT_COLORS.get(symbol, ELEMENT_COLORS["default"])
