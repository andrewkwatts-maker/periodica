"""
Enums for amino acid visualization properties and encodings.
Centralizes all amino acid-related property checking to prevent typos and improve maintainability.
"""

from enum import Enum


class AminoAcidLayoutMode(Enum):
    """Layout modes for amino acid visualization"""
    GRID = "grid"
    HYDROPATHY = "hydropathy"
    CHARGE = "charge"
    POLARITY = "polarity"
    CATEGORY = "category"
    MASS = "mass"
    PI_ORDER = "pi_order"
    STRUCTURE = "structure"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID

    @classmethod
    def get_display_name(cls, mode):
        """Get UI display name for a layout mode"""
        if isinstance(mode, str):
            mode = cls.from_string(mode)

        display_names = {
            cls.GRID: "Grid View",
            cls.HYDROPATHY: "By Hydropathy",
            cls.CHARGE: "By Charge",
            cls.POLARITY: "By Polarity",
            cls.CATEGORY: "By Category",
            cls.MASS: "By Mass",
            cls.PI_ORDER: "By Isoelectric Point",
            cls.STRUCTURE: "By Structure"
        }
        return display_names.get(mode, "Unknown")


class AminoAcidProperty(Enum):
    """Amino acid properties that can be visualized or used for ordering"""
    # Basic properties
    NAME = "name"
    SYMBOL = "symbol"
    THREE_LETTER_CODE = "three_letter_code"
    MOLECULAR_FORMULA = "molecular_formula"
    MOLECULAR_MASS = "molecular_mass"

    # Chemical properties
    PKA_CARBOXYL = "pKa_carboxyl"
    PKA_AMINO = "pKa_amino"
    PKA_SIDECHAIN = "pKa_sidechain"
    ISOELECTRIC_POINT = "isoelectric_point"
    CHARGE_PH7 = "charge_pH7"

    # Physical properties
    HYDROPATHY_INDEX = "hydropathy_index"
    POLARITY = "polarity"
    CATEGORY = "category"

    # Structure propensities
    HELIX_PROPENSITY = "helix_propensity"
    SHEET_PROPENSITY = "sheet_propensity"
    TURN_PROPENSITY = "turn_propensity"

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
            cls.SYMBOL: "Symbol",
            cls.THREE_LETTER_CODE: "3-Letter Code",
            cls.MOLECULAR_FORMULA: "Molecular Formula",
            cls.MOLECULAR_MASS: "Molecular Mass",
            cls.PKA_CARBOXYL: "pKa Carboxyl",
            cls.PKA_AMINO: "pKa Amino",
            cls.PKA_SIDECHAIN: "pKa Side Chain",
            cls.ISOELECTRIC_POINT: "Isoelectric Point",
            cls.CHARGE_PH7: "Charge at pH 7",
            cls.HYDROPATHY_INDEX: "Hydropathy Index",
            cls.POLARITY: "Polarity",
            cls.CATEGORY: "Category",
            cls.HELIX_PROPENSITY: "Helix Propensity",
            cls.SHEET_PROPENSITY: "Sheet Propensity",
            cls.TURN_PROPENSITY: "Turn Propensity",
            cls.NONE: "None"
        }
        return display_names.get(prop, "Unknown")

    @classmethod
    def get_numeric_properties(cls):
        """Get ordered list of all numeric properties for visual encoding"""
        return [
            cls.MOLECULAR_MASS,
            cls.PKA_CARBOXYL,
            cls.PKA_AMINO,
            cls.PKA_SIDECHAIN,
            cls.ISOELECTRIC_POINT,
            cls.CHARGE_PH7,
            cls.HYDROPATHY_INDEX,
            cls.HELIX_PROPENSITY,
            cls.SHEET_PROPENSITY,
            cls.TURN_PROPENSITY,
        ]

    @classmethod
    def get_color_properties(cls):
        """Get ordered list of properties suitable for color encoding"""
        return [
            cls.HYDROPATHY_INDEX,
            cls.CHARGE_PH7,
            cls.MOLECULAR_MASS,
            cls.ISOELECTRIC_POINT,
            cls.HELIX_PROPENSITY,
            cls.SHEET_PROPENSITY,
            cls.TURN_PROPENSITY,
            cls.PKA_SIDECHAIN,
            cls.NONE
        ]

    @classmethod
    def get_size_properties(cls):
        """Get ordered list of properties suitable for size encoding"""
        return [
            cls.MOLECULAR_MASS,
            cls.HYDROPATHY_INDEX,
            cls.ISOELECTRIC_POINT,
            cls.HELIX_PROPENSITY,
            cls.SHEET_PROPENSITY,
            cls.NONE
        ]


class AminoAcidCategory(Enum):
    """Amino acid category types based on side chain properties"""
    NONPOLAR_ALIPHATIC = "nonpolar_aliphatic"
    NONPOLAR_AROMATIC = "nonpolar_aromatic"
    POLAR_UNCHARGED = "polar_uncharged"
    POLAR_POSITIVE = "polar_positive"
    POLAR_NEGATIVE = "polar_negative"
    SPECIAL = "special"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        # Handle both snake_case and display names
        value_lower = value.lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        # Try matching display names
        display_map = {
            "nonpolar": cls.NONPOLAR_ALIPHATIC,
            "nonpolar_aliphatic": cls.NONPOLAR_ALIPHATIC,
            "nonpolar_aromatic": cls.NONPOLAR_AROMATIC,
            "aromatic": cls.NONPOLAR_AROMATIC,
            "polar": cls.POLAR_UNCHARGED,
            "polar_uncharged": cls.POLAR_UNCHARGED,
            "uncharged": cls.POLAR_UNCHARGED,
            "positive": cls.POLAR_POSITIVE,
            "polar_positive": cls.POLAR_POSITIVE,
            "basic": cls.POLAR_POSITIVE,
            "negative": cls.POLAR_NEGATIVE,
            "polar_negative": cls.POLAR_NEGATIVE,
            "acidic": cls.POLAR_NEGATIVE,
            "special": cls.SPECIAL,
        }
        return display_map.get(value_lower, cls.NONPOLAR_ALIPHATIC)

    @classmethod
    def get_display_name(cls, category):
        """Get UI display name for a category"""
        if isinstance(category, str):
            category = cls.from_string(category)

        display_names = {
            cls.NONPOLAR_ALIPHATIC: "Nonpolar Aliphatic",
            cls.NONPOLAR_AROMATIC: "Nonpolar Aromatic",
            cls.POLAR_UNCHARGED: "Polar Uncharged",
            cls.POLAR_POSITIVE: "Polar Positive (Basic)",
            cls.POLAR_NEGATIVE: "Polar Negative (Acidic)",
            cls.SPECIAL: "Special"
        }
        return display_names.get(category, "Unknown")

    @classmethod
    def get_color(cls, category):
        """Get color for a category"""
        if isinstance(category, str):
            category = cls.from_string(category)

        colors = {
            cls.NONPOLAR_ALIPHATIC: "#FFA726",  # Orange
            cls.NONPOLAR_AROMATIC: "#AB47BC",   # Purple
            cls.POLAR_UNCHARGED: "#66BB6A",     # Green
            cls.POLAR_POSITIVE: "#42A5F5",      # Blue
            cls.POLAR_NEGATIVE: "#EF5350",      # Red
            cls.SPECIAL: "#FFEE58"              # Yellow
        }
        return colors.get(category, "#9E9E9E")


class AminoAcidPolarity(Enum):
    """Amino acid polarity types"""
    NONPOLAR = "nonpolar"
    POLAR = "polar"
    ACIDIC = "acidic"
    BASIC = "basic"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.NONPOLAR

    @classmethod
    def get_display_name(cls, polarity):
        """Get UI display name for polarity"""
        if isinstance(polarity, str):
            polarity = cls.from_string(polarity)

        display_names = {
            cls.NONPOLAR: "Nonpolar",
            cls.POLAR: "Polar",
            cls.ACIDIC: "Acidic",
            cls.BASIC: "Basic"
        }
        return display_names.get(polarity, "Unknown")

    @classmethod
    def get_color(cls, polarity):
        """Get color for polarity"""
        if isinstance(polarity, str):
            polarity = cls.from_string(polarity)

        colors = {
            cls.NONPOLAR: "#FFA726",   # Orange
            cls.POLAR: "#66BB6A",      # Green
            cls.ACIDIC: "#EF5350",     # Red
            cls.BASIC: "#42A5F5"       # Blue
        }
        return colors.get(polarity, "#9E9E9E")


class ChargeState(Enum):
    """Amino acid charge state at physiological pH"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ZWITTERION = "zwitterion"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.NEUTRAL

    @classmethod
    def from_charge(cls, charge):
        """Determine charge state from numeric charge value"""
        if charge > 0.5:
            return cls.POSITIVE
        elif charge < -0.5:
            return cls.NEGATIVE
        else:
            return cls.NEUTRAL

    @classmethod
    def get_color(cls, state):
        """Get color for charge state"""
        if isinstance(state, str):
            state = cls.from_string(state)

        colors = {
            cls.POSITIVE: "#42A5F5",   # Blue
            cls.NEGATIVE: "#EF5350",   # Red
            cls.NEUTRAL: "#9E9E9E",    # Grey
            cls.ZWITTERION: "#AB47BC"  # Purple
        }
        return colors.get(state, "#9E9E9E")


class SecondaryStructure(Enum):
    """Protein secondary structure types"""
    HELIX = "helix"
    SHEET = "sheet"
    COIL = "coil"
    TURN = "turn"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.COIL

    @classmethod
    def get_color(cls, structure):
        """Get color for secondary structure"""
        if isinstance(structure, str):
            structure = cls.from_string(structure)

        colors = {
            cls.HELIX: "#FF4081",    # Pink
            cls.SHEET: "#448AFF",    # Blue
            cls.COIL: "#9E9E9E",     # Grey
            cls.TURN: "#69F0AE"      # Green
        }
        return colors.get(structure, "#9E9E9E")


# Standard amino acid data for reference
STANDARD_AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'three_letter': 'Ala', 'category': 'nonpolar_aliphatic'},
    'R': {'name': 'Arginine', 'three_letter': 'Arg', 'category': 'polar_positive'},
    'N': {'name': 'Asparagine', 'three_letter': 'Asn', 'category': 'polar_uncharged'},
    'D': {'name': 'Aspartic acid', 'three_letter': 'Asp', 'category': 'polar_negative'},
    'C': {'name': 'Cysteine', 'three_letter': 'Cys', 'category': 'special'},
    'E': {'name': 'Glutamic acid', 'three_letter': 'Glu', 'category': 'polar_negative'},
    'Q': {'name': 'Glutamine', 'three_letter': 'Gln', 'category': 'polar_uncharged'},
    'G': {'name': 'Glycine', 'three_letter': 'Gly', 'category': 'special'},
    'H': {'name': 'Histidine', 'three_letter': 'His', 'category': 'polar_positive'},
    'I': {'name': 'Isoleucine', 'three_letter': 'Ile', 'category': 'nonpolar_aliphatic'},
    'L': {'name': 'Leucine', 'three_letter': 'Leu', 'category': 'nonpolar_aliphatic'},
    'K': {'name': 'Lysine', 'three_letter': 'Lys', 'category': 'polar_positive'},
    'M': {'name': 'Methionine', 'three_letter': 'Met', 'category': 'nonpolar_aliphatic'},
    'F': {'name': 'Phenylalanine', 'three_letter': 'Phe', 'category': 'nonpolar_aromatic'},
    'P': {'name': 'Proline', 'three_letter': 'Pro', 'category': 'special'},
    'S': {'name': 'Serine', 'three_letter': 'Ser', 'category': 'polar_uncharged'},
    'T': {'name': 'Threonine', 'three_letter': 'Thr', 'category': 'polar_uncharged'},
    'W': {'name': 'Tryptophan', 'three_letter': 'Trp', 'category': 'nonpolar_aromatic'},
    'Y': {'name': 'Tyrosine', 'three_letter': 'Tyr', 'category': 'nonpolar_aromatic'},
    'V': {'name': 'Valine', 'three_letter': 'Val', 'category': 'nonpolar_aliphatic'},
}


# Property metadata for slider ranges and units
AMINO_ACID_PROPERTY_METADATA = {
    "molecular_mass": {
        "display_name": "Molecular Mass",
        "unit": "Da",
        "min_value": 75.0,
        "max_value": 205.0,
        "description": "Molecular mass of the amino acid in Daltons"
    },
    "pKa_carboxyl": {
        "display_name": "pKa Carboxyl",
        "unit": "",
        "min_value": 1.8,
        "max_value": 2.6,
        "description": "pKa of the carboxyl group"
    },
    "pKa_amino": {
        "display_name": "pKa Amino",
        "unit": "",
        "min_value": 8.8,
        "max_value": 10.8,
        "description": "pKa of the amino group"
    },
    "pKa_sidechain": {
        "display_name": "pKa Side Chain",
        "unit": "",
        "min_value": 1.0,
        "max_value": 14.0,
        "description": "pKa of the side chain (if ionizable)"
    },
    "isoelectric_point": {
        "display_name": "Isoelectric Point",
        "unit": "",
        "min_value": 2.5,
        "max_value": 11.0,
        "description": "pH at which the amino acid has no net charge"
    },
    "charge_pH7": {
        "display_name": "Charge at pH 7",
        "unit": "",
        "min_value": -1.0,
        "max_value": 1.0,
        "description": "Net charge at physiological pH"
    },
    "hydropathy_index": {
        "display_name": "Hydropathy Index",
        "unit": "",
        "min_value": -4.5,
        "max_value": 4.5,
        "description": "Kyte-Doolittle hydropathy index"
    },
    "helix_propensity": {
        "display_name": "Helix Propensity",
        "unit": "",
        "min_value": 0.5,
        "max_value": 1.6,
        "description": "Chou-Fasman helix propensity"
    },
    "sheet_propensity": {
        "display_name": "Sheet Propensity",
        "unit": "",
        "min_value": 0.3,
        "max_value": 1.8,
        "description": "Chou-Fasman sheet propensity"
    },
    "turn_propensity": {
        "display_name": "Turn Propensity",
        "unit": "",
        "min_value": 0.5,
        "max_value": 1.6,
        "description": "Turn propensity value"
    },
    "none": {
        "display_name": "None",
        "unit": "",
        "min_value": 0.0,
        "max_value": 100.0,
        "description": "No property encoding"
    }
}


def get_amino_acid_property_metadata(property_name):
    """Get metadata for an amino acid property"""
    return AMINO_ACID_PROPERTY_METADATA.get(
        property_name,
        AMINO_ACID_PROPERTY_METADATA["none"]
    )
