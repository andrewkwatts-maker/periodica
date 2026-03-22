"""
Enums for protein visualization and structural properties.
Includes secondary structure, folding states, and domain types.
"""

from enum import Enum


class ProteinLayoutMode(Enum):
    """Layout modes for protein visualization"""
    GRID = "grid"
    MASS = "mass"
    FUNCTION = "function"
    STRUCTURE = "structure"
    LOCALIZATION = "localization"
    ORGANISM = "organism"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID


class SecondaryStructureType(Enum):
    """Protein secondary structure types"""
    ALPHA_HELIX = "alpha_helix"
    BETA_SHEET = "beta_sheet"
    BETA_TURN = "beta_turn"
    RANDOM_COIL = "random_coil"
    THREE_TEN_HELIX = "310_helix"
    PI_HELIX = "pi_helix"
    POLYPROLINE_HELIX = "polyproline"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.RANDOM_COIL

    @classmethod
    def get_color(cls, structure):
        if isinstance(structure, str):
            structure = cls.from_string(structure)
        colors = {
            cls.ALPHA_HELIX: "#FF4081",      # Pink
            cls.BETA_SHEET: "#448AFF",       # Blue
            cls.BETA_TURN: "#69F0AE",        # Green
            cls.RANDOM_COIL: "#9E9E9E",      # Grey
            cls.THREE_TEN_HELIX: "#FF6E40",  # Orange
            cls.PI_HELIX: "#E040FB",         # Purple
            cls.POLYPROLINE_HELIX: "#FFEB3B" # Yellow
        }
        return colors.get(structure, "#9E9E9E")

    @classmethod
    def get_phi_psi_ranges(cls, structure):
        """Get typical phi/psi angle ranges for each structure type (Ramachandran)"""
        if isinstance(structure, str):
            structure = cls.from_string(structure)
        # (phi_min, phi_max, psi_min, psi_max) in degrees
        ranges = {
            cls.ALPHA_HELIX: (-80, -48, -59, -27),
            cls.BETA_SHEET: (-150, -90, 90, 150),
            cls.BETA_TURN: (-90, 0, -30, 60),
            cls.RANDOM_COIL: (-180, 180, -180, 180),
            cls.THREE_TEN_HELIX: (-74, -50, -18, -4),
            cls.PI_HELIX: (-76, -52, -53, -39),
            cls.POLYPROLINE_HELIX: (-78, -72, 145, 155),
        }
        return ranges.get(structure, (-180, 180, -180, 180))


class ProteinFunction(Enum):
    """Protein functional categories"""
    ENZYME = "enzyme"
    STRUCTURAL = "structural"
    TRANSPORT = "transport"
    SIGNALING = "signaling"
    RECEPTOR = "receptor"
    ANTIBODY = "antibody"
    STORAGE = "storage"
    REGULATORY = "regulatory"
    MOTOR = "motor"
    CHANNEL = "channel"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.STRUCTURAL

    @classmethod
    def get_color(cls, func):
        if isinstance(func, str):
            func = cls.from_string(func)
        colors = {
            cls.ENZYME: "#4CAF50",
            cls.STRUCTURAL: "#795548",
            cls.TRANSPORT: "#2196F3",
            cls.SIGNALING: "#FF9800",
            cls.RECEPTOR: "#E91E63",
            cls.ANTIBODY: "#9C27B0",
            cls.STORAGE: "#607D8B",
            cls.REGULATORY: "#00BCD4",
            cls.MOTOR: "#F44336",
            cls.CHANNEL: "#3F51B5",
        }
        return colors.get(func, "#9E9E9E")


class CellularLocalization(Enum):
    """Protein cellular localization"""
    CYTOPLASM = "cytoplasm"
    NUCLEUS = "nucleus"
    MITOCHONDRIA = "mitochondria"
    ENDOPLASMIC_RETICULUM = "endoplasmic_reticulum"
    GOLGI = "golgi"
    MEMBRANE = "membrane"
    EXTRACELLULAR = "extracellular"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    CYTOSKELETON = "cytoskeleton"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.CYTOPLASM

    @classmethod
    def get_color(cls, loc):
        if isinstance(loc, str):
            loc = cls.from_string(loc)
        colors = {
            cls.CYTOPLASM: "#81C784",
            cls.NUCLEUS: "#64B5F6",
            cls.MITOCHONDRIA: "#FFB74D",
            cls.ENDOPLASMIC_RETICULUM: "#BA68C8",
            cls.GOLGI: "#4DD0E1",
            cls.MEMBRANE: "#A1887F",
            cls.EXTRACELLULAR: "#90A4AE",
            cls.LYSOSOME: "#F06292",
            cls.PEROXISOME: "#AED581",
            cls.CYTOSKELETON: "#FF8A65",
        }
        return colors.get(loc, "#9E9E9E")


class FoldingState(Enum):
    """Protein folding states"""
    NATIVE = "native"
    PARTIALLY_FOLDED = "partially_folded"
    MOLTEN_GLOBULE = "molten_globule"
    UNFOLDED = "unfolded"
    MISFOLDED = "misfolded"
    AGGREGATED = "aggregated"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.NATIVE


class BondType(Enum):
    """Types of bonds in proteins"""
    PEPTIDE = "peptide"
    DISULFIDE = "disulfide"
    HYDROGEN = "hydrogen"
    IONIC = "ionic"
    HYDROPHOBIC = "hydrophobic"
    VAN_DER_WAALS = "van_der_waals"

    @classmethod
    def get_color(cls, bond):
        if isinstance(bond, str):
            for member in cls:
                if member.value == bond:
                    bond = member
                    break
        colors = {
            cls.PEPTIDE: "#607D8B",
            cls.DISULFIDE: "#FFC107",
            cls.HYDROGEN: "#03A9F4",
            cls.IONIC: "#F44336",
            cls.HYDROPHOBIC: "#8BC34A",
            cls.VAN_DER_WAALS: "#9E9E9E",
        }
        return colors.get(bond, "#9E9E9E")


# Ramachandran plot regions for validation
RAMACHANDRAN_REGIONS = {
    "favored_alpha": {"phi": (-80, -48), "psi": (-59, -27)},
    "favored_beta": {"phi": (-150, -90), "psi": (90, 150)},
    "favored_left_alpha": {"phi": (50, 70), "psi": (20, 60)},
    "allowed": {"phi": (-180, 180), "psi": (-180, 180)},
    "generously_allowed": {"phi": (-180, 180), "psi": (-180, 180)},
}

# Standard amino acid residue masses (for calculating protein MW)
RESIDUE_MASSES = {
    'A': 71.04, 'R': 156.10, 'N': 114.04, 'D': 115.03, 'C': 103.01,
    'E': 129.04, 'Q': 128.06, 'G': 57.02, 'H': 137.06, 'I': 113.08,
    'L': 113.08, 'K': 128.09, 'M': 131.04, 'F': 147.07, 'P': 97.05,
    'S': 87.03, 'T': 101.05, 'W': 186.08, 'Y': 163.06, 'V': 99.07,
}

# Water mass lost per peptide bond
WATER_MASS = 18.015


def calculate_protein_mass(sequence: str) -> float:
    """Calculate protein molecular mass from sequence"""
    mass = sum(RESIDUE_MASSES.get(aa, 110.0) for aa in sequence.upper())
    # Subtract water for each peptide bond formed
    if len(sequence) > 1:
        mass -= WATER_MASS * (len(sequence) - 1)
    # Add terminal groups (H + OH)
    mass += WATER_MASS
    return round(mass, 2)


def calculate_isoelectric_point(sequence: str) -> float:
    """Estimate protein isoelectric point from sequence"""
    # Count charged residues
    positive = sequence.upper().count('K') + sequence.upper().count('R') + sequence.upper().count('H')
    negative = sequence.upper().count('D') + sequence.upper().count('E')

    # Simple estimation based on charge balance
    if positive == negative:
        return 7.0
    elif positive > negative:
        return 7.0 + min(3.0, (positive - negative) * 0.3)
    else:
        return 7.0 - min(3.0, (negative - positive) * 0.3)


PROTEIN_PROPERTY_METADATA = {
    "molecular_mass": {
        "display_name": "Molecular Mass",
        "unit": "kDa",
        "min_value": 1.0,
        "max_value": 500.0,
    },
    "length": {
        "display_name": "Length",
        "unit": "residues",
        "min_value": 10,
        "max_value": 5000,
    },
    "isoelectric_point": {
        "display_name": "Isoelectric Point",
        "unit": "",
        "min_value": 3.0,
        "max_value": 12.0,
    },
    "helix_percent": {
        "display_name": "Helix Content",
        "unit": "%",
        "min_value": 0,
        "max_value": 100,
    },
    "sheet_percent": {
        "display_name": "Sheet Content",
        "unit": "%",
        "min_value": 0,
        "max_value": 100,
    },
}
