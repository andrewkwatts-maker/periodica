"""
Enums for cell component visualization and properties.
Includes organelle types, membrane types, and cellular compartments.
"""

from enum import Enum


class OrganelleType(Enum):
    """Types of cellular organelles"""
    RIBOSOME = "ribosome"
    MITOCHONDRION = "mitochondrion"
    NUCLEUS = "nucleus"
    ENDOPLASMIC_RETICULUM = "endoplasmic_reticulum"
    GOLGI_APPARATUS = "golgi_apparatus"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    CHLOROPLAST = "chloroplast"
    VACUOLE = "vacuole"
    CENTROSOME = "centrosome"
    CYTOSKELETON = "cytoskeleton"
    PLASMA_MEMBRANE = "plasma_membrane"
    PROTEASOME = "proteasome"
    SPLICEOSOME = "spliceosome"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.RIBOSOME

    @classmethod
    def get_color(cls, organelle):
        if isinstance(organelle, str):
            organelle = cls.from_string(organelle)
        colors = {
            cls.RIBOSOME: "#9C27B0",          # Purple
            cls.MITOCHONDRION: "#FF9800",     # Orange
            cls.NUCLEUS: "#2196F3",           # Blue
            cls.ENDOPLASMIC_RETICULUM: "#4CAF50",  # Green
            cls.GOLGI_APPARATUS: "#FFC107",   # Amber
            cls.LYSOSOME: "#F44336",          # Red
            cls.PEROXISOME: "#00BCD4",        # Cyan
            cls.CHLOROPLAST: "#8BC34A",       # Light green
            cls.VACUOLE: "#03A9F4",           # Light blue
            cls.CENTROSOME: "#E91E63",        # Pink
            cls.CYTOSKELETON: "#795548",      # Brown
            cls.PLASMA_MEMBRANE: "#607D8B",   # Blue grey
            cls.PROTEASOME: "#673AB7",        # Deep purple
            cls.SPLICEOSOME: "#009688",       # Teal
        }
        return colors.get(organelle, "#9E9E9E")


class MembraneType(Enum):
    """Types of biological membranes"""
    PLASMA = "plasma"
    NUCLEAR = "nuclear"
    MITOCHONDRIAL_OUTER = "mitochondrial_outer"
    MITOCHONDRIAL_INNER = "mitochondrial_inner"
    ER = "endoplasmic_reticulum"
    GOLGI = "golgi"
    LYSOSOMAL = "lysosomal"
    VACUOLAR = "vacuolar"
    THYLAKOID = "thylakoid"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.PLASMA


class CellularCompartment(Enum):
    """Cellular compartments/locations"""
    CYTOPLASM = "cytoplasm"
    NUCLEUS = "nucleus"
    NUCLEOLUS = "nucleolus"
    MITOCHONDRIA = "mitochondria"
    ER_LUMEN = "er_lumen"
    GOLGI_LUMEN = "golgi_lumen"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    EXTRACELLULAR = "extracellular"
    PLASMA_MEMBRANE = "plasma_membrane"
    CYTOSKELETON = "cytoskeleton"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.CYTOPLASM

    @classmethod
    def get_color(cls, compartment):
        if isinstance(compartment, str):
            compartment = cls.from_string(compartment)
        colors = {
            cls.CYTOPLASM: "#81C784",
            cls.NUCLEUS: "#64B5F6",
            cls.NUCLEOLUS: "#42A5F5",
            cls.MITOCHONDRIA: "#FFB74D",
            cls.ER_LUMEN: "#AED581",
            cls.GOLGI_LUMEN: "#FFD54F",
            cls.LYSOSOME: "#E57373",
            cls.PEROXISOME: "#4DD0E1",
            cls.EXTRACELLULAR: "#90A4AE",
            cls.PLASMA_MEMBRANE: "#A1887F",
            cls.CYTOSKELETON: "#BCAAA4",
        }
        return colors.get(compartment, "#9E9E9E")


class ComponentFunction(Enum):
    """Functional categories for cell components"""
    PROTEIN_SYNTHESIS = "protein_synthesis"
    ENERGY_PRODUCTION = "energy_production"
    GENETIC_STORAGE = "genetic_storage"
    PROTEIN_PROCESSING = "protein_processing"
    PROTEIN_SORTING = "protein_sorting"
    DEGRADATION = "degradation"
    SIGNALING = "signaling"
    TRANSPORT = "transport"
    STRUCTURAL = "structural"
    CELL_DIVISION = "cell_division"
    RNA_PROCESSING = "rna_processing"
    PHOTOSYNTHESIS = "photosynthesis"
    STORAGE = "storage"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.STRUCTURAL

    @classmethod
    def get_color(cls, func):
        if isinstance(func, str):
            func = cls.from_string(func)
        colors = {
            cls.PROTEIN_SYNTHESIS: "#9C27B0",
            cls.ENERGY_PRODUCTION: "#FF9800",
            cls.GENETIC_STORAGE: "#2196F3",
            cls.PROTEIN_PROCESSING: "#4CAF50",
            cls.PROTEIN_SORTING: "#FFC107",
            cls.DEGRADATION: "#F44336",
            cls.SIGNALING: "#E91E63",
            cls.TRANSPORT: "#00BCD4",
            cls.STRUCTURAL: "#795548",
            cls.CELL_DIVISION: "#673AB7",
            cls.RNA_PROCESSING: "#009688",
            cls.PHOTOSYNTHESIS: "#8BC34A",
            cls.STORAGE: "#607D8B",
        }
        return colors.get(func, "#9E9E9E")


class CellComponentLayoutMode(Enum):
    """Layout modes for cell component visualization"""
    GRID = "grid"
    TYPE = "type"
    FUNCTION = "function"
    SIZE = "size"
    COMPARTMENT = "compartment"
    COPY_NUMBER = "copy_number"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID


# Typical sizes of organelles (in micrometers)
ORGANELLE_SIZES = {
    'ribosome': 0.025,           # 25 nm
    'proteasome': 0.015,         # 15 nm
    'spliceosome': 0.025,        # 25 nm
    'lysosome': 0.5,             # 0.1-1.2 μm
    'peroxisome': 0.4,           # 0.1-1.0 μm
    'centrosome': 1.0,           # ~1 μm
    'endoplasmic_reticulum': 5.0,  # Network
    'golgi_apparatus': 3.0,      # ~1-3 μm
    'mitochondrion': 2.0,        # 1-10 μm length
    'nucleus': 6.0,              # 5-10 μm
    'chloroplast': 5.0,          # 5-10 μm
    'vacuole': 10.0,             # Variable, up to 80% of cell
}

# Typical copy numbers per cell
ORGANELLE_COPY_NUMBERS = {
    'ribosome': 10000000,        # 10 million
    'proteasome': 100000,        # ~100k
    'spliceosome': 50000,        # ~50k
    'lysosome': 300,             # 50-1000
    'peroxisome': 400,           # 100-1000
    'centrosome': 2,             # 1-2
    'endoplasmic_reticulum': 1,  # Continuous network
    'golgi_apparatus': 1,        # Single stack (or multiple)
    'mitochondrion': 1000,       # 100-2000+
    'nucleus': 1,                # 1 (usually)
    'chloroplast': 50,           # 10-100
    'vacuole': 1,                # 1 central vacuole in plants
}


CELL_COMPONENT_PROPERTY_METADATA = {
    "diameter": {
        "display_name": "Diameter",
        "unit": "μm",
        "min_value": 0.01,
        "max_value": 100,
    },
    "copy_number": {
        "display_name": "Copy Number",
        "unit": "per cell",
        "min_value": 1,
        "max_value": 100000000,
    },
    "protein_count": {
        "display_name": "Protein Count",
        "unit": "",
        "min_value": 1,
        "max_value": 10000,
    },
    "mass": {
        "display_name": "Mass",
        "unit": "MDa",
        "min_value": 0.1,
        "max_value": 10000,
    },
}
