"""
Enums for cell visualization and properties.
Includes cell types, organisms, cell cycle stages, and metabolic states.
"""

from enum import Enum


class CellType(Enum):
    """Types of cells"""
    # Blood cells
    ERYTHROCYTE = "erythrocyte"
    LEUKOCYTE = "leukocyte"
    PLATELET = "platelet"
    NEUTROPHIL = "neutrophil"
    LYMPHOCYTE = "lymphocyte"
    MONOCYTE = "monocyte"
    MACROPHAGE = "macrophage"

    # Epithelial cells
    EPITHELIAL = "epithelial"
    KERATINOCYTE = "keratinocyte"
    ENTEROCYTE = "enterocyte"

    # Connective tissue
    FIBROBLAST = "fibroblast"
    ADIPOCYTE = "adipocyte"
    CHONDROCYTE = "chondrocyte"
    OSTEOCYTE = "osteocyte"
    OSTEOBLAST = "osteoblast"
    OSTEOCLAST = "osteoclast"

    # Muscle cells
    MYOCYTE = "myocyte"
    CARDIOMYOCYTE = "cardiomyocyte"
    SMOOTH_MUSCLE = "smooth_muscle"

    # Neural cells
    NEURON = "neuron"
    ASTROCYTE = "astrocyte"
    OLIGODENDROCYTE = "oligodendrocyte"
    MICROGLIA = "microglia"
    SCHWANN_CELL = "schwann_cell"

    # Glandular cells
    HEPATOCYTE = "hepatocyte"
    PANCREATIC_BETA = "pancreatic_beta"
    THYROID = "thyroid"

    # Reproductive cells
    SPERMATOZOON = "spermatozoon"
    OOCYTE = "oocyte"

    # Stem cells
    STEM_CELL = "stem_cell"
    HEMATOPOIETIC = "hematopoietic"
    MESENCHYMAL = "mesenchymal"

    # Other
    OTHER = "other"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.OTHER

    @classmethod
    def get_color(cls, cell_type):
        if isinstance(cell_type, str):
            cell_type = cls.from_string(cell_type)
        colors = {
            # Blood cells - reds
            cls.ERYTHROCYTE: "#E53935",
            cls.LEUKOCYTE: "#F5F5F5",
            cls.PLATELET: "#FFCDD2",
            cls.NEUTROPHIL: "#FFEBEE",
            cls.LYMPHOCYTE: "#E8EAF6",
            cls.MONOCYTE: "#F3E5F5",
            cls.MACROPHAGE: "#EDE7F6",

            # Epithelial - blues
            cls.EPITHELIAL: "#2196F3",
            cls.KERATINOCYTE: "#64B5F6",
            cls.ENTEROCYTE: "#42A5F5",

            # Connective - greens
            cls.FIBROBLAST: "#4CAF50",
            cls.ADIPOCYTE: "#FFC107",
            cls.CHONDROCYTE: "#8BC34A",
            cls.OSTEOCYTE: "#CDDC39",
            cls.OSTEOBLAST: "#C0CA33",
            cls.OSTEOCLAST: "#AFB42B",

            # Muscle - purples
            cls.MYOCYTE: "#9C27B0",
            cls.CARDIOMYOCYTE: "#E91E63",
            cls.SMOOTH_MUSCLE: "#AB47BC",

            # Neural - oranges
            cls.NEURON: "#FF9800",
            cls.ASTROCYTE: "#FFB74D",
            cls.OLIGODENDROCYTE: "#FFA726",
            cls.MICROGLIA: "#FF7043",
            cls.SCHWANN_CELL: "#FFCC80",

            # Glandular - teals
            cls.HEPATOCYTE: "#009688",
            cls.PANCREATIC_BETA: "#26A69A",
            cls.THYROID: "#4DB6AC",

            # Reproductive - pinks
            cls.SPERMATOZOON: "#F48FB1",
            cls.OOCYTE: "#F06292",

            # Stem cells - gold
            cls.STEM_CELL: "#FFD700",
            cls.HEMATOPOIETIC: "#FFEB3B",
            cls.MESENCHYMAL: "#FDD835",
        }
        return colors.get(cell_type, "#9E9E9E")


class CellCyclePhase(Enum):
    """Cell cycle phases"""
    G0 = "g0"          # Quiescent
    G1 = "g1"          # Gap 1
    S = "s"            # DNA Synthesis
    G2 = "g2"          # Gap 2
    M_PROPHASE = "prophase"
    M_METAPHASE = "metaphase"
    M_ANAPHASE = "anaphase"
    M_TELOPHASE = "telophase"
    CYTOKINESIS = "cytokinesis"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.G1


class MetabolicState(Enum):
    """Metabolic states of cells"""
    AEROBIC = "aerobic"           # Normal oxidative phosphorylation
    ANAEROBIC = "anaerobic"       # Glycolysis only
    WARBURG = "warburg"           # Aerobic glycolysis (cancer)
    DORMANT = "dormant"           # Low metabolic activity
    ACTIVE = "active"             # High metabolic activity
    STRESSED = "stressed"         # Stress response

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.AEROBIC

    @classmethod
    def get_color(cls, state):
        if isinstance(state, str):
            state = cls.from_string(state)
        colors = {
            cls.AEROBIC: "#4CAF50",
            cls.ANAEROBIC: "#FF9800",
            cls.WARBURG: "#F44336",
            cls.DORMANT: "#9E9E9E",
            cls.ACTIVE: "#2196F3",
            cls.STRESSED: "#E91E63",
        }
        return colors.get(state, "#9E9E9E")


class Organism(Enum):
    """Organism types"""
    HOMO_SAPIENS = "homo_sapiens"
    MUS_MUSCULUS = "mus_musculus"
    RATTUS_NORVEGICUS = "rattus_norvegicus"
    ESCHERICHIA_COLI = "escherichia_coli"
    SACCHAROMYCES_CEREVISIAE = "saccharomyces_cerevisiae"
    CAENORHABDITIS_ELEGANS = "caenorhabditis_elegans"
    DROSOPHILA_MELANOGASTER = "drosophila_melanogaster"
    ARABIDOPSIS_THALIANA = "arabidopsis_thaliana"
    OTHER = "other"

    @classmethod
    def from_string(cls, value):
        value_lower = value.lower().replace(' ', '_')
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.OTHER

    @classmethod
    def get_display_name(cls, organism):
        if isinstance(organism, str):
            organism = cls.from_string(organism)
        names = {
            cls.HOMO_SAPIENS: "Homo sapiens",
            cls.MUS_MUSCULUS: "Mus musculus",
            cls.RATTUS_NORVEGICUS: "Rattus norvegicus",
            cls.ESCHERICHIA_COLI: "E. coli",
            cls.SACCHAROMYCES_CEREVISIAE: "S. cerevisiae",
            cls.CAENORHABDITIS_ELEGANS: "C. elegans",
            cls.DROSOPHILA_MELANOGASTER: "D. melanogaster",
            cls.ARABIDOPSIS_THALIANA: "A. thaliana",
        }
        return names.get(organism, "Unknown")


class TissueType(Enum):
    """Tissue types where cells are found"""
    BLOOD = "blood"
    BONE = "bone"
    CARTILAGE = "cartilage"
    MUSCLE_SKELETAL = "muscle_skeletal"
    MUSCLE_CARDIAC = "muscle_cardiac"
    MUSCLE_SMOOTH = "muscle_smooth"
    NERVOUS = "nervous"
    EPITHELIAL = "epithelial"
    CONNECTIVE = "connective"
    ADIPOSE = "adipose"
    LIVER = "liver"
    KIDNEY = "kidney"
    LUNG = "lung"
    SKIN = "skin"
    BONE_MARROW = "bone_marrow"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.CONNECTIVE

    @classmethod
    def get_color(cls, tissue):
        if isinstance(tissue, str):
            tissue = cls.from_string(tissue)
        colors = {
            cls.BLOOD: "#E53935",
            cls.BONE: "#FFF9C4",
            cls.CARTILAGE: "#E1F5FE",
            cls.MUSCLE_SKELETAL: "#F48FB1",
            cls.MUSCLE_CARDIAC: "#E91E63",
            cls.MUSCLE_SMOOTH: "#CE93D8",
            cls.NERVOUS: "#FFE0B2",
            cls.EPITHELIAL: "#BBDEFB",
            cls.CONNECTIVE: "#C8E6C9",
            cls.ADIPOSE: "#FFF59D",
            cls.LIVER: "#795548",
            cls.KIDNEY: "#8D6E63",
            cls.LUNG: "#F8BBD0",
            cls.SKIN: "#FFCCBC",
            cls.BONE_MARROW: "#FFCDD2",
        }
        return colors.get(tissue, "#9E9E9E")


class CellLayoutMode(Enum):
    """Layout modes for cell visualization"""
    GRID = "grid"
    TYPE = "type"
    TISSUE = "tissue"
    SIZE = "size"
    METABOLIC_RATE = "metabolic_rate"
    ORGANISM = "organism"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID


# Typical cell sizes (in micrometers)
CELL_SIZES = {
    'erythrocyte': 7.5,
    'leukocyte': 12.0,
    'platelet': 2.5,
    'neutrophil': 12.0,
    'lymphocyte': 8.0,
    'monocyte': 18.0,
    'macrophage': 21.0,
    'epithelial': 20.0,
    'fibroblast': 25.0,
    'adipocyte': 100.0,
    'hepatocyte': 20.0,
    'neuron': 30.0,
    'myocyte': 100.0,
    'cardiomyocyte': 100.0,
    'oocyte': 120.0,
    'spermatozoon': 50.0,  # Total length including tail
}

# Typical metabolic rates (femtowatts)
CELL_METABOLIC_RATES = {
    'erythrocyte': 1.2,
    'leukocyte': 15.0,
    'neutrophil': 20.0,
    'lymphocyte': 10.0,
    'fibroblast': 50.0,
    'hepatocyte': 200.0,
    'neuron': 100.0,
    'myocyte': 150.0,
    'cardiomyocyte': 300.0,
    'adipocyte': 30.0,
}

# Typical cell lifespans (in days, -1 for lifetime)
CELL_LIFESPANS = {
    'erythrocyte': 120,
    'platelet': 10,
    'neutrophil': 5,
    'lymphocyte': 1000,  # Memory cells
    'monocyte': 3,
    'epithelial': 5,
    'enterocyte': 4,
    'keratinocyte': 30,
    'hepatocyte': 200,
    'neuron': -1,  # Lifetime
    'cardiomyocyte': -1,  # Lifetime
    'osteocyte': 25000,  # ~70 years
}


CELL_PROPERTY_METADATA = {
    "diameter": {
        "display_name": "Diameter",
        "unit": "μm",
        "min_value": 1,
        "max_value": 200,
    },
    "volume": {
        "display_name": "Volume",
        "unit": "fL",
        "min_value": 1,
        "max_value": 10000000,
    },
    "mass": {
        "display_name": "Mass",
        "unit": "pg",
        "min_value": 0.1,
        "max_value": 1000000,
    },
    "metabolic_rate": {
        "display_name": "Metabolic Rate",
        "unit": "fW",
        "min_value": 0.1,
        "max_value": 1000,
    },
    "doubling_time": {
        "display_name": "Doubling Time",
        "unit": "hours",
        "min_value": 10,
        "max_value": 1000,
    },
    "lifespan": {
        "display_name": "Lifespan",
        "unit": "days",
        "min_value": 1,
        "max_value": 30000,
    },
}
