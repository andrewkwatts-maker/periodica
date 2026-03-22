"""
Enums for biological material visualization and properties.
Includes tissue types, ECM components, and mechanical property categories.
"""

from enum import Enum


class BiomaterialType(Enum):
    """Types of biological materials/tissues"""
    # Connective tissues
    BONE_CORTICAL = "bone_cortical"
    BONE_TRABECULAR = "bone_trabecular"
    CARTILAGE = "cartilage"
    TENDON = "tendon"
    LIGAMENT = "ligament"
    ADIPOSE = "adipose"

    # Muscle tissues
    MUSCLE_SKELETAL = "muscle_skeletal"
    MUSCLE_CARDIAC = "muscle_cardiac"
    MUSCLE_SMOOTH = "muscle_smooth"

    # Epithelial tissues
    SKIN = "skin"
    MUCOSA = "mucosa"
    ENDOTHELIUM = "endothelium"

    # Nervous tissue
    BRAIN_GRAY = "brain_gray"
    BRAIN_WHITE = "brain_white"
    NERVE = "nerve"

    # Specialized organs
    LIVER = "liver"
    KIDNEY = "kidney"
    LUNG = "lung"
    HEART = "heart"
    BLOOD = "blood"

    # Other
    OTHER = "other"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.OTHER

    @classmethod
    def get_color(cls, tissue_type):
        if isinstance(tissue_type, str):
            tissue_type = cls.from_string(tissue_type)
        colors = {
            # Bone - ivory/white
            cls.BONE_CORTICAL: "#FFF8E1",
            cls.BONE_TRABECULAR: "#FFFDE7",
            cls.CARTILAGE: "#E1F5FE",
            cls.TENDON: "#F5F5F5",
            cls.LIGAMENT: "#ECEFF1",
            cls.ADIPOSE: "#FFF9C4",

            # Muscle - reds/pinks
            cls.MUSCLE_SKELETAL: "#E57373",
            cls.MUSCLE_CARDIAC: "#EF5350",
            cls.MUSCLE_SMOOTH: "#F48FB1",

            # Epithelial - blues
            cls.SKIN: "#FFCCBC",
            cls.MUCOSA: "#F8BBD0",
            cls.ENDOTHELIUM: "#BBDEFB",

            # Nervous - oranges/yellows
            cls.BRAIN_GRAY: "#BDBDBD",
            cls.BRAIN_WHITE: "#F5F5F5",
            cls.NERVE: "#FFE0B2",

            # Organs
            cls.LIVER: "#795548",
            cls.KIDNEY: "#A1887F",
            cls.LUNG: "#FFCDD2",
            cls.HEART: "#C62828",
            cls.BLOOD: "#D32F2F",
        }
        return colors.get(tissue_type, "#9E9E9E")


class ECMComponent(Enum):
    """Extracellular matrix components"""
    COLLAGEN_I = "collagen_i"
    COLLAGEN_II = "collagen_ii"
    COLLAGEN_III = "collagen_iii"
    COLLAGEN_IV = "collagen_iv"
    ELASTIN = "elastin"
    FIBRONECTIN = "fibronectin"
    LAMININ = "laminin"
    HYALURONAN = "hyaluronan"
    PROTEOGLYCANS = "proteoglycans"
    GLYCOSAMINOGLYCANS = "glycosaminoglycans"
    HYDROXYAPATITE = "hydroxyapatite"
    WATER = "water"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.COLLAGEN_I

    @classmethod
    def get_color(cls, component):
        if isinstance(component, str):
            component = cls.from_string(component)
        colors = {
            cls.COLLAGEN_I: "#FFC107",
            cls.COLLAGEN_II: "#FFD54F",
            cls.COLLAGEN_III: "#FFEB3B",
            cls.COLLAGEN_IV: "#FFF59D",
            cls.ELASTIN: "#4CAF50",
            cls.FIBRONECTIN: "#FF9800",
            cls.LAMININ: "#9C27B0",
            cls.HYALURONAN: "#2196F3",
            cls.PROTEOGLYCANS: "#00BCD4",
            cls.GLYCOSAMINOGLYCANS: "#03A9F4",
            cls.HYDROXYAPATITE: "#FFFFFF",
            cls.WATER: "#E3F2FD",
        }
        return colors.get(component, "#9E9E9E")

    @classmethod
    def get_modulus(cls, component):
        """Get Young's modulus in MPa for ECM component."""
        if isinstance(component, str):
            component = cls.from_string(component)
        moduli = {
            cls.COLLAGEN_I: 1000,      # MPa (varies with hydration)
            cls.COLLAGEN_II: 800,
            cls.COLLAGEN_III: 600,
            cls.COLLAGEN_IV: 400,
            cls.ELASTIN: 0.6,          # Very compliant
            cls.FIBRONECTIN: 10,
            cls.LAMININ: 5,
            cls.HYALURONAN: 0.001,     # Gel-like
            cls.PROTEOGLYCANS: 0.01,
            cls.GLYCOSAMINOGLYCANS: 0.001,
            cls.HYDROXYAPATITE: 117000,  # Ceramic-like
            cls.WATER: 0,
        }
        return moduli.get(component, 1.0)


class MechanicalProperty(Enum):
    """Mechanical property categories"""
    STIFF = "stiff"           # E > 1 GPa
    COMPLIANT = "compliant"   # E < 100 MPa
    ELASTIC = "elastic"       # High elasticity
    VISCOELASTIC = "viscoelastic"
    BRITTLE = "brittle"
    TOUGH = "tough"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.COMPLIANT


class VascularizationLevel(Enum):
    """Degree of vascularization"""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.MODERATE


class BiomaterialLayoutMode(Enum):
    """Layout modes for biomaterial visualization"""
    GRID = "grid"
    TYPE = "type"
    STIFFNESS = "stiffness"
    DENSITY = "density"
    ORGAN_SYSTEM = "organ_system"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID


# Typical mechanical properties (Young's modulus in MPa)
TISSUE_MODULI = {
    'bone_cortical': 18000,      # 15-25 GPa
    'bone_trabecular': 700,      # 0.1-2 GPa
    'cartilage': 10,             # 1-100 MPa
    'tendon': 1500,              # 1-2 GPa
    'ligament': 400,             # 100-1000 MPa
    'muscle_skeletal': 0.5,      # Passive, relaxed
    'muscle_cardiac': 0.1,       # Passive
    'skin': 1,                   # 0.1-10 MPa
    'brain': 0.001,              # Very soft
    'liver': 0.01,
    'lung': 0.005,
    'adipose': 0.002,
}

# Typical densities (g/cm³)
TISSUE_DENSITIES = {
    'bone_cortical': 1.9,
    'bone_trabecular': 0.5,
    'cartilage': 1.1,
    'tendon': 1.15,
    'ligament': 1.1,
    'muscle_skeletal': 1.06,
    'muscle_cardiac': 1.05,
    'skin': 1.1,
    'brain': 1.04,
    'liver': 1.05,
    'lung': 0.3,  # Including air
    'adipose': 0.95,
    'blood': 1.06,
}


BIOMATERIAL_PROPERTY_METADATA = {
    "youngs_modulus": {
        "display_name": "Young's Modulus",
        "unit": "MPa",
        "min_value": 0.001,
        "max_value": 25000,
    },
    "ultimate_strength": {
        "display_name": "Ultimate Strength",
        "unit": "MPa",
        "min_value": 0.01,
        "max_value": 200,
    },
    "density": {
        "display_name": "Density",
        "unit": "g/cm³",
        "min_value": 0.1,
        "max_value": 2.5,
    },
    "porosity": {
        "display_name": "Porosity",
        "unit": "%",
        "min_value": 0,
        "max_value": 95,
    },
    "water_content": {
        "display_name": "Water Content",
        "unit": "%",
        "min_value": 0,
        "max_value": 90,
    },
}
