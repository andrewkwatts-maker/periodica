"""
Enums for material (engineering-level) visualization properties and encodings.
Materials represent the observable/engineering level above alloys.
"""

from enum import Enum


class MaterialLayoutMode(Enum):
    """Layout modes for material visualization"""
    CATEGORY = "category"
    PROPERTY_SCATTER = "property_scatter"
    STRENGTH_STIFFNESS = "strength_stiffness"
    THERMAL_MAP = "thermal_map"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.CATEGORY

    @classmethod
    def get_display_name(cls, mode):
        if isinstance(mode, str):
            mode = cls.from_string(mode)
        display_names = {
            cls.CATEGORY: "By Category",
            cls.PROPERTY_SCATTER: "Property Plot",
            cls.STRENGTH_STIFFNESS: "Strength vs Stiffness",
            cls.THERMAL_MAP: "Thermal Properties"
        }
        return display_names.get(mode, "Unknown")


class MaterialCategory(Enum):
    """Categories of engineering materials"""
    STRUCTURAL_STEEL = "Structural Steel"
    STAINLESS_STEEL = "Stainless Steel"
    ALUMINUM_ALLOY = "Aluminum Alloy"
    TITANIUM_ALLOY = "Titanium Alloy"
    COPPER_ALLOY = "Copper Alloy"
    NICKEL_SUPERALLOY = "Nickel Superalloy"
    POLYMER = "Polymer"
    CERAMIC = "Ceramic"
    COMPOSITE = "Composite"
    OTHER = "Other"

    @classmethod
    def from_string(cls, value):
        if value is None:
            return cls.OTHER
        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        # Fuzzy matching
        if 'steel' in value_lower and 'stainless' not in value_lower:
            return cls.STRUCTURAL_STEEL
        if 'stainless' in value_lower:
            return cls.STAINLESS_STEEL
        if 'aluminum' in value_lower or 'aluminium' in value_lower:
            return cls.ALUMINUM_ALLOY
        if 'titanium' in value_lower:
            return cls.TITANIUM_ALLOY
        if 'copper' in value_lower or 'brass' in value_lower or 'bronze' in value_lower:
            return cls.COPPER_ALLOY
        if 'nickel' in value_lower or 'inconel' in value_lower:
            return cls.NICKEL_SUPERALLOY
        if 'polymer' in value_lower or 'plastic' in value_lower or 'pe' in value_lower:
            return cls.POLYMER
        if 'ceramic' in value_lower or 'alumina' in value_lower or 'glass' in value_lower:
            return cls.CERAMIC
        if 'composite' in value_lower or 'fiber' in value_lower or 'cfrp' in value_lower:
            return cls.COMPOSITE
        return cls.OTHER

    @classmethod
    def get_color(cls, category):
        if isinstance(category, str):
            category = cls.from_string(category)
        colors = {
            cls.STRUCTURAL_STEEL: "#607D8B",
            cls.STAINLESS_STEEL: "#B0BEC5",
            cls.ALUMINUM_ALLOY: "#90CAF9",
            cls.TITANIUM_ALLOY: "#8E8E8E",
            cls.COPPER_ALLOY: "#B87333",
            cls.NICKEL_SUPERALLOY: "#FF6B35",
            cls.POLYMER: "#E8D5B7",
            cls.CERAMIC: "#FFFFFF",
            cls.COMPOSITE: "#1A1A1A",
            cls.OTHER: "#9E9E9E"
        }
        return colors.get(category, "#9E9E9E")


class MaterialProperty(Enum):
    """Properties that can be used for visual encodings"""
    YOUNGS_MODULUS = "youngs_modulus"
    YIELD_STRENGTH = "yield_strength"
    ULTIMATE_STRENGTH = "ultimate_strength"
    DENSITY = "density"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    MELTING_POINT = "melting_point"
    FRACTURE_TOUGHNESS = "fracture_toughness"
    HARDNESS = "hardness"
    ELONGATION = "elongation"
    FATIGUE_LIMIT = "fatigue_limit"
    SPECIFIC_STRENGTH = "specific_strength"
    SPECIFIC_STIFFNESS = "specific_stiffness"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.YOUNGS_MODULUS

    @classmethod
    def get_display_name(cls, prop):
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        display_names = {
            cls.YOUNGS_MODULUS: "Young's Modulus (GPa)",
            cls.YIELD_STRENGTH: "Yield Strength (MPa)",
            cls.ULTIMATE_STRENGTH: "Ultimate Strength (MPa)",
            cls.DENSITY: "Density (kg/m³)",
            cls.THERMAL_CONDUCTIVITY: "Thermal Conductivity (W/m·K)",
            cls.MELTING_POINT: "Melting Point (K)",
            cls.FRACTURE_TOUGHNESS: "Fracture Toughness (MPa√m)",
            cls.HARDNESS: "Hardness (HV)",
            cls.ELONGATION: "Elongation (%)",
            cls.FATIGUE_LIMIT: "Fatigue Limit (MPa)",
            cls.SPECIFIC_STRENGTH: "Specific Strength (kN·m/kg)",
            cls.SPECIFIC_STIFFNESS: "Specific Stiffness (MN·m/kg)"
        }
        return display_names.get(prop, prop.value)

    @classmethod
    def get_value_from_material(cls, prop, material_data):
        """Extract property value from material JSON data"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)

        elastic = material_data.get('ElasticProperties', {})
        strength = material_data.get('StrengthProperties', {})
        thermal = material_data.get('ThermalProperties', {})
        hardness = material_data.get('Hardness', {})
        ductility = material_data.get('Ductility', {})
        fatigue = material_data.get('FatigueProperties', {})
        fracture = material_data.get('FractureMechanics', {})
        physical = material_data.get('PhysicalProperties', {})

        extractors = {
            cls.YOUNGS_MODULUS: lambda: elastic.get('YoungsModulus_GPa', 0),
            cls.YIELD_STRENGTH: lambda: strength.get('YieldStrength_MPa', 0),
            cls.ULTIMATE_STRENGTH: lambda: strength.get('UltimateTensileStrength_MPa', 0),
            cls.DENSITY: lambda: physical.get('Density_kg_m3', 0),
            cls.THERMAL_CONDUCTIVITY: lambda: (
                thermal.get('ThermalConductivity_W_mK', 0)
                if isinstance(thermal.get('ThermalConductivity_W_mK'), (int, float))
                else thermal.get('ThermalConductivity_W_mK', {}).get('at_20C', 0)
            ),
            cls.MELTING_POINT: lambda: thermal.get('MeltingPoint_K', 0),
            cls.FRACTURE_TOUGHNESS: lambda: fracture.get('FractureToughness_KIC_MPa_sqrt_m', 0),
            cls.HARDNESS: lambda: hardness.get('Vickers_HV', hardness.get('Brinell_HB', 0)),
            cls.ELONGATION: lambda: ductility.get('Elongation_percent', 0),
            cls.FATIGUE_LIMIT: lambda: fatigue.get('FatigueLimit_MPa', 0),
            cls.SPECIFIC_STRENGTH: lambda: (
                strength.get('YieldStrength_MPa', 0) * 1000 / physical.get('Density_kg_m3', 1)
                if physical.get('Density_kg_m3', 0) > 0 else 0
            ),
            cls.SPECIFIC_STIFFNESS: lambda: (
                elastic.get('YoungsModulus_GPa', 0) * 1e6 / physical.get('Density_kg_m3', 1)
                if physical.get('Density_kg_m3', 0) > 0 else 0
            )
        }

        extractor = extractors.get(prop, lambda: 0)
        try:
            return extractor()
        except (KeyError, TypeError, ZeroDivisionError):
            return 0


def get_material_color(category):
    """Get color for a material category"""
    return MaterialCategory.get_color(category)
