#====== Playtow/PeriodicTable2/core/pt_enums.py ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

"""
Enums for periodic table visualization properties and encodings.
Centralizes all string-based property checking to prevent typos and improve maintainability.
"""

from enum import Enum, auto


class PTPropertyName(Enum):
    """Element properties that can be visualized or used for ordering"""
    # Basic atomic properties
    ATOMIC_NUMBER = "atomic_number"
    ATOMIC_MASS = "atomic_mass"
    MASS_NUMBER = "mass_number"
    ELECTRONS = "electrons"
    NEUTRONS = "neutrons"
    PROTONS = "protons"

    # Periodic table position
    GROUP = "group"
    PERIOD = "period"
    BLOCK = "block"
    CATEGORY = "category"

    # Electron configuration
    ELECTRON_CONFIG = "electron_config"
    OXIDATION = "oxidation"

    # Physical/chemical properties
    ELECTRONEGATIVITY = "electronegativity"
    IONIZATION_ENERGY = "ionization_energy"
    IONIZATION = "ionization"  # Alias for ionization_energy
    RADIUS = "radius"
    MELTING = "melting"
    BOILING = "boiling"
    DENSITY = "density"
    ELECTRON_AFFINITY = "electron_affinity"
    VALENCE = "valence"
    SPECIFIC_HEAT = "specific_heat"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    ELECTRICAL_CONDUCTIVITY = "electrical_conductivity"

    # Wavelength/spectrum properties
    WAVELENGTH = "wavelength"
    EMISSION_WAVELENGTH = "emission_wavelength"
    VISIBLE_EMISSION_WAVELENGTH = "visible_emission_wavelength"
    IONIZATION_WAVELENGTH = "ionization_wavelength"
    SPECTRUM = "spectrum"

    # Special value
    NONE = "none"

    @classmethod
    def is_wavelength_property(cls, prop):
        """Check if a property is wavelength-based"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        return prop in [
            cls.WAVELENGTH,
            cls.EMISSION_WAVELENGTH,
            cls.VISIBLE_EMISSION_WAVELENGTH,
            cls.IONIZATION_WAVELENGTH,
            cls.SPECTRUM
        ]

    @classmethod
    def is_categorical_property(cls, prop):
        """Check if a property is categorical (discrete values, not continuous)"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        return prop in [
            cls.BLOCK,
            cls.CATEGORY,
            cls.PERIOD,
            cls.GROUP
        ]

    @classmethod
    def get_color_properties(cls):
        """Get ordered list of properties suitable for color encoding"""
        return [
            cls.ATOMIC_NUMBER,
            cls.ATOMIC_MASS,
            cls.IONIZATION,
            cls.ELECTRONEGATIVITY,
            cls.MELTING,
            cls.BOILING,
            cls.RADIUS,
            cls.DENSITY,
            cls.ELECTRON_AFFINITY,
            cls.VALENCE,
            cls.GROUP,
            cls.PERIOD,
            cls.BLOCK,
            cls.SPECIFIC_HEAT,
            cls.THERMAL_CONDUCTIVITY,
            cls.ELECTRICAL_CONDUCTIVITY,
            cls.EMISSION_WAVELENGTH,
            cls.VISIBLE_EMISSION_WAVELENGTH,
            cls.IONIZATION_WAVELENGTH,
            cls.SPECTRUM,
            cls.NONE
        ]

    @classmethod
    def get_size_properties(cls):
        """Get ordered list of properties suitable for size encoding"""
        return [
            cls.ATOMIC_NUMBER,
            cls.ATOMIC_MASS,
            cls.RADIUS,
            cls.IONIZATION,
            cls.ELECTRONEGATIVITY,
            cls.MELTING,
            cls.BOILING,
            cls.DENSITY,
            cls.ELECTRON_AFFINITY,
            cls.VALENCE,
            cls.GROUP,
            cls.PERIOD,
            cls.SPECIFIC_HEAT,
            cls.THERMAL_CONDUCTIVITY,
            cls.ELECTRICAL_CONDUCTIVITY,
            cls.NONE
        ]

    @classmethod
    def get_intensity_properties(cls):
        """Get ordered list of properties suitable for intensity encoding"""
        return [
            cls.ATOMIC_NUMBER,
            cls.ATOMIC_MASS,
            cls.MELTING,
            cls.IONIZATION,
            cls.RADIUS,
            cls.BOILING,
            cls.DENSITY,
            cls.ELECTRON_AFFINITY,
            cls.ELECTRONEGATIVITY,
            cls.VALENCE,
            cls.GROUP,
            cls.PERIOD,
            cls.SPECIFIC_HEAT,
            cls.THERMAL_CONDUCTIVITY,
            cls.ELECTRICAL_CONDUCTIVITY,
            cls.NONE
        ]

    @classmethod
    def get_display_name(cls, prop):
        """Get UI display name for a property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)

        display_names = {
            cls.ATOMIC_NUMBER: "Atomic Number",
            cls.ATOMIC_MASS: "Atomic Mass",
            cls.IONIZATION: "Ionization Energy",
            cls.IONIZATION_ENERGY: "Ionization Energy",
            cls.ELECTRONEGATIVITY: "Electronegativity",
            cls.MELTING: "Melting Point",
            cls.BOILING: "Boiling Point",
            cls.RADIUS: "Atomic Radius",
            cls.DENSITY: "Density",
            cls.ELECTRON_AFFINITY: "Electron Affinity",
            cls.VALENCE: "Valence Electrons",
            cls.GROUP: "Group Number",
            cls.PERIOD: "Period Number",
            cls.BLOCK: "Orbital Block",
            cls.SPECIFIC_HEAT: "Specific Heat",
            cls.THERMAL_CONDUCTIVITY: "Thermal Conductivity",
            cls.ELECTRICAL_CONDUCTIVITY: "Electrical Conductivity",
            cls.WAVELENGTH: "Wavelength",
            cls.EMISSION_WAVELENGTH: "Emission Wavelength",
            cls.VISIBLE_EMISSION_WAVELENGTH: "Visible Emission",
            cls.IONIZATION_WAVELENGTH: "Ionization Wavelength",
            cls.SPECTRUM: "Spectrum Background",
            cls.NONE: "None"
        }
        return display_names.get(prop, "Unknown")

    @classmethod
    def from_display_name(cls, display_name):
        """Convert UI display name to property enum"""
        name_map = {
            "Atomic Number": cls.ATOMIC_NUMBER,
            "Atomic Mass": cls.ATOMIC_MASS,
            "Ionization Energy": cls.IONIZATION,
            "Electronegativity": cls.ELECTRONEGATIVITY,
            "Melting Point": cls.MELTING,
            "Boiling Point": cls.BOILING,
            "Atomic Radius": cls.RADIUS,
            "Density": cls.DENSITY,
            "Electron Affinity": cls.ELECTRON_AFFINITY,
            "Valence Electrons": cls.VALENCE,
            "Group Number": cls.GROUP,
            "Period Number": cls.PERIOD,
            "Orbital Block": cls.BLOCK,
            "Specific Heat": cls.SPECIFIC_HEAT,
            "Thermal Conductivity": cls.THERMAL_CONDUCTIVITY,
            "Electrical Conductivity": cls.ELECTRICAL_CONDUCTIVITY,
            "Wavelength": cls.WAVELENGTH,
            "Emission Wavelength": cls.EMISSION_WAVELENGTH,
            "Visible Emission": cls.VISIBLE_EMISSION_WAVELENGTH,
            "Ionization Wavelength": cls.IONIZATION_WAVELENGTH,
            "Spectrum Background": cls.SPECTRUM,
            "None": cls.NONE
        }
        return name_map.get(display_name, cls.NONE)

    @classmethod
    def from_string(cls, value):
        """Convert string to enum, return NONE if not found"""
        if value is None:
            return cls.NONE
        for member in cls:
            if member.value == value:
                return member
        return cls.NONE


class PTEncodingKey(Enum):
    """Visual encoding channels that can map properties to visual attributes"""
    FILL_COLOR = "fill_color"
    BORDER_COLOR = "border_color"
    BORDER_SIZE = "border_size"
    RING_COLOR = "ring_color"
    RING_SIZE = "ring_size"
    GLOW_COLOR = "glow_color"
    GLOW_INTENSITY = "glow_intensity"
    SYMBOL_TEXT_COLOR = "symbol_text_color"
    ATOMIC_NUMBER_TEXT_COLOR = "atomic_number_text_color"

    @classmethod
    def is_color_encoding(cls, key):
        """Check if an encoding key is for color"""
        if isinstance(key, str):
            key = cls.from_string(key)
        return key in [
            cls.FILL_COLOR,
            cls.BORDER_COLOR,
            cls.RING_COLOR,
            cls.GLOW_COLOR,
            cls.SYMBOL_TEXT_COLOR,
            cls.ATOMIC_NUMBER_TEXT_COLOR
        ]

    @classmethod
    def is_size_encoding(cls, key):
        """Check if an encoding key is for size"""
        if isinstance(key, str):
            key = cls.from_string(key)
        return key in [
            cls.BORDER_SIZE,
            cls.RING_SIZE
        ]

    @classmethod
    def is_intensity_encoding(cls, key):
        """Check if an encoding key is for intensity/alpha"""
        if isinstance(key, str):
            key = cls.from_string(key)
        return key == cls.GLOW_INTENSITY

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        if value is None:
            return None
        for member in cls:
            if member.value == value:
                return member
        return None


class PTControlType(Enum):
    """Types of visual encoding controls"""
    COLOR = "color"
    SIZE = "size"
    INTENSITY = "intensity"


class PTWavelengthMode(Enum):
    """Display modes for wavelength-based properties"""
    SPECTRUM = "spectrum"  # Use rainbow spectrum from wavelength_to_rgb
    GRADIENT = "gradient"  # Use A-B gradient lerp like other properties

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        if value is None or value == "spectrum":
            return cls.SPECTRUM
        if value == "gradient":
            return cls.GRADIENT
        return cls.SPECTRUM  # Default to spectrum


class PTElementDataKey(Enum):
    """Keys for accessing element data dictionary"""
    # Wavelength/spectrum data
    WAVELENGTH_NM = "wavelength_nm"
    EMISSION_WAVELENGTH = "emission_wavelength"
    VISIBLE_EMISSION_WAVELENGTH = "visible_emission_wavelength"
    IONIZATION_WAVELENGTH = "ionization_wavelength"
    SPECTRUM_LINES = "spectrum_lines"

    # Basic atomic properties
    ATOMIC_NUMBER = "z"
    SYMBOL = "symbol"
    NAME = "name"
    MASS_NUMBER = "mass_number"

    # Physical/chemical properties
    IONIZATION_ENERGY = "ie"
    ELECTRONEGATIVITY = "electronegativity"
    ATOMIC_RADIUS = "atomic_radius"
    MELTING_POINT = "melting_point"
    BOILING_POINT = "boiling_point"
    DENSITY = "density"
    ELECTRON_AFFINITY = "electron_affinity"
    VALENCE_ELECTRONS = "valence_electrons"

    # Periodic table position
    PERIOD = "period"
    GROUP = "group"
    BLOCK = "block"
    CATEGORY = "category"

    # Visualization data
    BLOCK_COLOR = "block_color"


class PTPropertyType(Enum):
    """Types of visual properties that can be mapped"""
    COLOR = "color"
    SIZE = "size"
    INTENSITY = "intensity"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        if value == "color":
            return cls.COLOR
        elif value == "size":
            return cls.SIZE
        elif value == "intensity":
            return cls.INTENSITY
        return cls.COLOR  # Default to color


class PTEncodingType(Enum):
    """Types of visual encodings (where properties are mapped)"""
    FILL = "fill"
    BORDER = "border"
    RING = "ring"
    GLOW = "glow"
    SYMBOL_TEXT = "symbol_text"
    ATOMIC_NUMBER_TEXT = "atomic_number_text"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.FILL  # Default to fill

    def get_wavelength_mode_attr(self):
        """Get the wavelength mode attribute name for this encoding type"""
        return f"{self.value}_wavelength_mode"

    def get_property_attr(self):
        """Get the property attribute name for this encoding type"""
        if self == PTEncodingType.FILL:
            return "fill_color_property"
        elif self == PTEncodingType.BORDER:
            return "border_color_property"
        elif self == PTEncodingType.RING:
            return "ring_color_property"
        elif self == PTEncodingType.GLOW:
            return "glow_color_property"
        elif self == PTEncodingType.SYMBOL_TEXT:
            return "symbol_text_color_property"
        elif self == PTEncodingType.ATOMIC_NUMBER_TEXT:
            return "atomic_number_text_color_property"
        return "fill_color_property"


class PTLayoutMode(Enum):
    """Layout modes for the periodic table visualization"""
    TABLE = "table"
    SPIRAL = "spiral"
    CIRCULAR = "circular"
    LINEAR = "linear"
    SERPENTINE = "serpentine"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.TABLE  # Default to table


# Mapping from encoding key strings to PTEncodingType
ENCODING_KEY_TO_TYPE = {
    "fill_color": PTEncodingType.FILL,
    "border_color": PTEncodingType.BORDER,
    "ring_color": PTEncodingType.RING,
    "glow_color": PTEncodingType.GLOW,
    "symbol_text_color": PTEncodingType.SYMBOL_TEXT,
    "atomic_number_text_color": PTEncodingType.ATOMIC_NUMBER_TEXT,
}

# Mapping from PTEncodingKey enum to PTEncodingType
ENCODING_KEY_ENUM_TO_TYPE = {
    PTEncodingKey.FILL_COLOR: PTEncodingType.FILL,
    PTEncodingKey.BORDER_COLOR: PTEncodingType.BORDER,
    PTEncodingKey.RING_COLOR: PTEncodingType.RING,
    PTEncodingKey.GLOW_COLOR: PTEncodingType.GLOW,
    PTEncodingKey.SYMBOL_TEXT_COLOR: PTEncodingType.SYMBOL_TEXT,
    PTEncodingKey.ATOMIC_NUMBER_TEXT_COLOR: PTEncodingType.ATOMIC_NUMBER_TEXT,
}
