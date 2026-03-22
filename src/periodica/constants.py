"""
Global constants for property names and configuration.
This file provides static strings to replace free strings throughout the codebase.
"""

# ============================================================================
# PROPERTY NAME CONSTANTS
# ============================================================================

# Visual Property Types (what visual channel to encode)
class PropertyType:
    """Visual property types for encoding data"""
    FILL = "fill"
    BORDER_COLOR = "border_color"
    BORDER_SIZE = "border_size"
    RING_COLOR = "ring_color"
    RING_SIZE = "ring_size"
    GLOW_COLOR = "glow_color"
    GLOW_INTENSITY = "glow_intensity"
    GLOW_RADIUS = "glow_radius"
    SYMBOL_TEXT_COLOR = "symbol_text"
    ATOMIC_NUMBER_TEXT_COLOR = "atomic_number_text"


# Data Properties (what data to encode)
class DataProperty:
    """Element data properties that can be visualized"""
    NONE = "none"
    ATOMIC_NUMBER = "atomic_number"
    IONIZATION = "ionization"
    ELECTRONEGATIVITY = "electronegativity"
    BLOCK = "block"
    WAVELENGTH = "wavelength"
    MELTING = "melting"
    RADIUS = "radius"
    DENSITY = "density"
    BOILING = "boiling"
    ELECTRON_AFFINITY = "electron_affinity"
    VALENCE = "valence"


# Element Data Keys (keys in element dictionaries)
class ElementKey:
    """Keys for element data dictionaries"""
    SYMBOL = "symbol"
    NAME = "name"
    Z = "z"
    IE = "ie"
    IONIZATION_ENERGY = "ionization_energy"
    BLOCK = "block"
    BLOCK_COLOR = "block_color"
    PERIOD = "period"
    GROUP = "group"
    FREQ_PHZ = "freq_phz"
    WAVELENGTH_NM = "wavelength_nm"
    ELECTRONEGATIVITY = "electronegativity"
    ATOMIC_RADIUS = "atomic_radius"
    MELTING_POINT = "melting_point"
    MELTING = "melting"
    BOILING_POINT = "boiling_point"
    BOILING = "boiling"
    DENSITY = "density"
    ELECTRON_AFFINITY = "electron_affinity"
    VALENCE_ELECTRONS = "valence_electrons"
    VALENCE = "valence"
    ELECTRON_CONFIG = "electron_config"
    ISOTOPES = "isotopes"
    SPECTRUM_LINES = "spectrum_lines"


# Layout Modes
class LayoutMode:
    """Layout display modes"""
    CIRCULAR = "circular"
    SPIRAL = "spiral"
    SERPENTINE = "serpentine"
    TABLE = "table"
    LINEAR = "linear"


# Glow Types
class GlowType:
    """Glow rendering types"""
    INTERNAL = "internal"
    EXTERNAL = "external"


# Orbital Blocks
class OrbitalBlock:
    """Orbital block types"""
    S = "s"
    P = "p"
    D = "d"
    F = "f"


# ============================================================================
# NUMERIC CONSTANTS
# ============================================================================

class PhysicalConstants:
    """Physical and mathematical constants"""
    SPEED_OF_LIGHT = 299792458  # m/s
    PLANCK_CONSTANT_EV_S = 4.135667696e-15  # eV·s
    PLANCK_CONSTANT_J_S = 6.62607015e-34  # J·s
    ELECTRON_CHARGE = 1.602176634e-19  # C (Coulombs) or J/eV
    RYDBERG_CONSTANT = 10973731.6  # m^-1
    NUCLEAR_RADIUS_CONSTANT = 1.2  # femtometers (fm)


class VisualizationConstants:
    """Constants for visualization parameters"""
    # Default ranges for properties
    IE_MIN = 3.5  # eV
    IE_MAX = 25.0  # eV
    ELECTRONEG_MIN = 0.0
    ELECTRONEG_MAX = 4.0
    MELTING_MIN = 0  # K
    MELTING_MAX = 4000  # K
    BOILING_MIN = 0  # K
    BOILING_MAX = 4000  # K
    RADIUS_MIN = 30  # pm
    RADIUS_MAX = 350  # pm
    DENSITY_MIN = 0.0001  # g/cm³
    DENSITY_MAX = 25  # g/cm³
    ELECTRON_AFFINITY_MIN = -10  # kJ/mol
    ELECTRON_AFFINITY_MAX = 350  # kJ/mol
    VALENCE_MIN = 1
    VALENCE_MAX = 8
    ATOMIC_NUMBER_MIN = 1
    ATOMIC_NUMBER_MAX = 118
    WAVELENGTH_MIN = 380  # nm (violet)
    WAVELENGTH_MAX = 780  # nm (red)

    # Spectrum visualization
    VISIBLE_SPECTRUM_MIN = 380  # nm (violet)
    VISIBLE_SPECTRUM_MAX = 750  # nm (red)
    SPECTRUM_INTENSITY_THRESHOLD = 0.1  # Minimum intensity to display
    SPECTRUM_MARK_MIN_LENGTH = 2  # pixels
    SPECTRUM_MARK_MAX_LENGTH = 6  # pixels
    SPECTRUM_LINE_WIDTH = 2  # pixels

    # Animation
    CLOUD_ANIMATION_FPS = 20
    CLOUD_ANIMATION_INTERVAL_MS = 50

    # Default values
    DEFAULT_CLOUD_OPACITY = 0.33
    DEFAULT_NUCLEUS_TO_SHELL_RATIO = 1.0
    DEFAULT_ZOOM = 1.0


class ColorConstants:
    """Color-related constants"""
    # Block colors (RGB)
    BLOCK_S_COLOR = (255, 80, 100)  # Neon red
    BLOCK_P_COLOR = (80, 150, 255)  # Electric blue
    BLOCK_D_COLOR = (255, 200, 80)  # Golden
    BLOCK_F_COLOR = (120, 255, 150)  # Neon green

    # Default colors
    DEFAULT_GRAY = (150, 150, 150)
    DEFAULT_BLUE_GRAY = (100, 100, 150)
    UV_IR_GRAY_BLUE = (120, 120, 150)

    # Electron cloud color
    ELECTRON_CLOUD_COLOR = (100, 180, 255)  # Light blue


class UIConstants:
    """UI-related constants for fonts, spacing, etc."""
    # Font names
    FONT_FAMILY = "Arial"

    # Font sizes
    LABEL_FONT_SIZE = 9
    VALUE_FONT_SIZE = 8
    SYMBOL_FONT_SIZE = 12
    ATOMIC_NUMBER_FONT_SIZE = 8

    # Spacing and margins
    LABEL_VERTICAL_SPACING = 15  # Pixels between min/max labels and reference lines
    LABEL_HORIZONTAL_MARGIN = 5  # Pixels margin from edge
    LABEL_HEIGHT = 20  # Height of label text area


# ============================================================================
# ELEMENT NAME MAPPING
# ============================================================================

ELEMENT_NAMES = {
    'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium', 'B': 'Boron',
    'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen', 'F': 'Fluorine', 'Ne': 'Neon',
    'Na': 'Sodium', 'Mg': 'Magnesium', 'Al': 'Aluminum', 'Si': 'Silicon', 'P': 'Phosphorus',
    'S': 'Sulfur', 'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
    'Sc': 'Scandium', 'Ti': 'Titanium', 'V': 'Vanadium', 'Cr': 'Chromium', 'Mn': 'Manganese',
    'Fe': 'Iron', 'Co': 'Cobalt', 'Ni': 'Nickel', 'Cu': 'Copper', 'Zn': 'Zinc',
    'Ga': 'Gallium', 'Ge': 'Germanium', 'As': 'Arsenic', 'Se': 'Selenium', 'Br': 'Bromine',
    'Kr': 'Krypton', 'Rb': 'Rubidium', 'Sr': 'Strontium', 'Y': 'Yttrium', 'Zr': 'Zirconium',
    'Nb': 'Niobium', 'Mo': 'Molybdenum', 'Tc': 'Technetium', 'Ru': 'Ruthenium', 'Rh': 'Rhodium',
    'Pd': 'Palladium', 'Ag': 'Silver', 'Cd': 'Cadmium', 'In': 'Indium', 'Sn': 'Tin',
    'Sb': 'Antimony', 'Te': 'Tellurium', 'I': 'Iodine', 'Xe': 'Xenon', 'Cs': 'Cesium',
    'Ba': 'Barium', 'La': 'Lanthanum', 'Ce': 'Cerium', 'Pr': 'Praseodymium', 'Nd': 'Neodymium',
    'Pm': 'Promethium', 'Sm': 'Samarium', 'Eu': 'Europium', 'Gd': 'Gadolinium', 'Tb': 'Terbium',
    'Dy': 'Dysprosium', 'Ho': 'Holmium', 'Er': 'Erbium', 'Tm': 'Thulium', 'Yb': 'Ytterbium',
    'Lu': 'Lutetium', 'Hf': 'Hafnium', 'Ta': 'Tantalum', 'W': 'Tungsten', 'Re': 'Rhenium',
    'Os': 'Osmium', 'Ir': 'Iridium', 'Pt': 'Platinum', 'Au': 'Gold', 'Hg': 'Mercury',
    'Tl': 'Thallium', 'Pb': 'Lead', 'Bi': 'Bismuth', 'Po': 'Polonium', 'At': 'Astatine',
    'Rn': 'Radon', 'Fr': 'Francium', 'Ra': 'Radium', 'Ac': 'Actinium', 'Th': 'Thorium',
    'Pa': 'Protactinium', 'U': 'Uranium', 'Np': 'Neptunium', 'Pu': 'Plutonium', 'Am': 'Americium',
    'Cm': 'Curium', 'Bk': 'Berkelium', 'Cf': 'Californium', 'Es': 'Einsteinium', 'Fm': 'Fermium',
    'Md': 'Mendelevium', 'No': 'Nobelium', 'Lr': 'Lawrencium', 'Rf': 'Rutherfordium', 'Db': 'Dubnium',
    'Sg': 'Seaborgium', 'Bh': 'Bohrium', 'Hs': 'Hassium', 'Mt': 'Meitnerium', 'Ds': 'Darmstadtium',
    'Rg': 'Roentgenium', 'Cn': 'Copernicium', 'Nh': 'Nihonium', 'Fl': 'Flerovium', 'Mc': 'Moscovium',
    'Lv': 'Livermorium', 'Ts': 'Tennessine', 'Og': 'Oganesson'
}
