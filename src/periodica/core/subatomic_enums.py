"""
Enums for subatomic particle visualization properties and encodings.
Centralizes all string-based property checking for the subatomic particle tab.
"""

from enum import Enum, auto


class SubatomicLayoutMode(Enum):
    """Layout modes for subatomic particle visualization"""
    BARYON_MESON = "baryon_meson"       # Separate groups for baryons and mesons
    MASS_ORDER = "mass_order"           # Ordered by mass
    CHARGE_ORDER = "charge_order"       # Ordered by charge
    DECAY_CHAIN = "decay_chain"         # Show decay relationships
    QUARK_CONTENT = "quark_content"     # Group by quark content
    EIGHTFOLD_WAY = "eightfold_way"     # Strangeness-Isospin plot (I3 vs Y)
    LIFETIME_SPECTRUM = "lifetime_spectrum"  # Logarithmic lifetime timeline
    QUARK_TREE = "quark_tree"           # Hierarchical quark composition tree
    DISCOVERY_TIMELINE = "discovery_timeline"  # Chronological discovery arrangement

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.BARYON_MESON  # Default

    @classmethod
    def get_display_name(cls, mode):
        """Get UI display name for a layout mode"""
        if isinstance(mode, str):
            mode = cls.from_string(mode)

        display_names = {
            cls.BARYON_MESON: "Baryon/Meson Groups",
            cls.MASS_ORDER: "Mass Order",
            cls.CHARGE_ORDER: "Charge Order",
            cls.DECAY_CHAIN: "Decay Chains",
            cls.QUARK_CONTENT: "Quark Content",
            cls.EIGHTFOLD_WAY: "Eightfold Way",
            cls.LIFETIME_SPECTRUM: "Lifetime Spectrum",
            cls.QUARK_TREE: "Quark Tree",
            cls.DISCOVERY_TIMELINE: "Discovery Timeline"
        }
        return display_names.get(mode, "Unknown")


class ParticleCategory(Enum):
    """Categories of subatomic particles"""
    BARYON = "baryon"
    MESON = "meson"
    LEPTON = "lepton"
    BOSON = "boson"
    QUARK = "quark"
    NUCLEON = "nucleon"
    STRANGE = "strange"
    CHARM = "charm"
    BOTTOM = "bottom"

    @classmethod
    def from_classification(cls, classification_list):
        """Determine particle category from classification list"""
        if not classification_list:
            return cls.BARYON

        classification_lower = [c.lower() for c in classification_list]

        if "baryon" in classification_lower:
            return cls.BARYON
        elif "meson" in classification_lower:
            return cls.MESON
        elif "lepton" in classification_lower:
            return cls.LEPTON
        elif "boson" in classification_lower and "meson" not in classification_lower:
            return cls.BOSON
        elif "quark" in classification_lower:
            return cls.QUARK

        return cls.BARYON  # Default

    @classmethod
    def get_color(cls, category):
        """Get default color for particle category (RGB tuple)"""
        if isinstance(category, str):
            category = cls.from_string(category)

        colors = {
            cls.BARYON: (102, 126, 234),    # Blue-purple
            cls.MESON: (240, 147, 251),     # Pink
            cls.LEPTON: (79, 195, 247),     # Light blue
            cls.BOSON: (255, 183, 77),      # Orange
            cls.QUARK: (129, 199, 132),     # Green
            cls.NUCLEON: (100, 181, 246),   # Blue
            cls.STRANGE: (255, 138, 128),   # Coral
            cls.CHARM: (255, 213, 79),      # Gold
            cls.BOTTOM: (186, 104, 200),    # Purple
        }
        return colors.get(category, (150, 150, 150))

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.BARYON


class SubatomicProperty(Enum):
    """Properties of subatomic particles that can be visualized"""
    MASS = "mass"
    MASS_LOG = "mass_log"  # Log scale for wide mass range
    CHARGE = "charge"
    SPIN = "spin"
    HALF_LIFE = "half_life"
    HALF_LIFE_LOG = "half_life_log"  # Log scale for lifetime
    MEAN_LIFETIME = "mean_lifetime"
    MEAN_LIFETIME_LOG = "mean_lifetime_log"
    DECAY_WIDTH = "decay_width"
    STABILITY = "stability"
    BARYON_NUMBER = "baryon_number"
    LEPTON_NUMBER = "lepton_number"
    STRANGENESS = "strangeness"
    CHARM = "charm"
    BOTTOMNESS = "bottomness"
    ISOSPIN = "isospin"
    ISOSPIN_I3 = "isospin_i3"
    PARITY = "parity"
    C_PARITY = "c_parity"
    QUARK_COUNT = "quark_count"
    NONE = "none"

    @classmethod
    def get_display_name(cls, prop):
        """Get UI display name for a property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)

        display_names = {
            cls.MASS: "Mass (MeV/c^2)",
            cls.MASS_LOG: "Mass (log scale)",
            cls.CHARGE: "Electric Charge (e)",
            cls.SPIN: "Spin (hbar)",
            cls.HALF_LIFE: "Half-Life (s)",
            cls.HALF_LIFE_LOG: "Half-Life (log)",
            cls.MEAN_LIFETIME: "Mean Lifetime (s)",
            cls.MEAN_LIFETIME_LOG: "Mean Lifetime (log)",
            cls.DECAY_WIDTH: "Decay Width (MeV)",
            cls.STABILITY: "Stability",
            cls.BARYON_NUMBER: "Baryon Number (B)",
            cls.LEPTON_NUMBER: "Lepton Number (L)",
            cls.STRANGENESS: "Strangeness (S)",
            cls.CHARM: "Charm (C)",
            cls.BOTTOMNESS: "Bottomness (B')",
            cls.ISOSPIN: "Isospin (I)",
            cls.ISOSPIN_I3: "Isospin I3",
            cls.PARITY: "Parity (P)",
            cls.C_PARITY: "C-Parity",
            cls.QUARK_COUNT: "Quark Count",
            cls.NONE: "None"
        }
        return display_names.get(prop, "Unknown")

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.NONE

    @classmethod
    def get_json_key(cls, prop):
        """Get JSON data key for property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        keys = {
            cls.MASS: "Mass_MeVc2",
            cls.MASS_LOG: "Mass_MeVc2",
            cls.CHARGE: "Charge_e",
            cls.SPIN: "Spin_hbar",
            cls.HALF_LIFE: "HalfLife_s",
            cls.HALF_LIFE_LOG: "HalfLife_s",
            cls.MEAN_LIFETIME: "MeanLifetime_s",
            cls.MEAN_LIFETIME_LOG: "MeanLifetime_s",
            cls.DECAY_WIDTH: "Width_MeV",
            cls.STABILITY: "Stability",
            cls.BARYON_NUMBER: "BaryonNumber_B",
            cls.LEPTON_NUMBER: "LeptonNumber_L",
            cls.STRANGENESS: "Strangeness",
            cls.CHARM: "Charm",
            cls.BOTTOMNESS: "Bottomness",
            cls.ISOSPIN: "Isospin_I",
            cls.ISOSPIN_I3: "Isospin_I3",
            cls.PARITY: "Parity_P",
            cls.C_PARITY: "CParity",
            cls.QUARK_COUNT: "quark_count",  # Computed property
            cls.NONE: None
        }
        return keys.get(prop, None)

    @classmethod
    def get_property_range(cls, prop):
        """Get default min/max range for a property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        ranges = {
            cls.MASS: (0.0, 12000.0),  # Pion to Upsilon (MeV/c^2)
            cls.MASS_LOG: (2.0, 4.5),  # Log10 scale
            cls.CHARGE: (-2.0, 2.0),  # In units of e
            cls.SPIN: (0.0, 3.5),  # In units of hbar
            cls.HALF_LIFE: (0.0, 1e-8),  # In seconds
            cls.HALF_LIFE_LOG: (-25, -5),  # Log10 scale
            cls.MEAN_LIFETIME: (0.0, 1e-8),  # In seconds
            cls.MEAN_LIFETIME_LOG: (-25, -5),  # Log10 scale
            cls.DECAY_WIDTH: (0.0, 200.0),  # In MeV
            cls.STABILITY: (0, 1),  # Binary stable/unstable
            cls.BARYON_NUMBER: (-1, 1),
            cls.LEPTON_NUMBER: (-1, 1),
            cls.STRANGENESS: (-3, 0),
            cls.CHARM: (-2, 2),
            cls.BOTTOMNESS: (-1, 1),
            cls.ISOSPIN: (0.0, 1.5),
            cls.ISOSPIN_I3: (-1.5, 1.5),
            cls.PARITY: (-1, 1),
            cls.C_PARITY: (-1, 1),
            cls.QUARK_COUNT: (2, 3),  # Mesons have 2, baryons have 3
            cls.NONE: (0, 100)
        }
        return ranges.get(prop, (0, 100))

    @classmethod
    def is_log_scale(cls, prop):
        """Check if property should use log scale"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        return prop in [cls.MASS_LOG, cls.HALF_LIFE_LOG, cls.MEAN_LIFETIME_LOG]

    @classmethod
    def get_color_properties(cls):
        """Get list of properties suitable for color encoding"""
        return [
            cls.MASS,
            cls.MASS_LOG,
            cls.CHARGE,
            cls.SPIN,
            cls.HALF_LIFE_LOG,
            cls.MEAN_LIFETIME_LOG,
            cls.DECAY_WIDTH,
            cls.STRANGENESS,
            cls.CHARM,
            cls.BOTTOMNESS,
            cls.ISOSPIN,
            cls.ISOSPIN_I3,
            cls.STABILITY,
            cls.NONE
        ]

    @classmethod
    def get_size_properties(cls):
        """Get list of properties suitable for size encoding"""
        return [
            cls.MASS,
            cls.MASS_LOG,
            cls.CHARGE,
            cls.SPIN,
            cls.QUARK_COUNT,
            cls.ISOSPIN,
            cls.NONE
        ]

    @classmethod
    def get_intensity_properties(cls):
        """Properties suitable for intensity encoding"""
        return [
            cls.MASS,
            cls.MASS_LOG,
            cls.SPIN,
            cls.STABILITY,
            cls.HALF_LIFE_LOG,
            cls.MEAN_LIFETIME_LOG,
            cls.DECAY_WIDTH,
            cls.NONE
        ]

    @classmethod
    def get_glow_properties(cls):
        """Properties suitable for glow/emission effect encoding"""
        return [
            cls.STABILITY,
            cls.HALF_LIFE_LOG,
            cls.MEAN_LIFETIME_LOG,
            cls.DECAY_WIDTH,
            cls.SPIN,
            cls.STRANGENESS,
            cls.CHARM,
            cls.BOTTOMNESS,
            cls.NONE
        ]

    @classmethod
    def get_border_properties(cls):
        """Properties suitable for border encoding"""
        return [
            cls.CHARGE,
            cls.BARYON_NUMBER,
            cls.LEPTON_NUMBER,
            cls.PARITY,
            cls.C_PARITY,
            cls.STRANGENESS,
            cls.CHARM,
            cls.BOTTOMNESS,
            cls.NONE
        ]


class QuarkType(Enum):
    """Types of quarks"""
    UP = "u"
    DOWN = "d"
    STRANGE = "s"
    CHARM = "c"
    BOTTOM = "b"
    TOP = "t"
    ANTI_UP = "u-bar"
    ANTI_DOWN = "d-bar"
    ANTI_STRANGE = "s-bar"
    ANTI_CHARM = "c-bar"
    ANTI_BOTTOM = "b-bar"
    ANTI_TOP = "t-bar"

    @classmethod
    def get_color(cls, quark):
        """Get display color for quark type (RGB tuple)"""
        if isinstance(quark, str):
            quark = cls.from_string(quark)

        colors = {
            cls.UP: (255, 100, 100),        # Red
            cls.DOWN: (100, 100, 255),      # Blue
            cls.STRANGE: (100, 255, 100),   # Green
            cls.CHARM: (255, 200, 100),     # Orange
            cls.BOTTOM: (200, 100, 255),    # Purple
            cls.TOP: (255, 255, 100),       # Yellow
            # Antiquarks are lighter versions
            cls.ANTI_UP: (255, 180, 180),
            cls.ANTI_DOWN: (180, 180, 255),
            cls.ANTI_STRANGE: (180, 255, 180),
            cls.ANTI_CHARM: (255, 230, 180),
            cls.ANTI_BOTTOM: (230, 180, 255),
            cls.ANTI_TOP: (255, 255, 180),
        }
        return colors.get(quark, (150, 150, 150))

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        value_lower = value.lower().replace('bar', '-bar').replace('anti-', '').replace('anti', '')

        # Handle various formats
        mappings = {
            'u': cls.UP,
            'd': cls.DOWN,
            's': cls.STRANGE,
            'c': cls.CHARM,
            'b': cls.BOTTOM,
            't': cls.TOP,
            'u-bar': cls.ANTI_UP,
            'd-bar': cls.ANTI_DOWN,
            's-bar': cls.ANTI_STRANGE,
            'c-bar': cls.ANTI_CHARM,
            'b-bar': cls.ANTI_BOTTOM,
            't-bar': cls.ANTI_TOP,
            'up': cls.UP,
            'down': cls.DOWN,
            'strange': cls.STRANGE,
            'charm': cls.CHARM,
            'bottom': cls.BOTTOM,
            'top': cls.TOP,
        }

        return mappings.get(value_lower, cls.UP)

    @classmethod
    def get_charge(cls, quark):
        """Get electric charge for quark type"""
        if isinstance(quark, str):
            quark = cls.from_string(quark)

        charges = {
            cls.UP: 2/3,
            cls.CHARM: 2/3,
            cls.TOP: 2/3,
            cls.DOWN: -1/3,
            cls.STRANGE: -1/3,
            cls.BOTTOM: -1/3,
            cls.ANTI_UP: -2/3,
            cls.ANTI_CHARM: -2/3,
            cls.ANTI_TOP: -2/3,
            cls.ANTI_DOWN: 1/3,
            cls.ANTI_STRANGE: 1/3,
            cls.ANTI_BOTTOM: 1/3,
        }
        return charges.get(quark, 0)


# Color mappings for visualization
PARTICLE_COLORS = {
    'nucleon': (100, 181, 246),       # Blue
    'delta': (255, 138, 128),         # Coral
    'sigma': (129, 199, 132),         # Green
    'xi': (255, 213, 79),             # Gold
    'omega': (186, 104, 200),         # Purple
    'lambda': (79, 195, 247),         # Light blue
    'pion': (255, 183, 77),           # Orange
    'kaon': (240, 147, 251),          # Pink
    'eta': (176, 190, 197),           # Gray-blue
    'jpsi': (255, 235, 59),           # Yellow
    'upsilon': (156, 39, 176),        # Deep purple
    'proton': (102, 126, 234),        # Blue-purple
    'neutron': (144, 164, 174),       # Blue-gray
}


def get_particle_family_color(particle_name):
    """Get color for a particle based on its family/type"""
    name_lower = particle_name.lower()

    for family, color in PARTICLE_COLORS.items():
        if family in name_lower:
            return color

    # Default colors based on classification
    if 'baryon' in name_lower:
        return (102, 126, 234)
    elif 'meson' in name_lower:
        return (240, 147, 251)

    return (150, 150, 150)
