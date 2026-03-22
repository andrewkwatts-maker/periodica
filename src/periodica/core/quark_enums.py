#!/usr/bin/env python3
"""
Enums for quark/particle visualization properties and encodings.
Centralizes all string-based property checking for the Quarks tab.
"""

from enum import Enum, auto


class QuarkLayoutMode(Enum):
    """Layout modes for particle visualization"""
    STANDARD_MODEL = "standard_model"  # Standard Model grid layout
    LINEAR = "linear"  # Linear arrangement by property
    CIRCULAR = "circular"  # Circular arrangement with categories
    ALTERNATIVE = "alternative"  # Alternative grouping (by interaction type)
    FORCE_NETWORK = "force_network"  # Force interaction network
    MASS_SPIRAL = "mass_spiral"  # Mass hierarchy spiral
    FERMION_BOSON = "fermion_boson"  # Fermion/Boson split
    CHARGE_MASS = "charge_mass"  # Charge-Mass grid

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.STANDARD_MODEL  # Default

    @classmethod
    def get_display_name(cls, mode):
        """Get display name for layout mode"""
        if isinstance(mode, str):
            mode = cls.from_string(mode)
        names = {
            cls.STANDARD_MODEL: "Standard Model",
            cls.LINEAR: "Linear Arrangement",
            cls.CIRCULAR: "Circular Layout",
            cls.ALTERNATIVE: "Alternative Grouping",
            cls.FORCE_NETWORK: "Force Network",
            cls.MASS_SPIRAL: "Mass Spiral",
            cls.FERMION_BOSON: "Fermion/Boson Split",
            cls.CHARGE_MASS: "Charge-Mass Grid"
        }
        return names.get(mode, "Unknown")


class ParticleType(Enum):
    """Types of fundamental particles"""
    QUARK = "quark"
    LEPTON = "lepton"
    GAUGE_BOSON = "gauge_boson"
    SCALAR_BOSON = "scalar_boson"
    ANTIPARTICLE = "antiparticle"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"

    @classmethod
    def from_classification(cls, classification_list):
        """Determine particle type from classification list"""
        if not classification_list:
            return cls.UNKNOWN

        classification_lower = [c.lower() for c in classification_list]

        if "quark" in classification_lower:
            return cls.QUARK
        elif "lepton" in classification_lower:
            return cls.LEPTON
        elif "gauge boson" in classification_lower or "force carrier" in classification_lower:
            return cls.GAUGE_BOSON
        elif "scalar boson" in classification_lower:
            return cls.SCALAR_BOSON
        elif any("anti" in c for c in classification_lower):
            return cls.ANTIPARTICLE
        elif "composite" in classification_lower or "hadron" in classification_lower:
            return cls.COMPOSITE
        return cls.UNKNOWN

    @classmethod
    def get_color(cls, particle_type):
        """Get default color for particle type (R, G, B)"""
        if isinstance(particle_type, str):
            particle_type = cls.from_string(particle_type)
        colors = {
            cls.QUARK: (230, 100, 100),  # Red-ish
            cls.LEPTON: (100, 180, 230),  # Blue
            cls.GAUGE_BOSON: (230, 180, 100),  # Orange/Gold
            cls.SCALAR_BOSON: (180, 100, 230),  # Purple
            cls.ANTIPARTICLE: (180, 180, 180),  # Gray
            cls.COMPOSITE: (100, 200, 150),  # Teal
            cls.UNKNOWN: (150, 150, 150)  # Gray
        }
        return colors.get(particle_type, (150, 150, 150))

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        for member in cls:
            if member.value == value:
                return member
        return cls.UNKNOWN


class QuarkProperty(Enum):
    """Particle properties that can be visualized"""
    MASS = "mass"
    MASS_LOG = "mass_log"  # Log scale for wide mass range
    CHARGE = "charge"
    SPIN = "spin"
    PARTICLE_TYPE = "particle_type"
    INTERACTION = "interaction"
    STABILITY = "stability"
    GENERATION = "generation"
    BARYON_NUMBER = "baryon_number"
    LEPTON_NUMBER = "lepton_number"
    ISOSPIN = "isospin"
    ISOSPIN_I3 = "isospin_i3"  # Third component of isospin
    PARITY = "parity"
    HALF_LIFE = "half_life"
    HALF_LIFE_LOG = "half_life_log"  # Log scale for lifetime
    DECAY_WIDTH = "decay_width"
    MAGNETIC_MOMENT = "magnetic_moment"
    NONE = "none"

    @classmethod
    def from_string(cls, value):
        """Convert string to enum"""
        if value is None:
            return cls.NONE
        for member in cls:
            if member.value == value:
                return member
        return cls.NONE

    @classmethod
    def get_display_name(cls, prop):
        """Get display name for property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        names = {
            cls.MASS: "Mass (MeV/c^2)",
            cls.MASS_LOG: "Mass (log scale)",
            cls.CHARGE: "Charge (e)",
            cls.SPIN: "Spin (hbar)",
            cls.PARTICLE_TYPE: "Particle Type",
            cls.INTERACTION: "Interaction Forces",
            cls.STABILITY: "Stability",
            cls.GENERATION: "Generation",
            cls.BARYON_NUMBER: "Baryon Number",
            cls.LEPTON_NUMBER: "Lepton Number",
            cls.ISOSPIN: "Isospin (I)",
            cls.ISOSPIN_I3: "Isospin I3",
            cls.PARITY: "Parity (P)",
            cls.HALF_LIFE: "Half-Life (s)",
            cls.HALF_LIFE_LOG: "Half-Life (log)",
            cls.DECAY_WIDTH: "Decay Width (MeV)",
            cls.MAGNETIC_MOMENT: "Magnetic Moment",
            cls.NONE: "None"
        }
        return names.get(prop, "Unknown")

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
            cls.PARTICLE_TYPE: "Classification",
            cls.INTERACTION: "InteractionForces",
            cls.STABILITY: "Stability",
            cls.GENERATION: "generation",
            cls.BARYON_NUMBER: "BaryonNumber_B",
            cls.LEPTON_NUMBER: "LeptonNumber_L",
            cls.ISOSPIN: "Isospin_I",
            cls.ISOSPIN_I3: "Isospin_I3",
            cls.PARITY: "Parity_P",
            cls.HALF_LIFE: "HalfLife_s",
            cls.HALF_LIFE_LOG: "HalfLife_s",
            cls.DECAY_WIDTH: "Width_MeV",
            cls.MAGNETIC_MOMENT: "MagneticDipoleMoment_J_T",
            cls.NONE: None
        }
        return keys.get(prop, None)

    @classmethod
    def get_property_range(cls, prop):
        """Get default min/max range for a property"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        ranges = {
            cls.MASS: (0.0, 175000.0),  # Up quark to top quark (MeV/c^2)
            cls.MASS_LOG: (0.0, 12.0),  # Log10 scale
            cls.CHARGE: (-1.0, 1.0),  # In units of e
            cls.SPIN: (0.0, 1.0),  # In units of hbar
            cls.PARTICLE_TYPE: (0, 5),  # Categorical
            cls.INTERACTION: (0, 4),  # Number of forces
            cls.STABILITY: (0, 1),  # Binary stable/unstable
            cls.GENERATION: (0, 3),  # 0=bosons, 1-3=fermion generations
            cls.BARYON_NUMBER: (-1.0, 1.0),
            cls.LEPTON_NUMBER: (-1.0, 1.0),
            cls.ISOSPIN: (0.0, 1.0),
            cls.ISOSPIN_I3: (-1.0, 1.0),
            cls.PARITY: (-1, 1),
            cls.HALF_LIFE: (0.0, 1e-10),  # In seconds
            cls.HALF_LIFE_LOG: (-25, 0),  # Log10 scale
            cls.DECAY_WIDTH: (0.0, 2500.0),  # In MeV (top quark is ~1.42 GeV)
            cls.MAGNETIC_MOMENT: (-1e-25, 1e-25),  # In J/T
            cls.NONE: (0, 1)
        }
        return ranges.get(prop, (0, 100))

    @classmethod
    def is_log_scale(cls, prop):
        """Check if property should use log scale"""
        if isinstance(prop, str):
            prop = cls.from_string(prop)
        return prop in [cls.MASS_LOG, cls.HALF_LIFE_LOG]

    @classmethod
    def get_color_properties(cls):
        """Properties suitable for color encoding"""
        return [
            cls.PARTICLE_TYPE,
            cls.MASS,
            cls.MASS_LOG,
            cls.CHARGE,
            cls.SPIN,
            cls.INTERACTION,
            cls.STABILITY,
            cls.GENERATION,
            cls.ISOSPIN,
            cls.ISOSPIN_I3,
            cls.PARITY,
            cls.HALF_LIFE_LOG,
            cls.NONE
        ]

    @classmethod
    def get_size_properties(cls):
        """Properties suitable for size encoding"""
        return [
            cls.MASS,
            cls.MASS_LOG,
            cls.CHARGE,
            cls.SPIN,
            cls.BARYON_NUMBER,
            cls.LEPTON_NUMBER,
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
            cls.DECAY_WIDTH,
            cls.NONE
        ]

    @classmethod
    def get_glow_properties(cls):
        """Properties suitable for glow/emission effect encoding"""
        return [
            cls.STABILITY,
            cls.HALF_LIFE_LOG,
            cls.DECAY_WIDTH,
            cls.INTERACTION,
            cls.SPIN,
            cls.NONE
        ]

    @classmethod
    def get_border_properties(cls):
        """Properties suitable for border encoding"""
        return [
            cls.CHARGE,
            cls.PARTICLE_TYPE,
            cls.GENERATION,
            cls.PARITY,
            cls.BARYON_NUMBER,
            cls.LEPTON_NUMBER,
            cls.NONE
        ]


class QuarkGeneration(Enum):
    """Particle generations in the Standard Model"""
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FORCE_CARRIER = 0  # Bosons don't have generations
    UNKNOWN = -1

    @classmethod
    def from_particle_name(cls, name):
        """Determine generation from particle name"""
        name_lower = name.lower()

        # First generation
        first_gen = ['up', 'down', 'electron', 'electron neutrino']
        if any(p in name_lower for p in first_gen):
            return cls.FIRST

        # Second generation
        second_gen = ['charm', 'strange', 'muon']
        if any(p in name_lower for p in second_gen):
            return cls.SECOND

        # Third generation
        third_gen = ['top', 'bottom', 'tau']
        if any(p in name_lower for p in third_gen):
            return cls.THIRD

        # Bosons
        bosons = ['photon', 'gluon', 'w boson', 'z boson', 'higgs']
        if any(p in name_lower for p in bosons):
            return cls.FORCE_CARRIER

        return cls.UNKNOWN


class InteractionForce(Enum):
    """Fundamental forces"""
    STRONG = "Strong"
    ELECTROMAGNETIC = "Electromagnetic"
    WEAK = "Weak"
    GRAVITATIONAL = "Gravitational"

    @classmethod
    def get_color(cls, force):
        """Get color for interaction force"""
        if isinstance(force, str):
            force_map = {
                "Strong": cls.STRONG,
                "Electromagnetic": cls.ELECTROMAGNETIC,
                "Weak": cls.WEAK,
                "Gravitational": cls.GRAVITATIONAL
            }
            force = force_map.get(force, None)
            if force is None:
                return (150, 150, 150)

        colors = {
            cls.STRONG: (255, 100, 100),  # Red
            cls.ELECTROMAGNETIC: (100, 150, 255),  # Blue
            cls.WEAK: (255, 200, 100),  # Orange
            cls.GRAVITATIONAL: (150, 255, 150)  # Green
        }
        return colors.get(force, (150, 150, 150))
