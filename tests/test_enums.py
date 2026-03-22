"""
Tests for all enum classes in the Periodics project.
Covers pt_enums, quark_enums, subatomic_enums, molecule_enums, alloy_enums,
material_enums, amino_acid_enums, protein_enums, nucleic_acid_enums,
biomaterial_enums, cell_enums, and cell_component_enums.
"""

import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# pt_enums.py
# ---------------------------------------------------------------------------
from periodica.core.pt_enums import (
    PTPropertyName, PTEncodingKey, PTControlType, PTWavelengthMode,
    PTElementDataKey, PTPropertyType, PTEncodingType, PTLayoutMode,
    ENCODING_KEY_TO_TYPE, ENCODING_KEY_ENUM_TO_TYPE,
)


class TestPTPropertyName:
    def test_members_are_nonempty(self):
        assert len(PTPropertyName) > 0

    def test_values_are_strings(self):
        for member in PTPropertyName:
            assert isinstance(member.value, str)

    def test_from_string_known(self):
        assert PTPropertyName.from_string("atomic_number") is PTPropertyName.ATOMIC_NUMBER
        assert PTPropertyName.from_string("density") is PTPropertyName.DENSITY

    def test_from_string_none(self):
        assert PTPropertyName.from_string(None) is PTPropertyName.NONE

    def test_from_string_unknown(self):
        assert PTPropertyName.from_string("nonexistent") is PTPropertyName.NONE

    def test_is_wavelength_property(self):
        assert PTPropertyName.is_wavelength_property(PTPropertyName.WAVELENGTH) is True
        assert PTPropertyName.is_wavelength_property(PTPropertyName.SPECTRUM) is True
        assert PTPropertyName.is_wavelength_property(PTPropertyName.ATOMIC_NUMBER) is False

    def test_is_wavelength_property_from_string(self):
        assert PTPropertyName.is_wavelength_property("wavelength") is True
        assert PTPropertyName.is_wavelength_property("density") is False

    def test_is_categorical_property(self):
        assert PTPropertyName.is_categorical_property(PTPropertyName.BLOCK) is True
        assert PTPropertyName.is_categorical_property(PTPropertyName.DENSITY) is False

    def test_is_categorical_property_from_string(self):
        assert PTPropertyName.is_categorical_property("category") is True

    def test_get_color_properties_nonempty(self):
        props = PTPropertyName.get_color_properties()
        assert len(props) > 0
        assert all(isinstance(p, PTPropertyName) for p in props)

    def test_get_size_properties_nonempty(self):
        props = PTPropertyName.get_size_properties()
        assert len(props) > 0
        assert PTPropertyName.NONE in props

    def test_get_intensity_properties_nonempty(self):
        props = PTPropertyName.get_intensity_properties()
        assert len(props) > 0

    def test_get_display_name_known(self):
        assert PTPropertyName.get_display_name(PTPropertyName.ATOMIC_NUMBER) == "Atomic Number"
        assert PTPropertyName.get_display_name(PTPropertyName.NONE) == "None"

    def test_get_display_name_from_string(self):
        assert PTPropertyName.get_display_name("melting") == "Melting Point"

    def test_get_display_name_unknown_member(self):
        # ELECTRONS is not in the display_names dict
        assert PTPropertyName.get_display_name(PTPropertyName.ELECTRONS) == "Unknown"

    def test_from_display_name_known(self):
        assert PTPropertyName.from_display_name("Atomic Number") is PTPropertyName.ATOMIC_NUMBER
        assert PTPropertyName.from_display_name("None") is PTPropertyName.NONE

    def test_from_display_name_unknown(self):
        assert PTPropertyName.from_display_name("Bogus") is PTPropertyName.NONE


class TestPTEncodingKey:
    def test_members_are_nonempty(self):
        assert len(PTEncodingKey) > 0

    def test_values_are_strings(self):
        for member in PTEncodingKey:
            assert isinstance(member.value, str)

    def test_from_string_known(self):
        assert PTEncodingKey.from_string("fill_color") is PTEncodingKey.FILL_COLOR

    def test_from_string_none_input(self):
        assert PTEncodingKey.from_string(None) is None

    def test_from_string_unknown(self):
        assert PTEncodingKey.from_string("bogus") is None

    def test_is_color_encoding(self):
        assert PTEncodingKey.is_color_encoding(PTEncodingKey.FILL_COLOR) is True
        assert PTEncodingKey.is_color_encoding(PTEncodingKey.BORDER_SIZE) is False

    def test_is_color_encoding_from_string(self):
        assert PTEncodingKey.is_color_encoding("glow_color") is True

    def test_is_size_encoding(self):
        assert PTEncodingKey.is_size_encoding(PTEncodingKey.BORDER_SIZE) is True
        assert PTEncodingKey.is_size_encoding(PTEncodingKey.FILL_COLOR) is False

    def test_is_intensity_encoding(self):
        assert PTEncodingKey.is_intensity_encoding(PTEncodingKey.GLOW_INTENSITY) is True
        assert PTEncodingKey.is_intensity_encoding(PTEncodingKey.FILL_COLOR) is False


class TestPTControlType:
    def test_members(self):
        assert PTControlType.COLOR.value == "color"
        assert PTControlType.SIZE.value == "size"
        assert PTControlType.INTENSITY.value == "intensity"


class TestPTWavelengthMode:
    def test_from_string_spectrum(self):
        assert PTWavelengthMode.from_string("spectrum") is PTWavelengthMode.SPECTRUM
        assert PTWavelengthMode.from_string(None) is PTWavelengthMode.SPECTRUM

    def test_from_string_gradient(self):
        assert PTWavelengthMode.from_string("gradient") is PTWavelengthMode.GRADIENT

    def test_from_string_unknown_defaults_spectrum(self):
        assert PTWavelengthMode.from_string("bogus") is PTWavelengthMode.SPECTRUM


class TestPTElementDataKey:
    def test_members_are_nonempty(self):
        assert len(PTElementDataKey) > 0

    def test_values_are_strings(self):
        for member in PTElementDataKey:
            assert isinstance(member.value, str)

    def test_specific_keys(self):
        assert PTElementDataKey.ATOMIC_NUMBER.value == "z"
        assert PTElementDataKey.SYMBOL.value == "symbol"


class TestPTPropertyType:
    def test_from_string(self):
        assert PTPropertyType.from_string("color") is PTPropertyType.COLOR
        assert PTPropertyType.from_string("size") is PTPropertyType.SIZE
        assert PTPropertyType.from_string("intensity") is PTPropertyType.INTENSITY
        assert PTPropertyType.from_string("bogus") is PTPropertyType.COLOR


class TestPTEncodingType:
    def test_from_string_known(self):
        assert PTEncodingType.from_string("fill") is PTEncodingType.FILL
        assert PTEncodingType.from_string("border") is PTEncodingType.BORDER

    def test_from_string_unknown_defaults_fill(self):
        assert PTEncodingType.from_string("bogus") is PTEncodingType.FILL

    def test_get_wavelength_mode_attr(self):
        assert PTEncodingType.FILL.get_wavelength_mode_attr() == "fill_wavelength_mode"
        assert PTEncodingType.GLOW.get_wavelength_mode_attr() == "glow_wavelength_mode"

    def test_get_property_attr(self):
        assert PTEncodingType.FILL.get_property_attr() == "fill_color_property"
        assert PTEncodingType.BORDER.get_property_attr() == "border_color_property"
        assert PTEncodingType.RING.get_property_attr() == "ring_color_property"
        assert PTEncodingType.GLOW.get_property_attr() == "glow_color_property"
        assert PTEncodingType.SYMBOL_TEXT.get_property_attr() == "symbol_text_color_property"
        assert PTEncodingType.ATOMIC_NUMBER_TEXT.get_property_attr() == "atomic_number_text_color_property"


class TestPTLayoutMode:
    def test_from_string_known(self):
        assert PTLayoutMode.from_string("table") is PTLayoutMode.TABLE
        assert PTLayoutMode.from_string("spiral") is PTLayoutMode.SPIRAL

    def test_from_string_unknown_defaults_table(self):
        assert PTLayoutMode.from_string("bogus") is PTLayoutMode.TABLE


class TestPTModuleLevelMappings:
    def test_encoding_key_to_type(self):
        assert ENCODING_KEY_TO_TYPE["fill_color"] is PTEncodingType.FILL
        assert ENCODING_KEY_TO_TYPE["glow_color"] is PTEncodingType.GLOW

    def test_encoding_key_enum_to_type(self):
        assert ENCODING_KEY_ENUM_TO_TYPE[PTEncodingKey.FILL_COLOR] is PTEncodingType.FILL


# ---------------------------------------------------------------------------
# quark_enums.py
# ---------------------------------------------------------------------------
from periodica.core.quark_enums import (
    QuarkLayoutMode, ParticleType, QuarkProperty, QuarkGeneration,
    InteractionForce,
)


class TestQuarkLayoutMode:
    def test_members_nonempty(self):
        assert len(QuarkLayoutMode) > 0

    def test_from_string(self):
        assert QuarkLayoutMode.from_string("standard_model") is QuarkLayoutMode.STANDARD_MODEL
        assert QuarkLayoutMode.from_string("bogus") is QuarkLayoutMode.STANDARD_MODEL

    def test_get_display_name(self):
        assert QuarkLayoutMode.get_display_name(QuarkLayoutMode.STANDARD_MODEL) == "Standard Model"
        assert QuarkLayoutMode.get_display_name("linear") == "Linear Arrangement"
        assert QuarkLayoutMode.get_display_name(QuarkLayoutMode.FERMION_BOSON) == "Fermion/Boson Split"


class TestParticleType:
    def test_from_classification_quark(self):
        assert ParticleType.from_classification(["Quark"]) is ParticleType.QUARK

    def test_from_classification_lepton(self):
        assert ParticleType.from_classification(["Lepton"]) is ParticleType.LEPTON

    def test_from_classification_gauge_boson(self):
        assert ParticleType.from_classification(["Gauge Boson"]) is ParticleType.GAUGE_BOSON
        assert ParticleType.from_classification(["Force Carrier"]) is ParticleType.GAUGE_BOSON

    def test_from_classification_scalar_boson(self):
        assert ParticleType.from_classification(["Scalar Boson"]) is ParticleType.SCALAR_BOSON

    def test_from_classification_antiparticle(self):
        assert ParticleType.from_classification(["Antiquark"]) is ParticleType.ANTIPARTICLE

    def test_from_classification_composite(self):
        assert ParticleType.from_classification(["Composite"]) is ParticleType.COMPOSITE
        assert ParticleType.from_classification(["Hadron"]) is ParticleType.COMPOSITE

    def test_from_classification_empty(self):
        assert ParticleType.from_classification([]) is ParticleType.UNKNOWN
        assert ParticleType.from_classification(None) is ParticleType.UNKNOWN

    def test_from_string(self):
        assert ParticleType.from_string("quark") is ParticleType.QUARK
        assert ParticleType.from_string("bogus") is ParticleType.UNKNOWN

    def test_get_color(self):
        color = ParticleType.get_color(ParticleType.QUARK)
        assert isinstance(color, tuple) and len(color) == 3

    def test_get_color_from_string(self):
        color = ParticleType.get_color("lepton")
        assert color == (100, 180, 230)


class TestQuarkProperty:
    def test_from_string_known(self):
        assert QuarkProperty.from_string("mass") is QuarkProperty.MASS
        assert QuarkProperty.from_string(None) is QuarkProperty.NONE

    def test_from_string_unknown(self):
        assert QuarkProperty.from_string("bogus") is QuarkProperty.NONE

    def test_get_display_name(self):
        assert QuarkProperty.get_display_name(QuarkProperty.MASS) == "Mass (MeV/c^2)"
        assert QuarkProperty.get_display_name("charge") == "Charge (e)"

    def test_get_json_key(self):
        assert QuarkProperty.get_json_key(QuarkProperty.MASS) == "Mass_MeVc2"
        assert QuarkProperty.get_json_key(QuarkProperty.NONE) is None
        assert QuarkProperty.get_json_key("spin") == "Spin_hbar"

    def test_get_property_range(self):
        r = QuarkProperty.get_property_range(QuarkProperty.CHARGE)
        assert r == (-1.0, 1.0)
        r2 = QuarkProperty.get_property_range("mass")
        assert r2[0] < r2[1]

    def test_is_log_scale(self):
        assert QuarkProperty.is_log_scale(QuarkProperty.MASS_LOG) is True
        assert QuarkProperty.is_log_scale(QuarkProperty.MASS) is False
        assert QuarkProperty.is_log_scale("half_life_log") is True

    def test_get_color_properties(self):
        props = QuarkProperty.get_color_properties()
        assert QuarkProperty.PARTICLE_TYPE in props
        assert QuarkProperty.NONE in props

    def test_get_size_properties(self):
        assert len(QuarkProperty.get_size_properties()) > 0

    def test_get_intensity_properties(self):
        assert len(QuarkProperty.get_intensity_properties()) > 0

    def test_get_glow_properties(self):
        assert QuarkProperty.NONE in QuarkProperty.get_glow_properties()

    def test_get_border_properties(self):
        assert QuarkProperty.CHARGE in QuarkProperty.get_border_properties()


class TestQuarkGeneration:
    def test_values_are_ints(self):
        for member in QuarkGeneration:
            assert isinstance(member.value, int)

    def test_from_particle_name_first(self):
        assert QuarkGeneration.from_particle_name("Up quark") is QuarkGeneration.FIRST
        assert QuarkGeneration.from_particle_name("electron") is QuarkGeneration.FIRST

    def test_from_particle_name_second(self):
        assert QuarkGeneration.from_particle_name("charm") is QuarkGeneration.SECOND
        assert QuarkGeneration.from_particle_name("muon") is QuarkGeneration.SECOND

    def test_from_particle_name_third(self):
        assert QuarkGeneration.from_particle_name("top") is QuarkGeneration.THIRD
        assert QuarkGeneration.from_particle_name("tau") is QuarkGeneration.THIRD

    def test_from_particle_name_boson(self):
        assert QuarkGeneration.from_particle_name("photon") is QuarkGeneration.FORCE_CARRIER
        assert QuarkGeneration.from_particle_name("Higgs") is QuarkGeneration.FORCE_CARRIER

    def test_from_particle_name_unknown(self):
        assert QuarkGeneration.from_particle_name("graviton") is QuarkGeneration.UNKNOWN


class TestInteractionForce:
    def test_get_color_enum(self):
        color = InteractionForce.get_color(InteractionForce.STRONG)
        assert color == (255, 100, 100)

    def test_get_color_string(self):
        color = InteractionForce.get_color("Electromagnetic")
        assert color == (100, 150, 255)

    def test_get_color_unknown_string(self):
        color = InteractionForce.get_color("Bogus")
        assert color == (150, 150, 150)


# ---------------------------------------------------------------------------
# subatomic_enums.py
# ---------------------------------------------------------------------------
from periodica.core.subatomic_enums import (
    SubatomicLayoutMode, ParticleCategory, SubatomicProperty, QuarkType,
    get_particle_family_color,
)


class TestSubatomicLayoutMode:
    def test_from_string(self):
        assert SubatomicLayoutMode.from_string("baryon_meson") is SubatomicLayoutMode.BARYON_MESON
        assert SubatomicLayoutMode.from_string("bogus") is SubatomicLayoutMode.BARYON_MESON

    def test_get_display_name(self):
        assert SubatomicLayoutMode.get_display_name(SubatomicLayoutMode.EIGHTFOLD_WAY) == "Eightfold Way"
        assert SubatomicLayoutMode.get_display_name("mass_order") == "Mass Order"


class TestParticleCategory:
    def test_from_classification(self):
        assert ParticleCategory.from_classification(["Baryon"]) is ParticleCategory.BARYON
        assert ParticleCategory.from_classification(["Meson"]) is ParticleCategory.MESON
        assert ParticleCategory.from_classification([]) is ParticleCategory.BARYON

    def test_from_string(self):
        assert ParticleCategory.from_string("Baryon") is ParticleCategory.BARYON
        assert ParticleCategory.from_string("MESON") is ParticleCategory.MESON

    def test_get_color(self):
        color = ParticleCategory.get_color(ParticleCategory.BARYON)
        assert isinstance(color, tuple) and len(color) == 3


class TestSubatomicProperty:
    def test_from_string(self):
        assert SubatomicProperty.from_string("mass") is SubatomicProperty.MASS
        assert SubatomicProperty.from_string("bogus") is SubatomicProperty.NONE

    def test_get_display_name(self):
        assert SubatomicProperty.get_display_name(SubatomicProperty.STRANGENESS) == "Strangeness (S)"

    def test_get_json_key(self):
        assert SubatomicProperty.get_json_key(SubatomicProperty.MASS) == "Mass_MeVc2"
        assert SubatomicProperty.get_json_key(SubatomicProperty.NONE) is None

    def test_get_property_range(self):
        r = SubatomicProperty.get_property_range(SubatomicProperty.CHARGE)
        assert r == (-2.0, 2.0)

    def test_is_log_scale(self):
        assert SubatomicProperty.is_log_scale(SubatomicProperty.MASS_LOG) is True
        assert SubatomicProperty.is_log_scale(SubatomicProperty.MASS) is False
        assert SubatomicProperty.is_log_scale(SubatomicProperty.MEAN_LIFETIME_LOG) is True

    def test_property_list_methods(self):
        assert len(SubatomicProperty.get_color_properties()) > 0
        assert len(SubatomicProperty.get_size_properties()) > 0
        assert len(SubatomicProperty.get_intensity_properties()) > 0
        assert len(SubatomicProperty.get_glow_properties()) > 0
        assert len(SubatomicProperty.get_border_properties()) > 0


class TestQuarkType:
    def test_values_are_strings(self):
        for member in QuarkType:
            assert isinstance(member.value, str)

    def test_from_string_single_letter(self):
        assert QuarkType.from_string("u") is QuarkType.UP
        assert QuarkType.from_string("d") is QuarkType.DOWN
        assert QuarkType.from_string("s") is QuarkType.STRANGE

    def test_from_string_name(self):
        assert QuarkType.from_string("up") is QuarkType.UP
        assert QuarkType.from_string("charm") is QuarkType.CHARM

    def test_get_color(self):
        color = QuarkType.get_color(QuarkType.UP)
        assert color == (255, 100, 100)
        color2 = QuarkType.get_color("d")
        assert color2 == (100, 100, 255)

    def test_get_charge(self):
        assert abs(QuarkType.get_charge(QuarkType.UP) - 2 / 3) < 1e-9
        assert abs(QuarkType.get_charge(QuarkType.DOWN) - (-1 / 3)) < 1e-9
        assert abs(QuarkType.get_charge(QuarkType.ANTI_UP) - (-2 / 3)) < 1e-9


class TestGetParticleFamilyColor:
    def test_known_family(self):
        assert get_particle_family_color("proton") == (102, 126, 234)
        assert get_particle_family_color("Kaon+") == (240, 147, 251)

    def test_baryon_fallback(self):
        assert get_particle_family_color("some baryon") == (102, 126, 234)

    def test_unknown(self):
        assert get_particle_family_color("xyzzy") == (150, 150, 150)


# ---------------------------------------------------------------------------
# molecule_enums.py
# ---------------------------------------------------------------------------
from periodica.core.molecule_enums import (
    MoleculeLayoutMode, MoleculeProperty, BondType, MolecularGeometry,
    MoleculePolarity, MoleculeCategory, MoleculeState, get_element_color,
)


class TestMoleculeLayoutMode:
    def test_from_string(self):
        assert MoleculeLayoutMode.from_string("grid") is MoleculeLayoutMode.GRID
        assert MoleculeLayoutMode.from_string("bogus") is MoleculeLayoutMode.GRID

    def test_get_display_name(self):
        assert MoleculeLayoutMode.get_display_name(MoleculeLayoutMode.PHASE_DIAGRAM) == "Phase Diagram"


class TestMoleculeProperty:
    def test_from_string(self):
        assert MoleculeProperty.from_string("density") is MoleculeProperty.DENSITY
        assert MoleculeProperty.from_string(None) is MoleculeProperty.NONE

    def test_get_display_name(self):
        assert MoleculeProperty.get_display_name(MoleculeProperty.DIPOLE_MOMENT) == "Dipole Moment"

    def test_property_lists(self):
        assert len(MoleculeProperty.get_color_properties()) > 0
        assert MoleculeProperty.NONE in MoleculeProperty.get_color_properties()
        assert len(MoleculeProperty.get_size_properties()) > 0
        assert len(MoleculeProperty.get_numeric_properties()) > 0
        assert len(MoleculeProperty.get_glow_properties()) > 0
        assert len(MoleculeProperty.get_border_properties()) > 0


class TestBondType:
    def test_from_string(self):
        assert BondType.from_string("Double") is BondType.DOUBLE
        assert BondType.from_string("Bogus") is BondType.SINGLE

    def test_get_color(self):
        assert BondType.get_color(BondType.SINGLE) == "#4CAF50"
        assert BondType.get_color("Triple") == "#9C27B0"


class TestMolecularGeometry:
    def test_from_string(self):
        assert MolecularGeometry.from_string("Tetrahedral") is MolecularGeometry.TETRAHEDRAL
        assert MolecularGeometry.from_string("Bogus") is MolecularGeometry.LINEAR

    def test_get_color(self):
        assert isinstance(MolecularGeometry.get_color(MolecularGeometry.BENT), str)


class TestMoleculePolarity:
    def test_from_string(self):
        assert MoleculePolarity.from_string("Polar") is MoleculePolarity.POLAR
        assert MoleculePolarity.from_string("Bogus") is MoleculePolarity.NONPOLAR

    def test_get_color(self):
        assert MoleculePolarity.get_color(MoleculePolarity.IONIC) == "#FF9800"


class TestMoleculeCategory:
    def test_from_string(self):
        assert MoleculeCategory.from_string("Organic") is MoleculeCategory.ORGANIC
        assert MoleculeCategory.from_string("Bogus") is MoleculeCategory.INORGANIC


class TestMoleculeState:
    def test_from_string(self):
        assert MoleculeState.from_string("Solid") is MoleculeState.SOLID
        assert MoleculeState.from_string("Bogus") is MoleculeState.GAS

    def test_get_color(self):
        assert MoleculeState.get_color("Liquid") == "#2196F3"


class TestMoleculeGetElementColor:
    def test_known_element(self):
        assert get_element_color("H") == "#FFFFFF"

    def test_unknown_element(self):
        assert get_element_color("Xx") == "#FF1493"


# ---------------------------------------------------------------------------
# alloy_enums.py
# ---------------------------------------------------------------------------
from periodica.core.alloy_enums import (
    AlloyLayoutMode, AlloyCategory, CrystalStructure, AlloyProperty,
    ComponentRole, get_ipf_color,
)


class TestAlloyLayoutMode:
    def test_from_string(self):
        assert AlloyLayoutMode.from_string("category") is AlloyLayoutMode.CATEGORY
        assert AlloyLayoutMode.from_string("bogus") is AlloyLayoutMode.CATEGORY

    def test_get_display_name(self):
        assert AlloyLayoutMode.get_display_name(AlloyLayoutMode.LATTICE) == "By Crystal Structure"


class TestAlloyCategory:
    def test_from_string_case_insensitive(self):
        assert AlloyCategory.from_string("steel") is AlloyCategory.STEEL
        assert AlloyCategory.from_string("ALUMINUM") is AlloyCategory.ALUMINUM

    def test_from_string_none(self):
        assert AlloyCategory.from_string(None) is AlloyCategory.OTHER

    def test_get_color(self):
        assert isinstance(AlloyCategory.get_color(AlloyCategory.BRONZE), str)
        assert AlloyCategory.get_color("Steel") == "#607D8B"


class TestCrystalStructure:
    def test_from_string_case_insensitive(self):
        assert CrystalStructure.from_string("fcc") is CrystalStructure.FCC
        assert CrystalStructure.from_string("BCC") is CrystalStructure.BCC

    def test_from_string_none(self):
        assert CrystalStructure.from_string(None) is CrystalStructure.UNKNOWN

    def test_get_color(self):
        assert isinstance(CrystalStructure.get_color(CrystalStructure.HCP), str)

    def test_get_description(self):
        desc = CrystalStructure.get_description(CrystalStructure.FCC)
        assert "Face-Centered Cubic" in desc
        desc2 = CrystalStructure.get_description("bcc")
        assert "Body-Centered Cubic" in desc2


class TestAlloyProperty:
    def test_from_string(self):
        assert AlloyProperty.from_string("density") is AlloyProperty.DENSITY
        assert AlloyProperty.from_string(None) is AlloyProperty.NONE
        assert AlloyProperty.from_string("bogus") is AlloyProperty.NONE

    def test_get_display_name(self):
        assert AlloyProperty.get_display_name(AlloyProperty.TENSILE_STRENGTH) == "Tensile Strength"

    def test_get_unit(self):
        assert AlloyProperty.get_unit(AlloyProperty.DENSITY) == "g/cm3"
        assert AlloyProperty.get_unit(AlloyProperty.MELTING_POINT) == "K"
        assert AlloyProperty.get_unit("yield_strength") == "MPa"

    def test_property_lists(self):
        assert len(AlloyProperty.get_scatter_x_properties()) > 0
        assert len(AlloyProperty.get_scatter_y_properties()) > 0
        assert len(AlloyProperty.get_color_properties()) > 0
        assert len(AlloyProperty.get_size_properties()) > 0
        assert len(AlloyProperty.get_intensity_properties()) > 0


class TestComponentRole:
    def test_from_string(self):
        assert ComponentRole.from_string("Base") is ComponentRole.BASE
        assert ComponentRole.from_string(None) is ComponentRole.OTHER

    def test_from_string_austenite(self):
        assert ComponentRole.from_string("Austenite former") is ComponentRole.STABILIZER

    def test_get_color(self):
        assert isinstance(ComponentRole.get_color(ComponentRole.STRENGTHENING), str)


class TestGetIpfColor:
    def test_returns_tuple(self):
        color = get_ipf_color((180, 45, 45))
        assert isinstance(color, tuple) and len(color) == 3
        assert all(0 <= c <= 255 for c in color)


# ---------------------------------------------------------------------------
# material_enums.py
# ---------------------------------------------------------------------------
from periodica.core.material_enums import (
    MaterialLayoutMode, MaterialCategory, MaterialProperty, get_material_color,
)


class TestMaterialLayoutMode:
    def test_from_string(self):
        assert MaterialLayoutMode.from_string("category") is MaterialLayoutMode.CATEGORY
        assert MaterialLayoutMode.from_string("bogus") is MaterialLayoutMode.CATEGORY

    def test_get_display_name(self):
        assert MaterialLayoutMode.get_display_name(MaterialLayoutMode.THERMAL_MAP) == "Thermal Properties"


class TestMaterialCategory:
    def test_from_string_exact(self):
        assert MaterialCategory.from_string("Structural Steel") is MaterialCategory.STRUCTURAL_STEEL

    def test_from_string_fuzzy(self):
        assert MaterialCategory.from_string("some steel alloy") is MaterialCategory.STRUCTURAL_STEEL
        assert MaterialCategory.from_string("stainless steel 304") is MaterialCategory.STAINLESS_STEEL
        assert MaterialCategory.from_string("aluminum 6061") is MaterialCategory.ALUMINUM_ALLOY
        assert MaterialCategory.from_string("aluminium sheet") is MaterialCategory.ALUMINUM_ALLOY
        assert MaterialCategory.from_string("brass fitting") is MaterialCategory.COPPER_ALLOY
        assert MaterialCategory.from_string("Inconel 718") is MaterialCategory.NICKEL_SUPERALLOY
        assert MaterialCategory.from_string("CFRP panel") is MaterialCategory.COMPOSITE

    def test_from_string_none(self):
        assert MaterialCategory.from_string(None) is MaterialCategory.OTHER

    def test_get_color(self):
        assert isinstance(MaterialCategory.get_color(MaterialCategory.POLYMER), str)


class TestMaterialProperty:
    def test_from_string(self):
        assert MaterialProperty.from_string("density") is MaterialProperty.DENSITY
        assert MaterialProperty.from_string("bogus") is MaterialProperty.YOUNGS_MODULUS

    def test_get_display_name(self):
        name = MaterialProperty.get_display_name(MaterialProperty.YIELD_STRENGTH)
        assert "Yield Strength" in name

    def test_get_value_from_material(self):
        material_data = {
            "ElasticProperties": {"YoungsModulus_GPa": 200},
            "StrengthProperties": {"YieldStrength_MPa": 300},
            "PhysicalProperties": {"Density_kg_m3": 7800},
        }
        val = MaterialProperty.get_value_from_material(MaterialProperty.YOUNGS_MODULUS, material_data)
        assert val == 200
        val2 = MaterialProperty.get_value_from_material("yield_strength", material_data)
        assert val2 == 300

    def test_get_value_from_material_missing_data(self):
        val = MaterialProperty.get_value_from_material(MaterialProperty.DENSITY, {})
        assert val == 0

    def test_specific_strength_calculation(self):
        material_data = {
            "StrengthProperties": {"YieldStrength_MPa": 500},
            "PhysicalProperties": {"Density_kg_m3": 2700},
            "ElasticProperties": {},
        }
        val = MaterialProperty.get_value_from_material(MaterialProperty.SPECIFIC_STRENGTH, material_data)
        expected = 500 * 1000 / 2700
        assert abs(val - expected) < 0.01


class TestGetMaterialColor:
    def test_delegates_to_category(self):
        assert get_material_color("Polymer") == MaterialCategory.get_color("Polymer")


# ---------------------------------------------------------------------------
# amino_acid_enums.py
# ---------------------------------------------------------------------------
from periodica.core.amino_acid_enums import (
    AminoAcidLayoutMode, AminoAcidProperty, AminoAcidCategory,
    AminoAcidPolarity, ChargeState, SecondaryStructure as AASecondaryStructure,
    get_amino_acid_property_metadata,
)


class TestAminoAcidLayoutMode:
    def test_from_string(self):
        assert AminoAcidLayoutMode.from_string("grid") is AminoAcidLayoutMode.GRID
        assert AminoAcidLayoutMode.from_string("bogus") is AminoAcidLayoutMode.GRID

    def test_get_display_name(self):
        assert AminoAcidLayoutMode.get_display_name(AminoAcidLayoutMode.HYDROPATHY) == "By Hydropathy"


class TestAminoAcidProperty:
    def test_from_string(self):
        assert AminoAcidProperty.from_string("molecular_mass") is AminoAcidProperty.MOLECULAR_MASS
        assert AminoAcidProperty.from_string(None) is AminoAcidProperty.NONE

    def test_get_display_name(self):
        assert AminoAcidProperty.get_display_name(AminoAcidProperty.HYDROPATHY_INDEX) == "Hydropathy Index"

    def test_property_lists(self):
        assert len(AminoAcidProperty.get_numeric_properties()) > 0
        assert len(AminoAcidProperty.get_color_properties()) > 0
        assert len(AminoAcidProperty.get_size_properties()) > 0


class TestAminoAcidCategory:
    def test_from_string_snake_case(self):
        assert AminoAcidCategory.from_string("nonpolar_aliphatic") is AminoAcidCategory.NONPOLAR_ALIPHATIC

    def test_from_string_alias(self):
        assert AminoAcidCategory.from_string("basic") is AminoAcidCategory.POLAR_POSITIVE
        assert AminoAcidCategory.from_string("acidic") is AminoAcidCategory.POLAR_NEGATIVE
        assert AminoAcidCategory.from_string("aromatic") is AminoAcidCategory.NONPOLAR_AROMATIC

    def test_get_display_name(self):
        assert "Positive" in AminoAcidCategory.get_display_name(AminoAcidCategory.POLAR_POSITIVE)

    def test_get_color(self):
        assert isinstance(AminoAcidCategory.get_color(AminoAcidCategory.SPECIAL), str)


class TestAminoAcidPolarity:
    def test_from_string(self):
        assert AminoAcidPolarity.from_string("polar") is AminoAcidPolarity.POLAR
        assert AminoAcidPolarity.from_string("Bogus") is AminoAcidPolarity.NONPOLAR

    def test_get_display_name(self):
        assert AminoAcidPolarity.get_display_name(AminoAcidPolarity.ACIDIC) == "Acidic"

    def test_get_color(self):
        assert isinstance(AminoAcidPolarity.get_color("basic"), str)


class TestChargeState:
    def test_from_string(self):
        assert ChargeState.from_string("positive") is ChargeState.POSITIVE
        assert ChargeState.from_string("bogus") is ChargeState.NEUTRAL

    def test_from_charge(self):
        assert ChargeState.from_charge(1.0) is ChargeState.POSITIVE
        assert ChargeState.from_charge(-1.0) is ChargeState.NEGATIVE
        assert ChargeState.from_charge(0.0) is ChargeState.NEUTRAL
        assert ChargeState.from_charge(0.3) is ChargeState.NEUTRAL

    def test_get_color(self):
        assert isinstance(ChargeState.get_color(ChargeState.ZWITTERION), str)


class TestAASecondaryStructure:
    def test_from_string(self):
        assert AASecondaryStructure.from_string("helix") is AASecondaryStructure.HELIX
        assert AASecondaryStructure.from_string("bogus") is AASecondaryStructure.COIL

    def test_get_color(self):
        assert AASecondaryStructure.get_color(AASecondaryStructure.SHEET) == "#448AFF"


class TestGetAminoAcidPropertyMetadata:
    def test_known_property(self):
        meta = get_amino_acid_property_metadata("molecular_mass")
        assert meta["unit"] == "Da"
        assert meta["min_value"] < meta["max_value"]

    def test_unknown_returns_none_default(self):
        meta = get_amino_acid_property_metadata("bogus")
        assert meta["display_name"] == "None"


# ---------------------------------------------------------------------------
# protein_enums.py
# ---------------------------------------------------------------------------
from periodica.core.protein_enums import (
    ProteinLayoutMode, SecondaryStructureType, ProteinFunction,
    CellularLocalization, FoldingState, BondType as ProteinBondType,
    calculate_protein_mass, calculate_isoelectric_point,
)


class TestProteinLayoutMode:
    def test_from_string(self):
        assert ProteinLayoutMode.from_string("grid") is ProteinLayoutMode.GRID
        assert ProteinLayoutMode.from_string("bogus") is ProteinLayoutMode.GRID

    def test_members_nonempty(self):
        assert len(ProteinLayoutMode) >= 5


class TestSecondaryStructureType:
    def test_from_string(self):
        assert SecondaryStructureType.from_string("alpha_helix") is SecondaryStructureType.ALPHA_HELIX
        assert SecondaryStructureType.from_string("bogus") is SecondaryStructureType.RANDOM_COIL

    def test_get_color(self):
        assert SecondaryStructureType.get_color(SecondaryStructureType.ALPHA_HELIX) == "#FF4081"
        assert SecondaryStructureType.get_color("beta_sheet") == "#448AFF"

    def test_get_phi_psi_ranges(self):
        ranges = SecondaryStructureType.get_phi_psi_ranges(SecondaryStructureType.ALPHA_HELIX)
        assert len(ranges) == 4
        assert ranges[0] < ranges[1]  # phi_min < phi_max

    def test_get_phi_psi_ranges_from_string(self):
        ranges = SecondaryStructureType.get_phi_psi_ranges("beta_sheet")
        assert len(ranges) == 4


class TestProteinFunction:
    def test_from_string(self):
        assert ProteinFunction.from_string("enzyme") is ProteinFunction.ENZYME
        assert ProteinFunction.from_string("bogus") is ProteinFunction.STRUCTURAL

    def test_get_color(self):
        assert isinstance(ProteinFunction.get_color(ProteinFunction.TRANSPORT), str)
        assert isinstance(ProteinFunction.get_color("signaling"), str)


class TestCellularLocalization:
    def test_from_string(self):
        assert CellularLocalization.from_string("nucleus") is CellularLocalization.NUCLEUS
        assert CellularLocalization.from_string("bogus") is CellularLocalization.CYTOPLASM

    def test_get_color(self):
        assert isinstance(CellularLocalization.get_color(CellularLocalization.MITOCHONDRIA), str)


class TestFoldingState:
    def test_from_string(self):
        assert FoldingState.from_string("native") is FoldingState.NATIVE
        assert FoldingState.from_string("misfolded") is FoldingState.MISFOLDED
        assert FoldingState.from_string("bogus") is FoldingState.NATIVE

    def test_members_nonempty(self):
        assert len(FoldingState) >= 4


class TestProteinBondType:
    def test_get_color_enum(self):
        assert ProteinBondType.get_color(ProteinBondType.DISULFIDE) == "#FFC107"

    def test_get_color_string(self):
        assert ProteinBondType.get_color("hydrogen") == "#03A9F4"

    def test_get_color_unknown_string(self):
        # When string doesn't match, bond remains a string, dict lookup returns default
        assert ProteinBondType.get_color("bogus") == "#9E9E9E"


class TestCalculateProteinMass:
    def test_single_residue(self):
        mass = calculate_protein_mass("A")
        # Ala residue mass + water for terminal groups
        assert mass > 0

    def test_dipeptide(self):
        mass = calculate_protein_mass("AG")
        assert mass > 0

    def test_unknown_residue(self):
        # Unknown defaults to 110.0
        mass = calculate_protein_mass("X")
        assert mass > 0


class TestCalculateIsoelectricPoint:
    def test_balanced(self):
        # Equal positive and negative
        pi = calculate_isoelectric_point("KD")
        assert pi == 7.0

    def test_basic(self):
        pi = calculate_isoelectric_point("KKK")
        assert pi > 7.0

    def test_acidic(self):
        pi = calculate_isoelectric_point("DDD")
        assert pi < 7.0


# ---------------------------------------------------------------------------
# nucleic_acid_enums.py
# ---------------------------------------------------------------------------
from periodica.core.nucleic_acid_enums import (
    NucleicAcidType, BaseType, SecondaryStructure as NASecondaryStructure,
    NucleicAcidFunction, Modification, NucleicAcidLayoutMode,
    translate_sequence, transcribe_dna, reverse_complement,
)


class TestNucleicAcidType:
    def test_from_string(self):
        assert NucleicAcidType.from_string("dna") is NucleicAcidType.DNA
        assert NucleicAcidType.from_string("RNA") is NucleicAcidType.RNA
        assert NucleicAcidType.from_string("bogus") is NucleicAcidType.DNA

    def test_get_color(self):
        assert NucleicAcidType.get_color(NucleicAcidType.DNA) == "#2196F3"
        assert isinstance(NucleicAcidType.get_color("mrna"), str)


class TestBaseType:
    def test_from_string(self):
        assert BaseType.from_string("A") is BaseType.ADENINE
        assert BaseType.from_string("t") is BaseType.THYMINE
        assert BaseType.from_string("X") is None

    def test_get_color(self):
        assert isinstance(BaseType.get_color(BaseType.ADENINE), str)

    def test_get_complement_dna(self):
        assert BaseType.get_complement(BaseType.ADENINE) is BaseType.THYMINE
        assert BaseType.get_complement(BaseType.GUANINE) is BaseType.CYTOSINE

    def test_get_complement_rna(self):
        assert BaseType.get_complement(BaseType.ADENINE, is_rna=True) is BaseType.URACIL

    def test_get_complement_from_string(self):
        assert BaseType.get_complement("A") is BaseType.THYMINE


class TestNASecondaryStructure:
    def test_from_string(self):
        assert NASecondaryStructure.from_string("double_helix") is NASecondaryStructure.DOUBLE_HELIX
        assert NASecondaryStructure.from_string("bogus") is NASecondaryStructure.SINGLE_STRAND

    def test_get_color(self):
        assert isinstance(NASecondaryStructure.get_color(NASecondaryStructure.HAIRPIN), str)


class TestNucleicAcidFunction:
    def test_from_string(self):
        assert NucleicAcidFunction.from_string("catalytic") is NucleicAcidFunction.CATALYTIC
        assert NucleicAcidFunction.from_string("bogus") is NucleicAcidFunction.GENETIC_STORAGE

    def test_get_color(self):
        assert isinstance(NucleicAcidFunction.get_color(NucleicAcidFunction.SPLICING), str)


class TestModification:
    def test_from_string(self):
        assert Modification.from_string("methylation") is Modification.METHYLATION
        assert Modification.from_string("m6A") is Modification.N6_METHYLADENOSINE
        assert Modification.from_string("bogus") is None


class TestNucleicAcidLayoutMode:
    def test_from_string(self):
        assert NucleicAcidLayoutMode.from_string("grid") is NucleicAcidLayoutMode.GRID
        assert NucleicAcidLayoutMode.from_string("bogus") is NucleicAcidLayoutMode.GRID


class TestTranslateSequence:
    def test_simple_codon(self):
        # AUG = M (start/methionine)
        assert translate_sequence("AUG") == "M"

    def test_stop_codon(self):
        # AUG UGA -> M then stop
        assert translate_sequence("AUGUGA") == "M"

    def test_dna_input(self):
        # DNA with T should be converted to U internally
        assert translate_sequence("ATGGCC") == "MA"


class TestTranscribeDna:
    def test_basic(self):
        assert transcribe_dna("ATGC") == "AUGC"


class TestReverseComplement:
    def test_dna(self):
        assert reverse_complement("ATGC") == "GCAT"

    def test_rna(self):
        assert reverse_complement("AUGC", is_rna=True) == "GCAU"


# ---------------------------------------------------------------------------
# biomaterial_enums.py
# ---------------------------------------------------------------------------
from periodica.core.biomaterial_enums import (
    BiomaterialType, ECMComponent, MechanicalProperty, VascularizationLevel,
    BiomaterialLayoutMode,
)


class TestBiomaterialType:
    def test_members_nonempty(self):
        assert len(BiomaterialType) > 10

    def test_from_string(self):
        assert BiomaterialType.from_string("bone_cortical") is BiomaterialType.BONE_CORTICAL
        assert BiomaterialType.from_string("BOGUS") is BiomaterialType.OTHER

    def test_get_color(self):
        color = BiomaterialType.get_color(BiomaterialType.MUSCLE_CARDIAC)
        assert isinstance(color, str) and color.startswith("#")


class TestECMComponent:
    def test_from_string(self):
        assert ECMComponent.from_string("elastin") is ECMComponent.ELASTIN
        assert ECMComponent.from_string("BOGUS") is ECMComponent.COLLAGEN_I

    def test_get_color(self):
        assert isinstance(ECMComponent.get_color(ECMComponent.WATER), str)

    def test_get_modulus(self):
        assert ECMComponent.get_modulus(ECMComponent.HYDROXYAPATITE) == 117000
        assert ECMComponent.get_modulus(ECMComponent.WATER) == 0
        assert ECMComponent.get_modulus("elastin") == 0.6


class TestMechanicalProperty:
    def test_from_string(self):
        assert MechanicalProperty.from_string("stiff") is MechanicalProperty.STIFF
        assert MechanicalProperty.from_string("BOGUS") is MechanicalProperty.COMPLIANT


class TestVascularizationLevel:
    def test_from_string(self):
        assert VascularizationLevel.from_string("high") is VascularizationLevel.HIGH
        assert VascularizationLevel.from_string("BOGUS") is VascularizationLevel.MODERATE


class TestBiomaterialLayoutMode:
    def test_from_string(self):
        assert BiomaterialLayoutMode.from_string("grid") is BiomaterialLayoutMode.GRID
        assert BiomaterialLayoutMode.from_string("bogus") is BiomaterialLayoutMode.GRID


# ---------------------------------------------------------------------------
# cell_enums.py
# ---------------------------------------------------------------------------
from periodica.core.cell_enums import (
    CellType, CellCyclePhase, MetabolicState, Organism, TissueType,
    CellLayoutMode,
)


class TestCellType:
    def test_members_nonempty(self):
        assert len(CellType) > 20

    def test_from_string(self):
        assert CellType.from_string("neuron") is CellType.NEURON
        assert CellType.from_string("BOGUS") is CellType.OTHER

    def test_get_color(self):
        color = CellType.get_color(CellType.ERYTHROCYTE)
        assert isinstance(color, str) and color.startswith("#")

    def test_get_color_from_string(self):
        color = CellType.get_color("hepatocyte")
        assert isinstance(color, str)


class TestCellCyclePhase:
    def test_from_string(self):
        assert CellCyclePhase.from_string("g1") is CellCyclePhase.G1
        assert CellCyclePhase.from_string("s") is CellCyclePhase.S
        assert CellCyclePhase.from_string("BOGUS") is CellCyclePhase.G1


class TestMetabolicState:
    def test_from_string(self):
        assert MetabolicState.from_string("warburg") is MetabolicState.WARBURG
        assert MetabolicState.from_string("BOGUS") is MetabolicState.AEROBIC

    def test_get_color(self):
        assert MetabolicState.get_color(MetabolicState.DORMANT) == "#9E9E9E"
        assert isinstance(MetabolicState.get_color("stressed"), str)


class TestOrganism:
    def test_from_string(self):
        assert Organism.from_string("homo_sapiens") is Organism.HOMO_SAPIENS
        assert Organism.from_string("Homo sapiens") is Organism.HOMO_SAPIENS
        assert Organism.from_string("bogus") is Organism.OTHER

    def test_get_display_name(self):
        assert Organism.get_display_name(Organism.HOMO_SAPIENS) == "Homo sapiens"
        assert Organism.get_display_name(Organism.ESCHERICHIA_COLI) == "E. coli"
        assert Organism.get_display_name(Organism.OTHER) == "Unknown"


class TestTissueType:
    def test_from_string(self):
        assert TissueType.from_string("blood") is TissueType.BLOOD
        assert TissueType.from_string("BOGUS") is TissueType.CONNECTIVE

    def test_get_color(self):
        assert isinstance(TissueType.get_color(TissueType.BONE), str)


class TestCellLayoutMode:
    def test_from_string(self):
        assert CellLayoutMode.from_string("grid") is CellLayoutMode.GRID
        assert CellLayoutMode.from_string("bogus") is CellLayoutMode.GRID


# ---------------------------------------------------------------------------
# cell_component_enums.py
# ---------------------------------------------------------------------------
from periodica.core.cell_component_enums import (
    OrganelleType, MembraneType, CellularCompartment, ComponentFunction,
    CellComponentLayoutMode,
)


class TestOrganelleType:
    def test_members_nonempty(self):
        assert len(OrganelleType) >= 10

    def test_from_string(self):
        assert OrganelleType.from_string("nucleus") is OrganelleType.NUCLEUS
        assert OrganelleType.from_string("BOGUS") is OrganelleType.RIBOSOME

    def test_get_color(self):
        color = OrganelleType.get_color(OrganelleType.MITOCHONDRION)
        assert color == "#FF9800"
        assert isinstance(OrganelleType.get_color("lysosome"), str)


class TestMembraneType:
    def test_from_string(self):
        assert MembraneType.from_string("plasma") is MembraneType.PLASMA
        assert MembraneType.from_string("BOGUS") is MembraneType.PLASMA

    def test_members_nonempty(self):
        assert len(MembraneType) >= 5


class TestCellularCompartment:
    def test_from_string(self):
        assert CellularCompartment.from_string("nucleus") is CellularCompartment.NUCLEUS
        assert CellularCompartment.from_string("BOGUS") is CellularCompartment.CYTOPLASM

    def test_get_color(self):
        assert isinstance(CellularCompartment.get_color(CellularCompartment.MITOCHONDRIA), str)
        assert isinstance(CellularCompartment.get_color("extracellular"), str)


class TestComponentFunction:
    def test_from_string(self):
        assert ComponentFunction.from_string("energy_production") is ComponentFunction.ENERGY_PRODUCTION
        assert ComponentFunction.from_string("BOGUS") is ComponentFunction.STRUCTURAL

    def test_get_color(self):
        assert ComponentFunction.get_color(ComponentFunction.PHOTOSYNTHESIS) == "#8BC34A"
        assert isinstance(ComponentFunction.get_color("signaling"), str)


class TestCellComponentLayoutMode:
    def test_from_string(self):
        assert CellComponentLayoutMode.from_string("grid") is CellComponentLayoutMode.GRID
        assert CellComponentLayoutMode.from_string("function") is CellComponentLayoutMode.FUNCTION
        assert CellComponentLayoutMode.from_string("bogus") is CellComponentLayoutMode.GRID
