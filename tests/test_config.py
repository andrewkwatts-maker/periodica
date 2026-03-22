"""
Tests for library constants.
"""
import pytest
from periodica.constants import PropertyType, DataProperty, ElementKey


# ==================== Constants Tests ====================

class TestPropertyType:
    """Test PropertyType constants."""

    def test_fill_constant(self):
        assert PropertyType.FILL == "fill"

    def test_border_constants(self):
        assert PropertyType.BORDER_COLOR == "border_color"
        assert PropertyType.BORDER_SIZE == "border_size"

    def test_glow_constants(self):
        assert PropertyType.GLOW_COLOR == "glow_color"
        assert PropertyType.GLOW_INTENSITY == "glow_intensity"

    def test_all_constants_are_strings(self):
        for attr in dir(PropertyType):
            if not attr.startswith('_'):
                assert isinstance(getattr(PropertyType, attr), str)


class TestDataProperty:
    """Test DataProperty constants."""

    def test_none_constant(self):
        assert DataProperty.NONE == "none"

    def test_key_properties_exist(self):
        assert hasattr(DataProperty, 'ATOMIC_NUMBER')
        assert hasattr(DataProperty, 'IONIZATION')
        assert hasattr(DataProperty, 'ELECTRONEGATIVITY')
        assert hasattr(DataProperty, 'BLOCK')
        assert hasattr(DataProperty, 'DENSITY')

    def test_all_constants_are_strings(self):
        for attr in dir(DataProperty):
            if not attr.startswith('_'):
                assert isinstance(getattr(DataProperty, attr), str)


class TestElementKey:
    """Test ElementKey constants."""

    def test_core_keys(self):
        assert ElementKey.SYMBOL == "symbol"
        assert ElementKey.NAME == "name"
        assert ElementKey.Z == "z"

    def test_all_constants_are_strings(self):
        for attr in dir(ElementKey):
            if not attr.startswith('_'):
                val = getattr(ElementKey, attr)
                if not callable(val):
                    assert isinstance(val, str)
