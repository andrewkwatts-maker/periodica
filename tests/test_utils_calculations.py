"""
Tests for utility calculation modules.
"""
import math
import pytest


# ==================== Calculations Module Tests ====================

class TestCalculations:
    """Test color and spectrum calculation functions."""

    def test_ev_to_frequency(self):
        from periodica.utils.color_math import ev_to_frequency
        # 1 eV should give about 0.2418 PHz
        freq = ev_to_frequency(1.0)
        assert abs(freq - 0.2418) < 0.001

    def test_ev_to_wavelength(self):
        from periodica.utils.color_math import ev_to_wavelength
        # 2 eV photon should be about 620 nm (red light)
        wl = ev_to_wavelength(2.0)
        assert 610 < wl < 630

    def test_ev_to_wavelength_visible_range(self):
        from periodica.utils.color_math import ev_to_wavelength
        # Visible light: ~1.7 eV (red) to ~3.1 eV (violet)
        red = ev_to_wavelength(1.77)
        violet = ev_to_wavelength(3.1)
        assert red > 600  # red wavelength > 600 nm
        assert violet < 420  # violet wavelength < 420 nm

    def test_get_block_color(self):
        from periodica.utils.color_math import get_block_color
        color = get_block_color('s')
        assert isinstance(color, tuple)
        assert color[0] > 200  # red channel: s-block is reddish

    def test_get_block_color_unknown(self):
        from periodica.utils.color_math import get_block_color
        color = get_block_color('x')
        assert color[0] == 200  # red channel: default gray

    def test_calculate_emission_spectrum(self):
        from periodica.utils.color_math import calculate_emission_spectrum
        # Hydrogen: IE = 13.6 eV, should generate spectral lines
        lines = calculate_emission_spectrum(1, 13.6)
        assert isinstance(lines, list)
        assert len(lines) > 0
        # Each line should be (wavelength, intensity) tuple
        for wl, intensity in lines:
            assert isinstance(wl, (int, float))
            assert isinstance(intensity, (int, float))

    def test_emission_spectrum_caching(self):
        from periodica.utils.color_math import calculate_emission_spectrum
        # Same call should return cached result
        lines1 = calculate_emission_spectrum(1, 13.6, 10)
        lines2 = calculate_emission_spectrum(1, 13.6, 10)
        assert lines1 is lines2  # Same object from cache


# ==================== Position Calculator Tests ====================

class TestPositionCalculator:
    """Test geometric positioning calculations."""

    def test_position_calculator_import(self):
        from periodica.utils.position_calculator import PositionCalculator
        calc = PositionCalculator()
        assert calc is not None

    def test_position_calculator_has_methods(self):
        from periodica.utils.position_calculator import PositionCalculator
        calc = PositionCalculator()
        assert hasattr(calc, 'get_table_position')
        assert hasattr(calc, 'get_circular_position')
        assert hasattr(calc, 'get_spiral_position')
        assert hasattr(calc, 'get_serpentine_position')


# ==================== Crystalline Math Tests ====================

class TestCrystallineMath:
    """Test crystal structure calculations."""

    def test_import(self):
        from periodica.utils.crystalline_math import (
            CrystalStructure,
            LatticeParameters,
            MicrostructureRenderer,
            PACKING_FACTORS,
        )

    def test_crystal_structure_enum(self):
        from periodica.utils.crystalline_math import CrystalStructure
        assert hasattr(CrystalStructure, 'FCC')
        assert hasattr(CrystalStructure, 'BCC')

    def test_packing_factors(self):
        from periodica.utils.crystalline_math import PACKING_FACTORS
        assert isinstance(PACKING_FACTORS, dict)
        assert len(PACKING_FACTORS) > 0


# ==================== Logger Tests ====================

class TestLogger:
    """Test the centralized logger utility."""

    def test_get_logger(self):
        from periodica.utils.logger import get_logger
        log = get_logger('test')
        assert log is not None
        assert log.name == 'periodica.test'

    def test_logger_has_handlers(self):
        from periodica.utils.logger import get_logger
        log = get_logger('test.handlers')
        assert len(log.handlers) > 0

    def test_logger_same_instance(self):
        from periodica.utils.logger import get_logger
        log1 = get_logger('test.same')
        log2 = get_logger('test.same')
        assert log1 is log2
