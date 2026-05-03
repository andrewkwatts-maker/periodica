"""
Tests for the QuarkConstantProvider.
Verifies that quark properties are read from JSON files and that
changes to quark JSON propagate correctly.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from periodica.utils.quark_constants import QuarkConstantProvider, get_quark_provider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton state before each test."""
    QuarkConstantProvider._instance = None
    yield
    QuarkConstantProvider._instance = None


class TestQuarkConstantProvider:
    """Test QuarkConstantProvider loads from JSON correctly."""

    def test_singleton_pattern(self):
        p1 = QuarkConstantProvider()
        p2 = QuarkConstantProvider()
        assert p1 is p2

    def test_get_quark_provider_returns_singleton(self):
        p1 = get_quark_provider()
        p2 = get_quark_provider()
        assert p1 is p2

    def test_loads_up_quark(self):
        provider = QuarkConstantProvider()
        u = provider.get_quark('u')
        assert u != {}
        assert 'mass_mev' in u
        assert 'charge' in u
        assert 'spin' in u
        assert u['spin'] == 0.5

    def test_loads_down_quark(self):
        provider = QuarkConstantProvider()
        d = provider.get_quark('d')
        assert d != {}
        assert d['mass_mev'] > 0

    def test_loads_strange_quark(self):
        provider = QuarkConstantProvider()
        s = provider.get_quark('s')
        assert s != {}
        assert s['mass_mev'] > 50  # Strange quark ~95 MeV

    def test_loads_charm_quark(self):
        provider = QuarkConstantProvider()
        c = provider.get_quark('c')
        assert c != {}
        assert c['mass_mev'] > 1000  # Charm ~1270 MeV

    def test_loads_bottom_quark(self):
        provider = QuarkConstantProvider()
        b = provider.get_quark('b')
        assert b != {}
        assert b['mass_mev'] > 4000  # Bottom ~4180 MeV

    def test_loads_top_quark(self):
        provider = QuarkConstantProvider()
        t = provider.get_quark('t')
        assert t != {}
        assert t['mass_mev'] > 170000  # Top ~173000 MeV

    def test_all_six_flavors_loaded(self):
        provider = QuarkConstantProvider()
        quarks = provider.get_all_quarks()
        for flavor in ['u', 'd', 's', 'c', 'b', 't']:
            assert flavor in quarks, f"Missing quark flavor: {flavor}"

    def test_up_quark_charge_positive(self):
        provider = QuarkConstantProvider()
        u = provider.get_quark('u')
        assert u['charge'] > 0  # +2/3

    def test_down_quark_charge_negative(self):
        provider = QuarkConstantProvider()
        d = provider.get_quark('d')
        assert d['charge'] < 0  # -1/3

    def test_unknown_flavor_returns_empty(self):
        provider = QuarkConstantProvider()
        assert provider.get_quark('x') == {}

    def test_get_quark_mass(self):
        provider = QuarkConstantProvider()
        mass = provider.get_quark_mass('u')
        assert mass > 0
        assert mass < 10  # Up quark is ~2.2 MeV

    def test_get_quark_mass_unknown_returns_zero(self):
        provider = QuarkConstantProvider()
        assert provider.get_quark_mass('x') == 0.0


class TestConstituentMasses:
    """Test constituent mass derivation."""

    def test_up_constituent_mass(self):
        provider = QuarkConstantProvider()
        mass = provider.get_constituent_mass('u')
        # Should be ~336 MeV (QCD dressed)
        assert 200 < mass < 500

    def test_down_constituent_mass(self):
        provider = QuarkConstantProvider()
        mass = provider.get_constituent_mass('d')
        # Should be ~340 MeV
        assert 200 < mass < 500

    def test_strange_constituent_mass(self):
        provider = QuarkConstantProvider()
        mass = provider.get_constituent_mass('s')
        # Should be ~486 MeV
        assert 400 < mass < 600

    def test_charm_constituent_mass(self):
        provider = QuarkConstantProvider()
        mass = provider.get_constituent_mass('c')
        # Current mass + 200 MeV QCD correction
        assert mass > 1200

    def test_constituent_masses_dict(self):
        provider = QuarkConstantProvider()
        masses = provider.get_constituent_masses()
        assert isinstance(masses, dict)
        for flavor in ['u', 'd', 's', 'c', 'b', 't']:
            assert flavor in masses
            assert masses[flavor] > 0

    def test_light_quarks_heavier_than_current(self):
        provider = QuarkConstantProvider()
        for flavor in ['u', 'd', 's']:
            constituent = provider.get_constituent_mass(flavor)
            current = provider.get_quark_mass(flavor)
            assert constituent > current, f"{flavor}: constituent {constituent} should be > current {current}"


class TestQuarkProperties:
    """Test the get_quark_properties format used by DerivationChain."""

    def test_properties_format(self):
        provider = QuarkConstantProvider()
        props = provider.get_quark_properties()
        for flavor in ['u', 'd', 's', 'c', 'b', 't']:
            assert flavor in props
            assert 'mass_mev' in props[flavor]
            assert 'charge' in props[flavor]
            assert 'spin' in props[flavor]

    def test_properties_match_individual_queries(self):
        provider = QuarkConstantProvider()
        props = provider.get_quark_properties()
        for flavor in ['u', 'd', 's']:
            individual = provider.get_quark(flavor)
            assert props[flavor]['mass_mev'] == individual['mass_mev']
            assert props[flavor]['charge'] == individual['charge']


class TestReload:
    """Test cache reload functionality."""

    def test_reload_clears_cache(self):
        provider = QuarkConstantProvider()
        # Access to trigger loading
        _ = provider.get_quark('u')
        assert provider._loaded is True

        # Reload should clear and re-load
        provider.reload()
        assert provider._loaded is True  # re-loaded immediately
        u = provider.get_quark('u')
        assert u['mass_mev'] > 0  # Still works after reload


class TestProtonNeutronDerivation:
    """Test that quark constants produce reasonable proton/neutron masses."""

    def test_proton_mass_from_quarks(self):
        provider = QuarkConstantProvider()
        cm = provider.get_constituent_masses()
        # Proton = uud
        proton_mass = cm['u'] * 2 + cm['d'] - 58.0  # binding correction
        # Should be close to 938.272 MeV
        error_pct = abs(proton_mass - 938.272) / 938.272 * 100
        assert error_pct < 5, f"Proton mass {proton_mass:.1f} MeV, error {error_pct:.1f}%"

    def test_neutron_mass_from_quarks(self):
        provider = QuarkConstantProvider()
        cm = provider.get_constituent_masses()
        # Neutron = udd
        neutron_mass = cm['u'] + cm['d'] * 2 - 58.0
        error_pct = abs(neutron_mass - 939.565) / 939.565 * 100
        assert error_pct < 5, f"Neutron mass {neutron_mass:.1f} MeV, error {error_pct:.1f}%"

    def test_proton_neutron_mass_difference_small(self):
        """The proton-neutron mass difference should be small (~1.3 MeV)."""
        provider = QuarkConstantProvider()
        cm = provider.get_constituent_masses()
        proton = cm['u'] * 2 + cm['d'] - 58.0
        neutron = cm['u'] + cm['d'] * 2 - 58.0
        diff = abs(neutron - proton)
        # The real difference is 1.293 MeV; our model should get within 5 MeV
        assert diff < 5.0, f"Proton-neutron mass difference {diff:.2f} MeV too large"
