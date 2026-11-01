"""Tests for the canonical property / unit layer (periodica.properties).

These pin the behaviour that makes curated data sheets usable: nested
property groups flatten, alternative spellings resolve to one canonical
property, and units convert both ways without accumulating float noise.
"""
from __future__ import annotations

import math

import pytest

from periodica import Get, data_sheet, sample
from periodica.properties import (
    DataSheet,
    canonical_of,
    convert,
    coverage,
    flatten,
    from_si,
    list_properties,
    parse_key,
    property_spec,
    schema,
    sheet,
    si_sheet,
    slugify,
    to_si,
    validate_sheet,
)


# ─────────────────────────────────────────────────────────────────────────
# Schema integrity
# ─────────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_loads(self):
        s = schema()
        assert s["properties"]
        assert s["units"]

    def test_every_property_has_a_known_si_unit(self):
        s = schema()
        si_targets = {u["si"] for u in s["units"].values()}
        for name, spec in s["properties"].items():
            assert spec["si_unit"] in si_targets, f"{name}: unknown si_unit"

    def test_every_default_unit_is_a_known_token(self):
        s = schema()
        for name, spec in s["properties"].items():
            unit = spec.get("default_unit")
            if unit is None:
                continue
            assert unit in s["units"], f"{name}: unknown default_unit {unit!r}"

    def test_default_unit_matches_the_property_dimension(self):
        s = schema()
        for name, spec in s["properties"].items():
            unit = spec.get("default_unit")
            if unit is None:
                continue
            assert s["units"][unit]["si"] == spec["si_unit"], (
                f"{name}: default_unit {unit!r} is not a {spec['si_unit']} unit"
            )

    def test_no_alias_is_claimed_by_two_properties(self):
        seen: dict = {}
        for name, spec in schema()["properties"].items():
            for alias in list(spec["aliases"]) + [name]:
                slug = slugify(alias if isinstance(alias, str) else alias["name"])
                assert slug not in seen or seen[slug] == name, (
                    f"alias {slug!r} claimed by both {seen.get(slug)!r} and {name!r}"
                )
                seen[slug] = name

    def test_ranges_are_ordered(self):
        for name, spec in schema()["properties"].items():
            rng = spec.get("range")
            if rng:
                assert rng[0] < rng[1], f"{name}: range not ascending"


# ─────────────────────────────────────────────────────────────────────────
# Slugify + key parsing
# ─────────────────────────────────────────────────────────────────────────

class TestSlugify:
    @pytest.mark.parametrize("raw, expect", [
        ("UltimateTensileStrength", "ultimate_tensile_strength"),
        ("youngs_modulus", "youngs_modulus"),
        ("Poisson's Ratio", "poissons_ratio"),
        ("YoungsModulus", "youngs_modulus"),
        ("HDPE", "hdpe"),
        ("a", "a"),
        ("", ""),
    ])
    def test_slugify(self, raw, expect):
        assert slugify(raw) == expect


class TestParseKey:
    @pytest.mark.parametrize("key, canonical, unit", [
        ("Density_g_cm3", "density", "g_cm3"),
        ("Density_kgm3", "density", "kgm3"),
        ("density", "density", "g_cm3"),
        ("UltimateTensileStrength_MPa", "tensile_strength", "MPa"),
        ("TensileStrength_MPa", "tensile_strength", "MPa"),
        ("YoungsModulus_GPa", "youngs_modulus", "GPa"),
        ("youngs_modulus_MPa", "youngs_modulus", "MPa"),
        ("MeltingPoint_K", "melting_point", "K"),
        ("melting_point", "melting_point", "K"),
        ("PoissonsRatio", "poissons_ratio", "ratio"),
        ("Elongation_percent", "elongation", "percent"),
        ("ThermalConductivity_W_mK", "thermal_conductivity", "W_mK"),
        ("Mass_amu", "atomic_mass", "amu"),
        ("atomic_radius", "atomic_radius", "pm"),
        ("ShoreD", "hardness_shore_d", "ShoreD"),
        ("a_pm", "lattice_a", "pm"),
    ])
    def test_recognised(self, key, canonical, unit):
        parsed = parse_key(key)
        assert parsed.canonical == canonical
        assert parsed.unit == unit

    def test_unknown_key_is_reported_not_guessed(self):
        parsed = parse_key("NonExistentProperty_XYZ")
        assert parsed.canonical is None

    def test_empty_key(self):
        assert parse_key("").canonical is None

    def test_canonical_of_matches_parse_key(self):
        assert canonical_of("Density_kgm3") == "density"
        assert canonical_of("nope") is None

    def test_longest_unit_suffix_wins(self):
        # MPa_sqrt_m must not be read as MPa.
        assert parse_key("FractureToughness_KIC_MPa_sqrt_m").unit == "MPa_sqrt_m"


# ─────────────────────────────────────────────────────────────────────────
# Unit conversion
# ─────────────────────────────────────────────────────────────────────────

class TestUnits:
    def test_density_to_si(self):
        assert to_si(7.874, "g_cm3") == 7874.0

    def test_density_round_trip_has_no_float_noise(self):
        assert to_si(0.97, "g_cm3") == 970.0
        assert from_si(970.0, "g_cm3") == 0.97

    def test_pressure_units(self):
        assert to_si(193, "GPa") == 193e9
        assert to_si(515, "MPa") == 515e6

    def test_percent_is_a_ratio(self):
        assert to_si(50, "percent") == 0.5

    def test_temperature_offset(self):
        assert to_si(0, "degC") == 273.15

    def test_convert_between_same_dimension(self):
        assert convert(1.0, "GPa", "MPa") == 1000.0
        assert convert(970, "kgm3", "g_cm3") == 0.97

    def test_convert_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError):
            convert(1.0, "GPa", "K")

    def test_unknown_unit_passes_value_through(self):
        assert to_si(5.0, "not_a_unit") == 5.0

    def test_non_numeric_is_none(self):
        assert to_si("BCC", "K") is None
        assert to_si(None, "K") is None
        assert to_si(True, "K") is None

    def test_numeric_string_is_parsed(self):
        assert to_si(" 7.874 ", "g_cm3") == 7874.0


# ─────────────────────────────────────────────────────────────────────────
# Flatten
# ─────────────────────────────────────────────────────────────────────────

class TestFlatten:
    def test_flattens_named_groups(self):
        flat = flatten({
            "Name": "X",
            "PhysicalProperties": {"Density_g_cm3": 7.8},
            "MechanicalProperties": {"YieldStrength_MPa": 300},
        })
        assert flat["Density_g_cm3"] == 7.8
        assert flat["YieldStrength_MPa"] == 300

    def test_root_scalars_survive(self):
        flat = flatten({"Mass_amu": 55.8, "Charge_e": 0})
        assert flat["Mass_amu"] == 55.8
        assert flat["Charge_e"] == 0

    def test_properties_block_outranks_a_deeper_group(self):
        flat = flatten({
            "Properties": {"Density_kgm3": 970},
            "PhysicalProperties": {"Density_kgm3": 999},
        })
        assert flat["Density_kgm3"] == 970

    def test_excluded_groups_are_skipped(self):
        flat = flatten({
            "Composition": {"Fe": 2},
            "_derivation": {"confidence": 0.7},
            "Properties": {"Density_kgm3": 1},
        })
        assert "Fe" not in flat
        assert "confidence" not in flat
        assert flat["Density_kgm3"] == 1

    def test_underscore_keys_are_skipped(self):
        assert "_private" not in flatten({"_private": 1, "Mass_amu": 2})

    def test_deeply_nested_structures_do_not_explode(self):
        entry = {"A": {"B": {"C": {"D": {"E": 1}}}}}
        flatten(entry)  # must terminate, depth-capped


# ─────────────────────────────────────────────────────────────────────────
# DataSheet lookups
# ─────────────────────────────────────────────────────────────────────────

class TestDataSheet:
    def test_source_key_returns_source_value(self):
        s = DataSheet({"Density_g_cm3": 7.874})
        assert s["Density_g_cm3"] == 7.874

    def test_alias_key_converts_units(self):
        s = DataSheet({"Density_g_cm3": 7.874})
        assert s["Density_kgm3"] == 7874.0
        assert s.get("density") == 7.874

    def test_iteration_shows_only_source_keys(self):
        s = DataSheet({"Density_g_cm3": 7.874})
        assert list(s) == ["Density_g_cm3"]
        assert len(s) == 1

    def test_contains_is_alias_aware(self):
        s = DataSheet({"Density_g_cm3": 7.874})
        assert "Density_kgm3" in s
        assert "NopeProperty_XYZ" not in s

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            DataSheet({})["Density_kgm3"]

    def test_get_default(self):
        assert DataSheet({}).get("Density_kgm3", 5) == 5

    def test_si_view(self):
        s = DataSheet({"YoungsModulus_GPa": 193})
        assert s.si("youngs_modulus") == 193e9

    def test_source_and_unit_introspection(self):
        s = DataSheet({"Density_g_cm3": 7.874})
        assert s.source_of("Density_kgm3") == "Density_g_cm3"
        assert s.unit_of("Density_kgm3") == "g_cm3"

    def test_canonical_keys(self):
        s = DataSheet({"Density_g_cm3": 1, "Junk_XYZ": 2})
        assert s.canonical_keys() == ["density"]

    def test_first_source_key_wins_for_a_canonical(self):
        # Insertion order decides, matching flatten()'s precedence.
        s = DataSheet({"Density_kgm3": 970, "Density_g_cm3": 8.0})
        assert s.si("density") == 970.0

    def test_cross_dimension_alias_does_not_match(self):
        s = DataSheet({"MeltingPoint_K": 1811})
        assert s.get("Density_kgm3") is None


# ─────────────────────────────────────────────────────────────────────────
# Real bundled data
# ─────────────────────────────────────────────────────────────────────────

class TestBundledData:
    def test_curated_material_sheet_is_not_empty(self):
        # The whole point: these sheets nest their numbers under
        # PhysicalProperties / MechanicalProperties and used to sample empty.
        s = data_sheet("Stainless_Steel_316L")
        assert len(s) > 10
        assert s.si("density") > 7000

    def test_sampling_a_curated_material_works(self):
        assert sample("Stainless_Steel_316L", "Density_kgm3") > 7000
        assert sample("Stainless_Steel_316L", "YoungsModulus_GPa") > 100

    def test_sampling_accepts_any_known_spelling(self):
        a = sample("Stainless_Steel_316L", "Density_kgm3")
        b = sample("Stainless_Steel_316L", "density")
        assert a == pytest.approx(b * 1000)

    def test_element_sheet_resolves_bare_keys(self):
        s = data_sheet(Get("Iron"))
        assert s.si("density") == 7874.0
        assert s.si("melting_point") == 1811.0

    def test_si_sheet_is_canonical_only(self):
        si = si_sheet(Get("Iron"))
        assert "density" in si
        assert all(canonical_of(k) == k for k in si)
        assert all(isinstance(v, float) and math.isfinite(v) for v in si.values())

    def test_coverage_reports_presence(self):
        cov = coverage(Get("Stainless_Steel_316L"), ["density", "youngs_modulus", "band_gap"])
        assert cov["density"] is True
        assert cov["band_gap"] is False

    def test_unknown_property_still_returns_none(self):
        assert sample("Stainless_Steel_316L", "NonExistentProperty_XYZ") is None


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────

class TestValidateSheet:
    def test_clean_sheet_has_no_issues(self):
        assert validate_sheet({"Properties": {"Density_kgm3": 7870}}) == []

    def test_impossible_poisson_ratio_is_an_error(self):
        issues = validate_sheet({"Properties": {"PoissonsRatio": 0.9}})
        assert [i.severity for i in issues] == ["error"]

    def test_negative_density_is_an_error(self):
        issues = validate_sheet({"Properties": {"Density_kgm3": -5}})
        assert issues and issues[0].severity == "error"

    def test_unit_mislabel_is_flagged_with_a_hint(self):
        # 7874 g/cm3 is 1000x too large -- the number is really kg/m3.
        issues = validate_sheet({"Properties": {"Density_g_cm3": 7874}})
        assert issues
        assert "unit mislabel" in issues[0].message

    def test_descriptive_strings_are_not_flagged(self):
        assert validate_sheet({"LatticeProperties": {"PrimaryStructure": "BCC"}}) == []

    def test_name_prefixes_the_key(self):
        issues = validate_sheet({"Properties": {"PoissonsRatio": 9}}, name="Widget")
        assert issues[0].key.startswith("Widget.")

    def test_every_bundled_entry_is_sane(self):
        """No bundled data sheet may carry a physically impossible value."""
        from periodica.get import _registry  # noqa: PLC0415  (test-only introspection)

        errors = []
        for tier, index in _registry().by_tier.items():
            for key, entry in index.exact.items():
                for issue in validate_sheet(entry, name=f"{tier}/{key}"):
                    if issue.severity == "error":
                        errors.append(str(issue))
        assert not errors, "impossible values in bundled data:\n" + "\n".join(sorted(set(errors))[:20])


class TestListProperties:
    def test_returns_sorted_canonical_names(self):
        props = list_properties()
        assert props == sorted(props)
        assert "density" in props

    def test_property_spec_lookup(self):
        assert property_spec("density")["si_unit"] == "kg_m3"
        assert property_spec("no_such_property") is None
