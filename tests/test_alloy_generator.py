"""
Tests for the AlloyGenerator.
Verifies alloy generation, composition validity, and schema compliance.
"""

import pytest
from periodica.utils.alloy_generator import AlloyGenerator


@pytest.fixture
def generator():
    return AlloyGenerator()


class TestGenerateAll:
    def test_generates_alloys(self, generator):
        alloys = generator.generate_all(count_limit=20)
        assert len(alloys) > 0
        assert len(alloys) <= 20

    def test_no_duplicate_formulas(self, generator):
        alloys = generator.generate_all(count_limit=40)
        formulas = [a['Formula'] for a in alloys]
        assert len(formulas) == len(set(formulas))

    def test_all_have_required_fields(self, generator):
        alloys = generator.generate_all(count_limit=10)
        required = ['Name', 'Formula', 'Category', 'Components',
                     'PhysicalProperties', 'MechanicalProperties', 'LatticeProperties']
        for alloy in alloys:
            for field in required:
                assert field in alloy, f"Missing {field} in {alloy.get('Name', '?')}"

    def test_all_have_derivation_metadata(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            assert '_derivation' in alloy
            assert alloy['_derivation']['source'] == 'auto_generated'

    def test_progress_callback(self, generator):
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        generator.generate_all(count_limit=10, progress_callback=cb)
        assert len(calls) > 0
        assert calls[-1][0] == 100

    def test_respects_count_limit(self, generator):
        for limit in [5, 15, 30]:
            alloys = generator.generate_all(count_limit=limit)
            assert len(alloys) <= limit


class TestComponentsValidity:
    def test_components_sum_near_100(self, generator):
        alloys = generator.generate_all(count_limit=20)
        for alloy in alloys:
            components = alloy['Components']
            # Use midpoint of min/max
            total = sum(
                (c['MinPercent'] + c['MaxPercent']) / 2.0
                for c in components
            )
            assert 95 <= total <= 105, (
                f"{alloy['Name']} components sum to {total:.1f}%"
            )

    def test_has_base_element(self, generator):
        alloys = generator.generate_all(count_limit=15)
        for alloy in alloys:
            roles = [c['Role'] for c in alloy['Components']]
            assert 'Base' in roles, f"{alloy['Name']} has no Base component"

    def test_components_have_required_keys(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            for comp in alloy['Components']:
                assert 'Element' in comp
                assert 'MinPercent' in comp
                assert 'MaxPercent' in comp
                assert 'Role' in comp

    def test_min_less_than_max(self, generator):
        alloys = generator.generate_all(count_limit=15)
        for alloy in alloys:
            for comp in alloy['Components']:
                assert comp['MinPercent'] <= comp['MaxPercent'], (
                    f"{alloy['Name']}: {comp['Element']} min > max"
                )


class TestSteelVariants:
    def test_generates_steels(self, generator):
        steels = generator.generate_steel_variants(count=5)
        assert len(steels) > 0
        assert len(steels) <= 5

    def test_all_fe_based(self, generator):
        steels = generator.generate_steel_variants(count=5)
        for steel in steels:
            base = next(c for c in steel['Components'] if c['Role'] == 'Base')
            assert base['Element'] == 'Fe'

    def test_category_is_steel(self, generator):
        steels = generator.generate_steel_variants(count=3)
        for steel in steels:
            assert steel['Category'] == 'Steel'

    def test_stainless_has_chromium(self, generator):
        steels = generator.generate_steel_variants(count=15)
        stainless = [s for s in steels if 'Stainless' in s['Name']]
        assert len(stainless) > 0
        for ss in stainless:
            cr = [c for c in ss['Components'] if c['Element'] == 'Cr']
            assert len(cr) > 0
            assert cr[0]['MaxPercent'] > 10


class TestAluminumAlloys:
    def test_generates_al_alloys(self, generator):
        alloys = generator.generate_aluminum_alloys(count=5)
        assert len(alloys) > 0

    def test_all_al_based(self, generator):
        alloys = generator.generate_aluminum_alloys(count=5)
        for alloy in alloys:
            base = next(c for c in alloy['Components'] if c['Role'] == 'Base')
            assert base['Element'] == 'Al'

    def test_category_is_aluminum(self, generator):
        alloys = generator.generate_aluminum_alloys(count=3)
        for alloy in alloys:
            assert alloy['Category'] == 'Aluminum Alloy'


class TestCopperAlloys:
    def test_generates_cu_alloys(self, generator):
        alloys = generator.generate_copper_alloys(count=5)
        assert len(alloys) > 0

    def test_all_cu_based(self, generator):
        alloys = generator.generate_copper_alloys(count=5)
        for alloy in alloys:
            base = next(c for c in alloy['Components'] if c['Role'] == 'Base')
            assert base['Element'] == 'Cu'


class TestBinaryAlloys:
    def test_generates_binary(self, generator):
        metals = ['Fe', 'Cu', 'Ni', 'Al', 'Zn', 'Ag']
        alloys = generator.generate_binary_alloys(metals, count=5)
        assert len(alloys) > 0

    def test_binary_has_two_elements(self, generator):
        metals = ['Cu', 'Ni', 'Zn', 'Ag']
        alloys = generator.generate_binary_alloys(metals, count=5)
        for alloy in alloys:
            assert len(alloy['Components']) == 2


class TestTernaryAlloys:
    def test_generates_ternary(self, generator):
        metals = ['Fe', 'Cr', 'Ni', 'Cu', 'Zn', 'Al']
        alloys = generator.generate_ternary_alloys(metals, count=5)
        assert len(alloys) > 0

    def test_ternary_has_three_elements(self, generator):
        metals = ['Fe', 'Cr', 'Ni', 'Mo', 'Mn']
        alloys = generator.generate_ternary_alloys(metals, count=5)
        for alloy in alloys:
            assert len(alloy['Components']) == 3


class TestPhysicalProperties:
    def test_positive_density(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            d = alloy['PhysicalProperties']['Density_g_cm3']
            assert d > 0, f"{alloy['Name']} has non-positive density"

    def test_positive_melting_point(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            mp = alloy['PhysicalProperties']['MeltingPoint_K']
            assert mp > 300, f"{alloy['Name']} melting point too low: {mp}"

    def test_positive_thermal_conductivity(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            tc = alloy['PhysicalProperties']['ThermalConductivity_W_mK']
            assert tc > 0


class TestMechanicalProperties:
    def test_positive_tensile_strength(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            ts = alloy['MechanicalProperties']['TensileStrength_MPa']
            assert ts > 0

    def test_yield_less_than_tensile(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            ts = alloy['MechanicalProperties']['TensileStrength_MPa']
            ys = alloy['MechanicalProperties']['YieldStrength_MPa']
            assert ys <= ts

    def test_positive_elongation(self, generator):
        alloys = generator.generate_all(count_limit=10)
        for alloy in alloys:
            el = alloy['MechanicalProperties']['Elongation_percent']
            assert el > 0


class TestMetallicElements:
    def test_returns_metals(self, generator):
        metals = generator.get_metallic_elements()
        assert 'Fe' in metals
        assert 'Cu' in metals
        assert 'Al' in metals

    def test_excludes_nonmetals(self, generator):
        metals = generator.get_metallic_elements()
        assert 'O' not in metals
        assert 'N' not in metals
        assert 'He' not in metals

    def test_filters_available(self, generator):
        metals = generator.get_metallic_elements(['Fe', 'O', 'Cu', 'N'])
        assert metals == ['Fe', 'Cu']
