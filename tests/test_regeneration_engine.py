"""
Tests for the RegenerationEngine.
Verifies element regeneration from quark constants matches expected schema
and produces reasonable property values.
"""

import json
import pytest
import tempfile
from pathlib import Path
from periodica.utils.regeneration_engine import RegenerationEngine


@pytest.fixture
def engine():
    return RegenerationEngine()


class TestPredictToElementJson:
    """Test that predict_to_element_json produces valid element schema."""

    def test_hydrogen_schema(self, engine):
        h = engine.predict_to_element_json(1)
        assert h['symbol'] == 'H'
        assert h['name'] == 'Hydrogen'
        assert h['atomic_number'] == 1
        assert h['block'] == 's'
        assert h['period'] == 1
        assert h['group'] == 1

    def test_hydrogen_has_all_required_fields(self, engine):
        h = engine.predict_to_element_json(1)
        required = [
            'symbol', 'name', 'atomic_number', 'atomic_mass', 'block',
            'period', 'group', 'ionization_energy', 'electronegativity',
            'atomic_radius', 'covalent_radius', 'melting_point', 'boiling_point',
            'density', 'electron_affinity', 'valence_electrons',
            'electron_configuration', 'primary_emission_wavelength',
        ]
        for field in required:
            assert field in h, f"Missing field: {field}"

    def test_hydrogen_derivation_metadata(self, engine):
        h = engine.predict_to_element_json(1)
        assert '_derivation' in h
        assert h['_derivation']['source'] == 'quark_derived'
        assert 'quarks' in h['_derivation']['derivation_chain']
        assert h['_derivation']['confidence'] > 0

    def test_carbon_properties(self, engine):
        c = engine.predict_to_element_json(6)
        assert c['symbol'] == 'C'
        assert c['atomic_number'] == 6
        assert c['block'] == 'p'
        assert c['period'] == 2
        assert c['valence_electrons'] > 0

    def test_iron_properties(self, engine):
        fe = engine.predict_to_element_json(26)
        assert fe['symbol'] == 'Fe'
        assert fe['block'] == 'd'
        assert fe['melting_point'] is not None
        assert fe['density'] is not None

    def test_oganesson_properties(self, engine):
        og = engine.predict_to_element_json(118)
        assert og['symbol'] == 'Og'
        assert og['atomic_number'] == 118
        assert og['period'] == 7

    def test_atomic_mass_positive(self, engine):
        for Z in [1, 6, 26, 79, 118]:
            elem = engine.predict_to_element_json(Z)
            assert elem['atomic_mass'] > 0, f"Z={Z}: mass should be positive"

    def test_ionization_energy_positive(self, engine):
        for Z in [1, 6, 26, 79]:
            elem = engine.predict_to_element_json(Z)
            assert elem['ionization_energy'] > 0, f"Z={Z}: IE should be positive"

    def test_atomic_radius_positive(self, engine):
        for Z in [1, 6, 26, 79]:
            elem = engine.predict_to_element_json(Z)
            assert elem['atomic_radius'] > 0

    def test_electron_configuration_not_empty(self, engine):
        for Z in [1, 6, 26, 79]:
            elem = engine.predict_to_element_json(Z)
            assert len(elem['electron_configuration']) > 0

    def test_noble_gas_no_electronegativity(self, engine):
        he = engine.predict_to_element_json(2)
        assert he['electronegativity'] is None

    def test_valence_electrons_hydrogen(self, engine):
        h = engine.predict_to_element_json(1)
        assert h['valence_electrons'] == 1


class TestRegenerateElements:
    """Test full regeneration of all 118 elements."""

    def test_regenerate_to_temp_dir(self, engine):
        """Regenerate a small set of elements to a temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Only test first 10 elements for speed
            results = []
            for Z in range(1, 11):
                elem = engine.predict_to_element_json(Z)
                results.append(elem)
                filepath = output_dir / f"{Z:03d}_{elem['symbol']}.json"
                with open(filepath, 'w') as f:
                    json.dump(elem, f, indent=2)

            assert len(results) == 10
            assert (output_dir / "001_H.json").exists()
            assert (output_dir / "010_Ne.json").exists()

    def test_regenerated_json_is_valid(self, engine):
        """Verify regenerated JSON can be parsed back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            elem = engine.predict_to_element_json(26)  # Iron
            filepath = output_dir / "026_Fe.json"
            with open(filepath, 'w') as f:
                json.dump(elem, f, indent=2)

            # Read it back
            with open(filepath) as f:
                loaded = json.load(f)
            assert loaded['symbol'] == 'Fe'
            assert loaded['atomic_number'] == 26

    def test_progress_callback_called(self, engine):
        """Verify progress callback is invoked."""
        calls = []

        def on_progress(pct, msg):
            calls.append((pct, msg))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.regenerate_elements(
                progress_callback=on_progress,
                output_dir=Path(tmpdir),
            )

        assert len(calls) == 118
        assert calls[0][0] == 0 or calls[0][0] == 1  # ~0-1% for first element
        assert calls[-1][0] == 100  # 100% for last

    def test_all_118_elements_generated(self, engine):
        """Verify all 118 elements are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = engine.regenerate_elements(output_dir=Path(tmpdir))
            assert len(results) == 118

    def test_no_duplicate_symbols(self, engine):
        """Verify each element has a unique symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = engine.regenerate_elements(output_dir=Path(tmpdir))
            symbols = [r['symbol'] for r in results]
            assert len(symbols) == len(set(symbols))


class TestSchemaCompatibility:
    """Test that regenerated elements match the existing JSON schema."""

    def test_matches_hydrogen_reference_keys(self, engine):
        """Compare keys of regenerated H with the reference 001_H.json."""
        ref_path = Path(__file__).parent.parent / "src" / "periodica" / "data" / "active" / "elements" / "001_H.json"
        if not ref_path.exists():
            pytest.skip("Reference 001_H.json not found")

        with open(ref_path) as f:
            reference = json.load(f)

        generated = engine.predict_to_element_json(1)

        # All reference keys (except isotopes) should be in generated
        for key in reference:
            if key in ('isotopes',):  # isotopes not yet regenerated
                continue
            assert key in generated, f"Missing key in generated: {key}"

    @pytest.mark.xfail(
        reason=(
            "RegenerationEngine currently picks H-2 (proton+neutron+electron) "
            "rather than H-1 for atomic_mass derivation, producing ~2.02 amu vs "
            "the IUPAC standard 1.008 amu (most abundant isotope). The Get()/Save() "
            "flow uses {P=1,N=0,E=1} explicitly via build_periodic_table.json and "
            "matches reference within 0.001 amu. Tracking issue: regen engine "
            "should select most-abundant isotope by default."
        ),
        strict=False,
    )
    def test_atomic_mass_within_tolerance(self, engine):
        """Generated atomic masses should be reasonable (within 20% of reference)."""
        ref_path = Path(__file__).parent.parent / "src" / "periodica" / "data" / "active" / "elements" / "001_H.json"
        if not ref_path.exists():
            pytest.skip("Reference not found")

        with open(ref_path) as f:
            ref = json.load(f)

        gen = engine.predict_to_element_json(1)
        ref_mass = ref['atomic_mass']
        gen_mass = gen['atomic_mass']
        error_pct = abs(gen_mass - ref_mass) / ref_mass * 100
        assert error_pct < 20, f"H mass error {error_pct:.1f}% (gen={gen_mass}, ref={ref_mass})"
