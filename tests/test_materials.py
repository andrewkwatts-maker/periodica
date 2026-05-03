"""Tests for complex-material tiers + 3D sampling.

Each material's generated data sheet is cross-checked against
data/reference/<tier>/<name>.json (real-world ground truth), confirming
that input-JSON-driven generation matches authoritative sources without
any hardcoded chemistry or properties in code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from periodica import (
    Get,
    Scope,
    data_sheet,
    list_tiers,
    register_field_model,
    reload_registry,
    sample,
)


REF = Path(__file__).resolve().parents[1] / "src" / "periodica" / "data" / "reference"


def _ref(folder: str, name: str) -> dict:
    return json.loads((REF / folder / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def _ensure_built():
    # Material tiers must exist before tests run. If they're missing,
    # invoke the CLI build to create them.
    reload_registry()
    if not all(t in list_tiers() for t in ("alloys", "polymers", "ceramics", "composites")):
        from periodica.__main__ import main as cli_main
        cli_main(["build", "-q"])
        reload_registry()
    yield


# ─────────────────────────────────────────────────────────────────────────
# Bulk data-sheet equality vs reference
# ─────────────────────────────────────────────────────────────────────────

class TestDataSheetVsReference:
    @pytest.mark.parametrize(
        "tier_dir, name, props_to_check",
        [
            ("alloys", "Steel-1018",        ["Density_kgm3", "YoungsModulus_GPa", "TensileStrength_MPa", "MeltingPoint_K"]),
            ("alloys", "Aluminum-6061",     ["Density_kgm3", "YoungsModulus_GPa", "TensileStrength_MPa", "MeltingPoint_K"]),
            ("alloys", "Titanium-Ti6Al4V",  ["Density_kgm3", "YoungsModulus_GPa", "TensileStrength_MPa", "MeltingPoint_K"]),
            ("polymers", "HDPE",            ["Density_kgm3", "YoungsModulus_GPa", "TensileStrength_MPa", "MeltingPoint_K"]),
            ("polymers", "PET",             ["Density_kgm3", "YoungsModulus_GPa", "TensileStrength_MPa", "MeltingPoint_K"]),
            ("ceramics", "Alumina",         ["Density_kgm3", "YoungsModulus_GPa", "CompressiveStrength_MPa", "MeltingPoint_K"]),
            ("ceramics", "SiliconCarbide",  ["Density_kgm3", "YoungsModulus_GPa", "CompressiveStrength_MPa", "MeltingPoint_K"]),
        ],
    )
    def test_generated_matches_reference(self, tier_dir, name, props_to_check):
        sheet = data_sheet(name)
        ref = _ref(tier_dir, name)
        for prop in props_to_check:
            assert sheet.get(prop) == ref[prop], (
                f"{name}.{prop}: generated={sheet.get(prop)} ref={ref[prop]}"
            )


# ─────────────────────────────────────────────────────────────────────────
# Sampling (homogeneous)
# ─────────────────────────────────────────────────────────────────────────

class TestHomogeneousSampling:
    def test_homogeneous_ignores_position(self):
        # Steel-1018 is homogeneous: same Young's modulus everywhere.
        e_origin = sample("Steel-1018", "YoungsModulus_GPa", at=(0, 0, 0))
        e_far    = sample("Steel-1018", "YoungsModulus_GPa", at=(1e6, 1e6, 1e6))
        e_none   = sample("Steel-1018", "YoungsModulus_GPa")
        assert e_origin == e_far == e_none == 200

    def test_homogeneous_ignores_scale(self):
        # No scale_dependent block on Aluminum-6061 -> scale_m irrelevant.
        for s in (1.0, 1e-3, 1e-9):
            assert sample("Aluminum-6061", "Density_kgm3", scale_m=s) == 2700

    def test_unknown_property_returns_None(self):
        assert sample("Steel-1018", "NonExistentProperty_XYZ") is None


# ─────────────────────────────────────────────────────────────────────────
# Sampling (anisotropic composites)
# ─────────────────────────────────────────────────────────────────────────

class TestAnisotropicSampling:
    def test_cfrp_axial_vs_transverse(self):
        ref = _ref("composites", "CFRP")
        e_axial      = sample("CFRP", "YoungsModulus_GPa", at=(1, 0, 0))
        e_transverse = sample("CFRP", "YoungsModulus_GPa", at=(0, 1, 0))
        assert e_axial == ref["AxialYoungsModulus_GPa"]
        assert e_transverse == ref["TransverseYoungsModulus_GPa"]

    def test_cfrp_off_axis_between_extremes(self):
        e_axial      = sample("CFRP", "YoungsModulus_GPa", at=(1, 0, 0))
        e_transverse = sample("CFRP", "YoungsModulus_GPa", at=(0, 1, 0))
        e_45         = sample("CFRP", "YoungsModulus_GPa", at=(1, 1, 0))
        assert e_transverse < e_45 < e_axial

    def test_cfrp_strength_is_anisotropic(self):
        s_axial     = sample("CFRP", "TensileStrength_MPa", at=(1, 0, 0))
        s_transverse = sample("CFRP", "TensileStrength_MPa", at=(0, 1, 0))
        # Real CFRP: axial tensile ~3500 MPa, transverse ~50 MPa (70x ratio).
        assert s_axial > 50 * s_transverse


# ─────────────────────────────────────────────────────────────────────────
# Sampling (scale-dependent)
# ─────────────────────────────────────────────────────────────────────────

class TestScaleDependentSampling:
    def test_concrete_macro_returns_bulk(self):
        # Macro scale -> bulk property regardless of `at`.
        assert sample("Concrete-Mix", "YoungsModulus_GPa", scale_m=1.0, at=(0, 0, 0)) == 30

    def test_concrete_micro_picks_a_phase(self):
        # Micro scale (< macro_scale_m=1e-2) -> phase-specific value.
        # Different points -> potentially different phases. Sample many points
        # and confirm at least two different phase values appear.
        seen = set()
        for i in range(50):
            v = sample(
                "Concrete-Mix",
                "YoungsModulus_GPa",
                at=(i * 1.0, 0, 0),
                scale_m=1e-3,
            )
            seen.add(v)
        # bulk value (30) plus phase-specific values (25, 60, 0) should appear.
        assert len(seen) >= 2, f"Expected phase variation; saw only {seen}"

    def test_micro_phase_values_match_field(self):
        # Phase YoungsModulus values declared in input: 25, 60, 0, plus bulk 30.
        valid = {25, 60, 0, 30}
        for i in range(20):
            v = sample(
                "Concrete-Mix",
                "YoungsModulus_GPa",
                at=(i * 0.5, i * 0.3, 0),
                scale_m=1e-4,
            )
            assert v in valid, f"Unexpected phase value {v}"


# ─────────────────────────────────────────────────────────────────────────
# Mass / Composition validation (the Get layer still works for materials)
# ─────────────────────────────────────────────────────────────────────────

class TestMaterialsCompose:
    def test_pure_copper_has_only_copper_constituent(self):
        cu = Get("Copper-C11000")
        assert cu["Composition"] == {"Cu": 1.0}

    def test_steel_composition_records_atomic_input(self):
        steel = Get("Steel-1018")
        assert steel["Composition"]["Fe"] == 0.99
        assert steel["Composition"]["C"] == 0.01

    def test_alumina_atomic_composition(self):
        alumina = Get("Alumina")
        assert alumina["Composition"] == {"Al": 2, "O": 3}


# ─────────────────────────────────────────────────────────────────────────
# Field-model registry extensibility
# ─────────────────────────────────────────────────────────────────────────

class TestFieldModelRegistry:
    def test_register_custom_model(self, tmp_path):
        # Register a "double" model that returns 2x the bulk property.
        def double(field, prop, at, scale_m, entry):
            from periodica.sample import _props_of  # type: ignore
            v = _props_of(entry).get(prop)
            return None if v is None else v * 2

        register_field_model("double", double)

        # Save into tmp_path (NOT the package data dir) so the test doesn't
        # leak entries into the user's installed library.
        from periodica import Save, reload_registry
        Save(
            "TestDouble",
            {
                "Properties": {"Density_kgm3": 1000},
                "Field": {"model": "double"},
            },
            dir=tmp_path,
        )
        # The Save path won't be in the registry walk (different folder),
        # so test the evaluator on the dict directly.
        entry = {
            "Properties": {"Density_kgm3": 1000},
            "Field": {"model": "double"},
        }
        assert sample(entry, "Density_kgm3") == 2000


# ─────────────────────────────────────────────────────────────────────────
# CLI integration
# ─────────────────────────────────────────────────────────────────────────

class TestSampleCLI:
    def test_sample_cli_returns_value(self, capsys):
        from periodica.__main__ import main as cli_main
        rc = cli_main(["sample", "Steel-1018", "Density_kgm3"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out == "7870"

    def test_sample_cli_with_at_anisotropic(self, capsys):
        from periodica.__main__ import main as cli_main
        rc = cli_main(["sample", "CFRP", "YoungsModulus_GPa", "--at", "1,0,0"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert float(out) == 230.0

    def test_sample_cli_full_data_sheet(self, capsys):
        from periodica.__main__ import main as cli_main
        rc = cli_main(["sample", "Alumina"])
        out = capsys.readouterr().out
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["Density_kgm3"] == 3950
        assert "YoungsModulus_GPa" in parsed
