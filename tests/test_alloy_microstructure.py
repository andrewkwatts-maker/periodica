"""Tests for the microstructure_voronoi field model on alloys."""
from __future__ import annotations

import pytest

from periodica import sample, reload_registry, list_tiers
from periodica.__main__ import main as cli_main


@pytest.fixture(scope="module", autouse=True)
def _ensure_built():
    reload_registry()
    if "alloys" not in list_tiers():
        cli_main(["build", "alloys", "-q"])
        reload_registry()
    yield


class TestMicrostructureVoronoi:
    def test_macro_scale_returns_bulk(self):
        # macro_scale_m=1e-4 in input; sampling at scale_m=1.0 returns bulk.
        v = sample("Steel-1018", "YoungsModulus_GPa", at=(0, 0, 0), scale_m=1.0)
        assert v == 200  # bulk Young's

    def test_micro_scale_picks_a_phase(self):
        # At sub-macro scale, the value is one of the declared phase values.
        seen = set()
        for i in range(40):
            v = sample(
                "Steel-1018",
                "YoungsModulus_GPa",
                at=(i * 1.0, i * 0.7, 0),
                scale_m=1e-7,
            )
            seen.add(v)
        # Should see ferrite (195) and pearlite (220) and possibly bulk fallback (200).
        assert {195, 220} & seen, f"phase values not seen; observed {seen}"

    def test_phase_values_match_input(self):
        valid = {195, 220, 200}  # ferrite, pearlite, bulk fallback
        for i in range(20):
            v = sample(
                "Steel-1018",
                "YoungsModulus_GPa",
                at=(i, 0, 0),
                scale_m=1e-7,
            )
            assert v in valid, f"unexpected value {v}"

    def test_at_same_point_is_deterministic(self):
        v1 = sample("Steel-1018", "YoungsModulus_GPa", at=(3.5, 1.2, 0.7), scale_m=1e-7)
        v2 = sample("Steel-1018", "YoungsModulus_GPa", at=(3.5, 1.2, 0.7), scale_m=1e-7)
        assert v1 == v2

    def test_no_at_returns_bulk(self):
        # Without `at`, even at micro scale we fall back to bulk.
        v = sample("Steel-1018", "YoungsModulus_GPa", scale_m=1e-7)
        assert v == 200

    def test_existing_homogeneous_alloys_still_work(self):
        # Aluminum-6061 stayed homogeneous; sampling at any point returns bulk.
        for at in [(0, 0, 0), (1e3, 1e3, 1e3), (1e-9, 1e-9, 1e-9)]:
            v = sample("Aluminum-6061", "YoungsModulus_GPa", at=at, scale_m=1e-9)
            assert v == 68.9

    def test_stainless_phase_values(self):
        # Austenite=195, ferrite=210; bulk fallback=193.
        valid = {195, 210, 193}
        for i in range(20):
            v = sample(
                "Stainless-304",
                "YoungsModulus_GPa",
                at=(i, i * 0.5, i * 0.3),
                scale_m=1e-7,
            )
            assert v in valid, f"Stainless-304: unexpected {v}"
