"""Tests for the engine layer: channels, sheet contract, backends, grid.

The guarantees under test, in order of how much damage a regression would do:

1. Strategy changes the schedule, never the answer. Two grids that differ
   only in `how=` must compare equal, including where phases vary per grain.
2. Every channel resolves for every entry -- a downstream app never gets None.
3. Confidence is honest: measured values rate 1.0, defaults rate low, and a
   default never silently feeds a formula.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from periodica import Get
from periodica.engine import backends
from periodica.engine.backends import Plan, Strategy, choose, plan
from periodica.engine.channels import (
    UnknownChannel,
    channel_groups,
    expand_channels,
    list_channels,
    resolve,
    resolve_all,
)
from periodica.engine.datasheet import (
    EngineSheet,
    SheetBelowFloor,
    engine_sheet,
    validate,
)
from periodica.engine.grid import (
    grid_points,
    is_vectorisable,
    plan_field,
    sample_field,
    sample_points,
    sample_rgb,
    sample_slice,
)

#: A homogeneous curated material, and one with a per-grain phase field.
HOMOGENEOUS = "Stainless_Steel_316L"
TWO_PHASE = "Stainless-304"
#: Bounds and scale that put TWO_PHASE below its macro scale, so grains show.
GRAIN_BOUNDS = ((0.0, 0.0, 0.0), (4e-4, 4e-4, 4e-4))
GRAIN_SCALE = 1e-5


# ─────────────────────────────────────────────────────────────────────────
# Channels
# ─────────────────────────────────────────────────────────────────────────

class TestChannelConfig:
    def test_channels_exist(self):
        assert len(list_channels()) >= 15

    def test_groups_reference_real_channels(self):
        known = set(list_channels())
        for group, members in channel_groups().items():
            unknown = set(members) - known
            assert not unknown, f"group {group!r} names unknown channels {unknown}"

    def test_expand_accepts_group_channel_and_none(self):
        assert expand_channels(None) == list_channels()
        assert expand_channels("density") == ["density"]
        assert set(expand_channels("thermal")) == set(channel_groups()["thermal"])

    def test_expand_deduplicates_keeping_order(self):
        assert expand_channels(["density", "density", "stiffness"]) == ["density", "stiffness"]

    def test_unknown_channel_raises(self):
        with pytest.raises(UnknownChannel):
            resolve(HOMOGENEOUS, "no_such_channel")


class TestChannelResolution:
    def test_measured_value_is_read_directly(self):
        cv = resolve(HOMOGENEOUS, "density")
        assert cv.is_measured
        assert cv.confidence == 1.0
        assert cv.value > 7000

    def test_derived_value_names_its_formula(self):
        cv = resolve(HOMOGENEOUS, "thermal_diffusivity")
        assert cv.method.startswith("formula:")
        assert 0 < cv.confidence < 1.0

    def test_mixture_value_names_its_weighting(self):
        cv = resolve("Alpha_Brass", "stiffness")
        assert cv.method.startswith(("property:", "mixture:", "formula:"))
        assert cv.value > 0

    def test_units_are_si(self):
        assert resolve(HOMOGENEOUS, "stiffness").unit == "Pa"
        assert resolve(HOMOGENEOUS, "stiffness").value > 1e9

    def test_colour_is_a_linear_rgb_triple(self):
        cv = resolve(HOMOGENEOUS, "colour")
        assert isinstance(cv.value, tuple) and len(cv.value) == 3
        assert all(0.0 <= c <= 1.0 for c in cv.value)

    def test_every_channel_resolves_for_every_tier_sample(self):
        """No channel may come back None: consumers do not handle it."""
        names = [HOMOGENEOUS, "Iron", "HDPE", "Alumina_Al2O3", "Gold", "Alpha_Brass"]
        missing = []
        for name in names:
            for channel, cv in resolve_all(name).items():
                if cv.value is None:
                    missing.append(f"{name}.{channel}")
        assert not missing, f"unresolved channels: {missing}"

    def test_confidence_is_between_zero_and_one(self):
        for cv in resolve_all(HOMOGENEOUS).values():
            assert 0.0 <= cv.confidence <= 1.0

    def test_a_default_does_not_feed_a_formula(self):
        """Iron has no resistivity, so a Wiedemann-Franz conductivity built on
        the insulating default used to come out 7e-18 W/mK. The chain must
        fall through to its own default instead."""
        cv = resolve("Iron", "thermal_conductivity")
        assert cv.value > 1e-3, f"implausible conductivity {cv.value} via {cv.method}"

    def test_resolution_accepts_an_entry_dict(self):
        entry = Get(HOMOGENEOUS)
        assert resolve(entry, "density").value == resolve(HOMOGENEOUS, "density").value

    def test_formula_rejects_disallowed_syntax(self):
        from periodica.engine.channels import _safe_eval

        with pytest.raises(ValueError):
            _safe_eval("__import__('os').system('true')", {})

    def test_formula_evaluates_arithmetic(self):
        from periodica.engine.channels import _safe_eval

        assert _safe_eval("2 * a + sqrt(b)", {"a": 3.0, "b": 16.0}) == 10.0


# ─────────────────────────────────────────────────────────────────────────
# Sheet contract
# ─────────────────────────────────────────────────────────────────────────

class TestEngineSheet:
    def test_sheet_is_complete(self):
        sheet = engine_sheet(HOMOGENEOUS)
        assert len(sheet) == len(list_channels())
        assert all(sheet[c] is not None for c in sheet)

    def test_sheet_reads_as_a_mapping(self):
        sheet = engine_sheet(HOMOGENEOUS)
        assert isinstance(sheet, EngineSheet)
        assert sheet["density"] > 7000
        assert "density" in dict(sheet)

    def test_provenance_is_available_per_channel(self):
        sheet = engine_sheet(HOMOGENEOUS)
        assert sheet.confidence("density") == 1.0
        assert sheet.method("density").startswith("property:")
        assert sheet.unit("density") == "kg_m3"

    def test_measured_and_defaulted_partition_sensibly(self):
        sheet = engine_sheet(HOMOGENEOUS)
        assert set(sheet.measured()).isdisjoint(sheet.defaulted())

    def test_weakest_is_sorted_ascending(self):
        weakest = engine_sheet(HOMOGENEOUS).weakest(5)
        assert [w.confidence for w in weakest] == sorted(w.confidence for w in weakest)

    def test_channel_subset_and_group_selection(self):
        assert set(engine_sheet(HOMOGENEOUS, ["density"])) == {"density"}
        assert set(engine_sheet(HOMOGENEOUS, "thermal")) == set(channel_groups()["thermal"])

    def test_strict_floor_raises_and_names_offenders(self):
        with pytest.raises(SheetBelowFloor) as excinfo:
            engine_sheet("Iron", "mechanical", min_confidence=0.99)
        assert excinfo.value.offenders

    def test_non_strict_floor_reports_instead_of_raising(self):
        sheet = engine_sheet("Iron", "mechanical", min_confidence=0.99, strict=False)
        assert sheet.below_floor()

    def test_a_well_covered_sheet_meets_a_reasonable_floor(self):
        engine_sheet(HOMOGENEOUS, "bulk", min_confidence=0.5)

    def test_mean_confidence_is_a_grade(self):
        assert 0.0 < engine_sheet(HOMOGENEOUS).mean_confidence <= 1.0

    def test_json_round_trips(self):
        import json

        payload = json.loads(engine_sheet(HOMOGENEOUS).to_json())
        assert payload["name"] == HOMOGENEOUS
        assert payload["channels"]["density"]["unit"] == "kg_m3"

    def test_uniforms_are_prefixed(self):
        uniforms = engine_sheet(HOMOGENEOUS).uniforms()
        assert all(k.startswith("pd_") for k in uniforms)


class TestValidate:
    def test_curated_material_validates(self):
        report = validate(HOMOGENEOUS)
        assert report["ok"] is True
        assert report["errors"] == []
        assert report["channels_total"] == len(list_channels())

    def test_report_counts_coverage(self):
        report = validate(HOMOGENEOUS)
        assert report["channels_measured"] > 0
        assert report["properties_recognised"] > 10

    def test_floor_violation_makes_it_not_ok(self):
        assert validate("Iron", min_confidence=0.99)["ok"] is False


# ─────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────

class TestStrategySelection:
    @pytest.mark.parametrize("points, expected", [
        (1, Strategy.SCALAR),
        (500, Strategy.SCALAR),
        (10_000, Strategy.VECTOR),
        (5_000_000, Strategy.VECTOR),
        (50_000_000, Strategy.CHUNKED),
    ])
    def test_auto_policy(self, points, expected):
        assert choose(points)[0] is expected

    def test_explicit_request_is_honoured(self):
        assert choose(10, how="vector")[0] is Strategy.VECTOR

    def test_reason_is_always_given(self):
        assert choose(10_000)[1]

    def test_non_vectorisable_falls_back_to_scalar_and_says_so(self):
        strategy, reason = choose(10_000, how="vector", vectorisable=False)
        assert strategy is Strategy.SCALAR
        assert "not" in reason

    def test_parallel_without_workers_degrades_to_chunked(self):
        assert choose(10**9, how="parallel", workers=1)[0] is Strategy.CHUNKED

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            choose(10, how="magic")

    def test_thresholds_are_overridable(self):
        assert choose(10, thresholds={"scalar_max": 1})[0] is not Strategy.SCALAR


class TestPlan:
    def test_plan_describes_without_running(self):
        p = plan(10_000_000)
        assert isinstance(p, Plan)
        assert p.points == 10_000_000
        assert p.bytes_peak > 0
        assert str(p)

    def test_chunked_plan_has_several_chunks(self):
        assert plan(50_000_000).chunks > 1

    def test_plan_serialises(self):
        assert plan(1000).as_dict()["strategy"] in {s.value for s in Strategy}

    def test_plan_field_uses_the_grid_size(self):
        p = plan_field(HOMOGENEOUS, resolution=64)
        assert p.points == 64 ** 3


class TestChunkHelpers:
    def test_slices_cover_exactly(self):
        slices = list(backends.chunk_slices(10, 3))
        assert [(s.start, s.stop) for s in slices] == [(0, 3), (3, 6), (6, 9), (9, 10)]

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError):
            list(backends.chunk_slices(10, 0))


# ─────────────────────────────────────────────────────────────────────────
# Grid geometry
# ─────────────────────────────────────────────────────────────────────────

class TestGridPoints:
    def test_3d_shape_and_count(self):
        points, shape = grid_points(((0, 0, 0), (1, 1, 1)), 4)
        assert shape == (4, 4, 4)
        assert points.shape == (64, 3)

    def test_2d_is_embedded_at_the_offset(self):
        points, shape = grid_points(((0, 0), (1, 1)), 4, dims=2, plane="xz", at=0.25)
        assert shape == (4, 4)
        assert points.shape == (16, 3)
        assert np.allclose(points[:, 1], 0.25)   # y is the axis xz omits

    def test_points_are_cell_centres_inside_the_bounds(self):
        points, _ = grid_points(((0, 0, 0), (1, 1, 1)), 2)
        assert points.min() > 0.0 and points.max() < 1.0

    def test_anisotropic_resolution(self):
        _points, shape = grid_points(((0, 0, 0), (1, 1, 1)), (2, 3, 4))
        assert shape == (2, 3, 4)

    def test_bad_dims_raises(self):
        with pytest.raises(ValueError):
            grid_points(((0, 0, 0), (1, 1, 1)), 4, dims=4)

    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError):
            grid_points(((1, 1, 1), (0, 0, 0)), 4)

    def test_unknown_plane_raises(self):
        with pytest.raises(ValueError):
            grid_points(((0, 0), (1, 1)), 4, dims=2, plane="qq")

    def test_resolution_length_must_match_dims(self):
        with pytest.raises(ValueError):
            grid_points(((0, 0, 0), (1, 1, 1)), (2, 3), dims=3)


# ─────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────

class TestSampleField:
    def test_homogeneous_field_is_constant(self):
        field = sample_field(HOMOGENEOUS, "density",
                             bounds=((0, 0, 0), (1, 1, 1)), resolution=8)
        assert field.shape == (8, 8, 8)
        assert len(np.unique(field)) == 1

    def test_channel_and_raw_property_both_work(self):
        bounds = ((0, 0, 0), (1, 1, 1))
        by_channel = sample_field(HOMOGENEOUS, "stiffness", bounds=bounds, resolution=4)
        by_property = sample_field(HOMOGENEOUS, "YoungsModulus_GPa", bounds=bounds, resolution=4)
        assert by_channel.flat[0] == pytest.approx(by_property.flat[0] * 1e9)

    def test_2d_slice_shape(self):
        slab = sample_slice(HOMOGENEOUS, "density",
                            bounds=((0, 0), (1, 1)), resolution=(16, 8))
        assert slab.shape == (16, 8)

    def test_unknown_property_is_nan_not_an_exception(self):
        field = sample_field(HOMOGENEOUS, "NoSuchProperty_XYZ",
                             bounds=((0, 0, 0), (1, 1, 1)), resolution=2)
        assert np.all(np.isnan(field))

    def test_rgb_sampling_has_a_colour_axis(self):
        rgb = sample_rgb(HOMOGENEOUS, bounds=((0, 0, 0), (1, 1, 1)), resolution=4)
        assert rgb.shape == (4, 4, 4, 3)

    def test_multi_component_channel_rejected_by_scalar_sampler(self):
        with pytest.raises(ValueError):
            sample_field(HOMOGENEOUS, "colour",
                         bounds=((0, 0, 0), (1, 1, 1)), resolution=2)

    def test_points_must_be_three_component(self):
        with pytest.raises(ValueError):
            sample_points(HOMOGENEOUS, "density", np.zeros((4, 2)))


class TestPhaseVariation:
    def test_grains_produce_more_than_one_value(self):
        field = sample_field(TWO_PHASE, "YoungsModulus_GPa",
                             bounds=GRAIN_BOUNDS, resolution=16, scale_m=GRAIN_SCALE)
        assert len(np.unique(field)) >= 2, "expected per-grain variation"

    def test_variation_reaches_channels_not_just_raw_keys(self):
        """A phase is a material in its own right: asking for the `stiffness`
        channel must give each grain its own stiffness, not the bulk value."""
        field = sample_field(TWO_PHASE, "stiffness",
                             bounds=GRAIN_BOUNDS, resolution=16, scale_m=GRAIN_SCALE)
        assert len(np.unique(field)) >= 2

    def test_above_the_macro_scale_it_is_bulk(self):
        field = sample_field(TWO_PHASE, "YoungsModulus_GPa",
                             bounds=((0, 0, 0), (1, 1, 1)), resolution=8, scale_m=1.0)
        assert len(np.unique(field)) == 1

    def test_assignment_is_reproducible(self):
        kwargs = dict(bounds=GRAIN_BOUNDS, resolution=12, scale_m=GRAIN_SCALE)
        first = sample_field(TWO_PHASE, "stiffness", **kwargs)
        second = sample_field(TWO_PHASE, "stiffness", **kwargs)
        assert np.array_equal(first, second)


class TestStrategyEquivalence:
    """Strategy changes the schedule, never the answer."""

    @pytest.mark.parametrize("how", ["scalar", "vector", "chunked", "parallel"])
    def test_homogeneous_agrees_across_strategies(self, how):
        kwargs = dict(bounds=((0, 0, 0), (1, 1, 1)), resolution=6)
        reference = sample_field(HOMOGENEOUS, "density", how="vector", **kwargs)
        assert np.array_equal(
            reference, sample_field(HOMOGENEOUS, "density", how=how, **kwargs),
            equal_nan=True,
        )

    @pytest.mark.parametrize("how", ["scalar", "vector", "chunked", "parallel"])
    def test_per_grain_field_agrees_across_strategies(self, how):
        """The hard case: phase assignment must not depend on the schedule.
        A separate scalar hash here is what would silently break it."""
        kwargs = dict(bounds=GRAIN_BOUNDS, resolution=8, scale_m=GRAIN_SCALE)
        reference = sample_field(TWO_PHASE, "stiffness", how="vector", **kwargs)
        assert np.array_equal(
            reference, sample_field(TWO_PHASE, "stiffness", how=how, **kwargs),
            equal_nan=True,
        )

    def test_vectorised_and_legacy_scalar_sampler_agree(self):
        """The grid sampler and periodica.sample.sample() share the phase hash,
        so the same point gets the same grain through either entry point."""
        from periodica.sample import sample as scalar_sample

        points, _shape = grid_points(GRAIN_BOUNDS, 6)
        vectorised = sample_points(TWO_PHASE, "YoungsModulus_GPa", points,
                                   scale_m=GRAIN_SCALE, how="vector")
        one_at_a_time = np.array([
            scalar_sample(TWO_PHASE, "YoungsModulus_GPa",
                          at=tuple(p), scale_m=GRAIN_SCALE)
            for p in points
        ], dtype=float)
        assert np.array_equal(vectorised, one_at_a_time, equal_nan=True)


class TestVectorisability:
    def test_builtin_models_are_vectorised(self):
        assert is_vectorisable(HOMOGENEOUS)
        assert is_vectorisable(TWO_PHASE)

    def test_custom_model_falls_back_but_still_samples(self):
        from periodica.sample import register_field_model

        register_field_model(
            "test_only_ramp",
            lambda field, prop, at, scale_m, entry: (at[0] if at else 0.0),
        )
        entry = {"Name": "Ramp", "Field": {"model": "test_only_ramp"}}
        assert is_vectorisable(entry) is False
        field = sample_field(entry, "anything",
                             bounds=((0, 0, 0), (1, 1, 1)), resolution=4)
        assert len(np.unique(field)) > 1


class TestScale:
    def test_scale_defaults_to_the_cell_size(self):
        """Left unset, scale_m is the cell size -- which is what tells a
        scale-dependent field whether the caller is looking at grains."""
        grainy = sample_field(TWO_PHASE, "YoungsModulus_GPa",
                              bounds=GRAIN_BOUNDS, resolution=16)
        assert len(np.unique(grainy)) >= 2

        coarse = sample_field(TWO_PHASE, "YoungsModulus_GPa",
                              bounds=((0, 0, 0), (1.0, 1.0, 1.0)), resolution=4)
        assert len(np.unique(coarse)) == 1
