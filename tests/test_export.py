"""Tests for periodica.export: STL, OBJ+MTL, VTK, SDF, HLSL."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from periodica import (
    export_hlsl,
    export_obj,
    export_sdf_raw,
    export_stl,
    export_vtk_legacy,
    list_tiers,
    reload_registry,
    voxel_phase_map,
    voxel_sample,
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_built():
    reload_registry()
    if "alloys" not in list_tiers():
        from periodica.__main__ import main as cli_main
        cli_main(["build", "alloys", "-q"])
        reload_registry()
    yield


BOUNDS = ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
VOXEL = 1.0
SCALE = 1e-7   # below Steel-1018's macro_scale_m so phases activate


# ─────────────────────────────────────────────────────────────────────────
# Voxel helpers
# ─────────────────────────────────────────────────────────────────────────

class TestVoxelHelpers:
    def test_voxel_phase_map_shape(self):
        grid, names = voxel_phase_map(
            "Steel-1018", bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE
        )
        assert grid.shape == (4, 4, 4)
        assert "ferrite" in names and "pearlite" in names

    def test_voxel_phase_map_only_declared_phase_ids(self):
        grid, names = voxel_phase_map(
            "Steel-1018", bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE
        )
        unique = set(int(v) for v in grid.flatten())
        # All ids must be valid indices into phase_names (or -1).
        for u in unique:
            assert u == -1 or 0 <= u < len(names)

    def test_voxel_sample_returns_property_grid(self):
        grid = voxel_sample(
            "Steel-1018", "YoungsModulus_GPa",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE,
        )
        assert grid.shape == (4, 4, 4)
        # Must be one of the declared values: 195 (ferrite), 220 (pearlite),
        # or 200 (bulk fallback).
        unique = set(np.unique(grid).tolist())
        assert unique <= {195.0, 200.0, 220.0}, unique


# ─────────────────────────────────────────────────────────────────────────
# STL
# ─────────────────────────────────────────────────────────────────────────

class TestSTL:
    def test_binary_stl_has_valid_header_and_count(self, tmp_path):
        out = export_stl(
            "Steel-1018", tmp_path / "x.stl",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE, binary=True,
        )
        data = out.read_bytes()
        assert len(data) > 84
        n_tris = struct.unpack("<I", data[80:84])[0]
        # Each triangle = 50 bytes (12 floats normal/3 verts + 2-byte attribute)
        assert len(data) == 84 + n_tris * 50, "STL file size doesn't match header"
        assert n_tris > 0

    def test_ascii_stl_format(self, tmp_path):
        out = export_stl(
            "Steel-1018", tmp_path / "x_ascii.stl",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE, binary=False,
        )
        text = out.read_text(encoding="ascii")
        assert text.startswith("solid Steel-1018")
        assert "facet normal" in text
        assert text.rstrip().endswith("endsolid Steel-1018")


# ─────────────────────────────────────────────────────────────────────────
# OBJ + MTL
# ─────────────────────────────────────────────────────────────────────────

class TestOBJ:
    def test_obj_and_mtl_written_and_referenced(self, tmp_path):
        obj, mtl = export_obj(
            "Steel-1018", tmp_path / "x.obj",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE,
        )
        assert obj.exists() and mtl.exists()
        text = obj.read_text(encoding="utf-8")
        assert f"mtllib {mtl.name}" in text

    def test_obj_groups_per_phase(self, tmp_path):
        obj, _ = export_obj(
            "Steel-1018", tmp_path / "x.obj",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE,
        )
        text = obj.read_text(encoding="utf-8")
        assert "usemtl ferrite" in text
        assert "usemtl pearlite" in text

    def test_mtl_has_per_phase_materials_with_color(self, tmp_path):
        _, mtl = export_obj(
            "Steel-1018", tmp_path / "x.obj",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE,
        )
        text = mtl.read_text(encoding="utf-8")
        assert "newmtl ferrite" in text and "newmtl pearlite" in text
        # Each material should have a Kd line.
        assert text.count("Kd ") == 2

    def test_obj_vertex_includes_color_extension(self, tmp_path):
        obj, _ = export_obj(
            "Steel-1018", tmp_path / "x.obj",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE,
        )
        text = obj.read_text(encoding="utf-8")
        # OBJ v line with trailing RGB has 7 tokens: v X Y Z R G B
        v_lines = [l for l in text.splitlines() if l.startswith("v ")]
        assert v_lines, "no vertex lines"
        for line in v_lines[:5]:
            assert len(line.split()) == 7, f"missing RGB on vertex: {line}"

    def test_mtl_includes_property_comments_when_requested(self, tmp_path):
        _, mtl = export_obj(
            "Steel-1018", tmp_path / "x.obj",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE,
            properties=["YoungsModulus_GPa"],
        )
        text = mtl.read_text(encoding="utf-8")
        assert "# YoungsModulus_GPa = 195" in text
        assert "# YoungsModulus_GPa = 220" in text


# ─────────────────────────────────────────────────────────────────────────
# VTK
# ─────────────────────────────────────────────────────────────────────────

class TestVTK:
    def test_vtk_legacy_structured_points(self, tmp_path):
        out = export_vtk_legacy(
            "Steel-1018", tmp_path / "x.vtk",
            bounds=BOUNDS, voxel_size=VOXEL,
            properties=["Density_kgm3", "YoungsModulus_GPa"], scale_m=SCALE,
        )
        text = out.read_text(encoding="utf-8")
        assert "DATASET STRUCTURED_POINTS" in text
        assert "DIMENSIONS 4 4 4" in text
        assert "SCALARS phase_id int 1" in text
        assert "SCALARS Density_kgm3 float 1" in text
        assert "SCALARS YoungsModulus_GPa float 1" in text


# ─────────────────────────────────────────────────────────────────────────
# SDF
# ─────────────────────────────────────────────────────────────────────────

class TestSDF:
    def test_sdf_occupancy_size_and_header(self, tmp_path):
        out = export_sdf_raw(
            "Steel-1018", tmp_path / "x.sdf",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE, mode="occupancy",
        )
        n_voxels = 4 * 4 * 4
        assert out.stat().st_size == n_voxels * 4   # float32
        header = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
        assert header["shape_xyz"] == [4, 4, 4]
        assert header["mode"] == "occupancy"
        assert header["dtype"] == "float32"

    def test_sdf_phase_mode_writes_phase_indices(self, tmp_path):
        out = export_sdf_raw(
            "Steel-1018", tmp_path / "y.sdf",
            bounds=BOUNDS, voxel_size=VOXEL, scale_m=SCALE, mode="phase",
        )
        data = np.frombuffer(out.read_bytes(), dtype=np.float32)
        assert data.size == 4 * 4 * 4
        # All values must be in {-1, 0, 1} cast to float
        assert set(np.unique(data).tolist()) <= {-1.0, 0.0, 1.0}


# ─────────────────────────────────────────────────────────────────────────
# HLSL
# ─────────────────────────────────────────────────────────────────────────

class TestHLSL:
    def test_hlsl_struct_and_function_emitted(self, tmp_path):
        out = export_hlsl(
            "Steel-1018", tmp_path / "x.hlsl",
            properties=["YoungsModulus_GPa", "TensileStrength_MPa", "Density_kgm3"],
        )
        text = out.read_text(encoding="utf-8")
        assert "struct Steel_1018Sample" in text
        assert "SampleSteel_1018(float3 pos, float scaleM)" in text
        assert "float YoungsModulus_GPa" in text
        assert "int   PhaseId" in text

    def test_hlsl_has_phase_dispatch_branches(self, tmp_path):
        out = export_hlsl("Steel-1018", tmp_path / "x.hlsl")
        text = out.read_text(encoding="utf-8")
        # Two phases declared in JSON => two cum/if branches.
        assert text.count("if (u < cum)") == 2
        # ferrite/pearlite property values must appear as float literals.
        assert "195.0" in text
        assert "220.0" in text

    def test_hlsl_handles_homogeneous_entry_with_no_phases(self, tmp_path):
        out = export_hlsl("Aluminum-6061", tmp_path / "al.hlsl")
        text = out.read_text(encoding="utf-8")
        # Without phases, the function still compiles to a bulk-only sample.
        assert "SampleAluminum_6061" in text
        assert "if (u < cum)" not in text


# ─────────────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────────────

class TestExportCLI:
    def test_export_stl_via_cli(self, tmp_path, capsys):
        from periodica.__main__ import main as cli_main
        out = tmp_path / "cli.stl"
        rc = cli_main([
            "export", "Steel-1018",
            "--format", "stl",
            "--output", str(out),
            "--bounds", "0,0,0,3,3,3",
            "--voxel-size", "1.0",
            "--scale", "1e-7",
        ])
        assert rc == 0
        assert out.exists() and out.stat().st_size > 84

    def test_export_hlsl_via_cli(self, tmp_path, capsys):
        from periodica.__main__ import main as cli_main
        out = tmp_path / "cli.hlsl"
        rc = cli_main(["export", "Steel-1018", "--format", "hlsl", "--output", str(out)])
        assert rc == 0
        assert "SampleSteel_1018" in out.read_text(encoding="utf-8")
