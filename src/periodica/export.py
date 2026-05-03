"""3D exports for any registered entry with a microstructure / phase field.

Generates voxelised representations of an entry's spatial structure plus a
selection of common artifact formats:

    voxel_sample(name, prop, ...)       3D ndarray of property samples
    voxel_phase_map(name, ...)          3D ndarray of phase indices + names
    export_stl(name, path, ...)         binary STL boundary mesh (per-phase groups)
    export_obj(name, path, ...)         OBJ + sibling .mtl with per-phase materials
    export_vtk_legacy(name, path, ...)  ASCII legacy VTK with point scalars
    export_sdf_raw(name, path, ...)     binary float32 3D volume (occupancy or distance)
    export_hlsl(name, path, ...)        HLSL Sample<Name>(float3 pos, float scaleM) function

All formats stay data-driven: the underlying field model is whatever is
declared in the entry's JSON `Field` block. No chemistry/physics is
hardcoded here. Pure-numpy.

Vertex colours encode phase identity by default (a deterministic palette
keyed by phase name); per-vertex sampled property values can be requested
via the `properties` argument.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from periodica.get import Get
from periodica.sample import sample, _props_of  # type: ignore


Bounds = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


# ─────────────────────────────────────────────────────────────────────────
# Voxel sampling helpers
# ─────────────────────────────────────────────────────────────────────────

def _validate_bounds(bounds: Bounds) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bounds must be ((x0,y0,z0), (x1,y1,z1))")
    if np.any(hi <= lo):
        raise ValueError(f"bounds upper {hi} must be > lower {lo}")
    return lo, hi


def _voxel_centers(bounds: Bounds, voxel_size: float) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    lo, hi = _validate_bounds(bounds)
    extent = hi - lo
    nx, ny, nz = (np.ceil(extent / voxel_size).astype(int)).tolist()
    nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)
    xs = lo[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = lo[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = lo[2] + (np.arange(nz) + 0.5) * voxel_size
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid, (nx, ny, nz)


def voxel_sample(
    name: str,
    prop: str,
    *,
    bounds: Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
) -> np.ndarray:
    """Return a 3D ndarray of property samples on a regular voxel grid.

    `scale_m` defaults to the voxel side length, which is the natural
    sampling resolution.
    """
    grid, (nx, ny, nz) = _voxel_centers(bounds, voxel_size)
    if scale_m is None:
        scale_m = voxel_size
    out = np.empty((nx, ny, nz), dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                v = sample(name, prop, at=tuple(grid[ix, iy, iz]), scale_m=scale_m)
                out[ix, iy, iz] = float("nan") if v is None else float(v)
    return out


def voxel_phase_map(
    name: str,
    *,
    bounds: Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return (int phase-id grid, [phase_name, ...]).

    The first phase listed is the one with the highest declared volume
    fraction. Voxels in entries without a phase field map to id -1 and
    name "<bulk>".
    """
    entry = Get(name)
    field = entry.get("Field") or {}
    phases = field.get("phases") or {}
    phase_names = sorted(phases.keys(), key=lambda k: -float(phases[k].get("fraction", 0)))
    if not phase_names:
        phase_names = ["<bulk>"]

    grid, (nx, ny, nz) = _voxel_centers(bounds, voxel_size)
    if scale_m is None:
        scale_m = voxel_size
    macro = float(field.get("macro_scale_m", 0.0))
    macro_active = scale_m < macro and bool(phases)

    g_density = float(field.get("grain_density", 1.0))
    grain_size = max(1e-12, 1.0 / max(1e-12, g_density)) ** (1.0 / 3.0)

    out = np.full((nx, ny, nz), -1, dtype=np.int32)
    if not macro_active:
        out.fill(0 if phase_names == ["<bulk>"] else -1)
        return out, phase_names

    fractions = [float(phases[n].get("fraction", 0.0)) for n in phase_names]
    cum = np.cumsum(fractions)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                px, py, pz = grid[ix, iy, iz]
                qx = round(px / grain_size)
                qy = round(py / grain_size)
                qz = round(pz / grain_size)
                h = hashlib.sha1(repr((qx, qy, qz)).encode()).digest()
                u = int.from_bytes(h[:8], "big") / 2 ** 64
                idx = int(np.searchsorted(cum, u))
                if idx >= len(phase_names):
                    idx = -1
                out[ix, iy, iz] = idx
    return out, phase_names


# ─────────────────────────────────────────────────────────────────────────
# Phase colour palette (deterministic from name)
# ─────────────────────────────────────────────────────────────────────────

def _phase_color(phase_name: str) -> Tuple[float, float, float]:
    """Deterministic RGB in [0, 1] from a phase/material name."""
    h = hashlib.sha1(phase_name.encode()).digest()
    return (h[0] / 255.0, h[1] / 255.0, h[2] / 255.0)


# ─────────────────────────────────────────────────────────────────────────
# STL: voxel-block boundary mesh
# ─────────────────────────────────────────────────────────────────────────

# 6 cube face directions (offset to neighbour, plus the 4 corner offsets).
_FACES = [
    # (delta neighbour, normal, 4 corner offsets)
    ((-1,  0,  0), ( -1.0, 0.0, 0.0), [(0,0,0),(0,1,0),(0,1,1),(0,0,1)]),
    (( 1,  0,  0), (  1.0, 0.0, 0.0), [(1,0,0),(1,0,1),(1,1,1),(1,1,0)]),
    (( 0, -1,  0), ( 0.0,-1.0, 0.0), [(0,0,0),(0,0,1),(1,0,1),(1,0,0)]),
    (( 0,  1,  0), ( 0.0, 1.0, 0.0), [(0,1,0),(1,1,0),(1,1,1),(0,1,1)]),
    (( 0,  0, -1), ( 0.0, 0.0,-1.0), [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]),
    (( 0,  0,  1), ( 0.0, 0.0, 1.0), [(0,0,1),(0,1,1),(1,1,1),(1,0,1)]),
]


def _boundary_quads(
    phase_grid: np.ndarray,
    bounds: Bounds,
    voxel_size: float,
) -> Iterable[Tuple[np.ndarray, np.ndarray, int]]:
    """Yield (quad_corners(4,3), normal(3,), phase_id) for each boundary face.

    A face is on a "boundary" if the neighbouring voxel has a different
    phase id (or is outside the grid).
    """
    lo, _hi = _validate_bounds(bounds)
    nx, ny, nz = phase_grid.shape
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                phase = int(phase_grid[ix, iy, iz])
                if phase < 0:
                    continue
                origin = lo + np.array([ix, iy, iz]) * voxel_size
                for (dx, dy, dz), normal, offsets in _FACES:
                    nx_ix, ny_iy, nz_iz = ix + dx, iy + dy, iz + dz
                    if (0 <= nx_ix < nx) and (0 <= ny_iy < ny) and (0 <= nz_iz < nz):
                        if int(phase_grid[nx_ix, ny_iy, nz_iz]) == phase:
                            continue   # interior face, skip
                    quad = np.array([
                        origin + np.array(off, dtype=float) * voxel_size
                        for off in offsets
                    ])
                    yield quad, np.asarray(normal, dtype=float), phase


def export_stl(
    name: str,
    path,
    *,
    bounds: Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
    binary: bool = True,
) -> Path:
    """Write a binary STL of the boundary surface.

    The output is a single solid; phase grouping is preserved in OBJ
    (use `export_obj` if you need per-phase materials).
    """
    phase_grid, phase_names = voxel_phase_map(
        name, bounds=bounds, voxel_size=voxel_size, scale_m=scale_m
    )
    triangles: List[Tuple[np.ndarray, np.ndarray]] = []  # (verts(3,3), normal(3,))
    for quad, normal, _phase in _boundary_quads(phase_grid, bounds, voxel_size):
        triangles.append((np.array([quad[0], quad[1], quad[2]]), normal))
        triangles.append((np.array([quad[0], quad[2], quad[3]]), normal))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        with open(out_path, "wb") as f:
            header = f"periodica STL of {name}".encode("ascii")[:80]
            f.write(header.ljust(80, b"\0"))
            f.write(struct.pack("<I", len(triangles)))
            for verts, normal in triangles:
                f.write(struct.pack("<3f", *normal))
                for v in verts:
                    f.write(struct.pack("<3f", *v))
                f.write(struct.pack("<H", 0))
    else:
        with open(out_path, "w", encoding="ascii") as f:
            f.write(f"solid {name}\n")
            for verts, normal in triangles:
                f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("  outer loop\n")
                for v in verts:
                    f.write(f"    vertex {v[0]} {v[1]} {v[2]}\n")
                f.write("  endloop\n")
                f.write("endfacet\n")
            f.write(f"endsolid {name}\n")
    return out_path


# ─────────────────────────────────────────────────────────────────────────
# OBJ + MTL: per-phase materials, vertex colours
# ─────────────────────────────────────────────────────────────────────────

def export_obj(
    name: str,
    path,
    *,
    bounds: Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
    properties: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path]:
    """Write OBJ + sibling MTL.

    - One `usemtl <phase>` per face group.
    - Vertex colors encode the phase via the deterministic palette.
    - If `properties` is given, those property values per phase are written
      as comments on the corresponding `newmtl` block (so the file remains
      a useful self-contained data sheet).

    Returns (obj_path, mtl_path).
    """
    obj_path = Path(path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")

    phase_grid, phase_names = voxel_phase_map(
        name, bounds=bounds, voxel_size=voxel_size, scale_m=scale_m
    )
    entry = Get(name)
    field = entry.get("Field") or {}
    phases_data = field.get("phases") or {}

    # Pre-compute per-phase property values for MTL comments.
    prop_table = {}
    if properties:
        for ph in phase_names:
            if ph == "<bulk>":
                bulk_props = _props_of(entry)
                prop_table[ph] = {p: bulk_props.get(p) for p in properties}
            else:
                row = phases_data.get(ph, {})
                prop_table[ph] = {p: row.get(p) for p in properties}

    # Write MTL.
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(f"# periodica MTL for {name}\n")
        for ph in phase_names:
            r, g, b = _phase_color(ph)
            f.write(f"\nnewmtl {ph}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            f.write(f"Ka {r * 0.2:.4f} {g * 0.2:.4f} {b * 0.2:.4f}\n")
            f.write(f"Ks 0.5 0.5 0.5\n")
            f.write(f"Ns 32\n")
            if properties and ph in prop_table:
                for k, v in prop_table[ph].items():
                    if v is not None:
                        f.write(f"# {k} = {v}\n")

    # Write OBJ.
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"# periodica OBJ of {name}\n")
        f.write(f"mtllib {mtl_path.name}\n")
        # Group by phase; within each group, emit unique vertices and faces.
        for phase_id, ph in enumerate(phase_names):
            f.write(f"\no {name}_{ph}\n")
            f.write(f"usemtl {ph}\n")
            r, g, b = _phase_color(ph)
            verts: List[Tuple[float, float, float]] = []
            faces: List[Tuple[int, int, int, int]] = []
            for quad, _normal, q_phase in _boundary_quads(phase_grid, bounds, voxel_size):
                if q_phase != phase_id:
                    continue
                base = len(verts) + 1   # OBJ is 1-indexed
                for v in quad:
                    verts.append((v[0], v[1], v[2]))
                faces.append((base, base + 1, base + 2, base + 3))
            for vx, vy, vz in verts:
                # OBJ extension: trailing RGB after vertex coords (Blender, etc.)
                f.write(f"v {vx} {vy} {vz} {r:.4f} {g:.4f} {b:.4f}\n")
            for a, b_, c, d in faces:
                f.write(f"f {a} {b_} {c} {d}\n")
    return obj_path, mtl_path


# ─────────────────────────────────────────────────────────────────────────
# VTK legacy ASCII: structured-points scalars
# ─────────────────────────────────────────────────────────────────────────

def export_vtk_legacy(
    name: str,
    path,
    *,
    bounds: Bounds,
    voxel_size: float,
    properties: Sequence[str],
    scale_m: Optional[float] = None,
) -> Path:
    """Write a STRUCTURED_POINTS legacy VTK file with one POINT_DATA scalar
    array per property in `properties`, plus a `phase_id` array.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase_grid, phase_names = voxel_phase_map(
        name, bounds=bounds, voxel_size=voxel_size, scale_m=scale_m
    )
    nx, ny, nz = phase_grid.shape
    lo, _ = _validate_bounds(bounds)

    prop_grids: List[Tuple[str, np.ndarray]] = []
    for prop in properties:
        prop_grids.append((prop, voxel_sample(
            name, prop, bounds=bounds, voxel_size=voxel_size, scale_m=scale_m,
        )))

    n = nx * ny * nz
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"periodica {name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {lo[0]} {lo[1]} {lo[2]}\n")
        f.write(f"SPACING {voxel_size} {voxel_size} {voxel_size}\n")
        f.write(f"POINT_DATA {n}\n")
        # phase_id
        f.write("SCALARS phase_id int 1\nLOOKUP_TABLE default\n")
        # VTK is x-fastest, our ndarray is (ix, iy, iz); flatten in Fortran order.
        flat = np.ascontiguousarray(phase_grid.transpose(2, 1, 0)).ravel()
        f.write("\n".join(str(int(v)) for v in flat))
        f.write("\n")
        # property scalars
        for prop, grid in prop_grids:
            f.write(f"SCALARS {prop} float 1\nLOOKUP_TABLE default\n")
            flat = np.ascontiguousarray(grid.transpose(2, 1, 0)).ravel()
            f.write("\n".join(repr(float(v)) for v in flat))
            f.write("\n")
    return out_path


# ─────────────────────────────────────────────────────────────────────────
# SDF: binary float32 occupancy / pseudo-distance field
# ─────────────────────────────────────────────────────────────────────────

def export_sdf_raw(
    name: str,
    path,
    *,
    bounds: Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
    mode: str = "occupancy",
) -> Path:
    """Write a binary float32 3D volume.

    `mode`:
      - "occupancy": 1.0 inside any phase, 0.0 outside (-1 phase id).
      - "phase":     float-cast phase id (-1, 0, 1, ...).

    Layout is x-fastest, then y, then z (matching VTK).
    A small header file `<path>.json` is written alongside with
    dimensions, voxel size, and the phase name list.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase_grid, phase_names = voxel_phase_map(
        name, bounds=bounds, voxel_size=voxel_size, scale_m=scale_m
    )
    if mode == "occupancy":
        vol = (phase_grid >= 0).astype(np.float32)
    elif mode == "phase":
        vol = phase_grid.astype(np.float32)
    else:
        raise ValueError(f"Unknown SDF mode {mode!r}; expected 'occupancy' or 'phase'.")
    flat = np.ascontiguousarray(vol.transpose(2, 1, 0)).ravel()
    with open(out_path, "wb") as f:
        f.write(flat.tobytes())
    header = {
        "name": name,
        "mode": mode,
        "shape_xyz": list(phase_grid.shape),
        "voxel_size_m": voxel_size,
        "bounds": [list(bounds[0]), list(bounds[1])],
        "phase_names": phase_names,
        "dtype": "float32",
        "layout": "x-fastest",
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(header, indent=2), encoding="utf-8"
    )
    return out_path


# ─────────────────────────────────────────────────────────────────────────
# HLSL shader code emission
# ─────────────────────────────────────────────────────────────────────────

def _hlsl_safe_ident(s: str) -> str:
    """Make a string usable as an HLSL identifier."""
    out = "".join(c if c.isalnum() else "_" for c in s)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


def export_hlsl(
    name: str,
    path,
    *,
    properties: Optional[Sequence[str]] = None,
) -> Path:
    """Emit an HLSL `Sample<Name>(float3 pos, float scaleM)` function.

    The generated code mirrors the entry's microstructure_voronoi field
    semantics (deterministic point-hash phase dispatch). All numbers
    come from the entry's JSON declaration -- nothing is hardcoded here.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry = Get(name)
    field = entry.get("Field") or {}
    bulk = _props_of(entry)
    phases = field.get("phases") or {}
    macro = float(field.get("macro_scale_m", 0.0))
    g_density = float(field.get("grain_density", 1.0))
    grain_size = max(1e-12, 1.0 / max(1e-12, g_density)) ** (1.0 / 3.0)

    # Determine the union of property names to expose in the struct.
    keys: List[str] = list(properties) if properties else []
    if not keys:
        seen = set()
        for source in [bulk] + [p for p in phases.values() if isinstance(p, dict)]:
            for k, v in source.items():
                if isinstance(v, (int, float)) and k != "fraction" and k not in seen:
                    keys.append(k); seen.add(k)
    keys = list(dict.fromkeys(keys))

    safe = _hlsl_safe_ident(name)
    fn_name = f"Sample{safe}"

    lines: List[str] = []
    lines.append(f"// Generated by periodica.export.export_hlsl for {name!r}.")
    lines.append("// All values come from the entry's JSON Field declaration.")
    lines.append("// Hash-based phase dispatch matches the Python sample() semantics")
    lines.append("// up to float-precision (the Python side uses sha1; here we use a")
    lines.append("// standard procedural-shader hash for performance/portability).")
    lines.append("")
    struct_name = f"{safe}Sample"
    lines.append(f"struct {struct_name} {{")
    for k in keys:
        lines.append(f"    float {_hlsl_safe_ident(k)};")
    lines.append("    int   PhaseId;")
    lines.append("};")
    lines.append("")
    lines.append(f"static const float {safe}_MACRO_SCALE_M = {macro};")
    lines.append(f"static const float {safe}_GRAIN_SIZE_M  = {grain_size};")
    lines.append("")
    lines.append(f"float {safe}_Hash(float3 q) {{")
    lines.append("    return frac(sin(dot(q, float3(12.9898, 78.233, 37.7191))) * 43758.5453);")
    lines.append("}")
    lines.append("")
    lines.append(f"{struct_name} {fn_name}(float3 pos, float scaleM) {{")
    lines.append(f"    {struct_name} s;")
    # bulk path
    for k in keys:
        v = bulk.get(k, 0.0) or 0.0
        lines.append(f"    s.{_hlsl_safe_ident(k)} = {float(v)};")
    lines.append("    s.PhaseId = -1;")
    lines.append(f"    if (scaleM >= {safe}_MACRO_SCALE_M) return s;")

    if phases:
        lines.append("    float3 q = floor(pos / max(1e-12, " + safe + "_GRAIN_SIZE_M));")
        lines.append(f"    float u = {safe}_Hash(q);")
        lines.append("    float cum = 0.0;")
        for idx, (ph_name, ph_data) in enumerate(phases.items()):
            frac = float(ph_data.get("fraction", 0.0))
            lines.append(f"    cum += {frac};")
            lines.append(f"    if (u < cum) {{")
            for k in keys:
                v = ph_data.get(k, bulk.get(k, 0.0)) or 0.0
                lines.append(f"        s.{_hlsl_safe_ident(k)} = {float(v)};")
            lines.append(f"        s.PhaseId = {idx};")
            lines.append("        return s;")
            lines.append("    }")
    lines.append("    return s;")
    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
