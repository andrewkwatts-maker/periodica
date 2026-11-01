#====== periodica/src/periodica/_periodica_core.pyi ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

# Rust impl: rust/periodica_core/src/pyfacade/
"""Type stub for the compiled ``periodica._periodica_core`` extension.

Declares exactly the names the extension exports -- no more, no fewer.
``tests/test_wrapper_conformance.py`` compares this file against the built
module in both directions, so adding a ``#[pyfunction]`` without declaring it
here (or the reverse) fails the suite.

These are the raw bindings. Public code calls the wrappers in ``periodica.*``;
the bindings may change shape between releases.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

_Vec3 = Tuple[float, float, float]
_Bounds = Tuple[_Vec3, _Vec3]

__version__: str

# ── sentinels ──────────────────────────────────────────────────────────────
def is_rust_backend() -> bool: ...
def version_rust() -> str: ...

# ── registry (pyfacade/registry.rs) ────────────────────────────────────────
def py_get(spec: str, scope_str: Optional[str] = None) -> Dict[str, Any]: ...
def py_save(name: str, data_json: str, tier_str: str) -> Optional[Dict[str, Any]]: ...
def py_list_tiers() -> List[str]: ...

# ── sampling (pyfacade/sample.rs) ──────────────────────────────────────────
def py_sample(
    name: str,
    property: str,
    at: Optional[_Vec3] = None,
    scale_m: Optional[float] = None,
) -> Optional[float | int]: ...
def py_data_sheet(name: str) -> Dict[str, Any]: ...

# ── export (pyfacade/export.rs) ────────────────────────────────────────────
def py_export_glsl(
    name: str,
    include_density: bool = True,
    include_ior: bool = True,
    include_sss: bool = True,
    include_caustic: bool = True,
) -> str: ...
def py_export_hlsl(
    name: str, out_path: str, properties: Optional[Sequence[str]] = None
) -> str: ...
def py_export_sdf_raw(
    name: str,
    out_path: str,
    bounds: _Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
    mode: str = "phase",
) -> str: ...
def py_export_vtk_legacy(
    name: str,
    out_path: str,
    bounds: _Bounds,
    voxel_size: float,
    properties: Optional[Sequence[str]] = None,
    scale_m: Optional[float] = None,
) -> str: ...
def py_export_stl(
    name: str,
    out_path: str,
    bounds: _Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
    binary: bool = True,
) -> str: ...
def py_export_obj(
    name: str,
    obj_path: str,
    bounds: _Bounds,
    voxel_size: float,
    scale_m: Optional[float] = None,
) -> Tuple[str, str]: ...
def py_bake_fourier(
    entry_name: str,
    property: str,
    bounds: _Bounds,
    grid_size: Tuple[int, int, int],
    truncate_threshold: float = 0.01,
) -> Dict[str, Any]: ...

# ── protein (pyfacade/protein.rs) ──────────────────────────────────────────
def py_kabsch_rmsd(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
) -> float: ...
def py_build_backbone(
    sequence: str, phi_psi_deg: Sequence[Tuple[float, float]]
) -> List[Dict[str, Any]]: ...
def py_build_backbone_from_entry(entry_name: str) -> List[Dict[str, Any]]: ...
def py_ramachandran_region(phi_deg: float, psi_deg: float) -> str: ...

# ── alloy (pyfacade/alloy.rs) ──────────────────────────────────────────────
def py_optimize_alloy(
    targets: Sequence[Dict[str, Any]],
    base: str,
    alloying_pool: Optional[Sequence[str]] = None,
    n_candidates: int = 1000,
    top_k: int = 5,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]: ...
