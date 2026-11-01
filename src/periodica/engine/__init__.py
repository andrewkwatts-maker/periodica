"""Engine-facing layer: what a renderer or simulation needs from a material.

The library's core answers "what does this material measure". This package
answers "what do I feed the engine", which is a different question with
different requirements: every value present, one unit, one name, honest about
how it was obtained, and fast in bulk.

    from periodica.engine import engine_sheet, sample_field

    sheet = engine_sheet("Stainless_Steel_316L")
    sheet["density"]              # 7990.0, always kg/m3, always present
    sheet.confidence("stiffness") # 1.0 -- read from the sheet, not derived

    field = sample_field("Stainless-304", "stiffness",
                         bounds=((0, 0, 0), (4e-4, 4e-4, 4e-4)),
                         resolution=128, scale_m=1e-5)   # per-grain, 128^3

Modules
-------
`channels`   the quantities an engine asks for, each defined as its fallback
             chain, resolved with provenance and confidence
`datasheet`  `EngineSheet`: the complete, graded, canonical sheet contract
`backends`   execution strategies (scalar/vector/chunked/parallel) and the
             documented policy that picks one
`grid`       2D and 3D bulk sampling, vectorised, strategy-selectable
"""
from periodica.engine.channels import (
    ChannelValue,
    UnknownChannel,
    channel,
    channel_groups,
    channel_spec,
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
from periodica.engine.backends import Plan, Strategy, plan
from periodica.engine.grid import (
    grid_points,
    is_vectorisable,
    plan_field,
    sample_field,
    sample_points,
    sample_rgb,
    sample_slice,
)

__all__ = [
    # Channels
    "ChannelValue", "UnknownChannel", "channel", "channel_groups",
    "channel_spec", "expand_channels", "list_channels", "resolve", "resolve_all",
    # Sheet contract
    "EngineSheet", "SheetBelowFloor", "engine_sheet", "validate",
    # Execution strategy
    "Plan", "Strategy", "plan",
    # Bulk sampling
    "grid_points", "is_vectorisable", "plan_field", "sample_field",
    "sample_points", "sample_rgb", "sample_slice",
]
