"""
Circular layout math for particles.

Arranges particles in concentric rings:
- Center: Higgs boson
- Inner ring: Gauge bosons
- Middle ring: Quarks
- Outer ring: Leptons
"""

from __future__ import annotations

import math

# Default ring radius ratios (fraction of max_radius)
_DEFAULT_RING_RATIOS = [0.25, 0.55, 0.85]

_DEFAULTS = {
    "min_size": 45,
    "max_size": 110,
    "margin": 50,
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    ring_ratios: list[float] | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
    margin: float | None = None,
) -> list[dict]:
    """
    Compute concentric-ring positions for particles.

    Each item should have:
        - Name (str)
        - particle_type (str): one of 'quark', 'lepton', 'gauge_boson', 'scalar_boson'

    Items whose Name contains 'higgs' (case-insensitive) are placed at center.

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    ratios = ring_ratios if ring_ratios is not None else _DEFAULT_RING_RATIOS
    mn = min_size if min_size is not None else _DEFAULTS["min_size"]
    mx = max_size if max_size is not None else _DEFAULTS["max_size"]
    mg = margin if margin is not None else _DEFAULTS["margin"]

    center_x = width / 2
    center_y = height / 2
    max_radius = min(width, height) / 2 - mg - 10

    cell_size = min(mx - 10, max_radius / 5)
    cell_size = max(mn, cell_size)

    inner_radius = max_radius * ratios[0]
    middle_radius = max_radius * ratios[1]
    outer_radius = max_radius * ratios[2]

    # Categorise
    higgs = []
    gauge_bosons = []
    quarks = []
    leptons = []
    others = []

    for item in items:
        name_lower = item.get("Name", "").lower()
        ptype = str(item.get("particle_type", "")).lower().replace(" ", "_")
        if "higgs" in name_lower:
            higgs.append(item)
        elif ptype == "gauge_boson":
            gauge_bosons.append(item)
        elif ptype == "quark":
            quarks.append(item)
        elif ptype == "lepton":
            leptons.append(item)
        else:
            others.append(item)

    results: list[dict] = []

    # Higgs at center
    for item in higgs:
        results.append({
            "x": center_x,
            "y": center_y,
            "w": cell_size * 1.2,
            "h": cell_size * 1.2,
            "label": item.get("Symbol", item.get("Name", "H")),
            "color_rgb": (180, 100, 230),
            "metadata": {"name": item.get("Name", ""), "ring": "center"},
        })

    def _place_ring(group, radius, ring_name, color_rgb):
        n = len(group)
        if n == 0:
            return
        angle_step = 2 * math.pi / n
        start_angle = -math.pi / 2
        for i, item in enumerate(group):
            angle = start_angle + i * angle_step
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            results.append({
                "x": x,
                "y": y,
                "w": cell_size,
                "h": cell_size,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": color_rgb,
                "metadata": {
                    "name": item.get("Name", ""),
                    "ring": ring_name,
                    "angle": angle,
                    "radius": radius,
                },
            })

    _place_ring(gauge_bosons, inner_radius, "inner", (230, 180, 100))
    _place_ring(quarks, middle_radius, "middle", (230, 100, 100))
    _place_ring(leptons, outer_radius, "outer", (100, 180, 230))
    if others:
        _place_ring(others, max_radius * 1.05, "extra", (150, 150, 150))

    return results
