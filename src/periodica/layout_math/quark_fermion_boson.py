"""
Fermion/Boson split layout math.

Left hemisphere: fermions (half-integer spin)
Right hemisphere: bosons (integer spin)
Within each hemisphere, arranged in a grid sorted by mass (heaviest first).
"""

from __future__ import annotations

import math

_DEFAULTS = {
    "min_size": 40,
    "max_size": 110,
    "margin_left": 50,
    "margin_right": 50,
    "margin_top": 100,
    "margin_bottom": 50,
    "section_spacing": 30,
    "inner_gap": 8,
}


def _is_fermion(item: dict) -> bool:
    spin = item.get("Spin_hbar", 0)
    if spin is None:
        spin = 0
    return (spin * 2) % 2 == 1


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
    section_spacing: float | None = None,
) -> list[dict]:
    """
    Compute fermion/boson split positions.

    Each item should have:
        - Spin_hbar (float)
        - Mass_MeVc2 (float)
        - particle_type (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    mt = margin_top if margin_top is not None else _DEFAULTS["margin_top"]
    mb = margin_bottom if margin_bottom is not None else _DEFAULTS["margin_bottom"]
    ss = section_spacing if section_spacing is not None else _DEFAULTS["section_spacing"]
    gap = _DEFAULTS["inner_gap"]
    mn = _DEFAULTS["min_size"]
    mx = _DEFAULTS["max_size"]

    fermions = []
    bosons = []
    for item in items:
        if _is_fermion(item):
            fermions.append(item)
        else:
            bosons.append(item)

    # Sort by mass descending
    fermions.sort(key=lambda p: p.get("Mass_MeVc2", 0) or 0, reverse=True)
    bosons.sort(key=lambda p: p.get("Mass_MeVc2", 0) or 0, reverse=True)

    available_w = width - _DEFAULTS["margin_left"] - _DEFAULTS["margin_right"]
    available_h = height - mt - mb - 80
    hemi_w = (available_w - ss * 2) / 2

    max_count = max(len(fermions), len(bosons), 1)
    cols_per = max(2, min(5, int(math.sqrt(max_count) + 0.5)))

    cell_size = min(
        mx,
        (hemi_w - 40) / cols_per,
        (available_h - 60) / (math.ceil(max_count / cols_per) + 1),
    )
    cell_size = max(mn, cell_size)

    results: list[dict] = []

    def _place_hemisphere(group, center_x, spin_type):
        n = len(group)
        cols = max(2, min(5, int(math.sqrt(n) + 0.5)))
        start_x = center_x - (cols * cell_size + (cols - 1) * gap) / 2 + cell_size / 2
        start_y = mt + 30 + cell_size / 2

        for i, item in enumerate(group):
            col = i % cols
            row = i // cols
            x = start_x + col * (cell_size + gap)
            y = start_y + row * (cell_size + gap)
            results.append({
                "x": x,
                "y": y,
                "w": cell_size,
                "h": cell_size,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": _type_color(item.get("particle_type", "")),
                "metadata": {
                    "name": item.get("Name", ""),
                    "spin_type": spin_type,
                    "hemisphere": spin_type,
                },
            })

    _place_hemisphere(fermions, width * 0.25, "fermion")
    _place_hemisphere(bosons, width * 0.75, "boson")
    return results


def _type_color(ptype: str) -> tuple[int, int, int]:
    colors = {
        "quark": (230, 100, 100),
        "lepton": (100, 180, 230),
        "gauge_boson": (230, 180, 100),
        "scalar_boson": (180, 100, 230),
        "antiparticle": (180, 180, 180),
        "composite": (100, 200, 150),
    }
    return colors.get(str(ptype).lower().replace(" ", "_"), (150, 150, 150))
