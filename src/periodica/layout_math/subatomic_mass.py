"""
Mass-ordered layout math for subatomic particles.

Sorts particles by mass and arranges them in a grid, lightest to heaviest.
"""

from __future__ import annotations

_DEFAULTS = {
    "card_width": 140,
    "card_height": 180,
    "card_spacing": 20,
    "header_height": 40,
    "margin_left": 50,
    "margin_right": 50,
    "margin_top": 100,
}

_DEFAULT_MASS_RANGES = [
    {"min": 0, "max": 200, "label": "Light (< 200 MeV)"},
    {"min": 200, "max": 1000, "label": "Medium (200-1000 MeV)"},
    {"min": 1000, "max": 2000, "label": "Heavy (1-2 GeV)"},
    {"min": 2000, "max": float("inf"), "label": "Very Heavy (> 2 GeV)"},
]


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    card_width: float | None = None,
    card_height: float | None = None,
    card_spacing: float | None = None,
    margin_left: float | None = None,
    margin_right: float | None = None,
    margin_top: float | None = None,
    mass_ranges: list[dict] | None = None,
) -> list[dict]:
    """
    Compute mass-ordered grid positions for subatomic particles.

    Each item should have:
        - Mass_MeVc2 (float)
        - Name (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    cw = card_width if card_width is not None else _DEFAULTS["card_width"]
    ch = card_height if card_height is not None else _DEFAULTS["card_height"]
    cs = card_spacing if card_spacing is not None else _DEFAULTS["card_spacing"]
    ml = margin_left if margin_left is not None else _DEFAULTS["margin_left"]
    mr = margin_right if margin_right is not None else _DEFAULTS["margin_right"]
    mt = margin_top if margin_top is not None else _DEFAULTS["margin_top"]
    hh = _DEFAULTS["header_height"]
    ranges = mass_ranges if mass_ranges is not None else _DEFAULT_MASS_RANGES

    available_w = width - ml - mr
    cols = max(1, int(available_w // (cw + cs)))

    sorted_items = sorted(items, key=lambda p: p.get("Mass_MeVc2", 0))

    y_offset = hh + 20 + hh  # header space
    x_start = ml

    results: list[dict] = []
    for i, item in enumerate(sorted_items):
        row = i // cols
        col = i % cols
        x = x_start + col * (cw + cs)
        y = y_offset + row * (ch + cs)

        mass = item.get("Mass_MeVc2", 0)
        mass_label = _mass_range_label(mass, ranges)

        results.append({
            "x": x,
            "y": y,
            "w": cw,
            "h": ch,
            "label": item.get("Symbol", item.get("Name", "?")),
            "color_rgb": (255, 183, 77),
            "metadata": {
                "name": item.get("Name", ""),
                "mass_rank": i + 1,
                "mass_range": mass_label,
            },
        })

    return results


def _mass_range_label(mass: float, ranges: list[dict]) -> str:
    for r in ranges:
        if r.get("min", 0) <= mass < r.get("max", float("inf")):
            return r.get("label", "")
    return ""
