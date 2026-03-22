"""
Linear layout math for particles.

Arranges particles in a horizontal or vertical line sorted by a property
(mass, charge, spin, generation, or name).
"""

from __future__ import annotations

_DEFAULTS = {
    "min_size": 45,
    "max_size": 120,
    "cell_spacing": 15,
    "margin_left": 50,
    "margin_right": 50,
    "margin_top": 100,
    "margin_bottom": 50,
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    sort_property: str = "mass",
    orientation: str = "horizontal",
    cell_spacing: float | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
    margin_left: float | None = None,
    margin_right: float | None = None,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
) -> list[dict]:
    """
    Compute linear sorted positions for particles.

    Args:
        sort_property: 'mass', 'charge', 'spin', 'generation', or 'name'
        orientation: 'horizontal' or 'vertical'

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    cs = cell_spacing if cell_spacing is not None else _DEFAULTS["cell_spacing"]
    mn = min_size if min_size is not None else _DEFAULTS["min_size"]
    mx = max_size if max_size is not None else _DEFAULTS["max_size"]
    ml = margin_left if margin_left is not None else _DEFAULTS["margin_left"]
    mr = margin_right if margin_right is not None else _DEFAULTS["margin_right"]
    mt = margin_top if margin_top is not None else _DEFAULTS["margin_top"]
    mb = margin_bottom if margin_bottom is not None else _DEFAULTS["margin_bottom"]

    sorted_items = _sort_items(items, sort_property)
    n = len(sorted_items)
    if n == 0:
        return []

    results: list[dict] = []

    if orientation == "horizontal":
        available = width - ml - mr
        cell_size = min(mx, (available - (n - 1) * cs) / n)
        cell_size = max(mn, cell_size)

        total_w = n * cell_size + (n - 1) * cs
        start_x = (width - total_w) / 2 + cell_size / 2
        center_y = height / 2

        for i, item in enumerate(sorted_items):
            x = start_x + i * (cell_size + cs)
            results.append(_make_entry(item, x, center_y, cell_size, i, sort_property))
    else:
        available = height - mt - mb
        cell_size = min(mx, (available - (n - 1) * cs) / n)
        cell_size = max(mn - 10, cell_size)

        total_h = n * cell_size + (n - 1) * cs
        start_y = (height - total_h) / 2 + cell_size / 2
        center_x = width / 2

        for i, item in enumerate(sorted_items):
            y = start_y + i * (cell_size + cs)
            results.append(_make_entry(item, center_x, y, cell_size, i, sort_property))

    return results


def _sort_items(items: list[dict], prop: str) -> list[dict]:
    key_map = {
        "mass": lambda p: p.get("Mass_MeVc2", 0) or 0,
        "charge": lambda p: p.get("Charge_e", 0) or 0,
        "spin": lambda p: p.get("Spin_hbar", 0) or 0,
        "generation": lambda p: p.get("generation_num", -1),
    }
    key_fn = key_map.get(prop, lambda p: p.get("Name", ""))
    return sorted(items, key=key_fn)


def _make_entry(item, x, y, size, sort_index, sort_property):
    return {
        "x": x,
        "y": y,
        "w": size,
        "h": size,
        "label": item.get("Symbol", item.get("Name", "?")),
        "color_rgb": (150, 150, 150),
        "metadata": {
            "name": item.get("Name", ""),
            "sort_index": sort_index,
            "sort_property": sort_property,
        },
    }
