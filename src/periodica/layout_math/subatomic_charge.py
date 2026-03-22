"""
Charge-grouped layout math for subatomic particles.

Groups particles by electric charge and arranges each group in a grid,
with groups stacked vertically in descending charge order.
"""

from __future__ import annotations

_DEFAULTS = {
    "card_width": 140,
    "card_height": 180,
    "card_spacing": 20,
    "section_spacing": 60,
    "header_height": 40,
    "margin_left": 50,
    "margin_right": 50,
}

_DEFAULT_CHARGE_ORDER = [2, 1, 0, -1, -2]

_CHARGE_COLORS = {
    2: (255, 100, 100),
    1: (255, 183, 77),
    0: (200, 200, 200),
    -1: (100, 181, 246),
    -2: (156, 39, 176),
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    card_width: float | None = None,
    card_height: float | None = None,
    card_spacing: float | None = None,
    section_spacing: float | None = None,
    charge_order: list[int] | None = None,
) -> list[dict]:
    """
    Compute charge-grouped grid positions for subatomic particles.

    Each item should have:
        - Charge_e (float/int)
        - Mass_MeVc2 (float)  -- used for sort within group
        - Name (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    cw = card_width if card_width is not None else _DEFAULTS["card_width"]
    ch = card_height if card_height is not None else _DEFAULTS["card_height"]
    cs = card_spacing if card_spacing is not None else _DEFAULTS["card_spacing"]
    ss = section_spacing if section_spacing is not None else _DEFAULTS["section_spacing"]
    hh = _DEFAULTS["header_height"]
    ml = _DEFAULTS["margin_left"]
    mr = _DEFAULTS["margin_right"]
    order = charge_order if charge_order is not None else _DEFAULT_CHARGE_ORDER

    available_w = width - ml - mr
    cols = max(1, int(available_w // (cw + cs)))

    # Group by charge
    charge_groups: dict[float, list[dict]] = {}
    for item in items:
        charge = item.get("Charge_e", 0)
        charge_groups.setdefault(charge, []).append(item)

    # Order groups
    ordered = [c for c in order if c in charge_groups]
    remaining = sorted([c for c in charge_groups if c not in ordered], reverse=True)
    all_charges = ordered + remaining

    y_offset = hh + 20
    x_start = ml
    results: list[dict] = []

    for charge in all_charges:
        group = charge_groups[charge]
        color = _CHARGE_COLORS.get(int(charge), (150, 150, 150))

        y_offset += hh  # section header space

        group_sorted = sorted(group, key=lambda p: p.get("Mass_MeVc2", 0))
        for i, item in enumerate(group_sorted):
            row = i // cols
            col = i % cols
            x = x_start + col * (cw + cs)
            y = y_offset + row * (ch + cs)

            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": color,
                "metadata": {
                    "name": item.get("Name", ""),
                    "charge_group": charge,
                },
            })

        group_rows = (len(group) + cols - 1) // cols
        y_offset += group_rows * (ch + cs) + ss

    return results
