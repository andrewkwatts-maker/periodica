"""
Alternative layout math -- groups particles by interaction force.

Groups:
- Strong (quarks & gluons)
- Electromagnetic (charged particles & photon)
- Weak Only (neutrinos, W/Z)
- Other (gravity only)

Groups arranged in a 2x2 (or 1xN) grid; particles in each group laid out
in a sub-grid.
"""

from __future__ import annotations

import math

_DEFAULTS = {
    "min_size": 45,
    "max_size": 110,
    "group_spacing": 40,
    "margin_left": 50,
    "margin_right": 50,
    "margin_top": 100,
    "margin_bottom": 50,
}

_GROUP_COLORS = {
    "Strong": (255, 100, 100),
    "Electromagnetic": (100, 150, 255),
    "Weak Only": (255, 200, 100),
    "Other": (150, 150, 150),
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    group_spacing: float | None = None,
) -> list[dict]:
    """
    Compute alternative (interaction-grouped) positions.

    Each item should have:
        - InteractionForces (list[str])

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    gs = group_spacing if group_spacing is not None else _DEFAULTS["group_spacing"]
    ml = _DEFAULTS["margin_left"]
    mr = _DEFAULTS["margin_right"]
    mt = _DEFAULTS["margin_top"]
    mb = _DEFAULTS["margin_bottom"]
    mn = _DEFAULTS["min_size"]
    mx = _DEFAULTS["max_size"]

    # Classify
    strong, em, weak_only, other = [], [], [], []
    for item in items:
        forces = item.get("InteractionForces", [])
        if "Strong" in forces:
            strong.append(item)
        elif "Electromagnetic" in forces:
            em.append(item)
        elif "Weak" in forces:
            weak_only.append(item)
        else:
            other.append(item)

    raw_groups = [
        ("Strong", strong),
        ("Electromagnetic", em),
        ("Weak Only", weak_only),
        ("Other", other),
    ]
    groups = [(name, parts) for name, parts in raw_groups if parts]
    if not groups:
        return []

    available_w = width - ml - mr
    available_h = height - mt - mb

    if len(groups) <= 2:
        gcols, grows = len(groups), 1
    else:
        gcols, grows = 2, math.ceil(len(groups) / 2)

    group_w = (available_w - (gcols - 1) * gs) / gcols
    group_h = (available_h - (grows - 1) * gs) / grows

    max_group_size = max(len(p) for _, p in groups)
    max_per_row = max(3, min(6, int(math.sqrt(max_group_size) + 1)))
    cell_size = min(mx - 10, (group_w - 40) / max_per_row, (group_h - 60) / max_per_row)
    cell_size = max(mn, cell_size)

    results: list[dict] = []

    for g_idx, (name, parts) in enumerate(groups):
        g_col = g_idx % gcols
        g_row = g_idx // gcols
        gx = ml + g_col * (group_w + gs)
        gy = mt + g_row * (group_h + gs)

        color = _GROUP_COLORS.get(name, (150, 150, 150))
        inner_cols = max(2, min(5, int(math.sqrt(len(parts)) + 0.5)))
        inner_start_x = gx + (group_w - inner_cols * cell_size) / 2 + cell_size / 2
        inner_start_y = gy + 50 + cell_size / 2

        for i, item in enumerate(parts):
            col = i % inner_cols
            row = i // inner_cols
            x = inner_start_x + col * (cell_size + 5)
            y = inner_start_y + row * (cell_size + 5)
            results.append({
                "x": x,
                "y": y,
                "w": cell_size,
                "h": cell_size,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": color,
                "metadata": {
                    "name": item.get("Name", ""),
                    "group_name": name,
                },
            })

    return results
