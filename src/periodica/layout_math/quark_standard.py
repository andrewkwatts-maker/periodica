"""
Standard Model grid layout math.

Arranges particles in the classic Standard Model table format:
- 3 generations of quarks (up-type and down-type)
- 3 generations of leptons (charged and neutrinos)
- Gauge bosons and Higgs boson
"""

from __future__ import annotations


# Default configuration
_DEFAULTS = {
    "cell_size": 70,
    "cell_spacing": 10,
    "section_spacing": 30,
    "min_size": 45,
    "max_size": 120,
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
    cell_spacing: float | None = None,
    section_spacing: float | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
    margin_left: float | None = None,
    margin_right: float | None = None,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
) -> list[dict]:
    """
    Compute Standard Model grid positions for particles.

    Each item should have:
        - sm_row (int): row in SM grid (0-3), or -1 if not placed
        - sm_col (int): column in SM grid (0-4), or -1 if not placed
        - Name / Symbol (str): display labels
        - particle_type (str): for color encoding

    Returns list of dicts with keys:
        x, y, w, h, label, color_rgb, metadata
    """
    cs = cell_spacing if cell_spacing is not None else _DEFAULTS["cell_spacing"]
    ss = section_spacing if section_spacing is not None else _DEFAULTS["section_spacing"]
    mn = min_size if min_size is not None else _DEFAULTS["min_size"]
    mx = max_size if max_size is not None else _DEFAULTS["max_size"]
    ml = margin_left if margin_left is not None else _DEFAULTS["margin_left"]
    mr = margin_right if margin_right is not None else _DEFAULTS["margin_right"]
    mt = margin_top if margin_top is not None else _DEFAULTS["margin_top"]
    mb = margin_bottom if margin_bottom is not None else _DEFAULTS["margin_bottom"]

    cols = 5  # 3 generations + bosons + Higgs
    rows = 4  # up-type quarks, down-type quarks, charged leptons, neutrinos

    available_width = width - ml - mr
    available_height = height - mt - mb

    cell_size = min(
        (available_width - (cols + 1) * cs) / cols,
        (available_height - (rows + 1) * cs) / rows,
    )
    cell_size = max(mn, min(mx, cell_size))

    # Center the grid horizontally
    start_x = (width - (cols * cell_size + (cols - 1) * cs)) / 2 + cell_size / 2
    start_y = mt + cell_size / 2

    results = []
    non_sm = []

    for item in items:
        sm_row = item.get("sm_row", -1)
        sm_col = item.get("sm_col", -1)

        if sm_row >= 0 and sm_col >= 0:
            x = start_x + sm_col * (cell_size + cs)
            y = start_y + sm_row * (cell_size + cs)
            results.append({
                "x": x,
                "y": y,
                "w": cell_size,
                "h": cell_size,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": _particle_type_color(item.get("particle_type", "")),
                "metadata": {
                    "name": item.get("Name", ""),
                    "sm_row": sm_row,
                    "sm_col": sm_col,
                    "in_layout": True,
                    "section": _section_for(sm_row, sm_col),
                },
            })
        else:
            non_sm.append(item)

    # Position non-SM particles below the main grid
    if non_sm:
        small_size = cell_size * 0.8
        extra_start_y = start_y + rows * (cell_size + cs) + ss * 2
        for i, item in enumerate(non_sm):
            col = i % 6
            row = i // 6
            x = start_x + col * (small_size + cs)
            y = extra_start_y + row * (small_size + cs)
            results.append({
                "x": x,
                "y": y,
                "w": small_size,
                "h": small_size,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": _particle_type_color(item.get("particle_type", "")),
                "metadata": {
                    "name": item.get("Name", ""),
                    "in_layout": False,
                },
            })

    return results


def _section_for(sm_row: int, sm_col: int) -> str:
    if sm_col >= 3:
        return "bosons"
    if sm_row in (0, 1):
        return "quarks"
    return "leptons"


def _particle_type_color(ptype: str) -> tuple[int, int, int]:
    colors = {
        "quark": (230, 100, 100),
        "lepton": (100, 180, 230),
        "gauge_boson": (230, 180, 100),
        "scalar_boson": (180, 100, 230),
    }
    return colors.get(str(ptype).lower().replace(" ", "_"), (150, 150, 150))
