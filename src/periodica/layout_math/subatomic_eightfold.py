"""
Eightfold Way layout math for subatomic particles.

Plots particles on an Isospin-I3 (x) vs Hypercharge-Y (y) diagram,
the classic multiplet representation from the Eightfold Way.
"""

from __future__ import annotations

_DEFAULTS = {
    "card_width": 119,   # 140 * 0.85
    "card_height": 153,  # 180 * 0.85
    "card_spacing": 15,
    "header_height": 40,
    "plot_margin": 100,
    "isospin_range": [-1.5, 1.5],
    "strangeness_range": [-3, 0],
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    plot_margin: float | None = None,
) -> list[dict]:
    """
    Compute Eightfold Way (I3 vs Y) positions for subatomic particles.

    Each item should have:
        - Isospin_I3 (float)
        - Strangeness (int)
        - BaryonNumber_B (int)
        - _is_baryon (bool)
        - _is_meson (bool)
        - Name (str)

    Hypercharge Y = Strangeness + BaryonNumber.

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    pm = plot_margin if plot_margin is not None else _DEFAULTS["plot_margin"]
    cw = _DEFAULTS["card_width"]
    ch = _DEFAULTS["card_height"]
    hh = _DEFAULTS["header_height"]

    baryons = [p for p in items if p.get("_is_baryon", False)]
    mesons = [p for p in items if p.get("_is_meson", False)]
    all_particles = baryons + mesons

    if not all_particles:
        return []

    # Compute coordinates
    coords: list[tuple[dict, float, float]] = []
    for p in all_particles:
        i3 = p.get("Isospin_I3", 0)
        strangeness = p.get("Strangeness", 0)
        baryon_num = p.get("BaryonNumber_B", 0)
        hypercharge = strangeness + baryon_num
        coords.append((p, i3, hypercharge))

    i3_vals = [c[1] for c in coords]
    y_vals = [c[2] for c in coords]

    i3_min = min(i3_vals) if i3_vals else _DEFAULTS["isospin_range"][0]
    i3_max = max(i3_vals) if i3_vals else _DEFAULTS["isospin_range"][1]
    y_min = min(y_vals) if y_vals else _DEFAULTS["strangeness_range"][0]
    y_max = max(y_vals) if y_vals else _DEFAULTS["strangeness_range"][1]

    i3_range = max(i3_max - i3_min, 3)
    y_range = max(y_max - y_min, 4)

    plot_left = pm + 50
    plot_right = width - pm
    plot_top = hh + 80
    plot_bottom = plot_top + 500
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    def coord_to_px(i3, y):
        x_norm = (i3 - (i3_min - 0.5)) / (i3_range + 1)
        y_norm = (y - (y_min - 0.5)) / (y_range + 1)
        px = plot_left + x_norm * plot_w
        py = plot_bottom - y_norm * plot_h
        return px, py

    overlap: dict[tuple[int, int], int] = {}
    results: list[dict] = []

    for p, i3, hypercharge in coords:
        px, py = coord_to_px(i3, hypercharge)
        px -= cw / 2
        py -= ch / 2

        key = (round(i3 * 2), round(hypercharge * 2))
        if key in overlap:
            px += overlap[key] * 25
            overlap[key] += 1
        else:
            overlap[key] = 1

        is_baryon = p.get("_is_baryon", False)
        color = (102, 126, 234) if is_baryon else (240, 147, 251)
        multiplet = _determine_multiplet(p)

        results.append({
            "x": px,
            "y": py,
            "w": cw,
            "h": ch,
            "label": p.get("Symbol", p.get("Name", "?")),
            "color_rgb": color,
            "metadata": {
                "name": p.get("Name", ""),
                "i3": i3,
                "hypercharge": hypercharge,
                "multiplet": multiplet,
            },
        })

    return results


def _determine_multiplet(particle: dict) -> str:
    classification = particle.get("Classification", [])
    cl_lower = str(classification).lower()
    if "delta" in cl_lower or "omega" in cl_lower:
        return "decuplet"
    if particle.get("_is_baryon"):
        return "baryon_octet"
    if particle.get("_is_meson"):
        return "meson_octet"
    return "other"
