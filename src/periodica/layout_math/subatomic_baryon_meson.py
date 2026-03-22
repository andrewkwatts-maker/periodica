"""
Baryon/Meson split layout math for subatomic particles.

Two sections stacked vertically:
  Top: Baryons (sorted by mass)
  Bottom: Mesons (sorted by mass)
Each section is a simple grid.
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

_BARYON_COLOR = (102, 126, 234)
_MESON_COLOR = (240, 147, 251)


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    card_width: float | None = None,
    card_height: float | None = None,
    card_spacing: float | None = None,
    section_spacing: float | None = None,
) -> list[dict]:
    """
    Compute baryon/meson split positions for subatomic particles.

    Each item should have:
        - _is_baryon (bool)
        - _is_meson (bool)
        - Mass_MeVc2 (float)
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

    baryons = sorted(
        [p for p in items if p.get("_is_baryon", False)],
        key=lambda p: p.get("Mass_MeVc2", 0),
    )
    mesons = sorted(
        [p for p in items if p.get("_is_meson", False)],
        key=lambda p: p.get("Mass_MeVc2", 0),
    )

    available_w = width - ml - mr
    cols = max(1, int(available_w // (cw + cs)))

    y_offset = hh + 20
    x_start = ml
    results: list[dict] = []

    # Baryons
    if baryons:
        y_offset += hh  # header
        for i, p in enumerate(baryons):
            row = i // cols
            col = i % cols
            x = x_start + col * (cw + cs)
            y = y_offset + row * (ch + cs)
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": p.get("Symbol", p.get("Name", "?")),
                "color_rgb": _BARYON_COLOR,
                "metadata": {
                    "name": p.get("Name", ""),
                    "particle_class": "baryon",
                },
            })
        baryon_rows = (len(baryons) + cols - 1) // cols
        y_offset += baryon_rows * (ch + cs) + ss

    # Mesons
    if mesons:
        y_offset += hh  # header
        for i, p in enumerate(mesons):
            row = i // cols
            col = i % cols
            x = x_start + col * (cw + cs)
            y = y_offset + row * (ch + cs)
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": p.get("Symbol", p.get("Name", "?")),
                "color_rgb": _MESON_COLOR,
                "metadata": {
                    "name": p.get("Name", ""),
                    "particle_class": "meson",
                },
            })

    return results
