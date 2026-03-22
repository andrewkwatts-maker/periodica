"""
Decay chain / stability layout math for subatomic particles.

Groups particles by stability (Stable, Long-lived, Medium, Short-lived),
arranges each group in a grid, and records decay-arrow relationships.
"""

from __future__ import annotations

_DEFAULTS = {
    "card_width": 140,
    "card_height": 180,
    "card_spacing": 30,
    "section_spacing": 60,
    "header_height": 40,
    "margin_left": 50,
    "margin_right": 50,
}

_DEFAULT_THRESHOLDS = {
    "long_lived": 1e-6,
    "short_lived": 1e-12,
}

_GROUP_COLORS = {
    "stable": (100, 255, 100),
    "long": (200, 255, 100),
    "medium": (255, 255, 100),
    "short": (255, 150, 100),
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    card_width: float | None = None,
    card_height: float | None = None,
    card_spacing: float | None = None,
    long_lived_threshold: float | None = None,
    short_lived_threshold: float | None = None,
) -> list[dict]:
    """
    Compute stability-grouped positions for subatomic particles.

    Each item should have:
        - Stability (str): 'Stable' or other
        - HalfLife_s (float | None)
        - _stability_factor (float)
        - DecayProducts (list[str], optional)
        - Name (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}

    The metadata includes a top-level 'decay_arrows' list with
    {'from': name, 'to': product_name} entries.
    """
    cw = card_width if card_width is not None else _DEFAULTS["card_width"]
    ch = card_height if card_height is not None else _DEFAULTS["card_height"]
    cs = card_spacing if card_spacing is not None else _DEFAULTS["card_spacing"]
    ss = _DEFAULTS["section_spacing"]
    hh = _DEFAULTS["header_height"]
    ml = _DEFAULTS["margin_left"]
    mr = _DEFAULTS["margin_right"]
    ll = long_lived_threshold if long_lived_threshold is not None else _DEFAULT_THRESHOLDS["long_lived"]
    sl = short_lived_threshold if short_lived_threshold is not None else _DEFAULT_THRESHOLDS["short_lived"]

    available_w = width - ml - mr
    cols = max(1, int(available_w // (cw + cs)))

    sorted_items = sorted(items, key=lambda p: -p.get("_stability_factor", 0))

    def _is_stable(p):
        return p.get("Stability") == "Stable"

    def _is_long(p):
        return not _is_stable(p) and (p.get("HalfLife_s") or 0) > ll

    def _is_medium(p):
        hl = p.get("HalfLife_s") or 0
        return hl and sl <= hl <= ll

    def _is_short(p):
        hl = p.get("HalfLife_s") or 0
        return hl and hl < sl

    groups = [
        ("stable", "Stable Particles", _is_stable),
        ("long", "Long-lived", _is_long),
        ("medium", "Medium", _is_medium),
        ("short", "Short-lived", _is_short),
    ]

    y_offset = hh + 20 + hh  # header
    x_start = ml
    results: list[dict] = []
    decay_arrows: list[dict] = []

    for gid, gname, filter_fn in groups:
        group = [p for p in sorted_items if filter_fn(p)]
        if not group:
            continue

        y_offset += 30  # sub-header
        color = _GROUP_COLORS.get(gid, (150, 150, 150))

        for i, item in enumerate(group):
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
                    "stability_group": gid,
                },
            })

            for product in item.get("DecayProducts", []):
                decay_arrows.append({"from": item["Name"], "to": product})

        group_rows = (len(group) + cols - 1) // cols
        y_offset += group_rows * (ch + cs) + ss

    # Attach arrows to first result's metadata (or return separately)
    for r in results:
        r["metadata"]["_decay_arrows"] = decay_arrows
        break  # only on first

    return results


def get_decay_arrows(results: list[dict]) -> list[dict]:
    """Extract decay arrows from compute_positions output."""
    for r in results:
        arrows = r.get("metadata", {}).get("_decay_arrows")
        if arrows is not None:
            return arrows
    return []
