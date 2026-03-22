"""
Quark composition tree layout math for subatomic particles.

Hierarchical tree with levels:
  Level 1: Light hadrons (u, d quarks only)
  Level 2: Strange hadrons (contain s quark)
  Level 3: Charm hadrons (contain c quark)
  Level 4: Bottom hadrons (contain b quark)

Within each level, baryons precede mesons, sorted by mass.
"""

from __future__ import annotations

_DEFAULTS = {
    "card_width": 130,   # 140 * 0.93
    "card_height": 160,  # 180 * 0.89
    "card_spacing": 20,
    "header_height": 40,
    "margin_left": 50,
    "margin_right": 50,
    "level_spacing": 220,
}

_LEVEL_COLORS = {
    "light": (255, 100, 100),
    "strange": (100, 255, 100),
    "charm": (255, 200, 100),
    "bottom": (200, 100, 255),
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    level_spacing: float | None = None,
) -> list[dict]:
    """
    Compute quark-tree positions for subatomic particles.

    Each item should have:
        - QuarkContent (str): e.g. 'uud'
        - Composition (list[dict]): with 'Constituent' keys
        - _is_baryon (bool)
        - _is_meson (bool)
        - Mass_MeVc2 (float)
        - Name (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    ls = level_spacing if level_spacing is not None else _DEFAULTS["level_spacing"]
    cw = _DEFAULTS["card_width"]
    ch = _DEFAULTS["card_height"]
    cs = _DEFAULTS["card_spacing"]
    hh = _DEFAULTS["header_height"]
    ml = _DEFAULTS["margin_left"]
    mr = _DEFAULTS["margin_right"]

    available_w = width - ml - mr
    cols = max(1, int(available_w // (cw + cs)))

    # Categorise
    light, strange, charm, bottom = [], [], [], []
    for p in items:
        qc = p.get("QuarkContent", "").lower()
        comp = p.get("Composition", [])
        has_b = "b" in qc or any("bottom" in c.get("Constituent", "").lower() for c in comp)
        has_c = "c" in qc or any("charm" in c.get("Constituent", "").lower() for c in comp)
        has_s = "s" in qc or any("strange" in c.get("Constituent", "").lower() for c in comp)

        if has_b:
            bottom.append(p)
        elif has_c:
            charm.append(p)
        elif has_s:
            strange.append(p)
        else:
            light.append(p)

    levels = [
        ("light", "LIGHT HADRONS (u, d quarks)", light),
        ("strange", "STRANGE HADRONS (contains s quark)", strange),
        ("charm", "CHARM HADRONS (contains c quark)", charm),
        ("bottom", "BOTTOM HADRONS (contains b quark)", bottom),
    ]

    results: list[dict] = []
    tree_connections: list[dict] = []
    y_offset = 30 + hh + 60  # header space
    prev_level_name = None
    prev_level_bottom_y = None

    for level_name, label, group in levels:
        y_offset += 40  # level header space
        color = _LEVEL_COLORS.get(level_name, (150, 150, 150))

        # Sort: baryons first then mesons, each by mass
        baryons = sorted([p for p in group if p.get("_is_baryon")], key=lambda p: p.get("Mass_MeVc2", 0))
        mesons = sorted([p for p in group if p.get("_is_meson")], key=lambda p: p.get("Mass_MeVc2", 0))
        ordered = baryons + mesons

        level_start_y = y_offset
        for i, p in enumerate(ordered):
            row = i // cols
            col = i % cols
            x = ml + col * (cw + cs)
            y = y_offset + row * (ch + cs)
            ptype = "baryon" if p.get("_is_baryon") else "meson"
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": p.get("Symbol", p.get("Name", "?")),
                "color_rgb": color,
                "metadata": {
                    "name": p.get("Name", ""),
                    "tree_level": level_name,
                    "particle_type": ptype,
                },
            })

        level_rows = max(1, (len(ordered) + cols - 1) // cols) if ordered else 0
        level_bottom = y_offset + level_rows * (ch + cs)

        # Record connection from previous level
        if prev_level_name and ordered:
            quark_labels = {"strange": "+s quark", "charm": "+c quark", "bottom": "+b quark"}
            tree_connections.append({
                "from_level": prev_level_name,
                "to_level": level_name,
                "from_y": prev_level_bottom_y,
                "to_y": level_start_y - 40,
                "label": quark_labels.get(level_name, ""),
            })

        if ordered:
            prev_level_name = level_name
            prev_level_bottom_y = level_bottom

        y_offset = level_bottom + ls

    # Attach tree connections to first result
    if results:
        results[0]["metadata"]["_tree_connections"] = tree_connections

    return results


def get_tree_connections(results: list[dict]) -> list[dict]:
    """Extract tree connections from compute_positions output."""
    for r in results:
        conns = r.get("metadata", {}).get("_tree_connections")
        if conns is not None:
            return conns
    return []
