"""
Force Interaction Network layout math.

Clusters particles by their primary interaction force (Strong > EM > Weak > Gravity).
Within each cluster, particles are arranged in a circle.
"""

from __future__ import annotations

import math

_DEFAULT_CLUSTER_POSITIONS = {
    "Strong": (0.25, 0.3),
    "Electromagnetic": (0.75, 0.3),
    "Weak": (0.5, 0.7),
    "Gravitational": (0.5, 0.5),
}

_CLUSTER_COLORS = {
    "Strong": (255, 100, 100),
    "Electromagnetic": (100, 150, 255),
    "Weak": (255, 180, 100),
    "Gravitational": (100, 255, 150),
}

_DEFAULTS = {
    "min_size": 40,
    "max_size": 110,
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
    cluster_positions: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """
    Compute force-network cluster positions for particles.

    Each item should have:
        - InteractionForces (list[str]): e.g. ['Strong', 'Electromagnetic']

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    cpos = cluster_positions if cluster_positions is not None else _DEFAULT_CLUSTER_POSITIONS
    ml = _DEFAULTS["margin_left"]
    mr = _DEFAULTS["margin_right"]
    mt = _DEFAULTS["margin_top"]
    mb = _DEFAULTS["margin_bottom"]
    mn = _DEFAULTS["min_size"]
    mx = _DEFAULTS["max_size"]

    available_w = width - ml - mr
    available_h = height - mt - mb

    # Categorise by primary force
    clusters: dict[str, list[dict]] = {
        "Strong": [],
        "Electromagnetic": [],
        "Weak": [],
        "Gravitational": [],
    }

    for item in items:
        forces = item.get("InteractionForces", [])
        if "Strong" in forces:
            primary = "Strong"
        elif "Electromagnetic" in forces:
            primary = "Electromagnetic"
        elif "Weak" in forces:
            primary = "Weak"
        else:
            primary = "Gravitational"
        clusters[primary].append(item)
        item["_primary_force"] = primary
        item["_all_forces"] = forces if forces else ["Gravitational"]

    max_cluster = max((len(v) for v in clusters.values()), default=1)
    cluster_radius = min(available_w, available_h) * 0.2

    cell_size = min(mx, cluster_radius * 1.5 / math.sqrt(max_cluster)) if max_cluster > 0 else mn
    cell_size = max(mn, cell_size)

    center_x = width / 2
    center_y = height / 2

    results: list[dict] = []

    for force_name, group in clusters.items():
        if not group:
            continue

        pos = cpos.get(force_name, (0.5, 0.5))
        cx = center_x + (pos[0] - 0.5) * available_w * 0.8
        cy = mt - 20 + pos[1] * (available_h - 40)
        color = _CLUSTER_COLORS.get(force_name, (150, 150, 150))

        n = len(group)
        if n == 1:
            results.append({
                "x": cx,
                "y": cy,
                "w": cell_size,
                "h": cell_size,
                "label": group[0].get("Symbol", group[0].get("Name", "?")),
                "color_rgb": color,
                "metadata": {
                    "name": group[0].get("Name", ""),
                    "primary_force": force_name,
                    "cluster_center": (cx, cy),
                },
            })
        else:
            r = min(cluster_radius, cell_size * n / (2 * math.pi) + cell_size)
            for i, item in enumerate(group):
                angle = 2 * math.pi * i / n - math.pi / 2
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                results.append({
                    "x": x,
                    "y": y,
                    "w": cell_size,
                    "h": cell_size,
                    "label": item.get("Symbol", item.get("Name", "?")),
                    "color_rgb": color,
                    "metadata": {
                        "name": item.get("Name", ""),
                        "primary_force": force_name,
                        "cluster_center": (cx, cy),
                    },
                })

    return results
