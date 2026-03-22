"""
Mass Hierarchy Spiral layout math.

- Logarithmic spiral where radius = f(log(mass))
- Angular position by generation (1st/2nd/3rd gen, bosons)
- Display size proportional to spin value
"""

from __future__ import annotations

import math

_DEFAULTS = {
    "base_cell_size": 50,
    "min_size": 45,
    "max_size": 120,
    "margin": 50,
    "min_radius_ratio": 0.15,
    "max_radius_ratio": 0.85,
    "angular_spread_deg": 60,
}

# Base angles for each generation sector
_GEN_BASE_ANGLES = {
    0: -math.pi / 2,       # Top (bosons)
    1: 0,                   # Right
    2: 2 * math.pi / 3,    # Lower right
    3: 4 * math.pi / 3,    # Lower left
    -1: math.pi,            # Left (unknown)
}

_GEN_COLORS = {
    0: (180, 100, 230),     # Purple for bosons
    1: (100, 200, 100),     # Green for 1st gen
    2: (200, 200, 100),     # Yellow for 2nd gen
    3: (200, 100, 100),     # Red for 3rd gen
    -1: (150, 150, 150),    # Gray for unknown
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    min_radius_ratio: float | None = None,
    max_radius_ratio: float | None = None,
    angular_spread_deg: float | None = None,
    margin: float | None = None,
) -> list[dict]:
    """
    Compute mass-spiral positions for particles.

    Each item should have:
        - Mass_MeVc2 (float)
        - generation_num (int): 0=boson, 1/2/3=gen, -1=unknown
        - Spin_hbar (float)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    min_rr = min_radius_ratio if min_radius_ratio is not None else _DEFAULTS["min_radius_ratio"]
    max_rr = max_radius_ratio if max_radius_ratio is not None else _DEFAULTS["max_radius_ratio"]
    spread = math.radians(angular_spread_deg if angular_spread_deg is not None else _DEFAULTS["angular_spread_deg"])
    mg = margin if margin is not None else _DEFAULTS["margin"]

    center_x = width / 2
    center_y = height / 2
    max_radius = min(width, height) / 2 - mg - 30
    base_cell = _DEFAULTS["base_cell_size"]

    # Mass range for log normalization
    masses = [item.get("Mass_MeVc2", 0) or 0.001 for item in items]
    min_mass = max(0.001, min(masses)) if masses else 0.001
    max_mass = max(masses) if masses else 1
    log_min = math.log10(min_mass)
    log_max = math.log10(max_mass) if max_mass > min_mass else log_min + 1

    # Group by generation
    gen_groups: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: [], -1: []}
    for item in items:
        gen = item.get("generation_num", -1)
        gen_groups.setdefault(gen, gen_groups[-1]).append(item)

    results: list[dict] = []

    for gen, group in gen_groups.items():
        if not group:
            continue

        base_angle = _GEN_BASE_ANGLES.get(gen, math.pi)
        group.sort(key=lambda p: p.get("Mass_MeVc2", 0) or 0)
        n = len(group)

        for i, item in enumerate(group):
            mass = item.get("Mass_MeVc2", 0) or 0.001
            log_mass = math.log10(max(0.001, mass))

            # Normalize to radius
            if log_max > log_min:
                norm_mass = (log_mass - log_min) / (log_max - log_min)
            else:
                norm_mass = 0.5
            radius = max_radius * (min_rr + (max_rr - min_rr) * norm_mass)

            # Angle within generation sector
            if n > 1:
                angle_offset = (i / (n - 1) - 0.5) * spread
            else:
                angle_offset = 0
            angle = base_angle + angle_offset

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            # Size proportional to spin
            spin = item.get("Spin_hbar", 0.5) or 0.5
            size_factor = 0.7 + 0.6 * min(spin, 1.5)
            display_size = base_cell * size_factor

            color = _GEN_COLORS.get(gen, (150, 150, 150))

            results.append({
                "x": x,
                "y": y,
                "w": display_size,
                "h": display_size,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": color,
                "metadata": {
                    "name": item.get("Name", ""),
                    "spiral_radius": radius,
                    "spiral_angle": angle,
                    "generation_num": gen,
                },
            })

    return results
