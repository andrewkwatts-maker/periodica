"""
Charge-Mass scatter plot layout math.

X-axis: electric charge
Y-axis: log(mass)
Jitter applied to separate overlapping particles.
"""

from __future__ import annotations

import math

_DEFAULTS = {
    "min_size": 35,
    "max_size": 110,
    "margin_left": 80,
    "margin_right": 40,
    "margin_top": 80,
    "margin_bottom": 100,
    "charge_min": -1.2,
    "charge_max": 1.2,
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    charge_min: float | None = None,
    charge_max: float | None = None,
    margin_left: float | None = None,
    margin_right: float | None = None,
    margin_top: float | None = None,
    margin_bottom: float | None = None,
) -> list[dict]:
    """
    Compute charge vs log-mass scatter positions.

    Each item should have:
        - Charge_e (float)
        - Mass_MeVc2 (float)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    cmin = charge_min if charge_min is not None else _DEFAULTS["charge_min"]
    cmax = charge_max if charge_max is not None else _DEFAULTS["charge_max"]
    ml = margin_left if margin_left is not None else _DEFAULTS["margin_left"]
    mr = margin_right if margin_right is not None else _DEFAULTS["margin_right"]
    mt = margin_top if margin_top is not None else _DEFAULTS["margin_top"]
    mb = margin_bottom if margin_bottom is not None else _DEFAULTS["margin_bottom"]

    plot_w = width - ml - mr
    plot_h = height - mt - mb

    # Log mass range
    masses = [item.get("Mass_MeVc2", 0) or 0.0001 for item in items]
    min_mass = max(0.0001, min(masses)) if masses else 0.0001
    max_mass = max(masses) if masses else 1
    log_min = math.log10(min_mass)
    log_max = math.log10(max_mass) if max_mass > min_mass else log_min + 1
    log_range = log_max - log_min
    log_min -= log_range * 0.1
    log_max += log_range * 0.1

    # Cell size
    n = len(items)
    mn = _DEFAULTS["min_size"]
    mx = _DEFAULTS["max_size"]
    cell_size = min(mx, max(mn, plot_w / (n ** 0.5) * 0.8)) if n > 0 else mn

    results: list[dict] = []
    for item in items:
        charge = item.get("Charge_e", 0) or 0
        mass = item.get("Mass_MeVc2", 0) or 0.0001

        charge_norm = (charge - cmin) / (cmax - cmin)
        x = ml + charge_norm * plot_w

        log_mass = math.log10(max(0.0001, mass))
        mass_norm = (log_mass - log_min) / (log_max - log_min)
        y = mt + (1 - mass_norm) * plot_h  # heavier is higher

        results.append({
            "x": x,
            "y": y,
            "w": cell_size,
            "h": cell_size,
            "label": item.get("Symbol", item.get("Name", "?")),
            "color_rgb": (150, 150, 150),
            "metadata": {
                "name": item.get("Name", ""),
                "grid_charge": charge,
                "grid_mass": mass,
                "grid_log_mass": log_mass,
            },
        })

    # Apply jitter to separate overlapping particles
    _apply_jitter(results, cell_size)
    return results


def _apply_jitter(results: list[dict], cell_size: float) -> None:
    """Separate overlapping particles with small circular offsets."""
    grid_size = cell_size * 0.8
    groups: dict[tuple[int, int], list[dict]] = {}

    for r in results:
        key = (int(r["x"] / grid_size), int(r["y"] / grid_size))
        groups.setdefault(key, []).append(r)

    for group in groups.values():
        if len(group) > 1:
            n = len(group)
            for i, r in enumerate(group):
                angle = 2 * math.pi * i / n
                dist = cell_size * 0.4
                r["x"] += dist * math.cos(angle)
                r["y"] += dist * math.sin(angle)
