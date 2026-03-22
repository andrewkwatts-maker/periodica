"""
Lifetime spectrum layout math for subatomic particles.

Positions particles on a horizontal logarithmic timeline based on half-life.
Stable particles are placed at the far right.
Unstable baryons and mesons are separated into their own rows.
"""

from __future__ import annotations

import math

_DEFAULTS = {
    "card_width": 119,   # 140 * 0.85
    "card_height": 140,  # 180 * 0.78
    "card_spacing": 15,
    "header_height": 40,
    "section_spacing": 80,
    "log_min": -24,
    "log_max": 4,
    "timeline_margin": 100,
}

_TIME_LABELS = {
    0: "1 s", -3: "1 ms", -6: "1 us", -9: "1 ns",
    -12: "1 ps", -15: "1 fs", -18: "1 as", -21: "1 zs",
    -24: "1 ys", 3: "1000 s",
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    log_min: int | None = None,
    log_max: int | None = None,
    timeline_margin: float | None = None,
) -> list[dict]:
    """
    Compute lifetime-spectrum positions for subatomic particles.

    Each item should have:
        - HalfLife_s (float | None)
        - Stability (str): 'Stable' or other
        - _is_baryon (bool)
        - _is_meson (bool)
        - Name (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    lmin = log_min if log_min is not None else _DEFAULTS["log_min"]
    lmax = log_max if log_max is not None else _DEFAULTS["log_max"]
    tmg = timeline_margin if timeline_margin is not None else _DEFAULTS["timeline_margin"]

    cw = _DEFAULTS["card_width"]
    ch = _DEFAULTS["card_height"]
    hh = _DEFAULTS["header_height"]
    ss = _DEFAULTS["section_spacing"]

    timeline_left = tmg
    timeline_right = width - tmg
    timeline_width = timeline_right - timeline_left

    def hl_to_x(half_life_s):
        if half_life_s is None or half_life_s <= 0:
            return timeline_left
        log_hl = max(lmin, min(lmax, math.log10(half_life_s)))
        norm = (log_hl - lmin) / (lmax - lmin)
        return timeline_left + norm * timeline_width

    stable = [p for p in items if p.get("Stability") == "Stable"]
    unstable_baryons = [p for p in items if p.get("_is_baryon") and p.get("Stability") != "Stable"]
    unstable_mesons = [p for p in items if p.get("_is_meson") and p.get("Stability") != "Stable"]

    y_offset = hh + 20 + hh + 80  # header + timeline axis space
    results: list[dict] = []

    # Stable particles at far right
    if stable:
        y_offset += 35
        for i, p in enumerate(stable):
            x = timeline_right + 50
            y = y_offset + (i % 2) * (ch + 10)
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": p.get("Symbol", p.get("Name", "?")),
                "color_rgb": (0, 184, 148),
                "metadata": {
                    "name": p.get("Name", ""),
                    "lifetime_category": "stable",
                    "is_stable": True,
                },
            })
        stable_rows = (len(stable) + 1) // 2
        y_offset += stable_rows * (ch + 10) + ss

    # Unstable baryons
    if unstable_baryons:
        y_offset += 35
        sorted_b = sorted(unstable_baryons, key=lambda p: p.get("HalfLife_s") or 0)
        x_positions: dict[int, int] = {}
        for p in sorted_b:
            hl = p.get("HalfLife_s")
            x = hl_to_x(hl) - cw / 2
            x_key = round(x / 50) * 50
            if x_key in x_positions:
                y = y_offset + x_positions[x_key] * (ch + 10)
                x_positions[x_key] += 1
            else:
                y = y_offset
                x_positions[x_key] = 1

            log_hl = math.log10(hl) if hl and hl > 0 else None
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": p.get("Symbol", p.get("Name", "?")),
                "color_rgb": (102, 126, 234),
                "metadata": {
                    "name": p.get("Name", ""),
                    "lifetime_category": _categorize(hl),
                    "log_half_life": log_hl,
                },
            })
        max_rows = max(x_positions.values()) if x_positions else 1
        y_offset += max_rows * (ch + 10) + ss

    # Unstable mesons
    if unstable_mesons:
        y_offset += 35
        sorted_m = sorted(unstable_mesons, key=lambda p: p.get("HalfLife_s") or 0)
        x_positions = {}
        for p in sorted_m:
            hl = p.get("HalfLife_s")
            x = hl_to_x(hl) - cw / 2
            x_key = round(x / 50) * 50
            if x_key in x_positions:
                y = y_offset + x_positions[x_key] * (ch + 10)
                x_positions[x_key] += 1
            else:
                y = y_offset
                x_positions[x_key] = 1

            log_hl = math.log10(hl) if hl and hl > 0 else None
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": p.get("Symbol", p.get("Name", "?")),
                "color_rgb": (240, 147, 251),
                "metadata": {
                    "name": p.get("Name", ""),
                    "lifetime_category": _categorize(hl),
                    "log_half_life": log_hl,
                },
            })

    return results


def _categorize(half_life_s) -> str:
    if half_life_s is None:
        return "unknown"
    log_hl = math.log10(half_life_s) if half_life_s > 0 else -30
    if log_hl > 0:
        return "long_lived"
    elif log_hl > -9:
        return "medium"
    elif log_hl > -18:
        return "short"
    return "ultra_short"


def get_time_markers(log_min: int = -24, log_max: int = 4, step: int = 3) -> list[dict]:
    """Return timeline axis markers."""
    markers = []
    for exp in range(log_min, log_max + 1, step):
        label = _TIME_LABELS.get(exp, f"10^{exp} s")
        markers.append({"log_value": exp, "label": label})
    return markers
