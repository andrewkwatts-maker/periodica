"""
Discovery timeline layout math for subatomic particles.

X-axis: discovery year
Y-axis: log(mass) -- showing progression of accessible energies
Particles without a discovery date go in a separate grid section below.
"""

from __future__ import annotations

import math

_DEFAULTS = {
    "card_width": 119,   # 140 * 0.85
    "card_height": 149,  # 180 * 0.83
    "card_spacing": 15,
    "header_height": 40,
    "timeline_height": 60,
    "margin_left": 50,
    "margin_right": 50,
    "timeline_margin": 100,
    "year_min": 1895,
    "year_max": 2020,
}

_DEFAULT_ERA_BOUNDARIES = {
    "classical": 1932,
    "nuclear": 1947,
    "strange": 1964,
    "quark_model": 1995,
}


def compute_positions(
    items: list[dict],
    width: float,
    height: float,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    era_boundaries: dict[str, int] | None = None,
) -> list[dict]:
    """
    Compute discovery-timeline positions for subatomic particles.

    Each item should have:
        - Discovery (dict with 'Year' key, optional 'Location')
        - Mass_MeVc2 (float)
        - Name (str)

    Returns list of dicts: {x, y, w, h, label, color_rgb, metadata}
    """
    eras = era_boundaries if era_boundaries is not None else _DEFAULT_ERA_BOUNDARIES
    cw = _DEFAULTS["card_width"]
    ch = _DEFAULTS["card_height"]
    cs = _DEFAULTS["card_spacing"]
    hh = _DEFAULTS["header_height"]
    ml = _DEFAULTS["margin_left"]
    mr = _DEFAULTS["margin_right"]
    tmg = _DEFAULTS["timeline_margin"]

    # Separate particles with/without dates
    with_date: list[tuple[dict, int]] = []
    without_date: list[dict] = []
    for item in items:
        disc = item.get("Discovery", {})
        yr = disc.get("Year") if isinstance(disc, dict) else None
        if yr:
            with_date.append((item, yr))
        else:
            without_date.append(item)

    with_date.sort(key=lambda x: x[1])

    # Timeline bounds
    timeline_left = tmg
    timeline_right = width - tmg
    timeline_width = timeline_right - timeline_left

    if with_date:
        actual_min = min(d[1] for d in with_date)
        actual_max = max(d[1] for d in with_date)
        yr_min = max(1890, actual_min - 5)
        yr_max = min(2025, actual_max + 5)
    else:
        yr_min = year_min if year_min is not None else _DEFAULTS["year_min"]
        yr_max = year_max if year_max is not None else _DEFAULTS["year_max"]

    # Mass range for Y axis
    masses = [p.get("Mass_MeVc2", 0) for p, _ in with_date]
    pos_masses = [m for m in masses if m > 0]
    if pos_masses:
        log_m_min = math.log10(min(pos_masses))
        log_m_max = math.log10(max(pos_masses))
    else:
        log_m_min, log_m_max = 0, 4

    plot_top = hh + 80 + _DEFAULTS["timeline_height"] + 20
    plot_height = 400

    def year_to_x(year):
        yr = max(yr_min, min(yr_max, year))
        return timeline_left + (yr - yr_min) / (yr_max - yr_min) * timeline_width

    def mass_to_y(mass):
        if mass <= 0:
            return plot_top + plot_height - 50
        log_m = math.log10(mass)
        if log_m_max == log_m_min:
            norm = 0.5
        else:
            norm = (log_m - log_m_min) / (log_m_max - log_m_min)
        return plot_top + plot_height - norm * (plot_height - 100)

    results: list[dict] = []
    grid: dict[tuple[int, int], int] = {}

    for item, year in with_date:
        mass = item.get("Mass_MeVc2", 100)
        x = year_to_x(year) - cw / 2
        y = mass_to_y(mass) - ch / 2

        key = (round(x / 80), round(y / 100))
        if key in grid:
            off = grid[key] * 30
            x += off % 60
            y += (off // 2) * 20
            grid[key] += 1
        else:
            grid[key] = 1

        disc = item.get("Discovery", {})
        location = disc.get("Location", "Unknown") if isinstance(disc, dict) else "Unknown"

        results.append({
            "x": x,
            "y": y,
            "w": cw,
            "h": ch,
            "label": item.get("Symbol", item.get("Name", "?")),
            "color_rgb": (255, 152, 0),
            "metadata": {
                "name": item.get("Name", ""),
                "discovery_year": year,
                "discovery_location": location,
                "era": _get_era(year, eras),
            },
        })

    # Particles without discovery date
    if without_date:
        unknown_y = plot_top + plot_height + 115
        available_w = width - ml - mr
        cols = max(1, int(available_w // (cw + cs)))

        for i, item in enumerate(without_date):
            row = i // cols
            col = i % cols
            x = ml + col * (cw + cs)
            y = unknown_y + row * (ch + cs)
            results.append({
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "label": item.get("Symbol", item.get("Name", "?")),
                "color_rgb": (150, 150, 150),
                "metadata": {
                    "name": item.get("Name", ""),
                    "discovery_year": None,
                    "era": "unknown",
                },
            })

    return results


def _get_era(year: int, boundaries: dict[str, int]) -> str:
    classical = boundaries.get("classical", 1932)
    nuclear = boundaries.get("nuclear", 1947)
    strange = boundaries.get("strange", 1964)
    quark_model = boundaries.get("quark_model", 1995)
    if year < classical:
        return "classical"
    elif year < nuclear:
        return "nuclear"
    elif year < strange:
        return "strange"
    elif year < quark_model:
        return "quark_model"
    return "modern"


def get_era_bands(
    plot_top: float,
    plot_height: float,
    era_boundaries: dict[str, int] | None = None,
) -> list[dict]:
    """Return era band definitions for visual reference."""
    b = era_boundaries if era_boundaries is not None else _DEFAULT_ERA_BOUNDARIES
    return [
        {"name": "Classical Era", "start": 1895, "end": b.get("classical", 1932),
         "color_rgba": (100, 100, 150, 50), "y_start": plot_top, "y_end": plot_top + plot_height},
        {"name": "Nuclear Era", "start": b.get("classical", 1932), "end": b.get("nuclear", 1947),
         "color_rgba": (100, 150, 100, 50), "y_start": plot_top, "y_end": plot_top + plot_height},
        {"name": "Strange Particles", "start": b.get("nuclear", 1947), "end": b.get("strange", 1964),
         "color_rgba": (150, 100, 100, 50), "y_start": plot_top, "y_end": plot_top + plot_height},
        {"name": "Quark Model Era", "start": b.get("strange", 1964), "end": b.get("quark_model", 1995),
         "color_rgba": (150, 150, 100, 50), "y_start": plot_top, "y_end": plot_top + plot_height},
        {"name": "Modern Era", "start": b.get("quark_model", 1995), "end": 2020,
         "color_rgba": (100, 150, 150, 50), "y_start": plot_top, "y_end": plot_top + plot_height},
    ]
