"""
Circular (wedge) layout positioner.

Elements are arranged in concentric rings by period, each element
occupying an angular wedge.  The returned dicts include polar
coordinates (radii and angles) in addition to the standard x/y centre
point, so renderers can draw wedge shapes if desired.

No Qt / PySide6 imports -- pure Python + math only.
"""

from __future__ import annotations

import math
from .base import LayoutPositioner


class CircularPositioner(LayoutPositioner):
    """
    Circular / wedge periodic-table layout.

    Each period occupies a concentric ring.  Elements within a period
    are evenly spaced around the full circle.

    Parameters
    ----------
    start_angle : float
        Angle (radians) where the first element of each period is placed.
        Default is ``-pi/2`` (12 o'clock).
    margin : float
        Minimum pixel margin from the edge of the viewport to the
        outermost ring.  Default ``20``.
    inner_fraction : float
        Fraction of the available radius used as the innermost ring
        starting point.  Default ``0.12`` (12 %).
    ring_fill : float
        Fraction of the ring-to-ring spacing occupied by the wedge
        (the rest is gap).  Default ``0.9`` (90 %).
    """

    def __init__(
        self,
        start_angle: float = -math.pi / 2,
        margin: float = 20.0,
        inner_fraction: float = 0.12,
        ring_fill: float = 0.9,
    ) -> None:
        self._start_angle = start_angle
        self._margin = margin
        self._inner_fraction = inner_fraction
        self._ring_fill = ring_fill

    # ---- core API --------------------------------------------------------

    def compute_positions(
        self,
        items: list[dict],
        width: float,
        height: float,
    ) -> list[dict]:
        """
        Lay out *items* in concentric period rings.

        Each returned dict contains::

            x, y          -- Cartesian centre of the wedge (relative to
                             viewport origin, *not* to the ring centre)
            w, h          -- radial thickness of the wedge (r_outer - r_inner)
                             repeated in both dimensions for convenience
            r_inner       -- inner radius of the wedge
            r_outer       -- outer radius of the wedge
            angle_start   -- start angle in radians
            angle_end     -- end angle in radians
            angle_mid     -- midpoint angle
            center_x      -- x of the global ring centre
            center_y      -- y of the global ring centre
            label         -- element symbol
            metadata      -- original item dict
        """
        if not items:
            return []

        center_x = width / 2.0
        center_y = height / 2.0

        num_periods = max(item["period"] for item in items)
        available_radius = min(width, height) / 2.0 - self._margin

        base_radius = available_radius * self._inner_fraction
        ring_spacing = (available_radius - base_radius) / num_periods

        # Pre-compute period radii
        period_radii: list[tuple[float, float]] = []
        for pidx in range(num_periods):
            r_inner = base_radius + pidx * ring_spacing
            r_outer = r_inner + ring_spacing * self._ring_fill
            period_radii.append((r_inner, r_outer))

        # Group items by period for angular distribution
        items_by_period: dict[int, list[dict]] = {}
        for item in items:
            p = item["period"]
            items_by_period.setdefault(p, []).append(item)

        result: list[dict] = []
        for item in items:
            period = item["period"]
            pidx = period - 1
            r_inner, r_outer = period_radii[pidx]

            period_items = items_by_period[period]
            num_in_period = len(period_items)
            idx_in_period = period_items.index(item)

            angle_per_elem = (2.0 * math.pi) / num_in_period
            angle_start = self._start_angle + idx_in_period * angle_per_elem
            angle_end = angle_start + angle_per_elem
            angle_mid = (angle_start + angle_end) / 2.0

            r_mid = (r_inner + r_outer) / 2.0
            x = center_x + r_mid * math.cos(angle_mid)
            y = center_y + r_mid * math.sin(angle_mid)
            thickness = r_outer - r_inner

            result.append({
                "x": x,
                "y": y,
                "w": thickness,
                "h": thickness,
                "r_inner": r_inner,
                "r_outer": r_outer,
                "angle_start": angle_start,
                "angle_end": angle_end,
                "angle_mid": angle_mid,
                "center_x": center_x,
                "center_y": center_y,
                "label": item.get("symbol", ""),
                "metadata": item,
            })

        return result
