"""
Spiral layout positioner.

Elements are placed on concentric period circles, connected by a spiral
that winds through all elements in Z order.  Isotope positions are
computed as radial offsets from the period circle.

No Qt / PySide6 imports -- pure Python + math only.
"""

from __future__ import annotations

import math
from .base import LayoutPositioner


class SpiralPositioner(LayoutPositioner):
    """
    Spiral periodic-table layout.

    Each element sits on its period circle at a continuously advancing
    angle.  The spiral makes ``total_turns`` full rotations across all
    elements.

    Isotope data, if present on each item under the key ``"isotopes"``
    (a list of ``(mass_number, abundance_percent)`` tuples), is resolved
    into radial offsets stored per-element in the output.

    Parameters
    ----------
    margin : float
        Pixel margin around the viewport edge.  Default ``50``.
    inner_fraction : float
        Fraction of available radius used as the innermost period
        circle.  Default ``0.18``.
    total_turns : float
        Number of full rotations the spiral makes over all elements.
        Default ``4.0``.
    """

    def __init__(
        self,
        margin: float = 50.0,
        inner_fraction: float = 0.18,
        total_turns: float = 4.0,
    ) -> None:
        self._margin = margin
        self._inner_fraction = inner_fraction
        self._total_turns = total_turns

    # ---- core API --------------------------------------------------------

    def compute_positions(
        self,
        items: list[dict],
        width: float,
        height: float,
    ) -> list[dict]:
        """
        Lay out *items* on a spiral.

        Each returned dict contains::

            x, y            -- Cartesian position on the period circle
            w, h            -- nominal element size (ring_spacing)
            angle           -- spiral angle (radians)
            radius          -- distance from centre (period circle radius)
            base_radius     -- same as radius (alias kept for symmetry)
            ring_spacing    -- radial gap between period circles
            center_x, center_y -- spiral centre
            isotope_positions -- list[dict] with per-isotope radial data
            label           -- element symbol
            metadata        -- original item dict
        """
        if not items:
            return []

        usable_w = width - 2.0 * self._margin
        usable_h = height - 2.0 * self._margin

        num_periods = max(item["period"] for item in items)
        available_radius = min(usable_w, usable_h) / 2.0 - 50.0
        base_radius = available_radius * self._inner_fraction
        ring_spacing = (
            (available_radius - base_radius) / (num_periods - 1)
            if num_periods > 1
            else 0.0
        )

        # Period circle radii (1-indexed)
        period_radii: dict[int, float] = {}
        for p in range(1, num_periods + 1):
            period_radii[p] = base_radius + (p - 1) * ring_spacing

        center_x = usable_w / 2.0 + self._margin
        center_y = usable_h / 2.0 + self._margin

        # Global neutron-offset range (for consistent isotope radial scaling)
        global_min_offset = 0.0
        global_max_offset = 0.0
        for item in items:
            z = item["z"]
            for mass, _abundance in item.get("isotopes", []):
                neutron_count = mass - z
                neutron_delta = neutron_count - z
                global_min_offset = min(global_min_offset, neutron_delta)
                global_max_offset = max(global_max_offset, neutron_delta)

        offset_range = max(abs(global_min_offset), abs(global_max_offset))
        if offset_range == 0:
            offset_range = 1.0

        # Angular spacing: total_turns full rotations over all elements
        total_elements = len(items)
        angular_step = (2.0 * math.pi * self._total_turns) / max(total_elements, 1)

        result: list[dict] = []
        current_angle = 0.0

        for elem_idx, item in enumerate(items):
            period = item["period"]
            pr = period_radii[period]
            angle = current_angle
            current_angle += angular_step

            x = center_x + pr * math.cos(angle)
            y = center_y + pr * math.sin(angle)

            # Build isotope sub-positions
            isotope_positions: list[dict] = []
            isotopes = item.get("isotopes", [])
            if not isotopes:
                # Fallback: synthesise a single default isotope
                isotopes = [(item["z"] * 2, 100.0)]

            z = item["z"]
            for iso_idx, (mass, abundance) in enumerate(isotopes):
                neutron_count = mass - z
                neutron_delta = neutron_count - z
                radius_offset = (neutron_delta / offset_range) * (ring_spacing / 2.0)
                iso_r = pr + radius_offset

                isotope_positions.append({
                    "angle": angle,
                    "radius": iso_r,
                    "base_radius": pr,
                    "mass": mass,
                    "abundance": abundance,
                    "neutron_count": neutron_count,
                    "neutron_delta": neutron_delta,
                    "isotope_index": iso_idx,
                    "x": center_x + iso_r * math.cos(angle),
                    "y": center_y + iso_r * math.sin(angle),
                })

            result.append({
                "x": x,
                "y": y,
                "w": ring_spacing,
                "h": ring_spacing,
                "angle": angle,
                "radius": pr,
                "base_radius": pr,
                "ring_spacing": ring_spacing,
                "center_x": center_x,
                "center_y": center_y,
                "isotope_positions": isotope_positions,
                "element_index": elem_idx,
                "label": item.get("symbol", ""),
                "metadata": item,
            })

        return result
