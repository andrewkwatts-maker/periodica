"""
Linear layout positioner.

Elements are arranged in a single horizontal row, sorted by a
configurable property.  The returned positions include data for
drawing property trend lines above and below the element row.

No Qt / PySide6 imports -- pure Python + math only.
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass
from .base import LayoutPositioner


# ---------------------------------------------------------------------------
# Property configuration (framework-free mirror of the Qt-based originals)
# ---------------------------------------------------------------------------

class PropertyKey(Enum):
    """Keys for the properties that can drive visual encoding."""
    IONIZATION = "ionization"
    ELECTRONEGATIVITY = "electronegativity"
    RADIUS = "radius"
    MELTING = "melting"
    BOILING = "boiling"
    DENSITY = "density"
    ELECTRON_AFFINITY = "electron_affinity"
    VALENCE = "valence"


@dataclass(frozen=True)
class PropertyConfig:
    """Metadata for one property trend line."""
    key: PropertyKey
    label: str
    element_key: str          # dict key used to read the value from an item
    display_name: str         # human-readable name with units
    color_rgb: tuple[int, int, int]  # base colour as (r, g, b)


PROPERTY_CONFIGS: dict[PropertyKey, PropertyConfig] = {
    PropertyKey.IONIZATION: PropertyConfig(
        PropertyKey.IONIZATION, "Ionization Energy", "ie",
        "Ionization Energy (eV)", (255, 100, 100)),
    PropertyKey.ELECTRONEGATIVITY: PropertyConfig(
        PropertyKey.ELECTRONEGATIVITY, "Electronegativity", "electronegativity",
        "Electronegativity", (100, 180, 255)),
    PropertyKey.RADIUS: PropertyConfig(
        PropertyKey.RADIUS, "Atomic Radius", "atomic_radius",
        "Atomic Radius (pm)", (100, 255, 180)),
    PropertyKey.MELTING: PropertyConfig(
        PropertyKey.MELTING, "Melting Point", "melting_point",
        "Melting Point (K)", (255, 180, 100)),
    PropertyKey.BOILING: PropertyConfig(
        PropertyKey.BOILING, "Boiling Point", "boiling_point",
        "Boiling Point (K)", (255, 100, 255)),
    PropertyKey.DENSITY: PropertyConfig(
        PropertyKey.DENSITY, "Density", "density",
        "Density (g/cm^3)", (180, 100, 255)),
    PropertyKey.ELECTRON_AFFINITY: PropertyConfig(
        PropertyKey.ELECTRON_AFFINITY, "Electron Affinity", "electron_affinity",
        "Electron Affinity (kJ/mol)", (255, 255, 100)),
    PropertyKey.VALENCE: PropertyConfig(
        PropertyKey.VALENCE, "Valence Electrons", "valence_electrons",
        "Valence Electrons", (100, 255, 255)),
}


# ---------------------------------------------------------------------------
# Normalisation helpers (pure functions)
# ---------------------------------------------------------------------------

_NORMALIZERS: dict[PropertyKey, object] = {
    PropertyKey.IONIZATION:        lambda e: (e.get("ie", 0) - 14) / 11,
    PropertyKey.ELECTRONEGATIVITY: lambda e: (e.get("electronegativity", 0) - 2) / 2,
    PropertyKey.MELTING:           lambda e: (e.get("melting_point", 0) - 1500) / 1500,
    PropertyKey.BOILING:           lambda e: (e.get("boiling_point", 0) - 2000) / 2000,
    PropertyKey.RADIUS:            lambda e: (e.get("atomic_radius", 0) - 150) / 150,
    PropertyKey.DENSITY:           lambda e: math.log10(max(e.get("density", 1), 0.001)) / 1.5,
    PropertyKey.ELECTRON_AFFINITY: lambda e: (e.get("electron_affinity", 0) - 100) / 150,
    PropertyKey.VALENCE:           lambda e: (e.get("valence_electrons", 1) - 4) / 4,
}


def get_normalized_value(item: dict, prop: PropertyKey) -> float:
    """Map an element's raw property value to the range ``[-1, 1]``."""
    fn = _NORMALIZERS.get(prop)
    return fn(item) if fn else 0.0


def get_property_range(items: list[dict], prop: PropertyKey) -> tuple[float, float]:
    """Return ``(min, max)`` of a property across *items*."""
    cfg = PROPERTY_CONFIGS.get(prop)
    if cfg is None:
        return (0.0, 0.0)
    values = [
        it.get(cfg.element_key, 0)
        for it in items
        if it.get(cfg.element_key) is not None
    ]
    if not values:
        return (0.0, 0.0)
    return (min(values), max(values))


# ---------------------------------------------------------------------------
# Ordering helper
# ---------------------------------------------------------------------------

_ORDER_MAP: dict[str, str] = {
    "atomic_number": "z",
    "ionization": "ie",
    "electronegativity": "electronegativity",
    "melting": "melting_point",
    "boiling": "boiling_point",
    "radius": "atomic_radius",
    "density": "density",
    "electron_affinity": "electron_affinity",
    "valence": "valence_electrons",
}


def _order_key(item: dict, order_property: str) -> float:
    key = _ORDER_MAP.get(order_property, "z")
    return item.get(key, 0)


# ---------------------------------------------------------------------------
# Public positioner
# ---------------------------------------------------------------------------

class LinearPositioner(LayoutPositioner):
    """
    Linear (horizontal strip) periodic-table layout.

    Elements are placed in a single row, sorted by *order_property*.
    The output also includes trend-line geometry for all eight standard
    element properties (four above the row, four below).

    Parameters
    ----------
    order_property : str
        Key controlling sort order.  One of ``"atomic_number"``,
        ``"ionization"``, ``"electronegativity"``, ``"melting"``,
        ``"boiling"``, ``"radius"``, ``"density"``,
        ``"electron_affinity"``, ``"valence"``.
        Default ``"atomic_number"``.
    desired_box_size : float
        Target side-length for each element box.  Default ``70``.
    """

    def __init__(
        self,
        order_property: str = "atomic_number",
        desired_box_size: float = 70.0,
    ) -> None:
        self._order_property = order_property
        self._desired_box_size = desired_box_size

    # ---- core API --------------------------------------------------------

    def compute_positions(
        self,
        items: list[dict],
        width: float,
        height: float,
    ) -> list[dict]:
        """
        Lay out *items* in a horizontal row.

        Each returned dict contains::

            x, y            -- centre of the element box
            w, h            -- box width and height (square)
            element_index   -- 0-based position in the sorted row
            period_boundaries -- list[float] of x coords where a new
                                period begins (shared, same object in
                                every dict)
            trend_lines     -- list[dict], one per property, describing
                               the trend line geometry (shared reference)
            label           -- element symbol
            metadata        -- original item dict
        """
        if not items:
            return []

        sorted_items = sorted(items, key=lambda e: _order_key(e, self._order_property))

        # Dynamic margins
        margin_left = max(150.0, width * 0.12)
        margin_right = max(20.0, width * 0.02)
        margin_top = max(40.0, height * 0.05)
        margin_bottom = max(40.0, height * 0.05)

        total_height = height - margin_top - margin_bottom
        max_box_h = total_height * 0.6
        box_size = min(self._desired_box_size, max_box_h)

        center_y = margin_top + total_height / 2.0

        # Detect period boundaries
        period_boundaries: list[float] = []
        last_period = None
        for idx, item in enumerate(sorted_items):
            box_x = margin_left + idx * box_size + box_size / 2.0
            if last_period is not None and item["period"] != last_period:
                period_boundaries.append(box_x - box_size / 2.0)
            last_period = item["period"]

        # Compute trend-line geometry
        trend_lines = self._compute_trend_lines(
            sorted_items, margin_left, box_size, center_y, width, height,
        )

        # Build element position list
        result: list[dict] = []
        for idx, item in enumerate(sorted_items):
            box_x = margin_left + idx * box_size + box_size / 2.0
            result.append({
                "x": box_x,
                "y": center_y,
                "w": box_size,
                "h": box_size,
                "element_index": idx,
                "period_boundaries": period_boundaries,
                "trend_lines": trend_lines,
                "label": item.get("symbol", ""),
                "metadata": item,
            })

        return result

    # ---- trend-line helpers ----------------------------------------------

    def _compute_trend_lines(
        self,
        sorted_items: list[dict],
        margin_left: float,
        box_size: float,
        center_y: float,
        width: float,
        height: float,
    ) -> list[dict]:
        """
        Compute the geometry for all eight property trend lines.

        Returns a list of dicts, each containing::

            property_key   -- PropertyKey enum member
            label          -- human-readable name
            color_rgb      -- (r, g, b)
            min_val, max_val -- raw value range
            points         -- list[(x, y)] tracing the normalised curve
            min_y, max_y   -- y-coordinate band for this trend line
            is_above       -- True if the line is above the element row
        """
        if not sorted_items:
            return []

        all_keys = list(PropertyKey)
        n = len(all_keys)
        n_above = n // 2
        n_below = n - n_above

        margin_top_paint = max(80.0, height * 0.1)

        space_above = center_y - box_size / 2.0 - margin_top_paint
        space_below = (height - margin_top_paint) - (center_y + box_size / 2.0)

        lh_above = space_above / n_above if n_above else 0.0
        lh_below = space_below / n_below if n_below else 0.0

        trend_lines: list[dict] = []

        # --- lines above centre ---
        for i in range(n_above):
            pk = all_keys[i]
            cfg = PROPERTY_CONFIGS[pk]
            min_val, max_val = get_property_range(sorted_items, pk)

            if i == 0:
                min_y = center_y - box_size / 2.0 - lh_above * 0.05
                max_y = min_y - lh_above * 0.85
            else:
                gap = lh_above * 0.10
                max_y = max_y - lh_above * 0.85 - gap  # noqa: F821
                min_y = max_y + lh_above * 0.85

            points = self._trace_property(sorted_items, pk, margin_left, box_size, min_y, max_y)

            trend_lines.append({
                "property_key": pk,
                "label": cfg.label,
                "color_rgb": cfg.color_rgb,
                "min_val": min_val,
                "max_val": max_val,
                "min_y": min_y,
                "max_y": max_y,
                "is_above": True,
                "points": points,
            })

        # --- lines below centre ---
        for i in range(n_below):
            pk = all_keys[n_above + i]
            cfg = PROPERTY_CONFIGS[pk]
            min_val, max_val = get_property_range(sorted_items, pk)

            if i == 0:
                min_y = center_y + box_size / 2.0 + lh_below * 0.05
                max_y = min_y + lh_below * 0.85
            else:
                gap = lh_below * 0.10
                min_y = max_y + gap  # noqa: F821
                max_y = min_y + lh_below * 0.85

            points = self._trace_property(sorted_items, pk, margin_left, box_size, min_y, max_y)

            trend_lines.append({
                "property_key": pk,
                "label": cfg.label,
                "color_rgb": cfg.color_rgb,
                "min_val": min_val,
                "max_val": max_val,
                "min_y": min_y,
                "max_y": max_y,
                "is_above": False,
                "points": points,
            })

        return trend_lines

    @staticmethod
    def _trace_property(
        sorted_items: list[dict],
        prop: PropertyKey,
        margin_left: float,
        box_size: float,
        min_y: float,
        max_y: float,
    ) -> list[tuple[float, float]]:
        """Return a polyline ``[(x, y), ...]`` for one property curve."""
        pts: list[tuple[float, float]] = []
        for idx, item in enumerate(sorted_items):
            x = margin_left + idx * box_size + box_size / 2.0
            nv = get_normalized_value(item, prop)
            y = min_y + (nv + 1.0) / 2.0 * (max_y - min_y)
            pts.append((x, y))
        return pts
