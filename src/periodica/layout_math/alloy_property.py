"""
Pure position math for alloy property scatter layout.
Plots alloys on a 2D scatter by two configurable numeric properties.
"""

import math
from typing import List, Dict, Any, Optional


DEFAULT_CATEGORY_COLORS: Dict[str, tuple] = {
    'Steel': (96, 125, 139),
    'Aluminum': (176, 190, 197),
    'Bronze': (205, 127, 50),
    'Brass': (218, 165, 32),
    'Copper': (184, 115, 51),
    'Titanium': (158, 158, 158),
    'Nickel': (192, 192, 192),
    'Superalloy': (233, 30, 99),
    'Precious': (255, 215, 0),
    'Solder': (120, 144, 156),
    'Other': (117, 117, 117),
}


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    x_property: str = 'density',
    y_property: str = 'tensile_strength',
    card_size: float = 60,
    padding: float = 80,
    axis_padding: float = 60,
    category_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute scatter positions for alloys based on two numeric properties.

    Args:
        items: List of alloy dicts.
        width: Available layout width.
        height: Available layout height.
        x_property: Key for X-axis values.
        y_property: Key for Y-axis values.
        card_size: Size of each alloy card.
        padding: Outer padding.
        axis_padding: Space for axis labels.
        category_colors: Optional color map for categories.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = category_colors or DEFAULT_CATEGORY_COLORS

    x_values = [a.get(x_property, 0) for a in items]
    y_values = [a.get(y_property, 0) for a in items]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Add 5% range padding
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05

    plot_left = padding + axis_padding
    plot_right = width - padding
    plot_top = padding
    plot_bottom = height - padding - axis_padding
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    results: List[Dict[str, Any]] = []

    for item in items:
        x_val = item.get(x_property, 0)
        y_val = item.get(y_property, 0)

        norm_x = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        norm_y = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5

        x = plot_left + norm_x * plot_width - card_size / 2
        y = plot_bottom - norm_y * plot_height - card_size / 2

        category = item.get('category', 'Other')
        color = colors.get(category, (117, 117, 117))

        results.append({
            'x': x,
            'y': y,
            'w': card_size,
            'h': card_size,
            'label': item.get('name', item.get('label', '')),
            'color_rgb': color,
            'metadata': {
                'x_value': x_val,
                'y_value': y_val,
                'x_property': x_property,
                'y_property': y_property,
                'category': category,
                'plot_bounds': {
                    'left': plot_left, 'right': plot_right,
                    'top': plot_top, 'bottom': plot_bottom,
                },
                'axis_ranges': {
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                },
            },
        })

    return results
