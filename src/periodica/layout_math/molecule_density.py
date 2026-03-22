"""
Pure position math for molecule density-mass scatter layout.
X-axis: molecular mass, Y-axis: density, color by category.
"""

from typing import List, Dict, Any, Optional


DEFAULT_CATEGORY_COLORS: Dict[str, tuple] = {
    'Organic': (76, 175, 80),
    'Inorganic': (33, 150, 243),
    'Ionic': (255, 152, 0),
}


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    min_card_size: float = 70,
    max_card_size: float = 130,
    padding: float = 80,
    axis_margin: float = 60,
    category_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute scatter positions for items based on mass and density.

    Each item should have: mass, density, category.

    Args:
        items: List of item dicts.
        width: Available layout width.
        height: Available layout height.
        min_card_size: Minimum card dimension.
        max_card_size: Maximum card dimension.
        padding: Outer padding.
        axis_margin: Space for axis labels.
        category_colors: Optional color map for categories.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = category_colors or DEFAULT_CATEGORY_COLORS

    # Filter valid items
    valid = [m for m in items
             if m.get('density', 0) > 0 and m.get('mass', 0) > 0]
    if not valid:
        valid = items

    masses = [m.get('mass', 1) for m in valid]
    densities = [m.get('density', 1) for m in valid]

    min_mass, max_mass = min(masses), max(masses)
    min_density, max_density = min(densities), max(densities)

    mass_range = max_mass - min_mass if max_mass > min_mass else 1
    density_range = max_density - min_density if max_density > min_density else 0.1

    plot_left = padding + axis_margin
    plot_right = width - padding
    plot_top = padding
    plot_bottom = height - padding - axis_margin
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    results: List[Dict[str, Any]] = []

    for item in valid:
        mass = item.get('mass', min_mass)
        density = item.get('density', min_density)
        category = item.get('category', 'Inorganic')

        x_ratio = (mass - min_mass) / mass_range if mass_range > 0 else 0.5
        y_ratio = (density - min_density) / density_range if density_range > 0 else 0.5

        cx = plot_left + x_ratio * plot_width
        cy = plot_bottom - y_ratio * plot_height  # invert Y

        # Size by packing efficiency (density / mass ratio)
        packing_ratio = density / max(mass, 1)
        card_size = min_card_size + (packing_ratio * 50) * (max_card_size - min_card_size)
        card_size = min(max(card_size, min_card_size), max_card_size)

        color = colors.get(category, (158, 158, 158))

        results.append({
            'x': cx - card_size / 2,
            'y': cy - card_size / 2,
            'w': card_size,
            'h': card_size,
            'label': item.get('name', item.get('label', '')),
            'color_rgb': color,
            'metadata': {
                'scatter_x': cx,
                'scatter_y': cy,
                'category': category,
                'mass': mass,
                'density': density,
                'packing_efficiency': packing_ratio,
                'plot_bounds': {
                    'left': plot_left, 'right': plot_right,
                    'top': plot_top, 'bottom': plot_bottom,
                },
            },
        })

    return results
