"""
Pure position math for molecule phase diagram scatter layout.
X-axis: melting point, Y-axis: boiling point, size by mass, color by state.
"""

from typing import List, Dict, Any, Optional


DEFAULT_STATE_COLORS: Dict[str, tuple] = {
    'Solid': (244, 67, 54),
    'Liquid': (33, 150, 243),
    'Gas': (76, 175, 80),
}


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    min_card_size: float = 60,
    max_card_size: float = 140,
    padding: float = 80,
    axis_margin: float = 60,
    state_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute scatter positions for items based on melting/boiling points.

    Each item should have: melting_point, boiling_point, mass, state.

    Args:
        items: List of item dicts.
        width: Available layout width.
        height: Available layout height.
        min_card_size: Minimum card dimension (for lightest item).
        max_card_size: Maximum card dimension (for heaviest item).
        padding: Outer padding.
        axis_margin: Space reserved for axis labels.
        state_colors: Optional color map for states.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = state_colors or DEFAULT_STATE_COLORS

    # Filter valid items
    valid = [m for m in items
             if m.get('melting_point', 0) > 0 and m.get('boiling_point', 0) > 0]
    if not valid:
        valid = items

    mps = [m.get('melting_point', 273) for m in valid]
    bps = [m.get('boiling_point', 373) for m in valid]
    masses = [m.get('mass', 1) for m in valid]

    min_mp, max_mp = min(mps), max(mps)
    min_bp, max_bp = min(bps), max(bps)
    min_mass, max_mass = min(masses), max(masses)

    mp_range = max_mp - min_mp if max_mp > min_mp else 100
    bp_range = max_bp - min_bp if max_bp > min_bp else 100
    mass_range = max_mass - min_mass if max_mass > min_mass else 1

    plot_left = padding + axis_margin
    plot_right = width - padding
    plot_top = padding
    plot_bottom = height - padding - axis_margin
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    results: List[Dict[str, Any]] = []

    for item in valid:
        mp = item.get('melting_point', min_mp)
        bp = item.get('boiling_point', min_bp)
        mass = item.get('mass', min_mass)
        state = item.get('state', 'Gas')

        x_ratio = (mp - min_mp) / mp_range if mp_range > 0 else 0.5
        y_ratio = (bp - min_bp) / bp_range if bp_range > 0 else 0.5

        cx = plot_left + x_ratio * plot_width
        cy = plot_bottom - y_ratio * plot_height  # invert Y

        mass_ratio = (mass - min_mass) / mass_range if mass_range > 0 else 0.5
        card_size = min_card_size + mass_ratio * (max_card_size - min_card_size)

        color = colors.get(state, (158, 158, 158))

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
                'state': state,
                'melting_point': mp,
                'boiling_point': bp,
                'mass': mass,
                'plot_bounds': {
                    'left': plot_left, 'right': plot_right,
                    'top': plot_top, 'bottom': plot_bottom,
                },
            },
        })

    return results
