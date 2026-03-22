"""
Pure position math for molecule dipole moment layout.
X-axis: dipole moment, grouped vertically by polarity.
"""

from typing import List, Dict, Any, Sequence, Optional


DEFAULT_POLARITY_COLORS: Dict[str, tuple] = {
    'Polar': (33, 150, 243),
    'Nonpolar': (76, 175, 80),
    'Ionic': (255, 152, 0),
}

DEFAULT_POLARITY_ORDER = ['Nonpolar', 'Polar', 'Ionic']


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    card_width: float = 130,
    card_height: float = 150,
    padding: float = 40,
    spacing: float = 15,
    group_spacing: float = 50,
    header_height: float = 45,
    axis_height: float = 50,
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute positions for items based on dipole moment and polarity grouping.

    Each item should have: dipole_moment, polarity.

    Args:
        items: List of item dicts.
        width: Available layout width.
        height: Available layout height.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = group_colors or DEFAULT_POLARITY_COLORS
    order = list(group_order or DEFAULT_POLARITY_ORDER)
    if 'Ionic' not in order:
        order.append('Ionic')

    # Group by polarity
    groups: Dict[str, List[Dict]] = {'Polar': [], 'Nonpolar': [], 'Ionic': []}
    for item in items:
        pol = item.get('polarity', 'Unknown')
        if pol in groups:
            groups[pol].append(item)
        else:
            groups['Nonpolar'].append(item)

    # Sort each group by dipole moment
    for g in groups:
        groups[g].sort(key=lambda m: m.get('dipole_moment', 0))

    # Dipole range for X-axis
    all_dipoles = [m.get('dipole_moment', 0) for m in items]
    max_dipole = max(all_dipoles) if all_dipoles else 5.0
    min_dipole = 0

    plot_left = padding + 20
    plot_right = width - padding
    plot_width = plot_right - plot_left

    results: List[Dict[str, Any]] = []
    current_y = padding + axis_height

    for group_name in order:
        group_items = groups.get(group_name, [])
        if not group_items:
            continue

        color = colors.get(group_name, (158, 158, 158))
        group_header_y = current_y
        current_y += header_height

        # Bin by similar dipole values (nearest 0.5 D)
        dipole_bins: Dict[float, List[Dict]] = {}
        for item in group_items:
            d = item.get('dipole_moment', 0)
            bin_key = round(d * 2) / 2
            dipole_bins.setdefault(bin_key, []).append(item)

        row_count = 0
        for bin_dipole in sorted(dipole_bins.keys()):
            bin_items = dipole_bins[bin_dipole]
            for row_idx, item in enumerate(bin_items):
                dipole = item.get('dipole_moment', 0)

                if max_dipole > min_dipole:
                    x_ratio = (dipole - min_dipole) / (max_dipole - min_dipole)
                else:
                    x_ratio = 0.5

                x = plot_left + x_ratio * (plot_width - card_width)
                y = current_y + row_idx * (card_height + spacing)

                results.append({
                    'x': x,
                    'y': y,
                    'w': card_width,
                    'h': card_height,
                    'label': item.get('name', item.get('label', '')),
                    'color_rgb': color,
                    'metadata': {
                        'group': group_name,
                        'group_header_y': group_header_y,
                        'dipole_moment': dipole,
                    },
                })

                row_count = max(row_count, row_idx + 1)

        current_y += row_count * (card_height + spacing) + group_spacing

    return results
