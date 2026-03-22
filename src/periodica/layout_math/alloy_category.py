"""
Pure position math for alloy category-grouped layout.
Groups alloys by category (Steel, Bronze, Brass, etc.) in centered rows.
"""

import math
from typing import List, Dict, Any, Sequence, Optional


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

DEFAULT_CATEGORY_ORDER = [
    'Steel', 'Aluminum', 'Bronze', 'Brass', 'Copper',
    'Titanium', 'Nickel', 'Superalloy', 'Precious', 'Solder', 'Other',
]


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    card_width: float = 160,
    card_height: float = 180,
    padding: float = 30,
    spacing: float = 15,
    group_spacing: float = 40,
    header_height: float = 45,
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute positions for alloys grouped by category.

    Each item should have: category.

    Args:
        items: List of alloy dicts.
        width: Available layout width.
        height: Available layout height.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = group_colors or DEFAULT_CATEGORY_COLORS
    order = list(group_order or DEFAULT_CATEGORY_ORDER)

    # Group by category
    groups: Dict[str, List[Dict]] = {}
    for item in items:
        cat = item.get('category', 'Other')
        groups.setdefault(cat, []).append(item)

    # Add categories not in predefined order
    for cat in sorted(groups.keys()):
        if cat not in order:
            order.append(cat)

    results: List[Dict[str, Any]] = []
    current_y = padding
    available_width = width - 2 * padding
    cols = max(1, int(available_width / (card_width + spacing)))

    for category in order:
        group_items = groups.get(category, [])
        if not group_items:
            continue

        color = colors.get(category, (117, 117, 117))
        group_header_y = current_y
        current_y += header_height

        for idx, item in enumerate(group_items):
            row = idx // cols
            col = idx % cols

            items_in_row = min(cols, len(group_items) - row * cols)
            row_width = items_in_row * card_width + (items_in_row - 1) * spacing
            start_x = (width - row_width) / 2

            x = start_x + col * (card_width + spacing)
            y = current_y + row * (card_height + spacing)

            results.append({
                'x': x,
                'y': y,
                'w': card_width,
                'h': card_height,
                'label': item.get('name', item.get('label', '')),
                'color_rgb': color,
                'metadata': {
                    'group': category,
                    'group_header_y': group_header_y,
                },
            })

        rows = math.ceil(len(group_items) / cols)
        current_y += rows * (card_height + spacing) + group_spacing

    return results
