"""
Pure position math for alloy composition layout.
Groups alloys by primary (base) element in centered rows.
"""

import math
from typing import List, Dict, Any, Sequence, Optional


DEFAULT_ELEMENT_COLORS: Dict[str, tuple] = {
    'Fe': (96, 125, 139),
    'Al': (176, 190, 197),
    'Cu': (184, 115, 51),
    'Ti': (158, 158, 158),
    'Ni': (192, 192, 192),
    'Zn': (120, 144, 156),
    'Sn': (189, 189, 189),
    'Ag': (192, 192, 192),
    'Au': (255, 215, 0),
    'Pb': (117, 117, 117),
}

DEFAULT_ELEMENT_NAMES: Dict[str, str] = {
    'Fe': 'Iron-based',
    'Al': 'Aluminum-based',
    'Cu': 'Copper-based',
    'Ti': 'Titanium-based',
    'Ni': 'Nickel-based',
    'Zn': 'Zinc-based',
    'Sn': 'Tin-based',
    'Ag': 'Silver-based',
    'Au': 'Gold-based',
    'Pb': 'Lead-based',
}


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
    element_colors: Optional[Dict[str, tuple]] = None,
    element_names: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute positions for alloys grouped by primary element.

    Each item should have: primary_element.

    Args:
        items: List of alloy dicts.
        width: Available layout width.
        height: Available layout height.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = element_colors or DEFAULT_ELEMENT_COLORS
    names = element_names or DEFAULT_ELEMENT_NAMES

    # Group by primary element
    groups: Dict[str, List[Dict]] = {}
    for item in items:
        primary = item.get('primary_element', 'Unknown')
        groups.setdefault(primary, []).append(item)

    # Sort elements by count (most common first)
    sorted_elements = sorted(groups.keys(), key=lambda e: -len(groups[e]))

    results: List[Dict[str, Any]] = []
    current_y = padding
    available_width = width - 2 * padding
    cols = max(1, int(available_width / (card_width + spacing)))

    for element in sorted_elements:
        group_items = groups.get(element, [])
        if not group_items:
            continue

        color = colors.get(element, (158, 158, 158))
        display_name = names.get(element, f'{element}-based')
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
                    'group': display_name,
                    'group_element': element,
                    'group_header_y': group_header_y,
                },
            })

        rows = math.ceil(len(group_items) / cols)
        current_y += rows * (card_height + spacing) + group_spacing

    return results
