"""
Pure position math for molecule geometry-grouped layout.
Groups items by molecular geometry and lays out in centered rows per group.
"""

import math
from typing import List, Dict, Any, Sequence, Optional


# Default color map for geometry groups (RGB tuples)
DEFAULT_GEOMETRY_COLORS: Dict[str, tuple] = {
    'Linear': (33, 150, 243),
    'Bent': (76, 175, 80),
    'Trigonal Planar': (255, 152, 0),
    'Trigonal Pyramidal': (156, 39, 176),
    'Tetrahedral': (244, 67, 54),
    'Octahedral': (0, 188, 212),
    'Unknown': (158, 158, 158),
}


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    card_width: float = 150,
    card_height: float = 170,
    padding: float = 80,
    spacing: float = 15,
    group_spacing: float = 40,
    header_height: float = 40,
    group_key: str = 'geometry',
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute positions for items grouped by molecular geometry.

    Args:
        items: List of item dicts, each with a geometry field.
        width: Available layout width.
        height: Available layout height.
        card_width: Card width.
        card_height: Card height.
        padding: Margin around layout.
        spacing: Gap between cards.
        group_spacing: Extra vertical gap between groups.
        header_height: Space reserved for group headers.
        group_key: Dict key for the grouping field.
        group_order: Preferred ordering of groups.
        group_colors: Mapping of group name to (r, g, b) color.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = group_colors or DEFAULT_GEOMETRY_COLORS

    # Group items
    groups: Dict[str, List[Dict]] = {}
    for item in items:
        g = item.get(group_key, 'Unknown')
        groups.setdefault(g, []).append(item)

    # Build ordered group list
    ordered: List[str] = []
    if group_order:
        for g in group_order:
            if g in groups:
                ordered.append(g)
    for g in sorted(groups.keys()):
        if g not in ordered:
            ordered.append(g)

    results: List[Dict[str, Any]] = []
    current_y = padding

    available_width = width - 2 * padding
    cols = max(1, int(available_width / (card_width + spacing)))

    for group_name in ordered:
        group_items = groups.get(group_name, [])
        if not group_items:
            continue

        color = colors.get(group_name, (158, 158, 158))
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
                    'group': group_name,
                    'group_header_y': group_header_y,
                },
            })

        rows = math.ceil(len(group_items) / cols)
        current_y += rows * (card_height + spacing) + group_spacing

    return results
