"""
Pure position math for alloy lattice (crystal structure) layout.
Groups alloys by crystal structure (FCC, BCC, HCP, etc.) in centered rows.
"""

import math
from typing import List, Dict, Any, Sequence, Optional


DEFAULT_STRUCTURE_COLORS: Dict[str, tuple] = {
    'FCC': (33, 150, 243),
    'BCC': (244, 67, 54),
    'HCP': (76, 175, 80),
    'BCT': (255, 152, 0),
    'Mixed': (156, 39, 176),
    'Unknown': (158, 158, 158),
}

DEFAULT_STRUCTURE_ORDER = ['FCC', 'BCC', 'HCP', 'BCT', 'Mixed', 'Unknown']

DEFAULT_STRUCTURE_DESCRIPTIONS: Dict[str, str] = {
    'FCC': 'Face-Centered Cubic',
    'BCC': 'Body-Centered Cubic',
    'HCP': 'Hexagonal Close-Packed',
    'BCT': 'Body-Centered Tetragonal',
    'Mixed': 'Multiple crystal phases',
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
    group_spacing: float = 50,
    header_height: float = 55,
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, tuple]] = None,
    structure_descriptions: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute positions for alloys grouped by crystal structure.

    Each item should have: crystal_structure.

    Args:
        items: List of alloy dicts.
        width: Available layout width.
        height: Available layout height.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = group_colors or DEFAULT_STRUCTURE_COLORS
    order = list(group_order or DEFAULT_STRUCTURE_ORDER)
    descriptions = structure_descriptions or DEFAULT_STRUCTURE_DESCRIPTIONS

    # Group by crystal structure
    groups: Dict[str, List[Dict]] = {}
    for item in items:
        struct = item.get('crystal_structure', 'Unknown')
        groups.setdefault(struct, []).append(item)

    # Add structures not in predefined order
    for struct in sorted(groups.keys()):
        if struct not in order:
            # Insert before Unknown/Other
            insert_idx = len(order)
            for sentinel in ('Unknown', 'Other'):
                if sentinel in order:
                    insert_idx = min(insert_idx, order.index(sentinel))
            order.insert(insert_idx, struct)

    results: List[Dict[str, Any]] = []
    current_y = padding
    available_width = width - 2 * padding
    cols = max(1, int(available_width / (card_width + spacing)))

    for structure in order:
        group_items = groups.get(structure, [])
        if not group_items:
            continue

        color = colors.get(structure, (158, 158, 158))
        description = descriptions.get(structure, '')
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
                    'group': structure,
                    'group_description': description,
                    'group_header_y': group_header_y,
                },
            })

        rows = math.ceil(len(group_items) / cols)
        current_y += rows * (card_height + spacing) + group_spacing

    return results
