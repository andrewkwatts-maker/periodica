"""
Pure position math for molecule bond-type-grouped layout.
Groups items by bond type (Covalent, Ionic, Metallic, etc.) in centered rows.
"""

import math
from typing import List, Dict, Any, Sequence, Optional


DEFAULT_BOND_COLORS: Dict[str, tuple] = {
    'Covalent': (33, 150, 243),
    'Ionic': (255, 152, 0),
    'Metallic': (156, 39, 176),
    'Hydrogen': (76, 175, 80),
    'Van der Waals': (0, 188, 212),
    'Unknown': (158, 158, 158),
}

DEFAULT_BOND_ORDER = ['Covalent', 'Ionic', 'Metallic', 'Hydrogen', 'Van der Waals']


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
    group_key: str = 'bond_type',
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Dict[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute positions for items grouped by bond type.

    Args:
        items: List of item dicts, each with a 'bond_type' key.
        width: Available layout width.
        height: Available layout height.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    colors = group_colors or DEFAULT_BOND_COLORS
    order_pref = list(group_order or DEFAULT_BOND_ORDER)

    # Group items
    groups: Dict[str, List[Dict]] = {}
    for item in items:
        bt = item.get(group_key, 'Unknown')
        groups.setdefault(bt, []).append(item)

    # Build final order
    ordered: List[str] = []
    for b in order_pref:
        if b in groups:
            ordered.append(b)
    for b in sorted(groups.keys()):
        if b not in ordered:
            ordered.append(b)

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
