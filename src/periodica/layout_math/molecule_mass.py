"""
Pure position math for molecule mass-ordered layout.
Arranges items sorted by mass with card sizes scaled proportionally.
"""

import math
from typing import List, Dict, Any


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    base_card_width: float = 120,
    base_card_height: float = 140,
    min_scale: float = 1.0,
    max_scale: float = 1.5,
    padding: float = 80,
    spacing: float = 20,
) -> List[Dict[str, Any]]:
    """
    Compute mass-ordered positions with variable card sizes.

    Items are sorted by 'mass' and card size scales with mass value.
    Rows are centered and wrap when they exceed the available width.

    Args:
        items: List of item dicts, each with a 'mass' key.
        width: Available layout width.
        height: Available layout height.
        base_card_width: Base card width before scaling.
        base_card_height: Base card height before scaling.
        min_scale: Minimum scale factor for lightest item.
        max_scale: Maximum scale factor for heaviest item.
        padding: Margin around layout.
        spacing: Gap between cards.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    sorted_items = sorted(items, key=lambda m: m.get('mass', 0))

    masses = [m.get('mass', 0) for m in sorted_items]
    min_mass = min(masses) if masses else 1
    max_mass = max(masses) if masses else 1
    mass_range = max_mass - min_mass if max_mass > min_mass else 1

    results: List[Dict[str, Any]] = []
    current_x = padding
    current_y = padding
    row_height = 0
    row_start_idx = 0

    for idx, item in enumerate(sorted_items):
        mass = item.get('mass', min_mass)
        scale = min_scale + (max_scale - min_scale) * ((mass - min_mass) / mass_range)

        cw = int(base_card_width * scale)
        ch = int(base_card_height * scale)

        # Wrap to next row if needed
        if current_x + cw > width - padding:
            # Center the completed row
            row_width = current_x - spacing - padding
            offset = (width - row_width - 2 * padding) / 2
            for i in range(row_start_idx, len(results)):
                results[i]['x'] += offset

            current_x = padding
            current_y += row_height + spacing
            row_height = 0
            row_start_idx = len(results)

        results.append({
            'x': current_x,
            'y': current_y,
            'w': cw,
            'h': ch,
            'label': item.get('name', item.get('label', '')),
            'color_rgb': item.get('color_rgb', (200, 200, 200)),
            'metadata': {
                'scale': scale,
                'mass_rank': idx + 1,
                'mass': mass,
            },
        })

        current_x += cw + spacing
        row_height = max(row_height, ch)

    # Center the last row
    if results:
        row_width = current_x - spacing - padding
        offset = (width - row_width - 2 * padding) / 2
        for i in range(row_start_idx, len(results)):
            results[i]['x'] += offset

    return results
