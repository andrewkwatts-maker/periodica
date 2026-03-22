"""
Pure position math for molecule grid layout.
Arranges items in a simple centered grid pattern.
"""

import math
from typing import List, Dict, Any


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    card_width: float = 150,
    card_height: float = 170,
    padding: float = 80,
    spacing: float = 15,
) -> List[Dict[str, Any]]:
    """
    Compute grid positions for items.

    Args:
        items: List of item dicts (each should have at least 'label').
        width: Available layout width in pixels.
        height: Available layout height in pixels.
        card_width: Width of each card.
        card_height: Height of each card.
        padding: Margin around the grid.
        spacing: Gap between cards.

    Returns:
        List of dicts with keys: x, y, w, h, label, color_rgb, metadata.
    """
    if not items:
        return []

    available_width = width - 2 * padding
    cols = max(1, int(available_width / (card_width + spacing)))

    total_grid_width = cols * card_width + (cols - 1) * spacing
    start_x = (width - total_grid_width) / 2

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        row = idx // cols
        col = idx % cols

        x = start_x + col * (card_width + spacing)
        y = padding + row * (card_height + spacing)

        results.append({
            'x': x,
            'y': y,
            'w': card_width,
            'h': card_height,
            'label': item.get('name', item.get('label', '')),
            'color_rgb': item.get('color_rgb', (200, 200, 200)),
            'metadata': {
                'row': row,
                'col': col,
                'index': idx,
            },
        })

    return results
