"""
Pure position math for molecule bond complexity tree layout.
Groups items by bond count into complexity levels, sub-groups by bond type diversity,
and sizes cards by atom count.
"""

import math
from typing import List, Dict, Any, Set, Optional


# Color by bond-type diversity count
DIVERSITY_COLORS: Dict[int, tuple] = {
    0: (158, 158, 158),   # Grey - no bonds
    1: (76, 175, 80),     # Green - single type
    2: (33, 150, 243),    # Blue - two types
    3: (156, 39, 176),    # Purple - three types
    4: (233, 30, 99),     # Pink - four types
}
DIVERSITY_COLOR_DEFAULT = (255, 87, 34)  # Orange for 5+


def _count_total_bonds(item: Dict) -> int:
    """Count total bonds."""
    return len(item.get('Bonds', []))


def _count_atom_total(item: Dict) -> int:
    """Count total atoms from composition."""
    composition = item.get('Composition', [])
    return sum(c.get('Count', 1) for c in composition)


def _get_bond_type_diversity(item: Dict) -> int:
    """Count unique bond types."""
    bonds = item.get('Bonds', [])
    return len(set(bond.get('Type', 'Single') for bond in bonds))


def _complexity_score(item: Dict) -> float:
    """Compute complexity score."""
    return _count_total_bonds(item) * 2 + _get_bond_type_diversity(item) * 3 + _count_atom_total(item)


def compute_positions(
    items: List[Dict[str, Any]],
    width: float,
    height: float,
    *,
    base_card_width: float = 120,
    base_card_height: float = 140,
    min_card_size: float = 90,
    max_card_size: float = 160,
    padding: float = 40,
    spacing: float = 20,
    level_spacing: float = 80,
    header_height: float = 45,
) -> List[Dict[str, Any]]:
    """
    Compute hierarchical tree positions based on bond complexity.

    Each item should have: Bonds (list of dicts with 'Type'), Composition (list of dicts with 'Count').

    Args:
        items: List of item dicts.
        width: Available layout width.
        height: Available layout height.

    Returns:
        List of position dicts.
    """
    if not items:
        return []

    # Compute metrics
    enriched = []
    for item in items:
        bond_count = _count_total_bonds(item)
        atom_count = _count_atom_total(item)
        diversity = _get_bond_type_diversity(item)
        score = _complexity_score(item)
        enriched.append({
            **item,
            '_bond_count': bond_count,
            '_atom_count': atom_count,
            '_diversity': diversity,
            '_score': score,
        })

    # Group by bond count into levels
    level_defs = [
        ('Simple (0-1 bonds)', 0, lambda bc: bc <= 1),
        ('Basic (2-3 bonds)', 1, lambda bc: 2 <= bc <= 3),
        ('Moderate (4-5 bonds)', 2, lambda bc: 4 <= bc <= 5),
        ('Complex (6+ bonds)', 3, lambda bc: bc >= 6),
    ]

    bond_groups: Dict[str, Dict] = {}
    for level_name, level_order, predicate in level_defs:
        members = [e for e in enriched if predicate(e['_bond_count'])]
        if members:
            bond_groups[level_name] = {'mols': members, 'order': level_order}

    sorted_levels = sorted(bond_groups.items(), key=lambda x: x[1]['order'])

    # Atom count range for sizing
    all_atoms = [e['_atom_count'] for e in enriched]
    min_atoms = min(all_atoms) if all_atoms else 1
    max_atoms = max(all_atoms) if all_atoms else 10
    atom_range = max_atoms - min_atoms if max_atoms > min_atoms else 1

    results: List[Dict[str, Any]] = []
    current_y = padding

    for level_name, group_data in sorted_levels:
        group_mols = group_data['mols']
        if not group_mols:
            continue

        # Sort by diversity then complexity
        group_mols.sort(key=lambda m: (-m['_diversity'], -m['_score']))

        group_header_y = current_y
        current_y += header_height

        # Sub-group by diversity
        diversity_groups: Dict[int, List[Dict]] = {}
        for mol in group_mols:
            d = mol['_diversity']
            diversity_groups.setdefault(d, []).append(mol)

        available_width = width - 2 * padding
        total_rows = 0

        for diversity in sorted(diversity_groups.keys(), reverse=True):
            div_mols = diversity_groups[diversity]
            indent = (diversity - 1) * 40
            indented_width = available_width - indent
            cols = max(1, int(indented_width / (base_card_width + spacing)))

            for idx, mol in enumerate(div_mols):
                row = idx // cols
                col = idx % cols

                # Size based on atom count
                atom_count = mol['_atom_count']
                size_ratio = (atom_count - min_atoms) / atom_range if atom_range > 0 else 0.5
                cw = min_card_size + size_ratio * (max_card_size - min_card_size)
                ch = cw * 1.1

                x = padding + indent + col * (base_card_width + spacing)
                y = current_y + (total_rows + row) * (base_card_height + spacing)

                color = DIVERSITY_COLORS.get(diversity, DIVERSITY_COLOR_DEFAULT)

                results.append({
                    'x': x,
                    'y': y,
                    'w': cw,
                    'h': ch,
                    'label': mol.get('name', mol.get('label', '')),
                    'color_rgb': color,
                    'metadata': {
                        'group': level_name,
                        'group_header_y': group_header_y,
                        'bond_count': mol['_bond_count'],
                        'atom_count': mol['_atom_count'],
                        'diversity': diversity,
                        'complexity_score': mol['_score'],
                    },
                })

            rows_in_div = math.ceil(len(div_mols) / cols)
            total_rows += rows_in_div

        current_y += total_rows * (base_card_height + spacing) + level_spacing

    return results
