"""
Table layout positioner.

Computes the traditional periodic-table grid (18 columns x 9 rows)
with lanthanides and actinides in separate rows 8-9.

No Qt / PySide6 imports -- pure Python + math only.
"""

from __future__ import annotations

from .base import LayoutPositioner


# ---------------------------------------------------------------------------
# Helpers for determining table position from atomic number / symbol
# ---------------------------------------------------------------------------

# Periods list: each sub-list holds the element symbols in that period,
# ordered by increasing atomic number.
# Users can supply their own via the ``periods`` kwarg, but this default
# covers elements 1-118.
_DEFAULT_PERIODS: list[list[str]] = [
    ["H", "He"],                                                                         # 1
    ["Li", "Be", "B", "C", "N", "O", "F", "Ne"],                                        # 2
    ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"],                                     # 3
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
     "Ga", "Ge", "As", "Se", "Br", "Kr"],                                               # 4
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
     "In", "Sn", "Sb", "Te", "I", "Xe"],                                                # 5
    ["Cs", "Ba",
     "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
     "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
     "Tl", "Pb", "Bi", "Po", "At", "Rn"],                                               # 6
    ["Fr", "Ra",
     "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
     "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
     "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],                                               # 7
]


def _get_period(symbol: str, periods: list[list[str]]) -> int:
    """Return 1-based period number for *symbol*."""
    for idx, row in enumerate(periods):
        if symbol in row:
            return idx + 1
    return 1


def _get_group_number(symbol: str, periods: list[list[str]]) -> int:
    """Return 1-based group (column) number for *symbol*."""
    period = _get_period(symbol, periods)
    period_list = periods[period - 1]
    pos = period_list.index(symbol)

    if period == 1:
        return 1 if pos == 0 else 18
    elif period in (2, 3):
        return pos + 1 if pos < 2 else pos + 11
    else:
        return pos + 1


def _is_lanthanide(z: int) -> bool:
    return 57 <= z <= 71


def _is_actinide(z: int) -> bool:
    return 89 <= z <= 103


def _get_table_position(
    z: int,
    symbol: str,
    periods: list[list[str]],
) -> tuple[int, int]:
    """
    Return ``(row, col)`` (both 1-based) for an element in the standard
    18-column periodic table.

    Lanthanides occupy row 8 (cols 3-17), actinides row 9 (cols 3-17).
    """
    period = _get_period(symbol, periods)
    group = _get_group_number(symbol, periods)

    if _is_lanthanide(z):
        row = 8
        col = (z - 57) + 3
    elif _is_actinide(z):
        row = 9
        col = (z - 89) + 3
    else:
        row = period
        col = group
        if period == 6 and z > 71:
            col = (z - 72) + 4
        elif period == 7 and z > 103:
            col = (z - 104) + 4

    return (row, col)


# ---------------------------------------------------------------------------
# Public positioner
# ---------------------------------------------------------------------------

class TablePositioner(LayoutPositioner):
    """
    Classic 18-column periodic table layout.

    Parameters
    ----------
    periods : list[list[str]] | None
        Override the built-in period lists (each sub-list is one period,
        elements in Z order).  ``None`` uses the default 1-118 table.
    """

    def __init__(self, periods: list[list[str]] | None = None) -> None:
        self._periods = periods or _DEFAULT_PERIODS

    # ---- core API --------------------------------------------------------

    def compute_positions(
        self,
        items: list[dict],
        width: float,
        height: float,
    ) -> list[dict]:
        """
        Lay out *items* in a standard periodic-table grid.

        Each returned dict contains::

            x, y         -- top-left pixel coordinate of the cell
            w, h         -- cell width and height (square)
            grid_row     -- 1-based row
            grid_col     -- 1-based column
            label        -- element symbol
            metadata     -- original item dict (pass-through)
        """
        num_cols = 18
        num_rows = 9  # 7 main + 2 for lanthanides/actinides

        # Dynamic margins (5-10 % of dimensions, minimum 30 px)
        margin_left = max(30.0, width * 0.05)
        margin_top = max(30.0, height * 0.05)
        margin_right = max(30.0, width * 0.05)
        margin_bottom = max(30.0, height * 0.05)

        avail_w = width - margin_left - margin_right
        avail_h = height - margin_top - margin_bottom

        cell_by_w = avail_w / num_cols
        cell_by_h = avail_h / num_rows
        cell_size = max(35.0, min(min(cell_by_w, cell_by_h), 80.0))

        result: list[dict] = []
        for item in items:
            symbol = item["symbol"]
            z = item["z"]
            row, col = _get_table_position(z, symbol, self._periods)

            x = margin_left + (col - 1) * cell_size
            y = margin_top + (row - 1) * cell_size

            result.append({
                "x": x,
                "y": y,
                "w": cell_size,
                "h": cell_size,
                "grid_row": row,
                "grid_col": col,
                "label": symbol,
                "metadata": item,
            })

        return result
