"""
Abstract base class for layout positioners.

All layout positioners compute element positions as pure data (dicts)
with zero dependency on any GUI framework.
"""

from abc import ABC, abstractmethod


class LayoutPositioner(ABC):
    """
    Abstract base for computing element positions.

    Subclasses implement ``compute_positions`` which returns a list of
    position dicts that any rendering backend can consume.

    Each dict in the returned list contains at minimum::

        {
            "x": float,       # horizontal pixel coordinate
            "y": float,       # vertical pixel coordinate
            "w": float,       # width of the element cell / region
            "h": float,       # height of the element cell / region
            "label": str,     # display label (typically the element symbol)
            "metadata": dict, # pass-through of the original element data
        }

    Individual positioners may add extra keys (angles, radii, etc.).
    """

    @abstractmethod
    def compute_positions(
        self,
        items: list[dict],
        width: float,
        height: float,
    ) -> list[dict]:
        """
        Compute positions for *items* inside a viewport of the given
        *width* and *height* (in pixels or abstract units).

        Parameters
        ----------
        items : list[dict]
            Each dict must have at least ``"symbol"`` (str), ``"z"`` (int),
            and ``"period"`` (int).  Extra keys are preserved in ``metadata``.
        width : float
            Available horizontal space.
        height : float
            Available vertical space.

        Returns
        -------
        list[dict]
            One dict per item with position/size keys described above.
        """
        ...

    # ------------------------------------------------------------------
    # Utility helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_dynamic_spacing(
        total_items: int,
        available_space: float,
        min_spacing: float = 5.0,
        max_spacing: float = 100.0,
    ) -> float:
        """Return optimal spacing for *total_items* in *available_space*."""
        if total_items <= 1:
            return 0.0
        spacing = available_space / (total_items - 1)
        return max(min_spacing, min(max_spacing, spacing))

    @staticmethod
    def calculate_period_radii(
        num_periods: int,
        base_radius: float = 45.0,
        ring_spacing: float = 55.0,
    ) -> list[tuple[float, float]]:
        """Return ``(r_inner, r_outer)`` for each period ring."""
        result: list[tuple[float, float]] = []
        for idx in range(num_periods):
            r_inner = base_radius + idx * ring_spacing
            r_outer = r_inner + ring_spacing - 5.0
            result.append((r_inner, r_outer))
        return result
