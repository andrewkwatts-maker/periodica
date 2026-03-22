"""
Position Calculator Module
Dynamically calculates element positions based on atomic properties.
NO HARDCODED positions - everything calculated from atomic properties.
"""

import math
from periodica.data.element_data import PERIODS, get_block, get_period, get_atomic_number


class PositionCalculator:
    """Calculate element positions dynamically based on atomic properties"""

    def __init__(self):
        # Group mapping based on period structure
        # Period 1: H=1, He=18
        # Period 2-3: Groups 1,2,13-18
        # Period 4+: Groups 1-18
        pass

    def get_group_number(self, symbol):
        """
        Calculate group number (1-18) from element symbol and its position in periods.
        Groups are the columns in the periodic table.
        """
        period = get_period(symbol)
        period_list = PERIODS[period - 1]
        position_in_period = period_list.index(symbol)

        if period == 1:
            # Period 1: H is group 1, He is group 18
            return 1 if position_in_period == 0 else 18
        elif period in [2, 3]:
            # Periods 2-3: First 2 elements are groups 1-2, rest are groups 13-18
            if position_in_period < 2:
                return position_in_period + 1
            else:
                return position_in_period + 11  # Maps to groups 13-18
        else:
            # Periods 4+: Full 18 columns
            return position_in_period + 1

    def is_lanthanide(self, z):
        """Check if element is a lanthanide (Z=57-71: La through Lu)"""
        return 57 <= z <= 71

    def is_actinide(self, z):
        """Check if element is an actinide (Z=89-103: Ac through Lr)"""
        return 89 <= z <= 103

    def get_table_position(self, atomic_number, element_symbol):
        """
        Calculate (row, column) for traditional periodic table layout.

        Args:
            atomic_number: The atomic number (Z)
            element_symbol: The element symbol (e.g., 'H', 'He', 'Li')

        Returns:
            Tuple of (row, column) where both are 1-indexed
        """
        period = get_period(element_symbol)
        group = self.get_group_number(element_symbol)

        # Handle lanthanides and actinides - they go in separate rows
        if self.is_lanthanide(atomic_number):
            # Lanthanides: row 8, columns 3-17 (La=3, Ce=4, ..., Lu=17)
            row = 8
            col = (atomic_number - 57) + 3  # La(57) -> col 3, Lu(71) -> col 17
        elif self.is_actinide(atomic_number):
            # Actinides: row 9, columns 3-17 (Ac=3, Th=4, ..., Lr=17)
            row = 9
            col = (atomic_number - 89) + 3  # Ac(89) -> col 3, Lr(103) -> col 17
        else:
            # Standard position
            row = period
            col = group

            # For periods 6 and 7, we need to adjust for the lanthanides/actinides gap
            # After La (Z=57) and before Hf (Z=72), insert lanthanides
            # After Ac (Z=89) and before Rf (Z=104), insert actinides
            if period == 6 and atomic_number > 71:
                # After lanthanides, shift to column 4+ for Hf onwards
                # Hf is Z=72, should be in column 4
                col = (atomic_number - 72) + 4
            elif period == 7 and atomic_number > 103:
                # After actinides, shift to column 4+ for Rf onwards
                # Rf is Z=104, should be in column 4
                col = (atomic_number - 104) + 4

        return (row, col)

    def get_circular_position(self, element, period_idx, elem_idx_in_period, num_elements_in_period):
        """
        Calculate position for circular/wedge layout.

        Args:
            element: Element dictionary with properties
            period_idx: Period index (0-6 for periods 1-7)
            elem_idx_in_period: Index of element within its period
            num_elements_in_period: Total number of elements in the period

        Returns:
            Dictionary with angular position data
        """
        # Define radii for each period (inner, outer)
        period_radii = [
            (45, 75), (75, 110), (110, 150), (150, 195),
            (195, 245), (245, 300), (300, 360)
        ]

        r_inner, r_outer = period_radii[period_idx]

        # Calculate angular span for this element
        # Full circle = 2π radians
        angle_per_elem = (2 * math.pi) / num_elements_in_period
        start_angle = -math.pi / 2  # Start at top

        angle_start = start_angle + elem_idx_in_period * angle_per_elem
        angle_end = angle_start + angle_per_elem
        angle_mid = (angle_start + angle_end) / 2

        return {
            'r_inner': r_inner,
            'r_outer': r_outer,
            'angle_start': angle_start,
            'angle_end': angle_end,
            'angle_mid': angle_mid
        }

    def get_spiral_position(self, element_index, isotope_index=0, total_isotopes=1):
        """
        Calculate position for spiral layout.

        Args:
            element_index: Index of element in full list (0-based)
            isotope_index: Index of isotope within element (0-based)
            total_isotopes: Total number of isotopes for this element

        Returns:
            Dictionary with spiral position data
        """
        # Spiral parameters
        a = 5  # Initial radius
        b = 8  # Spiral tightness (spacing between turns)

        # Calculate base angle from element index
        # Each element gets a small angular segment
        base_theta = element_index * 0.5  # Radians per element

        # If multiple isotopes, spread them slightly in the radial direction
        if total_isotopes > 1:
            isotope_offset = (isotope_index / total_isotopes) * 10
        else:
            isotope_offset = 0

        # Archimedean spiral: r = a + b*theta
        r = a + b * base_theta + isotope_offset
        theta = base_theta

        # Convert to Cartesian coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        return {
            'x': x,
            'y': y,
            'r': r,
            'theta': theta
        }

    def get_serpentine_position(self, element_index, period, num_elements_in_period):
        """
        Calculate position for serpentine/wave layout.

        Args:
            element_index: Index of element in full list
            period: Period number (1-7)
            num_elements_in_period: Total elements in this period

        Returns:
            Dictionary with serpentine position data
        """
        # Serpentine follows a wave pattern
        # X position is linear, Y oscillates based on period

        x = element_index * 50  # Linear spacing

        # Each period gets a different base Y with oscillation
        period_base_y = {
            1: 50, 2: 100, 3: 150, 4: 200,
            5: 250, 6: 300, 7: 350
        }

        base_y = period_base_y.get(period, 200)

        # Add wave oscillation
        wave_amplitude = 20
        wave_frequency = 0.1
        y_offset = wave_amplitude * math.sin(element_index * wave_frequency)

        y = base_y + y_offset

        return {
            'x': x,
            'y': y,
            'base_y': base_y,
            'wave_offset': y_offset
        }
