"""
Rule of Mixtures Predictor for Alloy Properties

Implements alloy property predictions using the rule of mixtures approach,
which calculates weighted averages of constituent element properties.

Key formulas:
- Density: 1/rho_alloy = sum(w_i / rho_i) (inverse rule of mixtures)
- Melting point: T_m = sum(w_i * T_i) * depression_factor
- Lattice parameter: a_alloy = sum(x_i * a_i) (Vegard's law)
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseAlloyPredictor, AlloyInput, AlloyResult
from ..registry import register_predictor


# Element property data for calculations
ELEMENT_DENSITIES = {
    'Fe': 7.874, 'Al': 2.70, 'Cu': 8.96, 'Ni': 8.908, 'Cr': 7.19,
    'Ti': 4.506, 'Zn': 7.14, 'Sn': 7.265, 'Mn': 7.21, 'Mo': 10.28,
    'W': 19.25, 'V': 6.11, 'Co': 8.90, 'Nb': 8.57, 'Si': 2.33,
    'Ag': 10.49, 'Au': 19.30, 'Pb': 11.34, 'C': 2.267, 'N': 1.251,
    'P': 1.823, 'S': 2.07, 'B': 2.34, 'Mg': 1.738
}

ELEMENT_MELTING_POINTS = {
    'Fe': 1811, 'Al': 933.5, 'Cu': 1357.8, 'Ni': 1728, 'Cr': 2180,
    'Ti': 1941, 'Zn': 692.7, 'Sn': 505.1, 'Mn': 1519, 'Mo': 2896,
    'W': 3695, 'V': 2183, 'Co': 1768, 'Nb': 2750, 'Si': 1687,
    'Ag': 1234.9, 'Au': 1337.3, 'Pb': 600.6, 'C': 3915, 'N': 63.15,
    'P': 317.3, 'S': 388.4, 'B': 2349, 'Mg': 923
}

ELEMENT_THERMAL_CONDUCTIVITY = {
    'Fe': 80.4, 'Al': 237, 'Cu': 401, 'Ni': 90.9, 'Cr': 93.9,
    'Ti': 21.9, 'Zn': 116, 'Sn': 66.8, 'Mn': 7.81, 'Mo': 138,
    'W': 173, 'V': 30.7, 'Co': 100, 'Nb': 53.7, 'Si': 149,
    'Ag': 429, 'Au': 317, 'Pb': 35.3, 'C': 140, 'Mg': 156
}

ELEMENT_RESISTIVITY = {
    'Fe': 9.71, 'Al': 2.65, 'Cu': 1.68, 'Ni': 6.99, 'Cr': 12.7,
    'Ti': 42.0, 'Zn': 5.92, 'Sn': 11.5, 'Mn': 144, 'Mo': 5.34,
    'W': 5.28, 'V': 20.1, 'Co': 6.24, 'Nb': 15.2, 'Si': 2300,
    'Ag': 1.59, 'Au': 2.44, 'Pb': 20.6, 'C': 3500
}

# Lattice packing factors
PACKING_FACTORS = {
    'FCC': 0.74,
    'BCC': 0.68,
    'HCP': 0.74,
    'BCT': 0.70,
    'Diamond': 0.34,
    'SC': 0.52
}


@register_predictor('alloy', 'rule_of_mixtures')
@register_predictor('alloy', 'default')
class RuleOfMixturesPredictor(BaseAlloyPredictor):
    """
    Alloy property predictor using the rule of mixtures.

    This predictor calculates alloy properties by combining constituent
    element properties using weighted averaging schemes appropriate for
    each property type.

    Key methods:
    - Density: Inverse rule of mixtures (harmonic mean)
    - Melting point: Weighted average with depression factor
    - Thermal conductivity: Weighted average with reduction factor
    - Electrical resistivity: Matthiessen's rule
    """

    @property
    def name(self) -> str:
        return "Rule of Mixtures Predictor"

    @property
    def description(self) -> str:
        return (
            "Predicts alloy properties using rule of mixtures calculations. "
            "Uses inverse rule for density, weighted averages for thermal properties, "
            "and includes empirical corrections for multi-component systems."
        )

    def predict(self, input_data: AlloyInput) -> AlloyResult:
        """
        Predict alloy properties from composition.

        Args:
            input_data: AlloyInput containing components and temperature

        Returns:
            AlloyResult with calculated properties
        """
        # Extract elements and weight fractions from components
        elements = []
        weight_fractions = []

        for comp in input_data.components:
            symbol = comp.get('symbol') or comp.get('element') or comp.get('Element', '')
            fraction = comp.get('fraction', comp.get('weight_fraction', 0.0))
            if symbol and fraction > 0:
                elements.append(symbol)
                weight_fractions.append(fraction)

        # Normalize fractions if needed
        total = sum(weight_fractions)
        if total > 0 and abs(total - 1.0) > 0.001:
            weight_fractions = [wf / total for wf in weight_fractions]

        # Calculate properties
        density = self.calculate_density(elements, weight_fractions)
        melting_point = self.calculate_melting_point(elements, weight_fractions)
        thermal_cond = self._calculate_thermal_conductivity(elements, weight_fractions)
        resistivity = self._calculate_electrical_resistivity(elements, weight_fractions)

        # Determine crystal structure from primary element
        crystal_structure = self._determine_crystal_structure(elements, weight_fractions)

        return AlloyResult(
            density_g_cm3=density,
            melting_point_k=melting_point,
            thermal_conductivity=thermal_cond,
            electrical_resistivity=resistivity,
            crystal_structure=crystal_structure
        )

    def calculate_density(
        self,
        elements: List[str],
        weight_fractions: List[float]
    ) -> float:
        """
        Calculate alloy density using inverse rule of mixtures.

        The inverse rule of mixtures (also called the harmonic mean) is
        appropriate for density because volume is additive:

        1/rho_alloy = sum(w_i / rho_i)

        where w_i is the weight fraction and rho_i is the density of element i.

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions for each element

        Returns:
            Alloy density in g/cm^3
        """
        if not elements or not weight_fractions:
            return 7.0  # Default steel-like density

        inverse_density_sum = 0.0

        for elem, wf in zip(elements, weight_fractions):
            elem_density = ELEMENT_DENSITIES.get(elem, 7.0)
            if elem_density > 0:
                inverse_density_sum += wf / elem_density

        if inverse_density_sum > 0:
            return 1.0 / inverse_density_sum
        return 7.0

    def calculate_melting_point(
        self,
        elements: List[str],
        weight_fractions: List[float]
    ) -> float:
        """
        Calculate alloy melting point with depression factor.

        Uses weighted average of constituent melting points, then applies
        a depression factor to account for the thermodynamic effect of
        mixing multiple components (entropy of mixing lowers the melting point).

        Depression factor: 1.0 - 0.03 * (n_components - 1)
        Clamped to range [0.85, 1.0]

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions for each element

        Returns:
            Estimated melting point in Kelvin
        """
        if not elements or not weight_fractions:
            return 1500.0  # Default value

        # Calculate weighted average melting point
        weighted_mp = 0.0
        for elem, wf in zip(elements, weight_fractions):
            mp = ELEMENT_MELTING_POINTS.get(elem, 1500)
            weighted_mp += wf * mp

        # Apply melting point depression for multi-component alloys
        # Each additional component typically depresses the melting point
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        depression_factor = 1.0 - 0.03 * (num_components - 1)
        depression_factor = max(0.85, min(1.0, depression_factor))

        return weighted_mp * depression_factor

    def _calculate_thermal_conductivity(
        self,
        elements: List[str],
        weight_fractions: List[float]
    ) -> float:
        """
        Calculate thermal conductivity using rule of mixtures with reduction factor.

        Alloys typically have lower thermal conductivity than their constituent
        elements due to phonon scattering at solute atoms.

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions for each element

        Returns:
            Thermal conductivity in W/(m*K)
        """
        if not elements or not weight_fractions:
            return 50.0  # Default value

        weighted_tc = 0.0
        for elem, wf in zip(elements, weight_fractions):
            tc = ELEMENT_THERMAL_CONDUCTIVITY.get(elem, 50)
            weighted_tc += wf * tc

        # Apply reduction factor for alloying (phonon scattering)
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        reduction = 0.7 ** (num_components - 1)
        reduction = max(0.3, min(1.0, reduction))

        return weighted_tc * reduction

    def _calculate_electrical_resistivity(
        self,
        elements: List[str],
        weight_fractions: List[float]
    ) -> float:
        """
        Calculate electrical resistivity using Matthiessen's rule.

        Matthiessen's rule states that resistivity contributions from different
        scattering mechanisms are additive. For alloys, we add a contribution
        from solute scattering:

        rho_alloy = rho_weighted + rho_alloying

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions for each element

        Returns:
            Electrical resistivity in Ohm*m
        """
        if not elements or not weight_fractions:
            return 10e-8  # Default value

        weighted_res = 0.0
        for elem, wf in zip(elements, weight_fractions):
            res = ELEMENT_RESISTIVITY.get(elem, 10)
            weighted_res += wf * res

        # Matthiessen's rule addition for alloying
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        alloying_addition = 5 * (num_components - 1)

        return (weighted_res + alloying_addition) * 1e-8

    def _determine_crystal_structure(
        self,
        elements: List[str],
        weight_fractions: List[float]
    ) -> str:
        """
        Determine likely crystal structure based on primary element.

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions for each element

        Returns:
            Crystal structure string (FCC, BCC, HCP, etc.)
        """
        if not elements or not weight_fractions:
            return "FCC"

        # Find primary element (highest weight fraction)
        max_idx = weight_fractions.index(max(weight_fractions))
        primary = elements[max_idx]

        # Common crystal structures by element
        structures = {
            'Fe': 'BCC', 'Al': 'FCC', 'Cu': 'FCC', 'Ni': 'FCC', 'Cr': 'BCC',
            'Ti': 'HCP', 'Zn': 'HCP', 'Mn': 'BCC', 'Mo': 'BCC', 'W': 'BCC',
            'V': 'BCC', 'Co': 'HCP', 'Nb': 'BCC', 'Ag': 'FCC', 'Au': 'FCC',
            'Pb': 'FCC', 'Mg': 'HCP'
        }

        return structures.get(primary, 'FCC')

    def get_packing_factor(self, structure: str) -> float:
        """
        Get the atomic packing factor for a crystal structure.

        Packing factors represent the fraction of volume occupied by atoms:
        - FCC: 0.74 (face-centered cubic, close-packed)
        - BCC: 0.68 (body-centered cubic)
        - HCP: 0.74 (hexagonal close-packed)

        Args:
            structure: Crystal structure name

        Returns:
            Atomic packing factor
        """
        return PACKING_FACTORS.get(structure, 0.68)

    def calculate_formation_enthalpy(
        self,
        element_data_a: Dict,
        element_data_b: Dict,
        x_a: float,
        x_b: float
    ) -> Optional[float]:
        """
        Calculate formation enthalpy using Miedema's semi-empirical model.

        Reads Miedema parameters (phi_star, nws_13, V_23) from element data
        sheets. Returns None if parameters are not available.

        Formula:
            DH = f(c_A, c_B) * [-P*(Dphi*)^2 + Q*(Dnws^(1/3))^2]
            f(c_A, c_B) = 2*c_A*c_B*(c_A*V_A^(2/3) + c_B*V_B^(2/3))

        Model constants: P = 12.35, Q/P = 9.4

        Args:
            element_data_a: Element A data dict (from element JSON data sheet)
            element_data_b: Element B data dict (from element JSON data sheet)
            x_a: Mole fraction of element A
            x_b: Mole fraction of element B

        Returns:
            Formation enthalpy in kJ/mol, or None if Miedema params missing
        """
        # Model constants (universal, not per-element)
        P = 12.35
        Q = P * 9.4  # = 116.09

        # Read Miedema parameters from element data sheets
        phi_a = element_data_a.get('miedema_phi_star')
        phi_b = element_data_b.get('miedema_phi_star')
        nws_a = element_data_a.get('miedema_nws_13')
        nws_b = element_data_b.get('miedema_nws_13')
        V_a = element_data_a.get('miedema_V_23')
        V_b = element_data_b.get('miedema_V_23')

        if any(v is None for v in [phi_a, phi_b, nws_a, nws_b, V_a, V_b]):
            return None

        # Concentration-dependent geometric factor
        f_conc = 2 * x_a * x_b * (x_a * V_a + x_b * V_b)

        # Electronegativity and electron density mismatch
        delta_phi = phi_a - phi_b
        delta_nws = nws_a - nws_b

        # Miedema formation enthalpy
        dH = f_conc * (-P * delta_phi ** 2 + Q * delta_nws ** 2)
        return round(dH, 2)

    def calculate_vegard_lattice(
        self,
        element_data_a: Dict,
        element_data_b: Dict,
        x_a: float,
        x_b: float
    ) -> Optional[float]:
        """
        Calculate alloy lattice parameter using Vegard's Law.

        a_alloy = x_A * a_A + x_B * a_B

        Reads lattice parameters from element data sheets (atomic_radius
        as proxy when lattice_parameter not available).

        Args:
            element_data_a: Element A data dict
            element_data_b: Element B data dict
            x_a: Mole fraction of A
            x_b: Mole fraction of B

        Returns:
            Estimated lattice parameter in pm, or None
        """
        # Use atomic_radius as proxy for lattice parameter
        a_a = element_data_a.get('atomic_radius')
        a_b = element_data_b.get('atomic_radius')

        if a_a is None or a_b is None:
            return None

        return round(x_a * float(a_a) + x_b * float(a_b), 1)

    def get_confidence(self, input_data: AlloyInput, result: AlloyResult) -> float:
        """
        Calculate confidence level for the prediction.

        Confidence is higher for:
        - Binary alloys (simpler mixing behavior)
        - Common alloying elements
        - Standard temperature ranges

        Args:
            input_data: The input data used for prediction
            result: The prediction result

        Returns:
            Confidence level between 0.0 and 1.0
        """
        base_confidence = 0.85

        # Reduce confidence for many-component alloys
        num_components = len(input_data.components)
        if num_components > 4:
            base_confidence -= 0.1
        if num_components > 6:
            base_confidence -= 0.1

        # Reduce confidence for unusual temperatures
        if input_data.temperature_k < 200 or input_data.temperature_k > 500:
            base_confidence -= 0.05

        # Check if all elements are in our database
        for comp in input_data.components:
            symbol = comp.get('symbol') or comp.get('element', '')
            if symbol not in ELEMENT_DENSITIES:
                base_confidence -= 0.1
                break

        return max(0.3, min(1.0, base_confidence))

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """
        Validate alloy input data.

        Args:
            input_data: The input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(input_data, AlloyInput):
            return False, f"Expected AlloyInput, got {type(input_data).__name__}"

        if not input_data.components:
            return False, "Alloy must have at least one component"

        total_fraction = 0.0
        for comp in input_data.components:
            fraction = comp.get('fraction', comp.get('weight_fraction', 0.0))
            total_fraction += fraction

        if abs(total_fraction - 1.0) > 0.01:
            return False, f"Component fractions must sum to 1.0, got {total_fraction}"

        return True, ""
