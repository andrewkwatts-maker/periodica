"""
Material Properties Predictor
Derives engineering material properties from alloy composition and processing.

Derivation chain: Quarks → Hadrons → Nuclei → Atoms → Molecules/Alloys → Materials

Key models used:
1. Rule of Mixtures for composites
2. Hall-Petch relation for grain size effects on strength
3. Ashby material property correlations
4. Empirical correlations for thermal/electrical properties
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

from ..base import BasePredictor
from ..registry import register_predictor


@dataclass
class MaterialInput:
    """Input for material property prediction."""
    alloy_data: Dict  # Alloy composition and properties
    processing: Optional[Dict] = None  # Heat treatment, cold work, etc.
    grain_size_um: float = 25.0  # Average grain size in micrometers
    temperature_K: float = 293.15  # Operating temperature (default 20°C)
    porosity: float = 0.0  # Volume fraction of porosity


@dataclass
class MaterialResult:
    """Complete material properties result."""
    name: str = ""
    category: str = ""

    # Elastic properties
    youngs_modulus_GPa: float = 0.0
    shear_modulus_GPa: float = 0.0
    bulk_modulus_GPa: float = 0.0
    poissons_ratio: float = 0.0

    # Strength properties
    yield_strength_MPa: float = 0.0
    ultimate_tensile_strength_MPa: float = 0.0
    compressive_strength_MPa: float = 0.0

    # Hardness
    hardness_HB: float = 0.0
    hardness_HRC: float = 0.0
    hardness_HV: float = 0.0

    # Toughness and ductility
    fracture_toughness_MPa_sqrt_m: float = 0.0
    elongation_percent: float = 0.0
    fatigue_limit_MPa: float = 0.0

    # Thermal properties
    melting_point_K: float = 0.0
    thermal_conductivity_W_mK: float = 0.0
    specific_heat_J_kgK: float = 0.0
    thermal_expansion_coeff_per_K: float = 0.0

    # Electrical properties
    electrical_resistivity_Ohm_m: float = 0.0

    # Physical
    density_kg_m3: float = 0.0

    # Confidence
    confidence: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@register_predictor('material', 'physics_based')
@register_predictor('material', 'default')
class MaterialPredictor(BasePredictor):
    """
    Physics-based material property predictor.

    Uses fundamental relationships:
    - E = 2G(1+ν) and K = E/3(1-2ν) for elastic moduli relations
    - Hall-Petch: σy = σ0 + k/√d for grain size strengthening
    - Wiedemann-Franz: κ/σT = L (thermal-electrical correlation)
    - Ashby correlations for property bounds
    """

    # Ashby correlation coefficients (from Cambridge Materials Selector)
    ASHBY_COEFFICIENTS = {
        'steel': {'E_rho': 26.0, 'sigma_y_E': 0.003, 'KIC_sigma_y': 0.1},
        'aluminum': {'E_rho': 26.0, 'sigma_y_E': 0.004, 'KIC_sigma_y': 0.15},
        'titanium': {'E_rho': 25.0, 'sigma_y_E': 0.005, 'KIC_sigma_y': 0.08},
        'copper': {'E_rho': 13.0, 'sigma_y_E': 0.001, 'KIC_sigma_y': 0.2},
        'polymer': {'E_rho': 1.5, 'sigma_y_E': 0.02, 'KIC_sigma_y': 0.5},
        'ceramic': {'E_rho': 50.0, 'sigma_y_E': 0.001, 'KIC_sigma_y': 0.02},
        'composite': {'E_rho': 30.0, 'sigma_y_E': 0.006, 'KIC_sigma_y': 0.1},
    }

    # Hall-Petch constants for common materials (σ0 in MPa, k in MPa·√μm)
    HALL_PETCH = {
        'steel': {'sigma_0': 100, 'k': 600},
        'aluminum': {'sigma_0': 20, 'k': 150},
        'titanium': {'sigma_0': 200, 'k': 400},
        'copper': {'sigma_0': 25, 'k': 110},
    }

    # Wiedemann-Franz Lorenz number (W·Ω/K²)
    LORENZ_NUMBER = 2.44e-8

    @property
    def name(self) -> str:
        return "Physics-Based Material Predictor"

    @property
    def description(self) -> str:
        return "Derives material properties using physics-based correlations"

    def predict(self, input_data: MaterialInput) -> MaterialResult:
        """Predict complete material properties from alloy data."""
        result = MaterialResult()
        alloy = input_data.alloy_data

        # Get base properties from alloy
        result.density_kg_m3 = alloy.get('density_g_cm3', 7.85) * 1000
        result.melting_point_K = alloy.get('melting_point_K', 1800)

        # Determine material category
        result.category = self._determine_category(alloy)
        result.name = alloy.get('name', 'Unknown Material')

        # Calculate elastic properties
        result.youngs_modulus_GPa = self._calculate_youngs_modulus(alloy, result.category)
        result.poissons_ratio = self._calculate_poissons_ratio(result.category)
        result.shear_modulus_GPa = result.youngs_modulus_GPa / (2 * (1 + result.poissons_ratio))
        result.bulk_modulus_GPa = result.youngs_modulus_GPa / (3 * (1 - 2 * result.poissons_ratio))

        # Calculate strength with Hall-Petch grain size effect
        result.yield_strength_MPa = self._calculate_yield_strength(
            alloy, result.category, input_data.grain_size_um
        )
        result.ultimate_tensile_strength_MPa = result.yield_strength_MPa * 1.2
        result.compressive_strength_MPa = result.yield_strength_MPa * 1.1

        # Calculate hardness (Tabor relation: HV ≈ 3σy)
        result.hardness_HV = result.yield_strength_MPa / 3.0
        result.hardness_HB = result.hardness_HV * 0.95
        result.hardness_HRC = self._convert_HB_to_HRC(result.hardness_HB)

        # Toughness and ductility
        result.fracture_toughness_MPa_sqrt_m = self._calculate_fracture_toughness(
            result.yield_strength_MPa, result.category
        )
        result.elongation_percent = self._calculate_elongation(result.category, alloy)
        result.fatigue_limit_MPa = result.ultimate_tensile_strength_MPa * 0.4

        # Thermal properties
        result.thermal_conductivity_W_mK = self._calculate_thermal_conductivity(alloy, result.category)
        result.specific_heat_J_kgK = self._calculate_specific_heat(alloy, result.category)
        result.thermal_expansion_coeff_per_K = self._calculate_thermal_expansion(result.category)

        # Electrical properties (Wiedemann-Franz law)
        result.electrical_resistivity_Ohm_m = self._calculate_resistivity(
            result.thermal_conductivity_W_mK, input_data.temperature_K
        )

        # Apply porosity correction
        if input_data.porosity > 0:
            result = self._apply_porosity_correction(result, input_data.porosity)

        # Calculate confidence
        result.confidence = self._calculate_confidence(alloy, input_data)

        return result

    def _determine_category(self, alloy: Dict) -> str:
        """Determine material category from alloy composition."""
        category = alloy.get('category', '').lower()
        composition = alloy.get('composition', [])

        # Check for known categories
        if 'steel' in category or 'iron' in category:
            return 'steel'
        elif 'aluminum' in category or 'aluminium' in category:
            return 'aluminum'
        elif 'titanium' in category:
            return 'titanium'
        elif 'copper' in category or 'brass' in category or 'bronze' in category:
            return 'copper'
        elif 'polymer' in category or 'plastic' in category:
            return 'polymer'
        elif 'ceramic' in category:
            return 'ceramic'
        elif 'composite' in category:
            return 'composite'

        # Infer from composition
        if composition:
            main_element = max(composition, key=lambda x: x.get('percentage', 0))
            symbol = main_element.get('element', '')
            if symbol == 'Fe':
                return 'steel'
            elif symbol == 'Al':
                return 'aluminum'
            elif symbol == 'Ti':
                return 'titanium'
            elif symbol == 'Cu':
                return 'copper'

        return 'steel'  # Default

    def _calculate_youngs_modulus(self, alloy: Dict, category: str) -> float:
        """Calculate Young's modulus from alloy properties."""
        # Base moduli by category (GPa)
        base_moduli = {
            'steel': 205,
            'aluminum': 70,
            'titanium': 114,
            'copper': 117,
            'polymer': 3,
            'ceramic': 350,
            'composite': 150,
        }

        E = alloy.get('youngs_modulus_GPa', base_moduli.get(category, 200))
        return E

    def _calculate_poissons_ratio(self, category: str) -> float:
        """Get typical Poisson's ratio for category."""
        ratios = {
            'steel': 0.29,
            'aluminum': 0.33,
            'titanium': 0.34,
            'copper': 0.34,
            'polymer': 0.40,
            'ceramic': 0.22,
            'composite': 0.30,
        }
        return ratios.get(category, 0.30)

    def _calculate_yield_strength(self, alloy: Dict, category: str, grain_size_um: float) -> float:
        """Calculate yield strength with Hall-Petch correction."""
        # Base yield strength from alloy
        base_strength = alloy.get('yield_strength_MPa', 250)

        # Apply Hall-Petch strengthening
        if category in self.HALL_PETCH:
            hp = self.HALL_PETCH[category]
            # Hall-Petch: σy = σ0 + k/√d
            hp_contribution = hp['k'] / math.sqrt(max(grain_size_um, 1.0))
            return hp['sigma_0'] + hp_contribution

        return base_strength

    def _calculate_fracture_toughness(self, yield_strength: float, category: str) -> float:
        """Calculate fracture toughness from yield strength."""
        coeffs = self.ASHBY_COEFFICIENTS.get(category, {'KIC_sigma_y': 0.1})
        # KIC ≈ coefficient × σy × √(characteristic_length)
        # Using simplified correlation
        KIC = yield_strength * coeffs['KIC_sigma_y']
        return max(KIC, 5.0)  # Minimum realistic value

    def _calculate_elongation(self, category: str, alloy: Dict) -> float:
        """Calculate elongation at break."""
        base_elongation = {
            'steel': 20,
            'aluminum': 12,
            'titanium': 14,
            'copper': 35,
            'polymer': 100,
            'ceramic': 0.1,
            'composite': 2,
        }
        return alloy.get('elongation_percent', base_elongation.get(category, 15))

    def _calculate_thermal_conductivity(self, alloy: Dict, category: str) -> float:
        """Calculate thermal conductivity."""
        base_k = {
            'steel': 50,
            'aluminum': 167,
            'titanium': 17,
            'copper': 385,
            'polymer': 0.2,
            'ceramic': 25,
            'composite': 5,
        }
        return alloy.get('thermal_conductivity_W_mK', base_k.get(category, 50))

    def _calculate_specific_heat(self, alloy: Dict, category: str) -> float:
        """Calculate specific heat capacity."""
        base_cp = {
            'steel': 490,
            'aluminum': 900,
            'titanium': 520,
            'copper': 385,
            'polymer': 1500,
            'ceramic': 800,
            'composite': 1000,
        }
        return alloy.get('specific_heat_J_kgK', base_cp.get(category, 500))

    def _calculate_thermal_expansion(self, category: str) -> float:
        """Calculate thermal expansion coefficient (per K)."""
        base_alpha = {
            'steel': 12e-6,
            'aluminum': 23e-6,
            'titanium': 9e-6,
            'copper': 17e-6,
            'polymer': 100e-6,
            'ceramic': 8e-6,
            'composite': 5e-6,
        }
        return base_alpha.get(category, 12e-6)

    def _calculate_resistivity(self, thermal_conductivity: float, temperature_K: float) -> float:
        """Calculate electrical resistivity using Wiedemann-Franz law."""
        # κ/σ = L×T, where L is Lorenz number
        # σ = κ/(L×T), ρ = 1/σ = L×T/κ
        if thermal_conductivity > 0:
            return self.LORENZ_NUMBER * temperature_K / thermal_conductivity
        return 1e-6  # Default metallic resistivity

    def _convert_HB_to_HRC(self, HB: float) -> float:
        """Convert Brinell hardness to Rockwell C."""
        # Empirical conversion for steels
        if HB < 200:
            return 0  # Below HRC range
        elif HB < 240:
            return (HB - 200) * 0.2
        else:
            return 8 + (HB - 240) * 0.11

    def _apply_porosity_correction(self, result: MaterialResult, porosity: float) -> MaterialResult:
        """Apply porosity correction to properties."""
        # Modulus correction: E_porous = E_dense × (1 - p)^2
        correction = (1 - porosity) ** 2
        result.youngs_modulus_GPa *= correction
        result.shear_modulus_GPa *= correction
        result.bulk_modulus_GPa *= correction

        # Strength correction: σ_porous = σ_dense × (1 - p)^1.5
        strength_correction = (1 - porosity) ** 1.5
        result.yield_strength_MPa *= strength_correction
        result.ultimate_tensile_strength_MPa *= strength_correction

        # Thermal conductivity: k_porous = k_dense × (1 - p)
        result.thermal_conductivity_W_mK *= (1 - porosity)

        result.warnings.append(f"Properties adjusted for {porosity*100:.1f}% porosity")
        return result

    def _calculate_confidence(self, alloy: Dict, input_data: MaterialInput) -> Dict[str, float]:
        """Calculate confidence in predictions."""
        confidence = {
            'elastic': 0.9,  # Well-established relationships
            'strength': 0.75,  # Hall-Petch varies by material
            'thermal': 0.85,  # Good correlations
            'electrical': 0.7,  # Wiedemann-Franz approximate
            'overall': 0.8,
        }

        # Reduce confidence for unusual conditions
        if input_data.temperature_K < 200 or input_data.temperature_K > 500:
            confidence['thermal'] *= 0.8
            confidence['overall'] *= 0.9

        if input_data.porosity > 0.1:
            confidence['strength'] *= 0.7
            confidence['overall'] *= 0.85

        return confidence

    def get_confidence(self, input_data: MaterialInput, result: MaterialResult) -> Dict[str, float]:
        return result.confidence

    def validate(self, input_data: MaterialInput, result: MaterialResult) -> Dict[str, Any]:
        """Validate result against physical bounds."""
        issues = []

        # Check physical bounds
        if result.youngs_modulus_GPa < 0.1 or result.youngs_modulus_GPa > 1000:
            issues.append(f"Young's modulus {result.youngs_modulus_GPa} GPa outside typical range")

        if result.poissons_ratio < 0 or result.poissons_ratio > 0.5:
            issues.append(f"Poisson's ratio {result.poissons_ratio} outside physical bounds")

        if result.yield_strength_MPa > result.ultimate_tensile_strength_MPa:
            issues.append("Yield strength exceeds UTS")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
        }
