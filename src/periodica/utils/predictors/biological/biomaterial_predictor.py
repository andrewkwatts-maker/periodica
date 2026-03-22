"""
Biomaterial Predictor Module
Predicts biological material properties using rule of mixtures and
composite material theories.

References:
- Voigt, W. (1889). Annalen der Physik, 274(12), 573-587.
- Reuss, A. (1929). ZAMM, 9(1), 49-58.
- Hashin, Z. & Shtrikman, S. (1963). J. Mech. Phys. Solids, 11(2), 127-140.
- Gibson, L.J. & Ashby, M.F. (1997). Cellular Solids. Cambridge University Press.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class BiomaterialPredictor:
    """Predicts biological material properties using rule of mixtures."""

    # Default reference moduli for ECM components (MPa)
    # Sources: Fung (1993), Holzapfel (2000), Wang et al. (2006)
    DEFAULT_ECM_MODULI = {
        'collagen_i': 1000,      # Type I collagen fiber
        'collagen_ii': 800,      # Type II (cartilage)
        'collagen_iii': 600,     # Type III (skin, vessels)
        'collagen_iv': 400,      # Type IV (basement membrane)
        'elastin': 0.6,          # Elastic fibers
        'fibronectin': 10,       # Glycoprotein
        'laminin': 5,            # Basement membrane
        'hyaluronan': 0.001,     # Hyaluronic acid
        'proteoglycans': 0.01,   # GAG-rich proteoglycans
        'hydroxyapatite': 117000,# Bone mineral
        'water': 0,              # No structural contribution
    }

    # Default reference densities (g/cm³)
    DEFAULT_ECM_DENSITIES = {
        'collagen_i': 1.3,
        'collagen_ii': 1.3,
        'collagen_iii': 1.3,
        'collagen_iv': 1.3,
        'elastin': 1.1,
        'fibronectin': 1.4,
        'laminin': 1.3,
        'hyaluronan': 1.0,
        'proteoglycans': 1.2,
        'hydroxyapatite': 3.16,
        'water': 1.0,
    }

    # Porosity model options
    POROSITY_MODELS = {
        'gibson_ashby': {'exponent': 2.0, 'description': 'Open-cell foam model'},
        'exponential': {'exponent': 1.5, 'description': 'General exponential decay'},
        'linear': {'exponent': 1.0, 'description': 'Linear porosity effect'},
    }

    def __init__(self, data_path: Optional[Path] = None, config_file: Optional[Path] = None):
        """
        Initialize the biomaterial predictor.

        Args:
            data_path: Path to biomaterial data directory
            config_file: Optional JSON config file for ECM properties
        """
        self._data_path = data_path or Path(__file__).parent.parent.parent.parent / "data" / "active" / "biological_materials"

        # Initialize with defaults
        self.ECM_MODULI = self.DEFAULT_ECM_MODULI.copy()
        self.ECM_DENSITIES = self.DEFAULT_ECM_DENSITIES.copy()
        self._porosity_model = 'gibson_ashby'

        # Load custom config if provided
        if config_file and config_file.exists():
            self._load_config(config_file)

    def _load_config(self, config_file: Path) -> None:
        """Load ECM properties from JSON config file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'ecm_moduli' in config:
                    self.ECM_MODULI.update(config['ecm_moduli'])
                if 'ecm_densities' in config:
                    self.ECM_DENSITIES.update(config['ecm_densities'])
                if 'porosity_model' in config:
                    self.set_porosity_model(config['porosity_model'])
                logger.info(f"Loaded biomaterial config from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config from {config_file}: {e}")

    def set_ecm_modulus(self, component: str, modulus: float) -> None:
        """Set modulus for a specific ECM component."""
        if modulus < 0:
            raise ValueError(f"Modulus must be non-negative, got {modulus}")
        self.ECM_MODULI[component.lower()] = modulus

    def set_ecm_density(self, component: str, density: float) -> None:
        """Set density for a specific ECM component."""
        if density <= 0:
            raise ValueError(f"Density must be positive, got {density}")
        self.ECM_DENSITIES[component.lower()] = density

    def set_porosity_model(self, model: str) -> None:
        """Set the porosity model to use."""
        if model not in self.POROSITY_MODELS:
            raise ValueError(f"Unknown porosity model '{model}'. Available: {list(self.POROSITY_MODELS.keys())}")
        self._porosity_model = model

    def get_available_components(self) -> List[str]:
        """Get list of available ECM components."""
        return list(self.ECM_MODULI.keys())

    def _validate_composition(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize composition dictionary."""
        if not composition:
            raise ValueError("Composition cannot be empty")

        validated = {}
        for component, fraction in composition.items():
            if fraction < 0:
                raise ValueError(f"Volume fraction for '{component}' cannot be negative: {fraction}")
            if fraction > 0:
                validated[component.lower()] = fraction

        total = sum(validated.values())
        if total <= 0:
            raise ValueError("Total volume fraction must be positive")

        # Warn if fractions don't sum to ~1
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Volume fractions sum to {total:.3f}, not 1.0. Consider normalizing.")

        return validated

    def calculate_voigt_modulus(self, composition: Dict[str, float]) -> float:
        """
        Calculate upper bound modulus using Voigt (iso-strain) model.

        E_voigt = Σ (f_i × E_i)

        Args:
            composition: Dict of {component: volume_fraction}

        Returns:
            Voigt bound modulus in MPa

        Note:
            Voigt bound assumes uniform strain (iso-strain). Gives upper bound
            for composite modulus. Most accurate for fiber-reinforced composites
            loaded parallel to fibers.
        """
        validated = self._validate_composition(composition)
        E_voigt = 0
        for component, fraction in validated.items():
            E_i = self.ECM_MODULI.get(component, 1.0)
            if component not in self.ECM_MODULI:
                logger.warning(f"Unknown component '{component}', using default modulus 1.0 MPa")
            E_voigt += fraction * E_i
        return E_voigt

    def calculate_reuss_modulus(self, composition: Dict[str, float]) -> float:
        """
        Calculate lower bound modulus using Reuss (iso-stress) model.

        1/E_reuss = Σ (f_i / E_i)

        Args:
            composition: Dict of {component: volume_fraction}

        Returns:
            Reuss bound modulus in MPa

        Note:
            Reuss bound assumes uniform stress (iso-stress). Gives lower bound
            for composite modulus. Components with E=0 (like water) are skipped.
        """
        validated = self._validate_composition(composition)
        inv_E = 0
        skipped_fraction = 0

        for component, fraction in validated.items():
            E_i = self.ECM_MODULI.get(component, 1.0)
            if component not in self.ECM_MODULI:
                logger.warning(f"Unknown component '{component}', using default modulus 1.0 MPa")

            if E_i > 0:
                inv_E += fraction / E_i
            else:
                # Skip zero-modulus components (water, etc.) but track them
                skipped_fraction += fraction
                logger.debug(f"Skipping '{component}' with zero modulus in Reuss calculation")

        if inv_E <= 0:
            logger.warning("All components have zero modulus, returning 0")
            return 0.0

        return 1.0 / inv_E

    def calculate_hashin_shtrikman_bounds(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate Hashin-Shtrikman bounds for composite modulus.
        Tighter bounds than Voigt-Reuss for two-phase composites.

        Args:
            composition: Dict of {component: volume_fraction}

        Returns:
            Tuple of (lower_bound, upper_bound) in MPa
        """
        # Simplified for two major components
        components = list(composition.items())
        if len(components) < 2:
            E = self.calculate_voigt_modulus(composition)
            return (E, E)

        # Sort by modulus
        sorted_comps = sorted(components,
                              key=lambda x: self.ECM_MODULI.get(x[0].lower(), 0))

        f1, f2 = sorted_comps[0][1], sorted_comps[1][1]
        E1 = self.ECM_MODULI.get(sorted_comps[0][0].lower(), 0.1)
        E2 = self.ECM_MODULI.get(sorted_comps[1][0].lower(), 0.1)

        # Simplified 2D bounds (assuming Poisson ratio = 0.3)
        K1, K2 = E1 / 2.4, E2 / 2.4  # Approximate bulk modulus

        lower = E1 + f2 / (1/(E2 - E1) + f1/(E1 + K1))
        upper = E2 + f1 / (1/(E1 - E2) + f2/(E2 + K2))

        return (max(0, lower), max(lower, upper))

    def calculate_composite_modulus(self, composition: Dict[str, float],
                                     model: str = "average") -> float:
        """
        Calculate composite modulus using specified model.

        Args:
            composition: Dict of {component: volume_fraction}
            model: "voigt", "reuss", "average", or "hs" (Hashin-Shtrikman)

        Returns:
            Effective modulus in MPa
        """
        if model == "voigt":
            return self.calculate_voigt_modulus(composition)
        elif model == "reuss":
            return self.calculate_reuss_modulus(composition)
        elif model == "hs":
            lower, upper = self.calculate_hashin_shtrikman_bounds(composition)
            return (lower + upper) / 2
        else:  # average
            voigt = self.calculate_voigt_modulus(composition)
            reuss = self.calculate_reuss_modulus(composition)
            return (voigt + reuss) / 2

    def calculate_composite_density(self, composition: Dict[str, float]) -> float:
        """
        Calculate composite density using rule of mixtures.

        ρ = Σ (f_i × ρ_i)

        Args:
            composition: Dict of {component: volume_fraction}

        Returns:
            Density in g/cm³
        """
        density = 0
        for component, fraction in composition.items():
            rho_i = self.ECM_DENSITIES.get(component.lower(), 1.0)
            density += fraction * rho_i
        return density

    def calculate_porosity_effect(self, E_dense: float, porosity: float,
                                     model: Optional[str] = None) -> float:
        """
        Adjust modulus for porosity using selected porosity model.

        Models available:
        - gibson_ashby: E/E_s = (1-φ)^2.0 (open-cell foam)
        - exponential: E/E_s = (1-φ)^1.5 (general porous solid)
        - linear: E/E_s = (1-φ) (simple linear)

        Args:
            E_dense: Dense (zero porosity) modulus in MPa
            porosity: Volume fraction of pores (0-1)
            model: Porosity model to use (default: instance setting)

        Returns:
            Effective modulus accounting for porosity in MPa

        Raises:
            ValueError: If porosity is outside [0, 1] range
        """
        if porosity < 0 or porosity > 1:
            raise ValueError(f"Porosity must be between 0 and 1, got {porosity}")

        if E_dense < 0:
            raise ValueError(f"Dense modulus must be non-negative, got {E_dense}")

        if porosity >= 1.0:
            return 0.0

        model = model or self._porosity_model
        if model not in self.POROSITY_MODELS:
            raise ValueError(f"Unknown porosity model '{model}'")

        n = self.POROSITY_MODELS[model]['exponent']
        return E_dense * ((1 - porosity) ** n)

    def estimate_ultimate_strength(self, youngs_modulus: float,
                                     tissue_type: str = "soft") -> float:
        """
        Estimate ultimate tensile strength from modulus.

        For soft tissues: σ ≈ E/10 to E/50
        For bone: σ ≈ E/100 to E/150

        Args:
            youngs_modulus: Young's modulus in MPa
            tissue_type: "soft" or "hard"

        Returns:
            Estimated UTS in MPa
        """
        if tissue_type == "hard":
            return youngs_modulus / 120  # Bone-like
        else:
            return youngs_modulus / 30   # Soft tissue

    def calculate_poissons_ratio(self, composition: Dict[str, float]) -> float:
        """
        Estimate Poisson's ratio for biological material.

        Most soft tissues: ν ≈ 0.45-0.50 (nearly incompressible)
        Bone: ν ≈ 0.3

        Args:
            composition: Dict of {component: volume_fraction}

        Returns:
            Estimated Poisson's ratio
        """
        hydroxyapatite_fraction = composition.get('hydroxyapatite', 0)
        water_fraction = composition.get('water', 0)

        # More mineral = lower Poisson's ratio (more compressible)
        # More water = higher Poisson's ratio (incompressible)
        base_nu = 0.35
        mineral_effect = -0.1 * hydroxyapatite_fraction
        water_effect = 0.15 * water_fraction

        return min(0.50, max(0.25, base_nu + mineral_effect + water_effect))

    def calculate_shear_modulus(self, E: float, nu: float) -> float:
        """
        Calculate shear modulus from Young's modulus and Poisson's ratio.

        G = E / (2(1 + ν))

        Args:
            E: Young's modulus in MPa
            nu: Poisson's ratio

        Returns:
            Shear modulus in MPa
        """
        return E / (2 * (1 + nu))

    def analyze_biomaterial(self, name: str, tissue_type: str,
                            ecm_composition: Dict[str, float],
                            cell_composition: Dict[str, float] = None,
                            porosity: float = 0.0) -> Dict:
        """
        Perform comprehensive biomaterial analysis.

        Args:
            name: Material name
            tissue_type: Type classification
            ecm_composition: Dict of {component: volume_fraction}
            cell_composition: Dict of {cell_type: volume_fraction}
            porosity: Porosity as fraction (0-1)

        Returns:
            Dictionary with all calculated properties
        """
        # Normalize composition
        total = sum(ecm_composition.values())
        if total > 0:
            ecm_normalized = {k: v/total for k, v in ecm_composition.items()}
        else:
            ecm_normalized = ecm_composition

        # Calculate mechanical properties
        E_voigt = self.calculate_voigt_modulus(ecm_normalized)
        E_reuss = self.calculate_reuss_modulus(ecm_normalized)
        E_average = (E_voigt + E_reuss) / 2
        E_effective = self.calculate_porosity_effect(E_average, porosity)

        density = self.calculate_composite_density(ecm_normalized)
        nu = self.calculate_poissons_ratio(ecm_normalized)
        G = self.calculate_shear_modulus(E_effective, nu)

        # Determine if hard or soft tissue
        is_hard = ecm_normalized.get('hydroxyapatite', 0) > 0.2
        UTS = self.estimate_ultimate_strength(E_effective, "hard" if is_hard else "soft")

        # Water content
        water_content = ecm_normalized.get('water', 0) * 100

        return {
            "name": name,
            "type": tissue_type,
            "ecm_composition": ecm_composition,
            "ecm_composition_normalized": ecm_normalized,
            "cell_composition": cell_composition or {},
            "porosity": porosity * 100,  # Convert to percentage
            "mechanical_properties": {
                "youngs_modulus_MPa": round(E_effective, 4),
                "youngs_modulus_voigt": round(E_voigt, 4),
                "youngs_modulus_reuss": round(E_reuss, 4),
                "ultimate_strength_MPa": round(UTS, 4),
                "poissons_ratio": round(nu, 3),
                "shear_modulus_MPa": round(G, 4),
            },
            "physical_properties": {
                "density_g_cm3": round(density, 3),
                "water_content_percent": round(water_content, 1),
            },
            "derived_properties": {
                "stiffness_category": self._categorize_stiffness(E_effective),
                "model_used": "Voigt-Reuss average with Gibson-Ashby porosity correction"
            }
        }

    def _categorize_stiffness(self, E_MPa: float) -> str:
        """Categorize tissue stiffness."""
        if E_MPa < 0.1:
            return "Ultra-soft"
        elif E_MPa < 1:
            return "Soft"
        elif E_MPa < 100:
            return "Intermediate"
        elif E_MPa < 1000:
            return "Stiff"
        else:
            return "Hard"

    def compare_with_cells(self, ecm_composition: Dict[str, float],
                           cell_fraction: float) -> Dict:
        """
        Compare ECM-only vs ECM+cells material properties.

        Cells are generally softer than ECM, so adding cells reduces stiffness.

        Args:
            ecm_composition: Dict of {component: volume_fraction}
            cell_fraction: Volume fraction occupied by cells (0-1)

        Returns:
            Comparison dictionary
        """
        # ECM-only properties
        E_ecm = self.calculate_composite_modulus(ecm_composition)

        # Cells are soft (E ≈ 0.1-1 kPa)
        E_cells = 0.001  # MPa

        # Adjusted modulus with cells
        E_with_cells = E_ecm * (1 - cell_fraction) + E_cells * cell_fraction

        return {
            "ecm_modulus_MPa": round(E_ecm, 4),
            "effective_modulus_MPa": round(E_with_cells, 4),
            "cell_fraction": cell_fraction,
            "stiffness_reduction_percent": round((E_ecm - E_with_cells) / E_ecm * 100, 1)
        }
