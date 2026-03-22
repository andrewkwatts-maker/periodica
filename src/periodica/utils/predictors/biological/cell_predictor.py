"""
Cell Predictor Module
Predicts cell properties using metabolic scaling laws and biophysical calculations.
Uses Kleiber's Law for metabolic rate scaling.

References:
- Kleiber, M. (1932). Hilgardia, 6(11), 315-353.
- West, G.B. et al. (2002). Nature, 413, 628-631.
- Savage, V.M. et al. (2007). PLOS Computational Biology, 3(4), e59.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class CellPredictor:
    """
    Predicts cell properties using metabolic scaling laws.

    Configurable parameters:
    - B0: Basal metabolic coefficient (default 3.5e-12 W)
    - ALPHA: Scaling exponent (default 0.75, range 0.67-0.85)
    - ATP_ENERGY: Energy per ATP (temperature dependent)
    - CELL_DENSITY: Default cell density (g/cm³)

    All parameters can be adjusted via set_* methods or loaded from config.
    """

    # Default Kleiber's Law constants
    DEFAULT_B0 = 3.5e-12   # Basal metabolic coefficient (watts)
    DEFAULT_ALPHA = 0.75   # Scaling exponent (3/4 power law)

    # Physical constants
    DEFAULT_ATP_ENERGY = 3.06e-20   # Joules per ATP at 25°C, pH 7
    DEFAULT_CELL_DENSITY = 1.05     # g/cm³ (typical mammalian cell)

    # Temperature dependence constants (Arrhenius)
    ACTIVATION_ENERGY = 0.65  # eV, metabolic activation energy
    BOLTZMANN_K = 8.617e-5    # eV/K

    # Valid ranges for parameters
    VALID_RANGES = {
        'B0': (1e-15, 1e-9),
        'ALPHA': (0.5, 1.0),
        'diameter_um': (0.1, 1000),
        'temperature_C': (0, 50),
    }

    def __init__(self, data_path: Optional[Path] = None, config_file: Optional[Path] = None):
        """
        Initialize the cell predictor.

        Args:
            data_path: Path to cell data directory
            config_file: Optional JSON config file for parameters
        """
        self._data_path = data_path or Path(__file__).parent.parent.parent.parent / "data" / "active" / "cells"

        # Initialize with defaults
        self.B0 = self.DEFAULT_B0
        self.ALPHA = self.DEFAULT_ALPHA
        self.ATP_ENERGY = self.DEFAULT_ATP_ENERGY
        self.CELL_DENSITY = self.DEFAULT_CELL_DENSITY
        self._temperature_C = 37.0  # Default body temperature

        # Load custom config if provided
        if config_file and config_file.exists():
            self._load_config(config_file)

    def _load_config(self, config_file: Path) -> None:
        """Load parameters from JSON config file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'B0' in config:
                    self.set_B0(config['B0'])
                if 'ALPHA' in config:
                    self.set_alpha(config['ALPHA'])
                if 'ATP_ENERGY' in config:
                    self.ATP_ENERGY = config['ATP_ENERGY']
                if 'CELL_DENSITY' in config:
                    self.CELL_DENSITY = config['CELL_DENSITY']
                if 'temperature_C' in config:
                    self.set_temperature(config['temperature_C'])
                logger.info(f"Loaded cell predictor config from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config from {config_file}: {e}")

    def set_B0(self, value: float) -> None:
        """Set basal metabolic coefficient with validation."""
        min_val, max_val = self.VALID_RANGES['B0']
        if not min_val <= value <= max_val:
            raise ValueError(f"B0 must be between {min_val} and {max_val}, got {value}")
        self.B0 = value

    def set_alpha(self, value: float) -> None:
        """Set scaling exponent with validation."""
        min_val, max_val = self.VALID_RANGES['ALPHA']
        if not min_val <= value <= max_val:
            raise ValueError(f"ALPHA must be between {min_val} and {max_val}, got {value}")
        self.ALPHA = value

    def set_temperature(self, temp_C: float) -> None:
        """Set temperature for temperature-dependent calculations."""
        min_val, max_val = self.VALID_RANGES['temperature_C']
        if not min_val <= temp_C <= max_val:
            raise ValueError(f"Temperature must be between {min_val} and {max_val}°C, got {temp_C}")
        self._temperature_C = temp_C

    def _validate_diameter(self, diameter_um: float) -> None:
        """Validate cell diameter."""
        min_val, max_val = self.VALID_RANGES['diameter_um']
        if not min_val <= diameter_um <= max_val:
            raise ValueError(f"Cell diameter must be between {min_val} and {max_val} μm, got {diameter_um}")

    def calculate_metabolic_rate(self, cell_mass_pg: float,
                                    temperature_C: Optional[float] = None) -> float:
        """
        Calculate metabolic rate using Kleiber's Law with optional temperature correction.

        B = B0 * M^α * exp(-E_a/(k*T))

        Args:
            cell_mass_pg: Cell mass in picograms
            temperature_C: Temperature in Celsius (default: instance setting)

        Returns:
            Metabolic rate in femtowatts (fW)

        Raises:
            ValueError: If cell mass is non-positive
        """
        if cell_mass_pg <= 0:
            raise ValueError(f"Cell mass must be positive, got {cell_mass_pg}")

        # Convert picograms to grams for Kleiber's law
        mass_g = cell_mass_pg * 1e-12

        # Base metabolic rate in watts
        B = self.B0 * (mass_g ** self.ALPHA)

        # Apply temperature correction (Arrhenius)
        temp = temperature_C if temperature_C is not None else self._temperature_C
        if temp != 37.0:  # Only correct if different from reference
            T_ref = 37.0 + 273.15  # Reference temperature in K
            T = temp + 273.15      # Actual temperature in K
            # Q10 ≈ 2-3 for most biological processes
            correction = math.exp(self.ACTIVATION_ENERGY / self.BOLTZMANN_K * (1/T_ref - 1/T))
            B *= correction

        # Convert to femtowatts
        return B * 1e15

    def calculate_cell_volume(self, diameter_um: float) -> float:
        """
        Calculate cell volume assuming spherical shape.

        Args:
            diameter_um: Cell diameter in micrometers

        Returns:
            Volume in femtoliters (fL)
        """
        radius_um = diameter_um / 2
        # Volume of sphere in μm³
        volume_um3 = (4/3) * math.pi * (radius_um ** 3)
        # 1 μm³ = 1 fL
        return volume_um3

    def calculate_cell_mass(self, volume_fl: float, density: float = None) -> float:
        """
        Calculate cell mass from volume.

        Args:
            volume_fl: Cell volume in femtoliters
            density: Cell density in g/cm³ (default 1.05)

        Returns:
            Mass in picograms
        """
        if density is None:
            density = self.CELL_DENSITY
        # 1 fL = 1 μm³ = 1e-12 cm³
        volume_cm3 = volume_fl * 1e-12
        # Mass in grams
        mass_g = volume_cm3 * density
        # Convert to picograms
        return mass_g * 1e12

    def calculate_doubling_time(self, metabolic_rate_fw: float, cell_mass_pg: float,
                                   growth_efficiency: float = 0.05) -> float:
        """
        Estimate cell doubling time based on metabolic rate.

        Args:
            metabolic_rate_fw: Metabolic rate in femtowatts
            cell_mass_pg: Cell mass in picograms
            growth_efficiency: Fraction of metabolic energy for biosynthesis (0-1, default 0.05)

        Returns:
            Estimated doubling time in hours

        Raises:
            ValueError: If inputs are invalid
        """
        if metabolic_rate_fw <= 0:
            raise ValueError(f"Metabolic rate must be positive, got {metabolic_rate_fw}")
        if cell_mass_pg <= 0:
            raise ValueError(f"Cell mass must be positive, got {cell_mass_pg}")
        if not 0 < growth_efficiency <= 1:
            raise ValueError(f"Growth efficiency must be between 0 and 1, got {growth_efficiency}")

        # Energy required to double mass (rough estimate)
        # ~2.5e-11 J per pg of new biomass (based on ATP cost of biosynthesis)
        energy_per_pg = 2.5e-11  # Joules per picogram

        energy_needed = cell_mass_pg * energy_per_pg  # Joules

        # Power available for growth
        growth_power_w = (metabolic_rate_fw * 1e-15) * growth_efficiency  # Watts

        # Time in seconds
        time_s = energy_needed / growth_power_w

        # Convert to hours
        return time_s / 3600

    def calculate_atp_turnover(self, metabolic_rate_fw: float) -> float:
        """
        Calculate ATP turnover rate.

        Args:
            metabolic_rate_fw: Metabolic rate in femtowatts

        Returns:
            ATP molecules hydrolyzed per second
        """
        # Convert fW to W
        power_w = metabolic_rate_fw * 1e-15
        # ATP molecules per second
        return power_w / self.ATP_ENERGY

    def calculate_oxygen_consumption(self, metabolic_rate_fw: float) -> float:
        """
        Estimate oxygen consumption rate.

        Assumes ~5 molecules O2 per ATP via oxidative phosphorylation.

        Args:
            metabolic_rate_fw: Metabolic rate in femtowatts

        Returns:
            O2 molecules consumed per second
        """
        atp_rate = self.calculate_atp_turnover(metabolic_rate_fw)
        # ~5 O2 per 30-36 ATP (oxidative phosphorylation)
        return atp_rate / 6

    def calculate_surface_area(self, diameter_um: float) -> float:
        """
        Calculate cell surface area assuming spherical shape.

        Args:
            diameter_um: Cell diameter in micrometers

        Returns:
            Surface area in μm²
        """
        radius_um = diameter_um / 2
        return 4 * math.pi * (radius_um ** 2)

    def calculate_surface_volume_ratio(self, diameter_um: float) -> float:
        """
        Calculate surface area to volume ratio.

        Important for nutrient/waste exchange efficiency.

        Args:
            diameter_um: Cell diameter in micrometers

        Returns:
            Surface area to volume ratio (1/μm)
        """
        radius_um = diameter_um / 2
        # SA/V = 3/r for a sphere
        return 3 / radius_um

    def predict_diffusion_time(self, diameter_um: float, molecule_d: float = 1e-9) -> float:
        """
        Estimate diffusion time across cell using Einstein relation.

        t = x² / (2D)

        Args:
            diameter_um: Cell diameter in micrometers
            molecule_d: Diffusion coefficient in m²/s (default ~small molecule in cytoplasm)

        Returns:
            Characteristic diffusion time in milliseconds
        """
        x_m = diameter_um * 1e-6  # Convert to meters
        t_s = (x_m ** 2) / (2 * molecule_d)
        return t_s * 1000  # Convert to milliseconds

    def analyze_cell(self, diameter_um: float, name: str = "Unknown Cell",
                     cell_type: str = "other", organism: str = "homo_sapiens",
                     custom_density: float = None) -> Dict:
        """
        Perform comprehensive cell analysis.

        Args:
            diameter_um: Cell diameter in micrometers
            name: Cell name
            cell_type: Cell type
            organism: Organism name
            custom_density: Optional custom cell density

        Returns:
            Dictionary with all calculated properties
        """
        density = custom_density or self.CELL_DENSITY

        volume = self.calculate_cell_volume(diameter_um)
        mass = self.calculate_cell_mass(volume, density)
        metabolic_rate = self.calculate_metabolic_rate(mass)
        doubling_time = self.calculate_doubling_time(metabolic_rate, mass)
        atp_turnover = self.calculate_atp_turnover(metabolic_rate)
        o2_consumption = self.calculate_oxygen_consumption(metabolic_rate)
        surface_area = self.calculate_surface_area(diameter_um)
        sa_v_ratio = self.calculate_surface_volume_ratio(diameter_um)
        diffusion_time = self.predict_diffusion_time(diameter_um)

        return {
            "name": name,
            "type": cell_type,
            "organism": organism,
            "diameter_um": diameter_um,
            "volume_fL": round(volume, 2),
            "mass_pg": round(mass, 2),
            "density_g_cm3": density,
            "surface_area_um2": round(surface_area, 2),
            "surface_volume_ratio": round(sa_v_ratio, 4),
            "metabolic_rate_fW": round(metabolic_rate, 2),
            "atp_turnover_per_s": round(atp_turnover, 0),
            "o2_consumption_per_s": round(o2_consumption, 0),
            "estimated_doubling_time_hours": round(doubling_time, 1),
            "diffusion_time_ms": round(diffusion_time, 3),
            "derived_properties": {
                "metabolic_scaling": {
                    "law": "Kleiber's Law",
                    "equation": "B = B₀ × M^0.75",
                    "B0": self.B0,
                    "exponent": self.ALPHA
                },
                "geometric_model": "Spherical approximation"
            }
        }

    def compare_cells(self, cells: List[Dict]) -> Dict:
        """
        Compare multiple cells and provide comparative analysis.

        Args:
            cells: List of cell data dictionaries

        Returns:
            Comparative analysis dictionary
        """
        if not cells:
            return {}

        analysis = {
            "cell_count": len(cells),
            "size_range": {
                "min_diameter_um": min(c.get("diameter_um", 0) for c in cells),
                "max_diameter_um": max(c.get("diameter_um", 0) for c in cells)
            },
            "metabolic_range": {
                "min_rate_fW": min(c.get("metabolic_rate_fW", 0) for c in cells),
                "max_rate_fW": max(c.get("metabolic_rate_fW", 0) for c in cells)
            }
        }

        # Calculate averages
        total_volume = sum(c.get("volume_fL", 0) for c in cells)
        total_mass = sum(c.get("mass_pg", 0) for c in cells)
        total_metabolic = sum(c.get("metabolic_rate_fW", 0) for c in cells)

        analysis["averages"] = {
            "avg_volume_fL": round(total_volume / len(cells), 2),
            "avg_mass_pg": round(total_mass / len(cells), 2),
            "avg_metabolic_rate_fW": round(total_metabolic / len(cells), 2)
        }

        return analysis

    def predict_tissue_metabolic_rate(self, cell_composition: Dict[str, int],
                                       cell_masses: Dict[str, float]) -> float:
        """
        Predict tissue metabolic rate from cell composition.

        Args:
            cell_composition: Dict of {cell_type: count}
            cell_masses: Dict of {cell_type: mass_pg}

        Returns:
            Total metabolic rate in femtowatts
        """
        total_rate = 0
        for cell_type, count in cell_composition.items():
            mass = cell_masses.get(cell_type, 100)  # Default 100 pg
            rate = self.calculate_metabolic_rate(mass)
            total_rate += rate * count
        return total_rate
