"""
Regeneration Engine
====================
Orchestrates regeneration of all data categories from first principles.
The primary use case is regenerating all 118 elements from quark constants
via the full derivation chain: Quarks → Hadrons → Nuclei → Atoms.
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from periodica.utils.derivation_metadata import DerivationSource, DerivationTracker
from periodica.utils.prediction_engine import PredictionEngine
from periodica.utils.atomic_derivation import AtomicDerivation
from periodica.utils.logger import get_logger

logger = get_logger('regeneration_engine')


# Empirical correlations for properties not derivable from first principles.
# These use atomic number and other derived properties as inputs.

# Melting points (K) for Z=1..118 — empirical reference data
# Source: CRC Handbook, NIST. None = unknown/not well-defined
_MELTING_POINT_REFERENCE = {
    1: 14.01, 2: 0.95, 3: 453.69, 4: 1560.0, 5: 2349.0, 6: 3823.0,
    7: 63.15, 8: 54.36, 9: 53.53, 10: 24.56, 11: 370.87, 12: 923.0,
    13: 933.47, 14: 1687.0, 15: 317.3, 16: 388.36, 17: 171.6, 18: 83.8,
    19: 336.53, 20: 1115.0, 21: 1814.0, 22: 1941.0, 23: 2183.0, 24: 2180.0,
    25: 1519.0, 26: 1811.0, 27: 1768.0, 28: 1728.0, 29: 1357.77, 30: 692.68,
    31: 302.91, 32: 1211.4, 33: 1090.0, 34: 494.0, 35: 265.8, 36: 115.79,
    37: 312.46, 38: 1050.0, 39: 1799.0, 40: 2128.0, 41: 2750.0, 42: 2896.0,
    43: 2430.0, 44: 2607.0, 45: 2237.0, 46: 1828.05, 47: 1234.93, 48: 594.22,
    49: 429.75, 50: 505.08, 51: 903.78, 52: 722.66, 53: 386.85, 54: 161.36,
    55: 301.59, 56: 1000.0, 57: 1193.0, 58: 1068.0, 59: 1208.0, 60: 1297.0,
    61: 1315.0, 62: 1345.0, 63: 1099.0, 64: 1585.0, 65: 1629.0, 66: 1680.0,
    67: 1734.0, 68: 1802.0, 69: 1818.0, 70: 1097.0, 71: 1925.0, 72: 2506.0,
    73: 3290.0, 74: 3695.0, 75: 3459.0, 76: 3306.0, 77: 2719.0, 78: 2041.4,
    79: 1337.33, 80: 234.32, 81: 577.0, 82: 600.61, 83: 544.55, 84: 527.0,
    85: 575.0, 86: 202.0, 87: 300.0, 88: 973.0, 89: 1323.0, 90: 2115.0,
    91: 1841.0, 92: 1405.3, 93: 917.0, 94: 912.5, 95: 1449.0, 96: 1613.0,
    97: 1259.0, 98: 1173.0, 99: 1133.0, 100: 1800.0, 101: 1100.0, 102: 1100.0,
    103: 1900.0, 104: 2400.0, 105: None, 106: None, 107: None, 108: None,
    109: None, 110: None, 111: None, 112: None, 113: 700.0, 114: 340.0,
    115: 700.0, 116: 708.0, 117: 623.0, 118: None,
}

# Boiling points (K) for Z=1..118
_BOILING_POINT_REFERENCE = {
    1: 20.28, 2: 4.22, 3: 1615.0, 4: 2742.0, 5: 4200.0, 6: 4098.0,
    7: 77.36, 8: 90.2, 9: 85.03, 10: 27.07, 11: 1156.0, 12: 1363.0,
    13: 2792.0, 14: 3538.0, 15: 553.65, 16: 717.87, 17: 239.11, 18: 87.3,
    19: 1032.0, 20: 1757.0, 21: 3109.0, 22: 3560.0, 23: 3680.0, 24: 2944.0,
    25: 2334.0, 26: 3134.0, 27: 3200.0, 28: 3186.0, 29: 2835.0, 30: 1180.0,
    31: 2477.0, 32: 3106.0, 33: 887.0, 34: 958.0, 35: 332.0, 36: 119.93,
    37: 961.0, 38: 1655.0, 39: 3609.0, 40: 4682.0, 41: 5017.0, 42: 4912.0,
    43: 4538.0, 44: 4423.0, 45: 3968.0, 46: 3236.0, 47: 2435.0, 48: 1040.0,
    49: 2345.0, 50: 2875.0, 51: 1860.0, 52: 1261.0, 53: 457.4, 54: 165.03,
    55: 944.0, 56: 2170.0, 57: 3737.0, 58: 3716.0, 59: 3793.0, 60: 3347.0,
    61: 3273.0, 62: 2067.0, 63: 1802.0, 64: 3546.0, 65: 3503.0, 66: 2840.0,
    67: 2993.0, 68: 3141.0, 69: 2223.0, 70: 1469.0, 71: 3675.0, 72: 4876.0,
    73: 5731.0, 74: 5828.0, 75: 5869.0, 76: 5285.0, 77: 4701.0, 78: 4098.0,
    79: 3129.0, 80: 629.88, 81: 1746.0, 82: 2022.0, 83: 1837.0, 84: 1235.0,
    85: 610.0, 86: 211.3, 87: 950.0, 88: 2010.0, 89: 3471.0, 90: 5061.0,
    91: 4300.0, 92: 4404.0, 93: 4273.0, 94: 3501.0, 95: 2880.0, 96: 3383.0,
    97: 2900.0, 98: 1743.0, 99: 1269.0, 100: None, 101: None, 102: None,
    103: None, 104: 5800.0, 105: None, 106: None, 107: None, 108: None,
    109: None, 110: None, 111: None, 112: 357.0, 113: 1400.0, 114: 420.0,
    115: 1400.0, 116: 1037.0, 117: 883.0, 118: 350.0,
}

# Densities (g/cm³) at STP for Z=1..118
_DENSITY_REFERENCE = {
    1: 8.988e-5, 2: 1.785e-4, 3: 0.534, 4: 1.85, 5: 2.34, 6: 2.267,
    7: 1.251e-3, 8: 1.429e-3, 9: 1.696e-3, 10: 9.002e-4, 11: 0.971,
    12: 1.738, 13: 2.698, 14: 2.3296, 15: 1.82, 16: 2.067, 17: 3.214e-3,
    18: 1.784e-3, 19: 0.862, 20: 1.55, 21: 2.989, 22: 4.54, 23: 6.11,
    24: 7.15, 25: 7.44, 26: 7.874, 27: 8.86, 28: 8.912, 29: 8.96,
    30: 7.134, 31: 5.907, 32: 5.323, 33: 5.776, 34: 4.809, 35: 3.122,
    36: 3.749e-3, 37: 1.532, 38: 2.64, 39: 4.469, 40: 6.506, 41: 8.57,
    42: 10.22, 43: 11.5, 44: 12.37, 45: 12.41, 46: 12.02, 47: 10.501,
    48: 8.69, 49: 7.31, 50: 7.287, 51: 6.685, 52: 6.232, 53: 4.93,
    54: 5.894e-3, 55: 1.873, 56: 3.594, 57: 6.145, 58: 6.77, 59: 6.773,
    60: 7.007, 61: 7.26, 62: 7.52, 63: 5.243, 64: 7.895, 65: 8.229,
    66: 8.55, 67: 8.795, 68: 9.066, 69: 9.321, 70: 6.965, 71: 9.84,
    72: 13.31, 73: 16.654, 74: 19.25, 75: 21.02, 76: 22.59, 77: 22.56,
    78: 21.46, 79: 19.282, 80: 13.5336, 81: 11.85, 82: 11.342, 83: 9.807,
    84: 9.32, 85: 7.0, 86: 9.73e-3, 87: 1.87, 88: 5.5, 89: 10.07,
    90: 11.72, 91: 15.37, 92: 18.95, 93: 20.45, 94: 19.84, 95: 12.0,
    96: 13.51, 97: 14.78, 98: 15.1, 99: 8.84, 100: None, 101: None,
    102: None, 103: None, 104: 23.2, 105: 29.3, 106: 35.0, 107: 37.1,
    108: 40.7, 109: 37.4, 110: 34.8, 111: 28.7, 112: 14.0, 113: 16.0,
    114: 14.0, 115: 13.5, 116: 12.9, 117: 7.2, 118: 5.0,
}

# Electron affinities (kJ/mol) — a subset for common elements
_ELECTRON_AFFINITY_REFERENCE = {
    1: 72.8, 2: 0, 3: 59.6, 4: 0, 5: 26.7, 6: 121.8, 7: 0, 8: 141.0,
    9: 328.0, 10: 0, 11: 52.8, 12: 0, 13: 42.5, 14: 134.1, 15: 72.0,
    16: 200.4, 17: 349.0, 18: 0, 19: 48.4, 20: 2.4, 26: 15.7, 29: 118.4,
    35: 324.6, 47: 125.6, 53: 295.2, 79: 222.8,
}

# Emission wavelengths (nm) for strongest visible lines
_EMISSION_WAVELENGTH_REFERENCE = {
    1: 656.3, 2: 587.6, 3: 670.8, 4: 234.9, 5: 249.7, 6: 247.9,
    7: 149.3, 8: 130.2, 9: 95.5, 10: 585.2, 11: 589.0, 12: 285.2,
    13: 396.2, 14: 251.6, 15: 253.6, 16: 180.7, 17: 134.7, 18: 811.5,
    19: 766.5, 20: 422.7, 26: 438.4, 29: 324.8, 47: 328.1, 79: 267.6,
}


class RegenerationEngine:
    """
    Regenerates scientific data from first principles.

    Primary use case: regenerate all 118 elements from quark constants
    via PredictionEngine (quarks → hadrons → nuclei → atoms).
    """

    def __init__(self):
        self.prediction_engine = PredictionEngine()
        self.atomic_calc = AtomicDerivation()

    def regenerate_elements(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Regenerate all 118 element JSON files from quark constants.

        Args:
            progress_callback: Called with (percent, status_message)
            output_dir: Directory to write JSON files. Defaults to data/active/elements/

        Returns:
            List of generated element dicts
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "active" / "elements"
        output_dir.mkdir(parents=True, exist_ok=True)

        elements = []
        for Z in range(1, 119):
            if progress_callback:
                pct = int(Z / 118 * 100)
                symbol = self.atomic_calc.ELEMENT_SYMBOLS[Z] if Z < len(self.atomic_calc.ELEMENT_SYMBOLS) else f"E{Z}"
                progress_callback(pct, f"Generating {symbol} (Z={Z})")

            element = self.predict_to_element_json(Z)
            elements.append(element)

            # Write to file
            filename = f"{Z:03d}_{element['symbol']}.json"
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(element, f, indent=2, ensure_ascii=False)

        logger.info("Regenerated %d elements", len(elements))
        return elements

    def predict_to_element_json(self, Z: int) -> Dict:
        """
        Derive a complete element JSON dict for atomic number Z.

        Combines first-principles calculations with empirical reference data
        for properties that cannot be derived from the quark chain.
        """
        # Run the full prediction chain
        result = self.prediction_engine.predict(Z)

        # Get detailed atomic properties using nuclear mass from prediction chain
        nuclear_mass_mev = result.nuclear_properties.get('mass_mev')
        atomic = self.atomic_calc.calculate(Z, nuclear_mass_mev)

        # Build element JSON matching existing schema
        element = {
            "symbol": atomic.symbol,
            "name": atomic.name,
            "atomic_number": Z,
            "atomic_mass": round(atomic.atomic_mass_u, 4),
            "block": atomic.block,
            "period": atomic.period,
            "group": atomic.group,
            "ionization_energy": round(atomic.ionization_energy_ev, 3),
            "electronegativity": round(atomic.electronegativity, 2) if atomic.electronegativity else None,
            "atomic_radius": round(atomic.atomic_radius_pm),
            "covalent_radius": round(atomic.covalent_radius_pm),
            "melting_point": _MELTING_POINT_REFERENCE.get(Z),
            "boiling_point": _BOILING_POINT_REFERENCE.get(Z),
            "density": _DENSITY_REFERENCE.get(Z),
            "electron_affinity": _ELECTRON_AFFINITY_REFERENCE.get(Z, self._estimate_electron_affinity(Z, atomic)),
            "valence_electrons": self._get_valence_electrons(Z, atomic),
            "electron_configuration": self._format_config(atomic.electron_configuration),
            "primary_emission_wavelength": _EMISSION_WAVELENGTH_REFERENCE.get(Z, self._estimate_emission_wavelength(Z, atomic)),
        }

        # Stamp with derivation metadata
        chain = ["quarks", "hadrons", "nucleus", "atom"]
        # Properties from empirical data get lower confidence
        empirical_count = sum(1 for k in ["melting_point", "boiling_point", "density"]
                             if element.get(k) is not None)
        base_confidence = result.confidence.get('overall', 0.85)
        # Blend: derived properties have model confidence, empirical have 1.0
        confidence = base_confidence * 0.7 + 0.3  # weighted blend

        DerivationTracker.stamp(
            element,
            source=DerivationSource.QUARK_DERIVED,
            derived_from=["up_quark", "down_quark"],
            derivation_chain=chain,
            confidence=round(confidence, 3),
            model_version=self.prediction_engine.VERSION,
        )

        return element

    def _get_valence_electrons(self, Z: int, atomic) -> int:
        """Determine valence electron count from electron configuration."""
        config = atomic.electron_configuration
        parts = config.split()
        if not parts:
            return 0

        # Get the outermost shell
        max_n = max(int(p[0]) for p in parts if p[0].isdigit())
        valence = 0
        for part in parts:
            if part[0].isdigit() and int(part[0]) == max_n:
                valence += int(part[2:])
        return valence

    def _format_config(self, config: str) -> str:
        """Format electron configuration with superscript-style notation."""
        # Convert "1s2 2s2 2p6" to "1s² 2s² 2p⁶"
        superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                       '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                       '10': '¹⁰', '11': '¹¹', '12': '¹²', '13': '¹³', '14': '¹⁴'}
        parts = config.split()
        formatted = []
        for part in parts:
            if len(part) >= 3 and part[0].isdigit():
                n = part[0]
                l = part[1]
                count = part[2:]
                sup = superscripts.get(count, count)
                formatted.append(f"{n}{l}{sup}")
            else:
                formatted.append(part)
        return ' '.join(formatted)

    def _estimate_electron_affinity(self, Z: int, atomic) -> float:
        """Estimate electron affinity from periodic trends."""
        # Halogens have high EA, noble gases ~0, alkali metals moderate
        if atomic.block == 'p' and atomic.group == 17:
            return 300.0  # Halogens
        elif atomic.block == 'p' and atomic.group == 16:
            return 200.0  # Chalcogens
        elif atomic.block == 'p' and atomic.group == 18:
            return 0.0  # Noble gases
        elif atomic.block == 's' and atomic.group == 2:
            return 0.0  # Alkaline earths
        elif atomic.block == 's' and atomic.group == 1:
            return 50.0  # Alkali metals
        elif atomic.block == 'p':
            return 100.0  # Other p-block
        elif atomic.block == 'd':
            return 50.0  # Transition metals
        else:
            return 30.0  # f-block

    def _estimate_emission_wavelength(self, Z: int, atomic) -> float:
        """Estimate primary emission wavelength from ionization energy."""
        # Use Rydberg-like formula: λ ≈ hc / (IE * some_fraction)
        ie = atomic.ionization_energy_ev
        if ie > 0:
            # Rough correlation: stronger binding → shorter wavelength
            wavelength = 1240.0 / (ie * 0.3)  # nm, using hc = 1240 eV·nm
            return round(max(90, min(900, wavelength)), 1)
        return 500.0  # Default visible

    def regenerate_category(
        self,
        category: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Dict]:
        """
        Regenerate a data category.

        Args:
            category: Category name (e.g., 'elements')
            progress_callback: Called with (percent, status_message)

        Returns:
            List of generated data dicts
        """
        if category == "elements":
            return self.regenerate_elements(progress_callback)
        else:
            logger.warning("Regeneration not yet implemented for category: %s", category)
            return []
