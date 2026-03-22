"""
Quark Constants Provider
=========================
Single source of truth for quark properties in the Periodics system.
Reads quark values from JSON files in data/active/quarks/ instead of
hardcoded Python constants. When a GUT updates quark JSON files,
calling reload() propagates the changes to all downstream calculations.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

from periodica.utils.logger import get_logger

logger = get_logger('quark_constants')


class QuarkConstantProvider:
    """
    Singleton provider that reads quark properties from JSON files.

    Replaces hardcoded QUARKS dicts in prediction_engine.py and chain.py.
    """

    _instance = None

    # Map quark symbols to JSON filenames
    QUARK_FILES = {
        'u': 'UpQuark.json',
        'd': 'DownQuark.json',
        's': 'StrangeQuark.json',
        'c': 'CharmQuark.json',
        'b': 'BottomQuark.json',
        't': 'TopQuark.json',
    }

    # Constituent quark masses (fitted to hadron spectroscopy).
    # These are QCD-dressed masses, much larger than current/bare masses.
    # Light quarks get ~1/3 of nucleon mass from gluon field energy.
    # The ratio between constituent and current mass is the key QCD dressing factor.
    _DEFAULT_CONSTITUENT_RATIOS = {
        'u': 155.6,   # 336 / 2.16 ≈ 155.6x dressing
        'd': 72.8,    # 340 / 4.67 ≈ 72.8x dressing
        's': 5.2,     # 486 / 93.4 ≈ 5.2x dressing
    }

    # For heavy quarks, constituent mass ≈ current mass + QCD correction
    _HEAVY_QUARK_QCD_CORRECTION = {
        'c': 200.0,   # MeV added to current mass
        'b': 100.0,
        't': 50.0,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._quarks = {}
            cls._instance._loaded = False
        return cls._instance

    def _ensure_loaded(self):
        """Lazy-load quark data on first access."""
        if not self._loaded:
            self._load_quarks()

    def _load_quarks(self):
        """Load quark properties from JSON files."""
        quarks_dir = Path(__file__).parent.parent / "data" / "active" / "quarks"

        for symbol, filename in self.QUARK_FILES.items():
            filepath = quarks_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Strip JS-style comments
                    content = re.sub(r'//.*?(?=\n|$)', '', content)
                    data = json.loads(content)
                    self._quarks[symbol] = {
                        'mass_mev': data.get('Mass_MeVc2', 0.0),
                        'charge': data.get('Charge_e', 0.0),
                        'spin': data.get('Spin_hbar', 0.5),
                        'baryon': data.get('BaryonNumber_B', 1/3),
                        'I3': data.get('Isospin_I3', 0.0),
                        'name': data.get('Name', ''),
                        'symbol': data.get('Symbol', symbol),
                    }
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Failed to load quark %s: %s", filename, e)
            else:
                logger.warning("Quark file not found: %s", filepath)

        self._loaded = True
        logger.info("Loaded %d quark definitions from JSON", len(self._quarks))

    def reload(self):
        """Force reload of quark data from JSON files. Call after modifying quark JSON."""
        self._quarks.clear()
        self._loaded = False
        self._ensure_loaded()

    def get_quark(self, flavor: str) -> Dict:
        """
        Get quark properties by flavor symbol.

        Args:
            flavor: Quark flavor symbol ('u', 'd', 's', 'c', 'b', 't')

        Returns:
            Dict with keys: mass_mev, charge, spin, baryon, I3, name, symbol
        """
        self._ensure_loaded()
        return self._quarks.get(flavor, {})

    def get_quark_mass(self, flavor: str) -> float:
        """Get the current (bare) quark mass in MeV/c²."""
        quark = self.get_quark(flavor)
        return quark.get('mass_mev', 0.0)

    def get_constituent_mass(self, flavor: str) -> float:
        """
        Get the constituent (dressed) quark mass in MeV.

        For light quarks (u, d, s), constituent mass is derived from
        the current mass using QCD dressing ratios.
        For heavy quarks (c, b, t), it's current mass + QCD correction.
        """
        current_mass = self.get_quark_mass(flavor)

        if flavor in self._DEFAULT_CONSTITUENT_RATIOS:
            return current_mass * self._DEFAULT_CONSTITUENT_RATIOS[flavor]
        elif flavor in self._HEAVY_QUARK_QCD_CORRECTION:
            return current_mass + self._HEAVY_QUARK_QCD_CORRECTION[flavor]
        else:
            return current_mass

    def get_all_quarks(self) -> Dict[str, Dict]:
        """Get all loaded quark properties."""
        self._ensure_loaded()
        return self._quarks.copy()

    def get_quark_properties(self) -> Dict[str, Dict]:
        """
        Get quark properties in the format used by DerivationChain and PredictionEngine.

        Returns dict like: {'u': {'mass_mev': 2.2, 'charge': 0.667, 'spin': 0.5}, ...}
        """
        self._ensure_loaded()
        result = {}
        for flavor, props in self._quarks.items():
            result[flavor] = {
                'mass_mev': props['mass_mev'],
                'charge': props['charge'],
                'spin': props['spin'],
            }
        return result

    def get_constituent_masses(self) -> Dict[str, float]:
        """
        Get constituent masses in the format used by DerivationChain.

        Returns dict like: {'u': 336.0, 'd': 340.0, 's': 486.0}
        """
        self._ensure_loaded()
        result = {}
        for flavor in self._quarks:
            result[flavor] = self.get_constituent_mass(flavor)
        return result


# Module-level convenience function
def get_quark_provider() -> QuarkConstantProvider:
    """Get the singleton QuarkConstantProvider instance."""
    return QuarkConstantProvider()
