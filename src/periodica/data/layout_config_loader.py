"""
Layout Configuration Loader
============================

Loads and provides access to layout configuration data from JSON files.
This ensures all layouts use data-driven positioning and styling.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class LayoutConfigLoader:
    """
    Singleton loader for layout configuration data.

    Provides data-driven configuration for:
    - Card/cell dimensions
    - Spacing and margins
    - Category/type orderings
    - Color schemes
    - Axis ranges and thresholds
    """

    _instance: Optional['LayoutConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        config_path = Path(__file__).parent / 'layout_config' / 'layout_config.json'

        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = json.load(f)
        else:
            # Fallback to default config
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            "global": {
                "min_card_size": 40,
                "max_card_size": 200,
                "default_spacing": 15,
                "default_padding": 30,
                "default_margin": 50
            },
            "quarks": {"card_size": {"default": 70}},
            "subatomic": {"card_size": {"width": 140, "height": 180}},
            "molecules": {"card_size": {"width": 150, "height": 170}},
            "alloys": {"card_size": {"width": 160, "height": 180}}
        }

    def reload(self) -> None:
        """Force reload configuration from file."""
        self._load_config()

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config or {}

    def get_global(self, key: str, default: Any = None) -> Any:
        """Get global configuration value."""
        return self.config.get('global', {}).get(key, default)

    def get_quark_config(self, *keys: str, default: Any = None) -> Any:
        """Get quark layout configuration value by nested keys."""
        return self._get_nested('quarks', keys, default)

    def get_subatomic_config(self, *keys: str, default: Any = None) -> Any:
        """Get subatomic layout configuration value by nested keys."""
        return self._get_nested('subatomic', keys, default)

    def get_molecule_config(self, *keys: str, default: Any = None) -> Any:
        """Get molecule layout configuration value by nested keys."""
        return self._get_nested('molecules', keys, default)

    def get_alloy_config(self, *keys: str, default: Any = None) -> Any:
        """Get alloy layout configuration value by nested keys."""
        return self._get_nested('alloys', keys, default)

    def get_color_scheme(self, category: str) -> Dict[str, str]:
        """Get color scheme for a category."""
        return self.config.get('color_schemes', {}).get(category, {})

    def _get_nested(self, category: str, keys: tuple, default: Any) -> Any:
        """Get nested value from config."""
        value = self.config.get(category, {})
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    # Convenience methods for common operations

    def get_card_size(self, category: str) -> Dict[str, int]:
        """Get card size configuration for a category."""
        return self.config.get(category, {}).get('card_size', {
            'width': 150, 'height': 170, 'default': 70
        })

    def get_spacing(self, category: str) -> Dict[str, int]:
        """Get spacing configuration for a category."""
        default = {'card': 15, 'section': 40, 'group': 40, 'header': 40}
        return self.config.get(category, {}).get('spacing', default)

    def get_margins(self, category: str) -> Dict[str, int]:
        """Get margin configuration for a category."""
        default = {'top': 80, 'right': 50, 'bottom': 50, 'left': 50}
        return self.config.get(category, {}).get('margins', default)

    def get_ordering(self, category: str, order_type: str) -> list:
        """Get ordering list for a category and type."""
        key = f'{order_type}_order'
        return self.config.get(category, {}).get(key, [])

    def calculate_responsive_card_size(
        self,
        category: str,
        item_count: int,
        available_width: int,
        available_height: int
    ) -> Dict[str, int]:
        """
        Calculate responsive card size based on item count and available space.

        Uses the configuration min/max values and scales appropriately.
        """
        card_config = self.get_card_size(category)
        spacing = self.get_spacing(category)
        margins = self.get_margins(category)

        # Get min/max from config
        min_width = card_config.get('min_width', card_config.get('min', 80))
        max_width = card_config.get('max_width', card_config.get('max', 200))
        default_width = card_config.get('width', card_config.get('default', 150))

        # Calculate how many cards can fit
        usable_width = available_width - margins.get('left', 50) - margins.get('right', 50)
        card_spacing = spacing.get('card', 15)

        # Estimate columns
        cols = max(1, int(usable_width / (default_width + card_spacing)))

        # Adjust card size based on item count
        if item_count <= cols:
            # Few items - use larger cards
            width = min(max_width, default_width * 1.2)
        elif item_count > cols * 5:
            # Many items - use smaller cards
            width = max(min_width, default_width * 0.8)
        else:
            width = default_width

        # Maintain aspect ratio
        aspect = card_config.get('height', 170) / card_config.get('width', 150)
        height = int(width * aspect)

        return {'width': int(width), 'height': height}


# Global instance
_loader: Optional[LayoutConfigLoader] = None


def get_layout_config() -> LayoutConfigLoader:
    """Get the global layout config loader instance."""
    global _loader
    if _loader is None:
        _loader = LayoutConfigLoader()
    return _loader


# Convenience functions
def get_quark_config(*keys, default=None):
    return get_layout_config().get_quark_config(*keys, default=default)

def get_subatomic_config(*keys, default=None):
    return get_layout_config().get_subatomic_config(*keys, default=default)

def get_molecule_config(*keys, default=None):
    return get_layout_config().get_molecule_config(*keys, default=default)

def get_alloy_config(*keys, default=None):
    return get_layout_config().get_alloy_config(*keys, default=default)
