"""
Data Manager
Handles add, edit, remove, and reset operations for JSON data files.
Manages defaults and active data directories.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class DataCategory(Enum):
    """Categories of data that can be managed.

    Values correspond to subdirectory names under data/active/ and data/defaults/.
    """
    # Physical sciences
    ELEMENTS = "elements"
    QUARKS = "quarks"
    ANTIQUARKS = "antiquarks"
    SUBATOMIC = "subatomic"
    MOLECULES = "molecules"
    ALLOYS = "alloys"
    MATERIALS = "materials"
    # Biological sciences
    AMINO_ACIDS = "amino_acids"
    PROTEINS = "proteins"
    NUCLEIC_ACIDS = "nucleic_acids"
    CELL_COMPONENTS = "cell_components"
    CELLS = "cells"
    BIOMATERIALS = "biological_materials"
    TISSUES = "tissues"


class DataManager:
    """
    Manages JSON data files with add, edit, remove, and reset functionality.
    Maintains separate defaults and active directories.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the data manager.

        Args:
            base_dir: Base directory for data. Defaults to 'data' relative to this file.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent

        self.base_dir = Path(base_dir)
        self.defaults_dir = self.base_dir / "defaults"
        self.active_dir = self.base_dir / "active"

        # Ensure directories exist
        self._ensure_directories()

        # Callbacks for data change notifications
        self._change_callbacks: Dict[DataCategory, List[Callable]] = {
            cat: [] for cat in DataCategory
        }

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for category in DataCategory:
            (self.defaults_dir / category.value).mkdir(parents=True, exist_ok=True)
            (self.active_dir / category.value).mkdir(parents=True, exist_ok=True)

    def get_active_path(self, category: DataCategory) -> Path:
        """Get the active data directory for a category"""
        return self.active_dir / category.value

    def get_defaults_path(self, category: DataCategory) -> Path:
        """Get the defaults data directory for a category"""
        return self.defaults_dir / category.value

    # ==================== Read Operations ====================

    def list_items(self, category: DataCategory) -> List[str]:
        """List all items (filenames without .json) in a category"""
        active_path = self.get_active_path(category)
        return sorted([f.stem for f in active_path.glob("*.json")])

    def get_item(self, category: DataCategory, name: str) -> Optional[Dict]:
        """
        Get a single item by name.

        Args:
            category: Data category
            name: Item name (filename without .json extension)

        Returns:
            Item data as dictionary, or None if not found
        """
        filepath = self._find_file(category, name)
        if filepath and filepath.exists():
            return self._load_json(filepath)
        return None

    def get_all_items(self, category: DataCategory) -> List[Dict]:
        """Get all items in a category"""
        items = []
        active_path = self.get_active_path(category)
        for filepath in sorted(active_path.glob("*.json")):
            data = self._load_json(filepath)
            if data:
                data['_filename'] = filepath.stem
                items.append(data)
        return items

    def _find_file(self, category: DataCategory, name: str) -> Optional[Path]:
        """Find a file by name (handles various naming patterns)"""
        active_path = self.get_active_path(category)

        # Try exact match
        exact = active_path / f"{name}.json"
        if exact.exists():
            return exact

        # Try pattern match (e.g., "001_H" for "H")
        for filepath in active_path.glob("*.json"):
            if name in filepath.stem or filepath.stem.endswith(f"_{name}"):
                return filepath

        return None

    def _load_json(self, filepath: Path) -> Optional[Dict]:
        """Load JSON file, handling JavaScript-style comments"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            # Remove JavaScript-style comments
            content = re.sub(r'//.*?(?=\n|$)', '', content)
            return json.loads(content)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    # ==================== Write Operations ====================

    def add_item(self, category: DataCategory, name: str, data: Dict) -> bool:
        """
        Add a new item.

        Args:
            category: Data category
            name: Item name (will be used as filename)
            data: Item data

        Returns:
            True if successful, False otherwise
        """
        active_path = self.get_active_path(category)
        filepath = active_path / f"{name}.json"

        if filepath.exists():
            print(f"Item '{name}' already exists in {category.value}")
            return False

        return self._save_json(filepath, data, category)

    def edit_item(self, category: DataCategory, name: str, data: Dict) -> bool:
        """
        Edit an existing item.

        Args:
            category: Data category
            name: Item name
            data: Updated item data

        Returns:
            True if successful, False otherwise
        """
        filepath = self._find_file(category, name)
        if not filepath:
            print(f"Item '{name}' not found in {category.value}")
            return False

        return self._save_json(filepath, data, category)

    def remove_item(self, category: DataCategory, name: str) -> bool:
        """
        Remove an item.

        Args:
            category: Data category
            name: Item name

        Returns:
            True if successful, False otherwise
        """
        filepath = self._find_file(category, name)
        if not filepath:
            print(f"Item '{name}' not found in {category.value}")
            return False

        try:
            filepath.unlink()
            self._notify_change(category)
            return True
        except Exception as e:
            print(f"Error removing {filepath}: {e}")
            return False

    def _save_json(self, filepath: Path, data: Dict, category: DataCategory) -> bool:
        """Save data to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._notify_change(category)
            return True
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
            return False

    # ==================== Reset Operations ====================

    def reset_category(self, category: DataCategory) -> bool:
        """
        Reset a category to defaults by copying all default files to active.

        Args:
            category: Data category to reset

        Returns:
            True if successful, False otherwise
        """
        defaults_path = self.get_defaults_path(category)
        active_path = self.get_active_path(category)

        try:
            # Remove all files in active directory
            for filepath in active_path.glob("*.json"):
                filepath.unlink()

            # Copy all files from defaults
            for filepath in defaults_path.glob("*.json"):
                shutil.copy2(filepath, active_path / filepath.name)

            self._notify_change(category)
            return True
        except Exception as e:
            print(f"Error resetting {category.value}: {e}")
            return False

    def reset_item(self, category: DataCategory, name: str) -> bool:
        """
        Reset a single item to its default.

        Args:
            category: Data category
            name: Item name

        Returns:
            True if successful, False otherwise
        """
        defaults_path = self.get_defaults_path(category)
        active_path = self.get_active_path(category)

        # Find the default file
        default_file = None
        for filepath in defaults_path.glob("*.json"):
            if name in filepath.stem or filepath.stem.endswith(f"_{name}"):
                default_file = filepath
                break

        if not default_file:
            print(f"No default found for '{name}' in {category.value}")
            return False

        try:
            shutil.copy2(default_file, active_path / default_file.name)
            self._notify_change(category)
            return True
        except Exception as e:
            print(f"Error resetting {name}: {e}")
            return False

    def reset_all(self) -> bool:
        """Reset all categories to defaults"""
        success = True
        for category in DataCategory:
            if not self.reset_category(category):
                success = False
        return success

    # ==================== Change Notifications ====================

    def register_change_callback(self, category: DataCategory, callback: Callable):
        """Register a callback to be notified when data changes"""
        self._change_callbacks[category].append(callback)

    def unregister_change_callback(self, category: DataCategory, callback: Callable):
        """Unregister a change callback"""
        if callback in self._change_callbacks[category]:
            self._change_callbacks[category].remove(callback)

    def _notify_change(self, category: DataCategory):
        """Notify all registered callbacks of a change"""
        for callback in self._change_callbacks[category]:
            try:
                callback()
            except Exception as e:
                print(f"Error in change callback: {e}")

    # ==================== Utility Methods ====================

    def get_item_count(self, category: DataCategory) -> int:
        """Get the number of items in a category"""
        return len(list(self.get_active_path(category).glob("*.json")))

    def has_changes(self, category: DataCategory) -> bool:
        """Check if active data differs from defaults"""
        defaults_path = self.get_defaults_path(category)
        active_path = self.get_active_path(category)

        default_files = set(f.name for f in defaults_path.glob("*.json"))
        active_files = set(f.name for f in active_path.glob("*.json"))

        # Check for added or removed files
        if default_files != active_files:
            return True

        # Check for modified files
        for filename in default_files:
            default_data = self._load_json(defaults_path / filename)
            active_data = self._load_json(active_path / filename)
            if default_data != active_data:
                return True

        return False

    def export_item(self, category: DataCategory, name: str, export_path: str) -> bool:
        """Export an item to a specified path"""
        data = self.get_item(category, name)
        if data:
            try:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"Error exporting: {e}")
        return False

    def import_item(self, category: DataCategory, import_path: str, name: Optional[str] = None) -> bool:
        """Import an item from a file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if name is None:
                name = Path(import_path).stem

            return self.add_item(category, name, data)
        except Exception as e:
            print(f"Error importing: {e}")
            return False


# Global data manager instance
_global_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """Get the global data manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = DataManager()
    return _global_manager
