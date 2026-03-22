"""
Test data integrity across all JSON data files.
Validates that all asset data in data/active/ is well-formed.
"""
import json
import re
import pytest
from pathlib import Path

from periodica.data.validation import (
    validate_required_fields,
    validate_numeric_fields,
    validate_json_data,
    REQUIRED_FIELDS,
    NUMERIC_FIELDS,
)

DATA_DIR = Path(__file__).parent.parent / "src" / "periodica" / "data" / "active"


def _load_json_safe(filepath: Path) -> dict:
    """Load a JSON file, stripping JavaScript-style comments if present."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Strip // line comments (used in some quark/subatomic JSON files)
    content = re.sub(r'//.*?(?=\n|$)', '', content)
    return json.loads(content)


def _load_json_files(category: str):
    """Load all JSON files from a data/active/ subdirectory."""
    category_dir = DATA_DIR / category
    if not category_dir.exists():
        return []
    results = []
    for json_file in sorted(category_dir.glob("*.json")):
        data = _load_json_safe(json_file)
        results.append((json_file.name, data))
    return results


# ==================== JSON Parse Tests ====================

class TestJsonParsing:
    """Verify all JSON files parse without errors."""

    @pytest.mark.parametrize("category", [
        "elements", "quarks", "antiquarks", "subatomic",
        "molecules", "alloys", "materials",
        "amino_acids", "proteins", "nucleic_acids",
        "cells", "cell_components", "biological_materials",
    ])
    def test_all_json_files_parse(self, category):
        """Every JSON file in data/active/{category} should parse cleanly."""
        category_dir = DATA_DIR / category
        if not category_dir.exists():
            pytest.skip(f"No {category} directory")

        json_files = list(category_dir.glob("*.json"))
        assert len(json_files) > 0, f"No JSON files found in {category}"

        for json_file in json_files:
            data = _load_json_safe(json_file)
            assert isinstance(data, dict), f"{json_file.name} is not a JSON object"

    def test_no_empty_json_files(self):
        """No JSON file should be empty or contain only whitespace."""
        for category_dir in DATA_DIR.iterdir():
            if not category_dir.is_dir():
                continue
            for json_file in category_dir.glob("*.json"):
                content = json_file.read_text(encoding='utf-8').strip()
                assert len(content) > 0, f"Empty JSON file: {json_file}"


# ==================== Required Fields Tests ====================

class TestRequiredFields:
    """Verify all data records have required fields."""

    def test_elements_required_fields(self):
        files = _load_json_files("elements")
        if not files:
            pytest.skip("No element data")
        for filename, data in files:
            missing = validate_required_fields(data, "elements", filename)
            assert missing == [], f"{filename} missing: {missing}"

    def test_quarks_required_fields(self):
        files = _load_json_files("quarks")
        if not files:
            pytest.skip("No quark data")
        for filename, data in files:
            missing = validate_required_fields(data, "quarks", filename)
            assert missing == [], f"{filename} missing: {missing}"

    def test_subatomic_required_fields(self):
        files = _load_json_files("subatomic")
        if not files:
            pytest.skip("No subatomic data")
        for filename, data in files:
            missing = validate_required_fields(data, "subatomic", filename)
            assert missing == [], f"{filename} missing: {missing}"

    def test_molecules_required_fields(self):
        files = _load_json_files("molecules")
        if not files:
            pytest.skip("No molecule data")
        for filename, data in files:
            missing = validate_required_fields(data, "molecules", filename)
            assert missing == [], f"{filename} missing: {missing}"

    def test_alloys_required_fields(self):
        files = _load_json_files("alloys")
        if not files:
            pytest.skip("No alloy data")
        for filename, data in files:
            missing = validate_required_fields(data, "alloys", filename)
            assert missing == [], f"{filename} missing: {missing}"

    def test_materials_required_fields(self):
        files = _load_json_files("materials")
        if not files:
            pytest.skip("No material data")
        for filename, data in files:
            missing = validate_required_fields(data, "materials", filename)
            assert missing == [], f"{filename} missing: {missing}"


# ==================== Numeric Fields Tests ====================

class TestNumericFields:
    """Verify numeric fields contain numeric values."""

    def test_elements_numeric_fields(self):
        files = _load_json_files("elements")
        if not files:
            pytest.skip("No element data")
        for filename, data in files:
            errors = validate_numeric_fields(data, "elements", filename)
            assert errors == [], f"{filename} non-numeric: {errors}"

    def test_quarks_numeric_fields(self):
        files = _load_json_files("quarks")
        if not files:
            pytest.skip("No quark data")
        for filename, data in files:
            errors = validate_numeric_fields(data, "quarks", filename)
            assert errors == [], f"{filename} non-numeric: {errors}"

    def test_molecules_numeric_fields(self):
        files = _load_json_files("molecules")
        if not files:
            pytest.skip("No molecule data")
        for filename, data in files:
            errors = validate_numeric_fields(data, "molecules", filename)
            assert errors == [], f"{filename} non-numeric: {errors}"


# ==================== Validation Utility Tests ====================

class TestValidationUtils:
    """Test the validation utility functions themselves."""

    def test_validate_required_fields_all_present(self):
        data = {"Name": "Test", "Formula": "H2O", "MolecularMass_amu": 18.0,
                "BondType": "Covalent", "Geometry": "Bent"}
        missing = validate_required_fields(data, "molecules", "test.json")
        assert missing == []

    def test_validate_required_fields_missing(self):
        data = {"Name": "Test"}
        missing = validate_required_fields(data, "molecules", "test.json")
        assert "Formula" in missing
        assert "MolecularMass_amu" in missing

    def test_validate_required_fields_none_value(self):
        data = {"Name": None, "Formula": "H2O", "MolecularMass_amu": 18.0,
                "BondType": "Covalent", "Geometry": "Bent"}
        missing = validate_required_fields(data, "molecules", "test.json")
        assert "Name" in missing

    def test_validate_numeric_fields_valid(self):
        data = {"atomic_number": 1, "atomic_mass": 1.008, "density": 0.09}
        errors = validate_numeric_fields(data, "elements", "test.json")
        assert errors == []

    def test_validate_numeric_fields_string_value(self):
        data = {"atomic_number": "not_a_number", "atomic_mass": 1.008}
        errors = validate_numeric_fields(data, "elements", "test.json")
        assert "atomic_number" in errors

    def test_validate_numeric_fields_none_skipped(self):
        data = {"atomic_number": None, "atomic_mass": 1.008}
        errors = validate_numeric_fields(data, "elements", "test.json")
        assert errors == []

    def test_validate_json_data_pass(self):
        data = {"symbol": "H", "name": "Hydrogen", "atomic_number": 1,
                "block": "s", "period": 1}
        assert validate_json_data(data, "elements", "test.json") is True

    def test_validate_json_data_fail(self):
        data = {"symbol": "H"}
        assert validate_json_data(data, "elements", "test.json") is False
