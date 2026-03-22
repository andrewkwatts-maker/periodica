"""
Data validation utilities for Periodics JSON data files.
Provides consistent validation across all data loaders.
"""
from typing import Dict, List, Optional

from periodica.utils.logger import get_logger

logger = get_logger('data.validation')


# Required fields per asset type
REQUIRED_FIELDS = {
    'elements': ['symbol', 'name', 'atomic_number', 'block', 'period'],
    'quarks': ['Name'],
    'subatomic': ['Name', 'Type'],
    'molecules': ['Name', 'Formula', 'MolecularMass_amu', 'BondType', 'Geometry'],
    'alloys': ['Name', 'Category', 'Components'],
    'materials': ['Name'],
    'amino_acids': ['Name'],
    'proteins': ['Name'],
    'nucleic_acids': ['Name'],
    'cells': ['Name'],
    'cell_components': ['Name'],
    'biological_materials': ['Name'],
}

# Numeric fields per asset type (fields that should contain int or float)
NUMERIC_FIELDS = {
    'elements': [
        'atomic_number', 'atomic_mass', 'ionization_energy',
        'electronegativity', 'atomic_radius', 'melting_point',
        'boiling_point', 'density', 'electron_affinity',
    ],
    'quarks': ['Mass_MeVc2', 'Charge_e', 'Spin_hbar'],
    'subatomic': ['Mass_MeVc2', 'Charge_e', 'Spin_hbar'],
    'molecules': ['MolecularMass_amu', 'BondAngle_deg', 'DipoleMoment_D'],
    'alloys': [],
    'materials': [],
}


def validate_required_fields(
    data: Dict, asset_type: str, filename: str
) -> List[str]:
    """Validate that all required fields are present.

    Args:
        data: The loaded JSON data dictionary
        asset_type: One of the keys in REQUIRED_FIELDS
        filename: Source filename for error messages

    Returns:
        List of missing required field names (empty if all present)
    """
    required = REQUIRED_FIELDS.get(asset_type, [])
    missing = [f for f in required if f not in data or data[f] is None]
    if missing:
        logger.warning(
            "Missing required fields %s in %s", missing, filename
        )
    return missing


def validate_numeric_fields(
    data: Dict, asset_type: str, filename: str
) -> List[str]:
    """Validate that numeric fields contain numeric values.

    Args:
        data: The loaded JSON data dictionary
        asset_type: One of the keys in NUMERIC_FIELDS
        filename: Source filename for error messages

    Returns:
        List of field names with non-numeric values
    """
    fields = NUMERIC_FIELDS.get(asset_type, [])
    errors = []
    for f in fields:
        value = data.get(f)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f)
    if errors:
        logger.warning(
            "Non-numeric values in fields %s in %s", errors, filename
        )
    return errors


def validate_json_data(
    data: Dict, asset_type: str, filename: str
) -> bool:
    """Run all validations on a data record.

    Args:
        data: The loaded JSON data dictionary
        asset_type: Asset type key
        filename: Source filename for error messages

    Returns:
        True if validation passes (no missing required fields)
    """
    missing = validate_required_fields(data, asset_type, filename)
    validate_numeric_fields(data, asset_type, filename)
    return len(missing) == 0
