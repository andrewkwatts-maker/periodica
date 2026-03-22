"""
Periodics Data Module
=====================

Public API for loading and accessing scientific data across all domains:
elements, molecules, quarks, subatomic particles, alloys, and more.

Quick start
-----------

Get a single element by symbol::

    from periodica.data import get_element
    hydrogen = get_element("H")

Load all molecules::

    from periodica.data import get_all_molecules
    molecules = get_all_molecules()

Use the unified DataManager for any category::

    from periodica.data import DataManager, DataCategory
    mgr = DataManager()
    items = mgr.get_all_items(DataCategory.AMINO_ACIDS)

Validate raw JSON before ingestion::

    from periodica.data import validate_json_data
    ok = validate_json_data(record, "elements", "001_H.json")
"""

# ---------------------------------------------------------------------------
# Element loader
# ---------------------------------------------------------------------------
from periodica.data.element_loader import (
    ElementDataLoader,
    get_loader as get_element_loader,
    get_element,
    get_element_by_z,
    get_all_elements,
    create_fallback_element_data,
)

# ---------------------------------------------------------------------------
# Molecule loader
# ---------------------------------------------------------------------------
from periodica.data.molecule_loader import (
    MoleculeDataLoader,
    get_molecule_loader,
    get_all_molecules,
)

# ---------------------------------------------------------------------------
# Quark / particle loader
# ---------------------------------------------------------------------------
from periodica.data.quark_loader import (
    QuarkDataLoader,
    get_quark_loader,
)

# ---------------------------------------------------------------------------
# Subatomic (composite) particle loader
# ---------------------------------------------------------------------------
from periodica.data.subatomic_loader import (
    SubatomicDataLoader,
    get_subatomic_loader,
    load_subatomic_data,
)

# ---------------------------------------------------------------------------
# Alloy loader
# ---------------------------------------------------------------------------
from periodica.data.alloy_loader import (
    AlloyDataLoader,
    get_alloy_loader,
    get_all_alloys,
)

# ---------------------------------------------------------------------------
# Data manager & categories
# ---------------------------------------------------------------------------
from periodica.data.data_manager import (
    DataManager,
    DataCategory,
    get_data_manager,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
from periodica.data.validation import validate_json_data

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
__all__ = [
    # Element loader
    "ElementDataLoader",
    "get_element_loader",
    "get_element",
    "get_element_by_z",
    "get_all_elements",
    "create_fallback_element_data",
    # Molecule loader
    "MoleculeDataLoader",
    "get_molecule_loader",
    "get_all_molecules",
    # Quark loader
    "QuarkDataLoader",
    "get_quark_loader",
    # Subatomic loader
    "SubatomicDataLoader",
    "get_subatomic_loader",
    "load_subatomic_data",
    # Alloy loader
    "AlloyDataLoader",
    "get_alloy_loader",
    "get_all_alloys",
    # Data manager
    "DataManager",
    "DataCategory",
    "get_data_manager",
    # Validation
    "validate_json_data",
]
