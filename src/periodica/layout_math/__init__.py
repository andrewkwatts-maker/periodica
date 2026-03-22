"""
periodics.layout_math -- pure-Python layout position computation.

This sub-package extracts the position-computation math from the
Qt-based layout renderers into framework-free modules.  Each positioner
takes a list of element dicts and viewport dimensions and returns a list
of position dicts that any rendering backend can consume.
"""

from .base import LayoutPositioner
from .element_table import TablePositioner
from .element_circular import CircularPositioner
from .element_spiral import SpiralPositioner
from .element_linear import (
    LinearPositioner,
    PropertyKey,
    PropertyConfig,
    PROPERTY_CONFIGS,
    get_normalized_value,
    get_property_range,
)

# Molecule layout math modules (pure compute_positions functions)
from . import (
    molecule_grid,
    molecule_mass,
    molecule_geometry,
    molecule_polarity,
    molecule_bond,
    molecule_phase_diagram,
    molecule_dipole,
    molecule_density,
    molecule_bond_complexity,
)

# Alloy layout math modules
from . import (
    alloy_property,
    alloy_composition,
    alloy_lattice,
    alloy_category,
)

# Quark layout math modules
from . import (
    quark_standard,
    quark_circular,
    quark_linear,
    quark_mass_spiral,
    quark_charge_mass,
    quark_fermion_boson,
    quark_force_network,
    quark_alternative,
)

# Subatomic layout math modules
from . import (
    subatomic_mass,
    subatomic_charge,
    subatomic_lifetime,
    subatomic_discovery,
    subatomic_decay,
    subatomic_eightfold,
    subatomic_quark_tree,
    subatomic_baryon_meson,
)

__all__ = [
    # Abstract base
    "LayoutPositioner",
    # Concrete positioners
    "TablePositioner",
    "CircularPositioner",
    "SpiralPositioner",
    "LinearPositioner",
    # Linear-layout helpers
    "PropertyKey",
    "PropertyConfig",
    "PROPERTY_CONFIGS",
    "get_normalized_value",
    "get_property_range",
    # Molecule layout math
    "molecule_grid",
    "molecule_mass",
    "molecule_geometry",
    "molecule_polarity",
    "molecule_bond",
    "molecule_phase_diagram",
    "molecule_dipole",
    "molecule_density",
    "molecule_bond_complexity",
    # Alloy layout math
    "alloy_property",
    "alloy_composition",
    "alloy_lattice",
    "alloy_category",
    # Quark layout math
    "quark_standard",
    "quark_circular",
    "quark_linear",
    "quark_mass_spiral",
    "quark_charge_mass",
    "quark_fermion_boson",
    "quark_force_network",
    "quark_alternative",
    # Subatomic layout math
    "subatomic_mass",
    "subatomic_charge",
    "subatomic_lifetime",
    "subatomic_discovery",
    "subatomic_decay",
    "subatomic_eightfold",
    "subatomic_quark_tree",
    "subatomic_baryon_meson",
]
