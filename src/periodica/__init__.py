"""
Periodica -- Scientific computation library.

Covers particle physics, nuclear physics, atomic chemistry, molecular chemistry,
materials science, alloy thermodynamics, and biological systems (amino acids,
proteins, nucleic acids, cells, biomaterials).

Quick start
-----------
Calculators::

    from periodica import AtomCalculator, PhysicsConstants
    from periodica import SubatomicCalculator, MoleculeCalculator

Data access::

    from periodica import DataManager, DataCategory
    from periodica import get_element, get_molecule_loader, get_quark_loader

Core enums::

    from periodica import PTPropertyName, PTLayoutMode
    from periodica import MoleculeLayoutMode, BondType
    from periodica import AlloyCategory, CrystalStructure
    from periodica import ProteinLayoutMode, CellType
"""

__version__ = "1.0.0"

# ── Calculators (periodica.utils) ───────────────────────────────────────
from periodica.utils.physics_calculator import (
    PhysicsConstants,
    AtomCalculator,
    SubatomicCalculator,
    MoleculeCalculator,
)

# ── Data access (periodica.data) ────────────────────────────────────────
from periodica.data.data_manager import DataManager, DataCategory
from periodica.data.element_loader import ElementDataLoader, get_element, get_element_by_z
from periodica.data.molecule_loader import MoleculeDataLoader, get_molecule_loader
from periodica.data.quark_loader import QuarkDataLoader, get_quark_loader
from periodica.data.subatomic_loader import SubatomicDataLoader, get_subatomic_loader
from periodica.data.alloy_loader import AlloyDataLoader, get_alloy_loader

# ── Core enums -- most-used subset ──────────────────────────────────────
from periodica.core.pt_enums import PTPropertyName, PTLayoutMode
from periodica.core.molecule_enums import MoleculeLayoutMode, BondType, MoleculeCategory, MolecularGeometry, MoleculePolarity
from periodica.core.quark_enums import QuarkLayoutMode, ParticleType, InteractionForce
from periodica.core.subatomic_enums import SubatomicLayoutMode, ParticleCategory
from periodica.core.alloy_enums import AlloyLayoutMode, AlloyCategory, CrystalStructure
from periodica.core.amino_acid_enums import AminoAcidLayoutMode, AminoAcidCategory
from periodica.core.material_enums import MaterialLayoutMode, MaterialCategory
from periodica.core.protein_enums import ProteinLayoutMode, SecondaryStructureType
from periodica.core.cell_enums import CellLayoutMode, CellType
from periodica.core.nucleic_acid_enums import NucleicAcidLayoutMode, NucleicAcidType
from periodica.core.biomaterial_enums import BiomaterialLayoutMode, BiomaterialType
from periodica.core.cell_component_enums import CellComponentLayoutMode, OrganelleType

# ── Generator convenience functions (periodica.generators) ────────────────
from periodica.generators import (
    generate_atom,
    generate_particle,
    generate_molecule,
    generate_alloy,
    generate_material,
    generate_protein,
    generate_nucleic_acid,
    generate_cell,
    generate_biomaterial,
    generate_alloys,
    generate_molecules,
    generate_amino_acids,
    generate_proteins,
    generate_nucleic_acids,
    generate_cells,
    generate_biomaterials,
    generate_materials,
    generate_documentation,
)

__all__ = [
    # Version
    '__version__',
    # Calculators
    'PhysicsConstants',
    'AtomCalculator',
    'SubatomicCalculator',
    'MoleculeCalculator',
    # Data access
    'DataManager',
    'DataCategory',
    'ElementDataLoader',
    'get_element',
    'get_element_by_z',
    'MoleculeDataLoader',
    'get_molecule_loader',
    'QuarkDataLoader',
    'get_quark_loader',
    'SubatomicDataLoader',
    'get_subatomic_loader',
    'AlloyDataLoader',
    'get_alloy_loader',
    # Element enums
    'PTPropertyName',
    'PTLayoutMode',
    # Molecule enums
    'MoleculeLayoutMode',
    'BondType',
    'MoleculeCategory',
    'MolecularGeometry',
    'MoleculePolarity',
    # Quark / subatomic enums
    'QuarkLayoutMode',
    'ParticleType',
    'InteractionForce',
    'SubatomicLayoutMode',
    'ParticleCategory',
    # Alloy enums
    'AlloyLayoutMode',
    'AlloyCategory',
    'CrystalStructure',
    # Amino acid enums
    'AminoAcidLayoutMode',
    'AminoAcidCategory',
    # Material enums
    'MaterialLayoutMode',
    'MaterialCategory',
    # Protein enums
    'ProteinLayoutMode',
    'SecondaryStructureType',
    # Cell enums
    'CellLayoutMode',
    'CellType',
    # Nucleic acid enums
    'NucleicAcidLayoutMode',
    'NucleicAcidType',
    # Biomaterial enums
    'BiomaterialLayoutMode',
    'BiomaterialType',
    # Cell component enums
    'CellComponentLayoutMode',
    'OrganelleType',
    # Generator functions
    'generate_atom',
    'generate_particle',
    'generate_molecule',
    'generate_alloy',
    'generate_material',
    'generate_protein',
    'generate_nucleic_acid',
    'generate_cell',
    'generate_biomaterial',
    'generate_alloys',
    'generate_molecules',
    'generate_amino_acids',
    'generate_proteins',
    'generate_nucleic_acids',
    'generate_cells',
    'generate_biomaterials',
    'generate_materials',
    'generate_documentation',
]
