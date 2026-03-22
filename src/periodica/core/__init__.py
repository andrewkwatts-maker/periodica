"""
Core enums and constants for the Periodics library.

Provides domain-specific enums for all 12 scientific domains: elements,
molecules, quarks, subatomic particles, alloys, amino acids, biomaterials,
cell components, cells, materials, nucleic acids, and proteins.

These enums are library-safe with no PySide6/QWidget dependencies.

Usage::

    from periodica.core import PTPropertyName, PTLayoutMode
    from periodica.core import MoleculeLayoutMode, BondType
    from periodica.core import AlloyCategory, CrystalStructure
    from periodica.core import ProteinLayoutMode, SecondaryStructureType
"""

# ── Element (periodic table) enums ──────────────────────────────────────
from periodica.core.pt_enums import (
    PTPropertyName,
    PTEncodingKey,
    PTControlType,
    PTWavelengthMode,
    PTElementDataKey,
    PTPropertyType,
    PTEncodingType,
    PTLayoutMode,
    ENCODING_KEY_TO_TYPE,
    ENCODING_KEY_ENUM_TO_TYPE,
)

# ── Molecule enums ──────────────────────────────────────────────────────
from periodica.core.molecule_enums import (
    MoleculeLayoutMode,
    MoleculeProperty,
    BondType,
    MolecularGeometry,
    MoleculePolarity,
    MoleculeCategory,
    MoleculeState,
    get_element_color,
)

# ── Quark / particle enums ──────────────────────────────────────────────
from periodica.core.quark_enums import (
    QuarkLayoutMode,
    QuarkProperty,
    ParticleType,
    QuarkGeneration,
    InteractionForce,
)

# ── Subatomic particle enums ────────────────────────────────────────────
from periodica.core.subatomic_enums import (
    SubatomicLayoutMode,
    SubatomicProperty,
    ParticleCategory,
    QuarkType,
    PARTICLE_COLORS,
    get_particle_family_color,
)

# ── Alloy enums ─────────────────────────────────────────────────────────
from periodica.core.alloy_enums import (
    AlloyLayoutMode,
    AlloyCategory,
    CrystalStructure,
    AlloyProperty,
    ComponentRole,
)

# ── Amino acid enums ────────────────────────────────────────────────────
from periodica.core.amino_acid_enums import (
    AminoAcidLayoutMode,
    AminoAcidProperty,
    AminoAcidCategory,
    AminoAcidPolarity,
    ChargeState,
    SecondaryStructure as AminoAcidSecondaryStructure,
)

# ── Biomaterial enums ───────────────────────────────────────────────────
from periodica.core.biomaterial_enums import (
    BiomaterialLayoutMode,
    BiomaterialType,
    ECMComponent,
    MechanicalProperty,
    VascularizationLevel,
)

# ── Cell component enums ────────────────────────────────────────────────
from periodica.core.cell_component_enums import (
    CellComponentLayoutMode,
    OrganelleType,
    MembraneType,
    CellularCompartment,
    ComponentFunction,
)

# ── Cell enums ──────────────────────────────────────────────────────────
from periodica.core.cell_enums import (
    CellLayoutMode,
    CellType,
    CellCyclePhase,
    MetabolicState,
    Organism,
    TissueType,
)

# ── Material (engineering) enums ────────────────────────────────────────
from periodica.core.material_enums import (
    MaterialLayoutMode,
    MaterialCategory,
    MaterialProperty,
)

# ── Nucleic acid enums ──────────────────────────────────────────────────
from periodica.core.nucleic_acid_enums import (
    NucleicAcidLayoutMode,
    NucleicAcidType,
    BaseType,
    NucleicAcidFunction,
)

# ── Protein enums ───────────────────────────────────────────────────────
from periodica.core.protein_enums import (
    ProteinLayoutMode,
    SecondaryStructureType,
    ProteinFunction,
    CellularLocalization,
    FoldingState,
)

__all__ = [
    # ── Element (periodic table) enums ──
    'PTPropertyName',
    'PTEncodingKey',
    'PTControlType',
    'PTWavelengthMode',
    'PTElementDataKey',
    'PTPropertyType',
    'PTEncodingType',
    'PTLayoutMode',
    'ENCODING_KEY_TO_TYPE',
    'ENCODING_KEY_ENUM_TO_TYPE',
    # ── Molecule enums ──
    'MoleculeLayoutMode',
    'MoleculeProperty',
    'BondType',
    'MolecularGeometry',
    'MoleculePolarity',
    'MoleculeCategory',
    'MoleculeState',
    'get_element_color',
    # ── Quark / particle enums ──
    'QuarkLayoutMode',
    'QuarkProperty',
    'ParticleType',
    'QuarkGeneration',
    'InteractionForce',
    # ── Subatomic particle enums ──
    'SubatomicLayoutMode',
    'SubatomicProperty',
    'ParticleCategory',
    'QuarkType',
    'PARTICLE_COLORS',
    'get_particle_family_color',
    # ── Alloy enums ──
    'AlloyLayoutMode',
    'AlloyCategory',
    'CrystalStructure',
    'AlloyProperty',
    'ComponentRole',
    # ── Amino acid enums ──
    'AminoAcidLayoutMode',
    'AminoAcidProperty',
    'AminoAcidCategory',
    'AminoAcidPolarity',
    'ChargeState',
    'AminoAcidSecondaryStructure',
    # ── Biomaterial enums ──
    'BiomaterialLayoutMode',
    'BiomaterialType',
    'ECMComponent',
    'MechanicalProperty',
    'VascularizationLevel',
    # ── Cell component enums ──
    'CellComponentLayoutMode',
    'OrganelleType',
    'MembraneType',
    'CellularCompartment',
    'ComponentFunction',
    # ── Cell enums ──
    'CellLayoutMode',
    'CellType',
    'CellCyclePhase',
    'MetabolicState',
    'Organism',
    'TissueType',
    # ── Material enums ──
    'MaterialLayoutMode',
    'MaterialCategory',
    'MaterialProperty',
    # ── Nucleic acid enums ──
    'NucleicAcidLayoutMode',
    'NucleicAcidType',
    'BaseType',
    'NucleicAcidFunction',
    # ── Protein enums ──
    'ProteinLayoutMode',
    'SecondaryStructureType',
    'ProteinFunction',
    'CellularLocalization',
    'FoldingState',
]
