"""
Periodica -- Scientific computation library.

Covers particle physics, nuclear physics, atomic chemistry, molecular chemistry,
materials science, alloy thermodynamics, and biological systems.

Quick start
-----------
Generic composition (data-driven, no hardcoded chemistry)::

    from periodica import Get, Save, Scope

    proton = Get("{u=2,d=1}")              # subatomic from quarks
    hydrogen = Get("{P=1,N=0,E=1}")        # atom from subatomic
    water = Get("{H=2,O=1}")               # molecule from atoms (post build)
    Get(Scope.SubAtomic, "P")              # disambiguate to Proton
    Get(Scope.Atom, "P")                   # disambiguate to Phosphorus

Lower-level physics (semi-empirical formulas, layout math, etc.)::

    from periodica import PhysicsConstants
    from periodica.utils.physics_calculator import (
        AtomCalculator, SubatomicCalculator, MoleculeCalculator,
    )
    from periodica.data.element_loader import ElementDataLoader
    from periodica.data.molecule_loader import MoleculeDataLoader
    from periodica.data.quark_loader import QuarkDataLoader
    from periodica.data.subatomic_loader import SubatomicDataLoader
    from periodica.data.alloy_loader import AlloyDataLoader

Core enums::

    from periodica import PTPropertyName, PTLayoutMode, BondType, CrystalStructure
"""

__version__ = "1.1.0"

# ── Generic composition + named registry (primary API) ─────────────────
from periodica.get import (
    Get,
    Save,
    Scope,
    UnknownName,
    UnknownConstituent,
    UnknownTier,
    RegistryCollision,
    reload_registry,
    list_tiers,
)
from periodica.sample import (
    sample,
    data_sheet,
    register_field_model,
)
from periodica.folding import (
    build_backbone,
    build_backbone_from_entry,
    extract_phi_psi,
    kabsch_rmsd,
    parse_pdb_backbone,
    backbone_array_from_pdb,
    fetch_alphafold,
    load_alphafold_reference,
    ramachandran_region,
    ramachandran_in_allowed,
    folding_rules,
)
from periodica.optimize import (
    optimize_protein_folding,
    optimize_alloy,
)
from periodica.export import (
    voxel_sample,
    voxel_phase_map,
    export_stl,
    export_obj,
    export_vtk_legacy,
    export_sdf_raw,
    export_hlsl,
)

# ── Constants & data manager (no hardcoded chemistry) ──────────────────
from periodica.utils.physics_calculator import PhysicsConstants
from periodica.data.data_manager import DataManager, DataCategory

# ── Core enums (descriptive metadata, no behaviour) ─────────────────────
from periodica.core.pt_enums import PTPropertyName, PTLayoutMode
from periodica.core.molecule_enums import (
    MoleculeLayoutMode, BondType, MoleculeCategory,
    MolecularGeometry, MoleculePolarity,
)
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


__all__ = [
    # Version
    '__version__',
    # Generic composition (primary API)
    'Get', 'Save', 'Scope',
    'UnknownName', 'UnknownConstituent', 'UnknownTier', 'RegistryCollision',
    'reload_registry', 'list_tiers',
    # Sampling / data sheets
    'sample', 'data_sheet', 'register_field_model',
    # Folding (3D backbone, Kabsch, AlphaFold reference)
    'build_backbone', 'build_backbone_from_entry', 'extract_phi_psi',
    'kabsch_rmsd', 'parse_pdb_backbone', 'backbone_array_from_pdb',
    'fetch_alphafold', 'load_alphafold_reference',
    'ramachandran_region', 'ramachandran_in_allowed', 'folding_rules',
    # Optimization
    'optimize_protein_folding', 'optimize_alloy',
    # 3D export (STL, OBJ+MTL, VTK, SDF, HLSL)
    'voxel_sample', 'voxel_phase_map',
    'export_stl', 'export_obj', 'export_vtk_legacy',
    'export_sdf_raw', 'export_hlsl',
    # Constants & data manager
    'PhysicsConstants',
    'DataManager', 'DataCategory',
    # Element enums
    'PTPropertyName', 'PTLayoutMode',
    # Molecule enums
    'MoleculeLayoutMode', 'BondType', 'MoleculeCategory',
    'MolecularGeometry', 'MoleculePolarity',
    # Quark / subatomic enums
    'QuarkLayoutMode', 'ParticleType', 'InteractionForce',
    'SubatomicLayoutMode', 'ParticleCategory',
    # Alloy enums
    'AlloyLayoutMode', 'AlloyCategory', 'CrystalStructure',
    # Amino acid enums
    'AminoAcidLayoutMode', 'AminoAcidCategory',
    # Material enums
    'MaterialLayoutMode', 'MaterialCategory',
    # Protein enums
    'ProteinLayoutMode', 'SecondaryStructureType',
    # Cell enums
    'CellLayoutMode', 'CellType',
    # Nucleic acid enums
    'NucleicAcidLayoutMode', 'NucleicAcidType',
    # Biomaterial enums
    'BiomaterialLayoutMode', 'BiomaterialType',
    # Cell component enums
    'CellComponentLayoutMode', 'OrganelleType',
]
