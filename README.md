# Periodica

A rigorous scientific computation library for particle physics, nuclear physics, atomic chemistry, molecular chemistry, materials science, alloy thermodynamics, and biological systems.

All 12 scientific domains are fully data-driven: properties come from bundled JSON data sheets (424 files covering 118 elements, quarks, hadrons, molecules, alloys, materials, amino acids, proteins, nucleic acids, cells, and biomaterials), not hardcoded lookup tables. Changes to data sheets cascade dynamically through the derivation chain.

## Installation

```bash
pip install periodica
```

## Quick Start

```python
# Top-level convenience imports
from periodica import AtomCalculator, PhysicsConstants, get_element, DataManager, DataCategory

# Get element data
iron = get_element("Fe")
print(iron["name"], iron["atomic_number"], iron["atomic_mass"])
# Iron 26 55.845

# Calculate atomic properties
calc = AtomCalculator()
mass = calc.calculate_atomic_mass(26, 30)  # Z=26, N=30 (Iron-56)
print(f"Atomic mass: {mass:.3f} u")

# Access any of the 14 data categories
dm = DataManager()
alloys = dm.get_all_items(DataCategory.ALLOYS)
proteins = dm.get_all_items(DataCategory.PROTEINS)
```

### Data Loaders

Each domain has a specialized loader with convenience functions:

```python
from periodica.data import (
    get_element, get_element_by_z,     # Elements by symbol or Z
    get_molecule_loader,                # Molecules by name or formula
    get_quark_loader,                   # Quarks by name or symbol
    get_subatomic_loader,               # Hadrons (baryons, mesons)
    get_alloy_loader,                   # Alloys by name or category
)

# Element lookup
hydrogen = get_element("H")
carbon = get_element_by_z(6)

# Molecule lookup
loader = get_molecule_loader()
water = loader.get_molecule_by_name("Water")
water_alt = loader.get_molecule_by_formula("H2O")

# Quark lookup
qloader = get_quark_loader()
up = qloader.get_particle_by_name("Up Quark")
print(up["Charge_e"])  # 0.6667

# Subatomic particles
sloader = get_subatomic_loader()
proton = sloader.get_particle_by_name("Proton")
baryons = sloader.get_baryons()
mesons = sloader.get_mesons()
```

### Physics Calculators

```python
from periodica import AtomCalculator, SubatomicCalculator, MoleculeCalculator, PhysicsConstants

# Atom calculator: atomic mass, ionization energy, electron configurations
atom = AtomCalculator()
mass = atom.calculate_atomic_mass(protons=82, neutrons=126)  # Lead-208

# Subatomic calculator: hadron masses from quark content
sub = SubatomicCalculator()

# Molecule calculator: molecular properties from atomic composition
mol = MoleculeCalculator()

# Physical constants
print(PhysicsConstants.PROTON_MASS_U)    # 1.007276466621
print(PhysicsConstants.RYDBERG_ENERGY_EV)  # 13.605693122994 eV
```

### Derivation Chain Predictors

The library includes a full derivation pipeline with semi-empirical models:

```python
from periodica.utils.predictors.hadron.hyperfine import calculate_hyperfine_mass
from periodica.utils.predictors.atomic.slater_predictor import SlaterPredictor
from periodica.utils.predictors.alloy.rule_of_mixtures import calculate_formation_enthalpy
from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor
```

**Models included:**
- De Rujula-Georgi-Glashow hyperfine splitting (hadron masses from quark content)
- Semi-empirical mass formula with shell correction + Wigner term (nuclear binding)
- Slater rules with relativistic corrections + Aufbau anomalies (atomic properties)
- VSEPR geometry + Huckel MO theory + vector bond dipoles (molecular properties)
- Miedema formation enthalpy + Vegard's law (alloy thermodynamics)
- Regular solution model + Lindemann melting (phase diagrams)
- Chou-Fasman secondary structure prediction (protein folding)

### Layout Math (UI-Agnostic)

Position computation for visualizations, decoupled from any rendering framework:

```python
from periodica.layout_math.element_table import TablePositioner
from periodica.layout_math.element_circular import CircularPositioner
from periodica.layout_math.quark_standard import compute_positions as quark_positions

# Compute element positions for an 800x600 canvas
positioner = TablePositioner()
elements = [{"symbol": "H", "z": 1, "period": 1, "group": 1}, ...]
positions = positioner.compute_positions(elements, width=800, height=600)
# Returns: [{"x": 40, "y": 20, "w": 38, "h": 50, "label": "H", "metadata": {...}}, ...]
```

**35 layout algorithms** across 5 domains: elements (table, circular, spiral, linear), quarks (8 modes), subatomic (8 modes), molecules (9 modes), alloys (4 modes).

### Enums (All 12 Domains)

```python
from periodica.core import (
    # Elements
    PTPropertyName, PTLayoutMode,
    # Molecules
    MoleculeLayoutMode, BondType, MolecularGeometry, MoleculePolarity,
    # Quarks
    QuarkLayoutMode, ParticleType, InteractionForce,
    # Subatomic
    SubatomicLayoutMode, ParticleCategory,
    # Alloys
    AlloyLayoutMode, AlloyCategory, CrystalStructure,
    # Materials
    MaterialLayoutMode, MaterialCategory,
    # Amino Acids
    AminoAcidLayoutMode, AminoAcidCategory,
    # Proteins
    ProteinLayoutMode, SecondaryStructureType,
    # Nucleic Acids
    NucleicAcidLayoutMode, NucleicAcidType,
    # Cells
    CellLayoutMode, CellType,
    # Cell Components
    CellComponentLayoutMode, OrganelleType,
    # Biomaterials
    BiomaterialLayoutMode, BiomaterialType,
)
```

## Supported Domains

| Domain | Data Files | Loader | Calculator | Predictors | Layouts |
|--------|-----------|--------|------------|------------|---------|
| Elements (118) | `data/active/elements/` | `ElementDataLoader` | `AtomCalculator` | Slater, relativistic | 4 modes |
| Quarks (6+) | `data/active/quarks/` | `QuarkDataLoader` | - | - | 8 modes |
| Subatomic | `data/active/subatomic/` | `SubatomicDataLoader` | `SubatomicCalculator` | Hyperfine | 8 modes |
| Molecules | `data/active/molecules/` | `MoleculeDataLoader` | `MoleculeCalculator` | VSEPR, Huckel | 9 modes |
| Alloys | `data/active/alloys/` | `AlloyDataLoader` | - | Miedema, Vegard | 4 modes |
| Materials | `data/active/materials/` | `DataManager` | - | Lindemann | Grid |
| Amino Acids | `data/active/amino_acids/` | `DataManager` | - | Chou-Fasman | Grid |
| Proteins | `data/active/proteins/` | `DataManager` | - | Secondary structure | Grid |
| Nucleic Acids | `data/active/nucleic_acids/` | `DataManager` | - | - | Grid |
| Cell Components | `data/active/cell_components/` | `DataManager` | - | - | Grid |
| Cells | `data/active/cells/` | `DataManager` | - | - | Grid |
| Biomaterials | `data/active/biological_materials/` | `DataManager` | - | - | Grid |

## Key Modules

| Module | Contents |
|--------|----------|
| `periodica.core` | 68 enum classes across 12 domains |
| `periodica.data` | 5 specialized loaders + DataManager (14 categories) + 424 JSON data sheets |
| `periodica.utils` | Physics calculators, pure math, SDF rendering, color math, simulation schema |
| `periodica.utils.predictors` | Semi-empirical models for hadrons, atoms, molecules, alloys, proteins |
| `periodica.utils.transforms` | Fourier fields, wavelets, stochastic fields, optical properties |
| `periodica.layout_math` | 35 layout algorithms returning framework-agnostic position dicts |

## Documentation

Generate full API docs:

```bash
pip install periodica[docs]
cd packages/periodica/docs
python generate_readme.py   # Generates docs/API_REFERENCE.md
python generate_docs.py     # Generates HTML API docs via pdoc
```

## License

MIT
