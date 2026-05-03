# Periodica

A rigorous scientific computation and estimation library for particle physics, nuclear physics, atomic chemistry, molecular chemistry, materials science, alloy thermodynamics, and biological systems.

All 12 scientific domains are fully data-driven: properties come from bundled JSON data sheets (424 files covering 118 elements, quarks, hadrons, molecules, alloys, materials, amino acids, proteins, nucleic acids, cells, and biomaterials), not hardcoded lookup tables. Changes to data sheets cascade dynamically through the derivation chain.

> **Looking for the GUI?** The companion KivyMD desktop + Android app lives at
> [github.com/andrewkwatts-maker/periodica-app](https://github.com/andrewkwatts-maker/periodica-app).
> It uses this library as its computation backend (`pip install periodica`) and
> ships as a Windows EXE plus an Android APK build (Buildozer/WSL).

## Installation

The slim install pulls only `numpy`. The `[full]` extra adds optional libraries
useful for visualization and AI-assisted workflows. Periodica itself never
imports the optional libraries — they are for downstream user code.

```bash
# Slim — core library (Get / Save / Scope / sample / fold / optimize / export / CLI)
pip install periodica

# Full — slim + matplotlib + scipy + google-generativeai
pip install "periodica[full]"

# Test extras for contributors (pytest, twine, build):
pip install "periodica[test]"

# Editable install from source:
git clone https://github.com/andrewkwatts-maker/periodica
cd periodica
pip install -e .
```

After install the `periodica` console script is on your `PATH`.

```bash
periodica list                           # registry tiers + counts
periodica show H2O                       # data sheet for an entry
periodica fold Crambin --validate        # CA RMSD vs AlphaFold (offline-capable)
periodica sample Steel-1018 YoungsModulus_GPa --at 0,0,0 --scale 1e-7
periodica optimize alloy --target YieldStrength_MPa_min=200 --target Density_kgm3_max=8000
periodica export Steel-1018 --format hlsl --output steel.hlsl
```

## Quick Start

### Generic `Get()` - compose anything from fundamentals

`Get()` is a content-agnostic composer: it walks JSON data files, sums
additive scalar properties, and knows nothing about specific elements
or particles. Pass a spec built from any fundamentals defined in JSON:

```python
from periodica import Get, Scope

# Compose subatomic from quarks
proton = Get("{u=2,d=1}")           # Charge_e=1, BaryonNumber_B=1

# Compose atom from subatomic + fundamentals
hydrogen = Get("{P=1,N=0,E=1}")     # Mass_amu ≈ 1.00783

# Compose molecule from atoms (after build_periodic_table)
water = Get("{H=2,O=1}")            # Mass_amu ≈ 18.15

# Bare-name lookup (after the corresponding generator script has run)
H = Get("H")                        # Hydrogen atom
H2O = Get("H2O")                    # Water molecule
Higgs = Get("Higgs")                # Higgs Boson via alias

# Disambiguate by scope when a symbol means different things in different tiers
Get(Scope.SubAtomic, "P")           # -> Proton
Get(Scope.Atom,      "P")           # -> Phosphorus
Get(Scope.Molecule,  "H2O")         # -> Water
```

### CLI

After install, the `periodica` command is on your PATH. Use it to inspect
the registry, regenerate the bundled defaults, or run your own input files.

```bash
periodica list                  # summary of registry tiers + entry counts
periodica list atoms            # list every entry in a tier
periodica show H                # pretty-print a registry entry by name
periodica show "Ca2+"           # ions resolve too (charge falls out of E vs P)
periodica inputs                # list the default input JSON files

periodica build                 # rebuild every default tier in dependency order
periodica build molecules       # only rebuild named tier(s)
periodica build --extra ./mine  # also include input JSONs from your folder
periodica build --extra ./my_tier.json

periodica run ./my_inputs.json  # one-shot run of a custom input file
periodica clear                 # wipe all generated tiers (asks for confirmation)
periodica clear ions            # wipe just one tier
periodica clear -y              # skip confirmation
```

Default tiers shipped pre-built in the wheel:

| tier         | source     | entries | example                    |
|--------------|------------|--------:|----------------------------|
| hadrons_gen  | quarks     |      15 | `Proton_q`                 |
| atoms        | subatomic+ |     118 | `H`, `Fe`, `Og`            |
| isotopes     | subatomic+ |      40 | `H-2`, `U-235`             |
| ions         | subatomic+ |      30 | `Ca2+`, `Cl-`              |
| molecules    | atoms      |      73 | `H2O`, `Caffeine`          |
| alloys       | atoms      |      10 | `Steel-1018`, `Inconel-718`|
| polymers     | atoms      |       9 | `HDPE`, `PET`, `Nylon-6`   |
| ceramics     | atoms      |       6 | `Alumina`, `SiliconCarbide`|
| composites   | atoms      |       4 | `CFRP`, `GFRP`, `Concrete-Mix`|

### Material data sheets and 3D sampling

Complex materials (alloys, polymers, ceramics, composites) ship with
full property data sheets and a generic `sample()` API for evaluating
properties at a 3D point.

```python
from periodica import data_sheet, sample

# Bulk data sheet
data_sheet("Steel-1018")
# {'Density_kgm3': 7870, 'YoungsModulus_GPa': 200, 'TensileStrength_MPa': 440, ...}

# Homogeneous materials: at/scale_m are no-ops
sample("Steel-1018", "Density_kgm3")                          # 7870
sample("Steel-1018", "YoungsModulus_GPa", at=(0, 0, 0))       # 200

# Anisotropic composites: response varies with the sample direction
sample("CFRP", "YoungsModulus_GPa", at=(1, 0, 0))             # 230 GPa axial
sample("CFRP", "YoungsModulus_GPa", at=(0, 1, 0))             #  15 GPa transverse
sample("CFRP", "YoungsModulus_GPa", at=(1, 1, 0))             # ~167 GPa off-axis

# Scale-dependent fields: small `scale_m` picks a phase from the field
sample("Concrete-Mix", "YoungsModulus_GPa", scale_m=1.0,  at=(0, 0, 0))   #  30  (bulk)
sample("Concrete-Mix", "YoungsModulus_GPa", scale_m=1e-3, at=(7, 3, 1))   #  60  (aggregate)
sample("Concrete-Mix", "YoungsModulus_GPa", scale_m=1e-3, at=(0, 0, 0))   #  25  (cement paste)
```

Field models are declared in JSON (`"Field": {"model": "anisotropic_axial",
"fiber_direction": [1,0,0], "axial": {...}, "transverse": {...}}`) - no
hardcoded chemistry. Add your own with `register_field_model("name", fn)`.

CLI: `periodica sample <name> [<prop>] [--at X,Y,Z] [--scale METRES]`.
Omit `<prop>` to print the full data sheet.

### Protein folding + AlphaFold validation

Proteins live as a first-class registry tier (active/proteins). The
existing JSONs already carry per-residue phi/psi dihedrals; `build_backbone`
reconstructs 3D Cartesian coordinates with the NeRF (Natural Extension
Reference Frame) algorithm. All bond lengths/angles are read from
`data/config/folding_rules.json` -- no chemistry hardcoded in code.

```python
from periodica import (
    build_backbone_from_entry, kabsch_rmsd,
    parse_pdb_backbone, backbone_array_from_pdb,
    load_alphafold_reference, optimize_protein_folding,
)

# Construct 3D backbone for a registered protein (N, CA, C per residue)
coords = build_backbone_from_entry("Crambin")          # ndarray (46, 3, 3)

# Compare against AlphaFold (bundled refs + optional live fetch)
af_pdb = load_alphafold_reference("P01542")            # bundled-first, network fallback
af = backbone_array_from_pdb(parse_pdb_backbone(af_pdb))
_, rmsd = kabsch_rmsd(coords[:, 1, :], af[:, 1, :])    # CA-only RMSD
print(f"Crambin vs AlphaFold: {rmsd:.2f} A")

# Simulated-annealing on phi/psi to maximise helical propensity
result = optimize_protein_folding(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGI",
    target="helical", iterations=2000, seed=42,
)
print(result["final_energy"], result["coords"].shape)
```

Five AlphaFold reference PDBs are bundled (Crambin, Insulin Chain A,
Ubiquitin, Lysozyme, Beta-Defensin) so unit tests run offline.
`fetch_alphafold(uniprot_id)` pulls any UniProt ID at runtime via the
AlphaFold API; results are cached in `~/.cache/periodica/alphafold/`.

### Alloy microstructure + composition optimization

Alloy data sheets accept a `microstructure_voronoi` field model that
returns phase-specific properties when sampled at a 3D point with a fine
`scale_m`. At macro scale, `sample()` returns the bulk value. Phases,
volume fractions, and grain density are declared per-entry in JSON.

```python
from periodica import sample, optimize_alloy

# Macro-scale: bulk Young's modulus
sample("Steel-1018", "YoungsModulus_GPa", at=(0, 0, 0), scale_m=1.0)   # 200

# Sub-macro: hits ferrite (195) or pearlite (220) by point hash
sample("Steel-1018", "YoungsModulus_GPa", at=(1, 0, 0), scale_m=1e-7)  # 195
sample("Steel-1018", "YoungsModulus_GPa", at=(0, 1, 0), scale_m=1e-7)  # 195
sample("Steel-1018", "YoungsModulus_GPa", at=(2, 3, 1), scale_m=1e-7)  # 220

# Search Hume-Rothery-bounded compositions for a property target
candidates = optimize_alloy(
    {"YieldStrength_MPa_min": 200, "Density_kgm3_max": 8000},
    base="Fe", candidates=500, top_k=5, seed=42,
)
for c in candidates:
    print(c["composition"], c["estimated_properties"])
```

CLI:
```bash
periodica fold Crambin --validate                # RMSD vs AlphaFold
periodica fold Ubiquitin --output ub.pdb         # write PDB
periodica optimize folding --sequence AAAAAA --target-region helical --seed 1
periodica optimize alloy --target YieldStrength_MPa_min=200 \
                         --target Density_kgm3_max=8000 --base Fe --top 3
```

### 3D export (STL / OBJ+MTL / VTK / SDF / HLSL)

Any registered entry with a microstructure or phase field can be exported as
a voxel-block boundary mesh, a procedural-shader function, or a raw scalar
volume. Phases are colour-coded by a deterministic palette; per-phase
property values come straight from the JSON `Field` declaration — no chemistry
is hardcoded in the exporter.

```python
from periodica import (
    export_stl, export_obj, export_vtk_legacy, export_sdf_raw, export_hlsl,
)

bounds = ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
v = 1.0
scale = 1e-7   # below the alloy's macro_scale_m so phases activate

# Binary STL, voxel-block boundary surfaces between grains.
export_stl("Steel-1018", "steel.stl",
           bounds=bounds, voxel_size=v, scale_m=scale)

# OBJ + sibling .mtl. One material per phase. Vertices carry RGB,
# encoding phase identity. MTL records property values as comments.
export_obj("Steel-1018", "steel.obj",
           bounds=bounds, voxel_size=v, scale_m=scale,
           properties=["YoungsModulus_GPa", "TensileStrength_MPa"])

# Legacy ASCII VTK with `phase_id` plus per-property scalars at every voxel.
export_vtk_legacy("Steel-1018", "steel.vtk",
                  bounds=bounds, voxel_size=v, scale_m=scale,
                  properties=["Density_kgm3", "YoungsModulus_GPa"])

# Raw float32 3D volume + JSON sidecar. mode='occupancy' gives 1/0;
# mode='phase' gives float-cast phase ids.
export_sdf_raw("Steel-1018", "steel.sdf",
               bounds=bounds, voxel_size=v, scale_m=scale, mode="phase")

# HLSL Sample<Name>(float3 pos, float scaleM) function with hash-based
# phase dispatch and per-phase property values. Drop into your shader.
export_hlsl("Steel-1018", "steel.hlsl",
            properties=["YoungsModulus_GPa", "TensileStrength_MPa", "Density_kgm3"])
```

CLI:
```bash
periodica export Steel-1018 --format stl  --output steel.stl --bounds 0,0,0,5,5,5 --voxel-size 1 --scale 1e-7
periodica export Steel-1018 --format obj  --output steel.obj --bounds 0,0,0,5,5,5 --voxel-size 1 --scale 1e-7 --properties YoungsModulus_GPa
periodica export Steel-1018 --format hlsl --output steel.hlsl
periodica export Steel-1018 --format sdf  --output steel.sdf --sdf-mode phase
```

### Custom input file shape

```json
{
  "from": ["subatomic", "fundamentals"],
  "tier": "my_tier",
  "requires": ["atoms"],
  "rows": [
    { "name": "MyEntry", "spec": { "P": 1, "N": 0, "E": 1 } }
  ]
}
```

Composition rules (which fields are additive, which folders are
fundamentals, etc.) live in `data/config/composition_rules.json` -
edit it to extend the system without touching code.

### Domain helpers (backward-compat)

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
