# Changelog

All notable changes to periodica are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-05-03

Major minor release: a generic, content-agnostic composition + sampling
core, advanced protein folding with AlphaFold validation, alloy
microstructure-aware sampling, optimization, 3D export, and a complete
CLI. All chemistry/physics constants live in JSON data sheets — no
hardcoded element tables in code.

### Added — generic composition layer
- `Get(spec, from_=...)` and `Get(scope, spec)` overload — compose anything
  from fundamentals declared in JSON. Single-letter symbol collisions
  resolved via a priority registry (alias_active > symbol_derived >
  stem_active > symbol_active > stem_derived).
- `Save(name, composed, *, tier, dir)` — persist composed entries; explicit
  `Symbol` lets callers shadow active-tier entries deliberately (e.g. atom
  Hydrogen shadows Higgs Boson `H`).
- `Scope` enum with friendly tier members for every shipped tier
  (`Fundamental`, `SubAtomic`, `Atom`, `Atoms`, `Molecule`, `Ion`,
  `Isotope`, `AminoAcid`, `Protein`, `Alloy`, `Polymer`, `Ceramic`,
  `Composite`, `HadronsGen`, plus singular/plural aliases).
- `data/config/composition_rules.json` — single config file declaring
  registry tiers, additive properties, derived root, placeholder prefixes.

### Added — registry tiers shipped pre-built
| Tier | Source | Entries |
|---|---|---:|
| fundamentals | active/quarks      | 19  |
| subatomic    | active/subatomic   | 24  |
| amino_acids  | active/amino_acids | 24  |
| proteins     | active/proteins    | 20  |
| atoms        | derived/atoms      | 118 |
| isotopes     | derived/isotopes   | 40  |
| ions         | derived/ions       | 30  |
| molecules    | derived/molecules  | 73  |
| hadrons_gen  | derived/hadrons_gen | 15 |
| alloys       | derived/alloys     | 10  |
| polymers     | derived/polymers   | 9   |
| ceramics     | derived/ceramics   | 6   |
| composites   | derived/composites | 4   |

### Added — 3D sampling
- `sample(name, prop, at=..., scale_m=...)` and `data_sheet(name)`.
- Field models `homogeneous`, `mixture`, `anisotropic_axial`,
  `backbone_path` (residue-aware for proteins), and
  `microstructure_voronoi` (point-hashed phase dispatch for alloys).
- `register_field_model(name, fn)` for user-defined field models.

### Added — protein folding + AlphaFold validation
- `build_backbone(sequence, phi_psi)` and `build_backbone_from_entry(name)` —
  pure-numpy NeRF construction of N/CA/C atom coords from per-residue
  phi/psi. Bond lengths / angles read from `data/config/folding_rules.json`.
- `kabsch_rmsd(P, Q)` — SVD-based optimal rotation + RMSD.
- `parse_pdb_backbone(path)` — pure-Python PDB ATOM parser.
- `fetch_alphafold(uniprot_id)` and `load_alphafold_reference(uniprot_id)` —
  bundled-first, cache-second, network-fallback. Five reference PDBs
  bundled (Crambin, Insulin Chain A, Ubiquitin, Lysozyme, Beta-Defensin).
- `ramachandran_region(phi, psi)`.

### Added — optimization
- `optimize_protein_folding(sequence, target='helical', iterations, seed)` —
  simulated annealing on phi/psi. Energy weights and Ramachandran regions
  in `folding_rules.json`.
- `optimize_alloy(targets={...}, base, candidates, top_k, seed)` — random
  search over Hume-Rothery-bounded compositions; alloying pool drawn from
  existing alloy inputs. Targets like `YieldStrength_MPa_min` and
  `Density_kgm3_max`.

### Added — generator scripts
- Generic `_runner.build_from_input(path)` accepts JSON inputs with
  `from`, `tier`, `requires`, `rows`, plus optional `properties`, `field`,
  `aliases`, `symbol` per row.
- `build_all()` topologically sorts inputs by their `requires`.
- Default scripts: `build_hadrons`, `build_periodic_table`, `build_isotopes`,
  `build_ions`, `build_molecules`, `build_alloys`, `build_polymers`,
  `build_ceramics`, `build_composites`.

### Added — 3D export
- `export_stl(name, path, ...)` — voxel-block boundary mesh, binary or ASCII.
- `export_obj(name, path, ...)` — OBJ + sibling MTL with per-phase materials,
  vertex RGB encoding phase identity, property comments in MTL.
- `export_vtk_legacy(name, path, properties=...)` — STRUCTURED_POINTS with
  `phase_id` plus per-property scalar arrays.
- `export_sdf_raw(name, path, mode=...)` — binary float32 3D volume +
  JSON sidecar header (occupancy or phase mode).
- `export_hlsl(name, path, properties=...)` — emits a
  `Sample<Name>(float3 pos, float scaleM)` HLSL function with hash-based
  phase dispatch and per-phase property values.

### Added — CLI (`periodica` console script)
- `build [SCRIPTS...] [--extra PATH] [-q]`
- `clear [TIER] [-y]`
- `run PATH`
- `list [TIER]`
- `show NAME`
- `inputs`
- `sample NAME [PROP] [--at X,Y,Z] [--scale METRES]`
- `fold NAME [--validate] [--output PDB] [--uniprot ID] [--offline]`
- `validate NAME [--uniprot ID] [--offline]`
- `optimize folding|alloy [...]`
- `export NAME --format stl|obj|vtk|sdf|hlsl --output PATH ...`

### Added — install variants
- Slim `pip install periodica` — only `numpy>=1.24`.
- `pip install "periodica[full]"` — slim plus matplotlib, scipy,
  google-generativeai (none imported by periodica itself).
- `pip install "periodica[test]"` — pytest, twine, build.

### Reference / validation data
- `data/reference/` (test-only oracle): atoms (H, He, C, O, Fe), molecules
  (H2O, CO2, CH4, NH3, NaCl), subatomic (Proton, Neutron), alloys
  (Steel-1018, Aluminum-6061, Titanium-Ti6Al4V), polymers (HDPE, PET),
  ceramics (Alumina, SiliconCarbide), composites (CFRP). Generated values
  cross-checked against CODATA / IUPAC / PDG / AlphaFold / MatWeb /
  ASM / Aluminum Association.
- 1297 unit tests; 137 module-skipped legacy tests (UnifiedTable orphans
  + Qt-dependent app tests; do not validate physics).

### Removed
- Top-level legacy generator wrappers (`generate_atom`, `generate_particle`,
  `generate_molecule`, etc.) — fully redundant with `Get()`.
- Top-level `get_element` / `*_loader` shortcuts — use `Get(name)` instead.
  Loader classes remain importable via `periodica.data.<loader>` for
  advanced users.

### Pure-numpy contract
`periodica.folding` and `periodica.optimize` import only the standard
library + `numpy`. Verified by AST inspection plus a subprocess test
that confirms `import periodica.folding` does not load scipy/biopython.

### Test suite hygiene
- 1304 passed, 11 skipped (Gemini API tests auto-skip when `GOOGLE_API_KEY`
  isn't set), 1 xfail (`RegenerationEngine` H-2 isotope-selection bug;
  the `Get('{P=1,N=0,E=1}')` flow is unaffected and matches IUPAC mass
  within 0.001 amu).
- Removed 9 orphan test files (test_layouts.py for deleted PySide6
  layouts, 5 UnifiedTable orphans, 3 app-side `periodica_app.ui`
  dependents).
- Fixed test paths so granite-material spatial-sampling tests and
  hydrogen reference-keys tests now run instead of skipping.

## [1.0.0]

Initial PyPI release. See [git tag v1.0.0](https://github.com/andrewkwatts-maker/periodica/releases/tag/v1.0.0)
for that release's surface.
