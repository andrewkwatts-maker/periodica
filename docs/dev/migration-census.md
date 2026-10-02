# periodica M0 — Python census and triage

Repo `H:\Github\MathsPhysicsEcosystem\periodica` @ `0ffd2c1` (app @ `b76b118`), 2026-10-02. **147 modules, 52,117 LOC** under `src/periodica`. Read-only analysis; the coverage run used a scratch copy of the repo (pytest tests -q -m 'not slow and not gemini' on a scratch copy: 1419 passed, 1 skipped, 11 deselected, 1 xfailed (28 s)). Both repos stayed clean.

## How to read this

- **Reachability** is a static `ast` import graph with *symbol-level* resolution: `from pkg import X` follows re-export chains through `__init__` files to the module that defines `X`, so a package init that eagerly imports 20 modules does not make all 20 "used". Lazy (function-level) imports count as edges. Roots: `public` = every name in `periodica.__all__`; `cli` = `__main__`; `scripts` = `periodica.scripts.*`; `app` = every `periodica.*` import in `periodica-app/src`; `tests` = `tests/*.py` (imports and `"periodica.x.y"` strings); `documented` = deep imports shown in README.md. `dead` = reached by nothing. "loaded on import" = executed by a bare `import periodica` (package-init side effects).
- **Classes**: PORT = science logic that moves to Rust (M6 survivors are the one canonical copy of each semi-empirical model, chosen here); ALREADY-NATIVE = a Rust implementation exists in `rust/periodica_core/src` (called-today / agrees columns below); WRAPPER = stays thin Python (I/O, enums, hooks, CLI, constants facade); MOVE-TO-APP = presentation; DELETE = dead, test-only duplicate, or superseded. Some modules split (e.g. `physics_calculator` = 45-line `PhysicsConstants` WRAPPER + 2,253 lines DELETE); split LOC are listed in `triage.json` (`split`).
- **Effort**: S ≤ 4 h, M 4–12 h, L > 12 h of agent time, for the remaining work (for ALREADY-NATIVE: finishing, fixing and switching over).

## Totals

| Class | Modules | LOC (whole-module) | LOC (split-adjusted) |
|---|---:|---:|---:|
| PORT | 14 | 5,828 | 5,828 |
| ALREADY-NATIVE | 5 | 2,116 | 1,884 |
| WRAPPER | 32 | 9,034 | 7,013 |
| MOVE-TO-APP | 38 | 5,813 | 5,558 |
| DELETE | 58 | 29,326 | 31,834 |
| **Total** | **147** | **52,117** | **52,117** |

Reachability (first matching root in the order public → cli → scripts → app → tests → dead):

| Reached first from | Modules | LOC |
|---|---:|---:|
| public | 24 | 10,148 |
| cli | 1 | 604 |
| scripts | 10 | 129 |
| app | 11 | 1,784 |
| tests | 57 | 32,520 |
| dead | 44 | 6,932 |

Only **24 modules / 10,148 LOC (19 %)** are reachable from the public API; the science on that path is `get`, `sample`, `export`, `folding`, `optimize` (2,116 LOC) plus `physics_calculator`/`nuclear_derivation`, which are reachable only because `PhysicsConstants` lives in the same module as `AtomCalculator`. **57 modules / 32,520 LOC (62 %) are reached only by tests**; 44 modules / 6,932 LOC are reached by nothing (26 layout_math files, 12 package inits, 6 others).

## Triage table

### Root, CLI, dispatch — 10 modules, 3,505 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `__init__.py` | 185 | public, cli, tests (loaded on import) | 100 | **WRAPPER** |  |  | Public surface. At 3.0: `_HAS_RUST` deprecated True; stop importing utils.physics_calculator (re-export PhysicsConstants from the native constants wrapper). Importing it today loads 38 periodica modules (279 ms measured incl. extension). |
| `__main__.py` | 604 | cli, tests | 58 | **WRAPPER** |  |  | CLI (argparse) over Get/sample/export/folding/optimize/_runner; stays Python, every subcommand becomes a call into native. |
| `_dispatch.py` | 332 | public, cli, tests (loaded on import) | 72 | **WRAPPER** |  |  | Becomes `_native.py` (@native decorator + conformance registry). ~60% (fallback counting, BackendMode.PYTHON, FallbackRequired, call_rust fallbacks) deleted at 3.0. |
| `_launcher.py` | 52 | public, tests (loaded on import) | 33 | **WRAPPER** |  |  | Companion-app launcher (I/O, subprocess). Allow-listed. |
| `constants.py` | 216 | tests | 100 | **DELETE** |  |  | Qt-era UI/visual constants (LayoutMode, GlowType, ColorConstants, UIConstants) + a PhysicalConstants copy (r0 = 1.2 fm). No library or app consumer; only tests/test_config.py. |
| `export.py` | 535 | public, cli, tests (loaded on import) | 96 | **ALREADY-NATIVE** | periodica-core export (export.rs; voxel_* in sample.rs) | M | Byte-identical goldens are the M3 gate; A9 (sdf_raw writes occupancy, hash spec, Fourier half-voxel). |
| `folding.py` | 354 | public, cli, tests (loaded on import) | 77 | **ALREADY-NATIVE** | periodica-core protein (protein.rs) | M | fetch_alphafold / load_alphafold_reference stay Python I/O; parse_pdb_backbone can stay Python (text I/O) or move with A8 chain-key fix. |
| `get.py` | 530 | public, cli, scripts, tests (loaded on import) | 94 | **ALREADY-NATIVE** | periodica-core registry (registry.rs, get.rs, data_loader.rs) | L | No periodica imports at all: the served numbers never touch utils/* predictors. |
| `optimize.py` | 319 | public, cli, tests (loaded on import) | 90 | **ALREADY-NATIVE** | periodica-core protein (SA) + alloy.rs -> periodica-mat estimates | M | Alloy property estimation is the 5th rule-of-mixtures copy - consolidate in periodica-mat. |
| `sample.py` | 378 | public, cli, tests (loaded on import) | 47 | **ALREADY-NATIVE** | periodica-core sample (sample.rs, sample_models.rs) | S | Remaining: Arc<Value>, batch py_sample_many, compiled-sampler cache; register_field_model must be consulted FIRST (today a Python-registered model only runs after Rust declines, and re-registering a built-in name is ignored by Rust). Python field-model bodies (~250 LOC) delete at 3.0; the hook stays. |

### core/ (enums) — 13 modules, 4,427 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `core/__init__.py` | 215 | documented (loaded on import) | 100 | **WRAPPER** |  |  | Enum re-export package (documented in README). |
| `core/alloy_enums.py` | 479 | public, tests (loaded on import) | 94 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). colour helpers get_element_color/get_ipf_color -> MOVE-TO-APP |
| `core/amino_acid_enums.py` | 455 | public, tests (loaded on import) | 95 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). get_amino_acid_property_metadata is display metadata |
| `core/biomaterial_enums.py` | 267 | public, tests (loaded on import) | 98 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). |
| `core/cell_component_enums.py` | 237 | public, tests (loaded on import) | 100 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). |
| `core/cell_enums.py` | 367 | public, tests (loaded on import) | 99 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). |
| `core/material_enums.py` | 188 | public, tests (loaded on import) | 92 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). get_material_color -> MOVE-TO-APP |
| `core/molecule_enums.py` | 407 | public, tests (loaded on import) | 95 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). get_element_color -> MOVE-TO-APP (third element-colour table) |
| `core/nucleic_acid_enums.py` | 305 | public, tests (loaded on import) | 98 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). translate_sequence/transcribe_dna/reverse_complement: string utilities, keep |
| `core/protein_enums.py` | 276 | public, tests (loaded on import) | 99 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). calculate_protein_mass/calculate_isoelectric_point (+ WATER_MASS 18.015) duplicate protein_predictor -> DELETE, use the native protein port |
| `core/pt_enums.py` | 433 | public, tests (loaded on import) | 98 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). |
| `core/quark_enums.py` | 358 | public, app, tests (loaded on import) | 99 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). used by the app via layout_math.quark_* |
| `core/subatomic_enums.py` | 440 | public, tests (loaded on import) | 91 | **WRAPPER** |  |  | Enums (descriptive, allow-listed). *LayoutMode enum is presentation -> MOVE-TO-APP at 3.0 (breaking: in __all__). get_particle_family_color -> MOVE-TO-APP |

### data/ — 12 modules, 3,438 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `data/__init__.py` | 143 | documented (loaded on import) | 100 | **WRAPPER** |  |  | Package init eagerly imports every loader (loaded by `import periodica`). Shrinks to DataManager + quark_source re-exports once loaders are removed. |
| `data/alloy_loader.py` | 237 | tests, documented (loaded on import) | 85 | **DELETE** |  |  | Superseded by Get over alloys tier. Second, divergent JSON read path; keep a 2.5 DeprecationWarning shim forwarding to Get, delete at 3.0. |
| `data/data_manager.py` | 373 | public, app, tests (loaded on import) | 80 | **WRAPPER** |  |  | CRUD over active/ + defaults/ JSON (add/edit/remove/reset) - user-data file I/O, used by the app (get_all_items/remove_item/reset_category). Read path should delegate to the native registry; writes must trigger reload_registry. |
| `data/element_data.py` | 641 | tests | 19 | **DELETE** |  |  | Electron-config/shell helpers; A4 makes element JSON `electron_configuration` the only source. Consumers (orbital_clouds, position_calculator) are themselves DELETE/MOVE. |
| `data/element_loader.py` | 421 | tests, documented (loaded on import) | 71 | **DELETE** |  |  | get_element/get_element_by_z (README). Superseded by Get/data_sheet over the atoms tier. Second, divergent JSON read path; keep a 2.5 DeprecationWarning shim forwarding to Get, delete at 3.0. |
| `data/layout_config_loader.py` | 202 | dead | 0 | **DELETE** |  |  | Dead (no importer anywhere, 0 % coverage). |
| `data/material_data.py` | 155 | tests | 89 | **DELETE** |  |  | Superseded by Get / periodica-mat MaterialRecord. Second, divergent JSON read path; keep a 2.5 DeprecationWarning shim forwarding to Get, delete at 3.0. |
| `data/molecule_loader.py` | 189 | tests, documented (loaded on import) | 81 | **DELETE** |  |  | Superseded by Get over molecules (and active/molecules must become a tier - M1). Second, divergent JSON read path; keep a 2.5 DeprecationWarning shim forwarding to Get, delete at 3.0. |
| `data/quark_loader.py` | 375 | app, tests, documented (loaded on import) | 86 | **MOVE-TO-APP** |  |  | The app's only data source for the quark screen. JSON loading -> Get/list over the quarks tier (native); sm_row/sm_col Standard-Model grid enrichment (l.240-300) is presentation -> app. |
| `data/quark_source.py` | 258 | app, tests (loaded on import) | 81 | **WRAPPER** |  |  | experimental/simulated source switch (optional metaphysica import = I/O policy). SURPRISE: it only affects quark_loader/load_quark - `Get('u')` never consults it, so `simulated` is invisible to the registry. Make it a registry overlay in M1 or move it with quark_loader. |
| `data/subatomic_loader.py` | 338 | tests, documented (loaded on import) | 81 | **DELETE** |  |  | Superseded by Get over subatomic tier. Second, divergent JSON read path; keep a 2.5 DeprecationWarning shim forwarding to Get, delete at 3.0. |
| `data/validation.py` | 106 | tests (loaded on import) | 100 | **DELETE** |  |  | Datasheet field validation; moves into native ingest (D7 typed errors, periodica-mat ingest). |

### layout_math/ — 35 modules, 4,757 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `layout_math/__init__.py` | 115 | app | 100 | **MOVE-TO-APP** |  |  | Package init eagerly imports all 35 layouts; the app imports `from periodica.layout_math import quark_*`. |
| `layout_math/alloy_category.py` | 115 | dead | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/alloy_composition.py` | 122 | dead | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/alloy_lattice.py` | 124 | dead | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/alloy_property.py` | 119 | dead | 11 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/base.py` | 89 | dead | 44 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/element_circular.py` | 141 | documented | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. README-documented. |
| `layout_math/element_linear.py` | 353 | dead | 29 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/element_spiral.py` | 173 | dead | 10 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/element_table.py` | 177 | documented | 18 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. README-documented. |
| `layout_math/molecule_bond.py` | 110 | dead | 12 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_bond_complexity.py` | 177 | dead | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_density.py` | 113 | dead | 8 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_dipole.py` | 127 | dead | 8 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_geometry.py` | 120 | dead | 10 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_grid.py` | 66 | dead | 18 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_mass.py` | 102 | dead | 8 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_phase_diagram.py` | 115 | dead | 8 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/molecule_polarity.py` | 104 | dead | 12 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/quark_alternative.py` | 129 | app | 9 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_charge_mass.py` | 120 | app | 10 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_circular.py` | 128 | app | 9 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_fermion_boson.py` | 125 | app | 12 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_force_network.py` | 139 | app | 12 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_linear.py` | 112 | app | 12 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_mass_spiral.py` | 137 | app | 12 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. |
| `layout_math/quark_standard.py` | 146 | app, documented | 10 | **MOVE-TO-APP** |  |  | Used by periodica-app quark screen. README-documented. |
| `layout_math/subatomic_baryon_meson.py` | 115 | dead | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_charge.py` | 108 | dead | 13 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_decay.py` | 147 | dead | 10 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_discovery.py` | 212 | dead | 7 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_eightfold.py` | 135 | dead | 6 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_lifetime.py` | 187 | dead | 8 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_mass.py` | 97 | dead | 16 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |
| `layout_math/subatomic_quark_tree.py` | 158 | dead | 8 | **MOVE-TO-APP** |  |  | Presentation; NO consumer in the app today (only the quark screen exists). Move with the screen that needs it, otherwise delete at 3.0. |

### scripts/ — 11 modules, 362 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `scripts/__init__.py` | 10 | scripts (loaded on import) | 100 | **WRAPPER** |  |  | Package docstring documents run order. |
| `scripts/_runner.py` | 233 | public, cli, scripts, tests (loaded on import) | 70 | **WRAPPER** | periodica-core registry (native compose) | S | Generic {name, spec} -> Get -> Save loop; becomes a thin CLI over native compose. Also read by optimize.py for _INPUTS_DIR (alloy pool). |
| `scripts/build_alloys.py` | 14 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_ceramics.py` | 10 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_composites.py` | 14 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_hadrons.py` | 14 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_ions.py` | 15 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_isotopes.py` | 14 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_molecules.py` | 14 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_periodic_table.py` | 14 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |
| `scripts/build_polymers.py` | 10 | scripts | 0 | **WRAPPER** |  |  | 10-15-line entry point naming an inputs JSON. |

### utils/ (excluding predictors, transforms) — 33 modules, 24,821 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `utils/__init__.py` | 111 | dead (loaded on import) | 100 | **DELETE** |  |  | Eager re-exports: `import periodica` (via utils.physics_calculator) also executes simulation_schema, pure_math, pure_array, backend_manager (4,146 LOC) that nothing public uses. |
| `utils/alloy_calculator.py` | 1375 | tests | 91 | **DELETE** |  |  | Category-heuristic strength model + Vegard/RoM; duplicates rule_of_mixtures, periodica-mat derive.rs and alloy.rs estimate. Only consumer alloy_generator (DELETE). |
| `utils/alloy_generator.py` | 674 | tests | 93 | **DELETE** |  |  | Generator superseded by scripts/_runner + inputs/alloys.json; Hume-Rothery sampling duplicates optimize_alloy. |
| `utils/atomic_derivation.py` | 396 | tests | 91 | **DELETE** |  |  | Old pipeline; duplicates slater_predictor (constants copy l.42). |
| `utils/backend_manager.py` | 467 | dead (loaded on import) | 13 | **DELETE** |  |  | Dead (only re-exported by utils/__init__); scipy-vs-pure switch superseded by _dispatch/native. |
| `utils/biological_component_factory.py` | 538 | tests | 59 | **DELETE** |  |  | Factory superseded by registry + scripts. |
| `utils/biological_derivation_chain.py` | 628 | tests | 85 | **DELETE** |  |  | Superseded by Get composition. |
| `utils/biological_generator.py` | 683 | tests | 84 | **DELETE** |  |  | Generator facade superseded by registry + scripts. |
| `utils/bonding_rules.py` | 175 | tests | 96 | **PORT** | periodica-chem | S | Electronegativity bond classification + valence/octet checks; useful for formula validation (M5) and bond perception. Consumers today: molecule_generator (DELETE), phase_diagram (lazy). |
| `utils/cascade_engine.py` | 396 | tests | 69 | **DELETE** |  |  | Plan list - verified: no importer; drives the DELETE generators. |
| `utils/color_math.py` | 472 | tests | 31 | **MOVE-TO-APP** |  |  | Colour maps for the periodic table (Qt-free); the app already has periodica_app/utils/color_utils. calculate_emission_spectrum is a crude Rydberg-like estimate - rebuild in periodica-qm if needed. |
| `utils/crystalline_math.py` | 1698 | tests | 18 | **DELETE** |  |  | Plan list - verified: no importer; tests/test_utils_calculations.py only. Lattices/Voronoi/noise superseded by periodica-runtime crystal + export. |
| `utils/derivation_metadata.py` | 150 | tests | 97 | **DELETE** |  |  | Provenance moves to native (periodica-mat Source, D3 MassSource). |
| `utils/logger.py` | 24 | tests (loaded on import) | 100 | **DELETE** |  |  | 24-line logging helper; every consumer is DELETE. Use logging.getLogger. |
| `utils/material_generator.py` | 344 | tests | 93 | **DELETE** |  |  | Generator superseded by scripts/_runner. |
| `utils/molecular_geometry.py` | 992 | tests | 11 | **DELETE** |  |  | Plan list - verified: no importer; 2 test files. Wrong coordinates (A6) -> curated Geometry3D (Q3/B2). |
| `utils/molecule_generator.py` | 531 | tests | 93 | **DELETE** |  |  | Generator superseded by scripts/_runner + inputs/molecules.json and M5 formula engine. |
| `utils/nuclear_derivation.py` | 347 | public, tests (loaded on import) | 89 | **DELETE** |  |  | Old pipeline; SEMF set A + r0 = 1.25. Public only through AtomCalculator.calculate_binding_energy (which uses set A while AtomCalculator.calculate_atomic_mass uses set B). |
| `utils/orbital_clouds.py` | 691 | tests | 53 | **DELETE** |  |  | A1 normaliser bug ((n+l)!**3). Replace with a deprecated shim over periodica-qm hydrogenic (A15), delete at 3.0. |
| `utils/phase_diagram.py` | 290 | tests | 96 | **PORT** | periodica-mat | S | Regular-solution model (omega, T_c, miscibility gap), Lindemann melting, Hume-Rothery solubility. Depends on rule_of_mixtures (lazy) and bonding_rules. |
| `utils/physics_calculator.py` | 2298 | public, tests, documented (loaded on import) | 51 | **WRAPPER** | constants (A5) -> Rust const | S | Public only for PhysicsConstants (l.15-57) -> facade over the single constants JSON / Rust const. AtomCalculator / SubatomicCalculator / MoleculeCalculator (README-documented, but README's `from periodica import AtomCalculator` raises ImportError) predict melting point/density/radius from Z and duplicate SEMF/hyperfine/VSEPR -> DELETE (2.5 deprecation). Unused NUCLEAR_RADIUS_CONST = 1.25. |
| `utils/physics_calculator_v2.py` | 4865 | tests | 67 | **DELETE** |  |  | Plan list - verified: no library/app importer; 6 test files only (test_data_driven_calculators, test_derivation_chain, test_deviation_report, test_json_validation, test_physics_calculations, test_prediction_chain). Own constants copy, r0 = 1.2, SEMF set B. |
| `utils/position_calculator.py` | 209 | tests | 18 | **MOVE-TO-APP** |  |  | Element positions for a layout (presentation). No app consumer today -> move with a screen or delete. |
| `utils/prediction_engine.py` | 235 | tests | 82 | **DELETE** |  |  | Old derivation pipeline (truncated m_p/m_n/m_e); superseded by predictors and now by Get composition. |
| `utils/predictive_physics.py` | 1020 | tests | 40 | **DELETE** |  |  | Property extrapolation engine; consumers alloy_calculator/v2 are DELETE. Constants copy l.30-36, r0 = 1.25. |
| `utils/pure_array.py` | 885 | dead (loaded on import) | 24 | **DELETE** |  |  | Dead except via sdf_core (DELETE); numpy replacement. |
| `utils/pure_math.py` | 1647 | tests (loaded on import) | 40 | **DELETE** |  |  | scipy.special replacement (Laguerre/Legendre/Y_lm) + ImprovedOrbitalCalculator (CLEMENTI_ZEFF, A3). Superseded by periodica-qm special functions (A1). Constants copy l.828. |
| `utils/quark_constants.py` | 176 | tests | 94 | **DELETE** |  |  | Plan list - verified: no importer; tests/test_quark_constants.py only. Quark masses come from datasheets. |
| `utils/regeneration_engine.py` | 304 | tests | 90 | **DELETE** |  |  | Orchestrates the old pipeline; no consumer except cascade_engine. |
| `utils/report_logger.py` | 190 | dead | 0 | **DELETE** |  |  | Plan list - verified: dead, 0 % coverage. |
| `utils/sdf_core.py` | 555 | dead | 14 | **DELETE** |  |  | Plan list - verified: dead (no importer at all). Nucleus/orbital SDFs superseded by periodica-render/qm. r0 = 1.25 default. |
| `utils/simulation_schema.py` | 1147 | tests (loaded on import) | 83 | **DELETE** |  |  | Dataclasses + propagate_* chain (4th SEMF copy, r0 = 1.2, constants copy). Loaded on every `import periodica` via utils/__init__. |
| `utils/validation_report.py` | 308 | tests | 93 | **DELETE** |  |  | Plan list - verified: no importer; replaced by generated docs/accuracy.html (ACC). |

### utils/predictors/** — 26 modules, 7,847 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `utils/predictors/__init__.py` | 41 | documented | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/alloy/__init__.py` | 14 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/alloy/rule_of_mixtures.py` | 474 | tests, documented | 34 | **PORT** | periodica-mat (Rust miedema stub lives in periodica_core) | M | Miedema dH_f + Vegard + RoM density/melting (README-documented). Becomes the ONE rule-of-mixtures implementation (sample `mixture`, optimize_alloy, alloy.rs estimate, biomaterial/alloy_calculator copies). |
| `utils/predictors/atomic/__init__.py` | 13 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/atomic/slater_predictor.py` | 496 | tests, documented | 88 | **PORT** | periodica-qm | M | Slater rules + relativistic corrections + Aufbau anomalies; per owner decision a teaching toggle beside RHF-STO. Must read configs from element JSON (A4). Rust predictors::slater is a stub. Constants copy l.34. |
| `utils/predictors/base.py` | 433 | tests | 72 | **DELETE** |  |  | Input/Result dataclasses + ABCs; replaced by the Rust Predictor trait + typed structs (.pyi). |
| `utils/predictors/biological/__init__.py` | 11 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/biological/amino_acid_predictor.py` | 408 | tests | 87 | **PORT** | periodica-core protein | S | Per-residue pKa/propensity lookups from amino-acid JSON; merge into the protein port. |
| `utils/predictors/biological/biomaterial_predictor.py` | 483 | tests | 71 | **DELETE** |  |  | Rule-of-mixtures composite estimate (another RoM copy); no consumer outside DELETE generators. |
| `utils/predictors/biological/cell_predictor.py` | 419 | tests | 66 | **DELETE** |  |  | Kleiber scaling 'predictions' - not a 99.9 % model, no consumer outside DELETE generators. Owner call: keep only as a labelled toy model. |
| `utils/predictors/biological/nucleic_acid_predictor.py` | 593 | tests | 76 | **PORT** | periodica-core protein (biopolymer) | M | Nearest-neighbour Tm/dG from config/{dna,rna}_thermodynamics.json. Owner call (no consumer today). |
| `utils/predictors/biological/protein_predictor.py` | 720 | tests, documented | 87 | **PORT** | periodica-core protein | M | MW, pI (Henderson-Hasselbalch), GRAVY, instability index (config/protein_instability.json), Chou-Fasman (Rust predictors::chou_fasman is a stub; optimize._aa_propensity is a 2nd copy). README-documented. |
| `utils/predictors/chain.py` | 1196 | tests | 42 | **DELETE** |  |  | DerivationChain quark->material (5th SEMF copy, set B, r0 = 1.25, truncated masses). Superseded by Get composition + composition_rules.json. |
| `utils/predictors/compat.py` | 37 | tests | 58 | **DELETE** |  |  | Legacy aliases PredictionEngine/NuclearDerivation/AtomicDerivation. |
| `utils/predictors/hadron/__init__.py` | 15 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/hadron/constituent_predictor.py` | 402 | tests | 77 | **PORT** | periodica-core predictors::constituent (stub) | S | Constituent-quark model on top of hyperfine; merge into one native `constituent` model. |
| `utils/predictors/hadron/hyperfine.py` | 184 | tests, documented | 78 | **PORT** | periodica-core predictors::constituent (stub) | S | Canonical De Rujula-Georgi-Glashow formula (README-documented). |
| `utils/predictors/material/__init__.py` | 3 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/material/material_predictor.py` | 393 | tests | 84 | **DELETE** |  |  | Heuristics (UTS = 1.2 sigma_y, fatigue = 0.4 UTS, Tabor 3.0 vs 3.3 in periodica-mat); exact relations already native in periodica-mat derive.rs. |
| `utils/predictors/molecule/__init__.py` | 10 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/molecule/huckel.py` | 113 | tests | 90 | **PORT** | periodica-qm | S | Hueckel pi-MO energies/aromaticity (plan names it); needs a symmetric eigen-solver (nalgebra). |
| `utils/predictors/molecule/vsepr_predictor.py` | 454 | tests | 86 | **PORT** | periodica-chem | S | VSEPR class + ideal angles as a labelled model; coordinates come from curated Geometry3D, never from here. Rust predictors::vsepr is a stub. |
| `utils/predictors/nuclear/__init__.py` | 24 | dead | 100 | **DELETE** |  |  | Package init of the Python predictor framework. Survivors get a new thin wrapper (e.g. `periodica.models`) over pyfacade/predictors.rs; old path keeps a 2.5 deprecation shim. |
| `utils/predictors/nuclear/semf_predictor.py` | 345 | tests | 92 | **PORT** | periodica-core predictors::semf (stub) | S | Canonical SEMF survivor (Z(Z-1) Coulomb, shell + Wigner terms). Rust semf.rs is a stub with a THIRD coefficient set (Krane 15.5/16.8/0.72/23.0/34.0) and r0 = 1.2; this file uses 15.56/17.23/0.7/23.285/12.0 and r0 = 1.25. Label as model; isotopes served from AME2020 (D3). |
| `utils/predictors/protocols.py` | 388 | dead | 80 | **DELETE** |  |  | Dead (only re-exported by predictors/__init__). |
| `utils/predictors/registry.py` | 178 | tests | 67 | **DELETE** |  |  | Singleton predictor registry; replaced by Rust registered_predictor_names(). |

### utils/transforms/ — 7 modules, 2,960 LOC

| Module | LOC | Reachable from | Cov % | Class | Target crate | Effort | Notes |
|---|---:|---|---:|---|---|---|---|
| `utils/transforms/__init__.py` | 39 | documented | 100 | **DELETE** |  |  | Package init of the transforms stack. |
| `utils/transforms/fourier_field.py` | 353 | tests | 65 | **DELETE** |  |  | Evaluator only; baking is native (fourier_bake.rs). If the 2 sheets' Fourier blocks must be served, add a field model in periodica-runtime instead. |
| `utils/transforms/material_sampler.py` | 459 | tests | 84 | **DELETE** |  |  | Duplicates sample() on a different schema; consumes Fourier/Wavelet/Stochastic/TemperatureDependencies blocks present in only 2 datasheets (Granite_Westerly, Steel_8620_CaseHardened). |
| `utils/transforms/optical_properties.py` | 645 | tests | 52 | **PORT** | periodica-mat | M | Drude/Lorentz/Sellmeier/Cauchy dispersion (1 sheet uses drude). Relevant to IOR baking for the renderer; port only with a consumer, else DELETE. Constants copy (h, hbar truncated). |
| `utils/transforms/stochastic_field.py` | 427 | tests | 78 | **DELETE** |  |  | Karhunen-Loeve heterogeneity; same disposition as fourier_field. |
| `utils/transforms/thermo_pressure.py` | 529 | tests | 80 | **PORT** | periodica-mat | M | Temperature dependence (tabulated/polynomial, data present in 2 sheets) + Birch-Murnaghan EOS. Port only if T-dependent properties become served data; else DELETE. |
| `utils/transforms/wavelet_field.py` | 508 | tests | 26 | **DELETE** |  |  | Same as fourier_field; 26 % coverage, no test imports it directly. |

## ALREADY-NATIVE detail (Rust in `rust/periodica_core/src`)

Spot-checked with `parity.py` (scratch) calling `periodica._periodica_core` directly against the Python code.

| Python module | Rust symbols | Called today? | Agrees? | Remaining effort |
|---|---|---|---|---|
| `export.py` | py_export_stl/obj/vtk_legacy/sdf_raw/hlsl (+ Rust-only py_export_glsl, py_bake_fourier) | NO - Python export.py never dispatches. | MIXED: VTK numerically identical (formatting '0' vs '0.0'); sdf_raw occupancy byte-identical but default mode differs (Python 'occupancy', Rust 'phase'); STL 348 vs 270 triangles; OBJ structure differs (per-phase objects vs flat); HLSL entirely different (44 vs 18 lines); API differs (OBJ `properties`, VTK `properties` required vs optional). voxel_sample/voxel_phase_map exist in sample.rs but are not exposed. | M (M3) |
| `folding.py` | py_kabsch_rmsd / py_build_backbone / py_build_backbone_from_entry / py_ramachandran_region | NO. | NO: py_kabsch_rmsd is WRONG (0.94-3.12 A over 8 random cases where the true RMSD is 0.145-0.190 A; Python matches an independent SVD reference to 1e-6); py_build_backbone collapses every atom after residue 1 (bond lengths 0), max deviation 9.4 A (10-mer), 45.6 A Crambin, 66.4 A Ubiquitin; returns list-of-dicts vs (n,3,3) array; ramachandran: 55/625 grid points differ, all case-only ('polyproline_ii' vs 'polyproline_II'), plus Python None vs Rust 'other'. | M (M4) |
| `get.py` | py_get / py_save / py_list_tiers (registry.rs, get.rs, data_loader.rs; reload not exported) | NO for Get/Save/list_tiers (pure Python). The Rust registry is used only inside py_sample/py_data_sheet. | PARTIAL: name resolution + list_tiers agree (parity tests; spot-check Fe, Steel-1018, H2O identical). Brace specs: numeric totals agree but Rust copies `Name` from the first constituent ('{u=2,d=1}' -> 'Up Quark') and returns Charge_e 1.0000000001 vs 1. Both share D1/D3/D4 ('{N=1,H=3}' = 4.032 u, '{H=2,O=1}' = 18.148 u). 'Water'/'Oxygen' fail in both. | L (M1 (+D1-D9)) |
| `optimize.py` | py_optimize_alloy (alloy.rs); none for optimize_protein_folding | NO. | NO: py_optimize_alloy returns [] for every input tried (correct target schema, with and without a pool) where Python returns results; different signature (list of {property,min_value,max_value,weight} vs mapping 'X_min'). Python itself returns 3 identical {'Fe': 1.0} rows for top_k=3 (no dedupe; D9 missing-data-as-0). | M (M4) |
| `sample.py` | py_sample / py_data_sheet | YES (call_rust for str names; dict form is Python). | YES - parity tests green across tiers/properties. | S (M2) |

**Rust predictors are stubs.** All six `periodica_core::predictors::{semf, slater, vsepr, miedema, chou_fasman, constituent}` entry points return `Err("… not yet implemented")` (only `semf::nuclear_radius` computes, with r0 = 1.2 fm); none is exported through `pyfacade.rs`, and their doc comments cite Python files that do not exist (`utils/predictors/semf.py`, `slater.py`, `miedema.py`, `vsepr.py`, `chou_fasman.py`, `constituent_quark.py`). They count as PORT targets, not native implementations. `py_bake_fourier` and `py_export_glsl` are Rust-only (no Python caller). `periodica-mat` and `periodica-runtime` have no Python surface at all.

## PORT items, dependencies and recommended order

| # | Module | Target | Effort | Milestone | Depends on |
|---:|---|---|---|---|---|
| 1 | utils/physics_calculator.py (PhysicsConstants only) | constants JSON → Rust `const` + generated Python facade | S | A5 (inside M1) | — |
| 2 | get.py | periodica-core registry | L | M1 + D1–D9 | constants (Mass_kg/amu derivation), curated mass sources (D3) |
| 3 | sample.py | periodica-core sample | S | M2 | M1 (Arc registry) |
| 4 | (new) formula engine | periodica-chem | M | M5 | M1 (tier resolution) |
| 5 | export.py | periodica-core export | M | M3 | M2 (voxel_sample batch over the sampler) |
| 6 | folding.py | periodica-core protein | M | M4 | M1 (entries); fix Kabsch + NeRF first |
| 7 | optimize.py | periodica-core protein + periodica-mat | M | M4 | folding; rule_of_mixtures/D9 curated element props for optimize_alloy |
| 8 | scripts/_runner.py (+ build_*.py) | thin CLI over native compose | S | M7 | M1, M5 |
| 9 | layout_math/*, quark_loader enrichment, color_math, position_calculator | → app | M | M9 | M1 (quark tier via Get) |
| 10 | predictors/nuclear/semf_predictor.py | periodica-core predictors::semf | S | M6 | constants; pick r0 + coefficient set |
| 11 | predictors/hadron/hyperfine.py → constituent_predictor.py | periodica-core predictors::constituent | S+S | M6 | constants; hyperfine before constituent |
| 12 | predictors/alloy/rule_of_mixtures.py | periodica-mat | M | M6 | M1 element data; D9 (missing ≠ 0) — also replaces alloy.rs estimate |
| 13 | utils/phase_diagram.py | periodica-mat | S | M6 | rule_of_mixtures, bonding_rules |
| 14 | predictors/biological/protein_predictor.py + amino_acid_predictor.py | periodica-core protein | M+S | M6 | M4 (shares Chou–Fasman with optimize_protein_folding) |
| 15 | predictors/molecule/huckel.py | periodica-qm | S | M6 | P1 (crate), nalgebra eigen |
| 16 | predictors/atomic/slater_predictor.py | periodica-qm | M | M6 | A4 (element JSON configs), A1/A6 (sits beside RHF-STO as teaching toggle) |
| 17 | predictors/molecule/vsepr_predictor.py, utils/bonding_rules.py | periodica-chem | S+S | M6 | M5 (formula parse), Q3 (geometry stays curated) |
| 18 | predictors/biological/nucleic_acid_predictor.py | periodica-core protein (biopolymer) | M | M6 (low) | M1 (config JSON) |
| 19 | transforms/thermo_pressure.py, transforms/optical_properties.py | periodica-mat | M+M | M6 (low, owner call) | periodica-mat property ids; a consumer |

M6 sizing: the core survivors (rows 10–17, incl. bonding_rules) are ≈ 40 h; the low-priority tail (rows 18–19) adds ≈ 17 h. The plan budgets 30 h for M6, so either drop the tail (DELETE thermo_pressure/optical/nucleic) or move it past 3.0. Everything classed DELETE can be removed at any time (it blocks nothing), but per the plan's oracle-freeze rule freeze its outputs into `tests/oracles/` first where a survivor replaces it (SEMF, hyperfine, Slater, VSEPR, RoM).

## DELETE verification

The plan's eight (≈ 9.2 k lines) are all confirmed: no importer from the public API, CLI, scripts or the app.

| Module | LOC | Importers (library) | Tests that import it |
|---|---:|---|---|
| `utils/physics_calculator_v2.py` | 4,865 | none | test_data_driven_calculators.py, test_derivation_chain.py, test_deviation_report.py, test_json_validation.py, test_physics_calculations.py, test_prediction_chain.py |
| `utils/crystalline_math.py` | 1,698 | none | test_utils_calculations.py |
| `utils/sdf_core.py` | 555 | none | none |
| `utils/molecular_geometry.py` | 992 | none | test_molecular_geometry_accuracy.py, test_molecules_complete.py |
| `utils/cascade_engine.py` | 396 | none | test_cascade_engine.py |
| `utils/validation_report.py` | 308 | none | test_validation_report.py |
| `utils/report_logger.py` | 190 | none | none |
| `utils/quark_constants.py` | 176 | none | test_quark_constants.py |
| **Total** | **9,180** | | |

Beyond the plan's list the census adds **50 more DELETE modules (20,146 LOC)**: the old derivation pipelines (prediction_engine/atomic_derivation/nuclear_derivation/regeneration_engine, predictive_physics, simulation_schema, predictors/chain + framework), the generators (alloy/molecule/material/biological + factory + derivation chain + derivation_metadata), the pure-Python numerics (pure_math, pure_array, backend_manager, orbital_clouds via an A15 shim), the JSON loaders superseded by the registry (2.5 deprecation shims), the heuristic calculators (alloy_calculator, material_predictor, cell/biomaterial predictors, AtomCalculator/SubatomicCalculator/MoleculeCalculator), the transforms field stack, and `constants.py`.

Test files that import only DELETE modules (delete or convert to oracle goldens with their module):

`test_all_118_elements.py`, `test_alloy_generator.py`, `test_biological_derivation.py`, `test_biological_generator.py`, `test_cascade_engine.py`, `test_config.py`, `test_data_driven_calculators.py`, `test_data_validation.py`, `test_derivation_chain.py`, `test_derivation_metadata.py`, `test_e2e_derivation_chain.py`, `test_enhanced_orbital_accuracy.py`, `test_material_generator.py`, `test_molecular_geometry_accuracy.py`, `test_molecule_generator.py`, `test_molecules_complete.py`, `test_nuclear_shell.py`, `test_prediction_chain.py`, `test_quark_constants.py`, `test_quark_to_hadron_complete.py`, `test_regeneration_engine.py`, `test_relativistic_atomic.py`, `test_validation_report.py`.

Test files that mix DELETE and surviving modules (need editing): `test_biological_predictors.py`, `test_data_loaders.py`, `test_deviation_report.py`, `test_hybridization.py`, `test_json_validation.py`, `test_material_sampler.py`, `test_material_validation.py`, `test_physics_calculations.py`, `test_predictors_coverage.py`, `test_utils_calculations.py`.

## Duplicated physical constants (A5)

19 independent constant tables/literals (the 8 named in the brief plus 11 more). None is on the serving path: `get`/`sample`/`export`/`folding`/`optimize` and the Rust core contain no physical constants — served masses are sums of datasheet values. Values are mostly CODATA 2018 (m_p 938.27208816, a0 52.9177210903 pm, u 1.66053906660e-27 kg); CODATA 2022 is the plan's reference, and several copies are truncated (938.272, 939.565, 0.511, 931.494, 1.66054e-27, 6.62607e-34, 8.617e-5).

| Site | Kind | Values |
|---|---|---|
| `src/periodica/utils/physics_calculator.py:15-57` | PhysicsConstants (PUBLIC) | m_p/m_n/m_e (u, MeV; CODATA 2018), quark masses, Ry 13.605693122994, a0 52.9177210903 pm, alpha 0.0072973525693, N_A, r0 1.25 (unused), SEMF set B; plus inline 931.494 at l.126, l.1569 |
| `src/periodica/utils/pure_math.py:828-829` | module globals | alpha = 1/137.035999084, Ry |
| `src/periodica/utils/atomic_derivation.py:42-45 (+l.116)` | class attrs | m_e MeV, m_e u truncated 0.000548579909, Ry, a0 pm; 931.494 |
| `src/periodica/utils/predictors/atomic/slater_predictor.py:34-37` | class attrs | m_e MeV, m_e u (truncated), Ry, a0 pm |
| `src/periodica/utils/predictive_physics.py:30-36 (+l.997)` | PredictiveConstants | Ry, alpha, a0 pm; r0 1.25 |
| `src/periodica/utils/physics_calculator_v2.py:28-61 (+l.1227, 2054, 2198, 3531, 3567)` | PhysicsConstantsV2 | MEV_TO_AMU 931.494, AMU_TO_KG 1.66054e-27 (6 s.f.), MEV_TO_JOULE 1.60218e-13, N_A, k_B, h, c, e, Ry, a0, alpha, SEMF set B; m_p 938.272; r0 1.2 (x3) |
| `src/periodica/utils/simulation_schema.py:56-70 (+l.970, 985, 1006)` | SimulationConstants | hbar, c, e, m_p/m_n/m_e kg (8-9 s.f.), a0 m, alpha, Ry, AMU_TO_KG 1.66054e-27, MEV_TO_KG; r0 1.2; SEMF set B; 931.494 |
| `rust/periodica-mat/src/canon.rs:114-118` | Rust const | AMU_TO_KG 1.66053906660e-27 (CODATA 2018), EV_TO_J, N_A (in KJ_MOL_TO_J) |
| `src/periodica/constants.py:100-110` | PhysicalConstants | c, h (eV s, J s), e, R_inf 10973731.6 (truncated), r0 1.2 |
| `src/periodica/utils/nuclear_derivation.py:46-55` | class attrs | SEMF set A, r0 1.25, m_p 938.272, m_n 939.565 (truncated) |
| `src/periodica/utils/predictors/nuclear/semf_predictor.py:48-57` | class attrs | SEMF set A, r0 1.25, m_p/m_n truncated |
| `src/periodica/utils/predictors/chain.py:251-252, 484-488, 508, 606` | inline | m_p/m_n truncated; SEMF set B; r0 1.25; 931.494 |
| `src/periodica/utils/prediction_engine.py:62-64` | class attrs | m_p 938.272, m_n 939.565, m_e 0.511 |
| `src/periodica/utils/transforms/optical_properties.py:36-38` | module globals | c, h 6.62607e-34, hbar 1.054571e-34 (truncated) |
| `src/periodica/utils/transforms/thermo_pressure.py:29` | module global | k_B |
| `src/periodica/utils/color_math.py:23` | module global | e (J/eV) |
| `src/periodica/utils/predictors/biological/cell_predictor.py:44` | class attr | k_B = 8.617e-5 eV/K (4 s.f.) |
| `rust/periodica_core/src/predictors/semf.rs:36-45` | Rust Default | SEMF set C (Krane 15.5/16.8/0.72/23.0/34.0), r0 1.2 |
| `src/periodica/core/protein_enums.py:216` | module global | WATER_MASS 18.015 (derived, not CODATA) |

**r0 (R = r0·A^1/3)**

- **1.25 fm**: `utils/physics_calculator.py:50 (NUCLEAR_RADIUS_CONST, declared but unused)`; `utils/nuclear_derivation.py:53 (R0, used l.200)`; `utils/predictors/nuclear/semf_predictor.py:55 (R0, used l.191)`; `utils/predictors/chain.py:508`; `utils/predictive_physics.py:997`; `utils/sdf_core.py:164 (default arg r0_fm)`
- **1.2 fm**: `constants.py:108 (NUCLEAR_RADIUS_CONSTANT)`; `utils/physics_calculator_v2.py:2054, 3531, 3567`; `utils/simulation_schema.py:970`; `rust/periodica_core/src/predictors/semf.rs:44 (radius_prefactor_fm)`

**SEMF coefficient sets** (the same physics with three parameterisations and two Coulomb forms):

- **A 15.56/17.23/0.70/23.285/12.0, Coulomb Z(Z-1)**: `utils/nuclear_derivation.py:46-50`; `utils/predictors/nuclear/semf_predictor.py:48-52`
- **B 15.75/17.8/0.711/23.7/11.2**: `utils/physics_calculator.py:51-55 (Coulomb Z^2, l.103)`; `utils/physics_calculator_v2.py:52-56 (Z^2)`; `utils/predictors/chain.py:484-488 (Z(Z-1))`; `utils/simulation_schema.py:985 (Z(Z-1))`
- **C 15.5/16.8/0.72/23.0/34.0 (Krane)**: `rust/periodica_core/src/predictors/semf.rs:36-45 (stub)`

`AtomCalculator` uses set B for `calculate_atomic_mass` and set A (via `NuclearDerivation`) for `calculate_binding_energy` — two different binding energies from one class. Recommendation: one constants JSON (CODATA 2022 with uncertainties) → Rust `const` via build script + generated `PhysicsConstants`; r0 becomes a named model parameter (1.2 fm charge-radius convention, labelled ~5–10 % model error) and tabulated charge radii serve any 0.1 % claim; SEMF keeps one coefficient set with its source.

## Public API inventory (`periodica.__all__`, 72 names)

| Name | Implementing module | Native today | Rust symbol | Should be |
|---|---|---|---|---|
| `__version__` | `periodica` | no |  | allow-list |
| `Get` | `periodica.get` | no | py_get | @native |
| `Save` | `periodica.get` | no | py_save | @native |
| `Scope` | `periodica.get` | no |  | allow-list (enum/exception; raised from native via pyo3 create_exception) |
| `UnknownName` | `periodica.get` | no |  | allow-list (enum/exception; raised from native via pyo3 create_exception) |
| `UnknownConstituent` | `periodica.get` | no |  | allow-list (enum/exception; raised from native via pyo3 create_exception) |
| `UnknownTier` | `periodica.get` | no |  | allow-list (enum/exception; raised from native via pyo3 create_exception) |
| `RegistryCollision` | `periodica.get` | no |  | allow-list (enum/exception; raised from native via pyo3 create_exception) |
| `reload_registry` | `periodica.get` | no | (data_loader::reload_registry, not exported) | @native |
| `list_tiers` | `periodica.get` | no | py_list_tiers | @native |
| `sample` | `periodica.sample` | yes | py_sample | @native |
| `data_sheet` | `periodica.sample` | yes | py_data_sheet | @native |
| `register_field_model` | `periodica.sample` | no |  | allow-list (user hook, consulted before native) |
| `assert_rust_backend` | `periodica._dispatch` | no | is_rust_backend / version_rust (handshake) | allow-list (diagnostics); backend_mode/fallback_count/BackendMode deprecated at 3.0 |
| `backend_report` | `periodica._dispatch` | no |  | allow-list (diagnostics); backend_mode/fallback_count/BackendMode deprecated at 3.0 |
| `backend_mode` | `periodica._dispatch` | no |  | allow-list (diagnostics); backend_mode/fallback_count/BackendMode deprecated at 3.0 |
| `fallback_count` | `periodica._dispatch` | no |  | allow-list (diagnostics); backend_mode/fallback_count/BackendMode deprecated at 3.0 |
| `BackendMode` | `periodica._dispatch` | no |  | allow-list (diagnostics); backend_mode/fallback_count/BackendMode deprecated at 3.0 |
| `RustBackendUnavailable` | `periodica._dispatch` | no |  | allow-list (diagnostics); backend_mode/fallback_count/BackendMode deprecated at 3.0 |
| `build_backbone` | `periodica.folding` | no | py_build_backbone (BROKEN) | @native |
| `build_backbone_from_entry` | `periodica.folding` | no | py_build_backbone_from_entry (BROKEN) | @native |
| `extract_phi_psi` | `periodica.folding` | no |  | @native |
| `kabsch_rmsd` | `periodica.folding` | no | py_kabsch_rmsd (WRONG) | @native |
| `parse_pdb_backbone` | `periodica.folding` | no |  | allow-list (I/O) |
| `backbone_array_from_pdb` | `periodica.folding` | no |  | @native |
| `fetch_alphafold` | `periodica.folding` | no |  | allow-list (I/O) |
| `load_alphafold_reference` | `periodica.folding` | no |  | allow-list (I/O) |
| `ramachandran_region` | `periodica.folding` | no | py_ramachandran_region (label case differs) | @native |
| `ramachandran_in_allowed` | `periodica.folding` | no |  | @native |
| `folding_rules` | `periodica.folding` | no |  | @native |
| `optimize_protein_folding` | `periodica.optimize` | no |  | @native |
| `optimize_alloy` | `periodica.optimize` | no | py_optimize_alloy (returns []) | @native |
| `voxel_sample` | `periodica.export` | no | (sample::voxel_sample, not exported) | @native |
| `voxel_phase_map` | `periodica.export` | no | (sample::voxel_phase_map, not exported) | @native |
| `export_stl` | `periodica.export` | no | py_export_stl (differs) | @native |
| `export_obj` | `periodica.export` | no | py_export_obj (differs) | @native |
| `export_vtk_legacy` | `periodica.export` | no | py_export_vtk_legacy (numerically equal) | @native |
| `export_sdf_raw` | `periodica.export` | no | py_export_sdf_raw (default mode differs) | @native |
| `export_hlsl` | `periodica.export` | no | py_export_hlsl (differs) | @native |
| `PhysicsConstants` | `periodica.utils.physics_calculator` | no |  | allow-list (constants facade generated from the native constants table, A5) |
| `DataManager` | `periodica.data.data_manager` | no |  | allow-list (file I/O) |
| `DataCategory` | `periodica.data.data_manager` | no |  | allow-list (enum) |
| `PTPropertyName` | `periodica.core.pt_enums` | no |  | allow-list (enum) |
| `PTLayoutMode` | `periodica.core.pt_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `MoleculeLayoutMode` | `periodica.core.molecule_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `BondType` | `periodica.core.molecule_enums` | no |  | allow-list (enum) |
| `MoleculeCategory` | `periodica.core.molecule_enums` | no |  | allow-list (enum) |
| `MolecularGeometry` | `periodica.core.molecule_enums` | no |  | allow-list (enum) |
| `MoleculePolarity` | `periodica.core.molecule_enums` | no |  | allow-list (enum) |
| `QuarkLayoutMode` | `periodica.core.quark_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `ParticleType` | `periodica.core.quark_enums` | no |  | allow-list (enum) |
| `InteractionForce` | `periodica.core.quark_enums` | no |  | allow-list (enum) |
| `SubatomicLayoutMode` | `periodica.core.subatomic_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `ParticleCategory` | `periodica.core.subatomic_enums` | no |  | allow-list (enum) |
| `AlloyLayoutMode` | `periodica.core.alloy_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `AlloyCategory` | `periodica.core.alloy_enums` | no |  | allow-list (enum) |
| `CrystalStructure` | `periodica.core.alloy_enums` | no |  | allow-list (enum) |
| `AminoAcidLayoutMode` | `periodica.core.amino_acid_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `AminoAcidCategory` | `periodica.core.amino_acid_enums` | no |  | allow-list (enum) |
| `MaterialLayoutMode` | `periodica.core.material_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `MaterialCategory` | `periodica.core.material_enums` | no |  | allow-list (enum) |
| `ProteinLayoutMode` | `periodica.core.protein_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `SecondaryStructureType` | `periodica.core.protein_enums` | no |  | allow-list (enum) |
| `CellLayoutMode` | `periodica.core.cell_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `CellType` | `periodica.core.cell_enums` | no |  | allow-list (enum) |
| `NucleicAcidLayoutMode` | `periodica.core.nucleic_acid_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `NucleicAcidType` | `periodica.core.nucleic_acid_enums` | no |  | allow-list (enum) |
| `BiomaterialLayoutMode` | `periodica.core.biomaterial_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `BiomaterialType` | `periodica.core.biomaterial_enums` | no |  | allow-list (enum) |
| `CellComponentLayoutMode` | `periodica.core.cell_component_enums` | no |  | allow-list (enum); presentation -> app at 3.0 |
| `OrganelleType` | `periodica.core.cell_component_enums` | no |  | allow-list (enum) |
| `Launch` | `periodica._launcher` | no |  | allow-list (process launcher) |

Only **2 of 72** names run native today (`sample`, `data_sheet`). 23 should become `@native`; the rest are allow-listed (enums, exceptions, I/O, hooks, diagnostics). README advertises `from periodica import AtomCalculator, get_element` — neither is in `__all__` (ImportError).

## Surprises

- **The served numbers never touch `utils/`.** `get.py` has zero periodica imports; `sample`/`export`/`folding`/`optimize` import only `get`/`_dispatch` (and each other). `utils/` (66 modules, 35.6 k LOC) — every calculator, predictor, generator and transform — is off the serving path; apart from `PhysicsConstants` it is reached only by tests.
- **Rust parity is much weaker than the existence of `py_*` symbols suggests.** `py_kabsch_rmsd` returns 0.9–3.1 Å where the true RMSD (and Python) is 0.15–0.19 Å; `py_build_backbone` collapses every atom after residue 1; `py_optimize_alloy` returns `[]` for every input tried; STL/OBJ/HLSL exports differ from Python. None of these is called, so the test suite is green.
- **All six Rust predictors are stubs** that return `Err(not yet implemented)` and cite non-existent Python files.
- **`import periodica` executes 38 modules (279 ms incl. the extension)**, including `simulation_schema`, `pure_math`, `pure_array`, `backend_manager` and every data loader — 6.6 k LOC that the public API never uses — because `utils/__init__` and `data/__init__` re-export eagerly. Plan target is < 150 ms.
- **`quark_source` (experimental vs simulated) does not affect `Get`.** It only feeds `quark_loader`/`load_quark`; the registry never consults it.
- **SEMF is implemented 6 times in Python (+1 Rust stub) with 3 coefficient sets and 2 Coulomb forms**; r0 is 1.25 fm at 6 sites and 1.2 fm at 6 sites (4 files incl. the Rust stub); rule-of-mixtures exists in at least 7 copies (sample `mixture`, optimize, alloy.rs, rule_of_mixtures, alloy_calculator, biomaterial_predictor, material_predictor).
- **The app reaches only 13 library modules** (data_manager, quark_loader → quark_source, the layout_math package + 8 `quark_*` layouts, quark_enums). The other 26 layout_math modules (3.6 k LOC, README-advertised as "35 layout algorithms") have no consumer anywhere.
- `optimize_alloy` (Python) returns three identical `{'Fe': 1.0}` rows for `top_k=3` — no de-duplication, and alloying never wins because missing element data scores as 0 (D9).
- `export_sdf_raw` defaults differ: Python `mode='occupancy'`, Rust `mode='phase'` (A9 already flags that the 'SDF' is occupancy).

Artifacts (scratch `census/`): `graph.py` (import graph), `graph.json`, `cov.json`, `parity.py`/`parity.json` (Rust spot-check), `classify.py` (judgements), `triage.json`, this file.