# Periodica — Library

Scientific computation library published as `pip install periodica`.

## Architecture

- **src layout**: `src/periodica/` — all source lives here
- **12 scientific domains**: quarks, subatomic, elements, molecules, alloys, materials, amino acids, proteins, nucleic acids, cell components, cells, biomaterials
- **Data-driven**: 627 JSON data sheets in `src/periodica/data/active/`, not hardcoded lookup tables
- **Zero GUI dependencies**: No PySide6, no Kivy — pure Python + numpy

## Key Modules

| Module | Purpose |
|--------|---------|
| `periodica.core` | 68 enum classes across 12 domains |
| `periodica.data` | 5 specialized loaders + DataManager + 627 JSON data sheets |
| `periodica.utils` | Physics calculators, predictors, transforms, derivation chains |
| `periodica.layout_math` | 35 layout algorithms returning position dicts (no rendering) |
| `periodica.generators` | Top-level convenience functions: `generate_atom()`, `generate_molecule()`, etc. |

## Public API (65 symbols)

```python
from periodica import generate_atom, generate_molecule, generate_alloy
from periodica import AtomCalculator, PhysicsConstants
from periodica.data import get_element, get_element_by_z
from periodica.core import PTPropertyName, MoleculeLayoutMode
```

## Commands

```bash
# Install editable
pip install -e ".[test]"

# Run tests (987 pass)
pytest tests/ -v --tb=short -m "not slow and not gemini"

# Build for PyPI
python -m build
python -m twine check dist/*
```

## Related Repos

- **periodica-app**: GUI application (`pip install periodica-app`) — imports this library
- **Periodics**: Original monorepo (sunset)

## Conventions

- All internal imports use `from periodica.` prefix
- Data access uses `importlib.resources` for installed-package compatibility
- Physics constants in `PhysicsConstants` class, not module-level globals
- Semi-empirical models in `utils/predictors/` follow a base → chain → registry pattern
- Generator functions in `generators.py` are thin wrappers around calculator/generator classes
