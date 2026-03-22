"""
Top-level generator functions for creating scientific objects.

Provides simple entry points for generating atoms, molecules, alloys,
materials, and biological structures from their constituent components.
"""

import re
import math
from typing import Dict, List, Optional

from periodica.utils.physics_calculator import (
    AtomCalculator,
    SubatomicCalculator,
    MoleculeCalculator,
)
from periodica.data.element_loader import ElementDataLoader, get_loader


# ── Quark name-to-symbol mapping ─────────────────────────────────────────

_QUARK_NAME_TO_SYMBOL = {
    "up": "u",
    "down": "d",
    "strange": "s",
    "charm": "c",
    "bottom": "b",
    "top": "t",
    "anti-up": "u\u0305",
    "anti-down": "d\u0305",
    "anti-strange": "s\u0305",
    "anti-charm": "c\u0305",
    "anti-bottom": "b\u0305",
    "anti-top": "t\u0305",
}


# ── Formula parser ────────────────────────────────────────────────────────

def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula like 'H2O' or 'Ca(OH)2' into {symbol: count}.

    Handles simple formulas with optional parenthesised groups and
    subscript digits (both ASCII and Unicode subscripts).
    """
    # Normalise Unicode subscripts to ASCII digits
    _sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    formula = formula.translate(_sub_map)

    # Tokenise: uppercase letter optionally followed by lowercase, then digits
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    result: Dict[str, int] = {}
    for symbol, count_str in tokens:
        if not symbol:
            continue
        count = int(count_str) if count_str else 1
        result[symbol] = result.get(symbol, 0) + count
    return result


# ═════════════════════════════════════════════════════════════════════════
#  Single-object generators
# ═════════════════════════════════════════════════════════════════════════

def generate_atom(symbol: str = None, *, z: int = None, a: int = None) -> dict:
    """Generate an atom with calculated properties.

    Provide either a symbol (``"Fe"``) or atomic number *z*.  Mass number *a*
    is optional (uses most stable isotope if omitted).

    Args:
        symbol: Element symbol (e.g. ``"Fe"``, ``"He"``, ``"U"``)
        z: Atomic number (alternative to *symbol*)
        a: Mass number.  If ``None``, uses most common isotope.

    Returns:
        Dict with keys: name, symbol, z, a, binding_energy_MeV,
        electron_config, ionization_energy_eV, atomic_radius_pm, etc.

    Example::

        >>> atom = generate_atom("Fe")
        >>> atom = generate_atom(z=26, a=56)
    """
    if symbol is None and z is None:
        raise ValueError("Provide either symbol or z (atomic number)")

    loader = get_loader()

    if symbol is not None:
        elem = loader.get_element_by_symbol(symbol)
        if elem is None:
            raise ValueError(f"Unknown element symbol: {symbol!r}")
    else:
        elem = loader.get_element_by_z(z)
        if elem is None:
            raise ValueError(f"Unknown atomic number: {z}")
        symbol = elem["symbol"]

    Z = elem["atomic_number"]

    # Determine mass number
    if a is not None:
        A = a
    else:
        # Use the most abundant isotope if available
        isotopes = elem.get("isotopes", [])
        if isotopes:
            best = max(isotopes, key=lambda iso: iso.get("abundance", 0))
            A = best.get("mass_number", Z + max(0, Z - 1))
        else:
            # Fallback: round atomic mass
            A = round(elem.get("atomic_mass", Z * 2))

    N = A - Z

    # Calculate properties via AtomCalculator
    calculated = AtomCalculator.create_atom_from_particles(
        protons=Z, neutrons=N, electrons=Z,
        name=elem.get("name", f"Element-{Z}"),
        symbol=symbol,
    )

    # Merge data-sheet values (from JSON) with calculated values.
    # Data-sheet values take priority for fields that exist and are non-None.
    merged = dict(calculated)
    for key, value in elem.items():
        if value is not None:
            merged[key] = value

    # Ensure the calculated fields that may be more specific are available
    # under user-friendly aliases
    merged.setdefault("z", Z)
    merged.setdefault("a", A)
    merged.setdefault("binding_energy_MeV",
                       _binding_energy(Z, N))
    merged.setdefault("electron_config",
                       merged.get("electron_configuration", ""))

    return merged


def _binding_energy(Z: int, N: int) -> float:
    """Semi-empirical mass formula binding energy in MeV."""
    A = Z + N
    if A == 0:
        return 0.0
    a_v, a_s, a_c, a_a, a_p = 15.75, 17.8, 0.711, 23.7, 11.2
    B = a_v * A - a_s * (A ** (2 / 3))
    if A > 0:
        B -= a_c * (Z ** 2) / (A ** (1 / 3))
    B -= a_a * ((N - Z) ** 2) / A
    if Z % 2 == 0 and N % 2 == 0:
        B += a_p / (A ** 0.5)
    elif Z % 2 == 1 and N % 2 == 1:
        B -= a_p / (A ** 0.5)
    magic = {2, 8, 20, 28, 50, 82, 126}
    if Z in magic:
        B += 2.5
    if N in magic:
        B += 2.5
    return round(B, 3)


# ─────────────────────────────────────────────────────────────────────────

def generate_particle(quarks: list, name: str = None) -> dict:
    """Generate a subatomic particle from quark composition.

    Args:
        quarks: List of quark names, e.g. ``["up", "up", "down"]`` for proton.
                Accepts full names (``"up"``, ``"anti-down"``) or one-letter
                symbols (``"u"``, ``"d\u0305"``).
        name: Optional particle name.

    Returns:
        Dict with keys: Name, Symbol, Mass_MeVc2, Charge_e, Spin_hbar,
        Composition, etc.

    Example::

        >>> proton = generate_particle(["up", "up", "down"], name="Proton")
    """
    if not quarks:
        raise ValueError("quarks list must not be empty")

    # Convert human-readable names to symbols
    symbols = []
    for q in quarks:
        q_lower = q.lower().strip()
        if q_lower in _QUARK_NAME_TO_SYMBOL:
            symbols.append(_QUARK_NAME_TO_SYMBOL[q_lower])
        elif q in SubatomicCalculator.QUARK_PROPERTIES:
            symbols.append(q)
        else:
            raise ValueError(
                f"Unknown quark: {q!r}.  Use names like 'up', 'down', "
                f"'strange' or symbols like 'u', 'd', 's'."
            )

    return SubatomicCalculator.create_particle_from_quarks(
        quarks=symbols, name=name,
    )


# ─────────────────────────────────────────────────────────────────────────

def generate_molecule(
    formula: str = None,
    *,
    atoms: dict = None,
    name: str = None,
) -> dict:
    """Generate a molecule from formula or atomic composition.

    Args:
        formula: Chemical formula, e.g. ``"H2O"``, ``"CH4"``
        atoms: Dict of ``{symbol: count}``, e.g. ``{"H": 2, "O": 1}``
        name: Optional molecule name

    Returns:
        Dict with keys: Name, Formula, MolecularMass_amu, Geometry,
        BondType, Polarity, etc.

    Example::

        >>> water = generate_molecule("H2O")
        >>> methane = generate_molecule(atoms={"C": 1, "H": 4}, name="Methane")
    """
    if formula is None and atoms is None:
        raise ValueError("Provide either formula or atoms dict")

    if atoms is None:
        atoms = _parse_formula(formula)

    # Build composition list expected by MoleculeCalculator
    composition = [
        {"Element": sym, "Count": count}
        for sym, count in atoms.items()
    ]

    return MoleculeCalculator.create_molecule_from_atoms(
        composition=composition,
        name=name,
    )


# ─────────────────────────────────────────────────────────────────────────

def generate_alloy(
    components: dict,
    name: str = None,
    category: str = None,
) -> dict:
    """Generate an alloy from element weight fractions.

    Args:
        components: Dict of ``{symbol: weight_percent}``, e.g.
                    ``{"Fe": 98.5, "C": 0.8, "Mn": 0.7}``
        name: Optional alloy name
        category: Optional category (e.g. ``"Steel"``, ``"Aluminum"``,
                  ``"Copper"``)

    Returns:
        Dict with keys: Name, Category, Components, density,
        melting_point, tensile_strength, yield_strength, etc.

    Example::

        >>> steel = generate_alloy({"Fe": 98.5, "C": 0.8, "Mn": 0.7},
        ...                        name="1080 Steel")
    """
    from periodica.utils.alloy_calculator import AlloyCalculator

    if not components:
        raise ValueError("components must not be empty")

    # Build component_data and weight_fractions lists
    loader = get_loader()

    symbols = list(components.keys())
    percents = list(components.values())
    total_pct = sum(percents)
    if total_pct <= 0:
        raise ValueError("Weight percentages must sum to a positive value")
    fractions = [p / total_pct for p in percents]

    # Look up element data for each component (or build a minimal stub)
    component_data = []
    for sym in symbols:
        elem = loader.get_element_by_symbol(sym)
        if elem is not None:
            component_data.append(elem)
        else:
            component_data.append({"symbol": sym})

    # Determine lattice type from primary element
    primary = max(zip(symbols, fractions), key=lambda x: x[1])[0]
    _lattice_map = {
        'Fe': 'BCC', 'Al': 'FCC', 'Cu': 'FCC', 'Ni': 'FCC', 'Ti': 'HCP',
        'Zn': 'HCP', 'Mg': 'HCP', 'Co': 'HCP', 'W': 'BCC', 'Cr': 'BCC',
        'V': 'BCC', 'Nb': 'BCC', 'Mo': 'BCC', 'Ag': 'FCC', 'Au': 'FCC',
        'Pb': 'FCC', 'Pt': 'FCC',
    }
    lattice_type = _lattice_map.get(primary, 'FCC')

    alloy = AlloyCalculator.create_alloy_from_components(
        component_data=component_data,
        weight_fractions=fractions,
        lattice_type=lattice_type,
        name=name,
    )

    # Override category if caller specified one
    if category is not None:
        alloy['Category'] = category
        alloy['category'] = category

    return alloy


# ─────────────────────────────────────────────────────────────────────────

def generate_material(
    alloy_data: dict,
    processing: str = "Hot rolled",
) -> dict:
    """Generate a material with microstructure from alloy data.

    Args:
        alloy_data: Alloy dict (from :func:`generate_alloy` or data loader)
        processing: Processing type (e.g. ``"Hot rolled"``, ``"Cold drawn"``,
                    ``"Annealed"``)

    Returns:
        Dict with microstructure: grains, defects, phases,
        mechanical properties.

    Example::

        >>> alloy = generate_alloy({"Fe": 99, "C": 1})
        >>> material = generate_material(alloy, processing="Annealed")
    """
    from periodica.utils.material_generator import MaterialGenerator

    gen = MaterialGenerator()
    return gen.generate_from_alloy(alloy_data, processing=processing)


# ─────────────────────────────────────────────────────────────────────────

def generate_protein(
    sequence: str,
    name: str = "Custom Protein",
) -> dict:
    """Generate a protein from amino acid sequence.

    Args:
        sequence: One-letter amino acid sequence
                  (e.g. ``"MKFLILLFNILCLFPVLAADNH..."``)
        name: Protein name

    Returns:
        Dict with keys: name, sequence, molecular_mass, isoelectric_point,
        secondary_structure, amino_acid_composition, etc.

    Example::

        >>> insulin = generate_protein(
        ...     "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEAL",
        ...     name="Insulin B-chain",
        ... )
    """
    from periodica.utils.predictors.biological.protein_predictor import (
        ProteinPredictor,
    )

    predictor = ProteinPredictor()
    return predictor.create_protein_json(
        sequence=sequence,
        name=name,
    )


# ─────────────────────────────────────────────────────────────────────────

def generate_nucleic_acid(
    sequence: str,
    name: str = "Custom",
    is_rna: bool = False,
) -> dict:
    """Generate a nucleic acid from base sequence.

    Args:
        sequence: Nucleotide sequence (e.g. ``"ATCGATCG"`` for DNA,
                  ``"AUCGAUCG"`` for RNA)
        name: Name for the nucleic acid
        is_rna: ``True`` for RNA, ``False`` for DNA

    Returns:
        Dict with keys: name, sequence, type, length, molecular_mass,
        gc_content, melting_temperature, etc.

    Example::

        >>> dna = generate_nucleic_acid("ATCGATCGATCG", name="Test DNA")
        >>> rna = generate_nucleic_acid("AUCGAUCG", name="mRNA fragment",
        ...                             is_rna=True)
    """
    from periodica.utils.predictors.biological.nucleic_acid_predictor import (
        NucleicAcidPredictor,
    )

    predictor = NucleicAcidPredictor()
    na_type = "RNA" if is_rna else "DNA"
    return predictor.create_nucleic_acid_json(
        sequence=sequence,
        name=name,
        na_type=na_type,
    )


# ─────────────────────────────────────────────────────────────────────────

def generate_cell(
    name: str,
    cell_type: str = "eukaryotic",
    components: list = None,
    diameter_um: float = 10.0,
) -> dict:
    """Generate a cell model.

    Args:
        name: Cell name (e.g. ``"Hepatocyte"``, ``"Neuron"``)
        cell_type: ``"eukaryotic"`` or ``"prokaryotic"``
        components: Optional list of cell component dicts
        diameter_um: Cell diameter in micrometers

    Returns:
        Dict with keys: name, type, diameter_um, volume_fL,
        surface_area_um2, metabolic_rate_fW, etc.
    """
    from periodica.utils.predictors.biological.cell_predictor import (
        CellPredictor,
    )

    predictor = CellPredictor()
    result = predictor.analyze_cell(
        diameter_um=diameter_um,
        name=name,
        cell_type=cell_type,
    )

    if components is not None:
        result["components"] = components

    return result


# ─────────────────────────────────────────────────────────────────────────

def generate_biomaterial(
    name: str,
    composition: dict = None,
    porosity: float = 0.0,
) -> dict:
    """Generate a biomaterial model.

    Args:
        name: Material name (e.g. ``"Cortical Bone"``, ``"Tendon"``)
        composition: Dict describing ECM composition, e.g.
                     ``{"collagen": 0.6, "hydroxyapatite": 0.3}``
        porosity: Porosity fraction (0.0 to 1.0)

    Returns:
        Dict with keys: name, type, ecm_composition, porosity,
        mechanical_properties, etc.
    """
    from periodica.utils.predictors.biological.biomaterial_predictor import (
        BiomaterialPredictor,
    )

    if composition is None:
        composition = {"collagen": 0.5}

    predictor = BiomaterialPredictor()
    return predictor.analyze_biomaterial(
        name=name,
        tissue_type="generic",
        ecm_composition=composition,
        cell_composition={},
        porosity=porosity,
    )


# ═════════════════════════════════════════════════════════════════════════
#  Batch generators
# ═════════════════════════════════════════════════════════════════════════

def generate_alloys(
    category: str = None,
    count: int = 10,
) -> list:
    """Generate a batch of alloys.

    Args:
        category: Filter by category (``"steel"``, ``"aluminum"``,
                  ``"copper"``, ``"binary"``, ``"ternary"``).
                  If ``None``, generates from all categories.
        count: Maximum number to generate.

    Returns:
        List of alloy dicts.
    """
    from periodica.utils.alloy_generator import AlloyGenerator

    gen = AlloyGenerator()

    if category is None:
        return gen.generate_all(count_limit=count)

    cat = category.lower()
    if cat == "steel":
        return gen.generate_steel_variants(count=count)
    elif cat in ("aluminum", "aluminium"):
        return gen.generate_aluminum_alloys(count=count)
    elif cat == "copper":
        return gen.generate_copper_alloys(count=count)
    elif cat == "binary":
        metals = gen.get_metallic_elements()
        return gen.generate_binary_alloys(metals, count=count)
    elif cat == "ternary":
        metals = gen.get_metallic_elements()
        return gen.generate_ternary_alloys(metals, count=count)
    else:
        # Treat as general generation and filter by Category field
        all_alloys = gen.generate_all(count_limit=count * 3)
        filtered = [
            a for a in all_alloys
            if a.get("Category", "").lower() == cat
        ]
        return filtered[:count]


def generate_molecules(count: int = 20, max_atoms: int = 8) -> list:
    """Generate a batch of molecules.

    Args:
        count: Maximum number to generate.
        max_atoms: Maximum atoms per molecule.

    Returns:
        List of molecule dicts.
    """
    from periodica.utils.molecule_generator import MoleculeGenerator

    gen = MoleculeGenerator()
    return gen.generate_all(count_limit=count, max_atoms=max_atoms)


def generate_amino_acids(count: int = 20) -> list:
    """Generate standard amino acid data.

    Args:
        count: Maximum number of amino acids to generate (up to 20).

    Returns:
        List of amino acid dicts (up to 20 standard amino acids).
    """
    from periodica.utils.biological_generator import BiologicalGenerator

    gen = BiologicalGenerator()
    return gen.generate_standard_amino_acids(count_limit=count)


def generate_proteins(count: int = 10) -> list:
    """Generate template protein data.

    Args:
        count: Maximum number of proteins to generate.

    Returns:
        List of protein dicts.
    """
    from periodica.utils.biological_generator import BiologicalGenerator

    gen = BiologicalGenerator()
    return gen.generate_proteins(count_limit=count)


def generate_nucleic_acids(count: int = 10) -> list:
    """Generate template nucleic acid data (DNA and RNA).

    Args:
        count: Maximum number to generate.

    Returns:
        List of nucleic acid dicts.
    """
    from periodica.utils.biological_generator import BiologicalGenerator

    gen = BiologicalGenerator()
    return gen.generate_nucleic_acids(count_limit=count)


def generate_cells(count: int = 10) -> list:
    """Generate cell type data.

    Args:
        count: Maximum number of cell types to generate.

    Returns:
        List of cell dicts with properties.
    """
    from periodica.utils.biological_generator import BiologicalGenerator

    gen = BiologicalGenerator()
    return gen.generate_cells(count_limit=count)


def generate_biomaterials(count: int = 10) -> list:
    """Generate biomaterial data.

    Args:
        count: Maximum number of biomaterials to generate.

    Returns:
        List of biomaterial dicts.
    """
    from periodica.utils.biological_generator import BiologicalGenerator

    gen = BiologicalGenerator()
    return gen.generate_biomaterials(count_limit=count)


def generate_materials(alloys: list = None, count: int = 10) -> list:
    """Generate materials with microstructure from alloys.

    Args:
        alloys: List of alloy dicts.  If ``None``, generates alloys first.
        count: Maximum number of materials to produce.

    Returns:
        List of material dicts with microstructure data.
    """
    from periodica.utils.material_generator import MaterialGenerator

    gen = MaterialGenerator()
    if alloys is None:
        alloys = generate_alloys(count=count)
    return gen.generate_all(alloys, count_limit=count)


# ═════════════════════════════════════════════════════════════════════════
#  Documentation generator
# ═════════════════════════════════════════════════════════════════════════

def generate_documentation(output_dir: str = None) -> str:
    """Generate comprehensive API documentation.

    Produces a markdown document listing all public classes, functions,
    and constants with their signatures and descriptions.

    Args:
        output_dir: Directory to write docs to.  If ``None``, returns as
                    string.

    Returns:
        Markdown string of the API reference.

    Example::

        >>> docs = generate_documentation()
        >>> print(docs[:200])
        # Periodica API Reference ...

        >>> generate_documentation("/path/to/docs/")  # writes API_REFERENCE.md
    """
    import importlib
    import inspect
    from pathlib import Path

    submodules = [
        "periodica",
        "periodica.core",
        "periodica.data",
        "periodica.utils",
        "periodica.layout_math",
    ]

    lines = [
        "# Periodica -- API Reference",
        "",
        "> Auto-generated by `generate_documentation()`.",
        "",
        "## Public API",
        "",
    ]

    for modname in submodules:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        all_names = getattr(mod, "__all__", None)
        if all_names is None:
            all_names = [n for n in dir(mod) if not n.startswith("_")]

        if not all_names:
            continue

        lines.append(f"### `{modname}`")
        lines.append("")
        lines.append("| Name | Kind | Description |")
        lines.append("|------|------|-------------|")

        for name in sorted(all_names):
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            if inspect.isclass(obj):
                kind = "class"
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                kind = "function"
            elif inspect.ismodule(obj):
                kind = "module"
            else:
                kind = "constant"

            doc = getattr(obj, "__doc__", None) or ""
            first_line = ""
            for dl in doc.strip().splitlines():
                stripped = dl.strip()
                if stripped:
                    first_line = stripped.replace("|", "\\|")
                    break

            lines.append(f"| `{name}` | {kind} | {first_line} |")

        lines.append("")

    # Quick Reference
    lines.append("## Quick Reference")
    lines.append("")
    lines.append("```python")
    lines.append("from periodica import (")
    lines.append("    generate_atom, generate_particle, generate_molecule,")
    lines.append("    generate_alloy, generate_material,")
    lines.append("    generate_protein, generate_nucleic_acid,")
    lines.append("    generate_cell, generate_biomaterial,")
    lines.append(")")
    lines.append("```")
    lines.append("")

    doc_text = "\n".join(lines)

    if output_dir is not None:
        out_path = Path(output_dir) / "API_REFERENCE.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc_text, encoding="utf-8")

    return doc_text


__all__ = [
    # Single-object generators
    "generate_atom",
    "generate_particle",
    "generate_molecule",
    "generate_alloy",
    "generate_material",
    "generate_protein",
    "generate_nucleic_acid",
    "generate_cell",
    "generate_biomaterial",
    # Batch generators
    "generate_alloys",
    "generate_molecules",
    "generate_amino_acids",
    "generate_proteins",
    "generate_nucleic_acids",
    "generate_cells",
    "generate_biomaterials",
    "generate_materials",
    # Documentation
    "generate_documentation",
]
