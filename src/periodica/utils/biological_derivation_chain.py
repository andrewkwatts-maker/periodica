"""
Biological Derivation Chain
Enables dynamic creation of biological structures from atomic/molecular components.

Derivation Hierarchy:
    Atoms → Molecules → Amino Acids/Nucleotides → Proteins/Nucleic Acids
         → Cell Components → Cells → Tissues → Biomaterials

Each level can be:
1. Loaded from JSON data files
2. Created dynamically from lower-level components
3. Mixed to create complex structures
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)


class BiologicalLevel(Enum):
    """Hierarchy levels in biological organization."""
    ATOM = 0
    MOLECULE = 1
    AMINO_ACID = 2
    NUCLEOTIDE = 3
    PROTEIN = 4
    NUCLEIC_ACID = 5
    CELL_COMPONENT = 6
    CELL = 7
    TISSUE = 8
    BIOMATERIAL = 9


@dataclass
class AtomicComposition:
    """Represents atomic composition of a molecule."""
    elements: Dict[str, int] = field(default_factory=dict)  # {element_symbol: count}

    def get_molecular_formula(self) -> str:
        """Generate molecular formula string."""
        formula = ""
        # Standard order: C, H, N, O, P, S, then alphabetical
        order = ['C', 'H', 'N', 'O', 'P', 'S']
        for elem in order:
            if elem in self.elements:
                count = self.elements[elem]
                formula += elem + (str(count) if count > 1 else "")

        for elem in sorted(self.elements.keys()):
            if elem not in order:
                count = self.elements[elem]
                formula += elem + (str(count) if count > 1 else "")

        return formula

    def get_molecular_mass(self, atomic_masses: Dict[str, float]) -> float:
        """Calculate molecular mass from atomic masses."""
        mass = 0.0
        for elem, count in self.elements.items():
            mass += atomic_masses.get(elem, 0) * count
        return mass

    def __add__(self, other: 'AtomicComposition') -> 'AtomicComposition':
        """Combine two compositions."""
        combined = self.elements.copy()
        for elem, count in other.elements.items():
            combined[elem] = combined.get(elem, 0) + count
        return AtomicComposition(combined)

    def __sub__(self, other: 'AtomicComposition') -> 'AtomicComposition':
        """Subtract composition (e.g., for water loss in condensation)."""
        result = self.elements.copy()
        for elem, count in other.elements.items():
            result[elem] = result.get(elem, 0) - count
            if result[elem] <= 0:
                del result[elem]
        return AtomicComposition(result)


@dataclass
class BiologicalComponent:
    """Base class for all biological components."""
    name: str
    level: BiologicalLevel
    composition: AtomicComposition = field(default_factory=AtomicComposition)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_components: List['BiologicalComponent'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Make component hashable by name and level."""
        return hash((self.name, self.level))

    def __eq__(self, other):
        """Compare components by name and level."""
        if not isinstance(other, BiologicalComponent):
            return False
        return self.name == other.name and self.level == other.level

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'level': self.level.name,
            'composition': self.composition.elements,
            'molecular_formula': self.composition.get_molecular_formula(),
            'properties': self.properties,
            'source_components': [c.name for c in self.source_components],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict, level: BiologicalLevel) -> 'BiologicalComponent':
        """Deserialize from dictionary."""
        return cls(
            name=data.get('name', 'Unknown'),
            level=level,
            composition=AtomicComposition(data.get('composition', {})),
            properties=data.get('properties', {}),
            metadata=data.get('metadata', {})
        )


class BiologicalDerivationChain:
    """
    Manages the derivation of biological structures from atomic components.
    Provides methods to build higher-level structures from lower-level ones.
    """

    # Standard atomic masses (Da)
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'P': 30.974, 'S': 32.065, 'Fe': 55.845, 'Mg': 24.305,
        'Ca': 40.078, 'K': 39.098, 'Na': 22.990, 'Cl': 35.453,
        'Zn': 65.38, 'Cu': 63.546, 'Mn': 54.938, 'Se': 78.971,
        'I': 126.904, 'Co': 58.933,
    }

    # Water composition for condensation reactions
    WATER = AtomicComposition({'H': 2, 'O': 1})

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the derivation chain.

        Args:
            data_path: Root path for biological data files
        """
        self._data_path = data_path or Path(__file__).parent.parent / "data" / "active"
        self._cache: Dict[str, BiologicalComponent] = {}
        self._custom_components: Dict[str, BiologicalComponent] = {}

    # === Amino Acid Building ===

    def get_amino_acid_composition(self, symbol: str) -> AtomicComposition:
        """Get atomic composition of an amino acid from data files."""
        aa_path = self._data_path / "amino_acids" / f"{symbol.lower()}.json"
        if aa_path.exists():
            with open(aa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'composition' in data:
                    return AtomicComposition(data['composition'])
                elif 'molecular_formula' in data:
                    return self._parse_formula(data['molecular_formula'])

        # Fallback to standard amino acid compositions
        return self._get_standard_aa_composition(symbol)

    def _get_standard_aa_composition(self, symbol: str) -> AtomicComposition:
        """Get standard amino acid atomic composition."""
        # Standard amino acid compositions
        AA_COMPOSITIONS = {
            'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2},   # Alanine
            'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2},  # Arginine
            'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3},   # Asparagine
            'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4},   # Aspartic acid
            'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},  # Cysteine
            'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4},   # Glutamic acid
            'Q': {'C': 5, 'H': 10, 'N': 2, 'O': 3},  # Glutamine
            'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2},   # Glycine
            'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2},   # Histidine
            'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2},  # Isoleucine
            'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2},  # Leucine
            'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2},  # Lysine
            'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},  # Methionine
            'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2},  # Phenylalanine
            'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2},   # Proline
            'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3},   # Serine
            'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3},   # Threonine
            'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2}, # Tryptophan
            'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3},  # Tyrosine
            'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2},  # Valine
        }
        return AtomicComposition(AA_COMPOSITIONS.get(symbol.upper(), {}))

    def _parse_formula(self, formula: str) -> AtomicComposition:
        """Parse a molecular formula string into AtomicComposition."""
        import re
        elements = {}
        # Match element symbols followed by optional numbers
        pattern = r'([A-Z][a-z]?)(\d*)'
        for match in re.finditer(pattern, formula):
            elem = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            elements[elem] = elements.get(elem, 0) + count
        return AtomicComposition(elements)

    # === Protein Building ===

    def build_protein_from_sequence(self, sequence: str, name: str = "Custom Protein") -> BiologicalComponent:
        """
        Build a protein from an amino acid sequence.

        Args:
            sequence: Amino acid sequence (one-letter codes)
            name: Protein name

        Returns:
            BiologicalComponent representing the protein
        """
        sequence = sequence.upper()
        total_composition = AtomicComposition({})

        # Sum amino acid compositions
        for aa in sequence:
            aa_comp = self.get_amino_acid_composition(aa)
            total_composition = total_composition + aa_comp

        # Subtract water for each peptide bond
        water_loss = AtomicComposition({
            'H': 2 * (len(sequence) - 1),
            'O': len(sequence) - 1
        })
        protein_composition = total_composition - water_loss

        # Calculate properties
        molecular_mass = protein_composition.get_molecular_mass(self.ATOMIC_MASSES)

        protein = BiologicalComponent(
            name=name,
            level=BiologicalLevel.PROTEIN,
            composition=protein_composition,
            properties={
                'sequence': sequence,
                'length': len(sequence),
                'molecular_mass': round(molecular_mass, 2),
                'molecular_formula': protein_composition.get_molecular_formula(),
            },
            metadata={
                'derived_from': 'amino_acid_sequence',
                'residue_count': len(sequence),
            }
        )

        return protein

    # === Nucleic Acid Building ===

    def get_nucleotide_composition(self, base: str, is_rna: bool = False) -> AtomicComposition:
        """Get atomic composition of a nucleotide."""
        # Nucleotide compositions (monophosphate form)
        DNA_NUCLEOTIDES = {
            'A': {'C': 10, 'H': 14, 'N': 5, 'O': 6, 'P': 1},  # dAMP
            'T': {'C': 10, 'H': 15, 'N': 2, 'O': 8, 'P': 1},  # dTMP
            'G': {'C': 10, 'H': 14, 'N': 5, 'O': 7, 'P': 1},  # dGMP
            'C': {'C': 9, 'H': 14, 'N': 3, 'O': 7, 'P': 1},   # dCMP
        }
        RNA_NUCLEOTIDES = {
            'A': {'C': 10, 'H': 14, 'N': 5, 'O': 7, 'P': 1},  # AMP
            'U': {'C': 9, 'H': 13, 'N': 2, 'O': 9, 'P': 1},   # UMP
            'G': {'C': 10, 'H': 14, 'N': 5, 'O': 8, 'P': 1},  # GMP
            'C': {'C': 9, 'H': 14, 'N': 3, 'O': 8, 'P': 1},   # CMP
        }

        nucleotides = RNA_NUCLEOTIDES if is_rna else DNA_NUCLEOTIDES
        return AtomicComposition(nucleotides.get(base.upper(), {}))

    def build_nucleic_acid_from_sequence(self, sequence: str, name: str = "Custom NA",
                                          is_rna: bool = False) -> BiologicalComponent:
        """
        Build a nucleic acid from a nucleotide sequence.

        Args:
            sequence: Nucleotide sequence
            name: Nucleic acid name
            is_rna: Whether this is RNA (default: DNA)

        Returns:
            BiologicalComponent representing the nucleic acid
        """
        sequence = sequence.upper()
        if is_rna:
            sequence = sequence.replace('T', 'U')
        else:
            sequence = sequence.replace('U', 'T')

        total_composition = AtomicComposition({})

        # Sum nucleotide compositions
        for base in sequence:
            nt_comp = self.get_nucleotide_composition(base, is_rna)
            total_composition = total_composition + nt_comp

        # Subtract water for phosphodiester bonds
        water_loss = AtomicComposition({
            'H': 2 * (len(sequence) - 1),
            'O': len(sequence) - 1
        })
        na_composition = total_composition - water_loss

        molecular_mass = na_composition.get_molecular_mass(self.ATOMIC_MASSES)

        # Calculate GC content
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = 100 * gc_count / len(sequence) if sequence else 0

        na = BiologicalComponent(
            name=name,
            level=BiologicalLevel.NUCLEIC_ACID,
            composition=na_composition,
            properties={
                'sequence': sequence,
                'length': len(sequence),
                'molecular_mass': round(molecular_mass, 2),
                'molecular_formula': na_composition.get_molecular_formula(),
                'type': 'RNA' if is_rna else 'DNA',
                'gc_content': round(gc_content, 1),
            },
            metadata={
                'derived_from': 'nucleotide_sequence',
                'is_rna': is_rna,
            }
        )

        return na

    # === Cell Component Building ===

    def build_cell_component(self, name: str, proteins: List[BiologicalComponent],
                              copy_number: int = 1,
                              component_type: str = "complex") -> BiologicalComponent:
        """
        Build a cell component from proteins.

        Args:
            name: Component name
            proteins: List of protein components
            copy_number: Number of copies in a cell
            component_type: Type of component (ribosome, proteasome, etc.)

        Returns:
            BiologicalComponent representing the cell component
        """
        total_composition = AtomicComposition({})
        total_mass = 0.0

        for protein in proteins:
            total_composition = total_composition + protein.composition
            total_mass += protein.properties.get('molecular_mass', 0)

        component = BiologicalComponent(
            name=name,
            level=BiologicalLevel.CELL_COMPONENT,
            composition=total_composition,
            properties={
                'molecular_mass': round(total_mass, 2),
                'molecular_formula': total_composition.get_molecular_formula(),
                'copy_number': copy_number,
                'component_type': component_type,
                'protein_count': len(proteins),
            },
            source_components=proteins,
            metadata={
                'derived_from': 'protein_assembly',
                'constituent_proteins': [p.name for p in proteins],
            }
        )

        return component

    # === Cell Building ===

    def build_cell(self, name: str, components: Dict[BiologicalComponent, int],
                   diameter_um: float = 10.0,
                   cell_type: str = "generic") -> BiologicalComponent:
        """
        Build a cell from cell components.

        Args:
            name: Cell name
            components: Dict of {component: count}
            diameter_um: Cell diameter in micrometers
            cell_type: Type of cell

        Returns:
            BiologicalComponent representing the cell
        """
        total_composition = AtomicComposition({})
        total_mass = 0.0

        component_list = []
        for component, count in components.items():
            for _ in range(count):
                total_composition = total_composition + component.composition
                total_mass += component.properties.get('molecular_mass', 0)
            component_list.append(component)

        # Calculate volume (spherical approximation)
        volume_fl = (4/3) * math.pi * (diameter_um/2) ** 3

        cell = BiologicalComponent(
            name=name,
            level=BiologicalLevel.CELL,
            composition=total_composition,
            properties={
                'diameter_um': diameter_um,
                'volume_fl': round(volume_fl, 2),
                'estimated_mass_pg': round(volume_fl * 1.05, 2),  # ~1.05 g/cm³
                'cell_type': cell_type,
                'component_count': sum(components.values()),
            },
            source_components=component_list,
            metadata={
                'derived_from': 'component_assembly',
                'component_counts': {c.name: n for c, n in components.items()},
            }
        )

        return cell

    # === Biomaterial Building ===

    def build_biomaterial(self, name: str,
                          ecm_composition: Dict[str, float],
                          cell_composition: Optional[Dict[BiologicalComponent, float]] = None,
                          porosity: float = 0.0) -> BiologicalComponent:
        """
        Build a biomaterial from ECM components and cells.

        Args:
            name: Material name
            ecm_composition: Dict of {ecm_type: volume_fraction}
            cell_composition: Optional dict of {cell: volume_fraction}
            porosity: Porosity as fraction (0-1)

        Returns:
            BiologicalComponent representing the biomaterial
        """
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor

        predictor = BiomaterialPredictor()

        # Calculate mechanical properties
        E_voigt = predictor.calculate_voigt_modulus(ecm_composition)
        E_reuss = predictor.calculate_reuss_modulus(ecm_composition)
        E_average = (E_voigt + E_reuss) / 2
        E_effective = predictor.calculate_porosity_effect(E_average, porosity)
        density = predictor.calculate_composite_density(ecm_composition)

        cell_fraction = sum(cell_composition.values()) if cell_composition else 0

        biomaterial = BiologicalComponent(
            name=name,
            level=BiologicalLevel.BIOMATERIAL,
            composition=AtomicComposition({}),  # Complex mixture
            properties={
                'ecm_composition': ecm_composition,
                'cell_fraction': cell_fraction,
                'porosity': porosity * 100,
                'youngs_modulus_MPa': round(E_effective, 4),
                'density_g_cm3': round(density, 3),
                'stiffness_category': predictor._categorize_stiffness(E_effective),
            },
            source_components=list(cell_composition.keys()) if cell_composition else [],
            metadata={
                'derived_from': 'ecm_cell_mixture',
                'model': 'Voigt-Reuss average with Gibson-Ashby porosity',
            }
        )

        return biomaterial

    # === Custom Component Registration ===

    def register_custom_component(self, component: BiologicalComponent) -> None:
        """Register a custom component for use in derivations."""
        key = f"{component.level.name}:{component.name}"
        self._custom_components[key] = component
        logger.info(f"Registered custom component: {key}")

    def get_component(self, level: BiologicalLevel, name: str) -> Optional[BiologicalComponent]:
        """Get a component by level and name."""
        key = f"{level.name}:{name}"
        return self._custom_components.get(key)

    # === JSON I/O ===

    def save_component(self, component: BiologicalComponent, path: Path) -> None:
        """Save a component to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(component.to_dict(), f, indent=2)
        logger.info(f"Saved component to {path}")

    def load_component(self, path: Path, level: BiologicalLevel) -> BiologicalComponent:
        """Load a component from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return BiologicalComponent.from_dict(data, level)


class BiologicalMixingSystem:
    """
    System for creating complex biological mixtures and composites.
    Combines components at different levels of the biological hierarchy.
    """

    def __init__(self, derivation_chain: BiologicalDerivationChain):
        self.chain = derivation_chain

    def create_tissue(self, name: str,
                      cells: Dict[BiologicalComponent, float],
                      ecm: Dict[str, float],
                      vascularization: float = 0.0) -> BiologicalComponent:
        """
        Create a tissue from cells and ECM.

        Args:
            name: Tissue name
            cells: Dict of {cell: fraction}
            ecm: Dict of {ecm_component: fraction}
            vascularization: Fraction of tissue that is vascular

        Returns:
            BiologicalComponent representing the tissue
        """
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor
        from periodica.utils.predictors.biological.cell_predictor import CellPredictor

        bio_pred = BiomaterialPredictor()
        cell_pred = CellPredictor()

        # Calculate tissue properties
        total_cell_fraction = sum(cells.values())
        ecm_fraction = 1 - total_cell_fraction - vascularization

        # Normalize ECM composition
        ecm_normalized = {k: v * ecm_fraction for k, v in ecm.items()}

        # Calculate mechanical properties
        E_ecm = bio_pred.calculate_composite_modulus(ecm)
        E_cells = 0.001  # Cells are soft (~1 kPa)
        E_tissue = E_ecm * ecm_fraction + E_cells * total_cell_fraction

        # Calculate metabolic rate
        total_metabolic_rate = 0
        for cell, fraction in cells.items():
            mass_pg = cell.properties.get('estimated_mass_pg', 100)
            rate = cell_pred.calculate_metabolic_rate(mass_pg)
            # Scale by fraction and typical cell count per mm³
            total_metabolic_rate += rate * fraction * 1e6

        tissue = BiologicalComponent(
            name=name,
            level=BiologicalLevel.TISSUE,
            composition=AtomicComposition({}),
            properties={
                'cell_types': {c.name: f for c, f in cells.items()},
                'ecm_composition': ecm,
                'total_cell_fraction': round(total_cell_fraction, 3),
                'vascularization': round(vascularization, 3),
                'estimated_modulus_MPa': round(E_tissue, 4),
                'metabolic_rate_fW_per_mm3': round(total_metabolic_rate, 2),
            },
            source_components=list(cells.keys()),
            metadata={
                'derived_from': 'cell_ecm_mixture',
                'tissue_type': name,
            }
        )

        return tissue

    def create_organ_model(self, name: str,
                           tissues: Dict[BiologicalComponent, float],
                           organ_type: str = "generic") -> BiologicalComponent:
        """
        Create a simplified organ model from tissues.

        Args:
            name: Organ name
            tissues: Dict of {tissue: volume_fraction}
            organ_type: Type of organ

        Returns:
            BiologicalComponent representing the organ model
        """
        # Aggregate properties
        avg_modulus = 0
        avg_metabolic_rate = 0

        for tissue, fraction in tissues.items():
            avg_modulus += tissue.properties.get('estimated_modulus_MPa', 0) * fraction
            avg_metabolic_rate += tissue.properties.get('metabolic_rate_fW_per_mm3', 0) * fraction

        organ = BiologicalComponent(
            name=name,
            level=BiologicalLevel.BIOMATERIAL,  # Organ as biomaterial for now
            composition=AtomicComposition({}),
            properties={
                'organ_type': organ_type,
                'tissue_composition': {t.name: f for t, f in tissues.items()},
                'estimated_modulus_MPa': round(avg_modulus, 4),
                'metabolic_rate_fW_per_mm3': round(avg_metabolic_rate, 2),
            },
            source_components=list(tissues.keys()),
            metadata={
                'derived_from': 'tissue_assembly',
                'model_type': 'simplified_organ',
            }
        )

        return organ
