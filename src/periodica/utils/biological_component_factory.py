"""
Biological Component Factory
Unified factory for creating, loading, and managing biological components
at all levels of the hierarchy.

Provides:
1. Dynamic component creation from lower-level components
2. JSON-based data loading and persistence
3. Component validation and property calculation
4. Complex material mixing capabilities
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union

from periodica.utils.biological_derivation_chain import (
    BiologicalDerivationChain,
    BiologicalMixingSystem,
    BiologicalComponent,
    BiologicalLevel,
    AtomicComposition
)

logger = logging.getLogger(__name__)


@dataclass
class ComponentTemplate:
    """Template for creating biological components."""
    name: str
    level: BiologicalLevel
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)
    validators: Dict[str, Callable] = field(default_factory=dict)


class BiologicalComponentFactory:
    """
    Factory for creating and managing biological components.
    Supports dynamic creation from atoms to complex tissues.
    """

    # Component templates for validation
    TEMPLATES = {
        BiologicalLevel.AMINO_ACID: ComponentTemplate(
            name="Amino Acid",
            level=BiologicalLevel.AMINO_ACID,
            required_fields=['symbol', 'name'],
            optional_fields=['molecular_mass', 'pKa_carboxyl', 'pKa_amino', 'pKa_sidechain',
                           'hydropathy_index', 'helix_propensity', 'sheet_propensity'],
            default_values={'category': 'standard'}
        ),
        BiologicalLevel.PROTEIN: ComponentTemplate(
            name="Protein",
            level=BiologicalLevel.PROTEIN,
            required_fields=['name', 'sequence'],
            optional_fields=['organism', 'function', 'localization', 'cofactors',
                           'post_translational_modifications', 'interactions'],
            default_values={'organism': 'Unknown', 'function': 'Unknown'}
        ),
        BiologicalLevel.NUCLEIC_ACID: ComponentTemplate(
            name="Nucleic Acid",
            level=BiologicalLevel.NUCLEIC_ACID,
            required_fields=['name', 'sequence', 'type'],
            optional_fields=['organism', 'function', 'secondary_structure'],
            default_values={'type': 'DNA', 'organism': 'Unknown'}
        ),
        BiologicalLevel.CELL_COMPONENT: ComponentTemplate(
            name="Cell Component",
            level=BiologicalLevel.CELL_COMPONENT,
            required_fields=['name', 'type'],
            optional_fields=['compartment', 'subunits', 'copy_number', 'activity'],
            default_values={'compartment': 'cytoplasm', 'copy_number': 1}
        ),
        BiologicalLevel.CELL: ComponentTemplate(
            name="Cell",
            level=BiologicalLevel.CELL,
            required_fields=['name', 'type'],
            optional_fields=['organism', 'tissue_origin', 'diameter_um', 'volume_fl',
                           'components', 'metabolic_properties'],
            default_values={'diameter_um': 10.0, 'organism': 'Homo sapiens'}
        ),
        BiologicalLevel.TISSUE: ComponentTemplate(
            name="Tissue",
            level=BiologicalLevel.TISSUE,
            required_fields=['name', 'type'],
            optional_fields=['cell_composition', 'ecm_composition', 'mechanical_properties',
                           'physical_properties', 'functions'],
            default_values={'type': 'connective_tissue'}
        ),
        BiologicalLevel.BIOMATERIAL: ComponentTemplate(
            name="Biomaterial",
            level=BiologicalLevel.BIOMATERIAL,
            required_fields=['name'],
            optional_fields=['ecm_composition', 'cell_composition', 'porosity',
                           'mechanical_properties', 'physical_properties'],
            default_values={'porosity': 0.0}
        ),
    }

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the component factory.

        Args:
            data_path: Root path for biological data files
        """
        self._data_path = data_path or Path(__file__).parent.parent / "data" / "active"
        self._chain = BiologicalDerivationChain(self._data_path)
        self._mixer = BiologicalMixingSystem(self._chain)
        self._component_cache: Dict[str, BiologicalComponent] = {}

        # Property calculators for each level
        self._calculators: Dict[BiologicalLevel, Callable] = {
            BiologicalLevel.PROTEIN: self._calculate_protein_properties,
            BiologicalLevel.NUCLEIC_ACID: self._calculate_nucleic_acid_properties,
            BiologicalLevel.CELL: self._calculate_cell_properties,
            BiologicalLevel.BIOMATERIAL: self._calculate_biomaterial_properties,
        }

    # === Component Creation ===

    def create_component(self, level: BiologicalLevel, data: Dict[str, Any],
                         calculate_properties: bool = True) -> BiologicalComponent:
        """
        Create a biological component from data dictionary.

        Args:
            level: Biological hierarchy level
            data: Component data dictionary
            calculate_properties: Whether to calculate derived properties

        Returns:
            BiologicalComponent instance
        """
        template = self.TEMPLATES.get(level)
        if template:
            # Apply defaults
            for key, value in template.default_values.items():
                if key not in data:
                    data[key] = value

            # Validate required fields
            missing = [f for f in template.required_fields if f not in data]
            if missing:
                raise ValueError(f"Missing required fields for {level.name}: {missing}")

        # Create base component
        component = BiologicalComponent(
            name=data.get('name', 'Unknown'),
            level=level,
            composition=AtomicComposition(data.get('composition', {})),
            properties=data.get('properties', {}),
            metadata=data.get('metadata', {})
        )

        # Copy additional data to properties
        for key, value in data.items():
            if key not in ['name', 'level', 'composition', 'properties', 'metadata']:
                component.properties[key] = value

        # Calculate derived properties
        if calculate_properties and level in self._calculators:
            self._calculators[level](component)

        return component

    def create_from_lower_level(self, level: BiologicalLevel,
                                 components: List[BiologicalComponent],
                                 name: str,
                                 **kwargs) -> BiologicalComponent:
        """
        Create a higher-level component from lower-level components.

        Args:
            level: Target biological level
            components: Source components
            name: Name for new component
            **kwargs: Additional properties

        Returns:
            New BiologicalComponent at specified level
        """
        if level == BiologicalLevel.PROTEIN:
            # Build protein from amino acid sequence
            sequence = kwargs.get('sequence', '')
            return self._chain.build_protein_from_sequence(sequence, name)

        elif level == BiologicalLevel.NUCLEIC_ACID:
            sequence = kwargs.get('sequence', '')
            is_rna = kwargs.get('is_rna', False)
            return self._chain.build_nucleic_acid_from_sequence(sequence, name, is_rna)

        elif level == BiologicalLevel.CELL_COMPONENT:
            proteins = [c for c in components if c.level == BiologicalLevel.PROTEIN]
            copy_number = kwargs.get('copy_number', 1)
            component_type = kwargs.get('component_type', 'complex')
            return self._chain.build_cell_component(name, proteins, copy_number, component_type)

        elif level == BiologicalLevel.CELL:
            component_counts = kwargs.get('component_counts', {})
            diameter_um = kwargs.get('diameter_um', 10.0)
            cell_type = kwargs.get('cell_type', 'generic')
            return self._chain.build_cell(name, component_counts, diameter_um, cell_type)

        elif level == BiologicalLevel.TISSUE:
            cells = kwargs.get('cells', {})
            ecm = kwargs.get('ecm', {})
            vascularization = kwargs.get('vascularization', 0.0)
            return self._mixer.create_tissue(name, cells, ecm, vascularization)

        elif level == BiologicalLevel.BIOMATERIAL:
            ecm = kwargs.get('ecm_composition', {})
            cells = kwargs.get('cell_composition', None)
            porosity = kwargs.get('porosity', 0.0)
            return self._chain.build_biomaterial(name, ecm, cells, porosity)

        else:
            raise ValueError(f"Cannot create {level.name} from lower-level components")

    # === Property Calculators ===

    def _calculate_protein_properties(self, component: BiologicalComponent) -> None:
        """Calculate derived properties for proteins."""
        from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor

        sequence = component.properties.get('sequence', '')
        if not sequence:
            return

        predictor = ProteinPredictor()
        analysis = predictor.analyze_protein(sequence, component.name)

        component.properties.update({
            'molecular_mass': analysis['molecular_mass'],
            'isoelectric_point': analysis['isoelectric_point'],
            'charge_pH7': analysis['charge_pH7'],
            'gravy': analysis['gravy'],
            'instability_index': analysis['instability_index'],
            'is_stable': analysis['is_stable'],
            'aliphatic_index': analysis['aliphatic_index'],
            'secondary_structure': analysis['secondary_structure'],
            'length': len(sequence),
        })

    def _calculate_nucleic_acid_properties(self, component: BiologicalComponent) -> None:
        """Calculate derived properties for nucleic acids."""
        from periodica.utils.predictors.biological.nucleic_acid_predictor import NucleicAcidPredictor

        sequence = component.properties.get('sequence', '')
        if not sequence:
            return

        is_rna = component.properties.get('type', 'DNA').upper() == 'RNA'
        predictor = NucleicAcidPredictor()
        analysis = predictor.analyze_sequence(sequence, component.name, is_rna)

        component.properties.update({
            'molecular_mass': analysis['molecular_mass'],
            'gc_content': analysis['gc_content'],
            'melting_temperature': analysis['melting_temperature']['nearest_neighbor'],
            'length': len(sequence),
            'complement': analysis['complement'],
        })

    def _calculate_cell_properties(self, component: BiologicalComponent) -> None:
        """Calculate derived properties for cells."""
        from periodica.utils.predictors.biological.cell_predictor import CellPredictor

        diameter_um = component.properties.get('diameter_um', 10.0)
        predictor = CellPredictor()

        volume = predictor.calculate_cell_volume(diameter_um)
        mass = predictor.calculate_cell_mass(volume)
        metabolic_rate = predictor.calculate_metabolic_rate(mass)

        component.properties.update({
            'volume_fl': round(volume, 2),
            'mass_pg': round(mass, 2),
            'metabolic_rate_fW': round(metabolic_rate, 2),
            'surface_area_um2': round(predictor.calculate_surface_area(diameter_um), 2),
            'surface_volume_ratio': round(predictor.calculate_surface_volume_ratio(diameter_um), 4),
        })

    def _calculate_biomaterial_properties(self, component: BiologicalComponent) -> None:
        """Calculate derived properties for biomaterials."""
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor

        ecm = component.properties.get('ecm_composition', {})
        if not ecm:
            return

        porosity = component.properties.get('porosity', 0.0)
        predictor = BiomaterialPredictor()

        E_voigt = predictor.calculate_voigt_modulus(ecm)
        E_reuss = predictor.calculate_reuss_modulus(ecm)
        E_avg = (E_voigt + E_reuss) / 2
        E_eff = predictor.calculate_porosity_effect(E_avg, porosity)
        density = predictor.calculate_composite_density(ecm)

        component.properties.update({
            'youngs_modulus_MPa': round(E_eff, 4),
            'youngs_modulus_voigt': round(E_voigt, 4),
            'youngs_modulus_reuss': round(E_reuss, 4),
            'density_g_cm3': round(density, 3),
            'stiffness_category': predictor._categorize_stiffness(E_eff),
        })

    # === Data Loading ===

    def load_component(self, level: BiologicalLevel, name: str) -> Optional[BiologicalComponent]:
        """
        Load a component from JSON data files.

        Args:
            level: Biological hierarchy level
            name: Component name (used to find file)

        Returns:
            BiologicalComponent if found, None otherwise
        """
        # Check cache first
        cache_key = f"{level.name}:{name}"
        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        # Determine subdirectory
        level_dirs = {
            BiologicalLevel.AMINO_ACID: 'amino_acids',
            BiologicalLevel.PROTEIN: 'proteins',
            BiologicalLevel.NUCLEIC_ACID: 'nucleic_acids',
            BiologicalLevel.CELL_COMPONENT: 'cell_components',
            BiologicalLevel.CELL: 'cells',
            BiologicalLevel.TISSUE: 'tissues',
            BiologicalLevel.BIOMATERIAL: 'biological_materials',
        }

        subdir = level_dirs.get(level, '')
        if not subdir:
            return None

        # Try to find the file
        data_dir = self._data_path / subdir
        if not data_dir.exists():
            return None

        # Try exact match first, then case-insensitive
        filename = name.lower().replace(' ', '_') + '.json'
        file_path = data_dir / filename

        if not file_path.exists():
            # Try to find by scanning
            for json_file in data_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get('name', '').lower() == name.lower():
                            file_path = json_file
                            break
                except Exception:
                    continue
            else:
                return None

        # Load the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            component = self.create_component(level, data)
            self._component_cache[cache_key] = component
            return component
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def list_available_components(self, level: BiologicalLevel) -> List[str]:
        """List available components at a given level."""
        level_dirs = {
            BiologicalLevel.AMINO_ACID: 'amino_acids',
            BiologicalLevel.PROTEIN: 'proteins',
            BiologicalLevel.NUCLEIC_ACID: 'nucleic_acids',
            BiologicalLevel.CELL_COMPONENT: 'cell_components',
            BiologicalLevel.CELL: 'cells',
            BiologicalLevel.TISSUE: 'tissues',
            BiologicalLevel.BIOMATERIAL: 'biological_materials',
        }

        subdir = level_dirs.get(level, '')
        if not subdir:
            return []

        data_dir = self._data_path / subdir
        if not data_dir.exists():
            return []

        names = []
        for json_file in data_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    names.append(data.get('name', json_file.stem))
            except Exception:
                names.append(json_file.stem)

        return sorted(names)

    # === Component Saving ===

    def save_component(self, component: BiologicalComponent) -> Path:
        """
        Save a component to JSON file.

        Args:
            component: Component to save

        Returns:
            Path to saved file
        """
        level_dirs = {
            BiologicalLevel.AMINO_ACID: 'amino_acids',
            BiologicalLevel.PROTEIN: 'proteins',
            BiologicalLevel.NUCLEIC_ACID: 'nucleic_acids',
            BiologicalLevel.CELL_COMPONENT: 'cell_components',
            BiologicalLevel.CELL: 'cells',
            BiologicalLevel.TISSUE: 'tissues',
            BiologicalLevel.BIOMATERIAL: 'biological_materials',
        }

        subdir = level_dirs.get(component.level, 'other')
        data_dir = self._data_path / subdir
        data_dir.mkdir(parents=True, exist_ok=True)

        filename = component.name.lower().replace(' ', '_') + '.json'
        file_path = data_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(component.to_dict(), f, indent=2)

        logger.info(f"Saved component to {file_path}")
        return file_path

    # === Complex Material Mixing ===

    def create_complex_material(self, name: str,
                                 protein_components: Optional[List[Dict]] = None,
                                 nucleic_acid_components: Optional[List[Dict]] = None,
                                 cell_types: Optional[Dict[str, float]] = None,
                                 ecm_composition: Optional[Dict[str, float]] = None,
                                 porosity: float = 0.0) -> BiologicalComponent:
        """
        Create a complex biomaterial by mixing multiple biological components.

        Args:
            name: Material name
            protein_components: List of {name, fraction} for proteins
            nucleic_acid_components: List of {name, fraction} for nucleic acids
            cell_types: Dict of {cell_name: fraction}
            ecm_composition: Dict of {ecm_type: fraction}
            porosity: Porosity fraction (0-1)

        Returns:
            Complex BiologicalComponent
        """
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor

        components = []
        total_composition = AtomicComposition({})
        total_mass = 0.0

        # Add protein contributions
        if protein_components:
            for pc in protein_components:
                protein = self.load_component(BiologicalLevel.PROTEIN, pc['name'])
                if protein:
                    fraction = pc.get('fraction', 0.1)
                    total_mass += protein.properties.get('molecular_mass', 0) * fraction
                    components.append(protein)

        # Add nucleic acid contributions
        if nucleic_acid_components:
            for nac in nucleic_acid_components:
                na = self.load_component(BiologicalLevel.NUCLEIC_ACID, nac['name'])
                if na:
                    fraction = nac.get('fraction', 0.1)
                    total_mass += na.properties.get('molecular_mass', 0) * fraction
                    components.append(na)

        # Calculate ECM mechanical properties
        predictor = BiomaterialPredictor()
        ecm = ecm_composition or {'collagen_i': 0.3, 'water': 0.7}

        E_voigt = predictor.calculate_voigt_modulus(ecm)
        E_reuss = predictor.calculate_reuss_modulus(ecm)
        E_avg = (E_voigt + E_reuss) / 2
        E_eff = predictor.calculate_porosity_effect(E_avg, porosity)
        density = predictor.calculate_composite_density(ecm)

        # Create the complex material
        material = BiologicalComponent(
            name=name,
            level=BiologicalLevel.BIOMATERIAL,
            composition=total_composition,
            properties={
                'ecm_composition': ecm,
                'cell_types': cell_types or {},
                'protein_content': [pc['name'] for pc in (protein_components or [])],
                'nucleic_acid_content': [nac['name'] for nac in (nucleic_acid_components or [])],
                'porosity': porosity * 100,
                'youngs_modulus_MPa': round(E_eff, 4),
                'density_g_cm3': round(density, 3),
                'stiffness_category': predictor._categorize_stiffness(E_eff),
                'estimated_mass_contribution': round(total_mass, 2),
            },
            source_components=components,
            metadata={
                'derived_from': 'complex_mixing',
                'component_count': len(components),
            }
        )

        return material

    # === Derivation Chain Access ===

    @property
    def derivation_chain(self) -> BiologicalDerivationChain:
        """Access the underlying derivation chain."""
        return self._chain

    @property
    def mixing_system(self) -> BiologicalMixingSystem:
        """Access the mixing system."""
        return self._mixer
