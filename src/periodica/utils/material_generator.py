"""
Material Generator
===================
Generates material data from alloy compositions, adding microstructure,
grain structure, defect data, and processing parameters.
"""

import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from periodica.utils.derivation_metadata import DerivationSource, DerivationTracker
from periodica.utils.logger import get_logger

logger = get_logger('material_generator')

# Grain size models by processing
_GRAIN_SIZE_BY_PROCESS = {
    'As-cast': (100, 500),       # coarse grains (um)
    'Hot rolled': (20, 80),
    'Cold rolled': (5, 20),
    'Annealed': (30, 100),
    'Forged': (10, 50),
    'Powder metallurgy': (1, 15),
    'Additive manufacturing': (10, 100),
}

# Default microstructure noise parameters
_DEFAULT_NOISE = {
    'NoiseType': 'Simplex',
    'NoiseScale': 0.08,
    'NoiseOctaves': 3,
    'NoisePersistence': 0.5,
    'PhaseThreshold': 0.95,
}

# Defect density ranges by processing condition
_DEFECT_LEVELS = {
    'low': {
        'VacancyConcentration': 1e-6,
        'DislocationDensity_per_m2': 1e10,
        'InclusionDensity_per_mm3': 10,
    },
    'medium': {
        'VacancyConcentration': 1e-5,
        'DislocationDensity_per_m2': 1e12,
        'InclusionDensity_per_mm3': 50,
    },
    'high': {
        'VacancyConcentration': 1e-4,
        'DislocationDensity_per_m2': 1e14,
        'InclusionDensity_per_mm3': 200,
    },
}


class MaterialGenerator:
    """Generates material data dicts from alloy compositions."""

    def generate_from_alloy(
        self,
        alloy_data: Dict,
        processing: str = 'Hot rolled',
        defect_level: str = 'medium',
    ) -> Dict:
        """
        Generate a material data dict from alloy data.

        Args:
            alloy_data: Alloy dict (from AlloyGenerator or loaded JSON)
            processing: Processing condition for microstructure
            defect_level: 'low', 'medium', or 'high'

        Returns:
            Material dict with full microstructure data
        """
        name = alloy_data.get('Name', 'Unknown Alloy')
        category = alloy_data.get('Category', 'Metal')
        physical = alloy_data.get('PhysicalProperties', {})
        mechanical = alloy_data.get('MechanicalProperties', {})
        lattice = alloy_data.get('LatticeProperties', {})
        components = alloy_data.get('Components', [])

        # Grain structure
        grain_range = _GRAIN_SIZE_BY_PROCESS.get(processing, (20, 80))
        avg_grain = (grain_range[0] + grain_range[1]) / 2.0
        astm_number = max(1, min(14, int(round(
            -6.644 * math.log10(avg_grain / 1000.0) - 3.288
        ))))

        grain_structure = {
            'AverageGrainSize_um': round(avg_grain, 1),
            'GrainSizeDistribution': 'LogNormal',
            'GrainSizeStdDev': round(0.3 + 0.1 * (avg_grain / 100), 2),
            'ASTMGrainSizeNumber': astm_number,
            'VoronoiSeedDensity_per_mm2': round(1e6 / (avg_grain ** 2), 0),
            'GrainBoundaryWidth_nm': 0.5,
            'GrainAspectRatio': 1.0 if processing != 'Cold rolled' else 2.5,
            'TwinDensity_per_mm': self._estimate_twin_density(lattice, mechanical),
        }

        # Defect data
        defects_base = _DEFECT_LEVELS.get(defect_level, _DEFECT_LEVELS['medium'])
        defects = {
            'VacancyConcentration': defects_base['VacancyConcentration'],
            'DislocationDensity_per_m2': defects_base['DislocationDensity_per_m2'],
            'StressField': {
                'Type': 'VonMises',
                'MaxStress_MPa': mechanical.get('YieldStrength_MPa', 200) * 0.3,
            },
        }

        # Inclusions
        inclusions = self._estimate_inclusions(components, defect_level)

        # Phase composition
        phases = self._estimate_phases(alloy_data)

        # Crystallographic orientation
        orientation = {
            'PreferredOrientation': processing in ('Cold rolled', 'Hot rolled'),
            'TextureType': 'Fiber' if processing == 'Cold rolled' else 'Random',
            'TextureStrength_mrd': 3.0 if processing == 'Cold rolled' else 1.0,
        }

        # Corrosion resistance
        corrosion = self._estimate_corrosion(components)

        material = {
            'Name': f"{name} ({processing})",
            'SourceAlloy': name,
            'Category': category,
            'Processing': processing,
            'Description': f"Material derived from {name} via {processing.lower()}",
            'Components': components,
            'PhysicalProperties': physical,
            'MechanicalProperties': mechanical,
            'LatticeProperties': lattice,
            'PhaseComposition': phases,
            'Microstructure': {
                'GrainStructure': grain_structure,
                'PhaseDistribution': dict(_DEFAULT_NOISE),
                'Defects': defects,
                'Inclusions': inclusions,
            },
            'CrystallographicOrientation': orientation,
            'CorrosionResistance': corrosion,
            'Applications': alloy_data.get('Applications', []),
            'ProcessingMethods': [processing],
            'Color': alloy_data.get('Color', '#808080'),
        }

        # Stamp with derivation metadata
        alloy_name = alloy_data.get('Name', 'unknown')
        DerivationTracker.stamp(
            material,
            source=DerivationSource.AUTO_GENERATED,
            derived_from=[f"alloy:{alloy_name}"],
            derivation_chain=['elements', 'alloy', 'material'],
            confidence=0.6,
        )

        return material

    def generate_all(
        self,
        alloys: List[Dict],
        processing_variants: Optional[List[str]] = None,
        count_limit: int = 50,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Generate materials from a list of alloys.

        Args:
            alloys: List of alloy data dicts
            processing_variants: Processing types to generate per alloy
            count_limit: Maximum number of materials
            progress_callback: fn(percent, message)

        Returns:
            List of material data dicts
        """
        if processing_variants is None:
            processing_variants = ['Hot rolled']

        materials = []
        total = len(alloys) * len(processing_variants)

        for i, alloy in enumerate(alloys):
            for proc in processing_variants:
                if len(materials) >= count_limit:
                    break

                mat = self.generate_from_alloy(alloy, processing=proc)
                materials.append(mat)

                if progress_callback:
                    done = len(materials)
                    pct = min(int(done / max(total, 1) * 100), 99)
                    progress_callback(pct, f"Generated material {done}/{total}")

            if len(materials) >= count_limit:
                break

        if progress_callback:
            progress_callback(100, f"Generated {len(materials)} materials")

        return materials

    def _estimate_twin_density(self, lattice: Dict, mechanical: Dict) -> float:
        """Estimate twin density from structure and properties."""
        structure = lattice.get('PrimaryStructure', 'FCC')
        # FCC metals twin more readily
        if structure == 'FCC':
            return 50.0
        elif structure == 'HCP':
            return 30.0
        elif structure == 'BCC':
            return 5.0
        return 10.0

    def _estimate_inclusions(
        self, components: List[Dict], defect_level: str
    ) -> Dict:
        """Estimate inclusion data from composition."""
        density = _DEFECT_LEVELS.get(defect_level, {}).get(
            'InclusionDensity_per_mm3', 50
        )

        # Check for common inclusion-forming elements
        inclusion_types = []
        elem_set = {c['Element'] for c in components}

        if 'S' in elem_set or 'Mn' in elem_set:
            inclusion_types.append({'Type': 'MnS', 'Shape': 'Elongated'})
        if 'Al' in elem_set:
            inclusion_types.append({'Type': 'Al2O3', 'Shape': 'Globular'})
        if 'Si' in elem_set:
            inclusion_types.append({'Type': 'SiO2', 'Shape': 'Globular'})

        if not inclusion_types:
            inclusion_types.append({'Type': 'Oxide', 'Shape': 'Globular'})

        return {
            'Density_per_mm3': density,
            'Types': inclusion_types,
            'AverageSize_um': 2.0 if defect_level == 'low' else 5.0,
        }

    def _estimate_phases(self, alloy_data: Dict) -> Dict:
        """Estimate phase composition from alloy data."""
        lattice = alloy_data.get('LatticeProperties', {})
        structure = lattice.get('PrimaryStructure', 'FCC')
        category = alloy_data.get('Category', '')

        # Default single-phase
        phases = [{
            'Name': self._structure_to_phase_name(structure),
            'Symbol': self._structure_to_symbol(structure),
            'Structure': structure,
            'VolumePercent': 100,
            'Magnetic': structure == 'BCC' and 'Steel' in category,
            'Hardness_HV': alloy_data.get('MechanicalProperties', {}).get('Hardness_HV', 200),
        }]

        return {
            'Phases': phases,
            'TransformationTemperatures': {},
        }

    def _structure_to_phase_name(self, structure: str) -> str:
        """Map crystal structure to conventional phase name."""
        return {
            'FCC': 'Austenite', 'BCC': 'Ferrite', 'HCP': 'Alpha',
            'BCT': 'Martensite',
        }.get(structure, 'Primary')

    def _structure_to_symbol(self, structure: str) -> str:
        """Map crystal structure to phase symbol."""
        return {
            'FCC': 'gamma', 'BCC': 'alpha', 'HCP': 'alpha',
            'BCT': 'alpha_prime',
        }.get(structure, 'alpha')

    def _estimate_corrosion(self, components: List[Dict]) -> Dict:
        """Estimate corrosion resistance from composition."""
        elem_pct = {}
        for c in components:
            elem = c.get('Element', '')
            pct = (c.get('MinPercent', 0) + c.get('MaxPercent', 0)) / 2.0
            elem_pct[elem] = pct

        # PREN = %Cr + 3.3*%Mo + 16*%N
        cr = elem_pct.get('Cr', 0)
        mo = elem_pct.get('Mo', 0)
        pren = cr + 3.3 * mo

        film = 'None'
        if cr > 10:
            film = 'Cr2O3'
        elif 'Al' in elem_pct and elem_pct['Al'] > 3:
            film = 'Al2O3'
        elif 'Ti' in elem_pct and elem_pct['Ti'] > 50:
            film = 'TiO2'

        return {
            'PREN': round(pren, 1),
            'PassivationFilmComposition': film,
            'PittingPotential_mV_SCE': round(pren * 10, 0) if pren > 0 else 0,
        }

    def save_materials(
        self,
        materials: List[Dict],
        output_dir: Optional[str] = None,
    ) -> int:
        """Save materials to JSON files. Returns count saved."""
        if output_dir is None:
            output_dir = str(
                Path(__file__).parent.parent / 'data' / 'active' / 'materials'
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved = 0

        for mat in materials:
            name = mat.get('Name', f'material_{saved}')
            safe = (name.replace(' ', '_').replace('/', '_')
                    .replace('(', '').replace(')', '').replace("'", ''))
            filepath = out_path / f"{safe}.json"

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(mat, f, indent=2, ensure_ascii=False)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

        logger.info(f"Saved {saved} materials to {output_dir}")
        return saved
