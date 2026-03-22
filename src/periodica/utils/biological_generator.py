"""
Biological Generator
=====================
Facade over the existing biological derivation chain and predictors.
Provides auto-generation methods for amino acids, proteins, nucleic acids,
cell components, cells, and biomaterials.
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from periodica.utils.derivation_metadata import DerivationSource, DerivationTracker
from periodica.utils.logger import get_logger

logger = get_logger('biological_generator')

# The 20 standard amino acids with their single-letter codes
_STANDARD_AA_SYMBOLS = [
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
]

# Common protein sequences for auto-generation
_PROTEIN_TEMPLATES = [
    {
        'name': 'Insulin Chain A',
        'sequence': 'GIVEQCCTSICSLYQLENYCN',
        'organism': 'Homo sapiens',
        'function': 'Glucose regulation',
        'localization': 'extracellular',
    },
    {
        'name': 'Ubiquitin',
        'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        'organism': 'Homo sapiens',
        'function': 'Protein degradation signaling',
        'localization': 'cytoplasm',
    },
    {
        'name': 'Ferredoxin',
        'sequence': 'ATYKVTLVTPTGNVEFQCPDDVYILDAAEEEGIDLPYSCRAGACSTCAGKLVSGTVDQSDQSFLDDDQIEAGYVL',
        'organism': 'Spinacia oleracea',
        'function': 'Electron transfer',
        'localization': 'chloroplast',
    },
    {
        'name': 'Thioredoxin',
        'sequence': 'MVKQIESKTAFQEALDAAGDKLVVVDFSATWCGPCKMIKPFFHSLSEKYSNVIFLEVDVDDCQDVASECEVKCMPTFQFFKKGQKVGEFSGANKEKLEATINELV',
        'organism': 'Escherichia coli',
        'function': 'Redox regulation',
        'localization': 'cytoplasm',
    },
    {
        'name': 'Rubredoxin',
        'sequence': 'MKKYVCTVCGYEYDPAEGDPDNGVKPGTSFDDLPADWVCPVCGAPKSEFERIED',
        'organism': 'Clostridium pasteurianum',
        'function': 'Electron transfer',
        'localization': 'cytoplasm',
    },
    {
        'name': 'Crambin',
        'sequence': 'TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN',
        'organism': 'Crambe hispanica',
        'function': 'Seed storage protein',
        'localization': 'seed',
    },
    {
        'name': 'Beta Defensin',
        'sequence': 'MRIHYLLFALLFLFLVPVPGHGGIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK',
        'organism': 'Homo sapiens',
        'function': 'Antimicrobial peptide',
        'localization': 'extracellular',
    },
    {
        'name': 'Metallothionein',
        'sequence': 'MDPNCSCAAGDSCTCAGSCKCKECKCTSCKKSCCSCPVGCAKCAQGCICKGASDKCSCCA',
        'organism': 'Homo sapiens',
        'function': 'Metal ion binding',
        'localization': 'cytoplasm',
    },
]

# Common nucleic acid sequences for auto-generation
_NUCLEIC_ACID_TEMPLATES = [
    {
        'name': 'tRNA-Phe Anticodon Stem-Loop',
        'sequence': 'GGGAGAUCGCCAAGAUGGCAGAACUCCGCCUCC',
        'na_type': 'RNA',
        'organism': 'Saccharomyces cerevisiae',
        'function': 'Transfer RNA',
    },
    {
        'name': 'TATA Box Promoter',
        'sequence': 'GCGCGCTATAAAAGGCGCGC',
        'na_type': 'DNA',
        'organism': 'Consensus',
        'function': 'Transcription initiation',
    },
    {
        'name': 'Kozak Consensus',
        'sequence': 'GCCGCCACCATGGCG',
        'na_type': 'DNA',
        'organism': 'Vertebrate consensus',
        'function': 'Translation initiation',
    },
    {
        'name': 'Poly-A Signal',
        'sequence': 'TTTTAATAAAAGATCCTTTATTTT',
        'na_type': 'DNA',
        'organism': 'Mammalian consensus',
        'function': 'mRNA polyadenylation signal',
    },
    {
        'name': 'CpG Island Fragment',
        'sequence': 'GCGCGCGTACGCGCGATCGCGCGCGATACGCGCG',
        'na_type': 'DNA',
        'organism': 'Synthetic',
        'function': 'Epigenetic regulation',
    },
    {
        'name': 'miR-21 Precursor',
        'sequence': 'UGUCGGGUAGCUUAUCAGACUGAUGUUGACUGUUGAAUCUCAUGGCAACACCAGUCGAUGGGCUGUC',
        'na_type': 'RNA',
        'organism': 'Homo sapiens',
        'function': 'MicroRNA regulation',
    },
    {
        'name': 'Shine-Dalgarno Sequence',
        'sequence': 'AAGGAGGUGATCATG',
        'na_type': 'RNA',
        'organism': 'Escherichia coli',
        'function': 'Prokaryotic ribosome binding',
    },
    {
        'name': 'Telomere Repeat',
        'sequence': 'TTAGGGTTAGGGTTAGGGTTAGGGTTAGGG',
        'na_type': 'DNA',
        'organism': 'Homo sapiens',
        'function': 'Chromosome end protection',
    },
]

# Cell component templates
_CELL_COMPONENT_TEMPLATES = [
    {
        'name': 'Ribosome (80S)',
        'component_type': 'organelle',
        'diameter_nm': 25,
        'copy_number': 10000000,
        'description': 'Eukaryotic ribosome for protein synthesis',
    },
    {
        'name': 'Proteasome (26S)',
        'component_type': 'complex',
        'diameter_nm': 15,
        'copy_number': 30000,
        'description': 'Protein degradation complex',
    },
    {
        'name': 'Plasma Membrane Patch',
        'component_type': 'membrane',
        'thickness_nm': 7.5,
        'description': 'Lipid bilayer with embedded proteins',
    },
    {
        'name': 'Nuclear Pore Complex',
        'component_type': 'complex',
        'diameter_nm': 120,
        'copy_number': 3000,
        'description': 'Nuclear envelope transport channel',
    },
    {
        'name': 'Spliceosome',
        'component_type': 'complex',
        'diameter_nm': 40,
        'copy_number': 100000,
        'description': 'Pre-mRNA splicing complex',
    },
]

# Cell type templates
_CELL_TEMPLATES = [
    {
        'name': 'Hepatocyte',
        'cell_type': 'epithelial',
        'organism': 'Homo sapiens',
        'diameter_um': 25,
        'description': 'Liver parenchymal cell',
    },
    {
        'name': 'Neuron (Cortical)',
        'cell_type': 'neuron',
        'organism': 'Homo sapiens',
        'diameter_um': 15,
        'description': 'Cortical pyramidal neuron',
    },
    {
        'name': 'Osteocyte',
        'cell_type': 'connective',
        'organism': 'Homo sapiens',
        'diameter_um': 10,
        'description': 'Bone matrix cell',
    },
    {
        'name': 'Macrophage (M1)',
        'cell_type': 'immune',
        'organism': 'Homo sapiens',
        'diameter_um': 20,
        'description': 'Classically activated macrophage',
    },
    {
        'name': 'Keratinocyte',
        'cell_type': 'epithelial',
        'organism': 'Homo sapiens',
        'diameter_um': 30,
        'description': 'Skin epidermal cell',
    },
    {
        'name': 'Chondrocyte',
        'cell_type': 'connective',
        'organism': 'Homo sapiens',
        'diameter_um': 13,
        'description': 'Cartilage cell',
    },
]

# Biomaterial templates
_BIOMATERIAL_TEMPLATES = [
    {
        'name': 'Trabecular Bone',
        'tissue_type': 'bone',
        'ecm_composition': {'hydroxyapatite': 0.5, 'collagen': 0.3},
        'porosity': 0.75,
        'description': 'Spongy bone tissue',
    },
    {
        'name': 'Tendon',
        'tissue_type': 'connective',
        'ecm_composition': {'collagen': 0.8, 'elastin': 0.05},
        'porosity': 0.1,
        'description': 'Dense connective tissue',
    },
    {
        'name': 'Corneal Stroma',
        'tissue_type': 'connective',
        'ecm_composition': {'collagen': 0.7, 'proteoglycan': 0.1},
        'porosity': 0.2,
        'description': 'Transparent corneal tissue',
    },
    {
        'name': 'Dermis',
        'tissue_type': 'connective',
        'ecm_composition': {'collagen': 0.6, 'elastin': 0.1, 'hyaluronic_acid': 0.05},
        'porosity': 0.3,
        'description': 'Skin dermal layer',
    },
    {
        'name': 'Meniscus',
        'tissue_type': 'cartilage',
        'ecm_composition': {'collagen': 0.6, 'proteoglycan': 0.1},
        'porosity': 0.3,
        'description': 'Knee fibrocartilage',
    },
    {
        'name': 'Intervertebral Disc',
        'tissue_type': 'cartilage',
        'ecm_composition': {'collagen': 0.5, 'proteoglycan': 0.2, 'elastin': 0.05},
        'porosity': 0.4,
        'description': 'Spinal shock absorber',
    },
]


class BiologicalGenerator:
    """
    Facade over existing biological derivation chain for auto-generation.
    Generates amino acids, proteins, nucleic acids, cell components,
    cells, and biomaterials.
    """

    def __init__(self):
        self._protein_predictor = None
        self._na_predictor = None
        self._aa_predictor = None
        self._cell_predictor = None
        self._biomaterial_predictor = None

    def _get_protein_predictor(self):
        if self._protein_predictor is None:
            from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor
            self._protein_predictor = ProteinPredictor()
        return self._protein_predictor

    def _get_na_predictor(self):
        if self._na_predictor is None:
            from periodica.utils.predictors.biological.nucleic_acid_predictor import NucleicAcidPredictor
            self._na_predictor = NucleicAcidPredictor()
        return self._na_predictor

    def _get_aa_predictor(self):
        if self._aa_predictor is None:
            from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidPredictor
            self._aa_predictor = AminoAcidPredictor()
        return self._aa_predictor

    def _get_cell_predictor(self):
        if self._cell_predictor is None:
            from periodica.utils.predictors.biological.cell_predictor import CellPredictor
            self._cell_predictor = CellPredictor()
        return self._cell_predictor

    def _get_biomaterial_predictor(self):
        if self._biomaterial_predictor is None:
            from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor
            self._biomaterial_predictor = BiomaterialPredictor()
        return self._biomaterial_predictor

    # Standard biochemistry pKa and ionization data for the 20 amino acids.
    # Source: CRC Handbook of Chemistry and Physics, Lehninger Principles of Biochemistry.
    _AMINO_ACID_BIOCHEMISTRY = {
        'A': {'pKa_carboxyl': 2.34, 'pKa_amino': 9.69, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'R': {'pKa_carboxyl': 2.17, 'pKa_amino': 9.04, 'pKa_sidechain': 12.48, 'sidechain_ionization': 'basic', 'can_form_disulfide': False},
        'N': {'pKa_carboxyl': 2.02, 'pKa_amino': 8.80, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'D': {'pKa_carboxyl': 2.09, 'pKa_amino': 9.82, 'pKa_sidechain': 3.65,  'sidechain_ionization': 'acidic','can_form_disulfide': False},
        'C': {'pKa_carboxyl': 1.96, 'pKa_amino': 10.28,'pKa_sidechain': 8.18,  'sidechain_ionization': 'acidic','can_form_disulfide': True},
        'E': {'pKa_carboxyl': 2.19, 'pKa_amino': 9.67, 'pKa_sidechain': 4.25,  'sidechain_ionization': 'acidic','can_form_disulfide': False},
        'Q': {'pKa_carboxyl': 2.17, 'pKa_amino': 9.13, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'G': {'pKa_carboxyl': 2.34, 'pKa_amino': 9.60, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'H': {'pKa_carboxyl': 1.82, 'pKa_amino': 9.17, 'pKa_sidechain': 6.00,  'sidechain_ionization': 'basic', 'can_form_disulfide': False},
        'I': {'pKa_carboxyl': 2.36, 'pKa_amino': 9.60, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'L': {'pKa_carboxyl': 2.36, 'pKa_amino': 9.60, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'K': {'pKa_carboxyl': 2.18, 'pKa_amino': 8.95, 'pKa_sidechain': 10.53, 'sidechain_ionization': 'basic', 'can_form_disulfide': False},
        'M': {'pKa_carboxyl': 2.28, 'pKa_amino': 9.21, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'F': {'pKa_carboxyl': 1.83, 'pKa_amino': 9.13, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'P': {'pKa_carboxyl': 1.99, 'pKa_amino': 10.60,'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'S': {'pKa_carboxyl': 2.21, 'pKa_amino': 9.15, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'T': {'pKa_carboxyl': 2.09, 'pKa_amino': 9.10, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'W': {'pKa_carboxyl': 2.83, 'pKa_amino': 9.39, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
        'Y': {'pKa_carboxyl': 2.20, 'pKa_amino': 9.11, 'pKa_sidechain': 10.07, 'sidechain_ionization': 'acidic','can_form_disulfide': False},
        'V': {'pKa_carboxyl': 2.32, 'pKa_amino': 9.62, 'pKa_sidechain': None,  'sidechain_ionization': None,    'can_form_disulfide': False},
    }

    # ─── Amino Acids ───────────────────────────────────────────────

    def generate_standard_amino_acids(
        self,
        count_limit: int = 20,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Generate the 20 standard amino acids with derived properties."""
        predictor = self._get_aa_predictor()
        results = []

        symbols = _STANDARD_AA_SYMBOLS[:count_limit]
        for i, symbol in enumerate(symbols):
            try:
                result = predictor.predict_from_symbol(symbol, pH=7.0)
                aa_dict = {
                    'name': result.name,
                    'symbol': symbol,
                    'category': result.category,
                    'charge_pH7': round(result.charge, 3),
                    'isoelectric_point': round(result.isoelectric_point, 2),
                    'hydropathy_index': result.hydropathy_index,
                    'helix_propensity': result.helix_propensity,
                    'sheet_propensity': result.sheet_propensity,
                    'turn_propensity': result.turn_propensity,
                }
                if result.molecular_mass is not None:
                    aa_dict['molecular_mass'] = round(result.molecular_mass, 3)
                if result.molecular_formula is not None:
                    aa_dict['molecular_formula'] = result.molecular_formula

                # Include standard biochemistry reference data
                biochem = self._AMINO_ACID_BIOCHEMISTRY.get(symbol, {})
                aa_dict.update(biochem)

                DerivationTracker.stamp(
                    aa_dict,
                    source=DerivationSource.AUTO_GENERATED,
                    derived_from=[f"element:C", f"element:H", f"element:N", f"element:O"],
                    derivation_chain=['elements', 'amino_acid_predictor'],
                    confidence=0.9,
                )
                results.append(aa_dict)
            except Exception as e:
                logger.warning(f"Failed to generate AA {symbol}: {e}")

            if progress_callback:
                pct = min(int((i + 1) / len(symbols) * 100), 99)
                progress_callback(pct, f"Generated amino acid {symbol}")

        if progress_callback:
            progress_callback(100, f"Generated {len(results)} amino acids")
        return results

    # ─── Proteins ──────────────────────────────────────────────────

    def generate_proteins(
        self,
        count_limit: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Generate proteins from template sequences."""
        predictor = self._get_protein_predictor()
        results = []

        templates = _PROTEIN_TEMPLATES[:count_limit]
        for i, tmpl in enumerate(templates):
            try:
                protein = predictor.create_protein_json(
                    sequence=tmpl['sequence'],
                    name=tmpl['name'],
                    organism=tmpl.get('organism', 'Unknown'),
                    function=tmpl.get('function', ''),
                    localization=tmpl.get('localization', 'unknown'),
                )

                DerivationTracker.stamp(
                    protein,
                    source=DerivationSource.AUTO_GENERATED,
                    derived_from=['amino_acids'],
                    derivation_chain=['amino_acids', 'protein_predictor'],
                    confidence=0.8,
                )
                results.append(protein)
            except Exception as e:
                logger.warning(f"Failed to generate protein {tmpl['name']}: {e}")

            if progress_callback:
                pct = min(int((i + 1) / len(templates) * 100), 99)
                progress_callback(pct, f"Generated protein {tmpl['name']}")

        if progress_callback:
            progress_callback(100, f"Generated {len(results)} proteins")
        return results

    # ─── Nucleic Acids ─────────────────────────────────────────────

    def generate_nucleic_acids(
        self,
        count_limit: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Generate nucleic acid sequences from templates."""
        predictor = self._get_na_predictor()
        results = []

        templates = _NUCLEIC_ACID_TEMPLATES[:count_limit]
        for i, tmpl in enumerate(templates):
            try:
                na = predictor.create_nucleic_acid_json(
                    sequence=tmpl['sequence'],
                    name=tmpl['name'],
                    na_type=tmpl.get('na_type', 'DNA'),
                    organism=tmpl.get('organism', 'Unknown'),
                    function=tmpl.get('function', ''),
                )

                DerivationTracker.stamp(
                    na,
                    source=DerivationSource.AUTO_GENERATED,
                    derived_from=['elements'],
                    derivation_chain=['elements', 'nucleic_acid_predictor'],
                    confidence=0.85,
                )
                results.append(na)
            except Exception as e:
                logger.warning(f"Failed to generate NA {tmpl['name']}: {e}")

            if progress_callback:
                pct = min(int((i + 1) / len(templates) * 100), 99)
                progress_callback(pct, f"Generated {tmpl['name']}")

        if progress_callback:
            progress_callback(100, f"Generated {len(results)} nucleic acids")
        return results

    # ─── Cell Components ───────────────────────────────────────────

    def generate_cell_components(
        self,
        count_limit: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Generate cell component data from templates."""
        results = []

        templates = _CELL_COMPONENT_TEMPLATES[:count_limit]
        for i, tmpl in enumerate(templates):
            try:
                component = {
                    'name': tmpl['name'],
                    'component_type': tmpl['component_type'],
                    'description': tmpl.get('description', ''),
                    'properties': {},
                }

                if 'diameter_nm' in tmpl:
                    component['properties']['diameter_nm'] = tmpl['diameter_nm']
                if 'thickness_nm' in tmpl:
                    component['properties']['thickness_nm'] = tmpl['thickness_nm']
                if 'copy_number' in tmpl:
                    component['properties']['copy_number'] = tmpl['copy_number']

                DerivationTracker.stamp(
                    component,
                    source=DerivationSource.AUTO_GENERATED,
                    derived_from=['proteins', 'nucleic_acids'],
                    derivation_chain=['proteins', 'nucleic_acids', 'cell_component'],
                    confidence=0.7,
                )
                results.append(component)
            except Exception as e:
                logger.warning(f"Failed to generate component {tmpl['name']}: {e}")

            if progress_callback:
                pct = min(int((i + 1) / len(templates) * 100), 99)
                progress_callback(pct, f"Generated {tmpl['name']}")

        if progress_callback:
            progress_callback(100, f"Generated {len(results)} cell components")
        return results

    # ─── Cells ─────────────────────────────────────────────────────

    def generate_cells(
        self,
        count_limit: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Generate cell type data using the cell predictor."""
        predictor = self._get_cell_predictor()
        results = []

        templates = _CELL_TEMPLATES[:count_limit]
        for i, tmpl in enumerate(templates):
            try:
                cell = predictor.analyze_cell(
                    diameter_um=tmpl['diameter_um'],
                    name=tmpl['name'],
                    cell_type=tmpl.get('cell_type', 'generic'),
                    organism=tmpl.get('organism', 'Unknown'),
                )
                cell['description'] = tmpl.get('description', '')

                DerivationTracker.stamp(
                    cell,
                    source=DerivationSource.AUTO_GENERATED,
                    derived_from=['cell_components'],
                    derivation_chain=['cell_components', 'cell_predictor'],
                    confidence=0.7,
                )
                results.append(cell)
            except Exception as e:
                logger.warning(f"Failed to generate cell {tmpl['name']}: {e}")

            if progress_callback:
                pct = min(int((i + 1) / len(templates) * 100), 99)
                progress_callback(pct, f"Generated {tmpl['name']}")

        if progress_callback:
            progress_callback(100, f"Generated {len(results)} cells")
        return results

    # ─── Biomaterials ──────────────────────────────────────────────

    def generate_biomaterials(
        self,
        count_limit: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Generate biomaterial data using the biomaterial predictor."""
        predictor = self._get_biomaterial_predictor()
        results = []

        templates = _BIOMATERIAL_TEMPLATES[:count_limit]
        for i, tmpl in enumerate(templates):
            try:
                biomat = predictor.analyze_biomaterial(
                    name=tmpl['name'],
                    tissue_type=tmpl.get('tissue_type', 'generic'),
                    ecm_composition=tmpl.get('ecm_composition', {'collagen': 0.5}),
                    cell_composition={},
                    porosity=tmpl.get('porosity', 0.2),
                )
                biomat['description'] = tmpl.get('description', '')

                DerivationTracker.stamp(
                    biomat,
                    source=DerivationSource.AUTO_GENERATED,
                    derived_from=['cells'],
                    derivation_chain=['cells', 'biomaterial_predictor'],
                    confidence=0.6,
                )
                results.append(biomat)
            except Exception as e:
                logger.warning(f"Failed to generate biomaterial {tmpl['name']}: {e}")

            if progress_callback:
                pct = min(int((i + 1) / len(templates) * 100), 99)
                progress_callback(pct, f"Generated {tmpl['name']}")

        if progress_callback:
            progress_callback(100, f"Generated {len(results)} biomaterials")
        return results

    # ─── Unified interface ─────────────────────────────────────────

    def generate_category(
        self,
        category: str,
        count_limit: int = 20,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Generate items for a specific biological category.

        Args:
            category: One of 'amino_acids', 'proteins', 'nucleic_acids',
                     'cell_components', 'cells', 'biomaterials'
            count_limit: Maximum items to generate
            progress_callback: fn(percent, message)
        """
        generators = {
            'amino_acids': self.generate_standard_amino_acids,
            'proteins': self.generate_proteins,
            'nucleic_acids': self.generate_nucleic_acids,
            'cell_components': self.generate_cell_components,
            'cells': self.generate_cells,
            'biomaterials': self.generate_biomaterials,
        }

        gen = generators.get(category)
        if gen is None:
            raise ValueError(f"Unknown category: {category}. "
                           f"Valid: {list(generators.keys())}")

        return gen(count_limit=count_limit, progress_callback=progress_callback)

    def save_items(
        self,
        items: List[Dict],
        category: str,
        output_dir: Optional[str] = None,
    ) -> int:
        """Save generated items to JSON files. Returns count saved."""
        category_dirs = {
            'amino_acids': 'amino_acids',
            'proteins': 'proteins',
            'nucleic_acids': 'nucleic_acids',
            'cell_components': 'cell_components',
            'cells': 'cells',
            'biomaterials': 'biological_materials',
        }

        if output_dir is None:
            subdir = category_dirs.get(category, category)
            output_dir = str(
                Path(__file__).parent.parent / 'data' / 'active' / subdir
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved = 0

        for item in items:
            name = item.get('name', item.get('Name', f'item_{saved}'))
            safe = (name.replace(' ', '_').replace('/', '_')
                    .replace('(', '').replace(')', '').replace("'", ''))
            filepath = out_path / f"{safe}.json"

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(item, f, indent=2, ensure_ascii=False)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

        logger.info(f"Saved {saved} {category} items to {output_dir}")
        return saved
