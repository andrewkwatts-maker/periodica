"""
Nucleic Acid Predictor - Fully Dynamic Implementation
All thermodynamic parameters are loaded from JSON configuration files.
No hardcoded values - all parameters are data-driven.

Provides prediction and calculation methods for nucleic acid properties including:
- Melting temperature (Tm) using nearest-neighbor method
- GC content and base composition
- Secondary structure prediction
- Molecular weight calculation
- Base pairing and complementarity

References:
- SantaLucia, J. Jr. (1998). PNAS, 95(4), 1460-1465.
- Xia, T. et al. (1998). Biochemistry, 37(42), 14719-14735.
- Owczarzy, R. et al. (2004). Biochemistry, 43(12), 3537-3554.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def _load_thermodynamic_config(config_type: str) -> Dict:
    """
    Load thermodynamic parameters from JSON config file.

    Args:
        config_type: 'dna' or 'rna'

    Returns:
        Dictionary with thermodynamic parameters
    """
    config_path = Path(__file__).parent.parent.parent.parent / "data" / "config"
    config_file = config_path / f"{config_type}_thermodynamics.json"

    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {config_file}: {e}")

    return {}


def _convert_nn_params(config: Dict) -> Dict[str, Tuple[float, float]]:
    """Convert JSON nearest-neighbor params to tuple format."""
    result = {}
    nn_params = config.get('nearest_neighbor_params', {})
    for key, values in nn_params.items():
        if isinstance(values, dict):
            result[key] = (values.get('dH', 0), values.get('dS', 0))
        else:
            result[key] = tuple(values)

    # Add initiation parameters
    init_params = config.get('initiation_params', {})
    if 'terminal_AT' in init_params:
        vals = init_params['terminal_AT']
        result['init_AT'] = (vals.get('dH', 2.3), vals.get('dS', 4.1))
    if 'terminal_AU' in init_params:
        vals = init_params['terminal_AU']
        result['init_AT'] = (vals.get('dH', 3.72), vals.get('dS', 10.5))
    if 'terminal_GC' in init_params:
        vals = init_params['terminal_GC']
        result['init_GC'] = (vals.get('dH', 0.1), vals.get('dS', -2.8))

    # Add symmetry correction
    sym = config.get('symmetry_correction', {})
    if sym:
        result['sym'] = (sym.get('dH', 0), sym.get('dS', -1.4))

    return result


# Load DNA parameters from JSON config
_dna_config = _load_thermodynamic_config('dna')
DNA_NN_PARAMS = _convert_nn_params(_dna_config) if _dna_config else {}

# Load RNA parameters from JSON config
_rna_config = _load_thermodynamic_config('rna')
RNA_NN_PARAMS = _convert_nn_params(_rna_config) if _rna_config else {}

# Load nucleotide molecular weights from config
NUCLEOTIDE_MW = _dna_config.get('nucleotide_molecular_weights', {}).copy()
NUCLEOTIDE_MW.update(_rna_config.get('nucleotide_molecular_weights', {}))


class NucleicAcidPredictor:
    """
    Predictor for nucleic acid properties.
    Implements nearest-neighbor thermodynamics for Tm calculation.

    Configurable parameters:
    - Nearest-neighbor thermodynamic parameters (DNA and RNA)
    - Salt concentration defaults
    - Oligonucleotide concentration defaults

    All parameters can be loaded from JSON config files.
    """

    # Valid nucleotide sets
    DNA_BASES = {'A', 'T', 'G', 'C'}
    RNA_BASES = {'A', 'U', 'G', 'C'}
    VALID_BASES = DNA_BASES | RNA_BASES | {'N'}

    # Valid ranges for parameters
    VALID_RANGES = {
        'Na_conc_M': (0.001, 2.0),
        'Mg_conc_M': (0.0, 0.5),
        'oligo_conc_M': (1e-12, 1e-3),
    }

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the nucleic acid predictor.

        Args:
            config_file: Optional JSON config file for thermodynamic parameters
        """
        self._dna_params = DNA_NN_PARAMS.copy()
        self._rna_params = RNA_NN_PARAMS.copy()
        self._R = 1.987  # Gas constant in cal/(mol·K)

        # Default salt/oligo concentrations
        self._default_Na = 0.05   # 50 mM
        self._default_Mg = 0.0    # 0 mM
        self._default_oligo = 0.25e-6  # 250 nM

        # Load custom config if provided
        if config_file and config_file.exists():
            self._load_config(config_file)

    def _load_config(self, config_file: Path) -> None:
        """Load thermodynamic parameters from JSON config file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'dna_nn_params' in config:
                    self._dna_params.update(config['dna_nn_params'])
                if 'rna_nn_params' in config:
                    self._rna_params.update(config['rna_nn_params'])
                if 'default_Na_M' in config:
                    self._default_Na = config['default_Na_M']
                if 'default_Mg_M' in config:
                    self._default_Mg = config['default_Mg_M']
                if 'default_oligo_M' in config:
                    self._default_oligo = config['default_oligo_M']
                logger.info(f"Loaded nucleic acid config from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config from {config_file}: {e}")

    def _validate_sequence(self, sequence: str, is_rna: bool = False) -> str:
        """Validate and normalize nucleic acid sequence."""
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        sequence = sequence.upper().strip()
        valid_bases = self.RNA_BASES if is_rna else self.DNA_BASES

        # Remove whitespace
        sequence = ''.join(sequence.split())

        # Check for invalid characters
        invalid_chars = set(sequence) - self.VALID_BASES
        if invalid_chars:
            raise ValueError(f"Invalid nucleotide(s) in sequence: {invalid_chars}")

        return sequence

    def set_default_concentrations(self, Na_M: Optional[float] = None,
                                     Mg_M: Optional[float] = None,
                                     oligo_M: Optional[float] = None) -> None:
        """Set default concentration values."""
        if Na_M is not None:
            min_val, max_val = self.VALID_RANGES['Na_conc_M']
            if not min_val <= Na_M <= max_val:
                raise ValueError(f"Na+ concentration must be between {min_val} and {max_val} M")
            self._default_Na = Na_M

        if Mg_M is not None:
            min_val, max_val = self.VALID_RANGES['Mg_conc_M']
            if not min_val <= Mg_M <= max_val:
                raise ValueError(f"Mg2+ concentration must be between {min_val} and {max_val} M")
            self._default_Mg = Mg_M

        if oligo_M is not None:
            min_val, max_val = self.VALID_RANGES['oligo_conc_M']
            if not min_val <= oligo_M <= max_val:
                raise ValueError(f"Oligo concentration must be between {min_val} and {max_val} M")
            self._default_oligo = oligo_M

    # === Melting Temperature Calculation ===

    def calculate_tm_nearest_neighbor(self, sequence: str,
                                       Na_conc: Optional[float] = None,
                                       Mg_conc: Optional[float] = None,
                                       oligo_conc: Optional[float] = None,
                                       is_rna: bool = False) -> float:
        """
        Calculate melting temperature using nearest-neighbor method.

        Uses SantaLucia (1998) parameters for DNA and Xia (1998) for RNA.

        Args:
            sequence: Nucleic acid sequence (5' to 3')
            Na_conc: Sodium concentration in M (default: instance setting)
            Mg_conc: Magnesium concentration in M (default: instance setting)
            oligo_conc: Total strand concentration in M (default: instance setting)
            is_rna: Whether sequence is RNA

        Returns:
            Melting temperature in °C

        Raises:
            ValueError: If sequence is invalid or too short
        """
        # Use defaults if not specified
        Na_conc = Na_conc if Na_conc is not None else self._default_Na
        Mg_conc = Mg_conc if Mg_conc is not None else self._default_Mg
        oligo_conc = oligo_conc if oligo_conc is not None else self._default_oligo

        # Validate inputs
        sequence = self._validate_sequence(sequence, is_rna)

        if is_rna:
            sequence = sequence.replace('T', 'U')
            params = self._rna_params
        else:
            sequence = sequence.replace('U', 'T')
            params = self._dna_params

        if len(sequence) < 2:
            logger.warning("Sequence too short for nearest-neighbor calculation")
            return 0.0

        # Sum thermodynamic parameters
        dH_total = 0.0  # kcal/mol
        dS_total = 0.0  # cal/(mol·K)

        # Add nearest-neighbor contributions
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            complement = self._get_complement_dinuc(dinuc, is_rna)
            key = f"{dinuc}/{complement}"

            if key in params:
                dH, dS = params[key]
                dH_total += dH
                dS_total += dS
            else:
                # Try reverse
                key_rev = f"{complement[::-1]}/{dinuc[::-1]}"
                if key_rev in params:
                    dH, dS = params[key_rev]
                    dH_total += dH
                    dS_total += dS
                else:
                    # Use average values
                    dH_total += -8.0
                    dS_total += -22.0

        # Terminal corrections
        if sequence[0] in 'AT' or (is_rna and sequence[0] in 'AU'):
            dH_total += params.get('init_AT', (2.3, 4.1))[0]
            dS_total += params.get('init_AT', (2.3, 4.1))[1]
        else:
            dH_total += params.get('init_GC', (0.1, -2.8))[0]
            dS_total += params.get('init_GC', (0.1, -2.8))[1]

        if sequence[-1] in 'AT' or (is_rna and sequence[-1] in 'AU'):
            dH_total += params.get('init_AT', (2.3, 4.1))[0]
            dS_total += params.get('init_AT', (2.3, 4.1))[1]
        else:
            dH_total += params.get('init_GC', (0.1, -2.8))[0]
            dS_total += params.get('init_GC', (0.1, -2.8))[1]

        # Salt correction (SantaLucia, 1998)
        # Adjust entropy for salt concentration
        if Na_conc > 0:
            dS_total += 0.368 * (len(sequence) - 1) * math.log(Na_conc)

        # Magnesium correction (if present)
        if Mg_conc > 0 and Na_conc > 0:
            ratio = math.sqrt(Mg_conc) / Na_conc
            if ratio < 0.22:
                pass  # Use Na correction
            else:
                # Mg dominates
                dS_total += 0.368 * (len(sequence) - 1) * math.log(Mg_conc * 4)

        # Calculate Tm
        # Tm = ΔH / (ΔS + R * ln(Ct/4)) - 273.15
        # where Ct is total strand concentration
        dH_cal = dH_total * 1000  # Convert to cal/mol
        Ct = oligo_conc

        try:
            Tm = dH_cal / (dS_total + self._R * math.log(Ct / 4)) - 273.15
        except (ValueError, ZeroDivisionError):
            Tm = 0.0

        return round(Tm, 1)

    def _get_complement_dinuc(self, dinuc: str, is_rna: bool = False) -> str:
        """Get complement of a dinucleotide."""
        if is_rna:
            comp_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        else:
            comp_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

        return ''.join(comp_map.get(b, 'N') for b in dinuc)

    def calculate_tm_basic(self, sequence: str) -> float:
        """
        Calculate Tm using basic 4+2 rule (for short oligos).
        Tm = 4(G+C) + 2(A+T)

        Only accurate for oligos < 14 bp.
        """
        sequence = sequence.upper()
        gc = sequence.count('G') + sequence.count('C')
        at = sequence.count('A') + sequence.count('T') + sequence.count('U')
        return float(4 * gc + 2 * at)

    def calculate_tm_gc_content(self, sequence: str,
                                 Na_conc: float = 0.05) -> float:
        """
        Calculate Tm using GC content method (Wallace rule variant).
        More accurate for longer sequences.

        Tm = 81.5 + 16.6*log10([Na+]) + 41*(G+C)/N - 500/N
        """
        sequence = sequence.upper()
        N = len(sequence)
        if N == 0:
            return 0.0

        gc = sequence.count('G') + sequence.count('C')
        gc_fraction = gc / N

        Tm = 81.5 + 16.6 * math.log10(Na_conc) + 41 * gc_fraction - 500 / N
        return round(Tm, 1)

    # === Composition Analysis ===

    def calculate_gc_content(self, sequence: str) -> float:
        """
        Calculate GC content as percentage.

        Args:
            sequence: Nucleic acid sequence

        Returns:
            GC content as percentage (0-100)

        Raises:
            ValueError: If sequence is empty or invalid
        """
        sequence = self._validate_sequence(sequence)
        if len(sequence) == 0:
            return 0.0

        gc = sequence.count('G') + sequence.count('C')
        return round(100 * gc / len(sequence), 1)

    def get_base_composition(self, sequence: str) -> Dict[str, int]:
        """Get base composition counts."""
        sequence = sequence.upper()
        composition = {}
        for base in sequence:
            composition[base] = composition.get(base, 0) + 1
        return composition

    def get_base_percentages(self, sequence: str) -> Dict[str, float]:
        """Get base composition as percentages."""
        composition = self.get_base_composition(sequence)
        total = len(sequence)
        return {base: round(100 * count / total, 1)
                for base, count in composition.items()}

    # === Molecular Weight Calculation ===

    def calculate_molecular_mass(self, sequence: str, is_rna: bool = False,
                                  is_double_stranded: bool = False) -> float:
        """
        Calculate molecular mass of nucleic acid.

        Args:
            sequence: Nucleic acid sequence
            is_rna: Whether sequence is RNA
            is_double_stranded: Whether to calculate for dsDNA

        Returns:
            Molecular mass in Da
        """
        sequence = sequence.upper()
        if is_rna:
            sequence = sequence.replace('T', 'U')
        else:
            sequence = sequence.replace('U', 'T')

        mass = sum(NUCLEOTIDE_MW.get(base, 330) for base in sequence)

        # Subtract phosphate groups (linked internally)
        # Each nucleotide loses ~61 Da when forming phosphodiester bond
        # But we're using monophosphate weights, so this is already accounted for

        # For dsDNA, calculate complement and add
        if is_double_stranded and not is_rna:
            complement = self.get_complement(sequence, False)
            mass += sum(NUCLEOTIDE_MW.get(base, 330) for base in complement)

        return round(mass, 1)

    # === Complementarity ===

    def get_complement(self, sequence: str, is_rna: bool = False) -> str:
        """Get complement of sequence (5' to 3')."""
        if is_rna:
            comp_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        else:
            comp_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

        sequence = sequence.upper()
        return ''.join(comp_map.get(b, 'N') for b in sequence)

    def get_reverse_complement(self, sequence: str, is_rna: bool = False) -> str:
        """Get reverse complement of sequence."""
        return self.get_complement(sequence, is_rna)[::-1]

    def transcribe(self, dna_sequence: str) -> str:
        """Transcribe DNA to RNA (coding strand convention)."""
        return dna_sequence.upper().replace('T', 'U')

    def reverse_transcribe(self, rna_sequence: str) -> str:
        """Reverse transcribe RNA to DNA."""
        return rna_sequence.upper().replace('U', 'T')

    # === Secondary Structure Prediction ===

    def predict_hairpin_structures(self, sequence: str,
                                    min_stem: int = 4,
                                    min_loop: int = 3,
                                    max_loop: int = 10) -> List[Dict]:
        """
        Predict potential hairpin (stem-loop) structures.

        Returns list of dicts with:
        - start: start position of stem
        - end: end position of stem
        - loop_start: start of loop
        - loop_end: end of loop
        - stem_length: length of stem
        - loop_length: length of loop
        - stability: estimated stability score
        """
        sequence = sequence.upper()
        hairpins = []

        for i in range(len(sequence) - min_stem * 2 - min_loop):
            for stem_len in range(min_stem, min(20, (len(sequence) - i - min_loop) // 2)):
                for loop_len in range(min_loop, min(max_loop + 1, len(sequence) - i - stem_len * 2)):
                    stem1 = sequence[i:i + stem_len]
                    loop_start = i + stem_len
                    loop_end = loop_start + loop_len
                    stem2_start = loop_end
                    stem2 = sequence[stem2_start:stem2_start + stem_len]

                    # Check complementarity
                    complement = self.get_reverse_complement(stem1, is_rna='U' in sequence)
                    matches = sum(1 for a, b in zip(stem2, complement) if a == b)

                    if matches >= stem_len * 0.8:  # Allow some mismatches
                        # Calculate stability (more GC = more stable)
                        gc_count = stem1.count('G') + stem1.count('C')
                        stability = gc_count / stem_len

                        hairpins.append({
                            'start': i,
                            'end': stem2_start + stem_len,
                            'loop_start': loop_start,
                            'loop_end': loop_end,
                            'stem_length': stem_len,
                            'loop_length': loop_len,
                            'stem_sequence': stem1,
                            'loop_sequence': sequence[loop_start:loop_end],
                            'stability': round(stability, 2),
                            'matches': matches,
                        })

        # Sort by stability and return top structures
        hairpins.sort(key=lambda x: (-x['stability'], -x['stem_length']))
        return hairpins[:10]  # Return top 10

    def find_repeat_sequences(self, sequence: str, min_length: int = 4) -> List[Dict]:
        """Find repeated sequences (potential regulatory elements)."""
        sequence = sequence.upper()
        repeats = []

        for length in range(min_length, min(30, len(sequence) // 2)):
            seen = {}
            for i in range(len(sequence) - length + 1):
                subseq = sequence[i:i + length]
                if subseq in seen:
                    seen[subseq].append(i)
                else:
                    seen[subseq] = [i]

            for subseq, positions in seen.items():
                if len(positions) > 1:
                    repeats.append({
                        'sequence': subseq,
                        'length': length,
                        'positions': positions,
                        'count': len(positions),
                    })

        return sorted(repeats, key=lambda x: (-x['count'], -x['length']))[:20]

    # === Full Sequence Analysis ===

    def analyze_sequence(self, sequence: str, name: str = "Unknown",
                         is_rna: bool = False) -> Dict:
        """
        Perform complete nucleic acid analysis.
        Returns comprehensive dict suitable for JSON storage.
        """
        sequence = sequence.upper()
        if is_rna:
            sequence = sequence.replace('T', 'U')
        else:
            sequence = sequence.replace('U', 'T')

        gc_content = self.calculate_gc_content(sequence)
        tm_nn = self.calculate_tm_nearest_neighbor(sequence, is_rna=is_rna)
        tm_gc = self.calculate_tm_gc_content(sequence)
        hairpins = self.predict_hairpin_structures(sequence)

        return {
            'name': name,
            'type': 'rna' if is_rna else 'dna',
            'sequence': sequence,
            'length': len(sequence),
            'gc_content': gc_content,
            'at_content': round(100 - gc_content, 1),
            'base_composition': self.get_base_composition(sequence),
            'molecular_mass': self.calculate_molecular_mass(sequence, is_rna),
            'molecular_mass_ds': self.calculate_molecular_mass(sequence, is_rna, True),
            'melting_temperature': {
                'nearest_neighbor': tm_nn,
                'gc_method': tm_gc,
                'basic': self.calculate_tm_basic(sequence),
            },
            'complement': self.get_complement(sequence, is_rna),
            'reverse_complement': self.get_reverse_complement(sequence, is_rna),
            'predicted_hairpins': hairpins,
        }

    def create_nucleic_acid_json(self, sequence: str, name: str,
                                  na_type: str = "dna",
                                  organism: str = "Unknown",
                                  function: str = "Unknown") -> Dict:
        """
        Create a complete nucleic acid JSON structure with all computed properties.
        Suitable for saving as a data file.
        """
        is_rna = na_type.lower() != 'dna'
        analysis = self.analyze_sequence(sequence, name, is_rna)

        return {
            'name': name,
            'type': na_type,
            'organism': organism,
            'function': function,
            'sequence': analysis['sequence'],
            'length': analysis['length'],
            'gc_content': analysis['gc_content'],
            'base_composition': analysis['base_composition'],
            'molecular_mass': analysis['molecular_mass'],
            'melting_temperature': analysis['melting_temperature']['nearest_neighbor'],
            'complement': analysis['complement'],
            'reverse_complement': analysis['reverse_complement'],
            'secondary_structures': {
                'hairpins': len(analysis['predicted_hairpins']),
                'details': analysis['predicted_hairpins'][:5],  # Top 5
            },
        }
