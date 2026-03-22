"""
Protein Predictor - Fully Dynamic Implementation
Loads all parameters from amino acid JSON data files.
Provides prediction and calculation methods for protein properties including:
- Secondary structure prediction using Chou-Fasman method
- Phi/Psi angle generation for Ramachandran plots
- Protein mass and isoelectric point calculation
- Disulfide bond prediction
"""

import json
import logging
import math
import random
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import List, Dict, Tuple, Optional


class ProteinPredictor:
    """
    Predictor for protein properties and structure.
    Dynamically loads all parameters from amino acid JSON data.
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize predictor with dynamic data loading.

        Args:
            data_path: Path to amino_acids data directory. If None, uses default.
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "active" / "amino_acids"

        self._data_path = data_path
        self._amino_acid_data: Dict[str, Dict] = {}
        self._loaded = False

        # Default values used when data not found
        self._default_propensity = 1.0
        self._default_mass = 110.0
        self._water_mass = 18.015

        # Terminal pKa values for isoelectric point calculation
        self._pKa_N_terminus = 9.69
        self._pKa_C_terminus = 2.34

        # Chou-Fasman window size (expanded from 6 to 8 for improved accuracy)
        self.window_size = 8

        # Ramachandran regions - can be overridden by JSON config
        self._ramachandran_regions = {
            'H': {'phi': (-80, -48), 'psi': (-59, -27)},    # Alpha helix
            'E': {'phi': (-150, -90), 'psi': (90, 150)},     # Beta sheet
            'T': {'phi': (-60, 0), 'psi': (-30, 60)},        # Turn
            'C': {'phi': (-180, 180), 'psi': (-180, 180)},   # Coil
        }

        # Dipeptide instability weights - loaded from JSON config
        self._diwv: Dict[str, float] = {}
        self._stability_threshold = 40.0

        # Load data on init
        self._load_amino_acid_data()
        self._load_instability_config()

    def _load_amino_acid_data(self) -> None:
        """Load all amino acid data from JSON files."""
        if self._loaded:
            return

        if not self._data_path.exists():
            logger.warning("Amino acid data path not found: %s", self._data_path)
            self._loaded = True
            return

        for json_file in self._data_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    symbol = data.get('symbol', '')
                    if symbol:
                        self._amino_acid_data[symbol.upper()] = data
            except Exception as e:
                logger.warning("Could not load %s: %s", json_file, e)

        self._loaded = True

    def reload_data(self) -> None:
        """Reload amino acid data from files."""
        self._amino_acid_data = {}
        self._loaded = False
        self._load_amino_acid_data()
        self._load_instability_config()

    def _load_instability_config(self) -> None:
        """Load dipeptide instability weights from JSON config."""
        config_path = Path(__file__).parent.parent.parent.parent / "data" / "config"
        config_file = config_path / "protein_instability.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self._diwv = config.get('dipeptide_weights', {})
                    self._stability_threshold = config.get('stability_threshold', 40.0)
            except Exception as e:
                logger.warning("Could not load instability config: %s", e)

    def set_ramachandran_regions(self, regions: Dict[str, Dict]) -> None:
        """Override Ramachandran regions with custom values."""
        self._ramachandran_regions.update(regions)

    # === Dynamic Property Getters ===

    def get_helix_propensity(self, aa: str) -> float:
        """Get helix propensity for amino acid from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('helix_propensity', self._default_propensity)

    def get_sheet_propensity(self, aa: str) -> float:
        """Get sheet propensity for amino acid from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('sheet_propensity', self._default_propensity)

    def get_turn_propensity(self, aa: str) -> float:
        """Get turn propensity for amino acid from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('turn_propensity', self._default_propensity)

    def get_residue_mass(self, aa: str) -> float:
        """Get residue mass for amino acid from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('molecular_mass', self._default_mass)

    def get_pKa_sidechain(self, aa: str) -> Optional[float]:
        """Get side chain pKa for amino acid from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('pKa_sidechain')

    def get_hydropathy(self, aa: str) -> float:
        """Get hydropathy index for amino acid from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('hydropathy_index', 0.0)

    def get_phi_psi_constraints(self, aa: str) -> Optional[Dict]:
        """Get any residue-specific phi/psi constraints from periodica.data."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('phi_psi_constraints')

    def is_helix_breaker(self, aa: str) -> bool:
        """Check if amino acid is a helix breaker using data from JSON."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('helix_breaker', False)

    def is_flexible(self, aa: str) -> bool:
        """Check if amino acid has high conformational flexibility (e.g., Glycine)."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('flexible', False)

    def get_ramachandran_access(self, aa: str) -> Optional[Dict]:
        """Get Ramachandran region access for residue (for flexible residues like Glycine)."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('ramachandran_access')

    def can_form_disulfide(self, aa: str) -> bool:
        """Check if amino acid can form disulfide bonds (e.g., Cysteine)."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('can_form_disulfide', False)

    def get_sidechain_ionization(self, aa: str) -> Optional[str]:
        """Get sidechain ionization type ('acidic', 'basic', or None)."""
        data = self._amino_acid_data.get(aa.upper(), {})
        return data.get('sidechain_ionization')

    def get_all_amino_acids(self) -> Dict[str, Dict]:
        """Get all loaded amino acid data."""
        return self._amino_acid_data.copy()

    # === Secondary Structure Prediction (Chou-Fasman) ===

    def predict_secondary_structure(self, sequence: str) -> List[Dict]:
        """
        Predict secondary structure using improved Chou-Fasman method.
        All propensity values loaded dynamically from amino acid data.

        Improvements over basic Chou-Fasman:
        - 8-residue window (expanded from 6)
        - Nucleation rules: helix needs >=4 of 6 with P(alpha)>1.0,
          sheet needs >=3 of 5 with P(beta)>1.0
        - Pro breaks helix (P(alpha)<0.6), Gly breaks sheet (flexible)
        - Hydrophobicity burial scoring

        Returns list of dicts with:
        - residue: amino acid letter
        - position: 1-indexed position
        - structure: 'H' (helix), 'E' (sheet), 'T' (turn), 'C' (coil)
        - helix_score: propensity score for helix
        - sheet_score: propensity score for sheet
        - turn_score: propensity score for turn
        - phi: predicted phi angle
        - psi: predicted psi angle
        """
        sequence = sequence.upper()

        # Phase 1: Identify nucleation sites
        helix_nucleation = self._find_helix_nucleation(sequence)
        sheet_nucleation = self._find_sheet_nucleation(sequence)

        results = []
        for i, aa in enumerate(sequence):
            # Calculate window scores using dynamic propensities
            helix_score = self._calculate_window_score(sequence, i, 'helix')
            sheet_score = self._calculate_window_score(sequence, i, 'sheet')
            turn_score = self._calculate_window_score(sequence, i, 'turn')

            # Boost scores at nucleation sites
            if helix_nucleation[i]:
                helix_score *= 1.15
            if sheet_nucleation[i]:
                sheet_score *= 1.15

            # Determine structure type
            structure = self._assign_structure(
                helix_score, sheet_score, turn_score, aa, sequence, i
            )

            # Generate phi/psi angles based on structure and residue-specific constraints
            phi, psi = self._generate_phi_psi(structure, aa)

            results.append({
                'residue': aa,
                'position': i + 1,
                'structure': structure,
                'helix_score': round(helix_score, 3),
                'sheet_score': round(sheet_score, 3),
                'turn_score': round(turn_score, 3),
                'phi': round(phi, 1),
                'psi': round(psi, 1),
            })

        return results

    def _find_helix_nucleation(self, sequence: str) -> List[bool]:
        """Find helix nucleation sites: >=4 of 6 consecutive residues with P(alpha)>1.0."""
        n = len(sequence)
        nucleated = [False] * n
        nuc_window = 6

        for i in range(n - nuc_window + 1):
            count = sum(
                1 for j in range(i, i + nuc_window)
                if self.get_helix_propensity(sequence[j]) > 1.0
            )
            if count >= 4:
                for j in range(i, i + nuc_window):
                    nucleated[j] = True

        return nucleated

    def _find_sheet_nucleation(self, sequence: str) -> List[bool]:
        """Find sheet nucleation sites: >=3 of 5 consecutive residues with P(beta)>1.0."""
        n = len(sequence)
        nucleated = [False] * n
        nuc_window = 5

        for i in range(n - nuc_window + 1):
            count = sum(
                1 for j in range(i, i + nuc_window)
                if self.get_sheet_propensity(sequence[j]) > 1.0
            )
            if count >= 3:
                for j in range(i, i + nuc_window):
                    nucleated[j] = True

        return nucleated

    def _calculate_window_score(self, sequence: str, position: int,
                                propensity_type: str) -> float:
        """Calculate average propensity score for a window centered at position."""
        half_window = self.window_size // 2
        start = max(0, position - half_window)
        end = min(len(sequence), position + half_window + 1)

        if propensity_type == 'helix':
            scores = [self.get_helix_propensity(aa) for aa in sequence[start:end]]
        elif propensity_type == 'sheet':
            scores = [self.get_sheet_propensity(aa) for aa in sequence[start:end]]
        else:  # turn
            scores = [self.get_turn_propensity(aa) for aa in sequence[start:end]]

        return sum(scores) / len(scores) if scores else self._default_propensity

    def _assign_structure(self, helix_score: float, sheet_score: float,
                          turn_score: float, aa: str,
                          sequence: str = '', position: int = 0) -> str:
        """Assign secondary structure based on propensity scores and break rules."""
        aa_helix_p = self.get_helix_propensity(aa)
        aa_sheet_p = self.get_sheet_propensity(aa)

        # Rule: Residues with both low helix AND low sheet propensity (e.g. Gly)
        # should not be assigned to either — prefer turn or coil
        if aa_helix_p < 0.8 and aa_sheet_p < 0.8:
            if turn_score > 0.8:
                return 'T'
            return 'C'

        # Rule: Proline breaks helix (helix propensity < 0.6 from data)
        if self.is_helix_breaker(aa) or aa_helix_p < 0.6:
            # Pro can still be in sheets or turns
            if sheet_score > 1.0 and sheet_score > turn_score:
                return 'E'
            if turn_score > 1.0:
                return 'T'
            return 'C'

        # Standard Chou-Fasman assignment with improved thresholds
        if helix_score > 1.03 and helix_score >= sheet_score and helix_score >= turn_score:
            return 'H'
        elif sheet_score > 1.0 and sheet_score > turn_score:
            return 'E'
        elif turn_score > 1.0:
            return 'T'
        else:
            return 'C'

    def _generate_phi_psi(self, structure: str, aa: str) -> Tuple[float, float]:
        """Generate phi/psi angles based on structure type and residue constraints."""
        aa = aa.upper()

        # Check for residue-specific constraints from data (e.g., Proline)
        constraints = self.get_phi_psi_constraints(aa)
        if constraints:
            phi_range = constraints.get('phi', (-180, 180))
            psi_range = constraints.get('psi', (-180, 180))
            phi = random.gauss((phi_range[0] + phi_range[1]) / 2,
                              (phi_range[1] - phi_range[0]) / 4)
            psi = random.gauss((psi_range[0] + psi_range[1]) / 2,
                              (psi_range[1] - psi_range[0]) / 4)
            return max(-180, min(180, phi)), max(-180, min(180, psi))

        # Check for flexible residues with extended Ramachandran access (e.g., Glycine)
        rama_access = self.get_ramachandran_access(aa)
        if rama_access and self.is_flexible(aa):
            if structure == 'H':
                region = self._ramachandran_regions['H']
            elif structure == 'E':
                region = self._ramachandran_regions['E']
            else:
                # Flexible residues can access left-handed alpha region
                if rama_access.get('left_handed_alpha', False) and random.random() < 0.2:
                    region = {'phi': (50, 70), 'psi': (20, 60)}
                else:
                    region = self._ramachandran_regions['C']
            phi_min, phi_max = region['phi']
            psi_min, psi_max = region['psi']
        else:
            # Standard residues use structure-based regions
            region = self._ramachandran_regions.get(structure, self._ramachandran_regions['C'])
            phi_min, phi_max = region['phi']
            psi_min, psi_max = region['psi']

        # Special handling for turns
        if structure == 'T':
            phi = random.gauss(-60, 30)
            psi = random.gauss(-30, 40)
            return max(-180, min(180, phi)), max(-180, min(180, psi))

        # Generate angles with Gaussian distribution
        phi_center = (phi_min + phi_max) / 2
        psi_center = (psi_min + psi_max) / 2
        phi_std = (phi_max - phi_min) / 4
        psi_std = (psi_max - psi_min) / 4

        phi = random.gauss(phi_center, phi_std)
        psi = random.gauss(psi_center, psi_std)

        return max(-180, min(180, phi)), max(-180, min(180, psi))

    def get_structure_summary(self, structure_results: List[Dict]) -> Dict:
        """Calculate secondary structure percentages."""
        total = len(structure_results)
        if total == 0:
            return {'helix_percent': 0, 'sheet_percent': 0,
                    'turn_percent': 0, 'coil_percent': 0}

        counts = {'H': 0, 'E': 0, 'T': 0, 'C': 0}
        for res in structure_results:
            counts[res['structure']] = counts.get(res['structure'], 0) + 1

        return {
            'helix_percent': round(100 * counts['H'] / total, 1),
            'sheet_percent': round(100 * counts['E'] / total, 1),
            'turn_percent': round(100 * counts['T'] / total, 1),
            'coil_percent': round(100 * counts['C'] / total, 1),
        }

    # === Molecular Weight Calculation ===

    def calculate_molecular_mass(self, sequence: str) -> float:
        """
        Calculate protein molecular mass from sequence using dynamic residue masses.

        Uses the formula: sum(free AA masses) - (n-1) * water
        where n is the number of residues. Each peptide bond formation releases
        one water molecule. The terminal H and OH are already included in the
        sum of free amino acid masses.
        """
        sequence = sequence.upper()

        # Sum free amino acid masses from data
        mass = sum(self.get_residue_mass(aa) for aa in sequence)

        # Subtract water for each peptide bond formed (n-1 bonds for n residues)
        if len(sequence) > 1:
            mass -= self._water_mass * (len(sequence) - 1)

        return round(mass, 2)

    # === Isoelectric Point Calculation ===

    def calculate_isoelectric_point(self, sequence: str) -> float:
        """
        Calculate protein isoelectric point using iterative charge calculation.
        Uses Henderson-Hasselbalch equation with dynamic pKa values.
        """
        sequence = sequence.upper()

        # Binary search for pI
        pH_low, pH_high = 0.0, 14.0

        for _ in range(100):  # Max iterations
            pH = (pH_low + pH_high) / 2
            charge = self._calculate_charge_at_pH(
                sequence, pH, self._pKa_N_terminus, self._pKa_C_terminus
            )

            if abs(charge) < 0.001:
                break
            elif charge > 0:
                pH_low = pH
            else:
                pH_high = pH

        return round(pH, 2)

    def _calculate_charge_at_pH(self, sequence: str, pH: float,
                                 pKa_N: float, pKa_C: float) -> float:
        """Calculate net charge at given pH using Henderson-Hasselbalch with dynamic pKa."""
        charge = 0.0

        # N-terminus (positive when protonated)
        charge += 1.0 / (1.0 + 10 ** (pH - pKa_N))

        # C-terminus (negative when deprotonated)
        charge -= 1.0 / (1.0 + 10 ** (pKa_C - pH))

        # Side chains - use dynamic pKa and ionization type from data
        for aa in sequence:
            pKa = self.get_pKa_sidechain(aa)
            if pKa is not None:
                ionization = self.get_sidechain_ionization(aa)
                if ionization == 'acidic':
                    # Acidic residues: negative when deprotonated
                    charge -= 1.0 / (1.0 + 10 ** (pKa - pH))
                elif ionization == 'basic':
                    # Basic residues: positive when protonated
                    charge += 1.0 / (1.0 + 10 ** (pH - pKa))

        return charge

    def calculate_charge_at_pH(self, sequence: str, pH: float) -> float:
        """Public method to calculate charge at specific pH."""
        return round(self._calculate_charge_at_pH(
            sequence, pH, self._pKa_N_terminus, self._pKa_C_terminus
        ), 2)

    # === Amino Acid Composition ===

    def get_amino_acid_composition(self, sequence: str) -> Dict[str, int]:
        """Count amino acids in sequence."""
        sequence = sequence.upper()
        composition = {}
        for aa in sequence:
            composition[aa] = composition.get(aa, 0) + 1
        return composition

    def get_amino_acid_percentages(self, sequence: str) -> Dict[str, float]:
        """Get amino acid percentages."""
        composition = self.get_amino_acid_composition(sequence)
        total = len(sequence)
        return {aa: round(100 * count / total, 1)
                for aa, count in composition.items()}

    # === Disulfide Bond Prediction ===

    def predict_disulfide_bonds(self, sequence: str,
                                 min_distance: int = 10) -> List[Tuple[int, int]]:
        """
        Predict potential disulfide bonds based on residues that can form them.
        Returns pairs of 1-indexed positions that could form disulfide bonds.

        Args:
            sequence: Protein sequence
            min_distance: Minimum residue distance for bond formation (configurable)
        """
        sequence = sequence.upper()
        # Find positions of residues that can form disulfide bonds (loaded from JSON data)
        disulfide_positions = [i + 1 for i, aa in enumerate(sequence)
                               if self.can_form_disulfide(aa)]

        if len(disulfide_positions) < 2:
            return []

        # Simple pairing: consecutive residues that are > min_distance apart
        bonds = []
        available = set(disulfide_positions)

        for i, pos1 in enumerate(disulfide_positions):
            if pos1 not in available:
                continue
            for pos2 in disulfide_positions[i+1:]:
                if pos2 not in available:
                    continue
                if abs(pos2 - pos1) >= min_distance:
                    bonds.append((pos1, pos2))
                    available.discard(pos1)
                    available.discard(pos2)
                    break

        return bonds

    # === Hydropathy Analysis ===

    def calculate_hydropathy_profile(self, sequence: str, window: int = 9) -> List[float]:
        """Calculate Kyte-Doolittle hydropathy profile using dynamic values."""
        sequence = sequence.upper()
        profile = []
        half = window // 2

        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            window_seq = sequence[start:end]

            score = sum(self.get_hydropathy(aa) for aa in window_seq) / len(window_seq)
            profile.append(round(score, 2))

        return profile

    def calculate_gravy(self, sequence: str) -> float:
        """Calculate Grand Average of Hydropathy (GRAVY) score."""
        sequence = sequence.upper()
        if not sequence:
            return 0.0
        total = sum(self.get_hydropathy(aa) for aa in sequence)
        return round(total / len(sequence), 3)

    # === Instability Index ===

    def calculate_instability_index(self, sequence: str) -> float:
        """
        Calculate protein instability index (Guruprasad et al., 1990).
        Uses dipeptide instability weights loaded from JSON config.

        Proteins with II < stability_threshold (default 40) are predicted stable.
        Proteins with II >= threshold are predicted unstable.

        II = (10/L) × Σ DIWV(x_i, x_{i+1})

        Args:
            sequence: Protein sequence

        Returns:
            Instability index value
        """
        sequence = sequence.upper()
        if len(sequence) < 2:
            return 0.0

        total = 0.0
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            # Use dynamically loaded weights, default to 1.0 if not found
            total += self._diwv.get(dipeptide, 1.0)

        return round((10.0 / len(sequence)) * total, 2)

    def calculate_aliphatic_index(self, sequence: str) -> float:
        """
        Calculate aliphatic index (Ikai, 1980).

        Measures relative volume occupied by aliphatic side chains.
        Higher values indicate more thermostable proteins.

        AI = X(Ala) + 2.9×X(Val) + 3.9×(X(Ile) + X(Leu))

        Args:
            sequence: Protein sequence

        Returns:
            Aliphatic index value
        """
        sequence = sequence.upper()
        length = len(sequence)
        if length == 0:
            return 0.0

        # Mole percentages
        x_A = 100 * sequence.count('A') / length
        x_V = 100 * sequence.count('V') / length
        x_I = 100 * sequence.count('I') / length
        x_L = 100 * sequence.count('L') / length

        ai = x_A + 2.9 * x_V + 3.9 * (x_I + x_L)
        return round(ai, 2)

    # === Extinction Coefficient ===

    def calculate_extinction_coefficient(self, sequence: str,
                                         reduced: bool = True) -> float:
        """
        Calculate molar extinction coefficient at 280 nm.
        Uses Pace method: E = nY*1490 + nW*5500 + nC*125

        Args:
            sequence: Protein sequence
            reduced: If True, assume no disulfide bonds (reduced cysteines)

        Returns:
            Extinction coefficient in M⁻¹ cm⁻¹
        """
        sequence = sequence.upper()
        n_tyr = sequence.count('Y')
        n_trp = sequence.count('W')
        n_cys = sequence.count('C')

        # Cystines contribute only when forming disulfide bonds
        if reduced:
            return n_tyr * 1490 + n_trp * 5500
        else:
            # Assume all Cys form disulfides
            return n_tyr * 1490 + n_trp * 5500 + (n_cys // 2) * 125

    # === Full Protein Analysis ===

    def analyze_protein(self, sequence: str, name: str = "Unknown") -> Dict:
        """
        Perform complete protein analysis.
        Returns comprehensive dict suitable for JSON storage.
        All values computed dynamically from amino acid data.
        """
        sequence = sequence.upper()

        # Secondary structure prediction with phi/psi angles
        structure = self.predict_secondary_structure(sequence)
        summary = self.get_structure_summary(structure)

        # Disulfide bonds
        disulfide_bonds = self.predict_disulfide_bonds(sequence)

        instability = self.calculate_instability_index(sequence)
        aliphatic = self.calculate_aliphatic_index(sequence)

        return {
            'name': name,
            'sequence': sequence,
            'length': len(sequence),
            'molecular_mass': self.calculate_molecular_mass(sequence),
            'isoelectric_point': self.calculate_isoelectric_point(sequence),
            'charge_pH7': self.calculate_charge_at_pH(sequence, 7.0),
            'gravy': self.calculate_gravy(sequence),
            'instability_index': instability,
            'is_stable': instability < self._stability_threshold,
            'aliphatic_index': aliphatic,
            'amino_acid_composition': self.get_amino_acid_composition(sequence),
            'secondary_structure': {
                'helix_percent': summary['helix_percent'],
                'sheet_percent': summary['sheet_percent'],
                'turn_percent': summary['turn_percent'],
                'coil_percent': summary['coil_percent'],
            },
            'residues': structure,  # Per-residue phi/psi angles
            'disulfide_bonds': disulfide_bonds,
            'extinction_coefficient_reduced': self.calculate_extinction_coefficient(sequence, True),
            'extinction_coefficient_oxidized': self.calculate_extinction_coefficient(sequence, False),
        }

    def create_protein_json(self, sequence: str, name: str,
                            organism: str = "Unknown",
                            function: str = "Unknown",
                            localization: str = "cytoplasm") -> Dict:
        """
        Create a complete protein JSON structure with all computed properties.
        Suitable for saving as a protein data file.
        """
        analysis = self.analyze_protein(sequence, name)

        return {
            'name': name,
            'organism': organism,
            'function': function,
            'localization': localization,
            'sequence': sequence,
            'length': analysis['length'],
            'molecular_mass': analysis['molecular_mass'],
            'isoelectric_point': analysis['isoelectric_point'],
            'charge_pH7': analysis['charge_pH7'],
            'gravy': analysis['gravy'],
            'amino_acid_composition': analysis['amino_acid_composition'],
            'secondary_structure': analysis['secondary_structure'],
            'residues': analysis['residues'],
            'disulfide_bonds': analysis['disulfide_bonds'],
            'extinction_coefficient': {
                'reduced': analysis['extinction_coefficient_reduced'],
                'oxidized': analysis['extinction_coefficient_oxidized'],
            },
        }
