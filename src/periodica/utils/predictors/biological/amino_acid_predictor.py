"""
Amino acid property predictor using Henderson-Hasselbalch equations.

Fully dynamic implementation - loads all parameters from amino acid JSON data files.

Implements calculations for:
- Charge at any pH (Henderson-Hasselbalch)
- Isoelectric point (pI)
- Hydropathy calculations
- Secondary structure propensities
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from periodica.utils.predictors.base import BasePredictor


# =============================================================================
# Dynamic Data Loading
# =============================================================================

def _load_amino_acid_data(data_path: Optional[Path] = None) -> Dict[str, Dict]:
    """Load all amino acid data from JSON files."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent.parent / "data" / "active" / "amino_acids"

    amino_acid_data = {}
    if not data_path.exists():
        return amino_acid_data

    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                symbol = data.get('symbol', '')
                if symbol:
                    amino_acid_data[symbol.upper()] = data
        except Exception:
            pass

    return amino_acid_data


# Module-level data cache (loaded once on import)
_AMINO_ACID_DATA: Dict[str, Dict] = _load_amino_acid_data()


def reload_amino_acid_data(data_path: Optional[Path] = None) -> None:
    """Reload amino acid data from JSON files."""
    global _AMINO_ACID_DATA
    _AMINO_ACID_DATA = _load_amino_acid_data(data_path)


# =============================================================================
# Input/Output Dataclasses
# =============================================================================

@dataclass
class AminoAcidInput:
    """Input parameters for amino acid predictions."""
    symbol: str  # One-letter code
    pKa_carboxyl: float = 2.0  # pKa of alpha-carboxyl
    pKa_amino: float = 9.5  # pKa of alpha-amino
    pKa_sidechain: Optional[float] = None  # pKa of side chain (if ionizable)
    pH: float = 7.0  # pH for charge calculation

    def __post_init__(self):
        if not self.symbol or len(self.symbol) != 1:
            raise ValueError(f"Symbol must be a single character, got '{self.symbol}'")


@dataclass
class AminoAcidResult:
    """Results from amino acid predictions."""
    symbol: str
    name: str
    charge: float  # Net charge at specified pH
    isoelectric_point: float  # pH where net charge = 0
    hydropathy_index: float  # Kyte-Doolittle index
    category: str  # nonpolar, polar, acidic, basic
    helix_propensity: float  # Chou-Fasman helix propensity
    sheet_propensity: float  # Chou-Fasman sheet propensity
    turn_propensity: float  # Turn propensity

    # Optional properties
    molecular_mass: Optional[float] = None
    molecular_formula: Optional[str] = None


# =============================================================================
# Dynamic Data Accessors
# =============================================================================

def get_amino_acid_pKa(symbol: str) -> Dict[str, Optional[float]]:
    """Get pKa values for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return {
        'carboxyl': data.get('pKa_carboxyl', 2.0),
        'amino': data.get('pKa_amino', 9.5),
        'sidechain': data.get('pKa_sidechain'),
    }


def get_hydropathy_index(symbol: str) -> float:
    """Get hydropathy index for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('hydropathy_index', 0.0)


def get_helix_propensity(symbol: str) -> float:
    """Get helix propensity for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('helix_propensity', 1.0)


def get_sheet_propensity(symbol: str) -> float:
    """Get sheet propensity for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('sheet_propensity', 1.0)


def get_turn_propensity(symbol: str) -> float:
    """Get turn propensity for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('turn_propensity', 1.0)


def get_amino_acid_name(symbol: str) -> str:
    """Get name for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('name', 'Unknown')


def get_amino_acid_category(symbol: str) -> str:
    """Get category for an amino acid from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('category', 'unknown')


def get_sidechain_ionization(symbol: str) -> Optional[str]:
    """Get sidechain ionization type ('acidic', 'basic', or None) from JSON data."""
    data = _AMINO_ACID_DATA.get(symbol.upper(), {})
    return data.get('sidechain_ionization')


def is_known_amino_acid(symbol: str) -> bool:
    """Check if amino acid symbol is in loaded data."""
    return symbol.upper() in _AMINO_ACID_DATA


# =============================================================================
# Predictor Implementation
# =============================================================================

class AminoAcidPredictor(BasePredictor):
    """
    Predictor for amino acid properties using Henderson-Hasselbalch equations.

    Calculates:
    - Net charge at any pH using Henderson-Hasselbalch equation
    - Isoelectric point (pI)
    - Secondary structure propensities (Chou-Fasman)
    - Hydropathy index (Kyte-Doolittle)
    """

    @property
    def name(self) -> str:
        return "Henderson-Hasselbalch Amino Acid Predictor"

    @property
    def description(self) -> str:
        return (
            "Predicts amino acid charge, isoelectric point, and secondary structure "
            "propensities using Henderson-Hasselbalch equations and Chou-Fasman method."
        )

    def calculate_charge_at_pH(
        self,
        pH: float,
        pKa_carboxyl: float,
        pKa_amino: float,
        pKa_sidechain: Optional[float] = None,
        sidechain_is_acidic: bool = True
    ) -> float:
        """
        Calculate net charge at given pH using Henderson-Hasselbalch equation.

        Henderson-Hasselbalch equation:
        pH = pKa + log([A-]/[HA])

        For acidic groups: charge contribution = -1 / (1 + 10^(pKa - pH))
        For basic groups: charge contribution = +1 / (1 + 10^(pH - pKa))

        Args:
            pH: The pH at which to calculate charge
            pKa_carboxyl: pKa of alpha-carboxyl group (~2.0)
            pKa_amino: pKa of alpha-amino group (~9.5)
            pKa_sidechain: pKa of ionizable side chain (optional)
            sidechain_is_acidic: Whether side chain is acidic (True) or basic (False)

        Returns:
            Net charge at the specified pH
        """
        # Alpha-carboxyl group (acidic) - negative when deprotonated
        charge_carboxyl = -1.0 / (1 + 10**(pKa_carboxyl - pH))

        # Alpha-amino group (basic) - positive when protonated
        charge_amino = 1.0 / (1 + 10**(pH - pKa_amino))

        # Side chain contribution (if ionizable)
        charge_sidechain = 0.0
        if pKa_sidechain is not None:
            if sidechain_is_acidic:
                # Acidic side chains (D, E, C, Y) - negative when deprotonated
                charge_sidechain = -1.0 / (1 + 10**(pKa_sidechain - pH))
            else:
                # Basic side chains (K, R, H) - positive when protonated
                charge_sidechain = 1.0 / (1 + 10**(pH - pKa_sidechain))

        return charge_carboxyl + charge_amino + charge_sidechain

    def calculate_isoelectric_point(
        self,
        pKa_carboxyl: float,
        pKa_amino: float,
        pKa_sidechain: Optional[float] = None,
        sidechain_is_acidic: bool = True
    ) -> float:
        """
        Calculate the isoelectric point (pI) where net charge = 0.

        For amino acids without ionizable side chains:
        pI = (pKa_carboxyl + pKa_amino) / 2

        For amino acids with ionizable side chains, average the two pKa values
        closest to pH 7.

        Args:
            pKa_carboxyl: pKa of alpha-carboxyl group
            pKa_amino: pKa of alpha-amino group
            pKa_sidechain: pKa of ionizable side chain (optional)
            sidechain_is_acidic: Whether side chain is acidic or basic

        Returns:
            Isoelectric point (pI)
        """
        if pKa_sidechain is None:
            # No ionizable side chain - average carboxyl and amino
            return (pKa_carboxyl + pKa_amino) / 2

        pKa_values = [pKa_carboxyl, pKa_amino, pKa_sidechain]
        pKa_values.sort()

        if sidechain_is_acidic:
            # Acidic: average the two lowest pKa values
            return (pKa_values[0] + pKa_values[1]) / 2
        else:
            # Basic: average the two highest pKa values
            return (pKa_values[1] + pKa_values[2]) / 2

    def is_sidechain_acidic(self, symbol: str) -> bool:
        """Determine if amino acid has acidic (True) or basic (False) side chain using JSON data."""
        ionization = get_sidechain_ionization(symbol)
        return ionization == 'acidic'

    def predict(self, input_data: AminoAcidInput) -> AminoAcidResult:
        """
        Make amino acid property predictions.

        Args:
            input_data: AminoAcidInput with symbol and optional pKa values

        Returns:
            AminoAcidResult with calculated properties
        """
        symbol = input_data.symbol.upper()

        # Get pKa values from JSON data (fall back to provided values if not found)
        pka_data = get_amino_acid_pKa(symbol)
        if not is_known_amino_acid(symbol):
            pka_data = {
                'carboxyl': input_data.pKa_carboxyl,
                'amino': input_data.pKa_amino,
                'sidechain': input_data.pKa_sidechain
            }

        pKa_carboxyl = pka_data['carboxyl']
        pKa_amino = pka_data['amino']
        pKa_sidechain = pka_data['sidechain']
        sidechain_is_acidic = self.is_sidechain_acidic(symbol)

        # Calculate charge at specified pH
        charge = self.calculate_charge_at_pH(
            input_data.pH,
            pKa_carboxyl,
            pKa_amino,
            pKa_sidechain,
            sidechain_is_acidic
        )

        # Calculate isoelectric point
        pI = self.calculate_isoelectric_point(
            pKa_carboxyl,
            pKa_amino,
            pKa_sidechain,
            sidechain_is_acidic
        )

        return AminoAcidResult(
            symbol=symbol,
            name=get_amino_acid_name(symbol),
            charge=round(charge, 3),
            isoelectric_point=round(pI, 2),
            hydropathy_index=get_hydropathy_index(symbol),
            category=get_amino_acid_category(symbol),
            helix_propensity=get_helix_propensity(symbol),
            sheet_propensity=get_sheet_propensity(symbol),
            turn_propensity=get_turn_propensity(symbol),
        )

    def predict_from_symbol(self, symbol: str, pH: float = 7.0) -> AminoAcidResult:
        """
        Convenience method to predict properties from just symbol and pH.

        Args:
            symbol: One-letter amino acid code
            pH: pH for charge calculation (default 7.0)

        Returns:
            AminoAcidResult with calculated properties
        """
        input_data = AminoAcidInput(symbol=symbol, pH=pH)
        return self.predict(input_data)

    def calculate_sequence_charge(self, sequence: str, pH: float = 7.0) -> float:
        """
        Calculate net charge of an amino acid sequence at given pH.

        Args:
            sequence: Amino acid sequence (one-letter codes)
            pH: pH for charge calculation

        Returns:
            Total net charge of the sequence
        """
        total_charge = 0.0

        for i, aa in enumerate(sequence.upper()):
            if not is_known_amino_acid(aa):
                continue

            pka_data = get_amino_acid_pKa(aa)
            pKa_carboxyl = pka_data['carboxyl']
            pKa_amino = pka_data['amino']
            pKa_sidechain = pka_data['sidechain']

            # N-terminus (first residue) and C-terminus (last residue)
            # have free alpha amino and carboxyl groups
            if i == 0:
                # N-terminus: include alpha-amino charge
                total_charge += 1.0 / (1 + 10**(pH - pKa_amino))
            if i == len(sequence) - 1:
                # C-terminus: include alpha-carboxyl charge
                total_charge += -1.0 / (1 + 10**(pKa_carboxyl - pH))

            # Side chain charges (for all residues)
            if pKa_sidechain is not None:
                sidechain_is_acidic = self.is_sidechain_acidic(aa)
                if sidechain_is_acidic:
                    total_charge += -1.0 / (1 + 10**(pKa_sidechain - pH))
                else:
                    total_charge += 1.0 / (1 + 10**(pH - pKa_sidechain))

        return round(total_charge, 2)

    def get_confidence(self, input_data: Any, result: Any) -> float:
        """
        Calculate confidence level for prediction.

        Henderson-Hasselbalch is highly accurate for standard amino acids,
        so confidence is based on whether we have reference data.
        """
        if isinstance(input_data, AminoAcidInput):
            symbol = input_data.symbol.upper()
            if is_known_amino_acid(symbol):
                return 0.95  # High confidence for known amino acids
        return 0.7  # Lower confidence for unknown/custom inputs

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """Validate amino acid input data."""
        if not isinstance(input_data, AminoAcidInput):
            return False, f"Expected AminoAcidInput, got {type(input_data).__name__}"

        symbol = input_data.symbol.upper()
        if len(symbol) != 1:
            return False, f"Symbol must be single character, got '{symbol}'"

        if not is_known_amino_acid(symbol):
            return False, f"Unknown amino acid symbol: {symbol}"

        if not 0 <= input_data.pH <= 14:
            return False, f"pH must be between 0 and 14, got {input_data.pH}"

        return True, ""
