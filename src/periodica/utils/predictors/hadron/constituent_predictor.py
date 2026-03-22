"""
Constituent Quark Model Predictor for Hadrons.

This module implements the constituent quark model for predicting hadron properties
from quark compositions. Uses experimental masses for known hadrons and the
constituent quark model with QCD corrections for unknown combinations.
"""

from typing import Any, Dict, List, Tuple

from ..base import BaseHadronPredictor, HadronInput, HadronResult
from ..registry import register_predictor
from .hyperfine import get_constituent_mass, calculate_hyperfine_mass

# Quark charges in units of e
QUARK_CHARGES = {
    'u': 2/3,
    'd': -1/3,
    's': -1/3,
    'c': 2/3,
    'b': -1/3,
    't': 2/3,
}

# =============================================================================
# Known Hadrons Lookup Table (Experimental Values)
# =============================================================================

# Format: (sorted tuple of quark symbols) -> (mass_MeV, spin, name)
# Keys MUST be in sorted order for lookup to work
# Uses combining overline (U+0305) for antiquarks
KNOWN_HADRONS = {
    # Mesons (quark-antiquark) - sorted alphabetically
    ('d', 'u\u0305'): (139.57, 0.0, 'Pion-'),      # pi- (d u-bar)
    ('d\u0305', 'u'): (139.57, 0.0, 'Pion+'),      # pi+ (u d-bar)
    ('u', 'u\u0305'): (134.98, 0.0, 'Pion0'),      # pi0 (u u-bar)
    ('d', 'd\u0305'): (134.98, 0.0, 'Pion0'),      # pi0 (d d-bar)
    ('s\u0305', 'u'): (493.68, 0.0, 'Kaon+'),      # K+ (u s-bar)
    ('s', 'u\u0305'): (493.68, 0.0, 'Kaon-'),      # K- (s u-bar)
    ('d', 's\u0305'): (497.61, 0.0, 'Kaon0'),      # K0 (d s-bar)
    ('d\u0305', 's'): (497.61, 0.0, 'Kaon0bar'),   # K0-bar (s d-bar)
    ('c', 'c\u0305'): (3096.90, 1.0, 'J/Psi'),     # J/psi (c c-bar)
    ('b', 'b\u0305'): (9460.30, 1.0, 'Upsilon'),   # Upsilon (b b-bar)

    # Baryons (three quarks) - Ground states (spin 1/2)
    ('d', 'u', 'u'): (938.27, 0.5, 'Proton'),      # p (uud)
    ('d', 'd', 'u'): (939.57, 0.5, 'Neutron'),     # n (udd)
    ('d', 's', 'u'): (1115.68, 0.5, 'Lambda'),     # Lambda (uds)
    ('s', 'u', 'u'): (1189.37, 0.5, 'Sigma+'),     # Sigma+ (uus)
    ('d', 's', 's'): (1321.71, 0.5, 'Xi-'),        # Xi- (dss)
    ('s', 's', 'u'): (1314.86, 0.5, 'Xi0'),        # Xi0 (uss)
    ('s', 's', 's'): (1672.45, 1.5, 'Omega-'),     # Omega- (sss)
    ('d', 'd', 's'): (1197.45, 0.5, 'Sigma-'),     # Sigma- (dds)

    # Delta baryons with unique quark content (spin 3/2)
    ('u', 'u', 'u'): (1232.0, 1.5, 'Delta++'),     # Delta++ (uuu)
    ('d', 'd', 'd'): (1232.0, 1.5, 'Delta-'),      # Delta- (ddd)
}


def _normalize_quark_symbol(symbol: str) -> str:
    """
    Normalize quark symbol for lookup.

    Converts various representations (full names, anti-prefixes) to
    standard single-letter symbols with combining overline for antiquarks.

    Args:
        symbol: Quark symbol or name to normalize

    Returns:
        Normalized single-letter symbol (with overline for antiquarks)
    """
    # Handle various antiparticle notations
    if symbol.startswith('Anti-') or symbol.startswith('anti-'):
        base = symbol.split('-', 1)[1].lower()
        if base.startswith('up'):
            return 'u\u0305'
        elif base.startswith('down'):
            return 'd\u0305'
        elif base.startswith('strange'):
            return 's\u0305'
        elif base.startswith('charm'):
            return 'c\u0305'
        elif base.startswith('bottom'):
            return 'b\u0305'
        elif base.startswith('top'):
            return 't\u0305'

    # Handle full names
    name_lower = symbol.lower()
    if 'up' in name_lower and 'anti' not in name_lower:
        return 'u'
    elif 'down' in name_lower and 'anti' not in name_lower:
        return 'd'
    elif 'strange' in name_lower and 'anti' not in name_lower:
        return 's'
    elif 'charm' in name_lower and 'anti' not in name_lower:
        return 'c'
    elif 'bottom' in name_lower and 'anti' not in name_lower:
        return 'b'
    elif 'top' in name_lower and 'anti' not in name_lower:
        return 't'

    # Already a symbol
    return symbol


def _get_base_flavor(symbol: str) -> str:
    """
    Get the base flavor letter from a quark symbol.

    Strips the combining overline to get the base flavor.

    Args:
        symbol: Normalized quark symbol

    Returns:
        Base flavor letter (u, d, s, c, b, t)
    """
    return symbol.replace('\u0305', '')


def _is_antiquark(symbol: str) -> bool:
    """Check if the symbol represents an antiquark."""
    return '\u0305' in symbol


@register_predictor('hadron', 'constituent')
@register_predictor('hadron', 'default')
class ConstituentQuarkPredictor(BaseHadronPredictor):
    """
    Hadron predictor using the constituent quark model.

    This predictor calculates hadron properties from quark compositions using:
    1. Experimental masses for known hadrons (proton, neutron, pions, etc.)
    2. Constituent quark model with QCD corrections for unknown combinations

    The constituent quark model uses "dressed" quark masses that include
    the effects of the gluon field, making predictions more accurate than
    using bare (current) quark masses.
    """

    @property
    def name(self) -> str:
        """Return the name of this predictor."""
        return "Constituent Quark Model"

    @property
    def description(self) -> str:
        """Return a description of what this predictor does."""
        return (
            "Predicts hadron properties using the constituent quark model. "
            "Uses experimental masses for known hadrons and calculates "
            "properties for unknown quark combinations using dressed quark masses."
        )

    def derive_hadron(self, quarks: List[Dict[str, Any]]) -> HadronResult:
        """
        Derive hadron properties from quark composition.

        Calculates mass, charge, spin, baryon number, and other quantum numbers
        from the constituent quarks.

        Args:
            quarks: List of quark dictionaries with properties like:
                   - Name/name: Quark name (e.g., "Up Quark")
                   - Symbol/symbol: Quark symbol (e.g., "u")
                   - Charge_e/charge: Electric charge in units of e
                   - Mass_MeVc2/mass: Current quark mass in MeV/c^2
                   - BaryonNumber_B: Baryon number contribution (1/3)
                   - Isospin_I3: Third component of isospin

        Returns:
            HadronResult with derived properties
        """
        if not quarks:
            return HadronResult(
                name="Unknown",
                mass_mev=0.0,
                charge_e=0.0,
                spin_hbar=0.0,
                baryon_number=0
            )

        # Calculate conserved quantities from quarks
        total_charge = 0.0
        total_baryon_number = 0.0
        strangeness = 0
        charm = 0
        bottomness = 0

        # Build normalized quark symbols for lookup
        symbols = []

        for q in quarks:
            # Get charge
            charge = q.get('Charge_e', q.get('charge', 0))
            total_charge += charge

            # Get baryon number
            baryon_num = q.get('BaryonNumber_B', q.get('baryon_number', 1/3))
            total_baryon_number += baryon_num

            # Determine quark flavor and if it's an antiquark
            name = q.get('Name', q.get('name', ''))
            sym = q.get('Symbol', q.get('symbol', '?'))

            is_anti = 'Anti' in name or 'anti' in name or baryon_num < 0

            # Normalize to standard symbol
            if name:
                normalized = _normalize_quark_symbol(name)
            else:
                normalized = sym
                if is_anti and '\u0305' not in normalized:
                    normalized = normalized + '\u0305'

            symbols.append(normalized)

            # Calculate flavor quantum numbers
            base_flavor = _get_base_flavor(normalized)
            antiquark = _is_antiquark(normalized)

            if base_flavor == 's':
                strangeness += 1 if antiquark else -1
            elif base_flavor == 'c':
                charm += -1 if antiquark else 1
            elif base_flavor == 'b':
                bottomness += 1 if antiquark else -1

        # Round baryon number to integer
        baryon_number_int = round(total_baryon_number)

        # Determine hadron type
        num_quarks = len(quarks)
        if num_quarks == 3:
            hadron_type = "baryon"
        elif num_quarks == 2:
            hadron_type = "meson"
        else:
            hadron_type = "exotic"

        # Try to look up known hadron by quark content
        lookup_key = tuple(sorted(symbols))
        known = KNOWN_HADRONS.get(lookup_key)

        if known:
            mass_mev = known[0]
            spin_hbar = known[1]
            hadron_name = known[2]
        else:
            # Calculate mass using constituent quark model
            mass_mev = self._calculate_constituent_mass(quarks, hadron_type)

            # Default spin based on particle type
            if hadron_type == "baryon":
                spin_hbar = 0.5  # Most baryons are spin-1/2
            elif hadron_type == "meson":
                spin_hbar = 0.0  # Pseudoscalar mesons are spin-0
            else:
                spin_hbar = 0.0

            hadron_name = self._generate_hadron_name(symbols, hadron_type)

        return HadronResult(
            name=hadron_name,
            mass_mev=mass_mev,
            charge_e=total_charge,
            spin_hbar=spin_hbar,
            baryon_number=baryon_number_int,
            strangeness=strangeness,
            charm=charm
        )

    def _calculate_constituent_mass(
        self,
        quarks: List[Dict[str, Any]],
        hadron_type: str,
        total_spin: float = None
    ) -> float:
        """
        Calculate hadron mass using the DRG hyperfine splitting model.

        Reads constituent quark masses from quark data sheets (ConstituentMass_MeV)
        and applies the De Rujula-Georgi-Glashow color-magnetic hyperfine
        interaction to split masses based on spin alignment.

        Args:
            quarks: List of quark data dictionaries (from quark data sheets)
            hadron_type: Type of hadron ("baryon", "meson", "exotic")
            total_spin: Total spin of the hadron (None = ground state)

        Returns:
            Calculated mass in MeV/c^2
        """
        constituent_masses = [get_constituent_mass(q) for q in quarks]
        return calculate_hyperfine_mass(constituent_masses, hadron_type, total_spin)

    def _generate_hadron_name(
        self,
        symbols: List[str],
        hadron_type: str
    ) -> str:
        """
        Generate a descriptive name for an unknown hadron.

        Args:
            symbols: List of normalized quark symbols
            hadron_type: Type of hadron

        Returns:
            Generated hadron name
        """
        quark_string = ''.join(symbols)
        return f"Unknown {hadron_type.capitalize()} ({quark_string})"

    def predict(self, input_data: HadronInput) -> HadronResult:
        """
        Make hadron predictions from input data.

        Args:
            input_data: HadronInput containing quark composition

        Returns:
            HadronResult with predicted properties
        """
        return self.derive_hadron(input_data.quarks)

    def get_confidence(self, input_data: Any, result: Any) -> float:
        """
        Calculate confidence level for a prediction.

        Higher confidence for known hadrons (experimental values),
        lower confidence for calculated predictions.

        Args:
            input_data: The input data used for prediction
            result: The prediction result

        Returns:
            Confidence level between 0.0 and 1.0
        """
        if not isinstance(input_data, HadronInput):
            return 0.0

        if not isinstance(result, HadronResult):
            return 0.0

        # Check if this is a known hadron
        quarks = input_data.quarks
        symbols = []

        for q in quarks:
            name = q.get('Name', q.get('name', ''))
            sym = q.get('Symbol', q.get('symbol', '?'))

            if name:
                normalized = _normalize_quark_symbol(name)
            else:
                normalized = sym

            symbols.append(normalized)

        lookup_key = tuple(sorted(symbols))

        if lookup_key in KNOWN_HADRONS:
            # High confidence for known hadrons (experimental values)
            return 0.95
        else:
            # Lower confidence for constituent quark model calculations
            # Confidence decreases with heavier quarks
            base_confidence = 0.7

            for sym in symbols:
                base_flavor = _get_base_flavor(sym)
                if base_flavor in ('c', 'b', 't'):
                    base_confidence -= 0.1

            return max(0.3, base_confidence)

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """
        Validate input data before prediction.

        Args:
            input_data: The input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(input_data, HadronInput):
            return False, f"Expected HadronInput, got {type(input_data).__name__}"

        if not input_data.quarks:
            return False, "Hadron must have at least one quark"

        quark_count = len(input_data.quarks)
        if quark_count not in (2, 3):
            return False, f"Hadrons must have 2 (meson) or 3 (baryon) quarks, got {quark_count}"

        return True, ""
