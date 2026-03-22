"""
Protocol definitions for the SOLID predictor framework.

This module defines structural subtyping protocols for various predictor types,
enabling loose coupling and dependency inversion in the prediction system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, runtime_checkable


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels for predictions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PredictionConfidence:
    """
    Data class representing the confidence of a prediction.

    Attributes:
        level: The qualitative confidence level
        score: Numerical confidence score between 0.0 and 1.0
        factors: Dictionary of factors contributing to the confidence
        uncertainty: Optional uncertainty margin for the prediction
        metadata: Additional metadata about the confidence calculation
    """
    level: ConfidenceLevel
    score: float
    factors: Dict[str, float] = field(default_factory=dict)
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence score is within bounds."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.score}")

    @classmethod
    def from_score(cls, score: float) -> "PredictionConfidence":
        """
        Create a PredictionConfidence from a numerical score.

        Args:
            score: Numerical confidence score between 0.0 and 1.0

        Returns:
            PredictionConfidence with appropriate level based on score
        """
        if score < 0.2:
            level = ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            level = ConfidenceLevel.LOW
        elif score < 0.6:
            level = ConfidenceLevel.MEDIUM
        elif score < 0.8:
            level = ConfidenceLevel.HIGH
        else:
            level = ConfidenceLevel.VERY_HIGH
        return cls(level=level, score=score)


# Type variable for prediction results
T = TypeVar("T")


@runtime_checkable
class PredictorProtocol(Protocol[T]):
    """
    Base protocol for all predictors in the framework.

    This protocol defines the minimal interface that all predictors must implement,
    following the Interface Segregation Principle (ISP) of SOLID.

    Type Parameters:
        T: The type of prediction result returned by the predictor
    """

    def predict(self, input_data: Dict[str, Any]) -> T:
        """
        Generate a prediction based on input data.

        Args:
            input_data: Dictionary containing input parameters for prediction

        Returns:
            Prediction result of type T
        """
        ...

    def get_confidence(self, input_data: Dict[str, Any]) -> PredictionConfidence:
        """
        Calculate the confidence level for a prediction.

        Args:
            input_data: Dictionary containing input parameters

        Returns:
            PredictionConfidence object with confidence metrics
        """
        ...

    def validate(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data before prediction.

        Args:
            input_data: Dictionary containing input parameters to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        ...


@runtime_checkable
class NuclearPredictorProtocol(Protocol):
    """
    Protocol for nuclear physics predictors.

    Defines methods for calculating nuclear properties such as binding energy,
    nuclear radius, and stability predictions.
    """

    def calculate_binding_energy(
        self,
        protons: int,
        neutrons: int
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the nuclear binding energy.

        Args:
            protons: Number of protons (atomic number Z)
            neutrons: Number of neutrons (N)

        Returns:
            Tuple of (binding_energy_MeV, confidence)
        """
        ...

    def calculate_radius(
        self,
        mass_number: int
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the nuclear radius using the empirical formula.

        Args:
            mass_number: Total number of nucleons (A = Z + N)

        Returns:
            Tuple of (radius_fm, confidence)
        """
        ...

    def check_stability(
        self,
        protons: int,
        neutrons: int
    ) -> Tuple[bool, PredictionConfidence]:
        """
        Check if a nucleus with given composition is stable.

        Args:
            protons: Number of protons
            neutrons: Number of neutrons

        Returns:
            Tuple of (is_stable, confidence)
        """
        ...


@runtime_checkable
class AtomicPredictorProtocol(Protocol):
    """
    Protocol for atomic property predictors.

    Defines methods for calculating atomic properties including electron
    configuration, ionization energy, and atomic radius.
    """

    def calculate_electron_configuration(
        self,
        atomic_number: int,
        charge: int = 0
    ) -> Tuple[str, PredictionConfidence]:
        """
        Calculate the electron configuration for an atom or ion.

        Args:
            atomic_number: The atomic number (Z)
            charge: Net charge of the species (default 0 for neutral atom)

        Returns:
            Tuple of (configuration_string, confidence)
        """
        ...

    def calculate_ionization_energy(
        self,
        atomic_number: int,
        ionization_level: int = 1
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the ionization energy for removing an electron.

        Args:
            atomic_number: The atomic number (Z)
            ionization_level: Which electron to remove (1 = first, 2 = second, etc.)

        Returns:
            Tuple of (ionization_energy_eV, confidence)
        """
        ...

    def calculate_atomic_radius(
        self,
        atomic_number: int,
        radius_type: str = "covalent"
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the atomic radius.

        Args:
            atomic_number: The atomic number (Z)
            radius_type: Type of radius ("covalent", "van_der_waals", "ionic")

        Returns:
            Tuple of (radius_pm, confidence)
        """
        ...


@runtime_checkable
class HadronPredictorProtocol(Protocol):
    """
    Protocol for hadron (quark composite) predictors.

    Defines methods for calculating properties of hadrons including
    mass, charge, and spin based on quark composition.
    """

    def calculate_mass(
        self,
        quark_composition: List[str]
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the hadron mass from quark composition.

        Args:
            quark_composition: List of quark flavors (e.g., ["u", "u", "d"] for proton)

        Returns:
            Tuple of (mass_MeV, confidence)
        """
        ...

    def calculate_charge(
        self,
        quark_composition: List[str]
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the total electric charge of the hadron.

        Args:
            quark_composition: List of quark flavors

        Returns:
            Tuple of (charge_e, confidence) where e is elementary charge
        """
        ...

    def calculate_spin(
        self,
        quark_composition: List[str],
        orbital_angular_momentum: int = 0
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the total spin of the hadron.

        Args:
            quark_composition: List of quark flavors
            orbital_angular_momentum: Orbital angular momentum quantum number L

        Returns:
            Tuple of (spin, confidence)
        """
        ...


@runtime_checkable
class AlloyPredictorProtocol(Protocol):
    """
    Protocol for alloy property predictors.

    Defines methods for predicting mechanical and physical properties
    of metal alloys based on composition.
    """

    def calculate_density(
        self,
        composition: Dict[str, float]
    ) -> Tuple[float, PredictionConfidence]:
        """
        Calculate the theoretical density of an alloy.

        Args:
            composition: Dictionary mapping element symbols to weight percentages

        Returns:
            Tuple of (density_g_cm3, confidence)
        """
        ...

    def calculate_strength(
        self,
        composition: Dict[str, float],
        treatment: Optional[str] = None
    ) -> Tuple[Dict[str, float], PredictionConfidence]:
        """
        Calculate strength properties of an alloy.

        Args:
            composition: Dictionary mapping element symbols to weight percentages
            treatment: Optional heat treatment or processing condition

        Returns:
            Tuple of (strength_properties_dict, confidence)
            where strength_properties_dict contains keys like:
            - "yield_strength_MPa"
            - "tensile_strength_MPa"
            - "hardness_HV"
        """
        ...


@runtime_checkable
class MoleculePredictorProtocol(Protocol):
    """
    Protocol for molecular geometry and structure predictors.

    Defines methods for determining molecular geometry and calculating
    bond angles based on molecular formula and structure.
    """

    def determine_geometry(
        self,
        molecular_formula: str,
        central_atom: Optional[str] = None
    ) -> Tuple[str, PredictionConfidence]:
        """
        Determine the molecular geometry using VSEPR theory.

        Args:
            molecular_formula: Chemical formula of the molecule
            central_atom: Optional specification of central atom

        Returns:
            Tuple of (geometry_name, confidence)
            Examples: "linear", "trigonal_planar", "tetrahedral", "octahedral"
        """
        ...

    def calculate_bond_angles(
        self,
        molecular_formula: str,
        bond_pair: Optional[Tuple[str, str, str]] = None
    ) -> Tuple[Dict[str, float], PredictionConfidence]:
        """
        Calculate bond angles in a molecule.

        Args:
            molecular_formula: Chemical formula of the molecule
            bond_pair: Optional tuple of (atom1, central_atom, atom2) for specific angle

        Returns:
            Tuple of (angles_dict, confidence)
            where angles_dict maps bond descriptions to angles in degrees
        """
        ...
