"""
Abstract base classes and data structures for physics predictors.

This module defines the common interfaces and data structures used by all
predictor implementations in the Periodics application.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Input Dataclasses
# =============================================================================

@dataclass
class NuclearInput:
    """Input parameters for nuclear predictions."""
    Z: int  # Atomic number (proton count)
    N: int  # Neutron number

    def __post_init__(self):
        if self.Z < 0:
            raise ValueError(f"Atomic number Z must be non-negative, got {self.Z}")
        if self.N < 0:
            raise ValueError(f"Neutron number N must be non-negative, got {self.N}")

    @property
    def A(self) -> int:
        """Mass number (total nucleons)."""
        return self.Z + self.N


@dataclass
class AtomicInput:
    """Input parameters for atomic predictions."""
    Z: int  # Atomic number
    nuclear_mass_mev: Optional[float] = None  # Nuclear mass in MeV/c^2

    def __post_init__(self):
        if self.Z < 1:
            raise ValueError(f"Atomic number Z must be positive, got {self.Z}")


@dataclass
class HadronInput:
    """Input parameters for hadron predictions."""
    quarks: List[Dict[str, Any]] = field(default_factory=list)
    # Each quark dict should contain: flavor, color, spin, etc.

    def __post_init__(self):
        if not self.quarks:
            raise ValueError("Hadron must have at least one quark")


@dataclass
class AlloyInput:
    """Input parameters for alloy predictions."""
    components: List[Dict[str, Any]] = field(default_factory=list)
    # Each component dict: element_symbol, fraction, etc.
    temperature_k: float = 298.15  # Room temperature default

    def __post_init__(self):
        if not self.components:
            raise ValueError("Alloy must have at least one component")


@dataclass
class MoleculeInput:
    """Input parameters for molecule predictions."""
    atoms: List[Dict[str, Any]] = field(default_factory=list)
    bonds: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.atoms:
            raise ValueError("Molecule must have at least one atom")


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class NuclearResult:
    """Results from nuclear predictions."""
    binding_energy_mev: float  # Total binding energy in MeV
    binding_per_nucleon_mev: float  # Binding energy per nucleon in MeV
    nuclear_radius_fm: float  # Nuclear radius in femtometers
    nuclear_mass_mev: float  # Nuclear mass in MeV/c^2
    is_stable: bool  # Whether the nucleus is stable
    stability_reason: str  # Explanation of stability/instability

    # Optional additional properties
    half_life_s: Optional[float] = None
    decay_modes: Optional[List[str]] = None
    magic_numbers: Optional[List[int]] = None


@dataclass
class AtomicResult:
    """Results from atomic predictions."""
    symbol: str  # Element symbol (e.g., "H", "He")
    name: str  # Element name (e.g., "Hydrogen")
    electron_configuration: str  # e.g., "1s2 2s2 2p6"
    ionization_energy_ev: float  # First ionization energy in eV
    atomic_radius_pm: float  # Atomic radius in picometers
    period: int  # Period in periodic table
    group: int  # Group in periodic table
    block: str  # Block (s, p, d, f)

    # Optional additional properties
    electronegativity: Optional[float] = None
    electron_affinity_ev: Optional[float] = None
    oxidation_states: Optional[List[int]] = None


@dataclass
class HadronResult:
    """Results from hadron predictions."""
    name: str  # Hadron name (e.g., "proton", "pion+")
    mass_mev: float  # Mass in MeV/c^2
    charge_e: float  # Electric charge in units of e
    spin_hbar: float  # Spin in units of hbar
    baryon_number: int  # Baryon number (1 for baryons, 0 for mesons)

    # Optional additional properties
    isospin: Optional[float] = None
    strangeness: Optional[int] = None
    charm: Optional[int] = None
    lifetime_s: Optional[float] = None


@dataclass
class AlloyResult:
    """Results from alloy predictions."""
    density_g_cm3: float  # Density in g/cm^3
    melting_point_k: float  # Melting point in Kelvin
    thermal_conductivity: float  # W/(m*K)
    electrical_resistivity: float  # Ohm*m

    # Optional additional properties
    hardness: Optional[float] = None
    yield_strength_mpa: Optional[float] = None
    crystal_structure: Optional[str] = None


@dataclass
class MoleculeResult:
    """Results from molecule predictions."""
    geometry: str  # Molecular geometry (e.g., "tetrahedral")
    bond_angles: List[float]  # Bond angles in degrees
    dipole_moment: float  # Dipole moment in Debye
    molecular_mass_amu: float  # Molecular mass in amu

    # Optional additional properties
    point_group: Optional[str] = None
    polarity: Optional[str] = None
    hybridization: Optional[Dict[int, str]] = None


# =============================================================================
# Abstract Base Classes
# =============================================================================

class BasePredictor(ABC):
    """
    Abstract base class for all predictors.

    Defines the common interface that all predictor implementations must follow.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this predictor."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this predictor does."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """
        Make a prediction based on input data.

        Args:
            input_data: The input data for prediction (type depends on predictor)

        Returns:
            The prediction result (type depends on predictor)
        """
        pass

    @abstractmethod
    def get_confidence(self, input_data: Any, result: Any) -> float:
        """
        Calculate confidence level for a prediction.

        Args:
            input_data: The input data used for prediction
            result: The prediction result

        Returns:
            Confidence level between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """
        Validate input data before prediction.

        Args:
            input_data: The input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class BaseNuclearPredictor(BasePredictor):
    """
    Abstract base class for nuclear physics predictors.

    Implementations predict nuclear properties such as binding energy,
    nuclear radius, stability, and decay modes.
    """

    @abstractmethod
    def calculate_binding_energy(self, Z: int, N: int) -> float:
        """
        Calculate the nuclear binding energy.

        Args:
            Z: Atomic number (proton count)
            N: Neutron number

        Returns:
            Binding energy in MeV
        """
        pass

    @abstractmethod
    def calculate_radius(self, A: int) -> float:
        """
        Calculate the nuclear radius.

        Args:
            A: Mass number (total nucleons)

        Returns:
            Nuclear radius in femtometers (fm)
        """
        pass

    def predict(self, input_data: NuclearInput) -> NuclearResult:
        """Make nuclear predictions from input data."""
        raise NotImplementedError("Subclasses must implement predict()")

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """Validate nuclear input data."""
        if not isinstance(input_data, NuclearInput):
            return False, f"Expected NuclearInput, got {type(input_data).__name__}"
        if input_data.Z < 0:
            return False, "Atomic number Z must be non-negative"
        if input_data.N < 0:
            return False, "Neutron number N must be non-negative"
        if input_data.Z == 0 and input_data.N == 0:
            return False, "Nucleus must have at least one nucleon"
        return True, ""


class BaseAtomicPredictor(BasePredictor):
    """
    Abstract base class for atomic physics predictors.

    Implementations predict atomic properties such as electron configuration,
    ionization energy, and atomic radius.
    """

    @abstractmethod
    def calculate_electron_configuration(self, Z: int) -> str:
        """
        Calculate the electron configuration.

        Args:
            Z: Atomic number

        Returns:
            Electron configuration string (e.g., "1s2 2s2 2p6")
        """
        pass

    @abstractmethod
    def calculate_ionization_energy(self, Z: int) -> float:
        """
        Calculate the first ionization energy.

        Args:
            Z: Atomic number

        Returns:
            First ionization energy in eV
        """
        pass

    def predict(self, input_data: AtomicInput) -> AtomicResult:
        """Make atomic predictions from input data."""
        raise NotImplementedError("Subclasses must implement predict()")

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """Validate atomic input data."""
        if not isinstance(input_data, AtomicInput):
            return False, f"Expected AtomicInput, got {type(input_data).__name__}"
        if input_data.Z < 1:
            return False, "Atomic number Z must be positive"
        if input_data.Z > 118:
            return False, "Atomic number Z exceeds known elements"
        return True, ""


class BaseHadronPredictor(BasePredictor):
    """
    Abstract base class for hadron physics predictors.

    Implementations predict hadron properties from quark compositions,
    including mass, charge, spin, and quantum numbers.
    """

    @abstractmethod
    def derive_hadron(self, quarks: List[Dict[str, Any]]) -> HadronResult:
        """
        Derive hadron properties from quark composition.

        Args:
            quarks: List of quark dictionaries with flavor, color, spin

        Returns:
            HadronResult with derived properties
        """
        pass

    def predict(self, input_data: HadronInput) -> HadronResult:
        """Make hadron predictions from input data."""
        raise NotImplementedError("Subclasses must implement predict()")

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """Validate hadron input data."""
        if not isinstance(input_data, HadronInput):
            return False, f"Expected HadronInput, got {type(input_data).__name__}"
        if not input_data.quarks:
            return False, "Hadron must have at least one quark"
        quark_count = len(input_data.quarks)
        if quark_count not in (2, 3):
            return False, f"Hadrons must have 2 (meson) or 3 (baryon) quarks, got {quark_count}"
        return True, ""


class BaseAlloyPredictor(BasePredictor):
    """
    Abstract base class for alloy property predictors.

    Implementations predict alloy properties such as density,
    melting point, and mechanical properties from composition.
    """

    @abstractmethod
    def calculate_density(self, components: List[Dict[str, Any]]) -> float:
        """
        Calculate alloy density from composition.

        Args:
            components: List of component dictionaries with element and fraction

        Returns:
            Density in g/cm^3
        """
        pass

    def predict(self, input_data: AlloyInput) -> AlloyResult:
        """Make alloy predictions from input data."""
        raise NotImplementedError("Subclasses must implement predict()")

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """Validate alloy input data."""
        if not isinstance(input_data, AlloyInput):
            return False, f"Expected AlloyInput, got {type(input_data).__name__}"
        if not input_data.components:
            return False, "Alloy must have at least one component"
        total_fraction = sum(c.get('fraction', 0) for c in input_data.components)
        if abs(total_fraction - 1.0) > 0.01:
            return False, f"Component fractions must sum to 1.0, got {total_fraction}"
        return True, ""


class BaseMoleculePredictor(BasePredictor):
    """
    Abstract base class for molecule property predictors.

    Implementations predict molecular properties such as geometry,
    bond angles, and dipole moment from atomic composition and bonding.
    """

    @abstractmethod
    def determine_geometry(self, atoms: List[Dict], bonds: List[Dict]) -> str:
        """
        Determine molecular geometry from atoms and bonds.

        Args:
            atoms: List of atom dictionaries
            bonds: List of bond dictionaries

        Returns:
            Geometry name (e.g., "tetrahedral", "linear")
        """
        pass

    def predict(self, input_data: MoleculeInput) -> MoleculeResult:
        """Make molecule predictions from input data."""
        raise NotImplementedError("Subclasses must implement predict()")

    def validate(self, input_data: Any) -> Tuple[bool, str]:
        """Validate molecule input data."""
        if not isinstance(input_data, MoleculeInput):
            return False, f"Expected MoleculeInput, got {type(input_data).__name__}"
        if not input_data.atoms:
            return False, "Molecule must have at least one atom"
        return True, ""
