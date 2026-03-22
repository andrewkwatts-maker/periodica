"""
Derivation Chain Orchestrator Module
======================================

Orchestrates the full derivation chain from fundamental particles to complex structures:
  Quarks -> Hadrons -> Nuclei -> Atoms -> Molecules/Alloys

Provides a unified interface for predicting properties at any level of the hierarchy,
combining results from multiple domain-specific predictors with confidence tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .registry import PredictorRegistry
from .base import (
    NuclearInput,
    NuclearResult,
    AtomicInput,
    AtomicResult,
    HadronInput,
    HadronResult,
    AlloyInput,
    AlloyResult,
    MoleculeInput,
    MoleculeResult,
)

# Import Material types if available
try:
    from .material.material_predictor import MaterialInput, MaterialResult
    HAS_MATERIAL_PREDICTOR = True
except ImportError:
    HAS_MATERIAL_PREDICTOR = False


# Element symbol and name lookup tables
ELEMENT_SYMBOLS = [
    '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

ELEMENT_NAMES = [
    '', 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
    'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium',
    'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur',
    'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium',
    'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
    'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium',
    'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
    'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
    'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver',
    'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine',
    'Xenon', 'Cesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium',
    'Neodymium', 'Promethium', 'Samarium', 'Europium', 'Gadolinium',
    'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium',
    'Ytterbium', 'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten',
    'Rhenium', 'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
    'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine', 'Radon',
    'Francium', 'Radium', 'Actinium', 'Thorium', 'Protactinium',
    'Uranium', 'Neptunium', 'Plutonium', 'Americium', 'Curium',
    'Berkelium', 'Californium', 'Einsteinium', 'Fermium', 'Mendelevium',
    'Nobelium', 'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium',
    'Bohrium', 'Hassium', 'Meitnerium', 'Darmstadtium', 'Roentgenium',
    'Copernicium', 'Nihonium', 'Flerovium', 'Moscovium', 'Livermorium',
    'Tennessine', 'Oganesson'
]


@dataclass
class ChainResult:
    """
    Result container for the full derivation chain prediction.

    Contains all derived properties from hadron through atomic levels,
    along with confidence scores and predictor metadata.

    Attributes:
        Z: Atomic number (proton count)
        N: Neutron number
        element_symbol: Chemical symbol (e.g., 'H', 'Fe', 'Au')
        element_name: Full element name (e.g., 'Hydrogen', 'Iron', 'Gold')
        hadron_properties: Derived hadron properties (proton/neutron masses, charges)
        nuclear_properties: Nuclear properties (binding energy, radius, stability)
        atomic_properties: Atomic properties (electron config, ionization, radius)
        confidence: Confidence scores for each prediction domain
        predictors_used: Names of predictors used for each domain
    """
    Z: int
    N: int
    element_symbol: str
    element_name: str
    hadron_properties: Dict[str, Any]
    nuclear_properties: Dict[str, Any]
    atomic_properties: Dict[str, Any]
    confidence: Dict[str, float]
    predictors_used: Dict[str, str]


@dataclass
class MoleculeChainResult:
    """
    Result container for molecule predictions.

    Attributes:
        atoms: List of atoms in the molecule
        bonds: List of bonds between atoms
        geometry: Molecular geometry (e.g., 'tetrahedral', 'linear')
        bond_angles: Bond angles in degrees
        dipole_moment: Molecular dipole moment in Debye
        molecular_mass_amu: Total molecular mass in atomic mass units
        confidence: Confidence scores for predictions
        predictors_used: Names of predictors used
    """
    atoms: List[Dict[str, Any]]
    bonds: List[Dict[str, Any]]
    geometry: str
    bond_angles: List[float]
    dipole_moment: float
    molecular_mass_amu: float
    confidence: Dict[str, float]
    predictors_used: Dict[str, str]
    point_group: Optional[str] = None
    polarity: Optional[str] = None


@dataclass
class AlloyChainResult:
    """
    Result container for alloy predictions.

    Attributes:
        elements: List of element symbols in the alloy
        weight_fractions: Weight fractions for each element
        lattice: Crystal lattice structure
        density_g_cm3: Alloy density in g/cm^3
        melting_point_k: Melting point in Kelvin
        thermal_conductivity: Thermal conductivity in W/(m*K)
        electrical_resistivity: Electrical resistivity in Ohm*m
        confidence: Confidence scores for predictions
        predictors_used: Names of predictors used
    """
    elements: List[str]
    weight_fractions: List[float]
    lattice: str
    density_g_cm3: float
    melting_point_k: float
    thermal_conductivity: float
    electrical_resistivity: float
    confidence: Dict[str, float]
    predictors_used: Dict[str, str]
    hardness: Optional[float] = None
    yield_strength_mpa: Optional[float] = None


@dataclass
class MaterialChainResult:
    """
    Result container for material (engineering-level) predictions.

    Represents the observable/engineering level above alloys with full
    mechanical, thermal, and simulation properties.

    Attributes:
        name: Material name
        category: Material category (steel, aluminum, etc.)
        base_alloy: Underlying alloy composition
        youngs_modulus_gpa: Young's modulus in GPa
        shear_modulus_gpa: Shear modulus in GPa
        yield_strength_mpa: Yield strength in MPa
        ultimate_strength_mpa: Ultimate tensile strength in MPa
        hardness_hv: Vickers hardness
        fracture_toughness: Fracture toughness in MPa√m
        elongation_percent: Elongation at break
        fatigue_limit_mpa: Fatigue limit in MPa
        melting_point_k: Melting point in Kelvin
        thermal_conductivity: Thermal conductivity in W/(m·K)
        density_kg_m3: Density in kg/m³
        confidence: Confidence scores for predictions
        predictors_used: Names of predictors used
    """
    name: str
    category: str
    base_alloy: Optional[str]
    youngs_modulus_gpa: float
    shear_modulus_gpa: float
    yield_strength_mpa: float
    ultimate_strength_mpa: float
    hardness_hv: float
    fracture_toughness: float
    elongation_percent: float
    fatigue_limit_mpa: float
    melting_point_k: float
    thermal_conductivity: float
    density_kg_m3: float
    confidence: Dict[str, float]
    predictors_used: Dict[str, str]
    poissons_ratio: Optional[float] = None
    specific_heat_j_kgk: Optional[float] = None
    thermal_expansion_per_k: Optional[float] = None
    electrical_resistivity_ohm_m: Optional[float] = None


class DerivationChain:
    """
    Orchestrates the full derivation chain from quarks to atoms and beyond.

    This class coordinates multiple domain-specific predictors to derive
    properties at each level of the physical hierarchy:

    1. Hadron level: Derive proton/neutron properties from quark model
    2. Nuclear level: Calculate binding energy, radius, stability
    3. Atomic level: Determine electron configuration, ionization energy, radius
    4. Molecular level: Predict geometry, bond angles, polarity
    5. Alloy level: Predict mechanical and thermal properties

    Predictors are loaded lazily from the PredictorRegistry, allowing
    for flexible configuration and testing.

    Example:
        chain = DerivationChain(nuclear_predictor='semf', atomic_predictor='slater')
        result = chain.predict(Z=26)  # Predict iron properties
        print(result.element_symbol)  # 'Fe'
        print(result.nuclear_properties['binding_energy_mev'])
    """

    # Quark properties for hadron derivation (PDG 2024)
    QUARK_PROPERTIES = {
        'u': {'mass_mev': 2.16, 'charge': 2/3, 'spin': 0.5},
        'd': {'mass_mev': 4.67, 'charge': -1/3, 'spin': 0.5},
        's': {'mass_mev': 93.4, 'charge': -1/3, 'spin': 0.5},
        'c': {'mass_mev': 1270.0, 'charge': 2/3, 'spin': 0.5},
        'b': {'mass_mev': 4180.0, 'charge': -1/3, 'spin': 0.5},
        't': {'mass_mev': 172760.0, 'charge': 2/3, 'spin': 0.5},
    }

    # Constituent quark masses (fitted to hadron spectroscopy)
    CONSTITUENT_MASSES = {'u': 336.0, 'd': 340.0, 's': 486.0}

    # Reference masses for validation
    PROTON_MASS_MEV = 938.272
    NEUTRON_MASS_MEV = 939.565

    def __init__(
        self,
        nuclear_predictor: str = 'default',
        atomic_predictor: str = 'default',
        hadron_predictor: str = 'default',
        alloy_predictor: str = 'default',
        molecule_predictor: str = 'default',
        material_predictor: str = 'default'
    ):
        """
        Initialize the derivation chain with specified predictors.

        Args:
            nuclear_predictor: Name of nuclear predictor to use (default: 'default')
            atomic_predictor: Name of atomic predictor to use (default: 'default')
            hadron_predictor: Name of hadron predictor to use (default: 'default')
            alloy_predictor: Name of alloy predictor to use (default: 'default')
            molecule_predictor: Name of molecule predictor to use (default: 'default')
            material_predictor: Name of material predictor to use (default: 'default')
        """
        self._registry = PredictorRegistry()

        # Store predictor names for lazy loading
        self._nuclear_predictor_name = nuclear_predictor
        self._atomic_predictor_name = atomic_predictor
        self._hadron_predictor_name = hadron_predictor
        self._alloy_predictor_name = alloy_predictor
        self._molecule_predictor_name = molecule_predictor
        self._material_predictor_name = material_predictor

        # Lazy-loaded predictor instances
        self._nuclear_predictor = None
        self._atomic_predictor = None
        self._hadron_predictor = None
        self._alloy_predictor = None
        self._molecule_predictor = None
        self._material_predictor = None

    def _get_nuclear_predictor(self):
        """Lazily load and return the nuclear predictor."""
        if self._nuclear_predictor is None:
            if self._registry.has_predictor('nuclear', self._nuclear_predictor_name):
                self._nuclear_predictor = self._registry.get('nuclear', self._nuclear_predictor_name)
        return self._nuclear_predictor

    def _get_atomic_predictor(self):
        """Lazily load and return the atomic predictor."""
        if self._atomic_predictor is None:
            if self._registry.has_predictor('atomic', self._atomic_predictor_name):
                self._atomic_predictor = self._registry.get('atomic', self._atomic_predictor_name)
        return self._atomic_predictor

    def _get_hadron_predictor(self):
        """Lazily load and return the hadron predictor."""
        if self._hadron_predictor is None:
            if self._registry.has_predictor('hadron', self._hadron_predictor_name):
                self._hadron_predictor = self._registry.get('hadron', self._hadron_predictor_name)
        return self._hadron_predictor

    def _get_alloy_predictor(self):
        """Lazily load and return the alloy predictor."""
        if self._alloy_predictor is None:
            if self._registry.has_predictor('alloy', self._alloy_predictor_name):
                self._alloy_predictor = self._registry.get('alloy', self._alloy_predictor_name)
        return self._alloy_predictor

    def _get_molecule_predictor(self):
        """Lazily load and return the molecule predictor."""
        if self._molecule_predictor is None:
            if self._registry.has_predictor('molecule', self._molecule_predictor_name):
                self._molecule_predictor = self._registry.get('molecule', self._molecule_predictor_name)
        return self._molecule_predictor

    def _get_material_predictor(self):
        """Lazily load and return the material predictor."""
        if self._material_predictor is None:
            if self._registry.has_predictor('material', self._material_predictor_name):
                self._material_predictor = self._registry.get('material', self._material_predictor_name)
        return self._material_predictor

    def _estimate_stable_N(self, Z: int) -> int:
        """
        Estimate the neutron number for the most stable isotope.

        Uses empirical relationships based on the nuclear valley of stability.

        Args:
            Z: Atomic number (proton count)

        Returns:
            Estimated neutron number for most stable isotope
        """
        if Z <= 20:
            return Z  # Light nuclei: N ~ Z
        elif Z <= 82:
            return int(Z * 1.3)  # Medium nuclei: slight neutron excess
        else:
            return int(Z * 1.5)  # Heavy nuclei: larger neutron excess

    def _get_element_symbol(self, Z: int) -> str:
        """Get element symbol from atomic number."""
        if 1 <= Z < len(ELEMENT_SYMBOLS):
            return ELEMENT_SYMBOLS[Z]
        return f"E{Z}"

    def _get_element_name(self, Z: int) -> str:
        """Get element name from atomic number."""
        if 1 <= Z < len(ELEMENT_NAMES):
            return ELEMENT_NAMES[Z]
        return f"Element {Z}"

    def _derive_hadron_properties(self) -> Tuple[Dict[str, Any], float]:
        """
        Derive proton and neutron properties from the quark model.

        Uses constituent quark masses and calculates binding/hyperfine
        corrections to match experimental hadron masses.

        Returns:
            Tuple of (hadron_properties_dict, confidence_score)
        """
        hadron_predictor = self._get_hadron_predictor()

        if hadron_predictor is not None:
            # Use registered predictor if available
            proton_input = HadronInput(quarks=[
                {'flavor': 'u', 'color': 'r', 'spin': 0.5},
                {'flavor': 'u', 'color': 'g', 'spin': 0.5},
                {'flavor': 'd', 'color': 'b', 'spin': -0.5},
            ])
            neutron_input = HadronInput(quarks=[
                {'flavor': 'u', 'color': 'r', 'spin': 0.5},
                {'flavor': 'd', 'color': 'g', 'spin': 0.5},
                {'flavor': 'd', 'color': 'b', 'spin': -0.5},
            ])

            try:
                proton_result = hadron_predictor.predict(proton_input)
                neutron_result = hadron_predictor.predict(neutron_input)

                proton_confidence = hadron_predictor.get_confidence(proton_input, proton_result)
                neutron_confidence = hadron_predictor.get_confidence(neutron_input, neutron_result)

                return {
                    'proton_mass_mev': proton_result.mass_mev,
                    'proton_charge_e': proton_result.charge_e,
                    'proton_spin': proton_result.spin_hbar,
                    'neutron_mass_mev': neutron_result.mass_mev,
                    'neutron_charge_e': neutron_result.charge_e,
                    'neutron_spin': neutron_result.spin_hbar,
                }, (proton_confidence + neutron_confidence) / 2
            except Exception:
                pass  # Fall through to built-in calculation

        # Built-in constituent quark model calculation
        # Proton: uud composition
        proton_mass = (
            self.CONSTITUENT_MASSES['u'] * 2 +
            self.CONSTITUENT_MASSES['d'] -
            58.0  # Binding + hyperfine correction
        )
        proton_charge = 2 * self.QUARK_PROPERTIES['u']['charge'] + self.QUARK_PROPERTIES['d']['charge']

        # Neutron: udd composition
        neutron_mass = (
            self.CONSTITUENT_MASSES['u'] +
            self.CONSTITUENT_MASSES['d'] * 2 -
            58.0  # Binding + hyperfine correction
        )
        neutron_charge = self.QUARK_PROPERTIES['u']['charge'] + 2 * self.QUARK_PROPERTIES['d']['charge']

        # Calculate error percentages for confidence estimation
        proton_error = abs(proton_mass - self.PROTON_MASS_MEV) / self.PROTON_MASS_MEV
        neutron_error = abs(neutron_mass - self.NEUTRON_MASS_MEV) / self.NEUTRON_MASS_MEV
        avg_error = (proton_error + neutron_error) / 2
        confidence = max(0.0, 1.0 - avg_error * 10)  # Scale error to confidence

        return {
            'proton_mass_mev': proton_mass,
            'proton_charge_e': proton_charge,
            'proton_spin': 0.5,
            'neutron_mass_mev': neutron_mass,
            'neutron_charge_e': neutron_charge,
            'neutron_spin': 0.5,
            'proton_error_pct': proton_error * 100,
            'neutron_error_pct': neutron_error * 100,
        }, confidence

    def _derive_nuclear_properties(
        self,
        Z: int,
        N: int,
        hadron_props: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Derive nuclear properties from nucleon counts and hadron properties.

        Args:
            Z: Proton count
            N: Neutron count
            hadron_props: Previously derived hadron properties

        Returns:
            Tuple of (nuclear_properties_dict, confidence_score)
        """
        nuclear_predictor = self._get_nuclear_predictor()

        if nuclear_predictor is not None:
            try:
                nuclear_input = NuclearInput(Z=Z, N=N)
                result = nuclear_predictor.predict(nuclear_input)
                confidence = nuclear_predictor.get_confidence(nuclear_input, result)

                return {
                    'binding_energy_mev': result.binding_energy_mev,
                    'binding_per_nucleon_mev': result.binding_per_nucleon_mev,
                    'nuclear_radius_fm': result.nuclear_radius_fm,
                    'nuclear_mass_mev': result.nuclear_mass_mev,
                    'is_stable': result.is_stable,
                    'stability_reason': result.stability_reason,
                    'half_life_s': result.half_life_s,
                    'decay_modes': result.decay_modes,
                }, confidence
            except Exception:
                pass  # Fall through to built-in calculation

        # Built-in Semi-Empirical Mass Formula (SEMF) calculation
        A = Z + N

        # SEMF coefficients (MeV)
        a_v = 15.75   # Volume term
        a_s = 17.8    # Surface term
        a_c = 0.711   # Coulomb term
        a_a = 23.7    # Asymmetry term
        a_p = 11.2    # Pairing term

        # Calculate binding energy terms
        volume = a_v * A
        surface = -a_s * (A ** (2/3))
        coulomb = -a_c * Z * (Z - 1) / (A ** (1/3))
        asymmetry = -a_a * ((A - 2*Z) ** 2) / A

        # Pairing term
        if Z % 2 == 0 and N % 2 == 0:
            pairing = a_p / (A ** 0.5)  # Even-even
        elif Z % 2 == 1 and N % 2 == 1:
            pairing = -a_p / (A ** 0.5)  # Odd-odd
        else:
            pairing = 0  # Odd-A

        binding_energy = volume + surface + coulomb + asymmetry + pairing
        binding_per_nucleon = binding_energy / A if A > 0 else 0

        # Nuclear radius: R = r0 * A^(1/3)
        r0 = 1.25  # fm
        nuclear_radius = r0 * (A ** (1/3))

        # Nuclear mass
        proton_mass = hadron_props.get('proton_mass_mev', self.PROTON_MASS_MEV)
        neutron_mass = hadron_props.get('neutron_mass_mev', self.NEUTRON_MASS_MEV)
        nuclear_mass = Z * proton_mass + N * neutron_mass - binding_energy

        # Stability check (simplified)
        # Check if nucleus is near the valley of stability
        optimal_N = self._estimate_stable_N(Z)
        n_deviation = abs(N - optimal_N)
        is_stable = n_deviation <= max(1, Z // 10)

        if is_stable:
            stability_reason = "Near valley of stability"
        elif N < optimal_N:
            stability_reason = "Neutron deficient (beta+ decay likely)"
        else:
            stability_reason = "Neutron rich (beta- decay likely)"

        # Confidence decreases for superheavy elements
        z_factor = max(0.3, 1.0 - Z / 150)
        confidence = 0.95 * z_factor

        return {
            'binding_energy_mev': binding_energy,
            'binding_per_nucleon_mev': binding_per_nucleon,
            'nuclear_radius_fm': nuclear_radius,
            'nuclear_mass_mev': nuclear_mass,
            'is_stable': is_stable,
            'stability_reason': stability_reason,
            'half_life_s': None,
            'decay_modes': None,
        }, confidence

    def _derive_atomic_properties(
        self,
        Z: int,
        nuclear_props: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Derive atomic properties from atomic number and nuclear properties.

        Args:
            Z: Atomic number
            nuclear_props: Previously derived nuclear properties

        Returns:
            Tuple of (atomic_properties_dict, confidence_score)
        """
        atomic_predictor = self._get_atomic_predictor()

        if atomic_predictor is not None:
            try:
                nuclear_mass_mev = nuclear_props.get('nuclear_mass_mev')
                atomic_input = AtomicInput(Z=Z, nuclear_mass_mev=nuclear_mass_mev)
                result = atomic_predictor.predict(atomic_input)
                confidence = atomic_predictor.get_confidence(atomic_input, result)

                return {
                    'symbol': result.symbol,
                    'name': result.name,
                    'atomic_mass_u': result.atomic_mass_u if hasattr(result, 'atomic_mass_u') else None,
                    'electron_configuration': result.electron_configuration,
                    'ionization_energy_ev': result.ionization_energy_ev,
                    'atomic_radius_pm': result.atomic_radius_pm,
                    'period': result.period,
                    'group': result.group,
                    'block': result.block,
                    'electronegativity': result.electronegativity,
                }, confidence
            except Exception:
                pass  # Fall through to built-in calculation

        # Built-in atomic property calculation
        symbol = self._get_element_symbol(Z)
        name = self._get_element_name(Z)

        # Electron configuration using Aufbau principle
        config = self._calculate_electron_configuration(Z)

        # Periodic table position
        period, group, block = self._get_periodic_position(Z)

        # Ionization energy estimation (hydrogen-like model with screening)
        ionization = self._estimate_ionization_energy(Z)

        # Atomic radius estimation
        atomic_radius = self._estimate_atomic_radius(Z, period, block)

        # Electronegativity (Mulliken scale approximation)
        electronegativity = self._estimate_electronegativity(Z, ionization)

        # Atomic mass from nuclear mass
        nuclear_mass_mev = nuclear_props.get('nuclear_mass_mev', Z * self.PROTON_MASS_MEV)
        electron_mass_mev = 0.511
        atomic_mass_mev = nuclear_mass_mev + Z * electron_mass_mev
        atomic_mass_u = atomic_mass_mev / 931.494  # Convert MeV/c^2 to u

        # Confidence decreases for heavy elements
        z_factor = max(0.3, 1.0 - Z / 150)
        confidence = 0.85 * z_factor

        return {
            'symbol': symbol,
            'name': name,
            'atomic_mass_u': atomic_mass_u,
            'electron_configuration': config,
            'ionization_energy_ev': ionization,
            'atomic_radius_pm': atomic_radius,
            'period': period,
            'group': group,
            'block': block,
            'electronegativity': electronegativity,
        }, confidence

    def _calculate_electron_configuration(self, Z: int) -> str:
        """Calculate electron configuration using Aufbau principle."""
        orbital_order = [
            (1, 's', 2), (2, 's', 2), (2, 'p', 6), (3, 's', 2), (3, 'p', 6),
            (4, 's', 2), (3, 'd', 10), (4, 'p', 6), (5, 's', 2), (4, 'd', 10),
            (5, 'p', 6), (6, 's', 2), (4, 'f', 14), (5, 'd', 10), (6, 'p', 6),
            (7, 's', 2), (5, 'f', 14), (6, 'd', 10), (7, 'p', 6),
        ]

        config_parts = []
        remaining = Z

        for n, orbital, max_e in orbital_order:
            if remaining <= 0:
                break
            electrons = min(remaining, max_e)
            config_parts.append(f"{n}{orbital}{electrons}")
            remaining -= electrons

        return ' '.join(config_parts)

    def _get_periodic_position(self, Z: int) -> Tuple[int, int, str]:
        """Determine period, group, and block from atomic number."""
        # Period boundaries: 2, 10, 18, 36, 54, 86, 118
        period_ends = [2, 10, 18, 36, 54, 86, 118, 168]

        period = 1
        for i, end in enumerate(period_ends):
            if Z <= end:
                period = i + 1
                break

        # Simplified group and block determination
        if Z <= 2:
            group = Z
            block = 's'
        elif Z <= 4:
            group = Z - 2
            block = 's'
        elif Z <= 10:
            group = Z - 2 + 10  # Groups 13-18
            block = 'p'
        elif Z <= 12:
            group = Z - 10
            block = 's'
        elif Z <= 18:
            group = Z - 10 + 10
            block = 'p'
        elif Z <= 20:
            group = Z - 18
            block = 's'
        elif Z <= 30:
            group = Z - 18 - 2 + 3  # d-block groups 3-12
            block = 'd'
        elif Z <= 36:
            group = Z - 28 + 12
            block = 'p'
        else:
            # Simplified for heavier elements
            block = 'd' if Z <= 48 else 'p'
            group = ((Z - 1) % 18) + 1

        return period, group, block

    def _estimate_ionization_energy(self, Z: int) -> float:
        """Estimate first ionization energy using screened hydrogen model."""
        # Simplified effective nuclear charge
        if Z == 1:
            z_eff = 1.0
        elif Z <= 2:
            z_eff = Z - 0.3
        elif Z <= 10:
            z_eff = Z - 2 - 0.35 * (Z - 3)
        else:
            z_eff = Z * 0.3 + 2

        # Hydrogen-like ionization: IE = 13.6 * Z_eff^2 / n^2
        _, _, block = self._get_periodic_position(Z)
        period = 1
        for end in [2, 10, 18, 36, 54, 86, 118]:
            if Z <= end:
                break
            period += 1

        n_eff = period + 0.5 if block in ['d', 'f'] else period
        ie = 13.6 * (z_eff ** 2) / (n_eff ** 2)

        return min(ie, 25.0)  # Cap at reasonable maximum

    def _estimate_atomic_radius(self, Z: int, period: int, block: str) -> float:
        """Estimate atomic radius in picometers."""
        # Base radius decreases across period, increases down group
        base_radius = 50 + period * 30

        # Position in period affects radius
        period_position = (Z - 1) % 18 + 1
        radius = base_radius - period_position * 2

        # Block corrections
        if block == 'd':
            radius += 20
        elif block == 'f':
            radius += 30

        return max(30, min(radius, 300))

    def _estimate_electronegativity(self, Z: int, ionization_ev: float) -> Optional[float]:
        """Estimate electronegativity from ionization energy."""
        # Noble gases have no meaningful electronegativity
        noble_gases = {2, 10, 18, 36, 54, 86, 118}
        if Z in noble_gases:
            return None

        # Simplified Mulliken scale: EN proportional to IE
        en = ionization_ev * 0.15
        return min(max(en, 0.7), 4.0)

    def _combine_confidence(
        self,
        hadron_conf: float,
        nuclear_conf: float,
        atomic_conf: float
    ) -> Dict[str, float]:
        """Combine confidence scores from all prediction domains."""
        # Overall confidence is product of individual confidences
        overall = hadron_conf * nuclear_conf * atomic_conf

        return {
            'hadron': hadron_conf,
            'nuclear': nuclear_conf,
            'atomic': atomic_conf,
            'overall': overall,
        }

    def predict(self, Z: int, N: Optional[int] = None) -> ChainResult:
        """
        Predict all properties for an atom with Z protons and N neutrons.

        This method orchestrates the full derivation chain:
        1. Derive hadron (proton/neutron) properties from quark model
        2. Calculate nuclear properties using SEMF or registered predictor
        3. Calculate atomic properties using quantum mechanics or registered predictor
        4. Combine confidence scores

        Args:
            Z: Atomic number (proton count, must be >= 1)
            N: Neutron number (if None, estimates most stable isotope)

        Returns:
            ChainResult containing all derived properties

        Raises:
            ValueError: If Z < 1
        """
        if Z < 1:
            raise ValueError(f"Atomic number Z must be at least 1, got {Z}")

        # Estimate N if not provided
        if N is None:
            N = self._estimate_stable_N(Z)

        # Step 1: Derive hadron properties
        hadron_props, hadron_conf = self._derive_hadron_properties()

        # Step 2: Derive nuclear properties
        nuclear_props, nuclear_conf = self._derive_nuclear_properties(Z, N, hadron_props)

        # Step 3: Derive atomic properties
        atomic_props, atomic_conf = self._derive_atomic_properties(Z, nuclear_props)

        # Step 4: Combine confidence scores
        confidence = self._combine_confidence(hadron_conf, nuclear_conf, atomic_conf)

        # Get element symbol and name
        element_symbol = atomic_props.get('symbol', self._get_element_symbol(Z))
        element_name = atomic_props.get('name', self._get_element_name(Z))

        # Track which predictors were used
        predictors_used = {
            'hadron': self._hadron_predictor_name if self._hadron_predictor else 'builtin',
            'nuclear': self._nuclear_predictor_name if self._nuclear_predictor else 'builtin',
            'atomic': self._atomic_predictor_name if self._atomic_predictor else 'builtin',
        }

        return ChainResult(
            Z=Z,
            N=N,
            element_symbol=element_symbol,
            element_name=element_name,
            hadron_properties=hadron_props,
            nuclear_properties=nuclear_props,
            atomic_properties=atomic_props,
            confidence=confidence,
            predictors_used=predictors_used,
        )

    def predict_molecule(
        self,
        atoms: List[Dict[str, Any]],
        bonds: List[Dict[str, Any]]
    ) -> MoleculeChainResult:
        """
        Predict molecular properties from constituent atoms and bonds.

        Args:
            atoms: List of atom dictionaries with at least 'symbol' or 'Z' key
            bonds: List of bond dictionaries with 'atom1', 'atom2', 'order' keys

        Returns:
            MoleculeChainResult with predicted molecular properties

        Example:
            # Water molecule
            atoms = [
                {'symbol': 'O', 'Z': 8},
                {'symbol': 'H', 'Z': 1},
                {'symbol': 'H', 'Z': 1},
            ]
            bonds = [
                {'atom1': 0, 'atom2': 1, 'order': 1},
                {'atom1': 0, 'atom2': 2, 'order': 1},
            ]
            result = chain.predict_molecule(atoms, bonds)
        """
        molecule_predictor = self._get_molecule_predictor()

        if molecule_predictor is not None:
            try:
                mol_input = MoleculeInput(atoms=atoms, bonds=bonds)
                result = molecule_predictor.predict(mol_input)
                confidence_score = molecule_predictor.get_confidence(mol_input, result)

                return MoleculeChainResult(
                    atoms=atoms,
                    bonds=bonds,
                    geometry=result.geometry,
                    bond_angles=result.bond_angles,
                    dipole_moment=result.dipole_moment,
                    molecular_mass_amu=result.molecular_mass_amu,
                    confidence={'molecule': confidence_score, 'overall': confidence_score},
                    predictors_used={'molecule': self._molecule_predictor_name},
                    point_group=result.point_group,
                    polarity=result.polarity,
                )
            except Exception:
                pass  # Fall through to built-in

        # Built-in molecular property estimation
        # Calculate molecular mass
        total_mass = 0.0
        for atom in atoms:
            Z = atom.get('Z', 0)
            if Z == 0 and 'symbol' in atom:
                symbol = atom['symbol']
                Z = ELEMENT_SYMBOLS.index(symbol) if symbol in ELEMENT_SYMBOLS else 0
            # Approximate atomic mass as ~2*Z for light elements
            total_mass += Z * 2 if Z <= 20 else Z * 2.5

        # Estimate geometry from bond count and atom types
        num_bonds = len(bonds)
        num_atoms = len(atoms)

        if num_atoms == 2:
            geometry = 'linear'
            bond_angles = []
        elif num_atoms == 3:
            geometry = 'bent' if num_bonds == 2 else 'linear'
            bond_angles = [104.5] if geometry == 'bent' else [180.0]
        elif num_atoms == 4:
            geometry = 'trigonal_pyramidal' if num_bonds == 3 else 'tetrahedral'
            bond_angles = [107.0, 107.0, 107.0]
        elif num_atoms == 5:
            geometry = 'tetrahedral'
            bond_angles = [109.5, 109.5, 109.5, 109.5, 109.5, 109.5]
        else:
            geometry = 'complex'
            bond_angles = [109.5] * (num_atoms - 1)

        # Estimate dipole moment based on symmetry
        dipole = 0.0 if geometry in ['linear', 'tetrahedral'] else 1.5

        return MoleculeChainResult(
            atoms=atoms,
            bonds=bonds,
            geometry=geometry,
            bond_angles=bond_angles,
            dipole_moment=dipole,
            molecular_mass_amu=total_mass,
            confidence={'molecule': 0.6, 'overall': 0.6},
            predictors_used={'molecule': 'builtin'},
            point_group=None,
            polarity='polar' if dipole > 0 else 'nonpolar',
        )

    def predict_alloy(
        self,
        elements: List[str],
        weight_fractions: List[float],
        lattice: str = 'FCC'
    ) -> AlloyChainResult:
        """
        Predict alloy properties from constituent elements and composition.

        Args:
            elements: List of element symbols (e.g., ['Fe', 'C', 'Ni'])
            weight_fractions: Weight fractions for each element (must sum to ~1.0)
            lattice: Crystal lattice structure ('FCC', 'BCC', 'HCP', etc.)

        Returns:
            AlloyChainResult with predicted alloy properties

        Raises:
            ValueError: If elements and weight_fractions have different lengths

        Example:
            # Stainless steel 304
            result = chain.predict_alloy(
                elements=['Fe', 'Cr', 'Ni', 'C'],
                weight_fractions=[0.70, 0.18, 0.10, 0.02],
                lattice='FCC'
            )
        """
        if len(elements) != len(weight_fractions):
            raise ValueError("Elements and weight_fractions must have same length")

        alloy_predictor = self._get_alloy_predictor()

        if alloy_predictor is not None:
            try:
                components = [
                    {'element': elem, 'fraction': frac}
                    for elem, frac in zip(elements, weight_fractions)
                ]
                alloy_input = AlloyInput(components=components)
                result = alloy_predictor.predict(alloy_input)
                confidence_score = alloy_predictor.get_confidence(alloy_input, result)

                return AlloyChainResult(
                    elements=elements,
                    weight_fractions=weight_fractions,
                    lattice=lattice,
                    density_g_cm3=result.density_g_cm3,
                    melting_point_k=result.melting_point_k,
                    thermal_conductivity=result.thermal_conductivity,
                    electrical_resistivity=result.electrical_resistivity,
                    confidence={'alloy': confidence_score, 'overall': confidence_score},
                    predictors_used={'alloy': self._alloy_predictor_name},
                    hardness=result.hardness,
                    yield_strength_mpa=result.yield_strength_mpa,
                )
            except Exception:
                pass  # Fall through to built-in

        # Built-in alloy property estimation using rule of mixtures
        # Element property data (simplified)
        element_densities = {
            'Fe': 7.87, 'C': 2.26, 'Ni': 8.91, 'Cr': 7.19, 'Mn': 7.44,
            'Al': 2.70, 'Cu': 8.96, 'Zn': 7.14, 'Ti': 4.54, 'Co': 8.90,
            'W': 19.3, 'Mo': 10.28, 'V': 6.11, 'Si': 2.33, 'Mg': 1.74,
        }
        element_melting_points = {
            'Fe': 1811, 'C': 3823, 'Ni': 1728, 'Cr': 2180, 'Mn': 1519,
            'Al': 933, 'Cu': 1358, 'Zn': 693, 'Ti': 1941, 'Co': 1768,
            'W': 3695, 'Mo': 2896, 'V': 2183, 'Si': 1687, 'Mg': 923,
        }
        element_thermal_cond = {
            'Fe': 80, 'C': 140, 'Ni': 91, 'Cr': 94, 'Mn': 7.8,
            'Al': 237, 'Cu': 401, 'Zn': 116, 'Ti': 22, 'Co': 100,
            'W': 173, 'Mo': 139, 'V': 31, 'Si': 148, 'Mg': 156,
        }

        # Calculate weighted averages (rule of mixtures)
        total_weight = sum(weight_fractions)

        density = sum(
            element_densities.get(elem, 5.0) * frac
            for elem, frac in zip(elements, weight_fractions)
        ) / total_weight

        melting_point = sum(
            element_melting_points.get(elem, 1500) * frac
            for elem, frac in zip(elements, weight_fractions)
        ) / total_weight

        thermal_cond = sum(
            element_thermal_cond.get(elem, 50) * frac
            for elem, frac in zip(elements, weight_fractions)
        ) / total_weight

        # Electrical resistivity estimation (inverse of conductivity trend)
        resistivity = 1e-6 * (10 + len(elements) * 2)  # Ohm*m

        # Hardness estimation based on lattice and composition
        base_hardness = {'FCC': 150, 'BCC': 200, 'HCP': 180}.get(lattice, 150)
        hardness = base_hardness + len(elements) * 20  # Solid solution hardening

        # Yield strength estimation
        yield_strength = hardness * 2.5

        return AlloyChainResult(
            elements=elements,
            weight_fractions=weight_fractions,
            lattice=lattice,
            density_g_cm3=density,
            melting_point_k=melting_point,
            thermal_conductivity=thermal_cond,
            electrical_resistivity=resistivity,
            confidence={'alloy': 0.7, 'overall': 0.7},
            predictors_used={'alloy': 'builtin'},
            hardness=hardness,
            yield_strength_mpa=yield_strength,
        )

    def predict_material(
        self,
        alloy_data: Dict[str, Any],
        processing: Optional[Dict[str, Any]] = None,
        grain_size_um: float = 25.0,
        temperature_k: float = 293.15,
    ) -> MaterialChainResult:
        """
        Predict engineering material properties from alloy composition.

        This is the final level of the derivation chain:
        Quarks -> Hadrons -> Nuclei -> Atoms -> Molecules/Alloys -> Materials

        Materials represent the observable/engineering level with properties
        suitable for FEA/CFD simulation and structural analysis.

        Args:
            alloy_data: Alloy composition and properties dictionary with keys like:
                - 'name': Alloy name
                - 'category': Material category (steel, aluminum, etc.)
                - 'composition': List of {element, percentage} dicts
                - 'density_g_cm3': Base density
                - 'melting_point_K': Melting point
            processing: Optional processing conditions dict with keys like:
                - 'heat_treatment': Heat treatment description
                - 'cold_work_percent': Amount of cold work
            grain_size_um: Average grain size in micrometers (affects strength via Hall-Petch)
            temperature_k: Operating temperature in Kelvin

        Returns:
            MaterialChainResult with engineering-level properties

        Example:
            # Predict properties for carbon steel
            alloy = {
                'name': 'Carbon Steel 1045',
                'category': 'steel',
                'density_g_cm3': 7.85,
                'melting_point_K': 1773,
                'composition': [{'element': 'Fe', 'percentage': 98.5}, {'element': 'C', 'percentage': 0.45}]
            }
            result = chain.predict_material(alloy, grain_size_um=20.0)
            print(f"Yield Strength: {result.yield_strength_mpa} MPa")
        """
        if not HAS_MATERIAL_PREDICTOR:
            # Fall back to built-in estimation
            return self._builtin_material_prediction(alloy_data, processing, grain_size_um, temperature_k)

        material_predictor = self._get_material_predictor()

        if material_predictor is not None:
            try:
                mat_input = MaterialInput(
                    alloy_data=alloy_data,
                    processing=processing,
                    grain_size_um=grain_size_um,
                    temperature_K=temperature_k,
                )
                result = material_predictor.predict(mat_input)
                confidence = material_predictor.get_confidence(mat_input, result)

                return MaterialChainResult(
                    name=result.name,
                    category=result.category,
                    base_alloy=alloy_data.get('name'),
                    youngs_modulus_gpa=result.youngs_modulus_GPa,
                    shear_modulus_gpa=result.shear_modulus_GPa,
                    yield_strength_mpa=result.yield_strength_MPa,
                    ultimate_strength_mpa=result.ultimate_tensile_strength_MPa,
                    hardness_hv=result.hardness_HV,
                    fracture_toughness=result.fracture_toughness_MPa_sqrt_m,
                    elongation_percent=result.elongation_percent,
                    fatigue_limit_mpa=result.fatigue_limit_MPa,
                    melting_point_k=result.melting_point_K,
                    thermal_conductivity=result.thermal_conductivity_W_mK,
                    density_kg_m3=result.density_kg_m3,
                    confidence=confidence,
                    predictors_used={'material': self._material_predictor_name},
                    poissons_ratio=result.poissons_ratio,
                    specific_heat_j_kgk=result.specific_heat_J_kgK,
                    thermal_expansion_per_k=result.thermal_expansion_coeff_per_K,
                    electrical_resistivity_ohm_m=result.electrical_resistivity_Ohm_m,
                )
            except Exception:
                pass  # Fall through to built-in

        return self._builtin_material_prediction(alloy_data, processing, grain_size_um, temperature_k)

    def _builtin_material_prediction(
        self,
        alloy_data: Dict[str, Any],
        processing: Optional[Dict[str, Any]],
        grain_size_um: float,
        temperature_k: float,
    ) -> MaterialChainResult:
        """Built-in material property estimation using physics correlations."""
        import math

        name = alloy_data.get('name', 'Unknown Material')
        category = alloy_data.get('category', 'steel').lower()

        # Base material properties by category
        base_props = {
            'steel': {'E': 205, 'sigma_y': 250, 'rho': 7850, 'k': 50, 'elong': 20},
            'aluminum': {'E': 70, 'sigma_y': 100, 'rho': 2700, 'k': 167, 'elong': 15},
            'titanium': {'E': 114, 'sigma_y': 400, 'rho': 4430, 'k': 7, 'elong': 14},
            'copper': {'E': 117, 'sigma_y': 70, 'rho': 8940, 'k': 385, 'elong': 40},
            'nickel': {'E': 207, 'sigma_y': 400, 'rho': 8190, 'k': 11, 'elong': 20},
        }

        # Get base properties for category
        props = base_props.get(category, base_props['steel'])

        # Hall-Petch strengthening
        hall_petch_k = {'steel': 600, 'aluminum': 150, 'titanium': 400, 'copper': 110}
        k_hp = hall_petch_k.get(category, 300)
        sigma_hp = k_hp / math.sqrt(max(grain_size_um, 1.0))

        # Calculate properties
        youngs_modulus = alloy_data.get('youngs_modulus_GPa', props['E'])
        poissons_ratio = 0.29 if 'steel' in category else 0.33
        shear_modulus = youngs_modulus / (2 * (1 + poissons_ratio))

        yield_strength = alloy_data.get('yield_strength_MPa', props['sigma_y']) + sigma_hp
        ultimate_strength = yield_strength * 1.3
        hardness = yield_strength / 3.0  # Tabor relation

        # Thermal properties
        density = alloy_data.get('density_g_cm3', props['rho'] / 1000) * 1000
        thermal_cond = alloy_data.get('thermal_conductivity_W_mK', props['k'])
        melting_point = alloy_data.get('melting_point_K', 1773)

        # Fracture and fatigue
        fracture_toughness = yield_strength * 0.08  # Rough correlation
        fatigue_limit = ultimate_strength * 0.4
        elongation = props['elong']

        # Confidence based on category match
        confidence_val = 0.7 if category in base_props else 0.5

        return MaterialChainResult(
            name=name,
            category=category,
            base_alloy=alloy_data.get('name'),
            youngs_modulus_gpa=youngs_modulus,
            shear_modulus_gpa=shear_modulus,
            yield_strength_mpa=yield_strength,
            ultimate_strength_mpa=ultimate_strength,
            hardness_hv=hardness,
            fracture_toughness=fracture_toughness,
            elongation_percent=elongation,
            fatigue_limit_mpa=fatigue_limit,
            melting_point_k=melting_point,
            thermal_conductivity=thermal_cond,
            density_kg_m3=density,
            confidence={'material': confidence_val, 'overall': confidence_val},
            predictors_used={'material': 'builtin'},
            poissons_ratio=poissons_ratio,
        )
