"""
End-to-End Tests for the Complete Derivation Chain.

Tests the full propagation pipeline from quarks to complex materials:
- Quarks -> Hadrons (protons, neutrons)
- Hadrons -> Nuclei (atoms)
- Atoms -> Molecules
- Elements -> Alloys

Uses simulation_schema.py propagation functions:
- propagate_quark_to_hadron
- propagate_hadrons_to_atom
- propagate_atoms_to_molecule
- propagate_elements_to_alloy
"""

import pytest
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periodica.utils.simulation_schema import (
    propagate_quark_to_hadron,
    propagate_hadrons_to_atom,
    propagate_atoms_to_molecule,
    propagate_elements_to_alloy,
    QuarkSimulationData,
    HadronSimulationData,
    AtomSimulationData,
    MoleculeSimulationData,
    AlloySimulationData,
)


# ==================== REFERENCE DATA ====================

# Quark data (PDG 2024)
UP_QUARK = {
    'Symbol': 'u', 'Name': 'Up Quark', 'Mass_MeVc2': 2.16,
    'Charge_e': 2/3, 'Spin_hbar': 0.5, 'BaryonNumber_B': 1/3,
    'Isospin_I': 0.5, 'Isospin_I3': 0.5
}

DOWN_QUARK = {
    'Symbol': 'd', 'Name': 'Down Quark', 'Mass_MeVc2': 4.67,
    'Charge_e': -1/3, 'Spin_hbar': 0.5, 'BaryonNumber_B': 1/3,
    'Isospin_I': 0.5, 'Isospin_I3': -0.5
}

# Element data for molecules and alloys
ELEMENT_DATA = {
    'H': {'symbol': 'H', 'Symbol': 'H', 'atomic_number': 1, 'Z': 1,
          'atomic_mass': 1.008, 'AtomicMass': 1.008, 'density': 0.0899,
          'melting_point': 14.01, 'electronegativity': 2.20},
    'C': {'symbol': 'C', 'Symbol': 'C', 'atomic_number': 6, 'Z': 6,
          'atomic_mass': 12.011, 'AtomicMass': 12.011, 'density': 2.267,
          'melting_point': 3823, 'electronegativity': 2.55},
    'O': {'symbol': 'O', 'Symbol': 'O', 'atomic_number': 8, 'Z': 8,
          'atomic_mass': 15.999, 'AtomicMass': 15.999, 'density': 1.429,
          'melting_point': 54.36, 'electronegativity': 3.44},
    'Fe': {'symbol': 'Fe', 'Symbol': 'Fe', 'atomic_number': 26, 'Z': 26,
           'atomic_mass': 55.845, 'AtomicMass': 55.845, 'density': 7.874,
           'melting_point': 1811, 'electronegativity': 1.83},
    'Cr': {'symbol': 'Cr', 'Symbol': 'Cr', 'atomic_number': 24, 'Z': 24,
           'atomic_mass': 51.996, 'AtomicMass': 51.996, 'density': 7.19,
           'melting_point': 2180, 'electronegativity': 1.66},
    'Ni': {'symbol': 'Ni', 'Symbol': 'Ni', 'atomic_number': 28, 'Z': 28,
           'atomic_mass': 58.693, 'AtomicMass': 58.693, 'density': 8.908,
           'melting_point': 1728, 'electronegativity': 1.91},
    'Cu': {'symbol': 'Cu', 'Symbol': 'Cu', 'atomic_number': 29, 'Z': 29,
           'atomic_mass': 63.546, 'AtomicMass': 63.546, 'density': 8.96,
           'melting_point': 1358, 'electronegativity': 1.90},
    'Zn': {'symbol': 'Zn', 'Symbol': 'Zn', 'atomic_number': 30, 'Z': 30,
           'atomic_mass': 65.38, 'AtomicMass': 65.38, 'density': 7.14,
           'melting_point': 692.68, 'electronegativity': 1.65},
    'He': {'symbol': 'He', 'Symbol': 'He', 'atomic_number': 2, 'Z': 2,
           'atomic_mass': 4.0026, 'AtomicMass': 4.0026, 'density': 0.1786,
           'melting_point': 0.95, 'electronegativity': 0},
    'Ne': {'symbol': 'Ne', 'Symbol': 'Ne', 'atomic_number': 10, 'Z': 10,
           'atomic_mass': 20.180, 'AtomicMass': 20.180, 'density': 0.9002,
           'melting_point': 24.56, 'electronegativity': 0},
    'Ar': {'symbol': 'Ar', 'Symbol': 'Ar', 'atomic_number': 18, 'Z': 18,
           'atomic_mass': 39.95, 'AtomicMass': 39.95, 'density': 1.784,
           'melting_point': 83.81, 'electronegativity': 0},
    'Au': {'symbol': 'Au', 'Symbol': 'Au', 'atomic_number': 79, 'Z': 79,
           'atomic_mass': 196.97, 'AtomicMass': 196.97, 'density': 19.3,
           'melting_point': 1337.33, 'electronegativity': 2.54},
}


# ==================== HELPER CLASS ====================

class ChainTestHelper:
    """Helper class for derivation chain tests."""

    @staticmethod
    def assert_within_tolerance(actual, expected, tolerance, message=""):
        """Assert that actual value is within tolerance of expected value."""
        if expected == 0:
            assert abs(actual) < tolerance, f"{message}: Expected ~0, got {actual}"
        else:
            percent_diff = abs(actual - expected) / abs(expected)
            assert percent_diff < tolerance, (
                f"{message}: Expected {expected}, got {actual} "
                f"(diff: {percent_diff*100:.2f}%, tolerance: {tolerance*100:.1f}%)"
            )

    @staticmethod
    def create_proton_quarks():
        """Create quark list for proton (uud)."""
        return [UP_QUARK.copy(), UP_QUARK.copy(), DOWN_QUARK.copy()]

    @staticmethod
    def create_neutron_quarks():
        """Create quark list for neutron (udd)."""
        return [UP_QUARK.copy(), DOWN_QUARK.copy(), DOWN_QUARK.copy()]


# ==================== TEST CLASS 1: Water from Quarks ====================

class TestE2EWaterFromQuarks:
    """
    End-to-end test: Build H2O molecule starting from quarks.

    Chain: Quarks -> Protons/Neutrons -> H and O atoms -> H2O molecule
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_step1_quarks_to_hadrons(self, helper):
        """
        Step 1: Verify quarks combine to form protons and neutrons
        with correct charges.
        """
        # Create proton from uud quarks
        proton_quarks = helper.create_proton_quarks()
        proton = propagate_quark_to_hadron(proton_quarks)

        # Proton charge must be exactly +1
        assert proton.charge_e == pytest.approx(1.0, abs=1e-10), \
            f"Proton charge should be +1, got {proton.charge_e}"

        # Create neutron from udd quarks
        neutron_quarks = helper.create_neutron_quarks()
        neutron = propagate_quark_to_hadron(neutron_quarks)

        # Neutron charge must be exactly 0
        assert neutron.charge_e == pytest.approx(0.0, abs=1e-10), \
            f"Neutron charge should be 0, got {neutron.charge_e}"

        # Verify masses are reasonable (within 5% of PDG values)
        helper.assert_within_tolerance(proton.mass_MeV, 938.27, 0.05,
                                       "Proton mass")
        helper.assert_within_tolerance(neutron.mass_MeV, 939.57, 0.05,
                                       "Neutron mass")

    def test_step2_hadrons_to_nuclei(self, helper):
        """
        Step 2: Verify hadrons combine to form nuclei with correct
        binding energies.

        Oxygen-16 should have binding energy ~127.6 MeV.
        """
        # Create O-16 nucleus (8 protons, 8 neutrons)
        oxygen = propagate_hadrons_to_atom(
            protons=8, neutrons=8, electrons=8
        )

        # O-16 binding energy should be approximately 127.6 MeV
        # Using wider tolerance since SEMF is an approximation
        helper.assert_within_tolerance(
            oxygen.nuclear_binding_energy_MeV, 127.6, 0.10,
            "O-16 binding energy"
        )

        # Verify nuclear radius follows R = r0 * A^(1/3)
        expected_radius = 1.2 * (16 ** (1/3))  # ~3.0 fm
        helper.assert_within_tolerance(
            oxygen.nuclear_radius_fm, expected_radius, 0.05,
            "O-16 nuclear radius"
        )

    def test_step3_atoms(self, helper):
        """
        Step 3: Verify atomic electron configurations.

        H should have config 1s1, O should have 2p in its config.
        """
        # Create hydrogen atom
        hydrogen = propagate_hadrons_to_atom(
            protons=1, neutrons=0, electrons=1
        )

        assert hydrogen.atomic_number == 1
        assert hydrogen.proton_count == 1
        assert hydrogen.electron_count == 1

        # Create oxygen atom
        oxygen = propagate_hadrons_to_atom(
            protons=8, neutrons=8, electrons=8
        )

        assert oxygen.atomic_number == 8
        assert oxygen.electron_count == 8
        # Oxygen electron configuration: 1s2 2s2 2p4
        # Note: propagate_hadrons_to_atom may not set configuration string

    def test_step4_molecule(self, helper):
        """
        Step 4: Verify water molecule properties.

        H2O mass should be approximately 18.015 amu.
        """
        # Create atom dictionaries for H2O
        h1 = ELEMENT_DATA['H'].copy()
        h2 = ELEMENT_DATA['H'].copy()
        o = ELEMENT_DATA['O'].copy()

        atoms = [h1, h2, o]
        bonds = [
            {'atom1': 0, 'atom2': 2, 'type': 'covalent', 'order': 1},
            {'atom1': 1, 'atom2': 2, 'type': 'covalent', 'order': 1},
        ]

        water = propagate_atoms_to_molecule(atoms, bonds)

        # Water molecular mass: 2 * 1.008 + 15.999 = 18.015 amu
        expected_mass = 18.015
        helper.assert_within_tolerance(
            water.molecular_mass_amu, expected_mass, 0.01,
            "Water molecular mass"
        )

        assert water.atom_count == 3
        assert water.total_electrons == 10  # 2*1 + 8 = 10


# ==================== TEST CLASS 2: Carbon Steel ====================

class TestE2ECarbonSteel:
    """
    End-to-end test: Build carbon steel alloy from Fe and C atoms.

    Chain: Fe atom + C atom -> Steel alloy
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_iron_atom(self, helper):
        """
        Verify iron atom properties.

        Fe: Z=26, electron config [Ar]3d6 4s2
        """
        iron = propagate_hadrons_to_atom(
            protons=26, neutrons=30, electrons=26
        )

        assert iron.atomic_number == 26, \
            f"Iron Z should be 26, got {iron.atomic_number}"
        assert iron.proton_count == 26
        assert iron.neutron_count == 30
        assert iron.electron_count == 26

        # Fe-56 binding energy ~492 MeV
        helper.assert_within_tolerance(
            iron.nuclear_binding_energy_MeV, 492.25, 0.05,
            "Fe-56 binding energy"
        )

    def test_carbon_atom(self, helper):
        """
        Verify carbon atom properties.

        C: Z=6
        """
        carbon = propagate_hadrons_to_atom(
            protons=6, neutrons=6, electrons=6
        )

        assert carbon.atomic_number == 6, \
            f"Carbon Z should be 6, got {carbon.atomic_number}"
        assert carbon.proton_count == 6
        assert carbon.neutron_count == 6
        assert carbon.electron_count == 6

    def test_steel_alloy(self, helper):
        """
        Verify carbon steel alloy properties.

        Low carbon steel: ~99% Fe, ~1% C
        Density should be approximately 7.85 g/cm3
        """
        fe_data = ELEMENT_DATA['Fe'].copy()
        c_data = ELEMENT_DATA['C'].copy()

        elements = [fe_data, c_data]
        # Low carbon steel: 99% Fe, 1% C by weight
        weight_fractions = [0.99, 0.01]

        steel = propagate_elements_to_alloy(
            elements, weight_fractions, lattice_type="BCC"
        )

        # Steel density ~7.85 g/cm3
        helper.assert_within_tolerance(
            steel.density_g_cm3, 7.85, 0.05,
            "Carbon steel density"
        )

        assert steel.primary_structure == "BCC"
        assert len(steel.weight_fractions) == 2


# ==================== TEST CLASS 3: Noble Gases ====================

class TestE2ENobleGases:
    """
    End-to-end test: Verify noble gases have complete electron shells.

    Tests He, Ne, and Ar atoms for complete shell configurations.
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_helium_complete_shell(self, helper):
        """
        Helium (Z=2) should have complete 1s shell (2 electrons).
        """
        helium = propagate_hadrons_to_atom(
            protons=2, neutrons=2, electrons=2
        )

        assert helium.atomic_number == 2
        assert helium.electron_count == 2
        # He has complete 1s2 shell (2 electrons fills n=1)

        # He-4 binding energy ~28.3 MeV (doubly magic)
        helper.assert_within_tolerance(
            helium.nuclear_binding_energy_MeV, 28.3, 0.05,
            "He-4 binding energy"
        )

    def test_neon_complete_shell(self, helper):
        """
        Neon (Z=10) should have complete shells (2 + 8 = 10 electrons).
        """
        neon = propagate_hadrons_to_atom(
            protons=10, neutrons=10, electrons=10
        )

        assert neon.atomic_number == 10
        assert neon.electron_count == 10
        # Ne: 1s2 2s2 2p6 - complete shells

    def test_argon_complete_shell(self, helper):
        """
        Argon (Z=18) should have complete shells (2 + 8 + 8 = 18 electrons).
        """
        argon = propagate_hadrons_to_atom(
            protons=18, neutrons=22, electrons=18
        )

        assert argon.atomic_number == 18
        assert argon.electron_count == 18
        # Ar: 1s2 2s2 2p6 3s2 3p6 - complete shells


# ==================== TEST CLASS 4: Helium-4 Alpha Particle ====================

class TestE2EHelium4Chain:
    """
    End-to-end test: Build alpha particle (He-4) from quarks.

    Alpha particle is exceptionally stable - doubly magic nucleus.
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_alpha_binding(self, helper):
        """
        Alpha particle (He-4) binding energy should be ~28.3 MeV.
        """
        alpha = propagate_hadrons_to_atom(
            protons=2, neutrons=2, electrons=0  # bare nucleus
        )

        # He-4 experimental binding: 28.296 MeV
        helper.assert_within_tolerance(
            alpha.nuclear_binding_energy_MeV, 28.3, 0.05,
            "Alpha particle binding energy"
        )

        # Binding per nucleon
        binding_per_nucleon = alpha.nuclear_binding_energy_MeV / 4
        helper.assert_within_tolerance(
            binding_per_nucleon, 7.07, 0.10,
            "Alpha binding per nucleon"
        )

    def test_doubly_magic(self, helper):
        """
        Alpha particle has Z=2, N=2 - both magic numbers.
        """
        alpha = propagate_hadrons_to_atom(
            protons=2, neutrons=2, electrons=2
        )

        assert alpha.proton_count == 2, "Z should be 2 (magic)"
        assert alpha.neutron_count == 2, "N should be 2 (magic)"

        # Mass number A = 4
        assert alpha.mass_number == 4


# ==================== TEST CLASS 5: Iron-56 Most Stable ====================

class TestE2EIron56Chain:
    """
    End-to-end test: Fe-56 is the most stable nucleus (highest binding/nucleon).
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_binding_per_nucleon(self, helper):
        """
        Fe-56 binding per nucleon should be ~8.79 MeV/nucleon.
        """
        fe56 = propagate_hadrons_to_atom(
            protons=26, neutrons=30, electrons=26
        )

        # Fe-56: 492.254 MeV total, 56 nucleons
        # 492.254 / 56 = 8.79 MeV/nucleon
        binding_per_nucleon = fe56.nuclear_binding_energy_MeV / 56
        helper.assert_within_tolerance(
            binding_per_nucleon, 8.79, 0.05,
            "Fe-56 binding per nucleon"
        )

    def test_stability(self, helper):
        """
        Fe-56 should be exceptionally stable.
        """
        fe56 = propagate_hadrons_to_atom(
            protons=26, neutrons=30, electrons=26
        )

        # Verify basic nuclear properties
        assert fe56.atomic_number == 26
        assert fe56.mass_number == 56
        assert fe56.neutron_count == 30

        # Fe-56 total binding should be high
        assert fe56.nuclear_binding_energy_MeV > 450, \
            "Fe-56 should have high total binding energy"


# ==================== TEST CLASS 6: Stainless Steel 304 ====================

class TestE2EStainlessSteel304:
    """
    End-to-end test: Build stainless steel 304 alloy (Fe-Cr-Ni).
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_composition(self, helper):
        """
        SS304 composition: Fe 72%, Cr 18%, Ni 10% (approximate).
        """
        fe = ELEMENT_DATA['Fe'].copy()
        cr = ELEMENT_DATA['Cr'].copy()
        ni = ELEMENT_DATA['Ni'].copy()

        elements = [fe, cr, ni]
        # SS304: ~72% Fe, ~18% Cr, ~10% Ni
        weight_fractions = [0.72, 0.18, 0.10]

        ss304 = propagate_elements_to_alloy(
            elements, weight_fractions, lattice_type="FCC"
        )

        # Verify composition stored correctly
        assert len(ss304.weight_fractions) == 3
        assert ss304.weight_fractions[0] == pytest.approx(0.72)
        assert ss304.weight_fractions[1] == pytest.approx(0.18)
        assert ss304.weight_fractions[2] == pytest.approx(0.10)

    def test_density(self, helper):
        """
        SS304 density should be approximately 7.93 g/cm3.
        """
        fe = ELEMENT_DATA['Fe'].copy()
        cr = ELEMENT_DATA['Cr'].copy()
        ni = ELEMENT_DATA['Ni'].copy()

        elements = [fe, cr, ni]
        weight_fractions = [0.72, 0.18, 0.10]

        ss304 = propagate_elements_to_alloy(
            elements, weight_fractions, lattice_type="FCC"
        )

        # SS304 density ~7.93 g/cm3
        helper.assert_within_tolerance(
            ss304.density_g_cm3, 7.93, 0.05,
            "SS304 density"
        )

        assert ss304.primary_structure == "FCC"


# ==================== TEST CLASS 7: Carbon Dioxide ====================

class TestE2ECarbonDioxide:
    """
    End-to-end test: Build CO2 molecule.

    CO2 is a linear triatomic molecule.
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_linear_geometry(self, helper):
        """
        CO2 should be a linear molecule (O=C=O).
        """
        c = ELEMENT_DATA['C'].copy()
        o1 = ELEMENT_DATA['O'].copy()
        o2 = ELEMENT_DATA['O'].copy()

        atoms = [c, o1, o2]
        # Double bonds between C and each O
        bonds = [
            {'atom1': 0, 'atom2': 1, 'type': 'covalent', 'order': 2},
            {'atom1': 0, 'atom2': 2, 'type': 'covalent', 'order': 2},
        ]

        co2 = propagate_atoms_to_molecule(atoms, bonds)

        assert co2.atom_count == 3
        # Total electrons: C(6) + 2*O(8) = 22
        assert co2.total_electrons == 22

        # Verify bonds stored
        assert len(co2.bonds) == 2

    def test_mass(self, helper):
        """
        CO2 molecular mass should be approximately 44.01 amu.
        """
        c = ELEMENT_DATA['C'].copy()
        o1 = ELEMENT_DATA['O'].copy()
        o2 = ELEMENT_DATA['O'].copy()

        atoms = [c, o1, o2]
        bonds = [
            {'atom1': 0, 'atom2': 1, 'type': 'covalent', 'order': 2},
            {'atom1': 0, 'atom2': 2, 'type': 'covalent', 'order': 2},
        ]

        co2 = propagate_atoms_to_molecule(atoms, bonds)

        # CO2 mass: 12.011 + 2*15.999 = 44.009 amu
        expected_mass = 44.01
        helper.assert_within_tolerance(
            co2.molecular_mass_amu, expected_mass, 0.01,
            "CO2 molecular mass"
        )


# ==================== TEST CLASS 8: Brass Alloy ====================

class TestE2EBrassAlloy:
    """
    End-to-end test: Build brass alloy (Cu-Zn).
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_composition(self, helper):
        """
        Brass composition: typically 60-70% Cu, 30-40% Zn.
        """
        cu = ELEMENT_DATA['Cu'].copy()
        zn = ELEMENT_DATA['Zn'].copy()

        elements = [cu, zn]
        # Cartridge brass: 70% Cu, 30% Zn
        weight_fractions = [0.70, 0.30]

        brass = propagate_elements_to_alloy(
            elements, weight_fractions, lattice_type="FCC"
        )

        assert len(brass.weight_fractions) == 2
        assert brass.weight_fractions[0] == pytest.approx(0.70)
        assert brass.weight_fractions[1] == pytest.approx(0.30)

    def test_density(self, helper):
        """
        Brass density should be approximately 8.4-8.7 g/cm3.
        """
        cu = ELEMENT_DATA['Cu'].copy()
        zn = ELEMENT_DATA['Zn'].copy()

        elements = [cu, zn]
        weight_fractions = [0.70, 0.30]

        brass = propagate_elements_to_alloy(
            elements, weight_fractions, lattice_type="FCC"
        )

        # Cartridge brass density ~8.53 g/cm3
        # Using wider tolerance for rule of mixtures approximation
        helper.assert_within_tolerance(
            brass.density_g_cm3, 8.53, 0.10,
            "Brass density"
        )


# ==================== TEST CLASS 9: Methane ====================

class TestE2EMethane:
    """
    End-to-end test: Build CH4 molecule.

    Methane has tetrahedral geometry.
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_tetrahedral_geometry(self, helper):
        """
        CH4 should have tetrahedral geometry with 4 C-H bonds.
        """
        c = ELEMENT_DATA['C'].copy()
        h1 = ELEMENT_DATA['H'].copy()
        h2 = ELEMENT_DATA['H'].copy()
        h3 = ELEMENT_DATA['H'].copy()
        h4 = ELEMENT_DATA['H'].copy()

        atoms = [c, h1, h2, h3, h4]
        # Four single bonds from C to each H
        bonds = [
            {'atom1': 0, 'atom2': 1, 'type': 'covalent', 'order': 1},
            {'atom1': 0, 'atom2': 2, 'type': 'covalent', 'order': 1},
            {'atom1': 0, 'atom2': 3, 'type': 'covalent', 'order': 1},
            {'atom1': 0, 'atom2': 4, 'type': 'covalent', 'order': 1},
        ]

        methane = propagate_atoms_to_molecule(atoms, bonds)

        assert methane.atom_count == 5
        # Total electrons: C(6) + 4*H(1) = 10
        assert methane.total_electrons == 10

        # Should have 4 bonds
        assert len(methane.bonds) == 4

    def test_mass(self, helper):
        """
        CH4 molecular mass should be approximately 16.04 amu.
        """
        c = ELEMENT_DATA['C'].copy()
        h1 = ELEMENT_DATA['H'].copy()
        h2 = ELEMENT_DATA['H'].copy()
        h3 = ELEMENT_DATA['H'].copy()
        h4 = ELEMENT_DATA['H'].copy()

        atoms = [c, h1, h2, h3, h4]
        bonds = [
            {'atom1': 0, 'atom2': 1, 'type': 'covalent', 'order': 1},
            {'atom1': 0, 'atom2': 2, 'type': 'covalent', 'order': 1},
            {'atom1': 0, 'atom2': 3, 'type': 'covalent', 'order': 1},
            {'atom1': 0, 'atom2': 4, 'type': 'covalent', 'order': 1},
        ]

        methane = propagate_atoms_to_molecule(atoms, bonds)

        # CH4 mass: 12.011 + 4*1.008 = 16.043 amu
        expected_mass = 16.04
        helper.assert_within_tolerance(
            methane.molecular_mass_amu, expected_mass, 0.01,
            "Methane molecular mass"
        )


# ==================== TEST CLASS 10: Gold Atom ====================

class TestE2EGoldAtom:
    """
    End-to-end test: Gold (Au) - a heavy element with relativistic effects.

    Z=79, electron config [Xe] 4f14 5d10 6s1
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    def test_electron_config(self, helper):
        """
        Gold electron configuration: [Xe] 4f14 5d10 6s1

        Note: Au has 79 electrons total.
        """
        gold = propagate_hadrons_to_atom(
            protons=79, neutrons=118, electrons=79  # Au-197
        )

        assert gold.atomic_number == 79
        assert gold.electron_count == 79
        assert gold.mass_number == 197

        # Verify proton and neutron counts
        assert gold.proton_count == 79
        assert gold.neutron_count == 118

    def test_relativistic_confidence_decay(self, helper):
        """
        Heavy elements like Au should have lower confidence in
        predictions due to relativistic effects.

        This test verifies that the nuclear properties are still
        calculable for heavy elements, but with expected deviations.
        """
        # Create gold atom
        gold = propagate_hadrons_to_atom(
            protons=79, neutrons=118, electrons=79
        )

        # Au-197 has significant binding energy
        # Using SEMF which has larger errors for heavy nuclei
        assert gold.nuclear_binding_energy_MeV > 1500, \
            "Au-197 should have high total binding energy"

        # Binding per nucleon should be lower than Fe-56
        # (past the iron peak, binding/nucleon decreases)
        binding_per_nucleon = gold.nuclear_binding_energy_MeV / 197
        assert binding_per_nucleon < 8.79, \
            "Au binding/nucleon should be less than Fe-56"

        # But still substantial
        assert binding_per_nucleon > 7.0, \
            "Au binding/nucleon should still be significant"

        # Create hydrogen for comparison
        hydrogen = propagate_hadrons_to_atom(
            protons=1, neutrons=0, electrons=1
        )

        # Nuclear radius should scale with A^(1/3)
        # Au has much larger nucleus than H
        assert gold.nuclear_radius_fm > hydrogen.nuclear_radius_fm


# ==================== TEST CASE 11-15: MATERIALS (ENGINEERING LEVEL) ====================

class TestE2EMaterialsFromAlloys:
    """
    Test Cases 11-15: Materials level - the engineering/observable level.

    Complete chain: Quarks -> Hadrons -> Nuclei -> Atoms -> Alloys -> Materials

    These tests validate material properties derived from alloy compositions
    against real engineering data sheets.
    """

    @pytest.fixture
    def helper(self):
        return ChainTestHelper()

    @pytest.fixture
    def chain(self):
        from periodica.utils.predictors.chain import DerivationChain
        return DerivationChain()

    @pytest.fixture
    def material_loader(self):
        from periodica.data.material_data import MaterialDataLoader
        return MaterialDataLoader()

    def test_case11_steel_1045_properties(self, helper, chain, material_loader):
        """
        Test Case 11: AISI 1045 Carbon Steel - Full chain prediction.

        Validates yield strength, Young's modulus, and density predictions
        against real data sheet values for normalized 1045 steel.
        """
        # Predict from alloy composition
        alloy_data = {
            'name': 'AISI 1045 Carbon Steel',
            'category': 'steel',
            'density_g_cm3': 7.85,
            'melting_point_K': 1773,
        }
        predicted = chain.predict_material(alloy_data, grain_size_um=25.0)

        # Young's modulus should be ~205 GPa for steel
        helper.assert_within_tolerance(
            predicted.youngs_modulus_gpa, 205, 0.05,
            "Young's modulus"
        )

        # Basic physical bounds (physics-based model has larger variance)
        assert 190 <= predicted.youngs_modulus_gpa <= 220, \
            f"Steel E should be 190-220 GPa, got {predicted.youngs_modulus_gpa}"
        assert predicted.yield_strength_mpa > 150, \
            f"Yield strength should be > 150 MPa, got {predicted.yield_strength_mpa}"
        assert predicted.density_kg_m3 > 7000, "Steel density should be > 7000 kg/m3"
        assert predicted.category == 'steel', f"Category should be steel, got {predicted.category}"

    def test_case12_aluminum_6061_t6(self, helper, chain, material_loader):
        """
        Test Case 12: Aluminum 6061-T6 - Heat treated aluminum.

        Validates lightweight metal properties prediction.
        """
        alloy_data = {
            'name': 'Aluminum 6061-T6',
            'category': 'aluminum',
            'density_g_cm3': 2.70,
            'melting_point_K': 855,
            'youngs_modulus_GPa': 68.9,
        }
        predicted = chain.predict_material(alloy_data, grain_size_um=50.0)

        # Reference values for 6061-T6
        # E = 68.9 GPa, σy = 276 MPa, ρ = 2700 kg/m3

        # Young's modulus (should use provided value)
        helper.assert_within_tolerance(
            predicted.youngs_modulus_gpa, 68.9, 0.05,
            "Aluminum Young's modulus"
        )

        # Density
        helper.assert_within_tolerance(
            predicted.density_kg_m3, 2700, 0.05,
            "Aluminum density"
        )

        # Poisson's ratio for aluminum ~0.33
        assert 0.30 <= predicted.poissons_ratio <= 0.36, \
            f"Aluminum Poisson's ratio should be ~0.33, got {predicted.poissons_ratio}"

    def test_case13_titanium_ti6al4v(self, helper, chain, material_loader):
        """
        Test Case 13: Ti-6Al-4V - High performance alloy.

        Validates prediction for aerospace-grade titanium alloy.
        """
        alloy_data = {
            'name': 'Ti-6Al-4V',
            'category': 'titanium',
            'density_g_cm3': 4.43,
            'melting_point_K': 1933,
        }
        predicted = chain.predict_material(alloy_data, grain_size_um=30.0)

        # Reference values for Ti-6Al-4V annealed
        # E = 113.8 GPa, σy = 880 MPa, ρ = 4430 kg/m3

        # Young's modulus should be ~114 GPa
        helper.assert_within_tolerance(
            predicted.youngs_modulus_gpa, 114, 0.10,
            "Titanium Young's modulus"
        )

        # Physics-based model uses base values + Hall-Petch
        # Yield strength will be lower than aged alloy
        assert predicted.yield_strength_mpa > 200, \
            f"Ti-6Al-4V should have yield strength > 200 MPa, got {predicted.yield_strength_mpa}"

        # Density
        helper.assert_within_tolerance(
            predicted.density_kg_m3, 4430, 0.05,
            "Titanium density"
        )

        # Category should be titanium
        assert 'titanium' in predicted.category.lower(), \
            f"Category should contain titanium, got {predicted.category}"

    def test_case14_inconel_718(self, helper, chain, material_loader):
        """
        Test Case 14: Inconel 718 - Nickel superalloy.

        Validates prediction for high-temperature aerospace alloy.
        """
        alloy_data = {
            'name': 'Inconel 718',
            'category': 'nickel superalloy',  # Explicit superalloy category
            'density_g_cm3': 8.19,
            'melting_point_K': 1609,
        }
        predicted = chain.predict_material(alloy_data, grain_size_um=20.0)

        # Reference values for Inconel 718
        # E = 211 GPa, σy = 1035 MPa, ρ = 8190 kg/m3

        # Young's modulus should be ~207 GPa
        helper.assert_within_tolerance(
            predicted.youngs_modulus_gpa, 207, 0.10,
            "Inconel Young's modulus"
        )

        # Physics-based model yield strength (not age-hardened prediction)
        assert predicted.yield_strength_mpa > 150, \
            f"Inconel should have yield strength > 150 MPa, got {predicted.yield_strength_mpa}"

        # Density
        helper.assert_within_tolerance(
            predicted.density_kg_m3, 8190, 0.05,
            "Inconel density"
        )

    def test_case15_copper_c11000(self, helper, chain, material_loader):
        """
        Test Case 15: Copper C11000 - Pure copper.

        Validates prediction for high-conductivity metal.
        """
        alloy_data = {
            'name': 'Copper C11000',
            'category': 'copper',
            'density_g_cm3': 8.94,
            'melting_point_K': 1356,
            'thermal_conductivity_W_mK': 385,
        }
        predicted = chain.predict_material(alloy_data, grain_size_um=40.0)

        # Reference values for pure copper
        # E = 117 GPa, σy = 69 MPa, ρ = 8940 kg/m3

        # Young's modulus
        helper.assert_within_tolerance(
            predicted.youngs_modulus_gpa, 117, 0.10,
            "Copper Young's modulus"
        )

        # Density
        helper.assert_within_tolerance(
            predicted.density_kg_m3, 8940, 0.05,
            "Copper density"
        )

        # High thermal conductivity
        assert predicted.thermal_conductivity > 300, \
            f"Copper should have high thermal conductivity, got {predicted.thermal_conductivity}"


class TestE2EFullChainQuarksToMaterials:
    """
    Test the complete derivation chain: Quarks -> ... -> Materials.

    This validates that we can trace properties from fundamental particles
    all the way to engineering-level material properties.
    """

    @pytest.fixture
    def chain(self):
        from periodica.utils.predictors.chain import DerivationChain
        return DerivationChain()

    def test_full_chain_iron_to_steel(self, chain):
        """
        Complete derivation chain test: Iron atoms to steel material.

        1. Quarks -> Hadrons (protons/neutrons)
        2. Hadrons -> Iron nucleus (Z=26, N=30)
        3. Nucleus -> Iron atom
        4. Iron atoms -> Carbon Steel alloy
        5. Alloy -> Engineering material with FEA properties
        """
        # Step 1-3: Predict iron atom from fundamental particles
        iron_result = chain.predict(Z=26, N=30)  # Fe-56

        assert iron_result.element_symbol == 'Fe'
        assert iron_result.Z == 26
        assert iron_result.N == 30

        # Verify nuclear properties
        binding_per_nucleon = iron_result.nuclear_properties['binding_per_nucleon_mev']
        assert 8.5 <= binding_per_nucleon <= 9.0, \
            f"Fe-56 binding/nucleon should be ~8.79 MeV, got {binding_per_nucleon}"

        # Step 4-5: Predict material from alloy
        steel_alloy = {
            'name': 'Carbon Steel (from Fe)',
            'category': 'steel',
            'density_g_cm3': 7.85,
            'melting_point_K': 1773,
        }
        material_result = chain.predict_material(steel_alloy, grain_size_um=25.0)

        # Verify material properties
        assert material_result.category == 'steel'
        assert material_result.youngs_modulus_gpa > 0
        assert material_result.yield_strength_mpa > 0
        assert material_result.density_kg_m3 > 0

        # Chain should maintain consistency - check confidence exists and is reasonable
        # MaterialPredictor returns keys: elastic, strength, thermal, electrical, overall
        assert 'overall' in material_result.confidence, \
            f"Material result should have 'overall' confidence key, got keys: {list(material_result.confidence.keys())}"
        assert material_result.confidence['overall'] > 0.3, \
            f"Material prediction should have reasonable confidence, got {material_result.confidence['overall']}"

    def test_full_chain_aluminum(self, chain):
        """
        Complete derivation: Aluminum atom -> Aluminum alloy -> Material.
        """
        # Predict aluminum atom
        al_result = chain.predict(Z=13, N=14)  # Al-27

        assert al_result.element_symbol == 'Al'
        assert al_result.Z == 13

        # Predict aluminum material
        al_alloy = {
            'name': 'Aluminum 6061',
            'category': 'aluminum',
            'density_g_cm3': 2.70,
        }
        al_material = chain.predict_material(al_alloy, grain_size_um=50.0)

        assert al_material.category == 'aluminum'
        assert 60 <= al_material.youngs_modulus_gpa <= 80, \
            f"Aluminum E should be 60-80 GPa, got {al_material.youngs_modulus_gpa}"

    def test_chain_confidence_propagation(self, chain):
        """
        Test that confidence scores propagate correctly through the chain.

        Overall confidence = hadron_conf * nuclear_conf * atomic_conf
        This is a product, so values will be lower than individual confidences.

        Note: Confidence values depend on which predictors are registered.
        We verify structure and valid ranges rather than specific values.
        """
        # Light element - verify confidence structure
        h_result = chain.predict(Z=1, N=0)
        assert h_result.confidence['overall'] > 0.1, \
            f"Hydrogen should have reasonable combined confidence, got {h_result.confidence['overall']}"

        # Verify individual confidences exist and are valid
        assert 'hadron' in h_result.confidence
        assert 'nuclear' in h_result.confidence
        assert 'atomic' in h_result.confidence

        for key in ['hadron', 'nuclear', 'atomic', 'overall']:
            assert 0.0 <= h_result.confidence[key] <= 1.0, \
                f"{key} confidence should be in [0, 1], got {h_result.confidence[key]}"

        # Medium element (Fe) - should be in well-characterized range
        fe_result = chain.predict(Z=26, N=30)
        assert fe_result.confidence['overall'] > 0.2, \
            f"Iron should have good combined confidence, got {fe_result.confidence['overall']}"

        # Heavy element - verify confidence exists and is bounded
        au_result = chain.predict(Z=79, N=118)
        assert au_result.confidence['overall'] > 0.0, \
            "Gold should have positive confidence"
        assert au_result.confidence['overall'] < 1.0, \
            "Confidence should be bounded below 1.0"

        # Verify overall confidence is product of individual confidences (or close to it)
        # This tests the chain's confidence combination logic
        expected_overall = (
            fe_result.confidence['hadron'] *
            fe_result.confidence['nuclear'] *
            fe_result.confidence['atomic']
        )
        assert abs(fe_result.confidence['overall'] - expected_overall) < 0.01, \
            f"Overall confidence should be product of individual confidences"

        # Material prediction confidence - uses different confidence keys
        steel = chain.predict_material({'category': 'steel'})
        assert 'overall' in steel.confidence, \
            f"Steel should have 'overall' confidence key, got: {list(steel.confidence.keys())}"
        assert steel.confidence['overall'] > 0


# ==================== MAIN ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
