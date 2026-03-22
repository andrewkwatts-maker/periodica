"""
Tests for all predictor classes in the utils/predictors/ package.

Covers import verification, instantiation, and key prediction methods
with known inputs for each predictor category.
"""

import pytest
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Nuclear Predictor (SEMF)
# =============================================================================

class TestSEMFNuclearPredictor:
    """Tests for the Semi-Empirical Mass Formula nuclear predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.nuclear.semf_predictor import SEMFNuclearPredictor
        return SEMFNuclearPredictor()

    def test_import(self):
        from periodica.utils.predictors.nuclear.semf_predictor import SEMFNuclearPredictor
        assert SEMFNuclearPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None
        assert predictor.name == "SEMF Nuclear Predictor"

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'calculate_binding_energy'))
        assert callable(getattr(predictor, 'calculate_radius'))
        assert callable(getattr(predictor, 'calculate_nuclear_mass'))
        assert callable(getattr(predictor, 'check_stability'))
        assert callable(getattr(predictor, 'predict'))
        assert callable(getattr(predictor, 'get_confidence'))

    def test_binding_energy_iron_56(self, predictor):
        """Iron-56 is the most tightly bound nucleus, ~8.8 MeV/nucleon."""
        Z, N = 26, 30  # Fe-56: 26 protons, 30 neutrons
        binding = predictor.calculate_binding_energy(Z, N)
        binding_per_nucleon = binding / 56

        # Well-known value: ~8.8 MeV/nucleon (experimental: 8.790)
        assert 8.0 < binding_per_nucleon < 9.5, (
            f"Fe-56 binding energy per nucleon should be ~8.8 MeV, got {binding_per_nucleon:.3f}"
        )

    def test_binding_energy_helium_4(self, predictor):
        """He-4 (alpha particle) binding energy ~7.07 MeV/nucleon."""
        Z, N = 2, 2
        binding = predictor.calculate_binding_energy(Z, N)
        binding_per_nucleon = binding / 4
        # SEMF is less accurate for light nuclei, but should be in range
        assert 3.0 < binding_per_nucleon < 10.0

    def test_binding_energy_zero(self, predictor):
        """Zero nucleons should give zero binding energy."""
        binding = predictor.calculate_binding_energy(0, 0)
        assert binding == 0.0

    def test_binding_energy_always_non_negative(self, predictor):
        """Binding energy should never be negative."""
        for Z in range(1, 20):
            for N in range(1, 20):
                binding = predictor.calculate_binding_energy(Z, N)
                assert binding >= 0.0

    def test_nuclear_radius(self, predictor):
        """Nuclear radius should follow R = R0 * A^(1/3)."""
        # Fe-56
        radius = predictor.calculate_radius(56)
        expected = 1.25 * (56 ** (1/3))
        assert abs(radius - expected) < 0.01

    def test_nuclear_radius_zero(self, predictor):
        radius = predictor.calculate_radius(0)
        assert radius == 0.0

    def test_nuclear_mass(self, predictor):
        """Nuclear mass = Z*mp + N*mn - B."""
        Z, N = 26, 30
        binding = predictor.calculate_binding_energy(Z, N)
        mass = predictor.calculate_nuclear_mass(Z, N, binding)
        # Mass should be positive and less than sum of constituent masses
        constituent_mass = Z * predictor.PROTON_MASS_MEV + N * predictor.NEUTRON_MASS_MEV
        assert 0 < mass < constituent_mass

    def test_stability_doubly_magic(self, predictor):
        """Doubly magic nuclei (e.g., He-4: Z=2, N=2) should be stable."""
        is_stable, reason = predictor.check_stability(2, 2)
        assert is_stable is True
        assert "magic" in reason.lower()

    def test_stability_neutron_rich(self, predictor):
        """Very neutron-rich nuclei should be unstable."""
        is_stable, reason = predictor.check_stability(10, 30)
        assert is_stable is False

    def test_predict_full(self, predictor):
        """Test full predict() method with NuclearInput."""
        from periodica.utils.predictors.base import NuclearInput
        input_data = NuclearInput(Z=26, N=30)
        result = predictor.predict(input_data)

        assert result.binding_energy_mev > 0
        assert result.binding_per_nucleon_mev > 0
        assert result.nuclear_radius_fm > 0
        assert result.nuclear_mass_mev > 0
        assert isinstance(result.is_stable, bool)
        assert isinstance(result.stability_reason, str)

    def test_confidence(self, predictor):
        """Test confidence calculation."""
        from periodica.utils.predictors.base import NuclearInput, NuclearResult
        input_data = NuclearInput(Z=26, N=30)
        result = predictor.predict(input_data)
        confidence = predictor.get_confidence(input_data, result)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_medium_nuclei_high(self, predictor):
        """SEMF should have high confidence for medium-mass nuclei."""
        from periodica.utils.predictors.base import NuclearInput
        input_data = NuclearInput(Z=26, N=30)
        result = predictor.predict(input_data)
        confidence = predictor.get_confidence(input_data, result)
        assert confidence >= 0.7

    def test_pairing_even_even(self, predictor):
        """Even-even nuclei should have positive pairing term."""
        pairing = predictor._calculate_pairing(26, 30, 56)
        assert pairing > 0

    def test_pairing_odd_odd(self, predictor):
        """Odd-odd nuclei should have negative pairing term."""
        pairing = predictor._calculate_pairing(7, 7, 14)
        assert pairing < 0

    def test_pairing_odd_even(self, predictor):
        """Odd-even nuclei should have zero pairing term."""
        pairing = predictor._calculate_pairing(7, 8, 15)
        assert pairing == 0.0

    def test_magic_numbers_defined(self, predictor):
        """Magic numbers set should contain known values."""
        expected_magic = {2, 8, 20, 28, 50, 82, 126}
        assert predictor.MAGIC_NUMBERS == expected_magic


# =============================================================================
# Atomic Predictor (Slater)
# =============================================================================

class TestSlaterAtomicPredictor:
    """Tests for the Slater atomic predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.atomic.slater_predictor import SlaterAtomicPredictor
        return SlaterAtomicPredictor()

    def test_import(self):
        from periodica.utils.predictors.atomic.slater_predictor import SlaterAtomicPredictor
        assert SlaterAtomicPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None
        assert predictor.name == "Slater Atomic Predictor"

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'calculate_electron_configuration'))
        assert callable(getattr(predictor, 'calculate_ionization_energy'))
        assert callable(getattr(predictor, 'calculate_effective_nuclear_charge'))
        assert callable(getattr(predictor, 'predict'))
        assert callable(getattr(predictor, 'get_confidence'))

    def test_electron_config_hydrogen(self, predictor):
        config = predictor.calculate_electron_configuration(1)
        assert config == "1s1"

    def test_electron_config_helium(self, predictor):
        config = predictor.calculate_electron_configuration(2)
        assert config == "1s2"

    def test_electron_config_carbon(self, predictor):
        config = predictor.calculate_electron_configuration(6)
        assert config == "1s2 2s2 2p2"

    def test_electron_config_neon(self, predictor):
        config = predictor.calculate_electron_configuration(10)
        assert config == "1s2 2s2 2p6"

    def test_electron_config_sodium(self, predictor):
        config = predictor.calculate_electron_configuration(11)
        assert config == "1s2 2s2 2p6 3s1"

    def test_electron_config_iron(self, predictor):
        config = predictor.calculate_electron_configuration(26)
        assert config == "1s2 2s2 2p6 3s2 3p6 4s2 3d6"

    def test_effective_nuclear_charge_hydrogen(self, predictor):
        """Hydrogen should have Z_eff = 1.0."""
        z_eff = predictor.calculate_effective_nuclear_charge(1)
        assert z_eff == 1.0

    def test_effective_nuclear_charge_helium(self, predictor):
        """Helium Z_eff should be less than Z=2 due to shielding."""
        z_eff = predictor.calculate_effective_nuclear_charge(2)
        assert 1.0 < z_eff < 2.0

    def test_effective_nuclear_charge_lithium(self, predictor):
        """Lithium Z_eff should be between 1 and 3."""
        z_eff = predictor.calculate_effective_nuclear_charge(3)
        assert 1.0 <= z_eff <= 3.0

    def test_effective_nuclear_charge_increases(self, predictor):
        """Z_eff should generally increase with Z within a period."""
        z_eff_li = predictor.calculate_effective_nuclear_charge(3)
        z_eff_ne = predictor.calculate_effective_nuclear_charge(10)
        assert z_eff_ne > z_eff_li

    def test_ionization_energy_positive(self, predictor):
        """Ionization energy should always be positive."""
        for Z in [1, 6, 11, 26, 79]:
            ie = predictor.calculate_ionization_energy(Z)
            assert ie > 0, f"IE for Z={Z} should be positive, got {ie}"

    def test_ionization_energy_lithium(self, predictor):
        """Lithium IE should be approximately 5.39 eV (known value)."""
        ie = predictor.calculate_ionization_energy(3)
        assert abs(ie - 5.39) < 0.01, f"Li IE should be ~5.39, got {ie}"

    def test_ionization_energy_sodium(self, predictor):
        """Sodium IE should be approximately 5.14 eV (known value)."""
        ie = predictor.calculate_ionization_energy(11)
        assert abs(ie - 5.14) < 0.01, f"Na IE should be ~5.14, got {ie}"

    def test_predict_full(self, predictor):
        """Test full predict() with AtomicInput."""
        from periodica.utils.predictors.base import AtomicInput
        input_data = AtomicInput(Z=26)
        result = predictor.predict(input_data)

        assert result.symbol == 'Fe'
        assert result.name == 'Iron'
        assert result.electron_configuration is not None
        assert result.ionization_energy_ev > 0
        assert result.atomic_radius_pm > 0
        assert result.period > 0
        assert result.block in ['s', 'p', 'd', 'f']

    def test_predict_hydrogen(self, predictor):
        from periodica.utils.predictors.base import AtomicInput
        result = predictor.predict(AtomicInput(Z=1))
        assert result.symbol == 'H'
        assert result.name == 'Hydrogen'
        assert result.period == 1
        assert result.block == 's'

    def test_confidence(self, predictor):
        from periodica.utils.predictors.base import AtomicInput
        input_data = AtomicInput(Z=6)
        result = predictor.predict(input_data)
        confidence = predictor.get_confidence(input_data, result)
        assert 0.0 <= confidence <= 1.0

    def test_slater_shielding_empty(self, predictor):
        sigma = predictor._calculate_slater_shielding({})
        assert sigma == 0.0


# =============================================================================
# Hadron Predictor (Constituent Quark Model)
# =============================================================================

class TestConstituentQuarkPredictor:
    """Tests for the constituent quark model hadron predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.hadron.constituent_predictor import ConstituentQuarkPredictor
        return ConstituentQuarkPredictor()

    def test_import(self):
        from periodica.utils.predictors.hadron.constituent_predictor import ConstituentQuarkPredictor
        assert ConstituentQuarkPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None
        assert predictor.name == "Constituent Quark Model"

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'derive_hadron'))
        assert callable(getattr(predictor, 'predict'))
        assert callable(getattr(predictor, 'get_confidence'))
        assert callable(getattr(predictor, 'validate'))

    def test_proton_derivation(self, predictor):
        """Proton (uud) should have mass ~938 MeV, charge +1, baryon number 1."""
        quarks = [
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Down Quark', 'symbol': 'd', 'Charge_e': -1/3, 'BaryonNumber_B': 1/3},
        ]
        result = predictor.derive_hadron(quarks)
        assert abs(result.mass_mev - 938.27) < 1.0
        assert abs(result.charge_e - 1.0) < 0.01
        assert result.baryon_number == 1
        assert "Proton" in result.name

    def test_neutron_derivation(self, predictor):
        """Neutron (udd) should have mass ~939 MeV, charge 0, baryon number 1."""
        quarks = [
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Down Quark', 'symbol': 'd', 'Charge_e': -1/3, 'BaryonNumber_B': 1/3},
            {'name': 'Down Quark', 'symbol': 'd', 'Charge_e': -1/3, 'BaryonNumber_B': 1/3},
        ]
        result = predictor.derive_hadron(quarks)
        assert abs(result.mass_mev - 939.57) < 1.0
        assert abs(result.charge_e) < 0.01
        assert result.baryon_number == 1
        assert "Neutron" in result.name

    def test_pion_plus_derivation(self, predictor):
        """Pion+ (u d-bar) should have mass ~140 MeV, charge +1, baryon number 0."""
        quarks = [
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Anti-Down Quark', 'symbol': 'd\u0305', 'Charge_e': 1/3, 'BaryonNumber_B': -1/3},
        ]
        result = predictor.derive_hadron(quarks)
        assert abs(result.mass_mev - 139.57) < 1.0
        assert abs(result.charge_e - 1.0) < 0.01
        assert result.baryon_number == 0

    def test_empty_quarks(self, predictor):
        result = predictor.derive_hadron([])
        assert result.name == "Unknown"
        assert result.mass_mev == 0.0

    def test_validate_valid_input(self, predictor):
        from periodica.utils.predictors.base import HadronInput
        quarks = [
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Down Quark', 'symbol': 'd', 'Charge_e': -1/3, 'BaryonNumber_B': 1/3},
        ]
        input_data = HadronInput(quarks=quarks)
        is_valid, msg = predictor.validate(input_data)
        assert is_valid is True

    def test_validate_wrong_quark_count(self, predictor):
        from periodica.utils.predictors.base import HadronInput
        quarks = [
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
        ]
        input_data = HadronInput(quarks=quarks)
        is_valid, msg = predictor.validate(input_data)
        assert is_valid is False

    def test_confidence_known_hadron(self, predictor):
        from periodica.utils.predictors.base import HadronInput
        quarks = [
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Up Quark', 'symbol': 'u', 'Charge_e': 2/3, 'BaryonNumber_B': 1/3},
            {'name': 'Down Quark', 'symbol': 'd', 'Charge_e': -1/3, 'BaryonNumber_B': 1/3},
        ]
        input_data = HadronInput(quarks=quarks)
        result = predictor.predict(input_data)
        confidence = predictor.get_confidence(input_data, result)
        assert confidence >= 0.9  # Known hadron should have high confidence


# =============================================================================
# Molecule Predictor (VSEPR)
# =============================================================================

class TestVSEPRPredictor:
    """Tests for the VSEPR molecular geometry predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.molecule.vsepr_predictor import VSEPRPredictor
        return VSEPRPredictor()

    def test_import(self):
        from periodica.utils.predictors.molecule.vsepr_predictor import VSEPRPredictor
        assert VSEPRPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None
        assert predictor.name == "VSEPR Predictor"

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'determine_geometry'))
        assert callable(getattr(predictor, 'calculate_molecular_mass'))
        assert callable(getattr(predictor, 'predict'))
        assert callable(getattr(predictor, 'get_confidence'))

    def test_water_geometry_bent(self, predictor):
        """Water (H2O): 2 bonding pairs + 2 lone pairs = bent, ~104.5 degrees."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=2, lone_pairs=2)
        assert geometry == 'Bent'
        assert abs(angle - 104.5) < 0.1

    def test_methane_geometry_tetrahedral(self, predictor):
        """Methane (CH4): 4 bonding pairs + 0 lone pairs = tetrahedral, 109.5 degrees."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=4, lone_pairs=0)
        assert geometry == 'Tetrahedral'
        assert abs(angle - 109.5) < 0.1

    def test_co2_geometry_linear(self, predictor):
        """CO2: 2 bonding pairs + 0 lone pairs = linear, 180 degrees."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=2, lone_pairs=0)
        assert geometry == 'Linear'
        assert abs(angle - 180.0) < 0.1

    def test_ammonia_geometry_trigonal_pyramidal(self, predictor):
        """NH3: 3 bonding pairs + 1 lone pair = trigonal pyramidal."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=3, lone_pairs=1)
        assert geometry == 'Trigonal Pyramidal'

    def test_bf3_geometry_trigonal_planar(self, predictor):
        """BF3: 3 bonding pairs + 0 lone pairs = trigonal planar."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=3, lone_pairs=0)
        assert geometry == 'Trigonal Planar'
        assert abs(angle - 120.0) < 0.1

    def test_sf6_geometry_octahedral(self, predictor):
        """SF6: 6 bonding pairs + 0 lone pairs = octahedral."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=6, lone_pairs=0)
        assert geometry == 'Octahedral'
        assert abs(angle - 90.0) < 0.1

    def test_xef4_square_planar(self, predictor):
        """XeF4: 4 bonding pairs + 2 lone pairs = square planar."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=4, lone_pairs=2)
        assert geometry == 'Square Planar'

    def test_unknown_geometry(self, predictor):
        """Unknown combination should return 'Unknown'."""
        geometry, angle = predictor.determine_geometry(bonding_pairs=10, lone_pairs=5)
        assert geometry == 'Unknown'

    def test_molecular_mass_water(self, predictor):
        """Water molecular mass should be ~18.015 amu."""
        atoms = [
            {'element': 'O'},
            {'element': 'H'},
            {'element': 'H'},
        ]
        mass = predictor.calculate_molecular_mass(atoms)
        assert abs(mass - 18.015) < 0.01

    def test_molecular_mass_methane(self, predictor):
        """Methane molecular mass should be ~16.043 amu."""
        atoms = [
            {'element': 'C'},
            {'element': 'H'},
            {'element': 'H'},
            {'element': 'H'},
            {'element': 'H'},
        ]
        mass = predictor.calculate_molecular_mass(atoms)
        assert abs(mass - 16.043) < 0.01

    def test_predict_water(self, predictor):
        """Test full prediction for water molecule."""
        from periodica.utils.predictors.base import MoleculeInput
        atoms = [
            {'element': 'O'},
            {'element': 'H'},
            {'element': 'H'},
        ]
        bonds = [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
        ]
        input_data = MoleculeInput(atoms=atoms, bonds=bonds)
        result = predictor.predict(input_data)

        assert result.geometry == 'Bent'
        assert result.molecular_mass_amu > 17
        assert len(result.bond_angles) > 0

    def test_predict_methane(self, predictor):
        """Test full prediction for methane molecule."""
        from periodica.utils.predictors.base import MoleculeInput
        atoms = [
            {'element': 'C'},
            {'element': 'H'},
            {'element': 'H'},
            {'element': 'H'},
            {'element': 'H'},
        ]
        bonds = [
            {'from': 0, 'to': 1, 'type': 'single'},
            {'from': 0, 'to': 2, 'type': 'single'},
            {'from': 0, 'to': 3, 'type': 'single'},
            {'from': 0, 'to': 4, 'type': 'single'},
        ]
        input_data = MoleculeInput(atoms=atoms, bonds=bonds)
        result = predictor.predict(input_data)

        assert result.geometry == 'Tetrahedral'
        assert result.dipole_moment == 0.0  # Symmetric

    def test_confidence_known_geometry(self, predictor):
        from periodica.utils.predictors.base import MoleculeInput, MoleculeResult
        result = MoleculeResult(
            geometry='Tetrahedral',
            bond_angles=[109.5],
            dipole_moment=0.0,
            molecular_mass_amu=16.0
        )
        input_data = MoleculeInput(
            atoms=[{'element': 'C'}],
            bonds=[]
        )
        confidence = predictor.get_confidence(input_data, result)
        assert confidence >= 0.9


# =============================================================================
# Material Predictor
# =============================================================================

class TestMaterialPredictor:
    """Tests for the physics-based material predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.material.material_predictor import MaterialPredictor
        return MaterialPredictor()

    def test_import(self):
        from periodica.utils.predictors.material.material_predictor import MaterialPredictor
        assert MaterialPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None
        assert predictor.name == "Physics-Based Material Predictor"

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'predict'))
        assert callable(getattr(predictor, 'get_confidence'))

    def test_predict_steel(self, predictor):
        from periodica.utils.predictors.material.material_predictor import MaterialInput
        input_data = MaterialInput(
            alloy_data={
                'name': 'Mild Steel',
                'category': 'steel',
                'density_g_cm3': 7.85,
                'melting_point_K': 1800,
            }
        )
        result = predictor.predict(input_data)

        assert result.name == 'Mild Steel'
        assert result.category == 'steel'
        assert result.youngs_modulus_GPa > 100  # Steel E ~205 GPa
        assert result.density_kg_m3 > 7000
        assert result.yield_strength_MPa > 0
        assert result.poissons_ratio > 0

    def test_predict_aluminum(self, predictor):
        from periodica.utils.predictors.material.material_predictor import MaterialInput
        input_data = MaterialInput(
            alloy_data={
                'name': 'Pure Aluminum',
                'category': 'aluminum',
                'density_g_cm3': 2.70,
            }
        )
        result = predictor.predict(input_data)

        assert result.category == 'aluminum'
        assert result.youngs_modulus_GPa > 50
        assert result.youngs_modulus_GPa < 150

    def test_hall_petch_grain_size_effect(self, predictor):
        """Smaller grains should give higher yield strength."""
        from periodica.utils.predictors.material.material_predictor import MaterialInput
        alloy_data = {'name': 'Steel', 'category': 'steel'}

        result_large_grain = predictor.predict(
            MaterialInput(alloy_data=alloy_data, grain_size_um=100)
        )
        result_small_grain = predictor.predict(
            MaterialInput(alloy_data=alloy_data, grain_size_um=5)
        )
        assert result_small_grain.yield_strength_MPa > result_large_grain.yield_strength_MPa

    def test_porosity_reduces_modulus(self, predictor):
        """Porosity should reduce Young's modulus."""
        from periodica.utils.predictors.material.material_predictor import MaterialInput
        alloy_data = {'name': 'Steel', 'category': 'steel'}

        result_dense = predictor.predict(
            MaterialInput(alloy_data=alloy_data, porosity=0.0)
        )
        result_porous = predictor.predict(
            MaterialInput(alloy_data=alloy_data, porosity=0.2)
        )
        assert result_porous.youngs_modulus_GPa < result_dense.youngs_modulus_GPa

    def test_confidence_output(self, predictor):
        from periodica.utils.predictors.material.material_predictor import MaterialInput
        input_data = MaterialInput(
            alloy_data={'name': 'Test', 'category': 'steel'}
        )
        result = predictor.predict(input_data)
        assert 'overall' in result.confidence
        assert 0 < result.confidence['overall'] <= 1.0

    def test_elastic_moduli_consistency(self, predictor):
        """G = E / (2*(1+nu)) should hold."""
        from periodica.utils.predictors.material.material_predictor import MaterialInput
        input_data = MaterialInput(
            alloy_data={'name': 'Test', 'category': 'steel'}
        )
        result = predictor.predict(input_data)
        expected_G = result.youngs_modulus_GPa / (2 * (1 + result.poissons_ratio))
        assert abs(result.shear_modulus_GPa - expected_G) < 0.01


# =============================================================================
# Alloy Predictor (Rule of Mixtures)
# =============================================================================

class TestRuleOfMixturesPredictor:
    """Tests for the rule of mixtures alloy predictor."""

    def test_import(self):
        from periodica.utils.predictors.alloy.rule_of_mixtures import RuleOfMixturesPredictor
        assert RuleOfMixturesPredictor is not None

    def test_instantiation(self):
        from periodica.utils.predictors.alloy.rule_of_mixtures import RuleOfMixturesPredictor
        predictor = RuleOfMixturesPredictor()
        assert predictor.name == "Rule of Mixtures Predictor"

    def test_has_required_methods(self):
        from periodica.utils.predictors.alloy.rule_of_mixtures import RuleOfMixturesPredictor
        predictor = RuleOfMixturesPredictor()
        assert callable(getattr(predictor, 'calculate_density'))
        assert callable(getattr(predictor, 'predict'))


# =============================================================================
# Biological Predictors
# =============================================================================

class TestAminoAcidPredictor:
    """Tests for the amino acid predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidPredictor
        return AminoAcidPredictor()

    def test_import(self):
        from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidPredictor
        assert AminoAcidPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None
        assert "Amino Acid" in predictor.name

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'calculate_charge_at_pH'))
        assert callable(getattr(predictor, 'calculate_isoelectric_point'))
        assert callable(getattr(predictor, 'predict'))

    def test_charge_at_low_pH(self, predictor):
        """At very low pH, amino acids should be positively charged."""
        charge = predictor.calculate_charge_at_pH(
            pH=1.0, pKa_carboxyl=2.0, pKa_amino=9.5
        )
        assert charge > 0

    def test_charge_at_high_pH(self, predictor):
        """At very high pH, amino acids should be negatively charged."""
        charge = predictor.calculate_charge_at_pH(
            pH=14.0, pKa_carboxyl=2.0, pKa_amino=9.5
        )
        assert charge < 0

    def test_isoelectric_point_no_sidechain(self, predictor):
        """pI without sidechain = average of carboxyl and amino pKa."""
        pI = predictor.calculate_isoelectric_point(
            pKa_carboxyl=2.0, pKa_amino=9.5
        )
        assert abs(pI - 5.75) < 0.01

    def test_isoelectric_point_acidic_sidechain(self, predictor):
        """Acidic sidechain should lower the pI."""
        pI = predictor.calculate_isoelectric_point(
            pKa_carboxyl=2.0, pKa_amino=9.5, pKa_sidechain=4.0,
            sidechain_is_acidic=True
        )
        assert pI < 5.75  # Lower than without sidechain

    def test_isoelectric_point_basic_sidechain(self, predictor):
        """Basic sidechain should raise the pI."""
        pI = predictor.calculate_isoelectric_point(
            pKa_carboxyl=2.0, pKa_amino=9.5, pKa_sidechain=12.0,
            sidechain_is_acidic=False
        )
        assert pI > 5.75  # Higher than without sidechain


class TestProteinPredictor:
    """Tests for the protein predictor."""

    @pytest.fixture
    def predictor(self):
        from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor
        return ProteinPredictor()

    def test_import(self):
        from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor
        assert ProteinPredictor is not None

    def test_instantiation(self, predictor):
        assert predictor is not None

    def test_has_required_methods(self, predictor):
        assert callable(getattr(predictor, 'predict_secondary_structure'))
        assert callable(getattr(predictor, 'calculate_molecular_mass'))
        assert callable(getattr(predictor, 'calculate_isoelectric_point'))
        assert callable(getattr(predictor, 'calculate_gravy'))
        assert callable(getattr(predictor, 'analyze_protein'))

    def test_molecular_mass_positive(self, predictor):
        mass = predictor.calculate_molecular_mass("ACDEFGHIKLMNPQRSTVWY")
        assert mass > 0

    def test_gravy_score(self, predictor):
        """GRAVY should return a finite value for any valid sequence."""
        gravy = predictor.calculate_gravy("AAAAAA")
        assert isinstance(gravy, float)

    def test_gravy_empty(self, predictor):
        gravy = predictor.calculate_gravy("")
        assert gravy == 0.0

    def test_secondary_structure_prediction(self, predictor):
        results = predictor.predict_secondary_structure("AAAAAA")
        assert len(results) == 6
        for res in results:
            assert 'structure' in res
            assert res['structure'] in ['H', 'E', 'T', 'C']
            assert 'phi' in res
            assert 'psi' in res

    def test_amino_acid_composition(self, predictor):
        comp = predictor.get_amino_acid_composition("AAACCC")
        assert comp['A'] == 3
        assert comp['C'] == 3


class TestNucleicAcidPredictor:
    """Tests for the nucleic acid predictor."""

    def test_import(self):
        from periodica.utils.predictors.biological.nucleic_acid_predictor import NucleicAcidPredictor
        assert NucleicAcidPredictor is not None

    def test_instantiation(self):
        from periodica.utils.predictors.biological.nucleic_acid_predictor import NucleicAcidPredictor
        predictor = NucleicAcidPredictor()
        assert predictor is not None


class TestBiomaterialPredictor:
    """Tests for the biomaterial predictor."""

    def test_import(self):
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor
        assert BiomaterialPredictor is not None

    def test_instantiation(self):
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor
        predictor = BiomaterialPredictor()
        assert predictor is not None

    def test_has_ecm_moduli(self):
        from periodica.utils.predictors.biological.biomaterial_predictor import BiomaterialPredictor
        predictor = BiomaterialPredictor()
        assert len(predictor.DEFAULT_ECM_MODULI) > 0


class TestCellPredictor:
    """Tests for the cell predictor."""

    def test_import(self):
        from periodica.utils.predictors.biological.cell_predictor import CellPredictor
        assert CellPredictor is not None

    def test_instantiation(self):
        from periodica.utils.predictors.biological.cell_predictor import CellPredictor
        predictor = CellPredictor()
        assert predictor is not None

    def test_has_kleiber_constants(self):
        from periodica.utils.predictors.biological.cell_predictor import CellPredictor
        predictor = CellPredictor()
        assert predictor.DEFAULT_B0 > 0
        assert 0.5 < predictor.DEFAULT_ALPHA < 1.0


# =============================================================================
# Predictor Registry and Framework
# =============================================================================

class TestPredictorRegistry:
    """Tests for the predictor registry."""

    def test_import_registry(self):
        from periodica.utils.predictors.registry import PredictorRegistry
        assert PredictorRegistry is not None

    def test_registry_has_nuclear(self):
        from periodica.utils.predictors.registry import PredictorRegistry
        registry = PredictorRegistry()
        predictor = registry.get('nuclear', 'semf')
        assert predictor is not None

    def test_registry_has_atomic(self):
        from periodica.utils.predictors.registry import PredictorRegistry
        registry = PredictorRegistry()
        predictor = registry.get('atomic', 'slater')
        assert predictor is not None

    def test_registry_has_hadron(self):
        from periodica.utils.predictors.registry import PredictorRegistry
        registry = PredictorRegistry()
        predictor = registry.get('hadron', 'constituent')
        assert predictor is not None

    def test_registry_defaults(self):
        from periodica.utils.predictors.registry import PredictorRegistry
        registry = PredictorRegistry()
        # Default aliases should work
        nuclear = registry.get('nuclear', 'default')
        atomic = registry.get('atomic', 'default')
        hadron = registry.get('hadron', 'default')
        assert nuclear is not None
        assert atomic is not None
        assert hadron is not None


class TestPredictorBaseClasses:
    """Tests for abstract base classes and data structures."""

    def test_nuclear_input(self):
        from periodica.utils.predictors.base import NuclearInput
        inp = NuclearInput(Z=26, N=30)
        assert inp.Z == 26
        assert inp.N == 30
        assert inp.A == 56

    def test_nuclear_input_validation(self):
        from periodica.utils.predictors.base import NuclearInput
        with pytest.raises(ValueError):
            NuclearInput(Z=-1, N=0)
        with pytest.raises(ValueError):
            NuclearInput(Z=0, N=-1)

    def test_atomic_input(self):
        from periodica.utils.predictors.base import AtomicInput
        inp = AtomicInput(Z=6)
        assert inp.Z == 6

    def test_atomic_input_validation(self):
        from periodica.utils.predictors.base import AtomicInput
        with pytest.raises(ValueError):
            AtomicInput(Z=0)

    def test_hadron_input(self):
        from periodica.utils.predictors.base import HadronInput
        quarks = [{'symbol': 'u'}, {'symbol': 'u'}, {'symbol': 'd'}]
        inp = HadronInput(quarks=quarks)
        assert len(inp.quarks) == 3

    def test_hadron_input_validation(self):
        from periodica.utils.predictors.base import HadronInput
        with pytest.raises(ValueError):
            HadronInput(quarks=[])

    def test_molecule_input(self):
        from periodica.utils.predictors.base import MoleculeInput
        atoms = [{'element': 'O'}, {'element': 'H'}, {'element': 'H'}]
        bonds = [{'from': 0, 'to': 1}, {'from': 0, 'to': 2}]
        inp = MoleculeInput(atoms=atoms, bonds=bonds)
        assert len(inp.atoms) == 3
        assert len(inp.bonds) == 2

    def test_molecule_input_validation(self):
        from periodica.utils.predictors.base import MoleculeInput
        with pytest.raises(ValueError):
            MoleculeInput(atoms=[], bonds=[])

    def test_alloy_input(self):
        from periodica.utils.predictors.base import AlloyInput
        components = [
            {'element': 'Fe', 'fraction': 0.7},
            {'element': 'C', 'fraction': 0.3},
        ]
        inp = AlloyInput(components=components)
        assert len(inp.components) == 2

    def test_alloy_input_validation(self):
        from periodica.utils.predictors.base import AlloyInput
        with pytest.raises(ValueError):
            AlloyInput(components=[])


class TestDerivationChain:
    """Tests for the derivation chain."""

    def test_import(self):
        from periodica.utils.predictors.chain import DerivationChain
        assert DerivationChain is not None

    def test_chain_result_import(self):
        from periodica.utils.predictors.chain import ChainResult
        assert ChainResult is not None


class TestLegacyCompat:
    """Tests for legacy compatibility module."""

    def test_import(self):
        from periodica.utils.predictors.compat import PredictionEngine
        assert PredictionEngine is not None

    def test_import_derivations(self):
        from periodica.utils.predictors.compat import NuclearDerivation, AtomicDerivation
        assert NuclearDerivation is not None
        assert AtomicDerivation is not None
