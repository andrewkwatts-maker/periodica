"""Physical constants exposed as ``periodica.PhysicsConstants``.

This is the class that used to live at the top of
``periodica.utils.physics_calculator``; the semi-empirical calculators that
shared that module were removed (migration census M0). Values are unchanged.
The plan (A5) is to generate this facade from the single native constants
table, so treat it as read-only data rather than a place to add physics.
"""


class PhysicsConstants:
    """Fundamental physical constants"""
    # Masses in atomic mass units (u) and MeV/c²
    PROTON_MASS_U = 1.007276466621  # u
    NEUTRON_MASS_U = 1.008664915  # u
    ELECTRON_MASS_U = 0.000548579909065  # u

    PROTON_MASS_MEV = 938.27208816  # MeV/c²
    NEUTRON_MASS_MEV = 939.56542052  # MeV/c²
    ELECTRON_MASS_MEV = 0.51099895  # MeV/c²

    # Quark masses (current quark masses in MeV/c²)
    UP_QUARK_MASS_MEV = 2.2
    DOWN_QUARK_MASS_MEV = 4.7
    STRANGE_QUARK_MASS_MEV = 95.0
    CHARM_QUARK_MASS_MEV = 1275.0
    BOTTOM_QUARK_MASS_MEV = 4180.0
    TOP_QUARK_MASS_MEV = 173100.0

    # Quark charges (in units of e)
    UP_QUARK_CHARGE = 2/3
    DOWN_QUARK_CHARGE = -1/3
    STRANGE_QUARK_CHARGE = -1/3
    CHARM_QUARK_CHARGE = 2/3
    BOTTOM_QUARK_CHARGE = -1/3
    TOP_QUARK_CHARGE = 2/3

    # Other constants
    RYDBERG_ENERGY_EV = 13.605693122994  # eV
    RYDBERG_ENERGY = RYDBERG_ENERGY_EV    # alias
    BOHR_RADIUS_PM = 52.9177210903  # pm
    FINE_STRUCTURE = 0.0072973525693  # α
    AVOGADRO = 6.02214076e23  # mol⁻¹

    # Nuclear constants
    NUCLEAR_RADIUS_CONST = 1.25  # fm, r₀ for R = r₀ * A^(1/3)
    BINDING_ENERGY_VOLUME = 15.75  # MeV, a_v
    BINDING_ENERGY_SURFACE = 17.8  # MeV, a_s
    BINDING_ENERGY_COULOMB = 0.711  # MeV, a_c
    BINDING_ENERGY_ASYMMETRY = 23.7  # MeV, a_a
    BINDING_ENERGY_PAIRING = 11.2  # MeV, a_p
