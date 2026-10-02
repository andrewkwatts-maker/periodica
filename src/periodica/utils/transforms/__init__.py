"""
Material property transforms
============================

Temperature / pressure dependence (tabulated and polynomial models, the
Birch-Murnaghan equation of state) and optical dispersion models
(``optical_properties``). Both are M6 port targets for periodica-mat.

Example::

    from periodica.utils.transforms.thermo_pressure import (
        BirchMurnaghanConfig, BirchMurnaghanEOS,
    )

    eos = BirchMurnaghanEOS(BirchMurnaghanConfig(K0_GPa=160, K0_prime=4.5))
    eos.bulk_modulus_at_pressure(10e9)   # Pa
"""

from .thermo_pressure import (
    BirchMurnaghanEOS,
    TemperatureDependence,
    ThermoPressureSampler
)

__all__ = [
    'BirchMurnaghanEOS',
    'TemperatureDependence',
    'ThermoPressureSampler',
]
